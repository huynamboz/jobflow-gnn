# Thuật toán tính Matching Score

## Công thức

```
final_score = min(base_score × role_penalty, must_have_cap)

base_score = α × text_similarity
           + β × semantic_skill_overlap
           + γ × seniority_score

α = 0.55, β = 0.30, γ = 0.15
```

---

## Cơ sở thiết kế công thức

### Tại sao hybrid scoring thay vì 1 model duy nhất?

Production job matching systems (LinkedIn LinkSAGE — KDD 2025, WHIN+CSAGNN — 2024) không bao giờ dùng 1 model duy nhất để quyết định matching. Lý do:

- **GNN/text embedding** tốt ở semantic understanding nhưng dễ match sai role (CV "Python developer" match JD "Data Scientist" vì text gần nhau)
- **Skill overlap** tốt ở exact matching nhưng miss related skills (Flask developer không match Django job dù cùng Python web framework)
- **Seniority/role** là hard constraints mà ML model thường bỏ qua

→ Kết hợp 3 signals cho kết quả robust hơn bất kỳ component nào đứng riêng.

### Tại sao α=0.55, β=0.30, γ=0.15?

Weights được xác định qua **3 vòng thực nghiệm iterative**, không phải chọn ngẫu nhiên:

| Version | α (text) | β (skill) | γ (seniority) | Vấn đề phát hiện |
|---------|----------|-----------|---------------|------------------|
| v1 | 0.85 | 0.10 | 0.05 | Text dominate → AI jobs lọt top cho Frontend CV |
| v2 | 0.80 | 0.15 | 0.05 | Vẫn text quá mạnh, skill matching yếu |
| **v3** | **0.55** | **0.30** | **0.15** | **Cân bằng — AI jobs biến mất, đúng role** |

**Validation từ trained MLP reranker:**

Khi train MLP reranker trên 20 features (5.300 labeled pairs), feature importance cho thấy:
```
role_penalty:     0.132  ← quan trọng nhất
gnn_rank:         0.127
stage1_score:     0.117
text_similarity:  0.115  ← chỉ #4, không phải #1
skill_weighted:   0.113
```

Text similarity chỉ đứng #4 trong feature importance → xác nhận rằng α=0.55 (giảm từ 0.85) là hướng đúng. Skill matching và role compatibility quan trọng hơn text similarity trong bài toán job matching.

### Tại sao role penalty = 0.7?

- Frontend CV match Frontend JD → penalty 1.0 (không giảm)
- Frontend CV match AI Security JD → penalty 0.7 (giảm 30%)
- Giảm 30% đủ để đẩy cross-role matches từ eligible (>0.65) xuống dưới threshold

Ví dụ: base_score = 0.75 (cao vì text gần nhau)
- Cùng role: 0.75 × 1.0 = 0.75 → eligible ✅
- Khác role: 0.75 × 0.7 = 0.525 → not eligible ❌

### Tại sao must-have cap = 0.55/0.70?

Logic nghiệp vụ: nếu JD yêu cầu Django (required, importance=5) mà CV không có Django → dù text similarity cao, CV không nên eligible.

- Thiếu 1 required skill → cap 0.70 (vẫn có thể eligible nếu mọi thứ khác tốt)
- Thiếu 2+ required skills → cap 0.55 (gần như chắc chắn not eligible)

### Tại sao Platt calibration trên balanced data?

Raw score 0.7 không có meaning — không nghĩa là "70% xác suất match". Platt scaling fit sigmoid trên validation data:

```
P(match) = sigmoid(0.865 × score - 0.379)
```

- Fit trên **balanced data** (500 positive + 500 negative) thay vì full data (75% negative)
- Lần đầu fit imbalanced → sigmoid bias → tất cả eligible=false → fix bằng balanced data
- Threshold 0.5 trên calibrated probability có meaning: "50% xác suất match trở lên"

### Tóm tắt nguồn gốc

| Thành phần | Nguồn gốc |
|-----------|-----------|
| Hybrid scoring (multi-signal) | Research: LinkSAGE, WHIN+CSAGNN |
| α, β, γ weights | Iterative experimentation (3 versions, test CV thật) |
| Role penalty, must-have cap | Failure case analysis (AI job cho Frontend CV) |
| Edge case penalties | Testing: generic CV, overqualified, tool-only match |
| Calibration | Statistical method (Platt scaling) trên validation set |
| Feature importance validation | MLP reranker trained trên 5.300 labeled pairs |

---

## 5 thành phần

### 1. Text Similarity (α = 0.55)

Cosine similarity giữa CV text embedding và JD text embedding.

```
cv_vec  = MiniLM(cv_text)    → 384-dim vector
jd_vec  = MiniLM(jd_text)    → 384-dim vector
sim     = cosine(cv_vec, jd_vec)
score   = (sim + 1) / 2      → normalize [0, 1]
```

### 2. Semantic Skill Overlap (β = 0.30)

Skill overlap có trọng số + semantic matching qua skill graph.

**Per-JD importance (1-5):**
```
5 — skill xuất hiện trong title JD
4 — trong section "required" / "must have"
2 — trong section "nice to have" / "preferred"
3 — body chung (default)
± 1 boost/penalty từ TF-IDF (rare skill +1, common skill -1)
```

**Matching logic cho mỗi JD skill:**
```
if skill in CV:
    credit = importance × 1.0           ← direct match (full)
elif related_skill in CV (via graph):
    credit = importance × similarity × 0.6  ← partial credit
else:
    credit = 0
```

**Ví dụ semantic matching:**
```
JD yêu cầu: django (importance=4)
CV có: flask (không có django)

Skill graph: flask relates_to django, PMI similarity = 0.7
→ credit = 4 × 0.7 × 0.6 = 1.68 (thay vì 0)
```

**Final:** `score = total_credit / total_importance`

### 3. Seniority Score (γ = 0.15)

```
distance = |cv_level - jd_level|

score = max(0, 1.0 - distance × 0.4)

  distance 0 → 1.0  (exact match)
  distance 1 → 0.6  (gần — junior↔mid)
  distance 2 → 0.2  (xa — junior↔senior)
  distance 3+ → 0.0  (mismatch — intern↔lead)
```

Levels: intern(0), junior(1), mid(2), senior(3), lead(4), manager(5)

### 4. Role Penalty (nhân vào base_score)

```
cv_role  = infer_role(cv_skills, cv_text)
jd_role  = infer_role(jd_skills, jd_text)

penalty:
  1.0 — cùng role hoặc compatible
  0.7 — khác role
```

**Compatible roles:**
```
frontend ↔ fullstack
backend  ↔ fullstack, devops, data
data     ↔ ml
mobile   ↔ frontend
security ↔ devops, backend
```

### 5. Must-Have Penalty (cap cuối cùng)

Skills có importance >= 4 là "must-have". Thiếu → cap score.

```
missing_required = count(jd_skills where importance >= 4 AND skill NOT in CV)

if missing_required >= 2:  cap score ≤ 0.55
if missing_required == 1:  cap score ≤ 0.70
if missing_required == 0:  no cap
```

---

## Pipeline tính score

```
Step 1: text_similarity   = cosine(MiniLM(cv), MiniLM(jd))
Step 2: skill_overlap     = semantic_weighted_overlap(cv_skills, jd_skills, skill_graph)
Step 3: seniority         = max(0, 1 - |cv_level - jd_level| × 0.4)
Step 4: base_score        = 0.55×text + 0.30×skill + 0.15×seniority
Step 5: role_penalty      = 1.0 (compatible) or 0.7 (mismatch)
Step 6: must_have_cap     = cap nếu thiếu required skills
Step 7: final_score       = min(base_score × role_penalty, must_have_cap)
Step 8: eligible          = final_score >= 0.65
```

---

## Ví dụ 1: Đúng role (Frontend CV → React Developer JD)

**CV:** React/Vue developer, skills = [react, typescript, javascript, git, vuejs, html_css]
**JD:** "React Developer", skills = [react(5), typescript(4), redux(4), html_css(3), tailwind(2)]

```
text_similarity   = 0.82
skill_overlap:
  react:      5 × 1.0 = 5.0  (direct match)
  typescript: 4 × 1.0 = 4.0  (direct match)
  redux:      4 × 0.5 × 0.6 = 1.2  (react relates_to redux via graph)
  html_css:   3 × 1.0 = 3.0  (direct match)
  tailwind:   2 × 0.0 = 0.0  (no match)
  total = 13.2 / 18 = 0.73
seniority     = 0.6 (junior→mid, distance=1)
role_penalty  = 1.0 (both frontend)
must_have     = 0 missing (react✓, typescript✓, redux partial) → no cap

base  = 0.55×0.82 + 0.30×0.73 + 0.15×0.6 = 0.451 + 0.219 + 0.09 = 0.76
final = 0.76 × 1.0 = 0.76  → ELIGIBLE ✅
```

## Ví dụ 2: Sai role (Frontend CV → AI Security Engineer JD)

**CV:** React developer, skills = [react, typescript, javascript, python, git]
**JD:** "AI Security Engineer", skills = [python(5), machine_learning(5), security(4), aws(4), docker(3)]

```
text_similarity   = 0.68
skill_overlap:
  python:           5 × 1.0 = 5.0  (direct)
  machine_learning: 5 × 0.0 = 0.0  (no match or relation)
  security:         4 × 0.0 = 0.0
  aws:              4 × 0.0 = 0.0
  docker:           3 × 0.0 = 0.0
  total = 5.0 / 21 = 0.24
seniority     = 0.6
role_penalty  = 0.7 (frontend → ml mismatch)
must_have     = 2 missing (machine_learning, security) → cap 0.55

base  = 0.55×0.68 + 0.30×0.24 + 0.15×0.6 = 0.374 + 0.072 + 0.09 = 0.536
after_role = 0.536 × 0.7 = 0.375
final = min(0.375, 0.55) = 0.375  → NOT ELIGIBLE ❌
```

## Ví dụ 3: Semantic matching (Flask CV → Django JD)

**CV:** Python backend, skills = [python, flask, postgresql, redis, docker]
**JD:** "Django Developer", skills = [django(5), python(4), postgresql(4), rest_api(3), git(2)]

```
skill_overlap:
  django:     5 × 0.7 × 0.6 = 2.1  (flask relates_to django, PMI=0.7)
  python:     4 × 1.0 = 4.0  (direct)
  postgresql: 4 × 1.0 = 4.0  (direct)
  rest_api:   3 × 0.0 = 0.0
  git:        2 × 0.0 = 0.0
  total = 10.1 / 18 = 0.56

→ Skill Overlap baseline: django NOT in CV → 0 match → score thấp
→ GNN-powered: flask relates_to django → partial credit → score cao hơn
```

---

## So sánh phiên bản scoring

| Phiên bản | α | β | γ | Role penalty | Must-have | Semantic skills |
|-----------|---|---|---|-------------|-----------|-----------------|
| v1 (ban đầu) | 0.85 | 0.10 | 0.05 | Không | Không | Không |
| v2 (+role) | 0.85 | 0.10 | 0.05 | 0.7/1.0 | Không | Không |
| **v3 (hiện tại)** | **0.55** | **0.30** | **0.15** | **0.7/1.0** | **Cap 0.55/0.70** | **Skill graph PMI** |

Kết quả v3: AI/ML jobs biến mất khỏi top 10 cho Frontend CV, VueJS jobs xuất hiện nhờ semantic matching.
