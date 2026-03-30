# Tổng hợp kỹ thuật đã áp dụng

---

## 1. Kiến trúc hệ thống

### Two-Stage Ranking Pipeline

```
Stage 1 (Retrieve): Hybrid scoring → Top 50 candidates
  ↓
Stage 2 (Rerank): MLP reranker trên 15 features → reorder
  ↓
Output: Top K results (score từ Stage 1, thứ tự từ Stage 2)
```

**Nguyên tắc:** Reranker quyết định ORDER, Stage 1 giữ SCORE + eligibility. Tách biệt ranking quality và score calibration.

### Module Architecture (DI Pattern)

```
embedding/     EmbeddingProvider ABC → EnglishProvider / MultilingualStub
crawler/       CrawlProvider ABC → JobSpyProvider (Indeed)
reranker/      FeatureExtractor + Reranker (MLP)
inference/     InferenceEngine (two-stage) + Checkpoint save/load
cv_parser/     Section-based parsing (PDF/DOCX/text)
api/           FastAPI (6 endpoints)
```

Thêm provider mới = implement ABC + register. Không sửa code hiện tại.

---

## 2. Graph Neural Network

### Model: HeteroGraphSAGE

```
Per-type Projection → GraphSAGE (2-3 layers, mean aggregation) → MLPDecoder
```

- **Tại sao GraphSAGE:** Inductive learning — CV/JD mới không cần retrain toàn graph
- **Alternative:** HeteroRGCN (relation-specific weights) — implemented nhưng GraphSAGE tốt hơn trên data size hiện tại

### Graph Schema

```
4 node types: CV(386), Job(386), Skill(208), Seniority(6)
9 edge types:
  Core:   has_skill, requires_skill, has_seniority, requires_seniority, match, no_match
  Enrich: skill→skill (co-occurrence PMI), job→job (similarity), cv→cv (similarity)
```

### Training

- **Loss:** BPR (Bayesian Personalized Ranking) — tối ưu ranking trực tiếp
- **Hard negative sampling:** 70% hard negatives (same role, similar skills, wrong match)
- **Label edge stripping:** Remove match/no_match edges trước GNN message passing (prevent leakage)
- **Early stopping:** Monitor val_mrr, patience=30

---

## 3. Scoring Algorithm (v3)

### Công thức

```
final_score = min(base_score × role_penalty, must_have_cap) × edge_case_penalty

base_score = 0.55 × text_similarity
           + 0.30 × semantic_skill_overlap
           + 0.15 × seniority_score
```

### 7 thành phần

| # | Thành phần | Kỹ thuật |
|---|-----------|---------|
| 1 | Text similarity | Cosine(MiniLM embedding), normalize [-1,1]→[0,1] |
| 2 | Semantic skill overlap | Per-JD importance (title=5, required=4, nice-to-have=2) + skill graph PMI partial credit |
| 3 | Seniority score | Linear decay: max(0, 1 - distance × 0.4) |
| 4 | Role penalty | Role classifier → compatible=1.0, mismatch=0.7 |
| 5 | Must-have cap | Thiếu 1 required skill → cap 0.70, thiếu 2+ → cap 0.55 |
| 6 | Edge case: generic CV | < 4 skills → ×0.85 |
| 7 | Edge case: tool-only match | Chỉ match git/jira, không core skill → ×0.75 |

### Semantic Skill Matching

```
JD yêu cầu: django (importance=4)
CV có: flask (không có django)

Skill graph (PMI co-occurrence): flask relates_to django, similarity=0.7
→ credit = 4 × 0.7 × 0.6 = 1.68 (thay vì 0)

Baseline chỉ thấy: django NOT in CV → 0 điểm
GNN qua skill graph: flask → django → partial credit
```

---

## 4. Skill Extraction

### Per-JD Importance (1-5)

```
5 — skill trong JD title
4 — trong "required" / "must have" section
2 — trong "nice to have" / "preferred"
3 — body chung (default)
±1 — TF-IDF boost (rare skill +1, common -1)
```

### Zone Detection

JD text split thành required zone vs nice-to-have zone bằng regex patterns:
```
required: "required", "must have", "requirements", "mandatory", "essential"
nice-to-have: "nice to have", "preferred", "bonus", "plus", "optional"
```

### False Positive Prevention

- Single-char skills ("c", "r") cần context patterns: "C programming", "C/C++", "R Studio"
- Skip tokens length <= 1

---

## 5. CV Parser

### Section-Based Extraction

```
Step 1: Split CV thành sections (SKILLS, EXPERIENCE, PROJECTS, EDUCATION)
Step 2: Scan FULL text cho skills (robust cho PDF formatting)
Step 3: Filter: vague skills (security, problem_solving, communication, teamwork)
        chỉ giữ nếu nằm trong SKILLS section
```

**Tại sao không strict section-only:** PDF extraction tạo formatting khác nhau, section headers không luôn match regex → full text scan + filter robust hơn.

### Seniority Inference

- Check title first (stronger signal): "Senior", "Junior", "Lead", "Manager"
- Fallback: check description first 500 chars
- "Task manager" không trigger MANAGER (chỉ match "engineering manager", "project manager")

---

## 6. Stage 2 Reranker

### MLP Architecture (PyTorch)

```
Input: 20 features → Linear(32) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Dropout(0.2) → Linear(1)
Output: match probability (sigmoid)
Training: BCEWithLogitsLoss, Adam optimizer, 50 epochs
Accuracy: 77.8%
```

### 20 Features

```
# Base features (1-15)
text_similarity          — cosine(cv_embedding, jd_embedding)
skill_overlap_jaccard    — |CV ∩ JD| / |CV ∪ JD|
skill_overlap_weighted   — weighted by JD importance
semantic_skill_overlap   — with graph relation partial credit
missing_required_count   — count of missing importance≥4 skills
missing_required_ratio   — missing / total required
matched_skill_count      — number of matched skills
total_job_skills         — JD total skill count
seniority_distance       — |cv_level - jd_level|
seniority_score          — linear decay
role_penalty             — role compatibility
experience_years         — CV experience
cv_skill_count           — number of CV skills
skill_specificity        — rarity of CV skills
tool_ratio               — fraction of CV skills that are tools

# Stage 1 + GNN signals (16-20)
stage1_score             — full hybrid score from Stage 1
gnn_score                — GNN text similarity component
gnn_rank                 — normalized rank from Stage 1 (0-1)
must_have_cap_triggered  — 0/0.5/1.0 based on missing required skills
edge_case_penalty_triggered — 1.0 if generic CV / overqualified / tool-only match
```

### Feature Importance (top 5 từ trained model)

```
role_penalty                   0.132  ← role match quan trọng nhất
gnn_rank                       0.127  ← GNN ranking signal
edge_case_penalty_triggered    0.124  ← edge case detection
gnn_score                      0.121  ← GNN similarity
stage1_score                   0.117  ← full hybrid score
```

GNN features (gnn_rank, gnn_score) nằm top 5 → GNN thực sự đóng góp vào reranking quality.

### Reranker = ORDER, Stage 1 = SCORE, Calibration = ELIGIBILITY

```
Reranker output: probability → quyết định thứ tự ranking
Stage 1 score: hybrid scoring → hiển thị cho user
Calibration: Platt scaling → quyết định eligible (threshold = 0.5)
```

Ba concerns tách biệt:
- ORDER: reranker MLP (trained trên labeled pairs)
- SCORE: Stage 1 hybrid (interpretable, 0.55×text + 0.30×skill + 0.15×seniority)
- ELIGIBILITY: Platt sigmoid calibration (fit trên balanced validation data)

---

## 6.5. Score Calibration (Platt Scaling)

### Vấn đề

Raw Stage 1 scores (0.65-0.85) không có probabilistic meaning. Score 0.7 không nghĩa là "70% xác suất match".

### Giải pháp

Platt scaling: fit sigmoid `P(match) = 1/(1 + exp(-(a×score + b)))` trên balanced validation data.

```
Parameters: a=0.865, b=-0.379
Fit trên: 1000 balanced pairs (500 positive + 500 negative)
Accuracy: ~50% (expected cho balanced calibration)
```

### Cách sử dụng

```
raw_score = 0.78 (Stage 1 hybrid)
calibrated = sigmoid(0.865 × 0.78 - 0.379) = 0.58
eligible = calibrated >= 0.5 → True ✅
```

### Tại sao fit trên balanced data

Lần đầu fit trên imbalanced data (75% negative) → sigmoid bias low → tất cả scores < 0.5 → mọi kết quả `eligible=false`. Fix: balanced data → sigmoid centered đúng.

---

## 7. Role Classification

### Role Categories

```
frontend, backend, fullstack, devops, data, ml, mobile, security, other
```

### Inference Logic

```
1. Title pattern matching (strongest): "front-end" → frontend, "devops" → devops
2. Skill-based (if ≥2 defining skills): react+vue → frontend
3. Special: frontend + backend skills → fullstack
```

### Compatibility Matrix

```
frontend ↔ fullstack, mobile
backend  ↔ fullstack, devops, data
data     ↔ ml
security ↔ devops, backend
```

---

## 8. Data Pipeline

### Crawling

- **Tool:** python-jobspy (Indeed)
- **Scale:** 4.854 unique JDs, 70 IT queries
- **Storage:** JSONL (append-safe, dedup by source_url)
- **Skill extraction:** n-gram matching + TF-IDF importance

### CV Dataset

- **Source:** HuggingFace datasetmaster/resumes (4.817 IT resumes, MIT)
- **Enrichment:** Extract skills từ BOTH structured fields AND full text (summary, responsibilities, projects)
- **Result:** 73 unique CV skills, avg 13.4 skills/CV (trước enrichment: 39 skills, 5.4/CV)

### Skill Dictionary

- **208 canonical skills** organized by category (technical, soft, tool, domain)
- **Alias mapping:** "ReactJS" → "react", "K8s" → "kubernetes", etc.
- **Skill graph:** PMI co-occurrence → 3.613 skill pairs

### Labeling

- Rule-based + skill relations + noise
- Positive: skill_overlap ≥ 0.5 AND seniority_dist ≤ 1, OR expanded_overlap ≥ 0.6 (via related skills)
- Hard negative: 0.15 ≤ overlap < 0.5 AND seniority_dist ≤ 1
- Label noise: 10% flip

---

## 9. Kết quả

### Benchmark (4.7K real JDs + 1K real CVs)

```
GNN AUC-ROC: 0.6815 (was 0.63 before skill expansion)
Reranker accuracy: 77.2% (MLP trên 20 features)
Calibration: balanced Platt scaling (a=0.865, b=-0.379)
Skill coverage: 208 skills, JD-CV overlap 80 skills
```

### GNN Inductive Inference

```
CV in graph:  real GNN decode → sigmoid(decoder(z_cv, z_job))
              blended 0.6×GNN + 0.4×text similarity
CV new upload: fallback to text similarity (MiniLM cosine)
```

Precompute cả CV + Job GNN embeddings tại init. z_dict cached cho fast decode.

### Inference Demo

**CV 1: Vue/React Frontend developer, 3 năm kinh nghiệm**

```
#1 React Developer        0.81  ✅  matched: react, typescript, tailwind, rest_api, git, html_css, javascript
#2 Web Developer           0.79  ✅  matched: vuejs, react, sass, ci_cd, rest_api, git, html_css, javascript
#3 Technical Writer        0.68  ✅
#4 IT Developer            0.66  ✅  matched: mongodb, mysql, nodejs, react, rest_api, git, javascript
#5 Software Engineer       0.66  ✅  matched: vuejs, react, typescript, nodejs, html_css, javascript
```

**CV 2: React/AI student, 1 năm kinh nghiệm**

```
#1 IT Developer            0.66  ✅  matched: express, mongodb, nodejs, postgresql, python, react, rest_api, git, javascript
#2 React Developer         0.82  ✅  matched: nextjs, react, redux, typescript, rest_api, git, html_css, javascript
#3 Front End Developer     0.70  ✅  matched: nextjs, react, typescript, javascript
#4 Web Developer           0.69  ✅  matched: figma, react, rest_api, git, html_css, javascript
#5 Software Engineer (Viz) 0.70  ✅  matched: machine_learning, python, react, typescript, javascript
```

**Không có AI/ML/Security/SAP trong top 10 cho cả 2 CVs ✅**
**Tất cả eligible=true với calibrated threshold ✅**
**New skills detected: nextjs, redux, figma, express, material_ui, machine_learning ✅**

### Improvement journey

| Version | Top issue | Fix | Result |
|---------|-----------|-----|--------|
| v1 | AI jobs in top 3 cho Frontend CV | — | Sai role matching |
| v2 | Text dominate (α=0.85) | Role penalty + rebalance weights | AI jobs tụt xuống #9 |
| v3 | Missing required skills | Must-have cap + semantic skills | AI jobs biến mất |
| v4 | Scores quá thấp | Tách order/score/eligibility | Scores 0.66-0.83 |
| v5 | Calibration imbalanced | Balanced Platt scaling | Threshold meaningful |
| v6 | Skill coverage 143 | Expand 208 skills + GNN decode | AUC +5%, 80 shared skills |

---

## 10. API Endpoints

```
POST /match/jd         — JD text → Top K CVs
POST /match/cv         — CV text → Top K Jobs
POST /match/cv/upload  — Upload CV PDF/DOCX → Top K Jobs
POST /parse/cv         — CV text → parsed structured data (debug)
POST /parse/cv/upload  — Upload CV PDF/DOCX → parsed data (debug)
GET  /health           — status + stats

Run: uvicorn ml_service.api.app:app --port 8001
Docs: http://localhost:8001/docs
```
