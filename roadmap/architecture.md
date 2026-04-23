# JobFlow GNN — Architecture & Flow

> Tài liệu này mô tả toàn bộ kiến trúc hệ thống, flow dữ liệu, và cách các thành phần liên kết với nhau.
> Mục đích: đảm bảo không đi sai hướng khi build thêm tính năng.

---

## Bài toán

**Matching CV ↔ Job** — Cho một CV, tìm các Job phù hợp nhất (và ngược lại).

Đây là bài toán **ranking**, không phải classification đơn thuần:
- Không chỉ trả lời "phù hợp hay không"
- Mà phải **sắp xếp thứ tự** từ phù hợp nhất → kém nhất

---

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────┐
│                    RAW DATA                             │
│  CVs (PDF/DOCX → raw_text)   JDs (LinkedIn crawl)      │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  CV Batch Extraction │      │   JD Batch Extraction   │
│  LLM → structured   │      │   LLM → structured      │
│  role_category       │      │   role_category          │
│  seniority (0-5)     │      │   seniority (0-5)        │
│  skills + proficiency│      │   skills + importance    │
│  experience_years    │      │   experience_min/max     │
└──────────┬──────────┘      └────────────┬────────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │      GRAPH DATABASE      │
           │  CV nodes ←→ Skill nodes │
           │  Job nodes ←→ Skill nodes│
           │  Seniority nodes         │
           └──────────────┬───────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │   PAIR GENERATION        │
           │  (CV, Job) candidate     │
           │  pairs với selection     │
           │  reason + split 70/15/15 │
           └──────────────┬───────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │   LLM LABELING           │
           │  skill_fit 0/1/2         │
           │  seniority_fit 0/1/2     │
           │  domain_fit 0/1/2        │
           │  overall 0/1/2           │
           └──────────────┬───────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │   MODEL TRAINING         │
           │  1. GNN (HeteroSAGE)     │
           │  2. XGBoost Reranker     │
           └──────────────┬───────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │   INFERENCE ENGINE       │
           │  Stage 1: Retrieve top 50│
           │  Stage 2: Rerank → top K │
           └──────────────────────────┘
```

---

## Node Features trong Graph

| Node type | Dims | Features |
|-----------|------|---------|
| CV | 386 | 384-dim sentence embedding (full text) + experience_years + education |
| Job | 386 | 384-dim sentence embedding (full text) + salary_min + salary_max |
| Skill | 385 | 384-dim embedding (skill name) + category |
| Seniority | 6 | One-hot (6 mức: Intern=0 → Manager=5) |

**Embedding model:** `all-MiniLM-L6-v2` (SentenceTransformers, 384-dim)
— Không phải Word2Vec hay TF-IDF. Mỗi text → 1 vector duy nhất capture semantic.

**Edge features:**
- `(CV) -[has_skill]→ (Skill)`: proficiency (1–5)
- `(Job) -[requires_skill]→ (Skill)`: importance (1–5)
- `(CV) -[has_seniority]→ (Seniority)`: binary
- `(CV) -[similar_profile]→ (CV)`: Jaccard overlap ≥ 0.3

---

## Inference Flow (2-Stage)

### Stage 1 — Retrieve (fast, top 50)

```
score = 0.55 × GNN_score + 0.30 × skill_overlap + 0.15 × seniority_score
```

Trong đó:
- `GNN_score = 0.6 × gnn_decode(cv_emb, job_emb) + 0.4 × cosine(cv_text, job_text)`
- `skill_overlap` = weighted Jaccard với importance, có PMI bonus (+0.6×) cho similar skills
- `seniority_score = max(0, 1 - |cv_seniority - job_seniority| × 0.4)`

**Lưu ý:** Trọng số `0.55/0.30/0.15` là **hardcode (heuristic)**, không học từ data.
Đây chỉ để lọc sơ top 50 candidate, không phải kết quả cuối.

**Penalty thêm:**
- Role mismatch penalty (nếu cv_role ≠ job_role)
- Must-have penalty (thiếu skill importance ≥ 4)
- Edge case: CV < 4 skills (×0.85), overqualified Senior → Junior job (×0.8)

### Stage 2 — Rerank (learned, top 50 → top K)

```
XGBoost(20 features) → match probability → sort
```

20 features bao gồm: text_sim, skill_overlap, seniority_diff, role_match, gnn_score, gnn_rank, ...

**Trọng số ở Stage 2 được HỌC từ labeled pairs** — không hardcode.
→ Đây là lý do cần đủ labeled pairs để train.

Nếu reranker chưa train → Stage 2 bị skip → chỉ dùng Stage 1 hardcode.

---

## LLM Extraction

### CV Extraction (7 fields)

| Field | Type | Mô tả |
|-------|------|-------|
| `name` | string | Tên candidate |
| `experience_years` | float | Số năm kinh nghiệm |
| `seniority` | int 0–5 | Intern/Junior/Mid/Senior/Lead/Manager |
| `role_category` | enum | backend/frontend/fullstack/mobile/devops/data_ml/data_eng/qa/design/ba/other |
| `education` | enum→int | none/college/bachelor/master/phd → 0–4 |
| `skills` | list | Canonical skill names + proficiency 1–5 |
| `work_experience` | list | title, company, duration, description |

### JD Extraction (fields)

| Field | Mô tả |
|-------|-------|
| `role_category` | Cùng enum với CV |
| `seniority` | 0–5 |
| `experience_min/max` | Số năm yêu cầu |
| `is_remote` | bool |
| `salary_usd_annual_min/max` | Lương (USD) |
| `skills` | Canonical names + importance 1–5 |

**Importance scale (JD):**
- 5 = Must-have (thiếu là loại ngay)
- 4 = Required (cần có để cân nhắc)
- 3 = Preferred (có thì tốt)
- 2 = Nice-to-have
- 1 = Bonus

**Rule:** Tối đa 30% skills ở importance=5. Phân bổ ≥ 3 mức cho jobs có ≥ 5 skills.

---

## Pair Generation Strategy

| Type | Cách chọn | Mục đích |
|------|-----------|---------|
| `high_overlap` | Jaccard ≥ 0.5 + same role | Positive examples rõ ràng |
| `medium_overlap` | Jaccard 0.2–0.5 + same/related role | Hard positive |
| `hard_negative` | Same role nhưng Jaccard < 0.1 | Hard negative (model phải phân biệt) |
| `random` | Random CV × Job | Easy negative |

**Target:** 3,500 pairs, split 70/15/15 (train/val/test), stratified by type.

---

## LLM Labeling

**Chi phí:** ~150đ/1M tokens → 3,500 pairs × 5,000 tokens ≈ 2,600đ.

**Label dimensions:**
- `skill_fit` 0/1/2: mức độ skill CV đáp ứng JD
- `seniority_fit` 0/1/2: seniority có phù hợp không
- `experience_fit` 0/1/2: số năm kinh nghiệm
- `domain_fit` 0/1/2: role_category có khớp không
- `overall` 0/1/2: kết luận tổng thể (0=không phù hợp, 1=phù hợp, 2=rất phù hợp)

**Target distribution:**
- overall=0: 40–50%
- overall=1: 30–40%
- overall=2: 10–20%

---

## Canonical Skill System

~200 canonical identifiers trong `skill-alias.json`.

**Quy tắc:**
- Tất cả skill names phải là canonical (lowercase, underscore)
- LLM được cung cấp danh sách canonical để map
- SkillNormalizer xử lý alias: "React.js" → "react", "PostgreSQL" → "postgresql"
- Skill không map được → bỏ qua (không tạo record)

---

## Dependency Chain (thứ tự bắt buộc)

```
1. CV batch extraction (role_category + seniority + skills đúng)
        ↓ bắt buộc hoàn thành trước
2. Generate pairs (cần role_category để filter có nghĩa)
        ↓
3. LLM labeling (cần pairs)
        ↓
4. Train GNN (cần labeled pairs)
        ↓
5. Train XGBoost reranker (cần labeled pairs + GNN embeddings)
        ↓
6. Stage 2 inference hoạt động → AUC-ROC > 0.71
```

**JD re-run** có thể chạy song song với bước 1 vì không phụ thuộc nhau.

---

## Những gì KHÔNG làm

- **Không dùng Word2Vec hay TF-IDF** — dùng SentenceTransformers (tốt hơn nhiều)
- **Không label thủ công** — dùng LLM auto-label (cost ~2,600đ cho 3,500 pairs)
- **Không hardcode trọng số final ranking** — XGBoost học từ data
- **Không include HR/PM CVs** — tập trung dev roles (AI/SE/Devops/Tester/UX_UI)
- **Không cosine similarity đơn giản** — dùng 2-stage: GNN graph + XGBoost reranker

---

## Baseline & Target

| Metric | Week 9 (baseline) | Week 10 (target) |
|--------|-------------------|------------------|
| AUC-ROC | 0.71 | > 0.75 |
| Labeled pairs | ~9,889 (proxy) | ≥ 3,000 (LLM labeled) |
| CV với role_category đúng | 0/365 | ≥ 300/365 |
| Reranker | Chưa train | Trained XGBoost |
