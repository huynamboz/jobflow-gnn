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
4 node types: CV(386), Job(386), Skill(143), Seniority(6)
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
Input: 15 features → Linear(32) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Dropout(0.2) → Linear(1)
Output: match probability (sigmoid)
Training: BCEWithLogitsLoss, Adam optimizer, 50 epochs
```

### 15 Features

```
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
skill_specificity        — rarity of CV skills (fewer co-occurrence partners = more specific)
tool_ratio               — fraction of CV skills that are tools
```

### Reranker = ORDER, Stage 1 = SCORE

```
Reranker output: probability → quyết định thứ tự ranking
Stage 1 score: hybrid scoring → hiển thị + eligibility check
```

Tách biệt vì reranker probability chưa calibrated (0.3-0.5 cho good matches), nhưng ordering chính xác hơn stage 1.

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

- **143 canonical skills** organized by category (technical, soft, tool, domain)
- **Alias mapping:** "ReactJS" → "react", "K8s" → "kubernetes", etc.
- **Skill graph:** PMI co-occurrence → 3.613 skill pairs

### Labeling

- Rule-based + skill relations + noise
- Positive: skill_overlap ≥ 0.5 AND seniority_dist ≤ 1, OR expanded_overlap ≥ 0.6 (via related skills)
- Hard negative: 0.15 ≤ overlap < 0.5 AND seniority_dist ≤ 1
- Label noise: 10% flip

---

## 9. Kết quả

### Benchmark (4.7K real JDs + 2K real CVs)

```
GNN AUC-ROC: 0.65 — beats all baselines
Reranker accuracy: 72.2% (MLP trên 15 features)
```

### Inference Demo

```
Input: Frontend developer CV (React, Vue, TypeScript)
Output top 3:
  #1 Software Engineer (matched 8 skills, có vuejs) — 0.70
  #2 Web Developer (matched 8 skills, vuejs + sass) — 0.78
  #3 IT Developer (matched 7 skills) — 0.66

Không có AI/ML/Security/SAP trong top 10 ✅
```

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
