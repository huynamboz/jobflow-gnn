# Tổng hợp kỹ thuật đã áp dụng

---

## 1. Kiến trúc hệ thống

### Two-Stage Ranking Pipeline

```
Stage 1 (Retrieve): Hybrid scoring (GNN + skill + seniority) → Top 50 candidates
  ↓
Stage 2 (Rerank): MLP reranker trên 20 features → reorder
  ↓
Output: Top K results (score từ Stage 1, thứ tự từ Stage 2, eligibility từ Calibration)
```

**Nguyên tắc:** Reranker quyết định ORDER, Stage 1 giữ SCORE, Platt Calibration quyết định ELIGIBILITY.

### Architecture

```
Django Backend (DRF + Admin + PostgreSQL)
  ├── apps/matching    → API endpoints (CV→Jobs matching)
  ├── apps/jobs        → Job CRUD + Admin
  ├── apps/cvs         → CV upload + Admin
  ├── apps/skills      → Skill dictionary + Admin
  ├── apps/users       → Auth (JWT)
  │
  └── ml_service/      → ML library (imported trực tiếp)
      ├── crawler/     → Multi-provider crawling (DI pattern)
      │   └── providers/  → auto-discovered
      │       ├── jobspy_provider.py     Indeed/Glassdoor
      │       ├── linkedin_provider.py   LinkedIn (Playwright + auth state)
      │       ├── adzuna_provider.py     Adzuna REST API
      │       └── remotive_provider.py   Remotive API
      ├── cv_parser/   → PDF/DOCX section-based parsing
      ├── data/        → Skill extraction, normalization, graph, LinkedIn CV loader
      ├── embedding/   → EmbeddingProvider ABC → EnglishProvider
      ├── graph/       → PyG HeteroData builder
      ├── inference/   → Two-stage engine + GNN inductive inference + checkpoint
      ├── models/      → HeteroGraphSAGE, HeteroRGCN, BPR loss
      ├── reranker/    → MLP reranker (20 features) + Platt calibration
      └── training/    → BPR trainer + hard negative sampling
```

---

## 2. Graph Neural Network

### Model: HeteroGraphSAGE

```
Per-type Projection → GraphSAGE (3 layers, 256 hidden, mean aggregation) → MLPDecoder
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

### GNN Inductive Inference (cho new CV upload)

```
CV mới upload (không có trong graph)
  → parse skills: [react, vuejs, typescript, ...]
  → tạo temporary CV node trong graph
  → connect tới Skill nodes: CV→react, CV→vuejs, CV→typescript
  → connect tới Seniority node: CV→MID
  → connect tới similar CVs (Jaccard ≥ 0.3)
  → GraphSAGE encode trên expanded graph
  → GNN embedding thật (128-dim) cho CV mới
  → decode(gnn_cv, gnn_job) cho mỗi job
  → Score discriminative (Vue CV gần Vue jobs, xa AI jobs)
```

### Training

- **Loss:** BPR (Bayesian Personalized Ranking) — tối ưu ranking trực tiếp
- **Hard negative sampling:** 70% hard negatives (same role, similar skills, wrong match)
- **Label edge stripping:** Remove match/no_match edges trước GNN message passing (prevent leakage)
- **Early stopping:** Monitor val_mrr, patience=50

---

## 3. Scoring Algorithm (v3)

### Công thức

```
final_score = base_score × role_penalty × must_have_penalty × edge_case_penalty

base_score = 0.55 × gnn_score
           + 0.30 × semantic_skill_overlap
           + 0.15 × seniority_score
```

### 7 thành phần

| # | Thành phần | Kỹ thuật |
|---|-----------|---------|
| 1 | GNN score | Inductive GNN decode (0.6×GNN + 0.4×text cosine), hoặc text cosine fallback |
| 2 | Semantic skill overlap | Per-JD importance (title=5, required=4, nice-to-have=2) + skill graph PMI partial credit |
| 3 | Seniority score | Linear decay: max(0, 1 - distance × 0.4) |
| 4 | Role penalty | Role classifier → compatible=1.0, mismatch=0.7 |
| 5 | Must-have penalty | Multiplicative: missing 1→×0.9, missing 2→×0.75, missing 3+→×0.6 |
| 6 | Edge case: generic CV | < 4 skills → ×0.85 |
| 7 | Edge case: tool-only match | Chỉ match git/jira, không core skill → ×0.75 |

### Semantic Skill Matching

```
JD yêu cầu: django (importance=4)
CV có: flask (không có django)

Skill graph (PMI co-occurrence): flask relates_to django, similarity=0.7
→ credit = 4 × 0.7 × 0.6 = 1.68 (thay vì 0)
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

### False Positive Prevention

- Single-char skills ("c", "r") cần context patterns
- Section-based: vague skills (security, problem_solving) chỉ giữ nếu trong SKILLS section

---

## 5. CV Parser

### Section-Based Extraction

```
Step 1: Split CV thành sections (SKILLS, EXPERIENCE, PROJECTS, EDUCATION)
Step 2: Scan FULL text cho skills (robust cho PDF formatting)
Step 3: Filter vague skills chỉ giữ nếu trong SKILLS section
```

---

## 6. Stage 2 Reranker

### MLP Architecture (PyTorch)

```
Input: 20 features → Linear(32) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Dropout(0.2) → Linear(1)
Output: match probability (sigmoid)
Training: BCEWithLogitsLoss, Adam optimizer, 50 epochs
Accuracy: ~75%
```

### 20 Features

```
# Base features (1-15)
text_similarity, skill_overlap_jaccard, skill_overlap_weighted, semantic_skill_overlap,
missing_required_count, missing_required_ratio, matched_skill_count, total_job_skills,
seniority_distance, seniority_score, role_penalty, experience_years, cv_skill_count,
skill_specificity, tool_ratio

# Stage 1 + GNN signals (16-20)
stage1_score, gnn_score, gnn_rank, must_have_cap_triggered, edge_case_penalty_triggered
```

### Reranker = ORDER, Stage 1 = SCORE, Calibration = ELIGIBILITY

Platt scaling (balanced fit) cho eligibility check: `P(match) = sigmoid(a×score + b)`

---

## 7. Multi-Provider Crawling

### Providers (auto-discovered DI pattern)

| Provider | Source | Auth | Jobs crawled |
|----------|--------|------|-------------|
| `jobspy` | Indeed | Không | 4,234 |
| `linkedin` | LinkedIn | Playwright + saved auth | 535 |
| `adzuna` | Adzuna API | API key | — |
| `remotive` | Remotive API | Không | — |

### LinkedIn Provider

- Playwright headless browser + saved auth state (cookies)
- CSS selectors trong JSON config (update khi LinkedIn đổi HTML)
- Stream save: mỗi job ghi ngay vào file (crash-safe)
- Fingerprint dedup: normalize(title) + normalize(company) + city → MD5

---

## 8. Data

### Hiện tại

| Data | Số lượng | Source |
|------|---------|--------|
| JDs | 4,769 unique | Indeed (US) + LinkedIn (VN + Remote) |
| CVs | 362 | LinkedIn PDF profiles (Vietnamese IT) |
| Skills | 208 canonical | skill-alias.json (aliases + categories) |
| Skill graph | 2,736 edges | PMI co-occurrence |

### CV Dataset (LinkedIn Vietnamese IT)

| Category | CVs | Avg skills |
|----------|-----|-----------|
| AI | filtered | 6.6 |
| DevOps | filtered | 9.1 |
| Software Engineer | filtered | 6.3 |
| Tester | filtered | 2.2 |
| Business Analyst | filtered | 1.9 |
| UX/UI | filtered | 0.6 |
| **Total (≥2 skills)** | **362** | **7.5** |

---

## 9. Kết quả

### Benchmark (4,641 JDs + 362 LinkedIn CVs)

```
Method                     auc_roc   recall@10   precision@10   ndcg@10
Cosine Similarity          0.5732    0.0343      0.7000         0.7215
Skill Overlap (Jaccard)    0.5095    0.0098      0.2000         0.2048
BM25                       0.5647    0.0196      0.4000         0.4627
GNN (Hybrid)               0.7225*   0.0245      0.5000         0.4572
```

**GNN AUC-ROC: 0.7225 — best, +21% vs best baseline**

### Matching Demo (Frontend developer CV, 3 năm, VueJS)

```
#1 Software Engineer - McKinsey Digital              0.668
#3 Fullstack Engineer (Typescript/PHP)               0.650  [matched vuejs!]
#4 React Developer                                   0.795
#8 Software Dev Engineer (Frontend) - Paradox DaNang  0.787  [Vietnamese company!]
```

- VueJS jobs xuất hiện nhờ LinkedIn VN JDs ✅
- Vietnamese companies trong top 10 ✅
- NO AI/ML/Security false positives ✅

---

## 10. API Endpoints

```
POST /api/matching/cv/           — CV text → Top K jobs
POST /api/matching/cv/upload/    — Upload CV PDF/DOCX → Top K jobs
POST /api/matching/parse/        — CV text → parsed skills (debug)
POST /api/matching/parse/upload/ — Upload CV → parsed skills (debug)
GET  /api/docs/                  — Swagger UI

Admin: /admin/ (Django Admin — full CRUD cho jobs, CVs, skills, users)

Run: cd backend && .venv/bin/python manage.py runserver 8000
```
