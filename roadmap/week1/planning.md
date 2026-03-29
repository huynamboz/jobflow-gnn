# Tuần 1 — Thiết kế bài toán & Nền tảng lý thuyết

**Thời gian:** Tuần 1 (15 tuần tổng)
**Mục tiêu:** Chốt graph schema, bài toán, embedding model, đọc paper đúng thứ tự, khởi tạo project.

> **Phạm vi phase này:** Chỉ tập trung vào **AI Matching Engine**.
> - Input: Job Description (JD)
> - Output: danh sách CV phù hợp (kèm score + eligibility flag)
> - **KHÔNG bao gồm:** gửi email apply, automation workflow, tracking apply
> — những phần đó là phase sau, tích hợp vào output của phase này.

> **Giả định dữ liệu:** Toàn bộ CV và JD là **tiếng Anh (English-only)** trong MVP.
> Kiến trúc giữ abstraction để nâng cấp multilingual sau — nhưng không implement trong MVP.

---

## Phase 1 — Chốt bài toán

### 1.0. Bài toán cụ thể

> **Đây là bài toán ranking / retrieval — không phải binary classification thuần.**
> Hệ thống phải trả về danh sách CV được sắp xếp theo độ phù hợp, không chỉ dự đoán match / không match.

**Job-driven retrieval:**
- **Input:** Job Description (JD) mới crawl về
- **Output:** Danh sách CV phù hợp kèm score và eligibility
- CV pool tương đối nhỏ, có thể precompute embedding trước
- JD stream lớn và liên tục cập nhật

**Output schema (decision-ready cho phase apply sau):**
```json
[
  { "cv_id": 123, "score": 0.82, "eligible": true  },
  { "cv_id": 456, "score": 0.61, "eligible": true  },
  { "cv_id": 789, "score": 0.43, "eligible": false }
]
```

**Threshold logic:**
```
eligible = True   if score >= threshold  (đề xuất: 0.65, tune sau)
eligible = False  if score <  threshold
```

**Inference flow (không chạy GNN full mỗi lần):**
```
JD (mới)
  ↓ extract skills + seniority
  ↓ build job node features
  ↓ encode → embedding vector
  ↓ compare với CV embeddings (precomputed)
  ↓ hybrid score = GNN_score + skill_overlap + seniority_match
  ↓ apply threshold → eligible flag
  → trả về Top K CV + score + eligible
```

**Vai trò của GNN (tránh nhầm lẫn):**
- **Training time:** GNN học embedding tốt hơn từ cấu trúc đồ thị → CV/JD embedding chất lượng cao hơn cosine thuần
- **Inference time:** KHÔNG chạy full GNN — dùng precomputed embedding + hybrid scoring
- GNN cải thiện embedding quality, không chạy realtime

---

### 1.1. Thiết kế Graph Schema (MVP)

> Schema sai thì GNN mạnh mấy cũng fail. Chốt ngay từ đầu, implement theo giai đoạn.

#### Node Types

| Node | Mô tả | Features | Giai đoạn |
|------|--------|---------|-----------|
| `CV` | Hồ sơ ứng viên | embedding (text), experience_years, education_level | **Core** |
| `Job` | Tin tuyển dụng | embedding (title+desc+requirements), salary_min, salary_max | **Core** |
| `Skill` | Kỹ năng canonical | embedding (skill name), category | **Core** |
| `Seniority` | Cấp độ kinh nghiệm | one-hot: intern/junior/mid/senior/lead/manager | **Core** |
| `Company` | Công ty tuyển dụng | embedding (name), size | Optional — Phase 2 |
| `Education` | Bằng cấp / trường | embedding (school name), degree_level | Optional — Phase 2 |
| `Industry` | Ngành nghề | embedding (name), one-hot category | Optional — Phase 2 |
| `Language` | Ngôn ngữ (business constraint) | one-hot (EN/VI/JP...), proficiency | Optional — Phase 2 (*) |

> (*) `Language` node KHÔNG dùng cho cross-language embedding matching.
> Chỉ dùng như business constraint (ví dụ: JD yêu cầu English B2).

**Node features chi tiết:**

```
CV node:
  - x: embedding vector (dim=384, all-MiniLM-L6-v2)
  - experience_years: float
  - education_level: int (0=none, 1=college, 2=bachelor, 3=master, 4=phd)

Job node:
  - x: embedding vector (dim=384)
  - salary_min: float (normalized)
  - salary_max: float (normalized)

Skill node:
  - x: embedding vector (dim=384)
  - category: int (0=technical, 1=soft, 2=tool, 3=domain)
  - canonical_name: str (index key)

Seniority node:
  - x: one-hot [intern, junior, mid, senior, lead, manager]  # shape [6, 6]

Company node (optional — Phase 2):
  - x: embedding vector
  - size: int (0=startup, 1=smb, 2=enterprise)

Education node (optional — Phase 2):
  - x: embedding vector (school name)
  - degree_level: int (0=associate, 1=bachelor, 2=master, 3=phd)

Industry node (optional — Phase 2):
  - x: embedding vector
  - category: int (IT/Finance/Marketing/Healthcare/...)
```

#### Edge Types

| Edge | Hướng | Weight | Mô tả | Giai đoạn |
|------|--------|--------|-------|-----------|
| `has_skill` | CV → Skill | proficiency: 1–5 | CV sở hữu kỹ năng này | **Core** |
| `requires_skill` | Job → Skill | importance: 1–5 | Job yêu cầu kỹ năng này | **Core** |
| `has_seniority` | CV → Seniority | — | Cấp độ ứng viên | **Core** |
| `requires_seniority` | Job → Seniority | — | Cấp độ Job yêu cầu | **Core** |
| `match` | CV → Job | — | Label edge — positive | **Core** |
| `no_match` | CV → Job | — | Label edge — negative | **Core** |
| `work_at` | CV → Company | duration_months | Từng làm tại đây | Optional — Phase 2 |
| `posted_by` | Job → Company | — | Job thuộc công ty này | Optional — Phase 2 |
| `has_degree` | CV → Education | gpa: float | Ứng viên tốt nghiệp trường này | Optional — Phase 2 |
| `prefers_degree` | Job → Education | — | Job ưu tiên bằng cấp này | Optional — Phase 2 |
| `belongs_to` | Job → Industry | — | Job thuộc ngành này | Optional — Phase 2 |
| `works_in` | CV → Industry | years: float | Kinh nghiệm trong ngành | Optional — Phase 2 |
| `requires_language` | Job → Language | — | Ngôn ngữ yêu cầu (business) | Optional — Phase 2 |
| `speaks` | CV → Language | proficiency: A1–C2 | Ngôn ngữ ứng viên biết | Optional — Phase 2 |

#### Cấu trúc HeteroData trong PyG

```python
data = HeteroData()

# === CORE (Phase 1 — build ngay) ===
data['cv'].x            = ...  # [num_cvs, 384]
data['job'].x           = ...  # [num_jobs, 384]
data['skill'].x         = ...  # [num_skills, 384]
data['seniority'].x     = ...  # [6, 6] one-hot

data['cv',  'has_skill',          'skill'].edge_index = ...
data['cv',  'has_skill',          'skill'].edge_attr  = ...  # proficiency weight [E]
data['job', 'requires_skill',     'skill'].edge_index = ...
data['job', 'requires_skill',     'skill'].edge_attr  = ...  # importance weight [E]
data['cv',  'has_seniority',      'seniority'].edge_index = ...
data['job', 'requires_seniority', 'seniority'].edge_index = ...
data['cv',  'match',              'job'].edge_index   = ...  # positive labels
data['cv',  'no_match',           'job'].edge_index   = ...  # negative labels

# === OPTIONAL (Phase 2 — uncomment khi implement) ===
# data['company'].x    = ...
# data['education'].x  = ...
# data['industry'].x   = ...
# data['language'].x   = ...  # business constraint only, NOT for embedding matching
# data['cv',  'work_at',           'company'].edge_index  = ...
# data['job', 'posted_by',         'company'].edge_index  = ...
# data['cv',  'has_degree',        'education'].edge_index = ...
# data['job', 'prefers_degree',    'education'].edge_index = ...
# data['cv',  'works_in',          'industry'].edge_index  = ...
# data['job', 'belongs_to',        'industry'].edge_index  = ...
# data['cv',  'speaks',            'language'].edge_index  = ...
# data['job', 'requires_language', 'language'].edge_index  = ...
```

#### Implement theo giai đoạn

```
Phase 1 (Tuần 1–8):
  Nodes: CV, Job, Skill, Seniority
  Edges: has_skill, requires_skill, has_seniority, requires_seniority, match, no_match

Phase 2 (Tuần 9–12):
  + Company, Education, Industry, Language (business constraint)
  + work_at, posted_by, has_degree, belongs_to, works_in, requires_language, speaks
```

---

### 1.2. Skill Normalization Strategy

> **Bắt buộc** — không phải optional. Không normalize thì graph bị duplicate node noise.

- [ ] Xác định canonical skill format (lowercase, no alias):
  - `"ReactJS"`, `"React.js"`, `"React"` → `"react"`
  - `"Node.js"`, `"NodeJS"` → `"nodejs"`
  - `"PostgreSQL"`, `"Postgres"` → `"postgresql"`
  - `"Machine Learning"`, `"ML"` → `"machine_learning"`
- [ ] Xây dựng skill alias dictionary (~100–150 skills phổ biến ban đầu)
- [ ] Pipeline: raw text → extract skills → normalize → lookup canonical → Skill node

---

### 1.3. Label Strategy & Negative Sampling

- [ ] **Nguồn nhãn (giai đoạn đầu — synthetic):**
  - Positive: skill overlap >= 50% AND seniority match → label = 1
  - Negative: skill overlap < 20% OR seniority mismatch >= 2 level → label = 0
  - Hard negative: cùng ngành, seniority match, nhưng thiếu 1–2 skill quan trọng

- [ ] **Tỷ lệ negative sampling:** 1 positive : 3 negatives (1 easy + 2 hard)

- [ ] **Synthetic data generation** (vì chưa có data thật):
  - Generate CV giả: chọn ngẫu nhiên 5–10 skills từ skill list + seniority + experience_years
  - Generate JD giả: chọn 4–8 skills + seniority + salary range
  - Positive pair: CV và JD share >= 3 skills AND seniority match
  - Scale: target ~2.000–5.000 cặp (CV, JD)

- [ ] **Evaluation Metrics** (ranking task):
  - **Recall@K** (K = 5, 10): trong top K JD gợi ý, bao nhiêu cái thực sự match
  - **MRR** (Mean Reciprocal Rank): job phù hợp nhất ở vị trí nào trong ranking
  - **NDCG@K**: chất lượng thứ tự ranking
  - **AUC-ROC**: cho edge classification (match / no_match)

---

### 1.4. Training Objective

- [ ] Dùng **BPR Loss (Bayesian Personalized Ranking)** — pairwise ranking:
  ```
  L = -log(σ(score_positive - score_negative))
  ```
  Ưu điểm hơn BCE: tối ưu trực tiếp ranking thay vì chỉ classification.

- [ ] **Hybrid Scoring** tại inference (không phụ thuộc hoàn toàn GNN):
  ```
  final_score = α × GNN_score
              + β × skill_overlap_ratio
              + γ × seniority_match_score

  eligible    = final_score >= threshold  (default: 0.65)
  ```
  α, β, γ là hyperparameters (bắt đầu: 0.6, 0.3, 0.1) — tune sau khi có evaluation data

---

## Phase 2 — Embedding Layer (English-only, abstraction-ready)

> Không implement multilingual trong MVP. Giữ abstraction để swap sau.
> **Không hardcode** `"english-only"` vào bất kỳ đâu ngoài config.

### Kiến trúc EmbeddingProvider

```
┌─────────────────────────────────────┐
│  Graph Builder / GNN Training       │
│  (chỉ nhận vector[], không biết     │
│   text đến từ ngôn ngữ nào)         │
└──────────────┬──────────────────────┘
               │ np.ndarray [N, dim]
┌──────────────▼──────────────────────┐
│  EmbeddingProvider (abstract)       │
│  ┌──────────────────────────────┐   │
│  │  EnglishProvider  [DEFAULT]  │───┼──▶ all-MiniLM-L6-v2 (80MB, dim=384)
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │  MultilingualProvider [TODO] │───┼──▶ LaBSE / multilingual-e5 (chưa implement)
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

Config swap: `EMBEDDING_PROVIDER=english` hoặc `multilingual`

### Việc cần làm

- [ ] Implement `EmbeddingProvider` abstract base:
  ```python
  # embedding/base.py
  class EmbeddingProvider:
      def encode(self, texts: list[str]) -> np.ndarray:
          raise NotImplementedError

      @property
      def dim(self) -> int:
          raise NotImplementedError
  ```

- [ ] Implement `EnglishProvider` (build ngay):
  ```python
  # embedding/english.py
  # Model: sentence-transformers/all-MiniLM-L6-v2
  # dim = 384, size ~80MB, inference ~0.5ms/text trên CPU
  class EnglishProvider(EmbeddingProvider): ...
  ```

- [ ] Tạo stub `MultilingualProvider` (placeholder — chưa implement):
  ```python
  # embedding/multilingual.py
  # TODO: implement khi có thời gian
  # Candidates: LaBSE (best quality), multilingual-e5 (best retrieval)
  # WARNING: không dùng PhoBERT+BERT riêng → vector không cùng không gian
  class MultilingualProvider(EmbeddingProvider):
      def encode(self, texts):
          raise NotImplementedError("Multilingual layer not yet implemented")
  ```

- [ ] Config:
  ```env
  EMBEDDING_PROVIDER=english   # switch sang "multilingual" khi sẵn sàng
  ```

---

## Phase 3 — Đọc paper theo thứ tự bám bài toán

> Mục tiêu: hiểu đủ để implement, không cần đọc toàn bộ.

- [ ] **PyG — Link Prediction on Heterogeneous Graphs** ← đọc đầu tiên
  - Tài liệu: [PyG Heterogeneous Graph Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html)
  - Focus: HeteroData, to_hetero(), LinkPrediction head

- [ ] **[From Text to Talent](https://arxiv.org/html/2503.17438)** (2025) ← gần nhất với approach của mình
  - LLM extract CV features → Heterogeneous GCN → predict match
  - Đọc: abstract + Section 3 (architecture) + Section 4 (experiments)

- [ ] **[WHIN + CSAGNN](https://arxiv.org/abs/2401.00010)** (2024)
  - RGCN trên heterogeneous graph, 5 node types, 9 edge types
  - Đọc: Section 3 (graph schema) — so sánh với schema của mình

- [ ] **GraphSAGE** — Hamilton et al., 2017
  - Focus: inductive learning — node CV/JD mới không cần retrain toàn graph
  - Đọc: Section 3 (method), skip phần proof

- [ ] **[LinkSAGE](https://arxiv.org/abs/2402.13430)** (LinkedIn, KDD 2025)
  - Production system — đọc để hiểu inference design (precompute embedding, nearline)

- [ ] **GAT** — Veličković et al., 2018
  - Focus: attention → biết skill nào đóng góp vào matching score (explainability)
  - Đọc: Section 2–3

- [ ] **GCN** — Kipf & Welling, 2017 ← đọc nhanh để nắm message passing cơ bản

---

## Phase 4 — Baseline (bắt buộc để defend project)

> GNN phải thắng baseline mới có giá trị. Build baseline trước khi có GNN.

- [ ] **Baseline 1:** Cosine similarity (all-MiniLM-L6-v2 embedding, không GNN)
- [ ] **Baseline 2:** Skill overlap ratio (Jaccard similarity trên skill sets)
- [ ] **Baseline 3:** BM25 trên raw text JD vs CV

---

## Phase 5 — Khởi tạo project & Infrastructure

- [ ] Tạo repository GitHub, thiết lập `.gitignore`, `README.md`
- [ ] Khởi tạo cấu trúc thư mục:
  ```
  jobflow-gnn/
  ├── jobflow-backend/         # FastAPI backend
  │   └── app/
  ├── crawler/                 # Scrapy / Katana
  ├── cv_parser/               # CV parsing module
  ├── embedding/               # EmbeddingProvider abstraction
  │   ├── base.py              # Abstract interface
  │   ├── english.py           # EnglishProvider (all-MiniLM-L6-v2)
  │   └── multilingual.py      # MultilingualProvider stub (TODO)
  ├── graph/                   # Graph construction + GNN
  │   ├── schema.py            # Node/edge type definitions
  │   ├── builder.py           # HeteroData builder
  │   ├── model.py             # GNN model (GraphSAGE)
  │   └── train.py             # Training pipeline (BPR loss)
  ├── email_generator/         # LLM email generation
  ├── frontend/                # Streamlit / React UI
  ├── .env.example
  └── docker-compose.yml
  ```
- [ ] Thiết lập `docker-compose.yml`: FastAPI, PostgreSQL, Redis, Celery worker
- [ ] Cài đặt dependencies (`requirements.txt`): fastapi, sqlalchemy, alembic, torch, pyg, sentence-transformers
- [ ] Thiết lập SQLAlchemy + Alembic, chạy migration đầu tiên
- [ ] Kiểm tra stack chạy với `docker compose up`

---

## Kết quả cần đạt cuối tuần 1

| Hạng mục | Trạng thái |
|----------|------------|
| Phạm vi xác định rõ (AI Matching Engine only, không bao gồm apply) | [ ] |
| Bài toán xác định rõ (ranking/retrieval, không phải classification) | [ ] |
| Output schema chốt (cv_id + score + eligible) | [ ] |
| Threshold strategy xác định (default 0.65, tune sau) | [ ] |
| Inference flow thiết kế xong (precompute CV embedding) | [ ] |
| Graph schema Phase 1 đã chốt (4 node types, 6 edge types) | [ ] |
| Graph schema Phase 2 đã document (optional nodes/edges) | [ ] |
| Skill normalization strategy + alias dict ~100 skills | [ ] |
| Label strategy + negative sampling đã thiết kế | [ ] |
| Synthetic data plan (2.000–5.000 cặp) | [ ] |
| Training objective: BPR loss + hybrid scoring đã chốt | [ ] |
| Evaluation metrics đã chọn (Recall@K, MRR, NDCG) | [ ] |
| EmbeddingProvider interface + EnglishProvider implement | [ ] |
| MultilingualProvider stub tạo sẵn (chưa implement) | [ ] |
| Đã đọc: From Text to Talent, WHIN+CSAGNN, GraphSAGE | [ ] |
| Đã đọc: PyG Heterogeneous Graph tutorial | [ ] |
| Baseline strategy xác định (cosine sim, skill overlap, BM25) | [ ] |
| Project khởi tạo, Docker Compose chạy được | [ ] |

---

## Ghi chú / Vấn đề phát sinh

> _(Ghi lại các vấn đề gặp phải trong tuần để xử lý hoặc điều chỉnh kế hoạch)_
