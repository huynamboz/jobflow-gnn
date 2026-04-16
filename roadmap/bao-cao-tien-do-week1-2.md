# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN TỐT NGHIỆP

**Đề tài:** Xây dựng hệ thống tự động thu thập tin tuyển dụng, phân tích nội dung và đánh giá mức độ phù hợp giữa công việc và hồ sơ ứng viên bằng mô hình Graph Neural Networks (GNN), hỗ trợ sinh nội dung và tự động gửi email ứng tuyển.

**Sinh viên:** Huy Nam

**Thời gian báo cáo:** Tuần 1 – Tuần 2 (18/03/2026 – 01/04/2026)

---

## 1. Tổng quan tiến độ

| Tuần | Thời gian | Mục tiêu ban đầu | Kết quả thực tế |
|------|-----------|-------------------|------------------|
| **Tuần 1** | 18/03 – 30/03 | Thiết kế bài toán, graph schema, chọn embedding model, khởi tạo project | Vượt kế hoạch — hoàn thành Phase 1 (Data pipeline), Phase 2 (GNN model + training), Phase 3 (Crawler + Inference) |
| **Tuần 2** | 30/03 – 01/04 | Scale dữ liệu thực (LinkedIn), fix lỗi hiệu suất, train reranker | Hoàn thành — fix 5 lỗi critical, cải thiện 80% NDCG@10, tích hợp Django backend |

---

## 2. Tuần 1 — Nền tảng & Xây dựng mô hình (18/03 – 30/03)

### 2.1. Công việc đã thực hiện

#### a) Data Pipeline
- Xây dựng module **Skill Extraction & Normalization**: trích xuất kỹ năng từ JD/CV, chuẩn hóa alias (vd: "reactjs" → "react", "js" → "javascript")
- Xây dựng **Synthetic Data Generator**: tạo 200 CV tổng hợp với phân phối kỹ năng từ dữ liệu thực
- Xây dựng **Pair Labeler**: gán nhãn match/no_match dựa trên skill overlap
- Tổng hợp **85 canonical skills** ban đầu

#### b) Graph Neural Network
- Thiết kế **Graph Schema**: 4 node types (CV, Job, Skill, Seniority), 6 edge types
- Implement **HeteroGraphSAGE** (3 layers, 256 hidden, mean aggregation) + MLP Decoder
- Implement **BPR Loss** (Bayesian Personalized Ranking) cho bài toán ranking
- Xây dựng **Training Pipeline** với hard negative sampling + early stopping

#### c) Job Crawler
- Xây dựng hệ thống crawler theo **Dependency Injection pattern** — dễ mở rộng provider mới
- **JobSpy Provider**: crawl 315 JD thực từ Indeed (8 queries lĩnh vực IT)
- Hỗ trợ deduplication bằng fingerprint

#### d) Inference Engine
- Xây dựng **Inference Pipeline**: load checkpoint → encode → score → rank
- Implement **Checkpoint Manager**: save/load model + metadata
- Xây dựng **Evaluation Module**: Recall@K, NDCG@K, MRR, AUC-ROC

#### e) Baselines
- Implement 3 phương pháp baseline để so sánh:
  - **Cosine Similarity** (embedding-based)
  - **Skill Overlap / Jaccard** (rule-based)
  - **BM25** (text retrieval)

#### f) Testing
- Viết **148 unit tests**, toàn bộ pass, 0 failures

### 2.2. Kết quả benchmark Tuần 1

**Dữ liệu:** 315 JD thực (Indeed) + 200 CV tổng hợp + 85 skills

| Phương pháp | Recall@5 | Recall@10 | AUC-ROC |
|-------------|----------|-----------|---------|
| Cosine Similarity | 0.0270 | 0.0405 | 0.4530 |
| Skill Overlap (Jaccard) | 0.0541 | 0.0811 | 0.6639 |
| BM25 | 0.0270 | 0.0270 | 0.6622 |
| **GNN (Hybrid)** | **0.0676** | **0.1216** | **0.6432** |

**Nhận xét:** GNN vượt trội so với tất cả baselines ở Recall@5 (+25%), Recall@10 (+50%), NDCG@5 (+17%), NDCG@10 (+35%).

---

## 3. Tuần 2 — Scale dữ liệu thực & Tối ưu hóa (30/03 – 01/04)

### 3.1. Mở rộng dữ liệu

| Chỉ số | Tuần 1 | Tuần 2 | Thay đổi |
|--------|--------|--------|----------|
| Số lượng JD | 315 (Indeed) | 6,020 (Indeed + LinkedIn) | +19x |
| Số lượng CV | 200 (synthetic) | 362 (LinkedIn Vietnam thực) | +81%, dữ liệu thực |
| Canonical Skills | 85 | 208 | +145% |
| Labeled Pairs | ~800 | 9,889 | +12x |

**Chi tiết nguồn dữ liệu:**
- **Indeed:** 4,234 JD từ 70 queries IT, thị trường US
- **LinkedIn:** 535 JD từ Vietnam + Remote, 16 queries
- **CV:** 362 hồ sơ IT thực từ LinkedIn Vietnam (6 categories: Frontend, Backend, Fullstack, Data, DevOps, Mobile)
- **Crawl 6 quốc gia:** US, Canada, Finland, Australia, Singapore, Vietnam

### 3.2. Vấn đề gặp phải và cách giải quyết

**Vấn đề:** Khi thêm ~5,000 JD từ LinkedIn, hiệu suất mô hình giảm 21% (AUC-ROC: 0.69 → 0.55)

**Nguyên nhân gốc:** Chuỗi 5 lỗi liên quan nhau

| # | Lỗi | Mức độ | Giải pháp |
|---|------|--------|-----------|
| 1 | Thiếu `extractor.fit()` trên corpus → IDF không được tính | **Critical** | Thêm `extractor.fit(raw_jobs)` trước khi extract |
| 2 | Công thức overlap thiên lệch với JD dài (LinkedIn) | **Critical** | Đổi từ `\|CV∩JD\|/\|JD\|` sang `\|CV∩JD\|/min(\|CV\|,\|JD\|)` (Sørensen-Dice) |
| 3 | Early stopping dựa trên metric không ổn định (`val_mrr`) | **High** | Chuyển sang `val_ndcg@10` (continuous, stable) |
| 4 | Training config không scale cho 6K jobs | **High** | Tăng NUM_POSITIVE_PAIRS (2000→3500), giảm NOISE_RATE (0.10→0.05), tăng patience (50→80) |
| 5 | Skill catalog thiếu | **Medium** | Kiểm tra → 208 skills đã đủ (chỉ thiếu `julia`, `sequelize`) |

**Tác động của việc fix:**
- Positive pairs tăng **4.9x** (2,000 → 9,889) nhờ fix overlap formula
- `best_epoch` chuyển từ **0 → 13** (model thực sự được train)
- Chỉ sửa **3 files, ~20 dòng code** nhưng cải thiện hiệu suất rất lớn

### 3.3. Xây dựng Backend Django

- **Database Schema:** Thiết kế lại hierarchy Platform → Company → Job, training history
- **Authentication:** JWT auth, permissions, admin endpoints
- **API Endpoints:**
  - User API: listing jobs, upload CV, skills, matching
  - Admin API: versioned training, model activation, dashboard
- **Service Layer:** DB ↔ ml_service converters, management commands
- **Crawler Integration:** Management commands cho crawl → DB import
- **Swagger:** API documentation với file upload support

### 3.4. Cải tiến ML Pipeline

- Mở rộng từ **1 provider** (Indeed) lên **4 providers**: Indeed, LinkedIn (Playwright), Adzuna API, Remotive API
- Implement **LinkedIn Crawler** bằng Playwright với auth state, cross-session dedup
- Implement **Two-Stage Reranker:**
  - Stage 1: GNN Hybrid scoring → Top 50 candidates
  - Stage 2: MLP Reranker (20 features) → reorder
  - Platt Calibration → eligibility threshold
- Implement **GNN Inductive Inference**: CV mới upload không cần retrain toàn bộ graph
- Mở rộng skill dictionary: 85 → 143 → 208 canonical skills
- Thêm HeteroRGCN model (alternative), skill-skill co-occurrence edges (PMI-based)

### 3.5. Kết quả benchmark Tuần 2 (sau khi fix)

**Dữ liệu:** 6,020 JD + 362 CV thực + 208 skills

| Phương pháp | Recall@5 | Recall@10 | Precision@5 | Precision@10 | NDCG@10 | AUC-ROC |
|-------------|----------|-----------|-------------|--------------|---------|---------|
| Cosine Similarity | 0.0130 | 0.0130 | 0.6000 | 0.7000 | 0.4441 | 0.5643 |
| Skill Overlap (Jaccard) | 0.0087 | 0.0173 | 0.4000 | 0.4000 | 0.5010 | 0.3753 |
| BM25 | 0.0130 | 0.0130 | 0.6000 | 0.6000 | 0.3188 | 0.5227 |
| **GNN (Hybrid)** | **0.0173** | **0.0390** | **0.8000** | **0.9000** | **0.7799** | **0.7122** |

**GNN vượt trội ở tất cả metrics:**
- **AUC-ROC: 0.7122** — cao nhất, vượt Cosine +15%, BM25 +16%, Jaccard +21%
- **Precision@10: 0.9000** — 9/10 kết quả là phù hợp
- **NDCG@10: 0.7799** — cải thiện 80% so với trước khi fix (0.43 → 0.78)

### 3.6. Demo thực tế

**Input:** CV Frontend Developer, 3 năm kinh nghiệm (React, Vue, TypeScript)

**Top 5 kết quả:**
1. React Developer (score: 0.79) — phù hợp
2. Fullstack Engineer TypeScript (score: 0.79) — phù hợp
3. Software Dev Engineer - Frontend (score: 0.79) — phù hợp
4. Fullstack Engineer TypeScript (score: 0.79) — phù hợp
5. Software Engineer L2 Visualization (score: 0.67) — phù hợp

**Nhận xét:** Không có job AI/ML/Security trong top kết quả → mô hình phân biệt role tốt.

---

## 4. Tổng kết kỹ thuật đã sử dụng

### 4.1. Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Django 5 + Django REST Framework |
| Database | PostgreSQL |
| ML Framework | PyTorch, PyTorch Geometric |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2, 384-dim) |
| GNN Model | HeteroGraphSAGE (primary), HeteroRGCN (alternative) |
| Reranker | MLP (20 features) + Platt Scaling |
| Crawling | Playwright (LinkedIn), JobSpy (Indeed), REST API (Adzuna, Remotive) |
| CV Parsing | pdfplumber (PDF), python-docx (DOCX) |
| Testing | pytest (155+ tests) |

### 4.2. Kiến trúc Two-Stage Ranking Pipeline

```
CV Input (text hoặc file PDF/DOCX)
  ↓
CV Parsing → trích xuất skills: [React, Vue, TypeScript, TailwindCSS, ...]
  ↓
GNN Encoding → 128-dim embedding vector
  ↓
Stage 1 — Retrieve (Hybrid Scoring)
  ├── GNN similarity (cosine giữa CV embedding và Job embedding)
  ├── Semantic skill overlap (dựa trên skill co-occurrence graph)
  └── Seniority matching
  → Top K=50 candidates
  ↓
Stage 2 — Rerank (MLP Reranker)
  └── 20 features: GNN score, skill overlap, text similarity, role match, location, ...
  → Reorder top K
  ↓
Stage 3 — Calibration (Platt Scaling)
  └── Convert scores → [0, 1] probability
  → Eligibility threshold: 0.5
  ↓
Output: Top K jobs với score + eligible/ineligible status
```

### 4.3. Graph Schema

```
4 Node Types:
  - CV      (362 nodes)  — feature: embedding 384-dim
  - Job     (6,020 nodes) — feature: embedding 384-dim
  - Skill   (208 nodes)  — feature: embedding 384-dim
  - Seniority (6 nodes)  — feature: one-hot 6-dim

9 Edge Types:
  Core:    has_skill, requires_skill, has_seniority, requires_seniority, match, no_match
  Enrich:  skill↔skill (co-occurrence PMI), job↔job (similarity), cv↔cv (similarity)
```

---

## 5. Cấu trúc project hiện tại

```
jobflow-gnn/
├── backend/                    # Django Backend + ML Service
│   ├── config/                 # Django settings
│   ├── apps/                   # Django apps
│   │   ├── users/              # Authentication (JWT)
│   │   ├── jobs/               # Job CRUD + Admin
│   │   ├── cvs/                # CV upload + parsing
│   │   ├── skills/             # Skill dictionary
│   │   └── matching/           # CV→Job matching API
│   ├── ml_service/             # ML Library (tích hợp trong Django)
│   │   ├── crawler/            # Multi-provider crawling (4 providers)
│   │   ├── cv_parser/          # PDF/DOCX parsing
│   │   ├── data/               # Skill extraction, normalization, labeling
│   │   ├── embedding/          # Sentence-Transformers
│   │   ├── graph/              # PyG HeteroData builder
│   │   ├── models/             # HeteroGraphSAGE, HeteroRGCN, BPR loss
│   │   ├── training/           # BPR trainer + hard negative sampling
│   │   ├── reranker/           # MLP reranker + Platt calibration
│   │   ├── inference/          # Two-stage ranking engine
│   │   └── evaluation/         # Metrics (Recall, NDCG, MRR, AUC-ROC)
│   └── tests_ml/               # 155+ tests
├── Dataset/                    # Dữ liệu JD (6 quốc gia)
└── roadmap/                    # Documentation + Reports
    ├── week1/                  # Tuần 1: planning, reports, technical docs
    └── week2/                  # Tuần 2: benchmark report
```

---

## 6. Metrics tổng kết

| Chỉ tiêu | Giá trị |
|-----------|---------|
| Tổng số JD thu thập | 6,020 (Indeed + LinkedIn, 6 quốc gia) |
| Tổng số CV thực | 362 (LinkedIn Vietnam, 6 categories IT) |
| Canonical skills | 208 |
| AUC-ROC (GNN) | **0.7122** |
| Precision@10 | **0.9000** |
| NDCG@10 | **0.7799** |
| Số test | 155+ (all passing) |
| Tổng commits | 40+ |

---

## 7. Khó khăn và bài học

### 7.1. Khó khăn
- **Scale dữ liệu:** Thêm dữ liệu LinkedIn khiến hiệu suất giảm mạnh do sự khác biệt phân phối (JD LinkedIn dài hơn, nhiều skills hơn so với Indeed)
- **Labeling tự động:** Thiếu ground truth — phải dùng skill overlap làm proxy label, cần thiết kế công thức cẩn thận
- **Training instability:** Early stopping trên metric nhỏ (MRR với 2 positive pairs) dẫn đến model không được train

### 7.2. Bài học rút ra
- **Data quality > Data quantity:** Fix 20 dòng code (labeling + training config) quan trọng hơn thêm 5K JD mới
- **Metric lựa chọn:** Cần dùng continuous metric (NDCG) thay vì discrete metric (MRR) cho early stopping khi val set nhỏ
- **Size-invariant formulas:** Công thức overlap phải không phụ thuộc vào độ dài JD để hoạt động tốt với nhiều nguồn dữ liệu

---

## 8. Kế hoạch tiếp theo (Tuần 3+)

| Ưu tiên | Nội dung | Kỳ vọng |
|---------|----------|---------|
| P1 | Hyperparameter tuning (hidden channels, num layers) | AUC-ROC: 0.71 → 0.75+ |
| P2 | Thêm feature location-aware matching | Cải thiện chất lượng match cho từng thị trường |
| P3 | Cải thiện evaluation protocol (per-CV evaluation) | Metrics ổn định hơn, phản ánh đúng production |
| P4 | Frontend dashboard + Demo | Có giao diện trực quan cho demo |

---

**Ngày báo cáo:** 04/04/2026
