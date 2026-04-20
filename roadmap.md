# JobFlow-GNN — Roadmap

> **Đề tài:** Xây dựng hệ thống tự động thu thập tin tuyển dụng, phân tích nội dung và đánh giá mức độ phù hợp giữa công việc và hồ sơ ứng viên bằng mô hình Graph Neural Networks (GNN), từ đó hỗ trợ sinh nội dung và tự động gửi email ứng tuyển.

---

## Tổng quan kiến trúc hệ thống

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌───────────┐
│  Job Crawler │───▶│  Data Store  │───▶│ Graph Builder  │───▶│ GNN Model │
│  (Scrapy)    │    │ (PostgreSQL) │    │ (PyG / DGL)    │    │(GraphSAGE)│
└─────────────┘    └──────────────┘    └────────────────┘    └─────┬─────┘
                          ▲                                        │
                   ┌──────┴──────┐                                 ▼
                   │ CV Parser   │                          ┌─────────────┐
                   │ (PDF/Text)  │                          │ Matching API│
                   └─────────────┘                          │  (FastAPI)  │
                                                            └──────┬──────┘
                                                                   │
                                                                   ▼
                                                     ┌──────────────────────┐
                                                     │ Email Generator (LLM)│
                                                     └───────────┬──────────┘
                                                                 │
                                                                 ▼
                                                     ┌──────────────────────┐
                                                     │ Auto Sender (SMTP)   │
                                                     └──────────────────────┘
```

---

## Tổng timeline — 3 tháng (12 tuần)

| Giai đoạn | Thời gian | Nội dung chính |
|-----------|-----------|----------------|
| Tháng 1   | Tuần 1–4  | Nền tảng hệ thống + Thu thập & xử lý dữ liệu |
| Tháng 2   | Tuần 5–8  | Xây dựng đồ thị + Huấn luyện mô hình GNN |
| Tháng 3   | Tuần 9–12 | Tự động hóa + Sinh email + Demo + Báo cáo |

---

# THÁNG 1 — THU THẬP DỮ LIỆU & NỀN TẢNG HỆ THỐNG (Tuần 1–4)

### Mục tiêu:
- Crawl được tin tuyển dụng từ các nguồn thực tế
- Parse và trích xuất thông tin từ CV
- Có bộ dataset sơ bộ phục vụ huấn luyện GNN

---

## Tuần 1 — Thiết kế hệ thống & Khởi tạo dự án

- [ ] Xác định phạm vi & nguồn dữ liệu tuyển dụng
  - TopCV, VietnamWorks, CareerBuilder, LinkedIn (nếu có API)
- [ ] Thiết kế cơ sở dữ liệu (PostgreSQL)
  - Bảng `jobs`: title, company, description, requirements, skills, location, salary, source_url
  - Bảng `cvs`: name, raw_text, parsed_skills, experience_years, education
  - Bảng `skills`: skill_name, category, aliases
  - Bảng `job_skill`, `cv_skill`: quan hệ nhiều-nhiều
- [ ] Thiết kế graph schema cho GNN
  - Node types: `Job`, `CV`, `Skill`
  - Edge types: `CV -[HAS_SKILL]-> Skill`, `Job -[REQUIRES_SKILL]-> Skill`
  - Dự kiến thêm edge: `CV -[MATCH]-> Job` (label cho training)
- [ ] Khởi tạo project
  - Backend: FastAPI
  - Database: PostgreSQL + SQLAlchemy/Alembic
  - Task queue: Celery + Redis
  - Cấu trúc thư mục chuẩn, Docker Compose

---

## Tuần 2 — Module thu thập tin tuyển dụng (Job Crawler)

- [ ] Xây dựng crawler với Scrapy / Selenium
  - Crawl các trường: title, company, description, requirements, location, salary
- [ ] Xử lý & làm sạch dữ liệu
  - Loại bỏ HTML tags, chuẩn hóa text (lowercase, remove special chars)
  - Trích xuất skills từ job description (rule-based + regex)
- [ ] Lưu trữ vào database
  - Deduplication theo source_url
  - Đánh dấu thời gian crawl
- [ ] Lập lịch crawl tự động
  - Celery beat hoặc cron: crawl hàng ngày

---

## Tuần 3 — Module xử lý hồ sơ ứng viên (CV Parser)

- [ ] Xây dựng API upload CV (PDF/DOCX)
- [ ] Trích xuất text từ CV
  - Sử dụng pdfplumber / PyMuPDF cho PDF
  - python-docx cho DOCX
- [ ] Phân tích & trích xuất thông tin có cấu trúc
  - Skills (kỹ năng)
  - Kinh nghiệm làm việc (số năm, vị trí, công ty)
  - Học vấn
  - Thông tin liên hệ
- [ ] Phương pháp trích xuất
  - Giai đoạn đầu: regex + rule-based
  - Nâng cao (nếu có thời gian): NER với spaCy hoặc LLM

---

## Tuần 4 — Xây dựng dataset & Gán nhãn

- [ ] Tạo bộ dataset gồm các cặp (CV, Job)
  - Label: match (1) / not match (0) / partial match (0.5)
- [ ] Xây dựng tiêu chí đánh giá mức độ phù hợp
  - Skill overlap ratio
  - Experience level match
  - Location match
- [ ] Sinh dữ liệu tổng hợp (synthetic data) nếu thiếu
  - Fake CV generator
  - Augmentation từ dữ liệu thật
- [ ] Kiểm tra chất lượng dataset
  - Phân bố label balanced
  - Tối thiểu 500–1000 cặp (CV, Job)

**Output:** Bộ dataset sẵn sàng cho việc xây dựng đồ thị và huấn luyện GNN.

---

# THÁNG 2 — MÔ HÌNH GNN & ĐÁNH GIÁ MỨC ĐỘ PHÙ HỢP (Tuần 5–8)

### Mục tiêu:
- Xây dựng đồ thị tri thức (Knowledge Graph) từ dữ liệu
- Huấn luyện mô hình GNN dự đoán matching score
- Đánh giá hiệu quả mô hình

---

## Tuần 5 — Xây dựng đồ thị (Graph Construction)

- [ ] Xây dựng heterogeneous graph
  - Node `CV`: feature vector từ embedded CV text
  - Node `Job`: feature vector từ embedded job description
  - Node `Skill`: feature vector (one-hot hoặc embedding)
- [ ] Tạo edges
  - `CV -[HAS_SKILL]-> Skill`: trọng số = proficiency level
  - `Job -[REQUIRES_SKILL]-> Skill`: trọng số = importance level
- [ ] Tạo node features
  - Text embedding: Sentence-BERT hoặc PhoBERT (cho tiếng Việt)
  - Skill embedding: Word2Vec hoặc pre-trained
- [ ] Convert sang PyTorch Geometric (PyG) format
  - HeteroData object
  - Train/val/test split

---

## Tuần 6 — Xây dựng mô hình GNN

- [ ] Implement mô hình GNN cho link prediction
  - Kiến trúc chính: GraphSAGE (ổn định, dễ scale)
  - Kiến trúc phụ (so sánh): GCN, GAT
- [ ] Thiết kế mô hình
  - Encoder: 2–3 lớp GNN layers
  - Decoder: dot product hoặc MLP để tính matching score
  - Loss function: Binary Cross Entropy hoặc Margin Ranking Loss
- [ ] Feature engineering
  - Node features: text embeddings + numerical features
  - Edge features (nếu cần): skill proficiency, importance weight
- [ ] Xử lý heterogeneous graph
  - Sử dụng `to_hetero()` hoặc custom message passing

---

## Tuần 7 — Huấn luyện & Đánh giá mô hình

- [ ] Huấn luyện mô hình
  - Optimizer: Adam
  - Learning rate scheduling
  - Early stopping
- [ ] Đánh giá mô hình
  - Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
  - Confusion matrix
- [ ] So sánh với baseline
  - Baseline 1: Cosine similarity (TF-IDF)
  - Baseline 2: Keyword matching
  - Baseline 3: MLP (không dùng graph)
- [ ] Điều chỉnh hyperparameters
  - Số lớp GNN, hidden dimensions, learning rate, dropout
- [ ] Fallback plan
  - Nếu GNN không đạt kết quả tốt: kết hợp GNN + cosine similarity (hybrid)

---

## Tuần 8 — Tích hợp API đánh giá mức độ phù hợp

- [ ] Xây dựng Matching API (FastAPI)
  - `POST /match`: input CV + Job → output matching score + giải thích
  - `GET /recommend/{cv_id}`: trả về danh sách job phù hợp, ranked
- [ ] Lưu kết quả matching vào database
  - Bảng `matches`: cv_id, job_id, score, matched_skills, created_at
- [ ] Tích hợp model inference
  - Load trained GNN model
  - Real-time prediction hoặc batch prediction
- [ ] API documentation (Swagger/OpenAPI)

---

# THÁNG 3 — CẢI THIỆN MÔ HÌNH & HOÀN THIỆN HỆ THỐNG (Tuần 9–12)

### Mục tiêu:
- Nâng cao chất lượng mô hình GNN (AUC-ROC >= 0.80)
- Pipeline end-to-end hoàn chỉnh: crawl → match → sinh email → gửi
- Có demo & báo cáo hoàn chỉnh

### Mục tiêu metrics cho demo:

| Metric | Hiện tại | Mục tiêu | Ý nghĩa |
|--------|----------|-----------|----------|
| AUC-ROC | 0.71 | **0.80+** | Ngưỡng "tốt" trong ML |
| NDCG@10 | 0.78 | **0.85+** | Ranking quality cao |
| Precision@10 | 0.90 | **0.90+** | Giữ nguyên — đã tốt |
| GNN vs Baselines | +15-21% | **+20%+** | Chứng minh GNN có giá trị |

---

## Tuần 9 — Cải thiện mô hình & Evaluation Protocol

### 9.1. Per-CV Evaluation (Ưu tiên cao) ✅ Đã implement
- [x] Implement per-CV evaluation: mỗi CV rank toàn bộ 6,020 jobs
  - Tính Recall@K, NDCG@K, Precision@K riêng cho từng CV
  - Lấy trung bình (macro-average) trên tất cả CV
- [ ] Chạy per-CV evaluation trên checkpoint hiện tại
- [ ] So sánh kết quả per-CV evaluation với global evaluation
- [ ] Phát hiện và phân tích các CV có kết quả kém (edge cases)

### 9.2. Cải thiện Data Pipeline ✅ Đã hoàn thành
- [x] Fix CV parser cho LinkedIn PDF format (experience, education, seniority)
- [x] Mở rộng skill catalog: 208 → 218 skills (+UX/UI, testing, BA, AI/ML terms)
- [x] Re-extract 511 CVs (từ 362) với catalog mới
- [x] Education distribution: COLLEGE 34% → 8%, BACHELOR 58% → 84%

### 9.3. Cải thiện mô hình GNN

> **Xem kế hoạch chi tiết:** [`roadmap/week9/plan/model-improvement.md`](week9/plan/model-improvement.md)

Kế hoạch 2 giai đoạn để nâng AUC-ROC từ 0.71 lên 0.83+:

**Phase 1 — Mức tốt (target AUC-ROC 0.78–0.82):**
- [ ] **P1.1** Tăng NUM_POSITIVE_PAIRS lên 3,500 và giảm NOISE_RATE xuống 0.05
- [ ] **P1.2** Hyperparameter grid search: hidden_channels, num_layers, lr, dropout
- [ ] **P1.3** DropEdge regularization (drop 20% edges/epoch — ngăn overfitting)
- [ ] **P1.4** Curriculum negative sampling (easy → hard theo epoch schedule)
- [ ] **P1.5** Skill co-occurrence edges (PMI-weighted skill-to-skill từ job postings)

**Phase 2 — Mức xuất sắc (target AUC-ROC 0.83–0.87):**
- [ ] **P2.1** Thay BPR → InfoNCE loss (multi-negative contrastive, richer gradient signal)
- [ ] **P2.2** Test HGT model (Heterogeneous Graph Transformer, relation-aware attention)
- [ ] **P2.3** Nâng cấp text embeddings: MiniLM-L6 → nomic-embed-text-v1.5 (8K context)
- [ ] **P2.4** GRACE contrastive pretraining (self-supervised trên unlabeled graph)

**Metrics mục tiêu:**

| Phase | AUC-ROC | NDCG@10 | Per-CV Recall@50 |
|-------|---------|---------|-----------------|
| Hiện tại | 0.71 | 0.78 | ~0.05 (ước tính) |
| Phase 1 | 0.78–0.82 | 0.85+ | 0.10–0.15 |
| Phase 2 | 0.83–0.87 | 0.90+ | 0.15–0.25 |

### 9.4. Pipeline tự động matching
- [ ] Thiết lập ngưỡng matching
  - Score >= 0.8: highly recommended
  - Score 0.5–0.8: potential match
  - Score < 0.5: not recommended

---

## Tuần 10 — Module sinh nội dung email ứng tuyển (Email Generator)

- [ ] Xây dựng email generator sử dụng LLM
  - Tích hợp OpenAI API / local LLM (Ollama)
  - Prompt engineering cho email ứng tuyển chuyên nghiệp
- [ ] Cá nhân hóa nội dung email
  - Đề cập tên công ty, vị trí ứng tuyển
  - Highlight skills phù hợp giữa CV và Job
  - Tóm tắt kinh nghiệm liên quan
- [ ] Template system
  - Nhiều template cho các loại job khác nhau (IT, Marketing, Finance...)
  - Hỗ trợ tiếng Việt & tiếng Anh
- [ ] Review & chỉnh sửa
  - Cho phép user review email trước khi gửi

---

## Tuần 11 — Module tự động gửi email + Giao diện người dùng

### 11.1. Auto Send Email
- [ ] Tích hợp SMTP để gửi email
  - Hỗ trợ Gmail, Outlook
  - Đính kèm CV (PDF)
- [ ] Xây dựng hệ thống hàng đợi
  - Celery task queue cho việc gửi email
  - Rate limiting (tránh bị đánh dấu spam)
  - Retry mechanism cho email thất bại
- [ ] Tracking & logging
  - Trạng thái email: sent, delivered, opened
  - Lịch sử ứng tuyển theo CV

### 11.2. Giao diện người dùng
- [ ] Xây dựng giao diện web (React)
  - Upload CV
  - Xem danh sách job đã crawl
  - Xem kết quả matching (score + matched skills giải thích)
  - Xem & chỉnh sửa email trước khi gửi
  - Dashboard: số job crawled, số email sent, match rate
- [ ] So sánh trực quan GNN vs Baselines trên cùng CV (cho demo)

---

## Tuần 12 — Demo & Báo cáo

- [ ] Chuẩn bị demo
  - Flow hoàn chỉnh: Upload CV → Auto Match → Generate Email → Send
  - Demo so sánh: GNN vs Cosine vs BM25 trên cùng CV
  - Video demo hoặc live demo
- [ ] Viết báo cáo đồ án
  - Chương 1: Giới thiệu đề tài, mục tiêu, phạm vi
  - Chương 2: Cơ sở lý thuyết (GNN, GraphSAGE, NLP, Email Automation)
  - Chương 3: Phân tích & thiết kế hệ thống
  - Chương 4: Triển khai & thực nghiệm
  - Chương 5: Kết quả & đánh giá (bao gồm per-CV evaluation, so sánh baselines)
  - Chương 6: Kết luận & hướng phát triển

---

# Công nghệ sử dụng (Tech Stack)

| Thành phần | Công nghệ |
|------------|-----------|
| Backend API | FastAPI (Python) |
| Database | PostgreSQL + SQLAlchemy |
| Task Queue | Celery + Redis |
| Crawler | Scrapy / Selenium |
| CV Parser | pdfplumber, PyMuPDF, python-docx |
| NLP / Embedding | Sentence-BERT, PhoBERT (tiếng Việt) |
| Graph ML | PyTorch Geometric (PyG) |
| GNN Model | GraphSAGE (chính), GCN/GAT (so sánh) |
| Email Generator | OpenAI API / Ollama (local LLM) |
| Email Sender | SMTP (smtplib) |
| Frontend | Streamlit hoặc React |
| Containerization | Docker + Docker Compose |

---

# Rủi ro & Phương án xử lý

| Rủi ro | Mức độ | Phương án |
|--------|--------|-----------|
| GNN khó huấn luyện / kết quả kém | Cao | Dùng GraphSAGE (đơn giản hơn GAT); fallback: hybrid GNN + cosine similarity |
| Thiếu dữ liệu huấn luyện | Cao | Sinh dữ liệu tổng hợp; dùng keyword matching để auto-label |
| Crawler bị chặn | Trung bình | Sử dụng proxy rotation, rate limiting; demo với dữ liệu local |
| Text tiếng Việt khó xử lý | Trung bình | Sử dụng PhoBERT, underthesea cho tokenization tiếng Việt |
| Email bị đánh dấu spam | Trung bình | Rate limiting, warm-up domain, nội dung cá nhân hóa |
| Model inference chậm | Thấp | Batch prediction, caching kết quả, model optimization |

---

# Mở rộng (nếu còn thời gian)

- [ ] Feedback loop: user accept/reject job → retrain model
- [ ] Recommendation system: gợi ý skill cần bổ sung
- [ ] Dashboard analytics: thống kê tỷ lệ match, response rate
- [ ] Multi-language support: mở rộng sang tiếng Anh hoàn chỉnh
- [ ] Mobile-friendly UI

---

# Tiêu chí đánh giá thành công

| Tiêu chí | Mục tiêu |
|----------|----------|
| Số lượng job crawl được | >= 1000 tin tuyển dụng |
| Độ chính xác matching (F1-Score) | >= 0.75 |
| GNN outperform baseline | Tốt hơn cosine similarity ít nhất 5% |
| Email sinh tự động | Nội dung tự nhiên, chuyên nghiệp |
| End-to-end pipeline | Chạy hoàn chỉnh từ crawl → match → email |
| Thời gian xử lý | Match 1 CV với 100 jobs < 5 giây |
