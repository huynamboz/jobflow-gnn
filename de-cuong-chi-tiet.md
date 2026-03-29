# ĐỀ CƯƠNG CHI TIẾT ĐỒ ÁN TỐT NGHIỆP

**Đề tài:** Xây dựng hệ thống tự động thu thập tin tuyển dụng, phân tích nội dung và đánh giá mức độ phù hợp giữa công việc và hồ sơ ứng viên bằng mô hình Graph Neural Networks (GNN), hỗ trợ sinh nội dung và tự động gửi email ứng tuyển.

---

## 1. Lý do chọn đề tài

### 1.1. Bối cảnh thực tiễn

Thị trường tuyển dụng tại Việt Nam đang phát triển mạnh mẽ với hàng trăm nghìn tin tuyển dụng mới được đăng tải mỗi ngày trên các nền tảng như TopCV, VietnamWorks, CareerBuilder và LinkedIn. Tuy nhiên, quá trình tìm kiếm việc làm hiện tại vẫn tồn tại nhiều hạn chế:

- **Tốn thời gian:** Ứng viên phải dành hàng giờ duyệt qua các trang tuyển dụng, đọc mô tả công việc, và tự đánh giá mức độ phù hợp của bản thân với từng vị trí.
- **Thiếu cá nhân hóa:** Các hệ thống gợi ý việc làm hiện tại chủ yếu dựa trên từ khóa (keyword matching) hoặc độ tương đồng văn bản đơn giản, chưa khai thác được mối quan hệ phức tạp giữa kỹ năng, kinh nghiệm, và yêu cầu công việc.
- **Quy trình ứng tuyển thủ công:** Việc soạn email ứng tuyển, cá nhân hóa nội dung cho từng vị trí, và gửi hồ sơ vẫn chủ yếu được thực hiện thủ công, gây lãng phí thời gian và công sức.

### 1.2. Tính cấp thiết về mặt khoa học

Graph Neural Networks (GNN) là một hướng nghiên cứu đang phát triển mạnh trong lĩnh vực Machine Learning, đặc biệt phù hợp với các bài toán có dữ liệu dạng đồ thị. Trong bài toán đánh giá mức độ phù hợp giữa ứng viên và công việc, dữ liệu tự nhiên có cấu trúc đồ thị:

- **Các thực thể (nodes):** Công việc (Job), Hồ sơ ứng viên (CV), Kỹ năng (Skill) tạo thành một đồ thị tri thức (Knowledge Graph).
- **Các mối quan hệ (edges):** CV sở hữu kỹ năng, công việc yêu cầu kỹ năng, CV phù hợp với công việc — thể hiện các quan hệ phong phú trên đồ thị.

Các phương pháp truyền thống như TF-IDF, cosine similarity hay keyword matching không thể nắm bắt được cấu trúc quan hệ đa chiều này. GNN, đặc biệt là kiến trúc GraphSAGE, cho phép học biểu diễn (representation learning) trên đồ thị không đồng nhất (heterogeneous graph), từ đó mang lại khả năng dự đoán tốt hơn về mức độ phù hợp giữa ứng viên và công việc.

### 1.3. Ý nghĩa thực tiễn

Hệ thống được xây dựng trong đồ án này hướng tới:

- **Tự động hóa toàn bộ quy trình:** từ thu thập tin tuyển dụng, phân tích hồ sơ, đánh giá mức độ phù hợp, đến sinh email ứng tuyển và gửi tự động.
- **Ứng dụng AI vào bài toán thực tế:** kết hợp GNN với NLP và LLM để giải quyết bài toán có giá trị thực tiễn cao trong lĩnh vực tuyển dụng.
- **Tiết kiệm thời gian và nâng cao hiệu quả** cho người tìm việc, đặc biệt là sinh viên mới ra trường và người đang tìm kiếm cơ hội chuyển đổi nghề nghiệp.

---

## 2. Mục tiêu và nhiệm vụ

### 2.1. Mục tiêu tổng quát

Xây dựng một hệ thống end-to-end tự động thu thập tin tuyển dụng, phân tích hồ sơ ứng viên, đánh giá mức độ phù hợp bằng mô hình GNN, và hỗ trợ sinh nội dung email ứng tuyển cá nhân hóa.

### 2.2. Mục tiêu cụ thể

| STT | Mục tiêu | Chỉ tiêu đánh giá |
|-----|----------|-------------------|
| 1 | Thu thập tự động tin tuyển dụng từ các nguồn thực tế | >= 1.000 tin tuyển dụng |
| 2 | Phân tích và trích xuất thông tin có cấu trúc từ CV | Trích xuất chính xác skills, kinh nghiệm, học vấn |
| 3 | Xây dựng đồ thị tri thức biểu diễn quan hệ Job-CV-Skill | Heterogeneous graph với đầy đủ node types và edge types |
| 4 | Huấn luyện mô hình GNN dự đoán mức độ phù hợp | F1-Score >= 0.75; outperform baseline ít nhất 5% |
| 5 | Sinh email ứng tuyển cá nhân hóa bằng LLM | Nội dung tự nhiên, chuyên nghiệp, phù hợp với vị trí |
| 6 | Tự động gửi email ứng tuyển | Pipeline hoàn chỉnh crawl → match → email → send |
| 7 | Xây dựng giao diện người dùng trực quan | Upload CV, xem kết quả matching, quản lý email |

### 2.3. Nhiệm vụ nghiên cứu

1. **Nghiên cứu lý thuyết:**
   - Tìm hiểu về Graph Neural Networks, đặc biệt là kiến trúc GraphSAGE, GCN, GAT.
   - Nghiên cứu các phương pháp biểu diễn văn bản (text embedding) sử dụng Sentence-BERT và PhoBERT cho tiếng Việt.
   - Tìm hiểu về bài toán link prediction trên đồ thị không đồng nhất (heterogeneous graph).
   - Nghiên cứu các phương pháp sinh nội dung tự động sử dụng Large Language Models (LLM).

2. **Phát triển hệ thống:**
   - Xây dựng module thu thập dữ liệu tuyển dụng (Job Crawler) sử dụng Scrapy/Selenium.
   - Xây dựng module phân tích hồ sơ ứng viên (CV Parser).
   - Xây dựng đồ thị tri thức và huấn luyện mô hình GNN.
   - Phát triển API đánh giá mức độ phù hợp (Matching API).
   - Xây dựng module sinh email ứng tuyển (Email Generator) tích hợp LLM.
   - Xây dựng module tự động gửi email (Auto Sender) qua SMTP.
   - Phát triển giao diện người dùng (Frontend).

3. **Thực nghiệm và đánh giá:**
   - Đánh giá hiệu quả mô hình GNN so với các phương pháp baseline.
   - Đánh giá chất lượng email sinh tự động.
   - Kiểm thử toàn bộ pipeline end-to-end.

---

## 3. Phương pháp và công nghệ sử dụng

### 3.1. Phương pháp nghiên cứu

#### 3.1.1. Thu thập và xử lý dữ liệu
- **Web Scraping:** Sử dụng Scrapy và Katana để crawl tin tuyển dụng từ các nguồn: TopCV, VietnamWorks, CareerBuilder.
- **Xử lý văn bản:** Loại bỏ HTML tags, chuẩn hóa text, trích xuất kỹ năng bằng rule-based + regex. Hỗ trợ tiếng Việt bằng thư viện underthesea.
- **CV Parsing:** Trích xuất text từ file PDF (pdfplumber, PyMuPDF) và DOCX (python-docx), sau đó phân tích cấu trúc để lấy thông tin kỹ năng, kinh nghiệm, học vấn.

#### 3.1.2. Xây dựng đồ thị tri thức (Knowledge Graph)
- **Thiết kế đồ thị không đồng nhất (Heterogeneous Graph):**
  - Node types: `Job`, `CV`, `Skill`
  - Edge types: `CV -[HAS_SKILL]-> Skill`, `Job -[REQUIRES_SKILL]-> Skill`, `CV -[MATCH]-> Job`
- **Tạo node features:**
  - Text embedding sử dụng Sentence-BERT (tiếng Anh) và PhoBERT (tiếng Việt)
  - Skill embedding sử dụng Word2Vec hoặc pre-trained embeddings
- **Chuyển đổi sang PyTorch Geometric (PyG) format:** HeteroData object với train/val/test split.

#### 3.1.3. Mô hình Graph Neural Networks
- **Kiến trúc chính: GraphSAGE (Graph Sample and Aggregate)**
  - Ưu điểm: Khả năng mở rộng tốt (inductive learning), phù hợp cho đồ thị lớn, hỗ trợ heterogeneous graph.
  - Cấu trúc: 2-3 lớp GNN layers → Decoder (dot product / MLP) → Matching score.
- **Kiến trúc so sánh:** GCN (Graph Convolutional Network), GAT (Graph Attention Network).
- **Bài toán:** Link prediction — dự đoán xác suất tồn tại cạnh `MATCH` giữa node `CV` và node `Job`.
- **Loss function:** Binary Cross Entropy hoặc Margin Ranking Loss.
- **Đánh giá:** Accuracy, Precision, Recall, F1-Score, AUC-ROC.

#### 3.1.4. Sinh nội dung email ứng tuyển
- **Prompt Engineering:** Thiết kế prompt chuyên biệt cho việc sinh email ứng tuyển, bao gồm thông tin ứng viên, vị trí ứng tuyển, kỹ năng phù hợp.
- **Cá nhân hóa:** Highlight các kỹ năng trùng khớp, tóm tắt kinh nghiệm liên quan, điều chỉnh giọng văn theo loại công việc.
- **Đa ngôn ngữ:** Hỗ trợ sinh email bằng tiếng Việt và tiếng Anh.

#### 3.1.5. So sánh với các phương pháp Baseline
| Phương pháp | Mô tả |
|-------------|--------|
| Baseline 1: TF-IDF + Cosine Similarity | So sánh độ tương đồng văn bản giữa CV và Job description |
| Baseline 2: Keyword Matching | Đếm số kỹ năng trùng khớp giữa CV và yêu cầu công việc |
| Baseline 3: MLP (không dùng graph) | Mạng MLP dựa trên feature concatenation, không khai thác cấu trúc đồ thị |
| **Đề xuất: GNN (GraphSAGE)** | **Học biểu diễn trên đồ thị, khai thác mối quan hệ đa chiều** |

### 3.2. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|------------|-----------|---------|
| **Backend API** | FastAPI (Python) | Xây dựng RESTful API cho toàn bộ hệ thống |
| **Cơ sở dữ liệu** | PostgreSQL + SQLAlchemy + Alembic | Lưu trữ dữ liệu có cấu trúc, ORM và migration |
| **Task Queue** | Celery + Redis | Xử lý tác vụ nền (crawl, matching, gửi email) |
| **Web Crawling** | Scrapy / Katana | Thu thập tin tuyển dụng tự động |
| **CV Parsing** | pdfplumber, PyMuPDF, python-docx | Trích xuất text từ file CV |
| **NLP / Embedding** | Sentence-BERT, PhoBERT | Tạo vector biểu diễn cho văn bản |
| **Graph ML** | PyTorch Geometric (PyG) | Xây dựng đồ thị và huấn luyện mô hình GNN |
| **Mô hình GNN** | GraphSAGE (chính), GCN/GAT (so sánh) | Dự đoán mức độ phù hợp CV-Job |
| **Sinh email** | OpenAI API / Ollama (local LLM) | Sinh nội dung email ứng tuyển cá nhân hóa |
| **Gửi email** | SMTP (smtplib) | Tự động gửi email ứng tuyển |
| **Frontend** | Streamlit hoặc React | Giao diện người dùng |
| **Containerization** | Docker + Docker Compose | Đóng gói và triển khai hệ thống |

---

## 4. Dự kiến kết quả đạt được

### 4.1. Sản phẩm phần mềm

1. **Hệ thống JobFlow-GNN hoàn chỉnh** gồm các module:
   - Module thu thập tin tuyển dụng tự động (Job Crawler)
   - Module phân tích hồ sơ ứng viên (CV Parser)
   - Module xây dựng đồ thị tri thức (Graph Builder)
   - Mô hình GNN đánh giá mức độ phù hợp (Matching Model)
   - API đánh giá và gợi ý việc làm (Matching API)
   - Module sinh email ứng tuyển (Email Generator)
   - Module tự động gửi email (Auto Sender)
   - Giao diện người dùng (Web UI)

2. **Pipeline end-to-end tự động:**
   - Crawl tin tuyển dụng mới hàng ngày
   - Tự động đánh giá mức độ phù hợp cho tất cả CV đang active
   - Sinh email ứng tuyển cá nhân hóa
   - Gửi email tự động theo lịch trình

### 4.2. Kết quả mô hình

| Chỉ tiêu | Mục tiêu dự kiến |
|-----------|-----------------|
| Số lượng tin tuyển dụng thu thập | >= 1.000 tin |
| Số cặp (CV, Job) trong dataset | >= 500-1.000 cặp |
| Độ chính xác matching (F1-Score) | >= 0.75 |
| GNN so với baseline (cosine similarity) | Cải thiện ít nhất 5% |
| Thời gian matching 1 CV với 100 jobs | < 5 giây |
| Chất lượng email sinh tự động | Tự nhiên, chuyên nghiệp, cá nhân hóa |

### 4.3. Kết quả khoa học

- Phân tích và đánh giá hiệu quả của mô hình GNN (GraphSAGE, GCN, GAT) trong bài toán đánh giá mức độ phù hợp ứng viên-công việc.
- So sánh chi tiết giữa phương pháp dựa trên đồ thị (GNN) với các phương pháp truyền thống (TF-IDF, keyword matching, MLP).
- Đề xuất kiến trúc đồ thị tri thức phù hợp cho miền dữ liệu tuyển dụng tại Việt Nam.

### 4.4. Tài liệu

- Báo cáo đồ án tốt nghiệp đầy đủ.
- Tài liệu hướng dẫn cài đặt và sử dụng hệ thống.
- Mã nguồn được tổ chức khoa học trên GitHub.

---

## 5. Nội dung các chương của đồ án

### Chương 1: Tổng quan đề tài

- 1.1. Đặt vấn đề
  - Thực trạng tìm kiếm việc làm hiện nay
  - Hạn chế của các hệ thống gợi ý việc làm truyền thống
  - Nhu cầu tự động hóa quy trình ứng tuyển
- 1.2. Mục tiêu và phạm vi đồ án
  - Mục tiêu tổng quát và cụ thể
  - Phạm vi nghiên cứu và giới hạn
- 1.3. Đối tượng và phương pháp nghiên cứu
  - Đối tượng nghiên cứu: dữ liệu tuyển dụng, hồ sơ ứng viên, mô hình GNN
  - Phương pháp nghiên cứu: thực nghiệm, so sánh, đánh giá
- 1.4. Ý nghĩa khoa học và thực tiễn
- 1.5. Bố cục đồ án

### Chương 2: Cơ sở lý thuyết

- 2.1. Tổng quan về Graph Neural Networks (GNN)
  - 2.1.1. Khái niệm đồ thị và biểu diễn đồ thị
  - 2.1.2. Message Passing Neural Networks
  - 2.1.3. Graph Convolutional Network (GCN)
  - 2.1.4. GraphSAGE (Graph Sample and Aggregate)
  - 2.1.5. Graph Attention Network (GAT)
  - 2.1.6. Heterogeneous Graph Neural Networks
- 2.2. Xử lý ngôn ngữ tự nhiên (NLP) cho bài toán tuyển dụng
  - 2.2.1. Biểu diễn văn bản (Text Representation)
  - 2.2.2. Sentence-BERT và PhoBERT
  - 2.2.3. Named Entity Recognition (NER) cho trích xuất thông tin
- 2.3. Link Prediction trên đồ thị
  - 2.3.1. Định nghĩa bài toán
  - 2.3.2. Các phương pháp giải quyết
  - 2.3.3. Hàm loss và metrics đánh giá
- 2.4. Large Language Models (LLM) và sinh nội dung tự động
  - 2.4.1. Tổng quan về LLM
  - 2.4.2. Prompt Engineering
  - 2.4.3. Ứng dụng LLM trong sinh email chuyên nghiệp
- 2.5. Các nghiên cứu liên quan
  - 2.5.1. Hệ thống gợi ý việc làm (Job Recommendation Systems)
  - 2.5.2. Ứng dụng GNN trong bài toán matching
  - 2.5.3. So sánh và định vị đề tài

### Chương 3: Phân tích và thiết kế hệ thống

- 3.1. Phân tích yêu cầu
  - 3.1.1. Yêu cầu chức năng
  - 3.1.2. Yêu cầu phi chức năng
  - 3.1.3. Biểu đồ Use Case
- 3.2. Kiến trúc tổng thể hệ thống
  - 3.2.1. Sơ đồ kiến trúc
  - 3.2.2. Luồng dữ liệu (Data Flow)
  - 3.2.3. Mô tả các thành phần
- 3.3. Thiết kế cơ sở dữ liệu
  - 3.3.1. Mô hình quan hệ (ERD)
  - 3.3.2. Thiết kế các bảng dữ liệu (jobs, cvs, skills, job_skill, cv_skill, matches)
  - 3.3.3. Chiến lược indexing và tối ưu truy vấn
- 3.4. Thiết kế đồ thị tri thức (Knowledge Graph)
  - 3.4.1. Schema đồ thị: node types, edge types
  - 3.4.2. Chiến lược tạo node features (text embeddings)
  - 3.4.3. Chiến lược tạo edge features và weights
- 3.5. Thiết kế mô hình GNN
  - 3.5.1. Kiến trúc mô hình (Encoder - Decoder)
  - 3.5.2. Lựa chọn hyperparameters
  - 3.5.3. Chiến lược huấn luyện và đánh giá
- 3.6. Thiết kế API
  - 3.6.1. RESTful API endpoints
  - 3.6.2. API documentation (OpenAPI/Swagger)
- 3.7. Thiết kế giao diện người dùng
  - 3.7.1. Wireframe các màn hình chính
  - 3.7.2. Luồng tương tác người dùng

### Chương 4: Triển khai hệ thống

- 4.1. Cài đặt môi trường phát triển
  - 4.1.1. Cấu trúc dự án
  - 4.1.2. Docker Compose và containerization
  - 4.1.3. Quản lý dependencies
- 4.2. Triển khai module thu thập dữ liệu (Job Crawler)
  - 4.2.1. Cài đặt Scrapy spiders và Katana crawler
  - 4.2.2. Xử lý và làm sạch dữ liệu
  - 4.2.3. Deduplication và lập lịch crawl
- 4.3. Triển khai module phân tích CV (CV Parser)
  - 4.3.1. Trích xuất text từ PDF/DOCX
  - 4.3.2. Phân tích cấu trúc và trích xuất thông tin
  - 4.3.3. Trích xuất kỹ năng (rule-based / NER)
- 4.4. Triển khai xây dựng đồ thị và mô hình GNN
  - 4.4.1. Xây dựng heterogeneous graph với PyTorch Geometric
  - 4.4.2. Cài đặt mô hình GraphSAGE
  - 4.4.3. Cài đặt các mô hình so sánh (GCN, GAT)
  - 4.4.4. Huấn luyện và lưu mô hình
- 4.5. Triển khai Matching API
  - 4.5.1. API endpoint matching và recommendation
  - 4.5.2. Model inference và caching
- 4.6. Triển khai module sinh email và gửi tự động
  - 4.6.1. Tích hợp LLM cho sinh nội dung email
  - 4.6.2. Hệ thống template đa ngôn ngữ
  - 4.6.3. SMTP integration và rate limiting
  - 4.6.4. Task queue và lập lịch tự động
- 4.7. Triển khai giao diện người dùng
  - 4.7.1. Các màn hình chính
  - 4.7.2. Tích hợp với Backend API

### Chương 5: Thực nghiệm và đánh giá

- 5.1. Mô tả dataset
  - 5.1.1. Thống kê dữ liệu thu thập
  - 5.1.2. Phân bố dữ liệu và tiền xử lý
  - 5.1.3. Chiến lược gán nhãn và tạo dữ liệu tổng hợp
- 5.2. Thiết lập thực nghiệm
  - 5.2.1. Môi trường thực nghiệm (phần cứng, phần mềm)
  - 5.2.2. Phân chia tập train/validation/test
  - 5.2.3. Hyperparameters và cấu hình huấn luyện
- 5.3. Kết quả thực nghiệm
  - 5.3.1. Kết quả mô hình GraphSAGE
  - 5.3.2. Kết quả các mô hình so sánh (GCN, GAT)
  - 5.3.3. Kết quả các baseline (TF-IDF, Keyword Matching, MLP)
  - 5.3.4. Bảng so sánh tổng hợp (Accuracy, Precision, Recall, F1, AUC-ROC)
- 5.4. Phân tích và thảo luận
  - 5.4.1. So sánh hiệu quả GNN với các phương pháp baseline
  - 5.4.2. Ảnh hưởng của hyperparameters
  - 5.4.3. Phân tích lỗi (Error Analysis)
  - 5.4.4. Confusion Matrix và các trường hợp đặc biệt
- 5.5. Đánh giá hệ thống tổng thể
  - 5.5.1. Hiệu năng pipeline end-to-end
  - 5.5.2. Đánh giá chất lượng email sinh tự động
  - 5.5.3. Đánh giá trải nghiệm người dùng

### Chương 6: Kết luận và hướng phát triển

- 6.1. Kết luận
  - 6.1.1. Tóm tắt kết quả đạt được
  - 6.1.2. Đóng góp của đồ án
  - 6.1.3. Hạn chế
- 6.2. Hướng phát triển
  - 6.2.1. Feedback loop: Cải thiện mô hình dựa trên phản hồi người dùng (accept/reject)
  - 6.2.2. Recommendation system: Gợi ý kỹ năng cần bổ sung cho ứng viên
  - 6.2.3. Dashboard analytics: Thống kê tỷ lệ match, response rate
  - 6.2.4. Mở rộng đa ngôn ngữ và đa nguồn dữ liệu
  - 6.2.5. Triển khai trên cloud và tối ưu hiệu năng cho quy mô lớn

---

## Tài liệu tham khảo (dự kiến)

1. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS.
2. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks* (GCN). ICLR.
3. Veličković, P., et al. (2018). *Graph Attention Networks* (GAT). ICLR.
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
5. Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained Language Models for Vietnamese*. EMNLP Findings.
6. Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop.
7. Zhang, M., & Chen, Y. (2018). *Link Prediction Based on Graph Neural Networks*. NeurIPS.

---

## 6. Dự kiến tiến độ thực hiện

| Tuần | Nội dung công việc | Kết quả dự kiến |
|------|-------------------|-----------------|
| **Tuần 1** | Nghiên cứu tổng quan, khảo sát các hệ thống liên quan | Báo cáo khảo sát, xác định phạm vi & nguồn dữ liệu tuyển dụng |
| **Tuần 2** | Thiết kế kiến trúc hệ thống & cơ sở dữ liệu | Thiết kế DB (PostgreSQL), graph schema (Job, CV, Skill), khởi tạo project (FastAPI, Docker Compose) |
| **Tuần 3** | Xây dựng module thu thập tin tuyển dụng (Job Crawler) | Crawler hoạt động với Scrapy/Katana, crawl được dữ liệu từ TopCV, VietnamWorks |
| **Tuần 4** | Xử lý dữ liệu crawl & lập lịch tự động | Làm sạch dữ liệu, trích xuất skills từ job description, deduplication, lập lịch crawl tự động |
| **Tuần 5** | Xây dựng module xử lý hồ sơ ứng viên (CV Parser) | API upload CV (PDF/DOCX), trích xuất skills, kinh nghiệm, học vấn từ CV |
| **Tuần 6** | Xây dựng dataset & Gán nhãn | Bộ dataset >= 500–1.000 cặp (CV, Job) với label match/not match, dữ liệu tổng hợp nếu cần |
| **Tuần 7** | Xây dựng đồ thị tri thức (Graph Construction) | Heterogeneous graph với node features (Sentence-BERT/PhoBERT), chuyển đổi sang PyG format |
| **Tuần 8** | Xây dựng mô hình GNN | Cài đặt GraphSAGE (chính), GCN/GAT (so sánh); thiết kế Encoder-Decoder cho link prediction |
| **Tuần 9** | Huấn luyện & Đánh giá mô hình | Huấn luyện mô hình, đánh giá F1 >= 0.75, so sánh với baseline (TF-IDF, Keyword Matching, MLP) |
| **Tuần 10** | Tích hợp Matching API & Pipeline tự động | API matching/recommendation (FastAPI), pipeline end-to-end: crawl → match → ranking |
| **Tuần 11** | Module sinh email ứng tuyển (Email Generator) | Tích hợp LLM (OpenAI/Ollama), sinh email cá nhân hóa đa ngôn ngữ (Việt/Anh) |
| **Tuần 12** | Module tự động gửi email (Auto Sender) | SMTP integration, task queue (Celery), rate limiting, lập lịch tự động hàng ngày |
| **Tuần 13** | Xây dựng giao diện người dùng (Frontend) | Web UI hoàn chỉnh (Streamlit/React): upload CV, xem matching, quản lý email |
| **Tuần 14** | Kiểm thử tổng thể & Chuẩn bị demo | Kiểm thử end-to-end, sửa lỗi, tối ưu hiệu năng, chuẩn bị demo |
| **Tuần 15** | Hoàn thiện báo cáo & Bảo vệ đồ án | Hoàn thiện báo cáo đồ án, slide thuyết trình, video demo |

### Tổng hợp theo giai đoạn

| Giai đoạn | Tuần | Trọng tâm |
|-----------|------|-----------|
| **Giai đoạn 1** — Nghiên cứu & Nền tảng | Tuần 1–4 | Khảo sát, thiết kế hệ thống, crawl dữ liệu tuyển dụng |
| **Giai đoạn 2** — Dữ liệu & Đồ thị | Tuần 5–7 | Parse CV, xây dựng dataset, xây dựng đồ thị tri thức |
| **Giai đoạn 3** — Mô hình GNN | Tuần 8–10 | Huấn luyện GNN, đánh giá, tích hợp Matching API |
| **Giai đoạn 4** — Tự động hóa & Hoàn thiện | Tuần 11–15 | Sinh email, gửi email, UI, kiểm thử, demo, báo cáo |

---

*Ngày lập: 25/03/2026*
