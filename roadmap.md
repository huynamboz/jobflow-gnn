# 🗓️ Tổng timeline 3 tháng (12 tuần)

* **Tháng 1:** Foundation + Data + Crawler
* **Tháng 2:** AI (GNN) + Matching
* **Tháng 3:** Automation + Email + Demo + Report

---

# 📦 THÁNG 1 — DATA + SYSTEM FOUNDATION (Tuần 1–4)

## 🎯 Goal:

* Crawl được job
* Parse được CV
* Có dataset sơ bộ

---

## ✅ Tuần 1 — Setup + Design

* [ ] Xác định scope (job sources: TopCV / VietnamWorks / LinkedIn clone API)

* [ ] Design database:

  * Job
  * CV
  * Skill

* [ ] Define graph schema:

  * Node: Job, CV, Skill
  * Edge:

    * CV → Skill
    * Job → Skill

* [ ] Setup project:

  * Backend (FastAPI / Django)
  * DB (PostgreSQL)

---

## ✅ Tuần 2 — Job Crawler

* [ ] Crawl job:

  * title, description, requirement
* [ ] Clean data (remove HTML, normalize text)
* [ ] Store vào DB

👉 Bonus:

* schedule crawl mỗi ngày

---

## ✅ Tuần 3 — CV Processing

* [ ] Upload CV (PDF)
* [ ] Extract text (pdfplumber / PyMuPDF)
* [ ] Extract:

  * skills
  * experience
  * keywords

👉 Có thể dùng:

* regex + rule-based (ban đầu)

---

## ✅ Tuần 4 — Dataset + Label

* [ ] Tạo dataset:

  * (CV, Job) → label (match / not match)
* [ ] Tạo synthetic data nếu thiếu

👉 Output:

* Dataset cho GNN

---

# 🧠 THÁNG 2 — AI (GNN MATCHING) (Tuần 5–8)

## 🎯 Goal:

* Train được model GNN
* Predict matching score

---

## ✅ Tuần 5 — Graph Construction

* [ ] Build graph:

  * Node:

    * CV
    * Job
    * Skill
  * Edge:

    * CV–Skill
    * Job–Skill

* [ ] Convert sang format:

  * PyTorch Geometric / DGL

---

## ✅ Tuần 6 — GNN Model

* [ ] Implement model:

  * GCN / GraphSAGE (khuyên dùng GraphSAGE)
* [ ] Input:

  * embedding của CV + Job + Skill
* [ ] Output:

  * matching score

---

## ✅ Tuần 7 — Training + Evaluation

* [ ] Train model
* [ ] Evaluate:

  * accuracy
  * precision / recall

👉 Nếu yếu:

* fallback = keyword matching + ML

---

## ✅ Tuần 8 — API Integration

* [ ] Build API:

  * input: CV + Job
  * output: score
* [ ] Save score vào DB

---

# ⚙️ THÁNG 3 — AUTOMATION + DEMO (Tuần 9–12)

## 🎯 Goal:

* Full system chạy end-to-end
* Có demo thật

---

## ✅ Tuần 9 — Matching Pipeline

* [ ] Auto match:

  * CV → list job phù hợp
* [ ] Ranking job

---

## ✅ Tuần 10 — Email Generator

* [ ] Generate email:

  * template + LLM
* [ ] Customize theo job

👉 Ví dụ:

* mention company name
* mention skill match

---

## ✅ Tuần 11 — Auto Apply System

* [ ] SMTP send mail
* [ ] Queue system (Celery / cron)
* [ ] Schedule:

  * mỗi ngày apply job mới

---

## ✅ Tuần 12 — UI + Report + Demo

* [ ] Simple UI:

  * upload CV
  * show job matched

* [ ] Demo flow:

  * upload CV → match → auto email

* [ ] Viết report:

  * intro
  * methodology (GNN)
  * result

---

# 🧱 Kiến trúc tổng (để bạn nhớ khi làm)

```
Crawler → DB → Graph Builder → GNN → Matching API → Email Generator → Auto Sender
```

---

# ⚠️ Risk & cách xử lý (rất quan trọng)

## ❌ GNN khó train

👉 Giải pháp:

* dùng GraphSAGE (đơn giản hơn GAT)
* fallback: cosine similarity

---

## ❌ Thiếu dataset

👉 Giải pháp:

* tự generate CV + job
* dùng keyword matching để label

---

## ❌ Crawl bị block

👉 Giải pháp:

* crawl ít, demo local data

---

# 🚀 Bonus (nếu còn thời gian)

* [ ] Feedback loop:

  * user accept/reject → retrain model
* [ ] Recommendation system
* [ ] Dashboard analytics

---

# 🎯 Kết luận (rất quan trọng)

Plan này đảm bảo bạn có:
✔ AI thật (GNN)
✔ System thật
✔ Demo thật
✔ Không quá tải trong 3 tháng
