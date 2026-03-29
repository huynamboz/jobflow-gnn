# Week 1 — Công việc tiếp theo

## Đã hoàn thành

- [x] Graph schema (4 node, 6 edge)
- [x] Skill alias dictionary (85 canonical skills)
- [x] Synthetic data pipeline (clean + noisy)
- [x] EmbeddingProvider (English, multilingual stub)
- [x] GraphBuilder (HeteroData)
- [x] GNN model (HeteroGraphSAGE + BPR loss)
- [x] Training pipeline (early stopping, hybrid scoring)
- [x] Baselines (cosine, Jaccard, BM25)
- [x] Evaluation metrics (Recall@K, MRR, NDCG@K, AUC-ROC)
- [x] Crawler module (JobSpy / Indeed, DI pattern)
- [x] 315 real JDs crawled
- [x] Benchmark: GNN outperform baselines trên real data
- [x] Inference pipeline (checkpoint save/load, InferenceEngine)
- [x] Demo inference (4 queries, kết quả hợp lý)
- [x] 148 tests, all passing

## Tiếp theo (ưu tiên cao → thấp)

### 1. FastAPI endpoint
- `POST /match` — input JD text → output ranked CVs
- `POST /cv/upload` — add CV vào pool
- `GET /health` — model status
- Wrap InferenceEngine vào REST API

### 2. Crawl thêm data
- Thêm Adzuna API provider (free, structured)
- Thêm Remotive API (remote tech jobs)
- Fix false positive skill "c" (cần min-length hoặc context filter)
- Target: 1.000+ JDs

### 3. CV Parser module
- Upload PDF/DOCX → extract text → skills → CVData
- pdfplumber + python-docx
- Có real CVs → retrain → kết quả thực tế hơn

### 4. Phase 2 Graph nodes
- Thêm Company, Education, Industry, Language nodes
- Thêm edges: work_at, posted_by, has_degree, etc.

### 5. Model improvements
- Thử GAT (attention → explainability)
- Increase training data (more CVs, more JDs)
- Fine-tune hybrid weights trên validation set
- Save best model checkpoint automatically

## Known issues

| Issue | Severity | Status |
|-------|----------|--------|
| Skill "c" false positive | Medium | Known, workaround: filter jobs >= 2 skills |
| numpy version warning (scipy) | Low | Cosmetic, no functional impact |
| jobspy regex version conflict | Low | Works despite warning |
| Multilingual not implemented | Low | Stub ready, implement khi cần |
