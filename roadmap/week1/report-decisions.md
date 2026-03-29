# Week 1 — Quyết định kiến trúc quan trọng

## 1. English-first, Multilingual-ready

**Quyết định:** MVP chỉ support tiếng Anh. Multilingual là optional layer.

**Lý do:**
- Giảm complexity cho MVP
- LaBSE/multilingual-e5 nặng (~1.8GB), chưa cần thiết
- Nếu dùng PhoBERT (VI) + BERT (EN) riêng → vector không cùng không gian → GNN fail silently

**Cách implement:**
- `EmbeddingProvider` abstract interface
- `EnglishProvider` (all-MiniLM-L6-v2, 80MB) là default
- `MultilingualProvider` stub (raise NotImplementedError)
- Swap bằng `EMBEDDING_PROVIDER=multilingual` trong .env
- Graph Builder chỉ nhận vector, không biết ngôn ngữ

---

## 2. Job-driven retrieval (không phải classification)

**Quyết định:** Bài toán là ranking/retrieval, output = Top K CVs + score + eligible.

**Lý do:**
- HR cần danh sách ranked, không phải yes/no
- Ranking metrics (Recall@K, MRR, NDCG) phù hợp hơn accuracy/F1
- BPR loss tối ưu ranking trực tiếp, tốt hơn BCE

**Output schema:**
```json
[
  {"cv_id": 123, "score": 0.82, "eligible": true},
  {"cv_id": 456, "score": 0.61, "eligible": true}
]
```

---

## 3. Hybrid scoring (không phụ thuộc 100% GNN)

**Quyết định:** `score = α×GNN + β×skill_overlap + γ×seniority_match`

**Lý do:**
- GNN score alone không đủ tin cậy trên small data
- Skill overlap là strong baseline, nên leverage thay vì bỏ
- Seniority match là hard constraint (intern apply senior → reject)
- Weights tunable qua config

**Giá trị cuối cùng:** α=0.8, β=0.15, γ=0.05 (GNN dominant, features bổ trợ)

---

## 4. GraphSAGE thay vì GCN/GAT

**Quyết định:** Dùng GraphSAGE làm backbone chính.

**Lý do:**
- Inductive learning: CV/JD mới không cần retrain toàn graph
- LinkedIn dùng GraphSAGE ở production (LinkSAGE, 1B+ nodes)
- Mean aggregation đơn giản, ổn định cho MVP
- PyG `to_hetero()` hỗ trợ tốt

---

## 5. DI pattern cho crawler

**Quyết định:** `CrawlProvider` ABC + Factory pattern, giống embedding module.

**Lý do:**
- Cần integrate nhiều nguồn (Indeed, Adzuna, TopCV, etc.)
- Mỗi nguồn có API/scraping khác nhau
- `register_provider("adzuna", AdzunaProvider)` — thêm nguồn mới = 1 file

**Interface:**
```python
class CrawlProvider(ABC):
    def name(self) -> str: ...
    def fetch(self, search_term, location, results_wanted) -> list[RawJob]: ...
```

---

## 6. JSONL storage thay vì PostgreSQL

**Quyết định:** Dùng JSONL (append-only) cho raw crawled data, chưa setup PostgreSQL.

**Lý do:**
- MVP cần iterate nhanh, JSONL đủ cho <10K records
- Không cần query phức tạp ở giai đoạn này
- Dedup bằng source_url đủ tốt
- PostgreSQL sẽ add khi cần API endpoint + concurrent access

---

## 7. Precompute CV embeddings cho inference

**Quyết định:** Encode full graph 1 lần, lookup embedding khi score.

**Lý do:**
- `model.encode()` chạy GNN qua toàn graph → tốn compute
- `model.decode()` chỉ MLP forward → rất nhanh
- CV pool ít thay đổi → precompute hợp lý
- New JD chỉ cần text embedding + hybrid scoring

---

## 8. Noisy synthetic data thay vì chỉ dùng clean data

**Quyết định:** Thêm 4 loại noise (implicit skills, synonyms, clusters, label flip).

**Lý do:**
- Clean synthetic data → circular evaluation → benchmark vô nghĩa
- Noise tạo gap giữa structured data (mà baseline thấy) và ground truth (mà label phản ánh)
- GNN khai thác text embeddings để "recover" implicit skills → advantage thực sự
- 12% label noise giống real world (recruiter reject candidate phù hợp)
