# Experiment — GNN vs Baselines Benchmark

## Overview

Script `run_experiment.py` chạy toàn bộ pipeline: generate data → build graph → train GNN → evaluate baselines → in bảng so sánh.

```bash
cd ml-service
python run_experiment.py
```

---

## Pipeline

```
Step 1: Load skill normalizer (85 canonical skills)
Step 2: Generate synthetic CVs (300) + JDs (600)
Step 3: Label pairs (800 positive, 2400 negative) + split (75/15/10)
Step 4: Build embeddings (all-MiniLM-L6-v2) + heterogeneous graph
Step 5: Train GNN (HeteroGraphSAGE + BPR loss, early stopping on val MRR)
Step 6: Evaluate 3 baselines on test set
Step 7: Print comparison table
```

---

## Methods so sánh

### Baselines (không dùng GNN)

| Method | Cách tính score | File |
|--------|----------------|------|
| **Cosine Similarity** | Cosine distance giữa embedding CV text và JD text | `baselines/cosine.py` |
| **Skill Overlap (Jaccard)** | \|CV_skills ∩ JD_skills\| / \|CV_skills ∪ JD_skills\| | `baselines/skill_overlap.py` |
| **BM25** | Okapi BM25 — CV text là document, JD text là query | `baselines/bm25.py` |

### GNN (Hybrid Scoring)

```
final_score = α × GNN_score + β × skill_overlap + γ × seniority_match

Default: α=0.6, β=0.3, γ=0.1
```

- **GNN_score**: HeteroGraphSAGE encode graph → MLPDecoder score (cv, job) pair → min-max normalize to [0,1]
- **skill_overlap**: Jaccard similarity trên skill sets
- **seniority_match**: 1.0 nếu match, giảm 0.25 mỗi level chênh lệch

---

## Evaluation Metrics

| Metric | Mô tả | Ý nghĩa |
|--------|--------|---------|
| **Recall@K** | Fraction positives trong top K | Trong K gợi ý, bao nhiêu thực sự match? |
| **MRR** | 1/rank của positive đầu tiên | Positive match xuất hiện ở vị trí nào? |
| **NDCG@K** | Normalized DCG | Chất lượng thứ tự ranking |
| **AUC-ROC** | Area under ROC curve | Khả năng phân biệt match vs no_match |

---

## Kết quả (synthetic data, 300 CVs × 600 JDs)

```
Method                          recall@5   recall@10         mrr      ndcg@5     ndcg@10     auc_roc
----------------------------------------------------------------------------------------------------
Cosine Similarity               0.0250      0.0625      1.0000      0.4704      0.5085      0.5545
Skill Overlap (Jaccard)         0.0625 *    0.1250 *    1.0000      1.0000 *    1.0000 *    0.9591 *
BM25                            0.0500      0.1000      0.5000      0.6608      0.7137      0.8558
GNN (Hybrid)                    0.0375      0.0750      1.0000      0.6844      0.6434      0.6243
```

### Phân tích

**Skill Overlap thắng** — dự đoán được vì:
- Synthetic data label bằng chính skill overlap + seniority distance
- Baseline đang "cheat" vì evaluation rule = labeling rule

**GNN underfitting** — nguyên nhân:
- Loss giảm chậm (0.69 → 0.66), early stop ở epoch 10
- val_mrr = 1.0 ngay epoch 0 (hybrid scoring quá dễ trên synthetic data)
- GNN chưa có đủ signal phức tạp để học vượt rule-based

**MRR = 1.0** cho 3/4 methods → test set quá dễ, positive pair luôn ranked đầu

### Kết luận

Kết quả này là **expected** cho synthetic data. GNN sẽ thể hiện tốt hơn khi:
1. Dùng real data (crawl JD + real CV) với pattern phức tạp hơn
2. Label bằng historical apply/hire thay vì rule-based
3. Thêm noise vào synthetic data

---

## Config

Điều chỉnh trong `run_experiment.py` hoặc `.env`:

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `NUM_CVS` | 300 | Số CV synthetic |
| `NUM_JOBS` | 600 | Số JD synthetic |
| `NUM_POSITIVE_PAIRS` | 800 | Số positive pairs |
| `SEED` | 42 | Random seed |
| `hidden_channels` | 128 | GNN hidden dim |
| `num_layers` | 2 | Số GNN layers |
| `lr` | 1e-3 | Learning rate |
| `epochs` | 50 | Max epochs |
| `patience` | 10 | Early stopping patience |
| `hybrid_alpha` | 0.6 | GNN score weight |
| `hybrid_beta` | 0.3 | Skill overlap weight |
| `hybrid_gamma` | 0.1 | Seniority match weight |
