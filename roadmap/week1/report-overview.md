# Week 1 — Báo cáo tổng quan

## Mục tiêu ban đầu

Thiết kế bài toán, chốt graph schema, chọn embedding model, đọc paper, khởi tạo project.

## Kết quả thực tế đạt được

Vượt kế hoạch — hoàn thành cả Phase 1 (data pipeline) + Phase 2 (GNN model + training) + Phase 3 (crawler + inference).

### Tổng kết modules đã build

| Module | Files | Mô tả |
|--------|-------|-------|
| `config/` | 1 | Pydantic Settings + .env |
| `embedding/` | 4 | EmbeddingProvider ABC, EnglishProvider, MultilingualStub, Factory |
| `graph/` | 2 | Schema (4 node types, 6 edge types) + GraphBuilder (HeteroData) |
| `data/` | 4 | SkillNormalizer, SyntheticDataGenerator, PairLabeler, SkillTaxonomy |
| `baselines/` | 4 | Scorer ABC, CosineSimilarity, SkillOverlap (Jaccard), BM25 |
| `evaluation/` | 1 | Recall@K, MRR, NDCG@K, AUC-ROC |
| `models/` | 2 | HeteroGraphSAGE + MLPDecoder, BPR loss |
| `training/` | 1 | Trainer (BPR + early stopping + hybrid scoring) |
| `crawler/` | 5 | CrawlProvider ABC, JobSpyProvider, SkillExtractor, Storage, Factory |
| `inference/` | 3 | Checkpoint (save/load), InferenceEngine, MatchResult |

### Kết quả benchmark (Real data)

```
Method                          recall@5   recall@10     auc_roc
Cosine Similarity               0.0270      0.0405      0.4530
Skill Overlap (Jaccard)         0.0541      0.0811      0.6639
BM25                            0.0270      0.0270      0.6622
GNN (Hybrid)                    0.0676 *    0.1216 *    0.6432
```

GNN thắng Recall@5 (+25%), Recall@10 (+50%), NDCG@5 (+17%), NDCG@10 (+35%).

### Tests

148 tests, all passing, 0 failures.

### Data

- 315 real JDs crawl từ Indeed (8 queries IT)
- 200 synthetic CVs (skill distribution từ real JDs)
- 85 canonical skills, 79 unique skills tìm thấy trong real data
