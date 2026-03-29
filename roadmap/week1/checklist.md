# ml-service Phase 1 + Phase 2 Checklist

## Phase 1 — Data Pipeline (complete)

- [x] **Embedding module** (`ml_service/embedding/`)
  - [x] `base.py` — `EmbeddingProvider` ABC
  - [x] `english.py` — SentenceTransformer (all-MiniLM-L6-v2)
  - [x] `multilingual.py` — Multilingual stub
  - [x] `factory.py` — Provider factory

- [x] **Graph schema** (`ml_service/graph/schema.py`)
  - [x] Node types: CV, Job, Skill, Seniority
  - [x] Edge types: has_skill, requires_skill, has_seniority, requires_seniority, match, no_match
  - [x] Data classes: CVData, JobData, LabeledPair, DatasetSplit

- [x] **Synthetic data generator** (`ml_service/data/generator.py`)
  - [x] `generate_cvs()` — realistic CV profiles
  - [x] `generate_jobs()` — realistic job descriptions

- [x] **Skill normalization** (`ml_service/data/skill_normalization.py`)
  - [x] Alias mapping from `skill-alias.json`
  - [x] Category-based skill catalog

- [x] **Pair labeling** (`ml_service/data/labeler.py`)
  - [x] Threshold-based labeling (skill overlap + seniority distance)
  - [x] 1:1:2 ratio sampling (positive : easy neg : hard neg)
  - [x] Stratified train/val/test split (75/15/10)

- [x] **Graph builder** (`ml_service/graph/builder.py`)
  - [x] CV nodes: embedding(384) + experience_years + education = 386-dim
  - [x] Job nodes: embedding(384) + salary_min_norm + salary_max_norm = 386-dim
  - [x] Skill nodes: embedding(384) + category = 385-dim
  - [x] Seniority nodes: 6x6 identity matrix
  - [x] All edge types with attributes

- [x] **Config** (`ml_service/config/settings.py`)
  - [x] Pydantic Settings with `.env` support
  - [x] Data generation params

- [x] **Tests**: 50 tests passing

---

## Phase 2 — Baselines, GNN Model, Training Pipeline (complete)

- [x] **Evaluation metrics** (`ml_service/evaluation/metrics.py`)
  - [x] `recall_at_k` — fraction of positives in top-k
  - [x] `mrr` — mean reciprocal rank
  - [x] `ndcg_at_k` — normalized discounted cumulative gain
  - [x] `auc_roc` — area under ROC curve
  - [x] `compute_all_metrics` — all-in-one convenience function

- [x] **Baselines** (`ml_service/baselines/`)
  - [x] `base.py` — `Scorer` ABC with `score()` and `score_batch()`
  - [x] `cosine.py` — `CosineSimilarityScorer` (embedding dot product)
  - [x] `skill_overlap.py` — `SkillOverlapScorer` (Jaccard similarity)
  - [x] `bm25.py` — `BM25Scorer` (inline Okapi BM25, no external deps)

- [x] **GNN model** (`ml_service/models/gnn.py`)
  - [x] `MLPDecoder` — concat(cv, job) → hidden → score
  - [x] `prepare_data_for_gnn()` — adds reverse edges via `ToUndirected()`
  - [x] `HeteroGraphSAGE` — per-type projections → `GraphSAGE` + `to_hetero()` → decoder
  - [x] `encode()`/`decode()` split for inference-time precomputation

- [x] **Losses** (`ml_service/models/losses.py`)
  - [x] `bpr_loss` — Bayesian Personalized Ranking: `-logsigmoid(pos - neg).mean()`

- [x] **Training pipeline** (`ml_service/training/trainer.py`)
  - [x] `TrainConfig` — all hyperparameters
  - [x] `TrainResult` — best_epoch, losses, val/test metrics
  - [x] Label edge stripping (prevent leakage)
  - [x] BPR triplet sampling (per-CV negatives with fallback)
  - [x] Training loop with early stopping on val MRR
  - [x] Hybrid scoring evaluation: α×GNN + β×skill_overlap + γ×seniority_match

- [x] **Config update** (`ml_service/config/settings.py`)
  - [x] GNN architecture: hidden_channels(128), num_layers(2)
  - [x] Training: lr(1e-3), weight_decay(1e-5), epochs(50), patience(10)
  - [x] Hybrid weights: alpha(0.6), beta(0.3), gamma(0.1)
  - [x] Eligibility threshold(0.65)

- [x] **Tests**: 52 new tests (102 total), 96% coverage

---

## Verification

```
102 passed, 96% coverage
ruff check: All checks passed
ruff format: All files formatted
```

## Files Created (Phase 2)

| File | Lines | Purpose |
|------|-------|---------|
| `ml_service/evaluation/__init__.py` | 0 | Package init |
| `ml_service/evaluation/metrics.py` | 73 | Ranking metrics |
| `ml_service/baselines/__init__.py` | 0 | Package init |
| `ml_service/baselines/base.py` | 16 | Scorer ABC |
| `ml_service/baselines/cosine.py` | 39 | Cosine similarity |
| `ml_service/baselines/skill_overlap.py` | 14 | Jaccard overlap |
| `ml_service/baselines/bm25.py` | 69 | Okapi BM25 |
| `ml_service/models/__init__.py` | 0 | Package init |
| `ml_service/models/gnn.py` | 90 | HeteroGraphSAGE |
| `ml_service/models/losses.py` | 10 | BPR loss |
| `ml_service/training/__init__.py` | 0 | Package init |
| `ml_service/training/trainer.py` | 265 | Training pipeline |
| `tests/test_metrics.py` | 115 | Metrics tests (18) |
| `tests/test_baselines.py` | 106 | Baseline tests (12) |
| `tests/test_losses.py` | 36 | Loss tests (5) |
| `tests/test_gnn_model.py` | 107 | GNN model tests (9) |
| `tests/test_trainer.py` | 96 | Trainer tests (8) |

## Files Modified (Phase 2)

| File | Change |
|------|--------|
| `ml_service/config/settings.py` | +14 lines: GNN/training hyperparams |
| `.env.example` | +16 lines: Corresponding env vars |
