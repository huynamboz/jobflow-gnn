# Week 1 — Kiến trúc hệ thống

## Tổng quan

```
ml-service/
├── ml_service/
│   ├── config/          Settings (Pydantic + .env)
│   ├── embedding/       EmbeddingProvider (pluggable)
│   ├── data/            Skill normalization, generator, labeler, taxonomy
│   ├── graph/           Schema + GraphBuilder (PyG HeteroData)
│   ├── models/          HeteroGraphSAGE + BPR loss
│   ├── baselines/       Cosine, SkillOverlap, BM25
│   ├── evaluation/      Recall@K, MRR, NDCG@K, AUC-ROC
│   ├── training/        Trainer (BPR + early stopping + hybrid eval)
│   ├── crawler/         CrawlProvider (DI) + JobSpy + SkillExtractor
│   ├── inference/       Checkpoint + InferenceEngine
│   └── utils/           Logging
├── tests/               148 tests
├── run_crawl.py         Crawl real JDs
├── run_experiment.py    Benchmark synthetic data
├── run_experiment_real.py  Benchmark real data
├── run_train_save.py    Train + save checkpoint
└── run_inference_demo.py   Demo inference
```

## Design Patterns

### 1. Dependency Injection (DI) — Provider Pattern

Dùng xuyên suốt cho extensibility:

```
EmbeddingProvider (ABC)
├── EnglishProvider        ← default
└── MultilingualProvider   ← stub (TODO)

CrawlProvider (ABC)
├── JobSpyProvider         ← default (Indeed)
└── (AdzunaProvider)       ← future

Scorer (ABC)
├── CosineSimilarityScorer
├── SkillOverlapScorer
└── BM25Scorer
```

Thêm provider mới = implement ABC + `register_provider()`. Không sửa code hiện tại.

### 2. Factory Pattern

```python
# Embedding
from ml_service.embedding import get_provider
provider = get_provider()          # → EnglishProvider
provider = get_provider("multilingual")  # → MultilingualProvider (khi implement)

# Crawler
from ml_service.crawler import get_provider
crawler = get_provider("jobspy")   # → JobSpyProvider
```

### 3. Frozen Dataclasses

Tất cả data objects đều immutable:
- `CVData`, `JobData`, `LabeledPair`, `DatasetSplit`
- `RawJob`, `MatchResult`

### 4. Encode/Decode Split (GNN)

```python
# Training: end-to-end
scores = model(data, cv_idx, job_idx)

# Inference: precompute once, decode many
z_dict = model.encode(data)              # expensive, chạy 1 lần
scores = model.decode(z_dict, cv_i, job_i)  # cheap, chạy N lần
```

## Data Flow

```
Crawl:    Indeed → JobSpy → RawJob → JSONL storage
Extract:  RawJob → SkillExtractor → JobData (skills, seniority, salary)
Generate: Real JD skill freq → Synthetic CVData
Label:    (CV, JD) pairs → skill overlap + seniority → positive/negative
Graph:    CVs + JDs + Skills + Seniority → HeteroData (PyG)
Train:    HeteroData → HeteroGraphSAGE + BPR loss → checkpoint
Infer:    JD text → SkillExtractor → score vs precomputed CVs → Top K results
```

## Graph Schema (Phase 1)

```
CV ──has_skill──► Skill         (weight: proficiency 1–5)
Job ──requires_skill──► Skill   (weight: importance 1–5)
CV ──has_seniority──► Seniority
Job ──requires_seniority──► Seniority
CV ──match──► Job               (label: positive)
CV ──no_match──► Job            (label: negative)
```

Node dimensions: CV(386), Job(386), Skill(385), Seniority(6).

## Hybrid Scoring

```
final_score = 0.8 × GNN_score + 0.15 × skill_overlap + 0.05 × seniority_match
eligible = final_score >= 0.65
```
