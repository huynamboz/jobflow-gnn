# ml-service — AI Matching Engine

Core ML module cho hệ thống JobFlow-GNN. Xây dựng heterogeneous graph từ dữ liệu CV/JD, huấn luyện GNN để học embedding chất lượng cao, phục vụ bài toán job-driven retrieval.

## Bài toán

```
Input:  Job Description (JD) mới
Output: Top K CV phù hợp + score + eligible flag
```

GNN dùng ở training time để học embedding tốt hơn. Inference dùng precomputed CV embedding + hybrid scoring.

## Kiến trúc

```
ml-service/
├── ml_service/
│   ├── config/              # Cấu hình (Pydantic Settings + .env)
│   │   └── settings.py
│   ├── data/                # Dữ liệu & tiền xử lý
│   │   ├── skill_normalization.py   # Canonical skill mapping (85 skills)
│   │   ├── generator.py             # Synthetic CV/JD generator
│   │   └── labeler.py               # Rule-based labeling + negative sampling
│   ├── embedding/           # Text embedding (pluggable)
│   │   ├── base.py          # Abstract EmbeddingProvider
│   │   ├── english.py       # all-MiniLM-L6-v2 (default, dim=384)
│   │   ├── multilingual.py  # Stub — LaBSE/multilingual-e5 (TODO)
│   │   └── factory.py       # Registry pattern provider factory
│   ├── graph/               # PyTorch Geometric graph
│   │   ├── schema.py        # Node/Edge types, dataclasses, enums
│   │   └── builder.py       # HeteroData builder
│   └── utils/
│       └── logging.py
├── tests/                   # 50 tests, all passing
├── pyproject.toml
└── .env.example
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

**4 Node types:**

| Node | Features | Dim |
|------|----------|-----|
| CV | embedding + experience_years + education_level | 386 |
| Job | embedding + salary_min + salary_max | 386 |
| Skill | embedding + category | 385 |
| Seniority | one-hot (6 levels) | 6 |

## Modules

### `data/skill_normalization.py`

Maps raw skill mentions to canonical names. Case-insensitive, alias-based lookup.

```python
normalizer = SkillNormalizer("skill-alias.json")
normalizer.normalize("ReactJS")    # → "react"
normalizer.normalize("Node.js")    # → "nodejs"
normalizer.normalize("unknown")    # → None
```

85 canonical skills, 4 categories: technical (0), soft (1), tool (2), domain (3).

### `data/generator.py`

Generates synthetic CV and JD data with realistic distributions.

```python
gen = SyntheticDataGenerator(normalizer, seed=42)
cvs = gen.generate_cvs(800)     # → list[CVData]
jobs = gen.generate_jobs(1500)   # → list[JobData]
```

- CV: 4–12 skills, seniority-weighted distributions, experience/education mapping
- JD: 3–8 skills, no intern-level JDs, salary ranges by seniority

### `data/labeler.py`

Rule-based labeling with stratified negative sampling.

```python
labeler = PairLabeler(cvs, jobs, seed=42)
pairs = labeler.create_pairs(num_positive=2000)  # ratio 1:3 (pos:neg)
split = labeler.split(pairs)                      # 75/15/10 train/val/test
```

| Label | Condition |
|-------|-----------|
| Positive | skill_overlap >= 0.5 AND seniority_distance <= 1 |
| Easy negative | overlap < 0.2 OR distance >= 3 |
| Hard negative | 0.2 <= overlap < 0.5 AND distance <= 1 |

### `embedding/`

Pluggable embedding providers. Switch via `EMBEDDING_PROVIDER` env var.

```python
from ml_service.embedding import get_provider

provider = get_provider()           # → EnglishProvider (default)
vectors = provider.encode(["text"]) # → np.ndarray (N, 384)
```

| Provider | Model | Status |
|----------|-------|--------|
| `english` | all-MiniLM-L6-v2 (80MB, dim=384) | Implemented |
| `multilingual` | LaBSE / multilingual-e5 | Stub (TODO) |

### `graph/schema.py`

Enums, dataclasses, mapping tables.

```python
from ml_service.graph.schema import NodeType, EdgeType, SeniorityLevel

NodeType.CV          # 4 node types
EdgeType.HAS_SKILL   # 6 edge types
SeniorityLevel.SENIOR  # 6 levels (intern → manager)
```

Key dataclasses: `CVData`, `JobData`, `LabeledPair`, `DatasetSplit`.

### `graph/builder.py`

Builds PyTorch Geometric `HeteroData` from CVs, jobs, and labeled pairs.

```python
builder = GraphBuilder(provider, normalizer)
data = builder.build(cvs, jobs, labeled_pairs)  # → HeteroData
```

Output `HeteroData` contains:
- `data['cv'].x` — shape `[N_cv, 386]`
- `data['job'].x` — shape `[N_job, 386]`
- `data['skill'].x` — shape `[N_skill, 385]`
- `data['seniority'].x` — shape `[6, 6]` (identity)
- 6 edge types with `edge_index` + `edge_attr` (where applicable)

## Setup

**Yêu cầu:** Python 3.11+

```bash
cd ml-service

# 1. Tạo virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Cài đặt package (editable mode + dev tools)
pip install -e ".[dev]"

# 3. Copy config
cp .env.example .env
```

> **Lưu ý:** Lần đầu chạy `EnglishProvider.encode()`, model `all-MiniLM-L6-v2` (~80MB) sẽ tự động download từ HuggingFace.

## Config

Chỉnh file `.env` nếu cần:

```env
# Embedding
EMBEDDING_PROVIDER=english        # "english" hoặc "multilingual"
EMBEDDING_DIM=384

# Data
DATA_DIR=./data
SKILL_ALIAS_PATH=../roadmap/week1/skill-alias.json

# Synthetic data generation
NUM_CVS=800
NUM_JOBS=1500
NUM_POSITIVE_PAIRS=2000
RANDOM_SEED=42
```

| Variable | Default | Mô tả |
|----------|---------|-------|
| `EMBEDDING_PROVIDER` | `english` | `english` or `multilingual` |
| `EMBEDDING_DIM` | `384` | Embedding dimension |
| `DATA_DIR` | `data` | Data directory |
| `SKILL_ALIAS_PATH` | — | Path to skill-alias.json |
| `NUM_CVS` | `800` | Synthetic CVs to generate |
| `NUM_JOBS` | `1500` | Synthetic JDs to generate |
| `NUM_POSITIVE_PAIRS` | `2000` | Positive labeled pairs |
| `RANDOM_SEED` | `42` | Reproducibility |

## Run

### 1. Chạy tests

```bash
# Chạy tất cả tests
pytest -q

# Chạy với coverage
pytest --cov=ml_service --cov-report=term-missing

# Chạy test cụ thể
pytest tests/test_embedding.py -v
pytest tests/test_graph_builder.py -v
```

### 2. Generate synthetic data + build graph (Python script)

```python
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.embedding import get_provider
from ml_service.graph.builder import GraphBuilder

# --- Step 1: Load skill normalizer ---
normalizer = SkillNormalizer("../roadmap/week1/skill-alias.json")
print(f"Loaded {len(normalizer.canonical_skills)} canonical skills")

# --- Step 2: Generate synthetic CVs & JDs ---
gen = SyntheticDataGenerator(normalizer, seed=42)
cvs = gen.generate_cvs(800)
jobs = gen.generate_jobs(1500)
print(f"Generated {len(cvs)} CVs, {len(jobs)} JDs")

# --- Step 3: Create labeled pairs ---
labeler = PairLabeler(cvs, jobs, seed=42)
pairs = labeler.create_pairs(num_positive=2000)
split = labeler.split(pairs)
print(f"Pairs: {len(split.train)} train, {len(split.val)} val, {len(split.test)} test")

# --- Step 4: Build heterogeneous graph ---
provider = get_provider()  # EnglishProvider (all-MiniLM-L6-v2)
builder = GraphBuilder(provider, normalizer)
data = builder.build(cvs, jobs, pairs)

# --- Step 5: Inspect graph ---
print(f"\n=== HeteroData ===")
print(f"CV nodes:        {data['cv'].x.shape}")
print(f"Job nodes:       {data['job'].x.shape}")
print(f"Skill nodes:     {data['skill'].x.shape}")
print(f"Seniority nodes: {data['seniority'].x.shape}")
print(f"\nEdge types:")
for edge_type in data.edge_types:
    ei = data[edge_type].edge_index
    print(f"  {edge_type}: {ei.shape[1]} edges")
```

Chạy:

```bash
# Từ thư mục ml-service
python -c "
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.embedding import get_provider
from ml_service.graph.builder import GraphBuilder

normalizer = SkillNormalizer('../roadmap/week1/skill-alias.json')
gen = SyntheticDataGenerator(normalizer, seed=42)
cvs = gen.generate_cvs(100)
jobs = gen.generate_jobs(200)
labeler = PairLabeler(cvs, jobs, seed=42)
pairs = labeler.create_pairs(num_positive=300)

provider = get_provider()
builder = GraphBuilder(provider, normalizer)
data = builder.build(cvs, jobs, pairs)

print('Graph built successfully!')
print(f'CV: {data[\"cv\"].x.shape}, Job: {data[\"job\"].x.shape}')
print(f'Skills: {data[\"skill\"].x.shape}, Seniority: {data[\"seniority\"].x.shape}')
for et in data.edge_types:
    print(f'  {et}: {data[et].edge_index.shape[1]} edges')
"
```

### 3. Sử dụng từng module riêng lẻ

```bash
# Skill normalization
python -c "
from ml_service.data.skill_normalization import SkillNormalizer
n = SkillNormalizer('../roadmap/week1/skill-alias.json')
for raw in ['ReactJS', 'Node.js', 'Python3', 'K8s', 'unknown']:
    print(f'  {raw:15s} → {n.normalize(raw)}')
"

# Embedding
python -c "
from ml_service.embedding import get_provider
p = get_provider()
v = p.encode(['Senior Python developer', 'Backend engineer with Django'])
print(f'Shape: {v.shape}, dim: {p.dim}')
"

# Synthetic data stats
python -c "
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.data.generator import SyntheticDataGenerator
n = SkillNormalizer('../roadmap/week1/skill-alias.json')
g = SyntheticDataGenerator(n, seed=42)
cvs = g.generate_cvs(100)
print(f'CVs generated: {len(cvs)}')
print(f'Example CV: seniority={cvs[0].seniority}, skills={cvs[0].skills[:5]}')
"
```

### 4. Lint & Format

```bash
# Check lint
ruff check ml_service/ tests/

# Auto-fix
ruff check --fix ml_service/ tests/

# Format
ruff format ml_service/ tests/
```

## Tests

```bash
pytest -q
# 50 passed
```

Test coverage:

| Module | Tests |
|--------|-------|
| `test_embedding.py` | Provider shape, normalization, factory, determinism |
| `test_generator.py` | CV/JD generation, distributions, constraints |
| `test_graph_builder.py` | Node shapes, edge types, edge attributes |
| `test_graph_schema.py` | Enums, dataclasses, mapping tables |
| `test_labeler.py` | Overlap calculation, pair creation, train/val/test split |
| `test_skill_normalization.py` | Canonical mapping, aliases, categories |

## TODO

- [ ] GNN model (GraphSAGE) — `graph/model.py`
- [ ] Training pipeline (BPR loss) — `graph/train.py`
- [ ] Hybrid scoring (GNN + skill_overlap + seniority_match)
- [ ] Baseline implementations (cosine sim, Jaccard, BM25)
- [ ] MultilingualProvider (LaBSE / multilingual-e5)
- [ ] Inference pipeline (precompute CV embeddings + ranking)
