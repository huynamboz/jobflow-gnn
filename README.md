# JobFlow-GNN

CV-Job matching system powered by Graph Neural Networks. Upload your CV, get matched with the most relevant jobs from 4,000+ real IT job postings.

## Architecture

```
Frontend (coming soon)
    │
    ▼
Django Backend (REST API + Admin)
    │
    ├── apps/matching    → CV-Job matching endpoints
    ├── apps/jobs        → Job CRUD + Admin
    ├── apps/cvs         → CV upload + parsing
    ├── apps/skills      → 208 IT skills dictionary
    └── apps/users       → Auth (JWT)
    │
    ├── ml_service/      → ML library (imported directly)
    │   ├── inference    → Two-stage ranking (retrieve + rerank)
    │   ├── models       → HeteroGraphSAGE + RGCN
    │   ├── reranker     → MLP reranker (20 features)
    │   ├── crawler      → Multi-provider job crawling
    │   ├── cv_parser    → PDF/DOCX parsing
    │   └── data         → Skill extraction, normalization, graph
    │
    └── PostgreSQL
```

## Key Features

- **GNN-powered matching** — GraphSAGE on heterogeneous graph (CV, Job, Skill, Seniority nodes)
- **Two-stage ranking** — Stage 1 retrieve (hybrid scoring) → Stage 2 rerank (MLP on 20 features)
- **Semantic skill matching** — Flask developer matches Django job via skill co-occurrence graph
- **Multi-provider crawling** — Indeed, LinkedIn (Playwright), Adzuna API, Remotive API
- **CV parsing** — Upload PDF/DOCX, extract skills + seniority + education
- **Role-aware scoring** — Frontend CV won't match AI/ML jobs
- **Django Admin** — Full CRUD for jobs, CVs, skills, users

## Quick Start

```bash
cd backend

# 1. Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Database
docker compose up -d              # PostgreSQL on port 5434
cp .env.example .env              # edit DB credentials if needed
python manage.py migrate
python manage.py createsuperuser

# 3. Run
python manage.py runserver 8000
```

- API: http://localhost:8000/api/docs/ (Swagger)
- Admin: http://localhost:8000/admin/

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/matching/cv/` | CV text → Top K matching jobs |
| POST | `/api/matching/cv/upload/` | Upload CV PDF/DOCX → Top K jobs |
| POST | `/api/matching/parse/` | CV text → parsed skills (debug) |
| POST | `/api/matching/parse/upload/` | Upload CV → parsed skills (debug) |

### Example

```bash
curl -X POST http://localhost:8000/api/matching/cv/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Frontend developer, 3 years experience. React, TypeScript, VueJS, NodeJS, TailwindCSS.",
    "top_k": 5
  }'
```

```json
[
  {"job_id": 1185, "score": 0.77, "eligible": true, "title": "React Developer", "matched_skills": ["react", "typescript", "tailwind", "html_css", "javascript"]},
  {"job_id": 3902, "score": 0.79, "eligible": true, "title": "Web Developer", "matched_skills": ["vuejs", "react", "sass", "ci_cd", "rest_api"]}
]
```

## Crawling Jobs

```bash
# Indeed (via JobSpy)
python run_crawl.py

# LinkedIn (requires one-time login)
python -m ml_service.crawler.providers.linkedin_auth   # login in browser
python -c "
from ml_service.crawler import get_provider
p = get_provider('linkedin', headless=True, save_path='data/raw_jobs.jsonl')
p.fetch('developer', location='Vietnam', results_wanted=250)
"
```

4 providers: `jobspy` (Indeed), `linkedin`, `adzuna`, `remotive`. See [crawler README](backend/ml_service/crawler/README.md).

## Training

```bash
cd backend

# Train GNN
python run_train_save.py

# Train reranker + calibration
python run_train_reranker.py
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Django 5 + DRF |
| ML | PyTorch, PyTorch Geometric, Sentence-Transformers |
| GNN | HeteroGraphSAGE, RGCN |
| Embedding | all-MiniLM-L6-v2 (384-dim) |
| Database | PostgreSQL |
| Crawling | Playwright (LinkedIn), JobSpy (Indeed), REST APIs |
| CV Parsing | pdfplumber, python-docx |

## Data

- **4,000+ real IT job postings** from Indeed + LinkedIn
- **4,800+ IT resumes** from HuggingFace (datasetmaster/resumes)
- **208 canonical skills** with alias mapping
- **Skill co-occurrence graph** (PMI-based, 3,600+ skill pairs)

## Scoring

```
final_score = base_score × role_penalty (capped by must-have penalty)

base_score = 0.55 × text_similarity
           + 0.30 × semantic_skill_overlap
           + 0.15 × seniority_score
```

See [scoring algorithm](roadmap/week1/scoring-algorithm.md) for details.

## Project Structure

```
jobflow-gnn/
├── backend/
│   ├── config/          Django settings
│   ├── apps/            Django apps (users, jobs, cvs, skills, matching)
│   ├── ml_service/      ML library
│   │   ├── crawler/     Multi-provider crawling
│   │   ├── cv_parser/   PDF/DOCX parsing
│   │   ├── data/        Skill extraction, normalization, graph
│   │   ├── embedding/   Text embedding (MiniLM)
│   │   ├── graph/       PyG graph builder
│   │   ├── inference/   Two-stage engine + checkpoint
│   │   ├── models/      GNN models (GraphSAGE, RGCN)
│   │   ├── reranker/    MLP reranker + calibration
│   │   └── training/    BPR trainer
│   ├── tests_ml/        155 tests
│   └── manage.py
└── roadmap/             Documentation + planning
```
