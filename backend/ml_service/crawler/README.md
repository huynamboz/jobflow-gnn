# Crawler Module

Multi-provider job crawling với DI pattern. Thêm provider mới = 1 file trong `providers/`, auto-discovered.

## Providers

| Provider | Source | Auth | Data quality |
|----------|--------|------|-------------|
| `jobspy` | Indeed, Glassdoor | Không cần | Tốt (title, company, description, salary) |
| `linkedin` | LinkedIn | Cần login 1 lần | Rất tốt (+ logo, company URL, job type, applicants) |
| `adzuna` | Adzuna API | Cần API key | Tốt (structured, có salary) |
| `remotive` | Remotive API | Không cần | OK (remote tech jobs only) |

## Setup

```bash
cd backend

# LinkedIn — login 1 lần, save session
.venv/bin/python -m ml_service.crawler.providers.linkedin_auth
# → Browser mở → login → quay lại terminal nhấn Enter
# → auth/linkedin_state.json saved

# Adzuna — set API key
# Đăng ký tại https://developer.adzuna.com/
export ADZUNA_APP_ID=your_id
export ADZUNA_APP_KEY=your_key
```

## Crawl

### Từng provider

```bash
# Indeed (qua JobSpy)
.venv/bin/python -c "
from ml_service.crawler import get_provider
from ml_service.crawler.storage import save_raw_jobs, deduplicate

p = get_provider('jobspy')
jobs = p.fetch('python developer', results_wanted=50)
jobs = deduplicate(jobs)
save_raw_jobs(jobs, 'data/raw_jobs.jsonl')
print(f'Saved {len(jobs)} jobs')
"

# LinkedIn (cần đã login)
.venv/bin/python -c "
from ml_service.crawler import get_provider

p = get_provider('linkedin', headless=True, save_path='data/raw_jobs.jsonl')
jobs = p.fetch('react developer', location='Vietnam', results_wanted=250)
print(f'Saved {len(jobs)} jobs')
"

# LinkedIn headless=False (xem browser crawl)
.venv/bin/python -c "
from ml_service.crawler import get_provider

p = get_provider('linkedin', headless=False, save_path='data/raw_jobs.jsonl')
jobs = p.fetch('developer', location='Vietnam', results_wanted=50)
"

# Remotive (remote tech jobs, free)
.venv/bin/python -c "
from ml_service.crawler import get_provider
from ml_service.crawler.storage import save_raw_jobs, deduplicate

p = get_provider('remotive')
jobs = p.fetch('frontend developer', results_wanted=50)
jobs = deduplicate(jobs)
save_raw_jobs(jobs, 'data/raw_jobs.jsonl')
print(f'Saved {len(jobs)} jobs')
"

# Adzuna (cần API key)
.venv/bin/python -c "
from ml_service.crawler import get_provider
from ml_service.crawler.storage import save_raw_jobs, deduplicate

p = get_provider('adzuna', app_id='xxx', app_key='yyy')
jobs = p.fetch('data engineer', location='london', results_wanted=50)
jobs = deduplicate(jobs)
save_raw_jobs(jobs, 'data/raw_jobs.jsonl')
print(f'Saved {len(jobs)} jobs')
"
```

### Multi-provider (scheduler)

```bash
# Crawl từ nhiều providers cùng lúc
.venv/bin/python -c "
from ml_service.crawler import CrawlScheduler

scheduler = CrawlScheduler(providers=['jobspy', 'remotive'])
scheduler.crawl_all(
    queries=['python developer', 'react developer', 'devops engineer'],
    results_per_query=50,
    save_path='data/raw_jobs.jsonl',
)
"

# Chạy script mặc định (8 queries × Indeed)
.venv/bin/python run_crawl.py
```

### Liệt kê providers

```bash
.venv/bin/python -c "
from ml_service.crawler import list_providers
print(list_providers())
"
# → ['adzuna', 'jobspy', 'linkedin', 'remotive']
```

## Check auth LinkedIn

```bash
.venv/bin/python -c "
from ml_service.crawler.providers.linkedin_auth import load_state_path
import json

path = load_state_path()
if not path:
    print('❌ Chưa login')
else:
    state = json.load(open(path))
    cookies = [c for c in state['cookies'] if 'linkedin' in c.get('domain','')]
    session = next((c for c in cookies if c['name'] == 'li_at'), None)
    print(f'✅ Logged in' if session else '❌ Session expired')
"
```

## Data format

Mỗi job lưu trong `data/raw_jobs.jsonl`:

```json
{
  "source": "linkedin",
  "source_url": "https://www.linkedin.com/jobs/view/123...",
  "title": "Senior React Developer",
  "company": "Google",
  "location": "Mountain View, CA",
  "description": "Full job description text...",
  "salary_min": 150000,
  "salary_max": 200000,
  "salary_currency": "USD",
  "date_posted": "2026-03-28T00:00:00",
  "seniority_hint": "2 weeks ago",
  "raw_skills": [],
  "company_logo_url": "https://media.licdn.com/...",
  "company_url": "https://www.linkedin.com/company/google/",
  "job_type": "remote, full-time",
  "applicant_count": "Over 100 applicants",
  "fingerprint": "897e25ae53704b..."
}
```

## Dedup

2 lớp:
1. **URL** — cùng source_url → skip
2. **Fingerprint** — normalize(title) + normalize(company) + city → MD5 hash
   - "Senior Python Developer @ Google Inc." = "Sr. Python Dev @ Google" → same fingerprint

## Thêm provider mới

Tạo file `providers/my_provider.py`:

```python
from ml_service.crawler.base import CrawlProvider, RawJob

class MyProvider(CrawlProvider):
    @property
    def name(self) -> str:
        return "my_source"

    def fetch(self, search_term, location="", results_wanted=100, **kwargs) -> list[RawJob]:
        # Crawl logic here
        return [RawJob(source="my_source", ...)]
```

Auto-discovered — không cần sửa factory hay bất kỳ file nào khác.

ml_service/
├── crawler/                   ← Chỉ crawling
│   ├── base.py                   CrawlProvider ABC + RawJob
│   ├── factory.py                Auto-discover providers
│   ├── scheduler.py              Multi-provider orchestrator
│   ├── storage.py                JSONL save/load + fingerprint dedup
│   ├── README.md                 Hướng dẫn crawl
│   └── providers/
│       ├── jobspy_provider.py    Indeed/Glassdoor
│       ├── linkedin_provider.py  LinkedIn (Playwright)
│       ├── linkedin_auth.py      LinkedIn login + save state
│       ├── linkedin_selectors.json  CSS selectors config
│       ├── adzuna_provider.py    Adzuna REST API
│       └── remotive_provider.py  Remotive API
│
├── data/                      ← Data processing
│   ├── skill_extractor.py        RawJob → JobData (skills, seniority)
│   ├── resume_loader.py          HuggingFace CV datasets
│   ├── skill_normalization.py    Alias mapping
│   ├── skill_graph.py            Skill co-occurrence PMI
│   ├── skill_taxonomy.py         Synonyms, clusters, templates
│   ├── generator.py              Synthetic data
│   ├── labeler.py                Pair labeling
│   └── skill-alias.json          208 skills dictionary
