# Datasets — Nguồn dữ liệu CV và JD

## Tổng quan

Hệ thống sử dụng **real data** từ 3 nguồn:

| Nguồn | Loại | Số lượng | Mô tả |
|-------|------|---------|-------|
| Indeed (JobSpy) | JD | 315 | Job postings thật, crawl qua python-jobspy |
| datasetmaster/resumes | CV | 4.817 | IT resumes, structured JSON |
| Suriyaganesh/54k-resume | CV | 54.933 (filter → IT) | Relational CSV, 6 tables |

---

## Dataset 1: datasetmaster/resumes (CV)

- **URL:** https://huggingface.co/datasets/datasetmaster/resumes
- **Số lượng:** 4.817 IT resumes
- **License:** MIT
- **Format:** JSONL, structured fields

### Cấu trúc dữ liệu

```json
{
  "personal_info": {
    "name": "...",
    "summary": "Python Developer with experience in...",
    "location": {"city": "Pune", "country": "India"}
  },
  "skills": {
    "technical": {
      "programming_languages": [{"name": "Python", "level": "intermediate"}],
      "frameworks": [{"name": "Django", "level": "advanced"}],
      "databases": [{"name": "PostgreSQL", "level": "intermediate"}],
      "tools": [...],
      "platforms": [...]
    },
    "soft_skills": ["Communication", "Teamwork"]
  },
  "experience": [
    {
      "title": "Python Developer",
      "company": "...",
      "level": "mid",
      "dates": {"duration": "2 years 3 months"},
      "responsibilities": [...]
    }
  ],
  "education": [
    {
      "degree": {"level": "Bachelor", "field": "Computer Science"},
      "institution": {"name": "..."}
    }
  ]
}
```

### Đặc điểm

- **100% IT** — Python Developer, Full Stack, Data Scientist, DevOps, etc.
- **77 unique job titles**
- **Structured skills** — có programming_languages, frameworks, databases, tools riêng biệt
- **Seniority level** trong experience (junior/mid/senior)
- **Avg 6.2 skills/resume**
- Seniority phân bố đều: junior ~3K, mid ~3K, senior ~3K

### Cách sử dụng

```python
from ml_service.crawler.resume_loader import load_kaggle_resumes
from ml_service.data.skill_normalization import SkillNormalizer

normalizer = SkillNormalizer("../roadmap/week1/skill-alias.json")
cvs = load_kaggle_resumes(normalizer, max_resumes=500)
# → list[CVData], mỗi CV có skills, seniority, education, text
```

---

## Dataset 2: Suriyaganesh/54k-resume (CV)

- **URL:** https://huggingface.co/datasets/Suriyaganesh/54k-resume
- **Kaggle mirror:** https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured
- **Số lượng:** 54.933 resumes (filter IT → hàng nghìn)
- **License:** MIT
- **Format:** 6 CSV tables (relational)

### Cấu trúc dữ liệu (6 tables)

| Table | Rows | Columns |
|-------|------|---------|
| `01_people.csv` | 54.933 | person_id, name, email, phone, linkedin |
| `02_abilities.csv` | 1.219.473 | person_id, ability |
| `03_education.csv` | 75.999 | person_id, institution, program, start_date, location |
| `04_experience.csv` | 265.404 | person_id, title, firm, start_date, end_date, location |
| `05_person_skills.csv` | 2.483.376 | person_id, skill |
| `06_skills.csv` | 226.760 | skill |

### Đặc điểm

- **Lớn nhất** — 54K people, 2.5M person-skill associations
- **Đa ngành nhưng IT-heavy**: Java Developer (8.253), Python Developer (6.159), Network Admin (5.442), Front End Developer (4.197), Database Admin (4.104)
- **Cần filter** — dùng keyword matching trên experience titles
- **Relational format** — join tables qua person_id
- **72 unique canonical skills** sau normalize (SQL, Linux, MySQL, Python, AWS, etc.)

### Cách sử dụng

```python
from ml_service.crawler.resume_loader import load_54k_resumes
from ml_service.data.skill_normalization import SkillNormalizer

normalizer = SkillNormalizer("../roadmap/week1/skill-alias.json")
cvs = load_54k_resumes(normalizer, max_resumes=1000)
# → list[CVData], chỉ IT resumes, filtered by title keywords
```

### IT filter logic

Giữ lại resumes có experience title chứa ít nhất 1 từ khóa IT:
> developer, engineer, programmer, architect, devops, java, python, frontend, backend, fullstack, web, software, data, database, cloud, network, security, qa, test, automation, machine learning, ml, ai, bi, etl, scrum, agile, mobile, android, ios, ...

---

## Dataset 3: Indeed Jobs (JD)

- **Nguồn:** Indeed (USA), crawl qua python-jobspy
- **Số lượng:** 315 job postings (8 queries × 50 results, dedup)
- **Format:** JSONL (`data/raw_jobs.jsonl`)

### Queries đã crawl

```
software engineer, python developer, frontend developer,
backend developer, fullstack developer, data engineer,
devops engineer, machine learning engineer
```

### Cách sử dụng

```python
from ml_service.crawler.storage import load_raw_jobs, deduplicate
from ml_service.crawler.skill_extractor import SkillExtractor

raw_jobs = deduplicate(load_raw_jobs("data/raw_jobs.jsonl"))
extractor = SkillExtractor(normalizer)
jobs = extractor.extract_batch(raw_jobs)
```

---

## Unified loader

```python
from ml_service.crawler.resume_loader import load_resumes

# Dataset #1 (default)
cvs = load_resumes(normalizer, source="datasetmaster", max_resumes=500)

# Dataset #2
cvs = load_resumes(normalizer, source="54k", max_resumes=1000)
```

---

## So sánh 2 CV datasets

| Đặc điểm | datasetmaster/resumes | Suriyaganesh/54k-resume |
|-----------|----------------------|------------------------|
| Số lượng | 4.817 | 54.933 (filter → IT) |
| IT-only | Có | Cần filter |
| Format | JSONL (nested JSON) | 6 CSV tables (relational) |
| Structured skills | Có (name + level) | Có (flat list) |
| Seniority | Trong experience.level | Infer từ title |
| Education | Degree level + field | Program name |
| Text quality | Summary + responsibilities | Title + skills only |
| Avg skills/CV | 6.2 | 3.7 |
| Recommend cho | Default, MVP, experiment | Scale lên khi cần nhiều data |

---

## Benchmark kết quả (Real JDs + Real CVs)

Dùng 311 real JDs (Indeed) + 500 real CVs (datasetmaster/resumes):

```
Method                          recall@5   recall@10     ndcg@10     auc_roc
Cosine Similarity               0.0000      0.0130      0.0636      0.4554
Skill Overlap (Jaccard)         0.0649      0.1039      0.8630      0.6443
BM25                            0.0260      0.0519      0.4600      0.5703
GNN (Hybrid)                    0.0649      0.1169 *    0.9364 *    0.7711 *
```

**GNN AUC-ROC: 0.7711 vs best baseline 0.6443 (+12.7%)**
