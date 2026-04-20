# Plan: Backend Labeling System (Week 10)

> Mục tiêu: Build Django backend cho admin labeling tool, thu thập real human labels để thay thế proxy labels trong ML training pipeline.

---

## Tại sao cần labeling system

Hệ thống hiện tại dùng proxy labels (skill overlap + seniority threshold). GNN học được AUC 0.70 nhưng bị giới hạn bởi chất lượng signal này. Với **200–500 human labels**, training signal sẽ phản ánh đúng "job này có phù hợp với người này không" từ góc nhìn recruiter.

---

## Kiến trúc tổng quan

```
React Admin UI
    │
    │ REST API
    ▼
Django Backend
  ├── PairQueue       (danh sách cặp CV-Job chờ label)
  ├── HumanLabel      (labels đã gán)
  ├── Smart Selector  (chọn cặp có giá trị thông tin cao nhất)
  └── Export API      (xuất ra format cho ML training)
    │
    ▼
ML Training Pipeline
  └── LabeledPair(cv_id, job_id, label, split)
```

---

## Django Models

### 1. `LabelingCV`
Lưu thông tin CV để hiển thị trên UI — tách khỏi raw data files.

```python
class LabelingCV(models.Model):
    cv_id        = models.IntegerField(unique=True)       # map với CVData.cv_id
    source       = models.CharField(max_length=50)        # "linkedin", "kaggle"
    skills       = models.JSONField()                     # ["python", "django", ...]
    seniority    = models.CharField(max_length=20)        # "JUNIOR", "MID", ...
    experience_years = models.FloatField()
    education    = models.CharField(max_length=20)        # "BACHELOR", "MASTER", ...
    text_summary = models.TextField()                     # first 500 chars của text
    created_at   = models.DateTimeField(auto_now_add=True)
```

### 2. `LabelingJob`
Lưu thông tin Job để hiển thị.

```python
class LabelingJob(models.Model):
    job_id     = models.IntegerField(unique=True)         # map với JobData.job_id
    title      = models.CharField(max_length=200)
    skills     = models.JSONField()                       # ["python", "fastapi", ...]
    seniority  = models.CharField(max_length=20)
    salary_min = models.IntegerField(null=True)
    salary_max = models.IntegerField(null=True)
    text_summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
```

### 3. `PairQueue`
Danh sách cặp CV-Job được chọn để gán nhãn. Pre-populated bởi smart selector.

```python
class SelectionReason(models.TextChoices):
    HIGH_OVERLAP   = "high_overlap"    # skill overlap >= 0.4 → likely positive
    MEDIUM_OVERLAP = "medium_overlap"  # 0.15 <= overlap < 0.4 → most informative
    HARD_NEGATIVE  = "hard_negative"   # same domain, wrong seniority → likely negative
    RANDOM         = "random"          # diversity sample

class PairStatus(models.TextChoices):
    PENDING = "pending"
    LABELED = "labeled"
    SKIPPED = "skipped"

class PairQueue(models.Model):
    cv                  = models.ForeignKey(LabelingCV, on_delete=models.CASCADE)
    job                 = models.ForeignKey(LabelingJob, on_delete=models.CASCADE)
    skill_overlap_score = models.FloatField()
    bm25_score          = models.FloatField(default=0.0)
    selection_reason    = models.CharField(max_length=30, choices=SelectionReason.choices)
    status              = models.CharField(max_length=20, choices=PairStatus.choices,
                                           default=PairStatus.PENDING)
    created_at          = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("cv", "job")
        ordering = ["selection_reason", "?"]  # medium_overlap first, then random
```

### 4. `HumanLabel`
Label thực tế từ admin. Một pair có thể có nhiều labels (inter-rater agreement).

```python
class LabelValue(models.IntegerChoices):
    NOT_SUITABLE = 0, "Không phù hợp"
    SUITABLE     = 1, "Phù hợp"
    STRONG_FIT   = 2, "Rất phù hợp"

class DimScore(models.IntegerChoices):
    BAD  = 0, "Không phù hợp"   # ❌
    OK   = 1, "Tạm được"        # ⚠️
    GOOD = 2, "Tốt"             # ✅

class HumanLabel(models.Model):
    pair       = models.ForeignKey(PairQueue, on_delete=models.CASCADE,
                                   related_name="labels")

    # --- Dimension scores (0=bad, 1=ok, 2=good) ---
    skill_fit       = models.IntegerField(choices=DimScore.choices)  # kỹ năng khớp
    seniority_fit   = models.IntegerField(choices=DimScore.choices)  # cấp bậc
    experience_fit  = models.IntegerField(choices=DimScore.choices)  # số năm kinh nghiệm
    domain_fit      = models.IntegerField(choices=DimScore.choices)  # domain/ngành

    # --- Overall (tự động gợi ý từ dims, labeler có thể override) ---
    overall         = models.IntegerField(choices=LabelValue.choices)

    note       = models.TextField(blank=True)
    labeled_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def binary_label(self) -> int:
        """Map sang LabeledPair.label: overall >= 1 → 1, overall == 0 → 0"""
        return 1 if self.overall >= 1 else 0

    @property
    def suggested_overall(self) -> int:
        """Gợi ý overall từ dimension scores — labeler có thể override."""
        avg = (self.skill_fit + self.seniority_fit +
               self.experience_fit + self.domain_fit) / 4
        if avg >= 1.5:
            return 2
        elif avg >= 0.75:
            return 1
        return 0
```

---

## Smart Pair Selection Algorithm

Không random — chọn cặp có **giá trị thông tin cao nhất** cho model:

```python
def generate_pair_queue(cvs, jobs, n_pairs=500):
    """
    Phân bổ n_pairs theo tỷ lệ:
      40% medium_overlap  (0.15 <= overlap < 0.40) → most uncertain, max info gain
      30% high_overlap    (overlap >= 0.40)         → confirm positives
      20% hard_negative   (same domain, |seniority| > 1) → confirm negatives
      10% random          (diversity)
    """
    scorer = SkillOverlapScorer()
    pairs = []

    for cv in cvs:
        for job in jobs:
            overlap = scorer.score(cv, job)
            seniority_gap = abs(int(cv.seniority) - int(job.seniority))
            domain_match = cv.domain == job.domain  # nếu có domain field

            if 0.15 <= overlap < 0.40:
                reason = SelectionReason.MEDIUM_OVERLAP
            elif overlap >= 0.40:
                reason = SelectionReason.HIGH_OVERLAP
            elif domain_match and seniority_gap > 1:
                reason = SelectionReason.HARD_NEGATIVE
            else:
                continue

            pairs.append((cv, job, overlap, reason))

    # Sample theo tỷ lệ, tránh duplicate
    return stratified_sample(pairs, n_pairs, ratio={
        SelectionReason.MEDIUM_OVERLAP: 0.40,
        SelectionReason.HIGH_OVERLAP:   0.30,
        SelectionReason.HARD_NEGATIVE:  0.20,
        SelectionReason.RANDOM:         0.10,
    })
```

**Tại sao phân bổ này:**
- Medium overlap: model không biết đây là positive hay negative → label này có giá trị cao nhất
- High overlap: cần confirm proxy labels đúng không
- Hard negative: dạy model phân biệt subtle mismatches
- Random: prevent bias, tăng diversity

---

## API Endpoints

### GET `/api/labeling/next/`
Trả pair tiếp theo chưa được label, ưu tiên medium_overlap.

```json
Response:
{
  "pair_id": 142,
  "cv": {
    "cv_id": 23,
    "skills": ["python", "django", "postgresql"],
    "seniority": "MID",
    "experience_years": 3.5,
    "education": "BACHELOR",
    "text_summary": "Backend developer with 3 years..."
  },
  "job": {
    "job_id": 891,
    "title": "Backend Engineer",
    "skills": ["python", "fastapi", "redis"],
    "seniority": "MID",
    "salary_min": 1500,
    "salary_max": 2500,
    "text_summary": "We are looking for..."
  },
  "hint": {
    "skill_overlap": 0.33,
    "common_skills": ["python"],
    "selection_reason": "medium_overlap"
  },
  "progress": {
    "labeled": 87,
    "total": 300,
    "remaining": 213
  }
}
```

### POST `/api/labeling/{pair_id}/submit/`
```json
Request:
{
  "skill_fit":      2,   // 0=bad, 1=ok, 2=good
  "seniority_fit":  1,
  "experience_fit": 2,
  "domain_fit":     2,
  "overall":        1,   // labeler confirm hoặc override suggested_overall
  "note":           ""   // optional
}

Response: {
  "next_pair_id": 143,
  "suggested_overall": 1   // gợi ý từ dims cho pair tiếp theo
}
```

### POST `/api/labeling/{pair_id}/skip/`
Bỏ qua pair không chắc chắn.

### GET `/api/labeling/stats/`
```json
{
  "total_pairs": 300,
  "labeled": 87,
  "skipped": 12,
  "label_distribution": {"0": 34, "1": 41, "2": 12},
  "by_reason": {
    "medium_overlap": {"labeled": 45, "total": 120},
    "high_overlap":   {"labeled": 30, "total": 90},
    "hard_negative":  {"labeled": 12, "total": 60}
  }
}
```

### GET `/api/labeling/export/?format=ml`
Export trực tiếp sang format ML training.

```json
[
  {
    "cv_id": 23, "job_id": 891,
    "label": 1, "split": "train",
    "skill_fit": 2, "seniority_fit": 1, "experience_fit": 2, "domain_fit": 2,
    "overall": 1
  },
  ...
]
```

---

## Export Pipeline → ML Training

### Label mapping
```
HumanLabel.overall = 0  →  LabeledPair.label = 0  (no match)
HumanLabel.overall = 1  →  LabeledPair.label = 1  (match)
HumanLabel.overall = 2  →  LabeledPair.label = 1  (strong match, still positive)
```

### Dimension scores — 3 cách dùng cho ML

**Cách 1 (ngay bây giờ):** Chỉ dùng `overall` → `binary_label` cho `LabeledPair`. Đơn giản, compatible với pipeline hiện tại.

**Cách 2 (sau khi có 200+ labels):** Dùng dimension scores để cải thiện `PairLabeler`:
```python
# Phân tích xem proxy labels sai ở dimension nào
# VD: skill_fit=2 nhưng domain_fit=0 → proxy label sai vì không có domain constraint
# → update PairLabeler để thêm domain matching
```

**Cách 3 (future):** Multi-task learning — train GNN predict cả 4 dimensions + overall simultaneously. Mỗi dimension là auxiliary loss, giúp model hiểu rõ hơn tại sao một pair là match.

### Split strategy
- Giữ **test set hoàn toàn từ human labels** (không mix proxy labels)
- Train/val có thể mix human + proxy labels
- Tỷ lệ: 70% train / 15% val / 15% test

```python
# Trong PairLabeler hoặc script training
human_labels = load_human_labels("export.json")  # từ API export
proxy_labels = labeler.create_pairs(...)          # existing proxy labels

# Test set = human labels only (honest evaluation)
test_pairs = [p for p in human_labels if p.split == "test"]

# Train set = proxy + human train labels
train_pairs = proxy_labels_train + [p for p in human_labels if p.split == "train"]
```

### Confidence filtering
Chỉ dùng labels có confidence ≥ 2 cho training. confidence = 1 → bỏ qua hoặc dùng với weight thấp hơn.

---

## Quality Control

**Inter-rater agreement** (nếu có nhiều người label):
- Mỗi pair được label bởi ≥ 2 người
- Nếu conflict → hiển thị lại để review
- Fleiss' kappa > 0.6 → label quality acceptable

**Single labeler (chỉ bạn label):**
- Sau mỗi 50 pairs, review lại 10 pairs đã label
- Đảm bảo consistency

---

## Management Command để populate PairQueue

```bash
python manage.py populate_pair_queue \
    --cvs-path data/linkedin_cvs.json \
    --jobs-path data/raw_jobs.jsonl \
    --n-pairs 300 \
    --strategy stratified
```

Script này:
1. Load CVs và Jobs từ files hiện có
2. Tính skill overlap và BM25 score cho tất cả cặp
3. Chọn 300 pairs theo stratified sampling
4. Insert vào `PairQueue`

---

## Implementation Order

1. **Django models** — LabelingCV, LabelingJob, PairQueue, HumanLabel
2. **Management command** — populate_pair_queue (smart selection)
3. **API endpoints** — next, submit, skip, stats, export
4. **Serializers + ViewSets** — DRF
5. **Test với React UI** (week11)

---

## Kết quả kỳ vọng sau labeling

| Labels | Expected AUC-ROC | Note |
|--------|-----------------|------|
| 0 (proxy only) | 0.70 (hiện tại) | Ceiling của proxy labels |
| 200 human | 0.72–0.75 | Mixed training |
| 500 human | 0.75–0.80 | Human test set credible |
| 1000 human | 0.80+ | Approaching paper-quality |

Quan trọng nhất: **test set từ human labels** cho phép đo performance thật sự, không bị inflated bởi proxy label bias.
