# Synthetic Data — Threshold & Label Strategy

## Bài toán label

Vì không có lịch sử apply thật, ta dùng **rule-based labeling** từ synthetic data.
Mục tiêu: tạo đủ positive/negative pairs, đảm bảo hard negatives có trong tập train.

---

## Threshold quyết định

### Positive (match = 1)

Pair (CV, JD) là **positive** khi thỏa **cả 2 điều kiện**:

```
skill_overlap_ratio >= 0.5   AND   seniority_distance <= 1
```

**skill_overlap_ratio** = số skill chung / số skill JD yêu cầu
```
ví dụ: JD yêu cầu [python, react, postgresql, docker] — 4 skills
        CV có      [python, react, nodejs, git]         — 2 match
        → overlap = 2/4 = 0.5  ✅ đủ điều kiện
```

**seniority_distance** = |cv_seniority_index - job_seniority_index|
```
seniority index: intern=0, junior=1, mid=2, senior=3, lead=4, manager=5

ví dụ: CV=junior(1), JD=mid(2) → distance = 1  ✅ acceptable
        CV=junior(1), JD=senior(3) → distance = 2  ❌ too far
```

---

### Negative (match = 0)

#### Easy negative
Pair là **easy negative** khi thỏa **ít nhất 1 trong 2**:

```
skill_overlap_ratio < 0.2   OR   seniority_distance >= 3
```

```
ví dụ: CV=backend dev, JD=graphic designer → overlap ≈ 0   ✅ easy neg
        CV=intern, JD=manager → distance = 5               ✅ easy neg
```

#### Hard negative *(quan trọng cho training)*
Pair là **hard negative** khi thỏa **tất cả**:

```
0.2 <= skill_overlap_ratio < 0.5   AND   seniority_distance <= 1
```

```
ví dụ: JD yêu cầu [python, react, postgresql, docker, aws, kubernetes]
        CV có      [python, react, git, linux]
        → overlap = 2/6 = 0.33   seniority match
        → hard negative: cùng ngành, gần đúng seniority, nhưng thiếu infra skills
```

---

## Tỷ lệ sampling

| Loại | Tỷ lệ | Lý do |
|------|--------|-------|
| Positive | 1 | Baseline |
| Easy negative | 1 | Cân bằng cơ bản |
| Hard negative | 2 | GNN cần học từ cases khó |
| **Tổng** | **1 : 1 : 2 = 1 : 3** | — |

---

## Synthetic data generation plan

### Bước 1: Tạo Skill Pool
- Dùng `skill-alias.json` — lấy list canonical skills (~120 skills)
- Phân nhóm theo category để generate CV/JD realistic hơn

### Bước 2: Generate CV giả

```python
# Mỗi CV gồm:
seniority     = random.choice(["intern","junior","mid","senior","lead","manager"])
experience    = seniority_to_years[seniority] + random.uniform(-0.5, 1.5)
num_skills    = random.randint(4, 12)  # senior thì nhiều skill hơn
skills        = sample_skills_by_seniority(seniority, num_skills)
edu_level     = random.choice([1, 2, 3])  # college, bachelor, master

# Skill sampling theo seniority:
# intern/junior: nhiều "soft", ít "technical"
# mid/senior:    nhiều "technical" + "tool"
# lead/manager:  thêm "domain" + leadership soft skills
```

### Bước 3: Generate JD giả

```python
# Mỗi JD gồm:
seniority_required  = random.choice(["junior","mid","senior","lead"])
num_required_skills = random.randint(3, 8)
required_skills     = sample_skills_for_job(seniority_required, num_required_skills)
salary_min          = seniority_to_salary_range[seniority_required][0]
salary_max          = seniority_to_salary_range[seniority_required][1]
```

### Bước 4: Generate Pairs

```python
for each JD:
    # Positive pairs
    positive_cvs = find_cvs_where(
        skill_overlap >= 0.5 AND seniority_distance <= 1
    )
    sample n_positive from positive_cvs

    # Hard negative pairs (2x positive count)
    hard_neg_cvs = find_cvs_where(
        0.2 <= skill_overlap < 0.5 AND seniority_distance <= 1
    )
    sample 2 * n_positive from hard_neg_cvs

    # Easy negative pairs (1x positive count)
    easy_neg_cvs = find_cvs_where(
        skill_overlap < 0.2 OR seniority_distance >= 3
    )
    sample n_positive from easy_neg_cvs
```

### Bước 5: Scale target

| Split | Positive pairs | Negative pairs | Tổng |
|-------|---------------|----------------|------|
| Train | 1.500 | 4.500 | 6.000 |
| Val | 300 | 900 | 1.200 |
| Test | 200 | 600 | 800 |
| **Total** | **2.000** | **6.000** | **8.000** |

---

## Seniority mapping

```python
seniority_to_years = {
    "intern":  (0, 0.5),
    "junior":  (0.5, 2),
    "mid":     (2, 5),
    "senior":  (5, 10),
    "lead":    (7, 15),
    "manager": (8, 20),
}

seniority_to_salary_usd = {
    "intern":  (500,  1_500),
    "junior":  (1_000, 2_500),
    "mid":     (2_000, 4_000),
    "senior":  (3_500, 6_000),
    "lead":    (5_000, 8_000),
    "manager": (5_000, 10_000),
}
```

---

## Edge cases cần xử lý

| Case | Xử lý |
|------|-------|
| CV không ghi số năm kinh nghiệm | Infer từ seniority trung vị |
| JD không ghi seniority rõ | Infer từ required skills complexity |
| Skill overlap đúng bằng threshold (= 0.5) | Tính là positive |
| CV có quá nhiều skills (> 15) | Cap ở 15, lấy top skills theo frequency |
| JD yêu cầu 0 skills | Skip — invalid JD |

---

## Validation sau khi generate

- [ ] Phân bố positive/negative: 1:3 ± 5%
- [ ] Phân bố seniority trong CV: roughly uniform (không bị lệch về 1 level)
- [ ] Phân bố skills: không có skill nào chiếm > 30% tổng edges
- [ ] Hard negative ratio: >= 50% của total negatives
- [ ] Không có data leak giữa train/val/test (CV và JD không xuất hiện ở cả 2 split)
