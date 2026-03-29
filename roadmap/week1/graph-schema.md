# Graph Schema — Phase 1

## Diagram

```
                    ┌─────────────┐
                    │  Seniority  │
                    │─────────────│
                    │ intern      │
                    │ junior      │
                    │ mid         │
                    │ senior      │
                    │ lead        │
                    │ manager     │
                    └──────┬──────┘
                           │ ▲
          has_seniority ───┘ └─── requires_seniority
                           │ │
          ┌────────────────┘ └────────────────┐
          │                                   │
    ┌─────┴──────┐                     ┌──────┴─────┐
    │     CV     │                     │    Job     │
    │────────────│                     │────────────│
    │ embedding  │                     │ embedding  │
    │ exp_years  │                     │ salary_min │
    │ edu_level  │                     │ salary_max │
    └─────┬──────┘                     └──────┬─────┘
          │                                   │
          │  match / no_match (label edges)   │
          └──────────────►────────────────────┘
          │                                   │
 has_skill│                                   │requires_skill
  (w=1–5) │                                   │ (w=1–5)
          │                                   │
          └──────────────►────────────────────┘
                          │
                    ┌─────┴──────┐
                    │   Skill    │
                    │────────────│
                    │ embedding  │
                    │ category   │
                    │ canonical  │
                    └────────────┘
```

---

## Node Types (Phase 1)

### CV
| Field | Type | Mô tả |
|-------|------|-------|
| `x` | float[384] | Embedding từ EnglishProvider (all-MiniLM-L6-v2) |
| `experience_years` | float | Số năm kinh nghiệm (0–30) |
| `education_level` | int | 0=none, 1=college, 2=bachelor, 3=master, 4=phd |

### Job
| Field | Type | Mô tả |
|-------|------|-------|
| `x` | float[384] | Embedding từ title + description + requirements |
| `salary_min` | float | Lương tối thiểu (normalized 0–1) |
| `salary_max` | float | Lương tối đa (normalized 0–1) |

### Skill
| Field | Type | Mô tả |
|-------|------|-------|
| `x` | float[384] | Embedding từ canonical skill name |
| `category` | int | 0=technical, 1=soft, 2=tool, 3=domain |
| `canonical_name` | str | Index key (lowercase, normalized) |

### Seniority
| Field | Type | Mô tả |
|-------|------|-------|
| `x` | float[6] | One-hot vector — 6 levels |

```
Index: 0=intern, 1=junior, 2=mid, 3=senior, 4=lead, 5=manager
```

---

## Edge Types (Phase 1)

### has_skill — CV → Skill
| Field | Type | Mô tả |
|-------|------|-------|
| `edge_index` | int[2, E] | Source CV → Target Skill |
| `edge_attr` | float[E] | Proficiency: 1=beginner, 2=basic, 3=intermediate, 4=advanced, 5=expert |

> Nếu CV không ghi proficiency → default = 3 (intermediate)

### requires_skill — Job → Skill
| Field | Type | Mô tả |
|-------|------|-------|
| `edge_index` | int[2, E] | Source Job → Target Skill |
| `edge_attr` | float[E] | Importance: 1=nice-to-have → 5=must-have |

> Phân loại: "required" → 5, "preferred" → 3, "bonus" → 1

### has_seniority — CV → Seniority
| Field | Type | Mô tả |
|-------|------|-------|
| `edge_index` | int[2, E] | Source CV → Target Seniority node |
| `edge_attr` | — | Không có weight |

### requires_seniority — Job → Seniority
| Field | Type | Mô tả |
|-------|------|-------|
| `edge_index` | int[2, E] | Source Job → Target Seniority node |
| `edge_attr` | — | Không có weight |

> Job có thể link tới nhiều Seniority node (ví dụ: junior–mid acceptable)

### match — CV → Job *(label: positive)*
| Field | Type | Mô tả |
|-------|------|-------|
| `edge_index` | int[2, E] | CV → Job (positive pairs) |
| `edge_attr` | — | Label = 1 (implicit) |

### no_match — CV → Job *(label: negative)*
| Field | Type | Mô tả |
|-------|------|-------|
| `edge_index` | int[2, E] | CV → Job (negative pairs) |
| `edge_attr` | — | Label = 0 (implicit) |

---

## Chiều của edges

```
CV  ──has_skill──►        Skill
Job ──requires_skill──►   Skill
CV  ──has_seniority──►    Seniority
Job ──requires_seniority──► Seniority
CV  ──match──►            Job   (positive label)
CV  ──no_match──►         Job   (negative label)
```

> Tất cả edges đều có hướng (directed).
> Không cần reverse edges cho Phase 1 — PyG to_hetero() xử lý tự động nếu cần.

---

## Sizing ước lượng (synthetic data Phase 1)

| Entity | Số lượng ước tính |
|--------|------------------|
| CV nodes | 500–1.000 |
| Job nodes | 1.000–2.000 |
| Skill nodes | 150–300 (canonical) |
| Seniority nodes | 6 (fixed) |
| has_skill edges | ~5.000–8.000 |
| requires_skill edges | ~8.000–15.000 |
| match edges (positive) | ~2.000–3.000 |
| no_match edges (negative) | ~6.000–9.000 (ratio 1:3) |
