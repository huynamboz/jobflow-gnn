# Week 1 — Vấn đề gặp phải & Giải pháp

## 1. Circular Evaluation (Synthetic Data)

**Vấn đề:** Benchmark đầu tiên trên synthetic data cho kết quả vô nghĩa — Skill Overlap baseline đạt AUC 0.96, GNN không thể thắng.

**Nguyên nhân:** Synthetic data được label bằng `skill_overlap >= 0.5 AND seniority_distance <= 1`. Khi evaluate, SkillOverlapScorer tính chính xác cùng formula → baseline "biết đáp án" sẵn.

**Bằng chứng:**
```
Lần 1 (clean synthetic):
  Skill Overlap AUC: 0.9591  ← gần perfect, vô nghĩa
  GNN AUC:           0.6243  ← thua baseline
  MRR = 1.0 cho 3/4 methods → test quá dễ
```

**Giải pháp:** Thêm 4 loại noise vào synthetic data:
1. **Implicit skills** (20%): CV có skill trong text nhưng không trong structured `skills` tuple → SkillOverlap baseline bỏ sót, GNN qua embedding vẫn match
2. **Synonym text** (30%): "React" → "ReactJS", "React.js" → cosine baseline kém hơn
3. **Skill clusters** (30%): JD yêu cầu "fullstack" → CV cần cover cluster, không chỉ individual skills
4. **Label noise** (12%): Flip labels ngẫu nhiên → không rule nào đạt perfect

**Kết quả sau fix:**
```
Lần 2 (noisy synthetic):
  Skill Overlap AUC: 0.6685  ← giảm từ 0.96 (expected)
  GNN AUC:           0.6022
  GNN Recall@5 = Skill Overlap Recall@5 → gap thu hẹp
```

**Files:** `data/skill_taxonomy.py`, sửa `generator.py` + `labeler.py`

---

## 2. GNN Underfitting (Loss không giảm)

**Vấn đề:** Training đầu tiên — loss giảm rất chậm (0.69 → 0.66), early stop ở epoch 10.

**Nguyên nhân:**
- Learning rate quá nhỏ (1e-3) cho graph nhỏ
- Hybrid scoring (β=0.3 skill_overlap) dominate → val_mrr=1.0 ngay epoch 0 → GNN chưa cần học gì
- Early stopping patience=10 quá ngắn

**Giải pháp:** Tune hyperparameters:
```
BEFORE: lr=1e-3, epochs=50, patience=10, alpha=0.6, beta=0.3, gamma=0.1
AFTER:  lr=5e-3, epochs=200, patience=30, alpha=0.8, beta=0.15, gamma=0.05
```

- Tăng lr 5x → loss giảm nhanh hơn
- Tăng alpha (GNN weight) 0.6→0.8, giảm beta (skill overlap) 0.3→0.15 → GNN có room học
- Tăng patience → cho GNN thêm thời gian converge

**Kết quả:** Loss giảm 0.69 → 0.43 (so với 0.69 → 0.66 trước đó)

---

## 3. Skill Extraction False Positive ("c")

**Vấn đề:** Skill `"c"` xuất hiện 121 lần trong crawl data — nhiều thứ 3 sau python và aws. Rõ ràng false positive.

**Nguyên nhân:** Tokenizer `re.findall(r"[\w#+.]+", text)` match chữ "C" đơn lẻ trong text (ví dụ "C-suite", "C level", "from A to C") → normalize thành `"c"` (C programming language).

**Trạng thái:** Đã nhận biết, chưa fix. Cần thêm min-length filter hoặc context-aware extraction.

**Workaround tạm:** Filter `len(j.skills) >= 2` khi load data → bỏ JDs chỉ match false positives.

---

## 4. Tokenizer bỏ sót skills do trailing punctuation

**Vấn đề:** Test `test_extract_skills_from_description` fail — "AWS" và "Git" không được extract.

**Nguyên nhân:** Regex `[\w#+.]+` match "AWS." (có dấu chấm cuối câu) → `normalize("AWS.")` → None.

**Giải pháp:** Strip trailing punctuation sau khi tokenize:
```python
# BEFORE
words = re.findall(r"[\w#+.]+", text)

# AFTER
words = [w.rstrip(".,;:") for w in re.findall(r"[\w#+.]+", text)]
```

**File:** `crawler/skill_extractor.py`

---

## 5. Seniority Inference sai thứ tự

**Vấn đề:** `title="Tech Lead"` nhưng description chứa "senior" → infer thành SENIOR thay vì LEAD.

**Nguyên nhân:** Pattern check title + description cùng lúc, "senior" match trước "lead" theo thứ tự pattern list.

**Giải pháp:** Check title trước (signal mạnh hơn), chỉ fallback sang description nếu title không match:
```python
# BEFORE
text = f"{title} {description[:500]}"
for pattern, level in _SENIORITY_PATTERNS:
    if pattern.search(text): return level

# AFTER
for pattern, level in _SENIORITY_PATTERNS:
    if pattern.search(title): return level  # title first
for pattern, level in _SENIORITY_PATTERNS:
    if pattern.search(description[:500]): return level  # fallback
```

**File:** `crawler/skill_extractor.py`

---

## 6. Model checkpoint load — metadata mismatch

**Vấn đề:** `load_checkpoint` fail với `Missing key(s) in state_dict` — model chứa edge types match/no_match mà saved model không có.

**Nguyên nhân:** Model được train trên data đã strip label edges, nhưng `load_checkpoint` build metadata từ raw data (có label edges) → model architecture khác nhau.

**Giải pháp:** Strip label edges trong `load_checkpoint` trước khi build metadata:
```python
# BEFORE
data_prepared = prepare_data_for_gnn(data)

# AFTER
data_clean = _strip_label_edges(data)
data_prepared = prepare_data_for_gnn(data_clean)
```

**File:** `inference/checkpoint.py`

---

## 7. Dependency conflict (regex version)

**Vấn đề:** Sau install `python-jobspy`, test `test_factory_unknown_raises` fail với `ImportError: regex>=2025.10.22 is required`.

**Nguyên nhân:** `python-jobspy` cài `regex==2024.11.6` nhưng `transformers` yêu cầu `regex>=2025.10.22`.

**Giải pháp:** `pip install "regex>=2025.10.22"` — chấp nhận jobspy compatibility warning (vẫn hoạt động).

---

## 8. Multilingual layer — quyết định architecture

**Vấn đề:** Ban đầu plan yêu cầu multilingual embedding (EN+VI) ngay từ MVP, tăng complexity không cần thiết.

**Quyết định:** Tách multilingual thành optional layer:
- Build English-first với `all-MiniLM-L6-v2`
- `EmbeddingProvider` abstract interface → swap bằng 1 dòng config
- `MultilingualProvider` stub (raise NotImplementedError)
- Không hardcode "english-only" vào code ngoài config

**Kết quả:** Kiến trúc clean, dễ upgrade sau mà không break code.
