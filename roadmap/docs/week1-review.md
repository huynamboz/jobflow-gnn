# Review Kiến Trúc & Output — Week 1

> Đánh giá toàn bộ ml-service sau Phase 1 (Data Pipeline) + Phase 2 (GNN + Training).
> Ngày review: 2026-03-29

---

## 1. Tổng Quan Kết Quả

| Chỉ số | Giá trị |
|---|---|
| Tổng file Python (source) | 38 file |
| Tổng file test | 17 file |
| Tổng test | 102 |
| Coverage | 96% |
| Ruff lint | All checks passed |
| Ruff format | All files formatted |

### Các module đã hoàn thành

| Phase | Module | Trạng thái |
|---|---|---|
| 1 | `config/` — Pydantic Settings | Done |
| 1 | `embedding/` — ABC + English + Factory | Done |
| 1 | `data/skill_normalization` — 85 skill chuẩn | Done |
| 1 | `data/skill_taxonomy` — synonym, cluster, template | Done |
| 1 | `data/generator` — tạo CV/Job tổng hợp | Done |
| 1 | `data/labeler` — gán nhãn + chia train/val/test | Done |
| 1 | `graph/schema` — enum, dataclass | Done |
| 1 | `graph/builder` — xây HeteroData (PyG) | Done |
| 1 | `crawler/` — CrawlProvider + JobSpy + storage + SkillExtractor | Done |
| 1 | `inference/` — InferenceEngine + checkpoint | Done |
| 2 | `evaluation/metrics` — Recall@K, MRR, NDCG, AUC-ROC | Done |
| 2 | `baselines/` — Cosine, Skill Overlap, BM25 | Done |
| 2 | `models/gnn` — HeteroGraphSAGE + MLPDecoder | Done |
| 2 | `models/losses` — BPR loss | Done |
| 2 | `training/trainer` — BPR loop + early stopping + hybrid eval | Done |

---

## 2. Đánh Giá Kiến Trúc

### 2.1 Điểm mạnh

**Abstraction tốt.** Module embedding dùng ABC + factory pattern, dễ dàng swap `english` sang `multilingual` sau này. Crawler module cũng theo pattern tương tự. Đúng theo planning doc: "không hardcode english-only".

**Frozen dataclass cho domain object.** `CVData`, `JobData`, `LabeledPair`, `RawJob`, `MatchResult` đều `frozen=True` — ngăn mutation ngoài ý muốn khi data flow qua nhiều tầng.

**Chống rò rỉ nhãn (label leakage).** `_strip_label_edges()` trong `trainer.py` đúng cách loại bỏ edge `match`/`no_match` trước khi GNN message passing. Đây là chi tiết quan trọng dễ bỏ sót trong heterogeneous graph.

**Dữ liệu tổng hợp thực tế.** Generator dùng phân phối có trọng số theo seniority cho skill, proficiency, education, salary. Thêm `synonym_rate`, `implicit_skill_rate`, `cluster_rate` tạo nhiễu có kiểm soát.

**Stratified splitting.** Labeler chia pos/neg riêng biệt thành 75/15/10, test kiểm tra không rò rỉ giữa các split.

**BPR loss phù hợp.** Cho bài toán ranking/retrieval, BPR tối ưu trực tiếp thứ tự xếp hạng — tốt hơn BCE vốn chỉ tối ưu từng điểm riêng lẻ.

**Test suite toàn diện.** 17 file test bao phủ edge case: empty corpus BM25, missing skill, frozen dataclass mutation, gradient flow, early stopping. `FakeEmbeddingProvider` trong conftest tránh download model thật.

**Hybrid scoring.** Kết hợp GNN + skill overlap + seniority cho interpretability và graceful degradation nếu GNN chưa tốt.

### 2.2 Data flow nhất quán

```
skill-alias.json → SkillNormalizer
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
  SyntheticDataGenerator  SkillExtractor  (cho crawled data)
          │
   CVData[] + JobData[]
          │
     PairLabeler
          │
  LabeledPair[] → DatasetSplit (75/15/10)
          │
  EmbeddingProvider.encode()
          │
     GraphBuilder.build()
          │
     HeteroData (PyG)
          │
  ┌───────┴────────┐
  ▼                ▼
_strip_label_edges   (loại match/no_match)
  │
prepare_data_for_gnn (thêm reverse edge)
  │
Trainer.train()
  │
TrainResult → save_checkpoint → InferenceEngine
```

Flow rõ ràng, mỗi bước có input/output xác định. Không có circular dependency.

---

## 3. Các Vấn Đề Phát Hiện

### Mức CRITICAL

#### ISSUE-1: `edge_attr` được tính nhưng GraphSAGE không dùng

**File:** `graph/builder.py:80-94`

```python
# builder.py — tính cẩn thận proficiency weight
data["cv", "has_skill", "skill"].edge_attr = torch.tensor(hs_attr, ...)
data["job", "requires_skill", "skill"].edge_attr = torch.tensor(rs_attr, ...)
```

Nhưng `GraphSAGE` **bỏ qua hoàn toàn** `edge_attr` — chỉ dùng `edge_index` cho neighborhood aggregation. `to_hetero()` không thay đổi hành vi này.

**Hậu quả:** Tín hiệu proficiency (1-5) và importance (1-5) — được thiết kế cẩn thận — hoàn toàn bị mất trong GNN message passing.

**Khắc phục:** Chuyển sang `GATConv` (đọc `edge_attr`) hoặc custom `MessagePassing` layer. Hoặc nếu giữ GraphSAGE, cần ghi nhận đây là limitation và lên kế hoạch cho Phase 3.

---

#### ISSUE-2: InferenceEngine không thực sự dùng GNN embedding

**File:** `inference/engine.py:199-205`

```python
def _score_pair(self, cv, job):
    # GNN score via decode would require the job to be in the graph,
    # so we use text embedding cosine as GNN proxy for new JDs
    gnn_score = self._text_similarity(cv, job)  # ← cosine, KHÔNG phải GNN
```

`_precompute_cv_embeddings()` (dòng 184) chạy `model.encode()` để tính GNN embedding, lưu vào `self._cv_embeddings`, nhưng **không bao giờ được đọc** trong `_score_pair()`. Thay vào đó dùng cosine similarity thuần.

**Hậu quả:** Toàn bộ GNN training trở nên vô nghĩa tại inference — hệ thống thoái hóa thành cosine + skill_overlap + seniority (tức là baselines).

**Khắc phục:** `_score_pair` cần dùng `self._cv_embeddings[cv_idx]` và tính GNN-based score cho job mới. Cách đơn giản nhất: encode job text thành embedding, project qua decoder. Đây là lý do thiết kế `encode()`/`decode()` split — cần triển khai đúng ở inference.

---

#### ISSUE-3: Thứ tự xử lý label edge sai trong InferenceEngine

**File:** `inference/engine.py:187-193`

```python
def _precompute_cv_embeddings(self):
    data_prepared = prepare_data_for_gnn(self._data)      # ToUndirected() → tạo rev_match, rev_no_match
    data_clean = _strip_label_edges(data_prepared)          # chỉ xoá match/no_match, KHÔNG xoá reverse
    data_clean = prepare_data_for_gnn(data_clean)           # ToUndirected() lần 2 — thừa
```

`ToUndirected()` gọi TRƯỚC `_strip_label_edges()` sẽ tạo reverse edge cho match/no_match (ví dụ: `("job", "rev_match", "cv")`). Sau đó `_strip_label_edges` chỉ xoá forward triplet, **để lại reverse label edge** → rò rỉ nhãn.

So sánh với `trainer.py:128-129` (đúng):
```python
data_clean = _strip_label_edges(data)       # strip TRƯỚC
data_clean = prepare_data_for_gnn(data_clean)  # reverse SAU
```

**Khắc phục:** Đảo thứ tự giống trainer, bỏ lần gọi `prepare_data_for_gnn` thừa.

---

### Mức MEDIUM

#### ISSUE-4: Skill overlap không nhất quán giữa labeler và scorer

**File:** `baselines/skill_overlap.py:16` vs `data/labeler.py:143`

| Nơi dùng | Công thức | Ví dụ (CV: 10 skill, Job: 4 skill, trùng 4) |
|---|---|---|
| `SkillOverlapScorer` | Jaccard: `\|A∩B\| / \|A∪B\|` | 4/10 = **0.40** |
| `PairLabeler` | Directional: `\|A∩B\| / \|B\|` | 4/4 = **1.00** |

Labeler gán label dựa trên directional overlap (tập trung vào "job cần gì, CV có bao nhiêu"). Nhưng hybrid scoring eval dùng Jaccard — **đo thứ khác**. CV có nhiều skill thừa sẽ bị phạt bởi Jaccard nhưng không bị phạt bởi labeler.

**Khắc phục:** `SkillOverlapScorer` nên dùng directional overlap giống labeler: `|A∩B| / |B|` (B = job skills). Hoặc document rõ tại sao Jaccard tốt hơn ở evaluation.

---

#### ISSUE-5: Trọng số hybrid khác nhau giữa train và inference

**File:** `config/settings.py:32-34` vs `inference/engine.py:61-63`

| Context | alpha | beta | gamma |
|---|---|---|---|
| `Settings` / `TrainConfig` | 0.6 | 0.3 | 0.1 |
| `InferenceEngine` default | 0.8 | 0.15 | 0.05 |

Training đánh giá với bộ trọng số A, inference scoring với bộ trọng số B. Kết quả evaluation không đại diện cho chất lượng inference thực tế.

**Khắc phục:** `InferenceEngine` nên đọc từ `Settings` hoặc từ checkpoint metadata, không hardcode riêng.

---

#### ISSUE-6: GNN score được min-max normalize theo batch

**File:** `training/trainer.py:240-244`

```python
gnn_min, gnn_max = gnn_scores.min(), gnn_scores.max()
if gnn_max - gnn_min > 1e-8:
    gnn_norm = (gnn_scores - gnn_min) / (gnn_max - gnn_min)
```

Min-max normalize phụ thuộc vào batch — batch toàn điểm cao sẽ bị dàn trải giả tạo, batch toàn điểm thấp cũng vậy. Hai batch khác nhau cho cùng một pair có thể ra normalized score khác nhau.

**Khắc phục:** Dùng sigmoid normalization hoặc fixed scale thay vì min-max per-batch.

---

#### ISSUE-7: `_strip_label_edges` là hàm private nhưng import ở 3 module

**File:** `training/trainer.py` (define) → `inference/engine.py`, `inference/checkpoint.py` (import)

```python
# engine.py, checkpoint.py
from ml_service.training.trainer import _strip_label_edges  # import hàm _private
```

Convention Python: `_` prefix = private. Nếu dùng ở 3 module khác nhau, nên là public utility.

**Khắc phục:** Chuyển sang `graph/utils.py` hoặc `graph/builder.py` với tên `strip_label_edges()` (không underscore).

---

### Mức LOW

#### ISSUE-8: `_sample_bpr_pairs` có thể sample pos_job == neg_job

**File:** `training/trainer.py:87-89`

```python
else:
    neg_job = rng.randint(0, num_jobs)  # có thể trùng pos_job
```

Xác suất thấp (~0.07% với 1500 job) nhưng thiếu rejection check.

**Khắc phục:** Thêm `while neg_job == pos_job: resample`.

---

#### ISSUE-9: `copy.copy()` (shallow copy) cho HeteroData

**File:** `training/trainer.py:51`, `tests/test_gnn_model.py:23`

Shallow copy chia sẻ reference đến tensor bên trong. Hiện tại không có in-place mutation nên chạy đúng, nhưng dễ vỡ nếu ai đó thêm in-place op.

**Khắc phục:** Dùng `copy.deepcopy()` hoặc construct HeteroData mới chỉ với edge types cần thiết.

---

#### ISSUE-10: `embedding_dim` trong Settings không được dùng ở đâu

**File:** `config/settings.py:9`

```python
embedding_dim: int = 384  # Dead config — không module nào đọc
```

Dimension thực tế hardcode trong `gnn.py:54` (`node_dims = {"cv": 386, ...}`).

---

#### ISSUE-11: `scikit-learn` không khai báo trong `pyproject.toml`

**File:** `evaluation/metrics.py:4`

```python
from sklearn.metrics import roc_auc_score  # transitive dep qua sentence-transformers
```

Hoạt động nhưng là undeclared dependency — nếu đổi embedding provider có thể mất sklearn.

---

#### ISSUE-12: `skill-alias.json` metadata ghi 75 skill nhưng test assert 85

**File:** `skill-alias.json:11` vs `tests/test_skill_normalization.py:14`

Metadata stale, cần cập nhật `"total_canonical_skills": 85`.

---

#### ISSUE-13: `dataclasses.asdict` import nhưng không dùng trong `checkpoint.py`

**File:** `inference/checkpoint.py:7`

```python
from dataclasses import asdict  # unused import
```

---

#### ISSUE-14: `_KEEP_COLS` trong `jobspy_provider.py` không được dùng

**File:** `crawler/jobspy_provider.py:14-25`

Defined nhưng `_to_raw_jobs()` không filter columns theo list này.

---

## 4. Thiếu Sót Test Coverage

| # | Mô tả | Mức độ |
|---|---|---|
| T-1 | **Không có integration test end-to-end** (generate → train → checkpoint → inference → verify) | High |
| T-2 | `utils/logging.py` — 0% coverage | Low |
| T-3 | Không test `_strip_label_edges` khi graph không có label edge | Low |
| T-4 | Không test `prepare_data_for_gnn` + `to_hetero` metadata consistency giữa construction và inference | Medium |
| T-5 | Không test `InferenceEngine._precompute_cv_embeddings` output shape | Medium |
| T-6 | Không test `skill_extractor.py` với description không có skill nào | Low |

---

## 5. Đánh Giá Thiết Kế

### `to_hetero()` + `ToUndirected()` — Đúng

Cách tiếp cận đúng cho PyG heterogeneous graph:
- `to_hetero()` dispatch homogeneous GraphSAGE operations per edge type
- `ToUndirected()` đảm bảo message passing hai chiều (skill/seniority node gửi message ngược lại cv/job)

**Lưu ý:** `ToUndirected()` PHẢI gọi SAU `_strip_label_edges()`. Nếu ngược lại, reverse label edge sẽ rò rỉ thông tin (ISSUE-3).

### BPR + Hybrid Scoring — Hợp lý

- BPR tối ưu GNN cho pairwise ranking
- Hybrid scoring thêm thành phần non-learned (skill overlap, seniority) cho interpretability
- Nếu GNN yếu (ít data), skill overlap + seniority vẫn đảm bảo chất lượng baseline

**Lưu ý:** GNN score normalize per-batch (ISSUE-6) làm giảm tính ổn định.

### Encode/Decode Split — Tốt nhưng chưa khai thác đúng

Thiết kế cho phép precompute CV embedding một lần, decode nhanh cho JD mới. Nhưng `InferenceEngine` hiện tại không dùng GNN embedding (ISSUE-2) — tính rồi bỏ.

---

## 6. Bảng Tóm Tắt Issues Theo Ưu Tiên

### Cần sửa trước Phase 3

| # | Issue | File chính | Mức |
|---|---|---|---|
| 1 | `edge_attr` bị bỏ qua bởi GraphSAGE | `models/gnn.py` | Critical |
| 2 | InferenceEngine không dùng GNN embedding | `inference/engine.py:199` | Critical |
| 3 | Thứ tự strip/reverse sai trong inference | `inference/engine.py:187` | Critical |
| 4 | Skill overlap Jaccard vs directional | `baselines/skill_overlap.py` | Medium |
| 5 | Trọng số hybrid không nhất quán | `inference/engine.py:61` | Medium |
| 7 | `_strip_label_edges` nên public | `training/trainer.py` | Medium |

### Nên sửa sớm

| # | Issue | File | Mức |
|---|---|---|---|
| 6 | Min-max normalize per-batch | `training/trainer.py:240` | Medium |
| 11 | `scikit-learn` undeclared dep | `pyproject.toml` | Low |
| T-1 | Thiếu integration test e2e | `tests/` | High |

### Sửa khi tiện

| # | Issue | File | Mức |
|---|---|---|---|
| 8 | BPR neg == pos possible | `training/trainer.py:89` | Low |
| 9 | Shallow copy HeteroData | `training/trainer.py:51` | Low |
| 10 | Dead config `embedding_dim` | `config/settings.py:9` | Low |
| 12 | Stale metadata 75→85 | `skill-alias.json:11` | Low |
| 13 | Unused import `asdict` | `inference/checkpoint.py:7` | Low |
| 14 | Unused `_KEEP_COLS` | `crawler/jobspy_provider.py:14` | Low |

---

## 7. Kết Luận

Kiến trúc Phase 1+2 **vững và được test tốt** (102 test, 96% coverage). Abstraction sạch, data flow rõ ràng, không circular dependency.

**3 vấn đề critical cần ưu tiên** trước khi Phase 3:
1. Inference engine phải dùng GNN embedding thực sự (không fallback cosine)
2. Sửa thứ tự strip/reverse trong inference engine
3. Lên kế hoạch chuyển sang GATConv để tận dụng edge weight

**Vấn đề medium ảnh hưởng chất lượng:** Jaccard vs directional overlap tạo misalignment giữa cách gán nhãn và cách đánh giá — sửa đơn giản, tác động lớn.

Sau khi sửa các critical issue, codebase sẵn sàng cho Phase 3 (API endpoints, GATConv migration, real data benchmark).
