# Plan: Per-CV Evaluation Protocol

## 1. Vấn đề hiện tại

### Evaluation hiện tại hoạt động thế nào
- `_evaluate_split()` trong `trainer.py` nhận danh sách `LabeledPair` (test split)
- Tính hybrid score cho **từng cặp CV-Job** trong test set
- Gọi `compute_all_metrics(y_true, y_scores)` trên **toàn bộ pairs cùng lúc**
- Metrics được tính **global** — không phân biệt pair nào thuộc CV nào

### Tại sao global evaluation có vấn đề
1. **Bias toward CVs with more pairs** — CV có 50 pairs ảnh hưởng gấp 10 lần CV có 5 pairs
2. **Không phản ánh production** — Thực tế user upload 1 CV → hệ thống rank toàn bộ jobs
3. **Metrics không ổn định** — Random split khác nhau → metrics dao động lớn (đã thấy ở Week 3)
4. **Không phát hiện edge cases** — Không biết CV nào model rank tốt, CV nào rank tệ

### References
- **Krichene & Rendle (KDD 2020)** — "On Sampled Metrics for Item Recommendation": chứng minh sampled metrics không nhất quán với full ranking, khuyến nghị dùng full ranking khi feasible
- **LightGCN (SIGIR 2020)** — Paper GNN-for-recommendation được cite nhiều nhất, dùng per-user full ranking evaluation với Recall@K và NDCG@K
- **RecBole framework** — Hỗ trợ `full` mode (rank all items) là gold standard

---

## 2. Per-CV Evaluation là gì

### Protocol chuẩn (theo literature)
Với **mỗi CV trong test set**:
1. Score **toàn bộ** jobs (6,020 jobs) cho CV đó
2. **Loại bỏ** jobs đã xuất hiện trong training set của CV đó (tránh trivial recommendation)
3. Rank jobs theo score giảm dần
4. So sánh ranked list với **ground truth** (các jobs mà CV match trong test set)
5. Tính Recall@K, NDCG@K, Precision@K, MRR **riêng cho CV đó**

Sau đó: **Lấy trung bình (macro-average)** tất cả CV

### Công thức per-CV

```
Recall@K(cv_i) = |{relevant jobs in top-K}| / |{all relevant jobs for cv_i in test}|

Precision@K(cv_i) = |{relevant jobs in top-K}| / K

NDCG@K(cv_i):
  DCG@K  = Σ(i=1→K) rel_i / log2(i + 1)
  IDCG@K = Σ(i=1→min(K, |relevant|)) 1 / log2(i + 1)
  NDCG@K = DCG@K / IDCG@K

MRR(cv_i) = 1 / rank_of_first_relevant_job

Aggregate:
  Metric = (1/N) * Σ(i=1→N) Metric(cv_i)    (N = số CV trong test)
```

---

## 3. Thách thức đặc thù của project này

### 3.1. Ground truth là proxy labels, không phải real interactions
- Hệ thống không có dữ liệu user click/apply thật
- Labels được tạo bởi `PairLabeler` dựa trên skill overlap thresholds:
  - Positive: `direct_overlap >= 0.4 AND seniority_distance <= 1`
  - Negative: `direct_overlap < 0.15 OR seniority_distance >= 3`
- **Hệ quả:** Ground truth bản thân đã dựa trên skill overlap → GNN cần học được patterns **beyond** skill overlap (ví dụ: related skills, graph structure) mới thực sự outperform baseline

### 3.2. Labeled pairs rất sparse
- 9,889 labeled pairs / (362 CVs × 6,020 Jobs) = **0.45%** coverage
- Mỗi CV trung bình chỉ có ~27 labeled pairs (trong đó ~9 positive)
- **Vấn đề:** Khi rank toàn bộ 6,020 jobs cho 1 CV, chỉ biết ~9 jobs là relevant → Recall@10 sẽ rất thấp nếu model rank đúng nhưng relevant jobs khác không có label

### 3.3. Quyết định: xử lý unlabeled pairs thế nào?
**Option A — Chỉ evaluate trên labeled pairs (hiện tại)**
- Pros: Đơn giản, không cần giả định
- Cons: Không phải full ranking, metrics không phản ánh production

**Option B — Full ranking, unlabeled = negative (standard in literature)**
- Pros: Đúng protocol chuẩn, phản ánh production
- Cons: Nhiều false negatives (CV-Job thực sự match nhưng chưa được label)
- **Đây là approach chuẩn** — LightGCN, RecBole đều dùng cách này

**Option C — Label toàn bộ CV-Job pairs trước khi evaluate**
- Pros: Ground truth đầy đủ nhất
- Cons: 362 × 6,020 = 2.18M pairs → chạy labeler rất lâu, và labels vẫn là proxy

### Quyết định: Dùng Option B (Full ranking, unlabeled = irrelevant)
**Lý do:**
- Đúng chuẩn literature (Krichene & Rendle 2020, LightGCN)
- Feasible: 6,020 jobs đủ nhỏ để full ranking (không cần sampling)
- Metrics sẽ thấp hơn hiện tại (vì harder task) nhưng **đáng tin cậy hơn**
- Phản ánh đúng production: user upload CV → rank ALL jobs

**Bổ sung:** Chạy labeler cho toàn bộ pairs **chỉ để phân tích** (không dùng làm ground truth chính), giúp biết bao nhiêu "relevant" jobs bị miss.

---

## 4. Kế hoạch implementation

### Phase 1: Per-CV Evaluator (core)

**File mới:** `backend/ml_service/evaluation/per_cv_evaluator.py`

```python
class PerCVEvaluator:
    """Evaluate model performance per-CV: each CV ranks ALL jobs."""

    def __init__(self, engine: InferenceEngine, cvs, jobs, dataset_split):
        ...

    def evaluate(self, ks=(5, 10, 20, 50)) -> PerCVResult:
        """
        For each test CV:
          1. Score ALL jobs using engine._score_pair()
          2. Exclude jobs that appeared in CV's training pairs
          3. Rank by score descending
          4. Compute metrics against test positive pairs
          5. Store per-CV breakdown
        Return aggregated metrics (macro-average)
        """

    def _get_cv_train_jobs(self, cv_id) -> set[int]:
        """Jobs this CV interacted with in training (to exclude)."""

    def _get_cv_test_positives(self, cv_id) -> set[int]:
        """Jobs labeled positive for this CV in test split."""

    def _evaluate_single_cv(self, cv, all_jobs, exclude_jobs, relevant_jobs, ks):
        """Score all jobs, rank, compute metrics for one CV."""
```

**Output:** `PerCVResult`
```python
@dataclass
class PerCVResult:
    # Aggregated (macro-average across all CVs)
    avg_metrics: dict[str, float]  # recall@5, ndcg@10, ...

    # Per-CV breakdown
    per_cv_metrics: dict[int, dict[str, float]]  # cv_id → metrics

    # Analysis
    num_cvs_evaluated: int
    num_cvs_with_test_positives: int
    avg_test_positives_per_cv: float
    worst_cvs: list[int]  # CVs with lowest metrics
    best_cvs: list[int]   # CVs with highest metrics
```

### Phase 2: Tích hợp vào training loop

**Sửa file:** `backend/ml_service/training/trainer.py`

- Sau khi train xong (final test evaluation), **thêm** per-CV evaluation
- Không thay thế `_evaluate_split()` hiện tại — chạy song song cả hai để so sánh
- Log kết quả per-CV vào `TrainResult`

### Phase 3: Tích hợp vào experiment scripts

**Sửa files:**
- `backend/run_experiment_linkedin_cv.py`
- `backend/run_experiment_real_cv.py`

- Thêm per-CV evaluation cho cả GNN và baselines
- So sánh: GNN vs baselines dưới per-CV protocol

### Phase 4: Analysis & Reporting

- **Edge case analysis:** Tìm CVs có metrics thấp nhất, phân tích tại sao
  - Thiếu skills? Role hiếm? Ít positive pairs?
- **Stratified evaluation:** Nhóm CVs theo:
  - Số lượng skills (ít vs nhiều)
  - Role (Frontend, Backend, Data, DevOps, Mobile)
  - Số training interactions (cold-start analysis)
- **Baseline comparison:** So sánh per-CV metrics của GNN vs Cosine vs Jaccard vs BM25

---

## 5. Thứ tự triển khai

| Step | Task | File | Estimate |
|------|------|------|----------|
| 1 | Implement `PerCVEvaluator` class | `evaluation/per_cv_evaluator.py` | Core logic |
| 2 | Viết unit tests | `tests_ml/test_per_cv_evaluator.py` | Đảm bảo metrics đúng |
| 3 | Chạy per-CV evaluation trên checkpoint hiện tại | Script/notebook | Xem baseline metrics |
| 4 | So sánh kết quả per-CV vs global evaluation | Analysis | Hiểu sự khác biệt |
| 5 | Tích hợp vào training loop | `training/trainer.py` | Dùng cho hyperparameter tuning |
| 6 | Tích hợp vào experiment scripts | `run_experiment_*.py` | Benchmark baselines |
| 7 | Edge case analysis + stratified report | `roadmap/week9/report/` | Viết findings |

---

## 6. Metrics kỳ vọng

Per-CV evaluation sẽ cho metrics **thấp hơn** global evaluation vì:
- Task khó hơn: rank 6,020 jobs thay vì vài chục pairs
- Nhiều false negatives: relevant jobs chưa được label
- Recall@10 kỳ vọng: **0.01 – 0.05** (10 / 6020 = rất nhỏ)

**Không nên so sánh trực tiếp** per-CV metrics với global metrics. Thay vào đó, focus vào:
- **GNN vs Baselines gap** — GNN có outperform không dưới protocol mới?
- **Per-CV variance** — Metrics ổn định hay dao động lớn giữa các CV?
- **Worst-case analysis** — CV nào model rank tệ nhất? Tại sao?

---

## 7. Rủi ro & Mitigation

| Rủi ro | Mức độ | Giải pháp |
|--------|--------|-----------|
| Quá ít test positives per CV → metrics không reliable | Cao | Filter: chỉ evaluate CVs có >= 3 test positives |
| Full ranking chậm (362 CVs × 6,020 jobs) | Thấp | ~2.18M scores, feasible với batch scoring |
| Proxy labels sai → per-CV evaluation cũng sai | Trung bình | Bổ sung manual evaluation trên 10-20 CVs |
| Metrics quá thấp → khó báo cáo | Trung bình | Dùng K lớn hơn (K=100, 200) và focus vào relative improvement |
| GNN score normalization ảnh hưởng ranking | Cao | Dùng min-max normalization per-CV thay vì sigmoid (đã fix) |

**Ghi chú:** Lần chạy đầu dùng sigmoid normalization cho GNN scores → tất cả per-CV metrics = 0. Nguyên nhân: sigmoid nén scores vào ~0.5, mất khả năng phân biệt. Fix: dùng min-max normalization per-CV (consistent với training evaluation). K values cũng cần mở rộng đến K=100, 200 vì test positives rất sparse (~1-2 jobs / CV trong 6020).

---

## 8. References

### [1] Krichene & Rendle — On Sampled Metrics (KDD 2020)
- **Citation:** W. Krichene and S. Rendle, "On Sampled Metrics for Item Recommendation," in *Proc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining (KDD '20)*, 2020, pp. 1748–1757.
- **DOI:** [10.1145/3394486.3403226](https://doi.org/10.1145/3394486.3403226)
- **Relevance:** Chứng minh sampled ranking metrics không nhất quán với full ranking — là cơ sở lý thuyết cho việc dùng full ranking evaluation trong project này.

### [2] He et al. — LightGCN (SIGIR 2020)
- **Citation:** X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," in *Proc. 43rd Int. ACM SIGIR Conf. Research and Development in Information Retrieval (SIGIR '20)*, 2020, pp. 639–648.
- **DOI:** [10.1145/3397271.3401063](https://doi.org/10.1145/3397271.3401063)
- **arXiv:** [2002.02126](https://arxiv.org/abs/2002.02126)
- **Relevance:** Paper GNN-for-recommendation được cite nhiều nhất, dùng per-user full ranking evaluation với Recall@K, NDCG@K — là protocol chuẩn mà project này follow.

### [3] Zhao et al. — RecBole (CIKM 2021)
- **Citation:** W. X. Zhao, S. Mu, Y. Hou, Z. Lin, et al., "RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms," in *Proc. 30th ACM Int. Conf. Information & Knowledge Management (CIKM '21)*, 2021, pp. 4653–4664.
- **DOI:** [10.1145/3459637.3482016](https://doi.org/10.1145/3459637.3482016)
- **arXiv:** [2011.01731](https://arxiv.org/abs/2011.01731)
- **Relevance:** Framework recommendation chuẩn hóa evaluation protocol, `full` ranking mode là gold standard.

### [4] Meng et al. — Data Splitting Strategies (RecSys 2020)
- **Citation:** Z. Meng, R. McCreadie, C. Macdonald, and I. Ounis, "Exploring Data Splitting Strategies for the Evaluation of Recommendation Models," in *Proc. 14th ACM Conf. Recommender Systems (RecSys '20)*, 2020, pp. 681–686.
- **DOI:** [10.1145/3383313.3418479](https://doi.org/10.1145/3383313.3418479)
- **arXiv:** [2007.13237](https://arxiv.org/abs/2007.13237)
- **Relevance:** Chứng minh splitting strategy là confounding variable lớn — hỗ trợ quyết định dùng per-CV evaluation thay vì global random split.

### [5] Liu et al. — LinkSAGE (KDD 2025)
- **Citation:** P. Liu, H. Wei, X. Hou, J. Shen, et al., "LinkSAGE: Optimizing Job Matching Using Graph Neural Networks," in *Proc. 31st ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD '25)*, Toronto, Canada, 2025.
- **DOI:** [10.1145/3690624.3709396](https://doi.org/10.1145/3690624.3709396)
- **arXiv:** [2402.13430](https://arxiv.org/abs/2402.13430)
- **Relevance:** Hệ thống GNN job matching ở production (LinkedIn) — reference cho kiến trúc và approach tương tự project này.
