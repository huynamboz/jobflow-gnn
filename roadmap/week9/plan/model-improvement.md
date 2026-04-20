# Plan: Cải thiện mô hình GNN — Từ tốt đến xuất sắc

> Tài liệu này mô tả chi tiết các kỹ thuật cải thiện mô hình GNN theo 2 giai đoạn:
> - **Phase 1 (Good):** AUC-ROC 0.71 → **0.78–0.80**, thực hiện trong 1–2 tuần
> - **Phase 2 (Excellent):** AUC-ROC 0.80 → **0.83–0.87**, thực hiện sau khi Phase 1 ổn định

---

## Trạng thái hiện tại (Baseline)

| Thông số | Giá trị |
|----------|---------|
| Model | HeteroGraphSAGE |
| hidden_channels | 256 |
| num_layers | 3 |
| Loss | BPR (Bayesian Personalized Ranking) |
| Negative sampling | Hard negatives (70% hard / 30% random) |
| Hybrid scoring | α=0.55·GNN + β=0.30·SkillOverlap + γ=0.15·Seniority |
| Dataset | 511 CVs, 6,020 jobs, 9,889 labeled pairs |
| AUC-ROC | ~0.71 (global eval) |
| NDCG@10 | ~0.78 (global eval) |

**Bottlenecks đã xác định:**
1. Dataset nhỏ (511 CVs × 6,020 jobs, chỉ 0.32% coverage)
2. Proxy labels (skill-overlap thresholds, không phải real user data)
3. Chưa có skill-to-skill semantic edges → GNN không học được "Flask relates to Django"
4. Regularization yếu (không có DropEdge, chưa có dropout tuning)
5. Negative sampling chưa có curriculum schedule
6. Evaluation hiện tại (global) có bias — cần per-CV protocol

---

# PHASE 1 — Mức tốt (AUC-ROC 0.78–0.80)

> **Triết lý:** Không thay đổi kiến trúc lớn. Squeeze maximum performance từ pipeline hiện tại bằng cách fix các điểm yếu đã biết.

---

## P1.1. Tăng và làm sạch training data

**Vấn đề:** 511 CVs còn ít. Graph thưa thớt → message passing không hiệu quả. Proxy labels từ skill overlap thresholds có nhiễu.

**Giải pháp:**

### Tăng NUM_POSITIVE_PAIRS
```python
# Trước
NUM_POSITIVE_PAIRS = 2000

# Sau — 511 CVs × avg 8 positives/CV ≈ 4,000 pairs tối đa
NUM_POSITIVE_PAIRS = 3500
```
Với 511 CVs (từ 362), số eligible positive pairs tăng. Tăng pairs giúp model thấy nhiều patterns hơn.

### Giảm NOISE_RATE
```python
# Trước
NOISE_RATE = 0.10

# Sau — data đã tốt hơn sau khi fix catalog
NOISE_RATE = 0.05
```

**Kỳ vọng:** +1–2% AUC-ROC.

---

## P1.2. Hyperparameter Tuning — Grid Search

**Vấn đề:** Config hiện tại (`hidden=256, layers=3, lr=1e-3`) chưa tối ưu cho dataset kích thước này.

**Grid search:**

| Tham số | Thử nghiệm | Hiện tại |
|---------|-----------|---------|
| hidden_channels | 128, **256**, 384, 512 | 256 |
| num_layers | 2, **3**, 4 | 3 |
| lr | 1e-4, **5e-4**, 1e-3 | 1e-3 |
| dropout | 0.0, **0.1**, 0.2, 0.3 | 0.0 |
| hybrid_alpha (GNN weight) | 0.45, **0.55**, 0.65 | 0.55 |

**Chú ý về num_layers:**
- 2 layers: mỗi node nhận thông tin từ 2-hop neighbors → ổn cho graph nhỏ
- 3 layers: tốt nhất với dense graph
- 4 layers: dễ bị over-smoothing trên small graph (tất cả node embeddings converge)
- **Recommended:** thử 2 và 3 trước khi thêm layer

**Chú ý về dropout:**
- Dataset nhỏ → dễ overfit → dropout 0.1–0.2 giúp generalize
- Áp dụng sau mỗi GNN layer, không phải trên edge features

**Kỳ vọng:** +2–3% AUC-ROC.

---

## P1.3. DropEdge Regularization

**Vấn đề:** Với graph nhỏ (6,596 nodes), GNN dễ memorize training patterns thay vì generalize.

**Kỹ thuật:** DropEdge (Rong et al., ICLR 2020) — randomly drop một tỷ lệ edges trong mỗi training iteration. Ngăn model overfit vào cấu trúc cụ thể của graph, đồng thời giảm over-smoothing khi dùng nhiều layers.

**Thực nghiệm trên các benchmark:**
- Citeseer (2-layer GCN): +0.9% accuracy
- Citeseer (64-layer GCN): +13.5% accuracy
- → **Tác dụng lớn hơn khi dùng nhiều layers**

**Implementation:**

```python
# Trong trainer.py — thêm vào TrainConfig
@dataclass
class TrainConfig:
    ...
    drop_edge_rate: float = 0.2  # Drop 20% edges during training

# Trong training loop — trước encode()
def _apply_drop_edge(data: HeteroData, rate: float) -> HeteroData:
    """Randomly drop edges for regularization (DropEdge)."""
    if rate <= 0:
        return data
    data = copy.copy(data)
    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        n_edges = ei.size(1)
        mask = torch.rand(n_edges) > rate  # Keep (1-rate) fraction
        data[edge_type].edge_index = ei[:, mask]
    return data

# Trong training step:
data_aug = _apply_drop_edge(data_clean, cfg.drop_edge_rate) if model.training else data_clean
z_dict = model.encode(data_aug)
```

**DropEdge rate recommendations:**
- Graph nhỏ (< 1K nodes): 0.1–0.2
- Graph vừa (1K–10K nodes): 0.2–0.3
- Graph lớn: 0.3–0.5

**Kỳ vọng:** +2–4% AUC-ROC (đặc biệt khi dùng 3 layers).

**Reference:** Rong et al., "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification," ICLR 2020. [[arxiv:1907.10903]](https://arxiv.org/abs/1907.10903)

---

## P1.4. Curriculum Negative Sampling

**Vấn đề hiện tại:** Hard negatives (70%) ngay từ đầu training → model chưa có representation tốt → học sai signal → instability.

**Kỹ thuật:** Curriculum learning cho negative sampling — bắt đầu với easy negatives, dần chuyển sang hard negatives khi model đã stable.

**Schedule:**

```
Epochs 1–5   : 100% random negatives (warmup — model học basic patterns)
Epochs 6–20  : 30% hard / 70% random (transition)
Epochs 21+   : 70% hard / 30% random (current default — khai thác)
```

**Implementation trong `_sample_bpr_pairs()`:**

```python
def _sample_bpr_pairs(pairs, rng, ..., hard_neg_ratio=0.7, epoch=0):
    # Curriculum schedule
    if epoch < 5:
        effective_hard_ratio = 0.0   # Warmup: all random
    elif epoch < 20:
        effective_hard_ratio = 0.3   # Transition
    else:
        effective_hard_ratio = hard_neg_ratio  # Full hard negatives
    ...
```

**Lý do hoạt động:** Khi model chưa được train, hard negatives gần với positives về embedding → model nhận gradient sai, dẫn đến saddle points. Curriculum cho phép model học representation đơn giản trước, rồi mới tập trung vào hard cases.

**Reference:** Cascianelli et al., "CuCo: Graph Representation with Curriculum Contrastive Learning," IJCAI 2021. [[link]](https://www.ijcai.org/proceedings/2021/0317.pdf)

**Kỳ vọng:** +1–2% AUC-ROC, đặc biệt cải thiện training stability.

---

## P1.5. Skill Co-occurrence Edges (Skill Graph Enrichment)

**Vấn đề:** Hiện tại, GNN không biết Flask và Django có liên quan nhau. CV có Flask không được match với Job yêu cầu Django → false negative.

**Kỹ thuật:** Thêm `skill_similar_to` edges dựa trên **PMI (Pointwise Mutual Information)** tính từ job postings.

**Công thức PMI:**
```
PMI(skill_A, skill_B) = log[ P(A,B) / (P(A) × P(B)) ]

Trong đó:
  P(A,B)  = tỷ lệ job postings chứa cả skill A và skill B
  P(A)    = tỷ lệ job postings chứa skill A
  
Chỉ giữ cặp có PMI > 0 (co-occur nhiều hơn expected) và count >= 5 jobs
```

**Implementation:**

```python
# backend/ml_service/data/skill_graph.py (đã có infrastructure)
def build_skill_cooccurrence(cvs, jobs, min_count=5, min_pmi=0.5):
    """Build PMI-weighted skill co-occurrence from job postings."""
    from collections import Counter
    import math
    
    # Count skill occurrences
    skill_count = Counter()
    pair_count = Counter()
    total_docs = len(jobs)
    
    for job in jobs:
        skills = set(job.skills)
        for s in skills:
            skill_count[s] += 1
        for s1 in skills:
            for s2 in skills:
                if s1 < s2:  # deduplicate
                    pair_count[(s1, s2)] += 1
    
    # Compute PMI
    edges = {}
    for (s1, s2), count in pair_count.items():
        if count < min_count:
            continue
        p_ab = count / total_docs
        p_a = skill_count[s1] / total_docs
        p_b = skill_count[s2] / total_docs
        pmi = math.log(p_ab / (p_a * p_b)) if p_a * p_b > 0 else 0
        if pmi > min_pmi:
            edges[(s1, s2)] = pmi
    
    return edges

# Thêm vào GraphBuilder.build():
# skill_to_idx đã có
# skill_cooc_edges = build_skill_cooccurrence(cvs, jobs)
# → thêm edge type ("skill", "similar_to", "skill")
```

**Edge schema sau khi thêm:**
- `("skill", "similar_to", "skill")` — PMI-weighted, bidirectional

**Tại sao quan trọng:** Graph hiện tại chỉ kết nối Skill qua CV và Job. Không có direct Skill-Skill edges. Thêm skill co-occurrence cho phép GNN tìm path: `CV → Flask → [similar_to] → Django → Job`.

**Cẩn thận:** Skill-skill edges có thể tạo noise nếu PMI thấp. Cần threshold tối thiểu: `min_pmi=0.5`, `min_count=5`.

**Reference:** Nguyen et al., "Job Posting-Enriched Knowledge Graph for Skills-based Matching," arxiv:2109.02554. [[link]](https://arxiv.org/abs/2109.02554)

**Kỳ vọng:** +2–4% AUC-ROC, đặc biệt cải thiện Recall@K.

---

## Tóm tắt Phase 1

| Kỹ thuật | Độ khó | Thời gian | Kỳ vọng |
|----------|--------|-----------|---------|
| P1.1 Tăng training data | Thấp | 30 phút | +1–2% |
| P1.2 Hyperparameter tuning | Thấp | 1–2 ngày chạy | +2–3% |
| P1.3 DropEdge | Thấp | 2 giờ code | +2–4% |
| P1.4 Curriculum negatives | Thấp | 2 giờ code | +1–2% |
| P1.5 Skill co-occurrence | Trung bình | 1 ngày | +2–4% |
| **Tổng cộng** | | **3–4 ngày** | **+8–15%** |

**Target Phase 1:** AUC-ROC **0.78–0.82**, GNN consistently outperforms tất cả baselines.

---

# PHASE 2 — Mức xuất sắc (AUC-ROC 0.83–0.87)

> **Triết lý:** Thay đổi kiến trúc và loss function. Áp dụng sau khi Phase 1 đã ổn định và có per-CV evaluation results làm baseline mới.

---

## P2.1. Thay BPR → InfoNCE Loss

**Vấn đề với BPR:** BPR so sánh 1 positive với 1 negative → gradient signal yếu. Với proxy labels có nhiễu, BPR dễ bị confuse bởi near-positive negatives.

**InfoNCE (Noise Contrastive Estimation):** So sánh 1 positive với **N negatives** đồng thời:

```
L_InfoNCE = -log( exp(score_pos / τ) / Σ_j exp(score_j / τ) )

Trong đó:
  τ (temperature) = learnable hoặc fixed (0.07–0.2)
  Σ_j = tổng trên 1 positive + N negatives trong cùng batch
```

**Tại sao InfoNCE tốt hơn BPR:**
1. **Multi-negative gradient:** Mỗi step update dùng gradient từ N negatives thay vì 1 → richer signal
2. **Temperature τ:** Điều chỉnh được độ sharp của distribution → ít nhạy cảm với noisy labels hơn
3. **In-batch negatives:** Reuse các items khác trong batch làm negatives → không cần explicit negative sampling

**Implementation:**

```python
# backend/ml_service/models/losses.py

def infonce_loss(
    pos_scores: torch.Tensor,   # (B,) — scores of positive pairs
    neg_scores: torch.Tensor,   # (B, K) — K negatives per positive
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss.
    
    Args:
        pos_scores: (B,) scores for positive pairs
        neg_scores: (B, K) scores for K negative pairs per positive
        temperature: softmax temperature (lower = sharper, harder task)
    """
    # Concatenate: [pos_score, neg_score_1, ..., neg_score_K]
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # (B, K+1)
    all_scores = all_scores / temperature
    
    # Label: 0 is always the positive (first column)
    labels = torch.zeros(all_scores.size(0), dtype=torch.long, device=all_scores.device)
    
    return torch.nn.functional.cross_entropy(all_scores, labels)
```

**Sampling để dùng với InfoNCE:**

```python
# Trong training loop — sample multiple negatives per positive
K = 8  # Số negatives per positive (thay vì 1 trong BPR)
# Sample 8 negatives: 5 hard + 3 random per CV-positive pair
```

**Kỳ vọng:** +2–4% AUC-ROC. Cải thiện rõ nhất trên per-CV Recall@K.

**Reference:** Oord et al., "Representation Learning with Contrastive Predictive Coding," 2018. [[arxiv:1807.03748]](https://arxiv.org/abs/1807.03748); Mao et al., "Revisiting Recommendation Loss Functions through Contrastive Learning," 2023. [[arxiv:2312.08520]](https://arxiv.org/abs/2312.08520)

---

## P2.2. Test HGT (Heterogeneous Graph Transformer)

**Vấn đề với GraphSAGE + to_hetero():** `to_hetero()` của PyG áp dụng cùng 1 GraphSAGE backbone cho tất cả edge types → không phân biệt semantics của `has_skill` vs `requires_skill` vs `has_seniority`.

**HGT (Hu et al., KDD 2020):** Thiết kế riêng cho heterogeneous graphs. Mỗi edge type có **relation-specific attention weights**:

```
Attention(s, e, t) = softmax( Q(t)·K(s)·W_att(e) ) 
Message(s, e, t) = W_msg(e) · V(s)

Trong đó:
  s = source node type
  e = edge type  
  t = target node type
  W_att, W_msg = learnable matrices per edge type
```

**Khi nào HGT tốt hơn GraphSAGE:**
- Khi có nhiều loại edge với semantics khác nhau (đúng với hệ thống này)
- Khi dataset đủ lớn để train relation-specific weights (cần ≥ 5K pairs/edge type) → **risk với dataset nhỏ**
- Graph lớn và đa dạng (nhiều node types, edge types)

**Risk với dataset nhỏ:** HGT có nhiều parameters hơn → dễ overfit hơn GraphSAGE. Cần combine với DropEdge và stronger regularization.

**Implementation:** PyG đã có `HGTConv`:

```python
from torch_geometric.nn import HGTConv

class HeteroHGT(nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, num_heads=4):
        super().__init__()
        self.projections = nn.ModuleDict(...)  # same as before
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, 
                    heads=num_heads, group='sum')
            for _ in range(num_layers)
        ])
        self.decoder = MLPDecoder(hidden_channels)
    
    def encode(self, data):
        x_dict = {ntype: proj(data[ntype].x) for ntype, proj in self.projections.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        return x_dict
```

**Cách test:** Chạy HGT với cùng config tốt nhất từ Phase 1. So sánh per-CV NDCG@10:
- Nếu HGT > GraphSAGE +2%: switch
- Nếu không: giữ GraphSAGE (simpler = more robust)

**Reference:** Hu et al., "Heterogeneous Graph Transformer," KDD 2020. [[arxiv:2003.01332]](https://arxiv.org/abs/2003.01332)

**Kỳ vọng:** +1–4% AUC-ROC (uncertain — cần A/B test).

---

## P2.3. Nâng cấp Text Embedding

**Hiện tại:** `all-MiniLM-L6-v2` (384 dim, 22M params)

**Vấn đề:** CV texts dài (1000+ tokens), MiniLM được thiết kế cho short sentences. Truncate CV → mất thông tin.

**Lựa chọn:**

| Model | Dim | Params | STS-B | Context | Notes |
|-------|-----|--------|-------|---------|-------|
| all-MiniLM-L6-v2 | 384 | 22M | 84.8 | 512 tokens | Current |
| all-mpnet-base-v2 | 768 | 109M | 87.8 | 512 tokens | Best quality Sentence-BERT |
| nomic-embed-text-v1.5 | 768 | 137M | 88.0+ | 8192 tokens | **Long doc, multilingual** |
| jina-embeddings-v3 | 1024 | 572M | 90.0+ | 8192 tokens | Best, nhưng nặng |

**Recommendation:** `nomic-embed-text-v1.5`
- Context window 8192 tokens → không cắt CV text
- Multilingual — tốt cho Vietnamese CVs
- Free, open-source, chạy local

**Tác động lên architecture:**
- CV/Job node features: 384 + 2 metadata → **768 + 2 = 770 dims**
- Skill node features: 384 + 1 category → **768 + 1 = 769 dims**
- Cần update `node_dims` trong HeteroGraphSAGE

**Kỳ vọng:** +2–3% AUC-ROC từ richer semantic representations.

**Reference:** Nussbaum et al., "Nomic Embed: Training a Reproducible Long Context Text Embedder," arxiv:2402.01613. [[link]](https://arxiv.org/abs/2402.01613)

---

## P2.4. Contrastive Pretraining (GRACE/GraphCL)

**Vấn đề:** Với 9,889 labeled pairs, supervised signal rất sparse. Phần lớn CV-job pairs không có label.

**Kỹ thuật:** Self-supervised pretraining trên unlabeled graph structure, trước khi fine-tune với BPR/InfoNCE.

**GRACE (Graph Contrastive Representation Learning, ICML 2020):**
1. Tạo 2 augmented views của graph bằng DropEdge + Feature Masking
2. Train model để maximize agreement giữa embeddings của cùng node từ 2 views khác nhau
3. Fine-tune với supervised BPR/InfoNCE

```
View 1: G' = DropEdge(G, rate=0.2) + MaskFeature(X, rate=0.1)
View 2: G'' = DropEdge(G, rate=0.3) + MaskFeature(X, rate=0.2)

L_GRACE = -log( exp(sim(h_i, h_i') / τ) / Σ_j exp(sim(h_i, h_j') / τ) )

Trong đó h_i, h_i' là embeddings của node i từ 2 views
```

**Protocol:**
1. Pretraining: 50–100 epochs với L_GRACE (unsupervised)
2. Fine-tuning: 200–300 epochs với L_InfoNCE (supervised)

**Kỳ vọng:** +3–5% AUC-ROC từ richer initialization, đặc biệt cải thiện cold-start (CVs/jobs chưa nhiều labeled pairs).

**Complexity:** Cao nhất trong tất cả techniques. Implement sau khi P2.1–P2.3 ổn định.

**Reference:** Zhu et al., "Deep Graph Contrastive Representation Learning," ICML 2020 workshop. [[arxiv:2006.04131]](https://arxiv.org/abs/2006.04131)

---

## Tóm tắt Phase 2

| Kỹ thuật | Độ khó | Thời gian | Kỳ vọng |
|----------|--------|-----------|---------|
| P2.1 InfoNCE loss | Trung bình | 1 ngày | +2–4% |
| P2.2 HGT model | Trung bình | 1–2 ngày | +1–4% (uncertain) |
| P2.3 Nomic embeddings | Thấp | 2 giờ + rerun | +2–3% |
| P2.4 GRACE pretraining | Cao | 3–4 ngày | +3–5% |
| **Tổng cộng** | | **1–2 tuần** | **+8–16%** |

**Target Phase 2:** AUC-ROC **0.83–0.87** (nếu ít nhất 3/4 kỹ thuật hoạt động tốt).

---

# Thứ tự thực hiện khuyến nghị

```
Week 9 (hiện tại):
  1. [x] Fix data pipeline (511 CVs, skill catalog fixed)
  2. [ ] P1.3 DropEdge  ← 2 giờ, impact cao
  3. [ ] P1.4 Curriculum negatives  ← 2 giờ, impact cao
  4. [ ] P1.1 Tăng NUM_POSITIVE_PAIRS=3500  ← 30 phút
  5. [ ] Chạy experiment → có baseline mới

Week 10:
  6. [ ] P1.2 Hyperparameter grid search  ← song song
  7. [ ] P1.5 Skill co-occurrence edges  ← cần data engineering
  8. [ ] Per-CV evaluation với best config

Week 11:
  9. [ ] P2.3 Upgrade Nomic embeddings  ← nếu thời gian cho phép
  10. [ ] P2.1 InfoNCE loss  ← nếu thời gian cho phép
  11. [ ] P2.2 HGT A/B test  ← optional

Week 12:
  12. [ ] P2.4 GRACE pretraining  ← chỉ nếu còn thời gian
  13. [ ] Final benchmark, phân tích kết quả, viết report
```

---

# Metrics kỳ vọng theo từng phase

## Global Evaluation (có thể đạt)

| Phase | AUC-ROC | NDCG@10 | Recall@10 | Precision@5 |
|-------|---------|---------|----------|-------------|
| Baseline (hiện tại) | 0.71 | 0.78 | 0.039 | 0.80 |
| Phase 1 (tốt) | **0.78–0.82** | **0.85+** | 0.05–0.08 | **0.90+** |
| Phase 2 (xuất sắc) | **0.83–0.87** | **0.90+** | 0.08–0.12 | **0.95+** |

## Per-CV Full-Ranking (realistic lower bound)

| Phase | Recall@50 | Recall@100 | NDCG@10 | MRR |
|-------|----------|-----------|---------|-----|
| Baseline | ~0.05–0.08 | ~0.10–0.15 | ~0.15–0.25 | ~0.10–0.15 |
| Phase 1 | **0.10–0.15** | **0.20–0.25** | **0.25–0.35** | **0.15–0.25** |
| Phase 2 | **0.15–0.25** | **0.25–0.35** | **0.35–0.45** | **0.20–0.35** |

> **Lưu ý:** Per-CV metrics thấp hơn global là bình thường và đã được giải thích trong literature (Krichene & Rendle, KDD 2020). Điều quan trọng là GNN phải consistently better hơn baselines ở cả 2 protocol.

---

# Rủi ro và mitigation

| Rủi ro | Xác suất | Impact | Mitigation |
|--------|----------|--------|-----------|
| DropEdge làm training không ổn định | Thấp | Trung bình | Giảm drop rate về 0.1 |
| HGT không improve với dataset nhỏ | Cao | Thấp | Giữ GraphSAGE, không switch |
| InfoNCE sensitive với temperature τ | Trung bình | Trung bình | Grid search τ ∈ {0.05, 0.1, 0.2} |
| Nomic embedding chậm hơn MiniLM | Trung bình | Thấp | Cache embeddings, chỉ compute 1 lần |
| Skill co-occurrence có noisy edges | Trung bình | Trung bình | Tăng min_pmi threshold lên 0.8 |
| AUC-ROC vẫn không đạt 0.83 | Trung bình | Trung bình | Focus vào relative improvement GNN > baselines, không phải absolute numbers |

---

# References

1. **DropEdge:** Rong et al., "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification," ICLR 2020. [arxiv:1907.10903](https://arxiv.org/abs/1907.10903)

2. **InfoNCE / Contrastive Loss:** Oord et al., "Representation Learning with Contrastive Predictive Coding," 2018. [arxiv:1807.03748](https://arxiv.org/abs/1807.03748)

3. **Revisiting Recommendation Loss Functions:** Mao et al., 2023. [arxiv:2312.08520](https://arxiv.org/abs/2312.08520)

4. **HGT:** Hu et al., "Heterogeneous Graph Transformer," KDD 2020. [arxiv:2003.01332](https://arxiv.org/abs/2003.01332)

5. **GRACE Contrastive Pretraining:** Zhu et al., "Deep Graph Contrastive Representation Learning," ICML 2020 workshop. [arxiv:2006.04131](https://arxiv.org/abs/2006.04131)

6. **Curriculum Negatives:** Cascianelli et al., "CuCo: Graph Representation with Curriculum Contrastive Learning," IJCAI 2021. [link](https://www.ijcai.org/proceedings/2021/0317.pdf)

7. **Skill Graph / Knowledge Graph:** Nguyen et al., "Job Posting-Enriched Knowledge Graph for Skills-based Matching," arxiv:2109.02554. [link](https://arxiv.org/abs/2109.02554)

8. **Nomic Embeddings:** Nussbaum et al., "Nomic Embed: Training a Reproducible Long Context Text Embedder," arxiv:2402.01613. [link](https://arxiv.org/abs/2402.01613)

9. **Negative Sampling in GNN:** Yang et al., "Understanding Negative Sampling in GNN," KDD 2020. [link](https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD20-Yang-et-al-Understanding_Negative_Sampling_in_GNN.pdf)

10. **LinkSAGE (LinkedIn production):** Liu et al., "LinkSAGE: Optimizing Job Matching Using Graph Neural Networks," KDD 2025. [arxiv:2402.13430](https://arxiv.org/abs/2402.13430)

11. **Krichene & Rendle (sampled metrics):** Krichene & Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020. DOI: [10.1145/3394486.3403226](https://doi.org/10.1145/3394486.3403226)
