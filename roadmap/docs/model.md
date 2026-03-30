# Model — HeteroGraphSAGE

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    HeteroGraphSAGE                      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Step 1: Per-type Linear Projection              │   │
│  │                                                  │   │
│  │  CV      (386) ──► Linear ──► (128)              │   │
│  │  Job     (386) ──► Linear ──► (128)              │   │
│  │  Skill   (385) ──► Linear ──► (128)              │   │
│  │  Seniority (6) ──► Linear ──► (128)              │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Step 2: GraphSAGE (to_hetero)                   │   │
│  │                                                  │   │
│  │  2 layers × SAGEConv (mean aggregation)          │   │
│  │  Message passing qua tất cả edge types           │   │
│  │  (bao gồm reverse edges từ ToUndirected)         │   │
│  │                                                  │   │
│  │  Output: z_dict = {                              │   │
│  │    "cv":        [N_cv, 128],                     │   │
│  │    "job":       [N_job, 128],                    │   │
│  │    "skill":     [N_skill, 128],                  │   │
│  │    "seniority": [6, 128],                        │   │
│  │  }                                               │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Step 3: MLPDecoder                              │   │
│  │                                                  │   │
│  │  concat(z_cv, z_job) ──► (256)                   │   │
│  │  Linear(256→128) ──► ReLU                        │   │
│  │  Linear(128→1)                                   │   │
│  │  Output: scalar score per (cv, job) pair         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Per-type Linear Projection

Mỗi node type có input dim khác nhau, cần project về cùng `hidden_channels`:

| Node Type | Input Dim | Cấu thành | Output Dim |
|-----------|-----------|-----------|------------|
| CV | 386 | embedding(384) + experience_years + education_level | 128 |
| Job | 386 | embedding(384) + salary_min_norm + salary_max_norm | 128 |
| Skill | 385 | embedding(384) + category | 128 |
| Seniority | 6 | one-hot (6 levels) | 128 |

```python
self.projections = nn.ModuleDict({
    "cv":        nn.Linear(386, 128),
    "job":       nn.Linear(386, 128),
    "skill":     nn.Linear(385, 128),
    "seniority": nn.Linear(6, 128),
})
```

### 2. GraphSAGE Backbone

Dùng PyG `GraphSAGE` + `to_hetero()` để wrap cho heterogeneous graph.

```python
backbone = GraphSAGE(
    in_channels=128,
    hidden_channels=128,
    num_layers=2,
    out_channels=128,
)
self.gnn = to_hetero(backbone, metadata, aggr="mean")
```

**Tại sao GraphSAGE?**
- **Inductive learning**: CV/JD mới không cần retrain toàn graph
- **Scalable**: LinkedIn dùng GraphSAGE ở production (LinkSAGE, 1B+ nodes)
- **Mean aggregation**: đơn giản, ổn định, phù hợp MVP

**Message passing:**
- Mỗi layer: node nhận message từ neighbors, aggregate bằng mean, update embedding
- 2 layers = mỗi node "nhìn thấy" 2-hop neighbors
- Ví dụ: CV → Skill → Job (CV gián tiếp nhận thông tin từ Job qua Skill chung)

**Reverse edges:**
- `ToUndirected()` thêm reverse edge cho mỗi edge type
- CV ← has_skill ← Skill (skill cũng nhận thông tin từ các CV sử dụng nó)
- Đảm bảo mọi node type là message destination

### 3. MLPDecoder

Scoring head cho cặp (CV, Job):

```python
class MLPDecoder(nn.Module):
    def forward(self, z_cv, z_job):
        z = torch.cat([z_cv, z_job], dim=-1)  # [B, 256]
        z = torch.relu(self.lin1(z))           # [B, 128]
        return self.lin2(z).squeeze(-1)        # [B]
```

---

## Encode / Decode Split

Thiết kế tách `encode()` và `decode()` phục vụ inference:

```python
# Training: end-to-end
scores = model(data, cv_indices, job_indices)

# Inference: precompute once, decode many
z_dict = model.encode(data)           # chạy 1 lần
scores = model.decode(z_dict, cv_i, job_i)  # chạy N lần (nhanh)
```

**Tại sao tách?**
- `encode()` chạy GNN qua toàn graph — tốn compute
- `decode()` chỉ lookup embedding + MLP forward — rất nhanh
- Inference: encode 1 lần, decode cho mọi cặp (cv, job)

---

## Loss Function — BPR (Bayesian Personalized Ranking)

```python
def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()
```

**Tại sao BPR thay vì BCE?**
- BCE tối ưu classification (match/no_match riêng lẻ)
- BPR tối ưu **ranking** trực tiếp: positive pair phải score cao hơn negative pair
- Phù hợp hơn cho bài toán retrieval (trả về Top K)

**BPR triplet sampling:**
```
Với mỗi positive pair (cv, job_pos):
  → sample 1 negative job_neg (từ explicit negatives hoặc random)
  → loss = -log(σ(score(cv,job_pos) - score(cv,job_neg)))
```

---

## Training Pipeline

```python
trainer = Trainer(TrainConfig(
    hidden_channels=128,
    num_layers=2,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=50,
    patience=10,
    hybrid_alpha=0.6,
    hybrid_beta=0.3,
    hybrid_gamma=0.1,
))
result = trainer.train(data, dataset, cvs, jobs)
```

### Training loop (mỗi epoch)

```
1. Strip label edges (match/no_match) khỏi graph
   → tránh label leakage trong message passing

2. Sample BPR triplets từ training pairs
   → (cv_idx, pos_job_idx, neg_job_idx)

3. Forward pass:
   z_dict = model.encode(data_clean)
   pos_scores = model.decode(z_dict, cv_idx, pos_job_idx)
   neg_scores = model.decode(z_dict, cv_idx, neg_job_idx)
   loss = bpr_loss(pos_scores, neg_scores)

4. Backward + optimizer step

5. Validate bằng hybrid scoring trên val set
   → track val_mrr cho early stopping

6. Early stopping nếu val_mrr không improve sau `patience` epochs
```

### Label leakage prevention

```python
def _strip_label_edges(data):
    """Remove match/no_match edges trước khi GNN message passing."""
    del data[("cv", "match", "job")]
    del data[("cv", "no_match", "job")]
    return data
```

Nếu không strip → GNN "thấy" trực tiếp CV nào match Job nào → cheat.

### Early stopping

- Monitor: **val_mrr** (Mean Reciprocal Rank trên validation set)
- Patience: 10 epochs
- Restore best model state khi training kết thúc

---

## Hybrid Scoring (Inference)

GNN score alone không đủ tin cậy. Kết hợp 3 signals:

```
final_score = α × GNN_score + β × skill_overlap + γ × seniority_match
```

| Component | Weight | Cách tính |
|-----------|--------|-----------|
| GNN_score | α = 0.6 | MLPDecoder output, min-max normalized to [0,1] |
| skill_overlap | β = 0.3 | Jaccard similarity trên skill sets |
| seniority_match | γ = 0.1 | 1.0 nếu match, giảm 0.25/level chênh lệch |

```python
# Seniority match scoring
def _seniority_match_score(cv, job):
    dist = abs(cv.seniority - job.seniority)
    return max(0.0, 1.0 - dist * 0.25)
    # match → 1.0, 1 level → 0.75, 2 → 0.5, 3 → 0.25, 4+ → 0.0
```

### Eligibility threshold

```python
eligible = final_score >= 0.65  # configurable
```

---

## Hyperparameters

| Parameter | Default | Env var | Mô tả |
|-----------|---------|---------|-------|
| `hidden_channels` | 128 | `GNN_HIDDEN_CHANNELS` | GNN hidden dimension |
| `num_layers` | 2 | `GNN_NUM_LAYERS` | Số GraphSAGE layers (2-hop) |
| `lr` | 1e-3 | `LEARNING_RATE` | Adam learning rate |
| `weight_decay` | 1e-5 | `WEIGHT_DECAY` | L2 regularization |
| `epochs` | 50 | `NUM_EPOCHS` | Max training epochs |
| `patience` | 10 | `EARLY_STOPPING_PATIENCE` | Early stopping patience |
| `hybrid_alpha` | 0.6 | `HYBRID_ALPHA` | GNN score weight |
| `hybrid_beta` | 0.3 | `HYBRID_BETA` | Skill overlap weight |
| `hybrid_gamma` | 0.1 | `HYBRID_GAMMA` | Seniority match weight |
| `threshold` | 0.65 | `ELIGIBILITY_THRESHOLD` | Eligibility cutoff |

---

## Files

| File | Lines | Mô tả |
|------|-------|-------|
| `models/gnn.py` | 101 | HeteroGraphSAGE + MLPDecoder + prepare_data_for_gnn |
| `models/losses.py` | 13 | BPR loss |
| `training/trainer.py` | 263 | Training loop, BPR sampling, hybrid eval, early stopping |
| `evaluation/metrics.py` | 75 | Recall@K, MRR, NDCG@K, AUC-ROC |
| `baselines/cosine.py` | 43 | Cosine similarity scorer |
| `baselines/skill_overlap.py` | 17 | Jaccard skill overlap scorer |
| `baselines/bm25.py` | 78 | Okapi BM25 scorer |
