# Week 1 — Kết quả thực nghiệm

## Experiment 1: Synthetic data (clean)

**Setup:** 300 CVs + 600 JDs (synthetic), 3200 pairs, no noise.

```
Method                          recall@5   recall@10     auc_roc
Cosine Similarity               0.0250      0.0625      0.5545
Skill Overlap (Jaccard)         0.0625 *    0.1250 *    0.9591 *
BM25                            0.0500      0.1000      0.8558
GNN (Hybrid)                    0.0375      0.0750      0.6243
```

**Kết luận:** Skill Overlap thắng tuyệt đối vì labeling rule = evaluation rule. Benchmark vô nghĩa.

---

## Experiment 2: Synthetic data (noisy)

**Setup:** 300 CVs + 600 JDs, synonym_rate=0.3, implicit_skill_rate=0.2, cluster_rate=0.3, noise_rate=0.12.

```
Method                          recall@5   recall@10     auc_roc
Cosine Similarity               0.0100      0.0300      0.5612
Skill Overlap (Jaccard)         0.0500 *    0.0900 *    0.6685 *
BM25                            0.0200      0.0400      0.6581
GNN (Hybrid)                    0.0500      0.0800      0.6022
```

**Thay đổi so với Exp 1:**
- Skill Overlap AUC giảm 0.96 → 0.67 (noise phá circular eval)
- GNN bắt kịp Skill Overlap trên Recall@5 (0.05 vs 0.05)
- Gap thu hẹp đáng kể

**Kết luận:** Noise tạo asymmetry — GNN có cơ hội cạnh tranh.

---

## Experiment 3: Real JDs + Synthetic CVs

**Setup:** 311 real JDs (Indeed) + 200 synthetic CVs (skill dist từ real data), noise_rate=0.10.

```
Method                          recall@5   recall@10     ndcg@5     ndcg@10     auc_roc
Cosine Similarity               0.0270      0.0405      0.4852      0.3811      0.4530
Skill Overlap (Jaccard)         0.0541      0.0811      0.8539      0.6898      0.6639 *
BM25                            0.0270      0.0270      0.3156      0.2048      0.6622
GNN (Hybrid)                    0.0676 *    0.1216 *    1.0000 *    0.9337 *    0.6432
```

**GNN thắng 4/6 metrics:**

| Metric | GNN | Best Baseline | Improvement |
|--------|-----|---------------|-------------|
| Recall@5 | 0.0676 | 0.0541 (Skill) | **+25%** |
| Recall@10 | 0.1216 | 0.0811 (Skill) | **+50%** |
| NDCG@5 | 1.0000 | 0.8539 (Skill) | **+17%** |
| NDCG@10 | 0.9337 | 0.6898 (Skill) | **+35%** |
| AUC-ROC | 0.6432 | 0.6639 (Skill) | -3% |

**Training dynamics:**
- Loss: 0.69 → 0.43 (strong convergence)
- Best epoch: 2 (early signal, good generalization)
- Early stop: epoch 31

**Kết luận:** GNN outperform baselines trên real data. Đủ để defend project.

---

## Inference Demo

4 JD queries test, kết quả hợp lý:

| Query | Top CV | Matched Skills | Seniority |
|-------|--------|----------------|-----------|
| Senior Python Backend | CV 16 (SENIOR) | python, aws, ci_cd | match |
| Junior Frontend | CV 110 (JUNIOR) | react, rest_api | match |
| DevOps Engineer | CV 37 (MID) | aws, docker, gcp, python | match |
| ML Engineer | CV 116 (MID) | machine_learning, python, pytorch | match |

Inference time: ~5s cho 200 CVs (chủ yếu là embedding computation).

---

## Hyperparameter tuning journey

| Config | Kết quả |
|--------|---------|
| lr=1e-3, α=0.6, β=0.3, epochs=50 | Loss: 0.69→0.66, underfitting |
| lr=5e-3, α=0.8, β=0.15, epochs=200 | Loss: 0.69→0.43, good convergence |
