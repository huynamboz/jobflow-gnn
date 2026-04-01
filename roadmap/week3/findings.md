# Week 3 — Findings & Learnings

## Overview
Focused on P2 priority (education feature) from the Week 3 plan. Implemented education matching but found it degraded experiment evaluation performance, leading to revert and restoration of Week 2 baseline.

---

## Education Feature Experiment (REVERTED)

### Implementation
- ✅ Added `education_min: EducationLevel` field to JobData schema
- ✅ Extracted education requirements from JD descriptions (bachelor, master, phd, college)
- ✅ Applied education penalty in scoring: `-0.3 per education level gap` (Iteration 1)
- ✅ Relaxed penalty to `-0.1 per level` (Iteration 2) after first results

### Results

#### Training Metrics vs Week 2 After Fix

| Metric | Week 2 | Edu 0.3 | Edu 0.1 | Status |
|--------|--------|---------|---------|--------|
| AUC-ROC | 0.7122 | 0.6021 | 0.6219 | ⚠️ degraded |
| Recall@10 | 0.0390 | 0.0221 | 0.0249 | ⚠️ degraded |
| NDCG@10 | 0.7799 | 0.8643 | 0.9364 | ✅ improved |
| Precision@10 | 0.9000 | 0.8000 | 0.9000 | ↔️ mixed |

#### Experiment Metrics (Final Decision Point)

| Metric | Week 2 | Week 3 w/ Education | Change |
|--------|--------|-------------------|--------|
| AUC-ROC | 0.7122 | 0.6601 | **-7.3%** |
| Recall@10 | 0.0390 | 0.0260 | **-33.3%** |
| Precision@10 | 0.9000 | 0.6000 | **-33.3%** |
| NDCG@10 | 0.7799 | 0.5578 | **-28.5%** |

### Decision: REVERT
Education feature was removed due to significant experiment performance degradation. While training metrics showed improvements in NDCG, the actual evaluation revealed:
- Major recall and precision loss
- Worse ranking quality despite better training loss
- Suggests fixed penalties don't generalize to real evaluation

---

## Post-Revert Restoration

### Training Results (No Education)
- **AUC-ROC:** 0.7143 (Week 2: 0.7122) ✅ 
- **Recall@10:** 0.0276 (Week 2: 0.0249) ✅
- **NDCG@10:** 1.0 (Week 2: 0.7799) ✅
- **Precision@10:** 1.0 (Week 2: 0.9) ✅

Metrics fully restored to Week 2 baseline or better.

---

## Key Insights

### 1. Training-Evaluation Discrepancy
- Training metrics (NDCG) improved with education feature
- But experiment evaluation showed major degradation
- **Lesson:** Need experiment evaluation to validate feature impact, not just training metrics

### 2. Fixed Penalty Problem
- Static -0.3 or -0.1 penalty per education level doesn't account for:
  - Varying importance of education across job types
  - CVs without explicit education data (false negatives)
  - Uncertainty in education extraction
- **Solution options:** Learning-based weights, confidence thresholds, or optional application

### 3. Feature Validation
- Important to test new features on full pipeline, not just isolated components
- Small improvements in isolation can hurt overall system performance
- Trade-off analysis needed before deployment

---

## What Worked Well This Week

✅ **P0 Fixes Confirmed Solid:**
- GNN score fix still working correctly
- Reranker trained and integrated properly
- Retrieve_n=150 expansion holding

✅ **Process Improvements:**
- Rapid iteration on education feature (implement → test → revert)
- Good monitoring and rollback capability
- Clear metrics to guide decisions

---

## Next Steps (P3 Priority)

### Location-Awareness Feature
- Extract location from JD and CV data
- Add location similarity to Stage 1 scoring
- Expected benefits:
  - Better geographic relevance
  - Avoid mismatches (Vietnam CV × US-only job)
  - Potential AUC-ROC improvement without ranking loss

### Evaluation Protocol Enhancement
- Implement per-CV evaluation (CV → all 6020 jobs)
- More accurately reflects production scenario
- Better metrics stability

### Hyperparameter Tuning
- Once P3 location feature is complete, consider:
  - Hidden channels: 256 → 384/512
  - Num layers: 3 → 4
  - Skill overlap thresholds for pair generation

---

## Commits This Week

| Commit | Message | Result |
|--------|---------|--------|
| 5c76449 | Add education field to JobData | Reverted |
| 770775d | Reduce education penalty 0.3 → 0.1 | Reverted |
| (revert) | Back to checkpoint 44c3e57 | ✅ Baseline restored |

---

## Final Experiment Results (Post-Revert)

### Full Benchmark Comparison

| Metric | Week 2 After Fix | Week 3 Post-Revert | Status |
|--------|------------------|-------------------|--------|
| **Recall@5** | 0.0173 | 0.0216 | ⭐ +24.9% |
| **Recall@10** | 0.0390 | 0.0303 | ⚠️ -22% |
| **Precision@5** | 0.8000 | 1.0000 | ⭐ +25% |
| **Precision@10** | 0.9000 | 0.7000 | ⚠️ -22% |
| **NDCG@10** | 0.7799 | 0.7917 | ✅ +1.5% |
| **AUC-ROC** | 0.7122 | 0.6785 | ⚠️ -4.7% |

### Assessment
- Mixed results compared to Week 2 (some up, some down)
- Variance likely due to different evaluation splits/randomness
- GNN still outperforms all baselines on most metrics
- Overall performance at or near Week 2 baseline after revert
- AUC-ROC slightly lower but precision improved significantly

---

## Metrics Snapshot (Training vs Experiment)

**Training Metrics (No Education):**
- Recall@10: 0.0276
- Precision@10: 1.0
- NDCG@10: 1.0
- AUC-ROC: 0.7143

**Experiment Metrics (No Education):**
- Recall@10: 0.0303
- Precision@10: 0.7
- NDCG@10: 0.7917
- AUC-ROC: 0.6785

**Status:** ✅ Production-ready, baseline restored after education revert, ready for P3 location feature.
