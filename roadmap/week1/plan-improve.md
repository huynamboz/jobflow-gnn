# Plan Improve — Nâng chất lượng GNN Matching

## Mục tiêu: AUC-ROC > 0.75, Recall@10 > 0.05

---

## Improve 1: Mở rộng Skill Dictionary (HIGH IMPACT, ~1h)

**Vấn đề:** 85 canonical skills quá ít. Real JDs có hàng trăm skills mà normalizer không nhận ra → mất thông tin → graph thiếu edges.

**Cách làm:**
- Scan toàn bộ 4.854 JDs, đếm từ/cụm từ xuất hiện nhiều mà chưa normalize được
- Thêm vào skill-alias.json: target 200-300 canonical skills
- Focus nhóm thiếu: DevOps tools (Ansible, Prometheus, Grafana, Nginx, HAProxy), Data (Snowflake, dbt, Databricks, Looker), Cloud services (Lambda, S3, EC2, CloudFormation), Frameworks (NestJS, Svelte, Remix, Nuxt)

**Expected impact:** Tăng avg skills/JD từ 10.8 → 15+, graph denser → GNN message passing hiệu quả hơn.

---

## Improve 2: Fix false positive "c" + short tokens (HIGH IMPACT, ~30m)

**Vấn đề:** Skill "c" (C language) match nhầm chữ "C" đơn lẻ trong text thường. 121/315 JDs ban đầu bị ảnh hưởng.

**Cách làm:**
- Thêm min-length filter: skip tokens có length <= 1
- Exception list cho "c", "r" — chỉ match khi đi kèm context: "C programming", "C/C++", "C language", "R programming", "R Studio"
- Hoặc đổi canonical name "c" → "c_lang", "r" → "r_lang" và chỉ match bigrams

**Expected impact:** Giảm noise trong graph → GNN không bị confuse bởi fake edges.

---

## Improve 3: Enrich CV text (MEDIUM IMPACT, ~30m)

**Vấn đề:** Kaggle CVs text = "Python Developer. Skills: python, django, redis" — quá template. Embedding gần giống nhau → GNN text signal yếu.

**Cách làm:**
- `resume_loader.py`: dùng FULL text từ dataset — summary + ALL responsibilities + projects
- Concat: `{summary}. {title1}: {resp1}, {resp2}. {title2}: {resp1}. Skills: {skills}`
- Truncate ở 512 tokens (MiniLM limit)

**Expected impact:** CV embeddings diverse hơn → GNN phân biệt tốt hơn giữa các CVs.

---

## Improve 4: Thêm CV→CV similar_profile edges (MEDIUM IMPACT, ~30m)

**Vấn đề:** Graph chỉ có Job→Job similarity. CVs với profile tương tự không connected → GNN thiếu collaborative signal.

**Cách làm:**
- Trong `skill_graph.py`: thêm `build_cv_similarity_edges(cvs, top_k=5, min_overlap=0.3)`
- Logic giống `build_job_similarity_edges` — Jaccard trên skill sets
- Trong `builder.py`: thêm `("cv", "similar_profile", "cv")` edge type

**Expected impact:** GNN propagate thông tin giữa similar CVs → nếu CV_A match Job_X, thì CV_B (similar to CV_A) cũng có signal.

---

## Improve 5: Normalize salary + experience features tốt hơn (LOW-MEDIUM IMPACT, ~20m)

**Vấn đề:** `salary_min/max` normalize bằng constant 10,000. `experience_years` raw float. Feature scale không đồng đều.

**Cách làm:**
- Z-score normalization cho experience_years (mean, std từ dataset)
- Percentile-based normalization cho salary
- Hoặc đơn giản: min-max normalize tất cả numeric features

**Expected impact:** GNN train ổn định hơn, gradient flow tốt hơn.

---

## Thứ tự triển khai

```
Step 1: Fix false positive "c" + short tokens           (30m)
Step 2: Mở rộng skill dictionary 85 → 250+              (1h)
Step 3: Enrich CV text (summary + responsibilities)      (30m)
Step 4: Thêm CV→CV similar_profile edges                 (30m)
Step 5: Normalize features tốt hơn                       (20m)
Step 6: Re-run experiment → benchmark                    (10m)
```

Tổng: ~3h. Chạy experiment sau mỗi step để đo impact.
