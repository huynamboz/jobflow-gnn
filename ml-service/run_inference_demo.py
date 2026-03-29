"""
Demo inference — match a JD against the CV pool.

Usage:
    cd ml-service
    python run_train_save.py      # train + save checkpoint (run once)
    python run_inference_demo.py  # run inference
"""

from __future__ import annotations

import logging

from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("inference_demo")

CHECKPOINT_DIR = "checkpoints/latest"
SKILL_ALIAS_PATH = "../roadmap/week1/skill-alias.json"

# Example JD queries to test
DEMO_QUERIES = [
    {
        "name": "Senior Python Backend",
        "text": (
            "We are hiring a senior backend engineer with strong Python experience. "
            "Must have expertise in Django or FastAPI, PostgreSQL, Redis, and Docker. "
            "Experience with AWS and CI/CD pipelines is required. "
            "Salary: $5000-$8000/month."
        ),
    },
    {
        "name": "Junior Frontend Developer",
        "text": (
            "Looking for a junior frontend developer proficient in React and TypeScript. "
            "Should know HTML/CSS, Git, and have basic understanding of REST APIs. "
            "Fresh graduates welcome."
        ),
    },
    {
        "name": "DevOps Engineer",
        "text": (
            "Seeking a mid-level DevOps engineer. Required: Kubernetes, Docker, Terraform, "
            "AWS or GCP, CI/CD, Linux administration. "
            "Nice to have: monitoring, security, Python scripting."
        ),
    },
    {
        "name": "ML Engineer",
        "text": (
            "Machine learning engineer needed. Must be proficient in Python, PyTorch, "
            "and have experience with NLP or computer vision. "
            "Knowledge of data engineering, SQL, and cloud platforms is a plus."
        ),
    },
]


def main() -> None:
    logger.info("Loading model from checkpoint: %s", CHECKPOINT_DIR)
    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    provider = get_provider()

    engine = InferenceEngine.from_checkpoint(
        CHECKPOINT_DIR,
        normalizer=normalizer,
        embedding_provider=provider,
    )
    logger.info("Engine ready: %d CVs in pool", engine.num_cvs)

    for query in DEMO_QUERIES:
        print(f"\n{'='*70}")
        print(f"  JD: {query['name']}")
        print(f"{'='*70}")

        results = engine.match(query["text"], top_k=5)

        if not results:
            print("  No matches found.")
            continue

        for i, r in enumerate(results, 1):
            cv = engine.cv_pool[r.cv_id]
            status = "ELIGIBLE" if r.eligible else "not eligible"
            sen = "yes" if r.seniority_match else "no"
            print(f"\n  #{i} CV {r.cv_id} — score: {r.score:.4f} [{status}]")
            print(f"     Seniority: {cv.seniority.name} (match: {sen})")
            print(f"     CV skills: {', '.join(cv.skills[:10])}")
            print(f"     Matched:   {', '.join(r.matched_skills) or '(none)'}")
            print(f"     Missing:   {', '.join(r.missing_skills) or '(none)'}")


if __name__ == "__main__":
    main()
