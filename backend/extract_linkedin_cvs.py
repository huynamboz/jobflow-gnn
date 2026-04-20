"""
Extract LinkedIn CV PDFs → JSON cache.

Usage:
    cd backend
    .venv/bin/python extract_linkedin_cvs.py

Output: data/linkedin_cvs.json

Also prints a verification report for a sample CV from each category.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("extract_linkedin_cvs")

DATASET_DIR = Path("/Users/huynam/Documents/PROJECT/jobflow-gnn/Dataset")
OUTPUT_PATH = Path("data/linkedin_cvs.json")
SKILL_ALIAS_PATH = "ml_service/data/skill-alias.json"

CATEGORIES = ["AI", "Devops", "Software Engineer", "Tester", "Business Analyst", "UX_UI"]


def main() -> None:
    t0 = time.time()

    from ml_service.cv_parser import CVParser
    from ml_service.data.skill_normalization import SkillNormalizer

    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    parser = CVParser(normalizer)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_cvs: list[dict] = []
    per_category: dict[str, list[dict]] = {}
    skipped = 0
    cv_id = 0

    for category in CATEGORIES:
        category_dir = DATASET_DIR / category
        if not category_dir.exists():
            logger.warning("Category dir not found: %s", category_dir)
            continue

        pdfs = sorted(category_dir.glob("*.pdf"))
        logger.info("Processing %s: %d PDFs", category, len(pdfs))
        cat_cvs: list[dict] = []

        for pdf_path in pdfs:
            try:
                cv = parser.parse_file(str(pdf_path), cv_id=cv_id)

                if len(cv.skills) < 2:
                    skipped += 1
                    continue

                record = {
                    "cv_id": cv_id,
                    "source_file": pdf_path.name,
                    "category": category,
                    "seniority": cv.seniority.value,
                    "experience_years": cv.experience_years,
                    "education": cv.education.value,
                    "skills": list(cv.skills),
                    "skill_proficiencies": list(cv.skill_proficiencies),
                    "text": cv.text,
                }
                all_cvs.append(record)
                cat_cvs.append(record)
                cv_id += 1

            except Exception as e:
                logger.debug("Failed %s: %s", pdf_path.name, e)
                skipped += 1

        per_category[category] = cat_cvs
        logger.info("  → %d CVs extracted", len(cat_cvs))

    # Save JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_cvs, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    logger.info("Saved %d CVs to %s (skipped %d, %.1fs)", len(all_cvs), OUTPUT_PATH, skipped, elapsed)

    # ── Verification report: 1 sample per category ────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION REPORT — 1 sample per category")
    print("=" * 70)

    for category, cvs in per_category.items():
        if not cvs:
            print(f"\n[{category}] — no CVs extracted")
            continue

        sample = cvs[0]
        print(f"\n[{category}] {sample['source_file']}  (cv_id={sample['cv_id']})")
        print(f"  Skills ({len(sample['skills'])}): {sample['skills'][:10]}")
        print(f"  Experience : {sample['experience_years']} years")
        print(f"  Education  : {sample['education']}")
        print(f"  Seniority  : {sample['seniority']}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total CVs   : {len(all_cvs)}")
    print(f"Skipped     : {skipped}")
    print(f"Time        : {elapsed:.1f}s")
    print()

    from collections import Counter
    sen_dist = Counter(c["seniority"] for c in all_cvs)
    edu_dist = Counter(c["education"] for c in all_cvs)
    exp_vals = [c["experience_years"] for c in all_cvs]
    skill_counts = [len(c["skills"]) for c in all_cvs]

    print("Seniority distribution:")
    for k, v in sorted(sen_dist.items()):
        print(f"  {k:<20} {v}")

    print("\nEducation distribution:")
    for k, v in sorted(edu_dist.items()):
        print(f"  {k:<20} {v}")

    print(f"\nExperience years:")
    print(f"  avg={sum(exp_vals)/len(exp_vals):.1f}  min={min(exp_vals):.1f}  max={max(exp_vals):.1f}")
    print(f"  zero-exp CVs: {sum(1 for e in exp_vals if e == 0.0)}")

    print(f"\nSkills per CV:")
    print(f"  avg={sum(skill_counts)/len(skill_counts):.1f}  min={min(skill_counts)}  max={max(skill_counts)}")
    print(f"  CVs with < 5 skills: {sum(1 for s in skill_counts if s < 5)}")

    print(f"\nOutput: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
