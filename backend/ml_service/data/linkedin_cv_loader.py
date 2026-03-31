"""Load CV dataset from LinkedIn profile PDFs organized by role category.

Dataset structure:
    Dataset/
    ├── AI/              114 PDFs
    ├── Business Analyst/ 76 PDFs
    ├── Devops/          100 PDFs
    ├── Software Engineer/101 PDFs
    ├── Tester/          121 PDFs
    └── ...

Each PDF is a LinkedIn profile export → parse to CVData.
Category folder name = role label.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ml_service.cv_parser import CVParser
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import CVData

logger = logging.getLogger(__name__)

# Map folder names to IT-relevant categories (skip non-IT if needed)
_CATEGORY_MAP = {
    "AI": "ml",
    "Business Analyst": "data",
    "Devops": "devops",
    "HR": "other",
    "Project Manager": "other",
    "Software Engineer": "backend",
    "Tester": "backend",
    "UX_UI": "frontend",
}


def load_linkedin_cvs(
    dataset_dir: str | Path,
    normalizer: SkillNormalizer | None = None,
    *,
    min_skills: int = 2,
    max_cvs: int | None = None,
    categories: list[str] | None = None,
) -> list[CVData]:
    """Load LinkedIn profile PDFs from dataset directory.

    Args:
        dataset_dir: Path to Dataset/ folder with category subfolders
        min_skills: Skip CVs with fewer than this many extracted skills
        max_cvs: Cap total CVs loaded
        categories: Only load these categories (folder names). None = all.

    Returns:
        list of CVData with sequential cv_ids
    """
    dataset_dir = Path(dataset_dir)
    if normalizer is None:
        normalizer = SkillNormalizer()

    parser = CVParser(normalizer)
    cvs: list[CVData] = []
    skipped = 0

    for category_dir in sorted(dataset_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        if categories and category_name not in categories:
            continue

        for pdf_path in sorted(category_dir.glob("*.pdf")):
            if max_cvs and len(cvs) >= max_cvs:
                break

            try:
                cv = parser.parse_file(str(pdf_path), cv_id=len(cvs))

                if len(cv.skills) < min_skills:
                    skipped += 1
                    continue

                cvs.append(cv)

            except Exception as e:
                logger.debug("Failed to parse %s: %s", pdf_path.name, e)
                skipped += 1

        if max_cvs and len(cvs) >= max_cvs:
            break

    logger.info(
        "Loaded %d CVs from %s (skipped %d with < %d skills)",
        len(cvs), dataset_dir, skipped, min_skills,
    )
    return cvs
