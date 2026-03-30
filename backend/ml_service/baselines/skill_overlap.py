from __future__ import annotations

from ml_service.baselines.base import Scorer
from ml_service.graph.schema import CVData, JobData


class SkillOverlapScorer(Scorer):
    """Jaccard similarity between CV skills and Job required skills."""

    def score(self, cv: CVData, job: JobData) -> float:
        cv_set = set(cv.skills)
        job_set = set(job.skills)
        union = cv_set | job_set
        if not union:
            return 0.0
        return len(cv_set & job_set) / len(union)
