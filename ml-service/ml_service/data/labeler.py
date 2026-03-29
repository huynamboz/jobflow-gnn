from __future__ import annotations

import random as _random_mod

from ml_service.graph.schema import CVData, DatasetSplit, JobData, LabeledPair


class PairLabeler:
    """Threshold-based pair labeling with stratified train/val/test split."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = _random_mod.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_pairs(
        self,
        cvs: list[CVData],
        jobs: list[JobData],
        num_positive: int,
    ) -> list[LabeledPair]:
        """Classify all CV-Job pairs, then sample with 1:1:2 ratio.

        Target: num_positive positives, num_positive easy negatives,
        2 * num_positive hard negatives  →  total = 4 * num_positive.
        """
        positives: list[tuple[int, int]] = []
        easy_negs: list[tuple[int, int]] = []
        hard_negs: list[tuple[int, int]] = []

        for cv in cvs:
            for job in jobs:
                overlap = self._skill_overlap(cv, job)
                dist = self._seniority_distance(cv, job)

                if overlap >= 0.5 and dist <= 1:
                    positives.append((cv.cv_id, job.job_id))
                elif overlap < 0.2 or dist >= 3:
                    easy_negs.append((cv.cv_id, job.job_id))
                elif 0.2 <= overlap < 0.5 and dist <= 1:
                    hard_negs.append((cv.cv_id, job.job_id))
                # else: ambiguous — skip

        # Sample desired counts (cap to available)
        n_pos = min(num_positive, len(positives))
        n_easy = min(n_pos, len(easy_negs))
        n_hard = min(2 * n_pos, len(hard_negs))

        sampled_pos = self._rng.sample(positives, n_pos)
        sampled_easy = self._rng.sample(easy_negs, n_easy)
        sampled_hard = self._rng.sample(hard_negs, n_hard)

        pairs: list[LabeledPair] = []
        for cv_id, job_id in sampled_pos:
            pairs.append(LabeledPair(cv_id=cv_id, job_id=job_id, label=1))
        for cv_id, job_id in sampled_easy:
            pairs.append(LabeledPair(cv_id=cv_id, job_id=job_id, label=0))
        for cv_id, job_id in sampled_hard:
            pairs.append(LabeledPair(cv_id=cv_id, job_id=job_id, label=0))

        self._rng.shuffle(pairs)
        return pairs

    def split(self, pairs: list[LabeledPair]) -> DatasetSplit:
        """Stratified 75/15/10 split (by label)."""
        pos = [p for p in pairs if p.label == 1]
        neg = [p for p in pairs if p.label == 0]
        self._rng.shuffle(pos)
        self._rng.shuffle(neg)

        def _split_list(items: list[LabeledPair]) -> tuple[list, list, list]:
            n = len(items)
            n_train = int(n * 0.75)
            n_val = int(n * 0.15)
            train = [LabeledPair(p.cv_id, p.job_id, p.label, "train") for p in items[:n_train]]
            val = [
                LabeledPair(p.cv_id, p.job_id, p.label, "val")
                for p in items[n_train : n_train + n_val]
            ]
            test = [
                LabeledPair(p.cv_id, p.job_id, p.label, "test") for p in items[n_train + n_val :]
            ]
            return train, val, test

        pos_train, pos_val, pos_test = _split_list(pos)
        neg_train, neg_val, neg_test = _split_list(neg)

        train = pos_train + neg_train
        val = pos_val + neg_val
        test = pos_test + neg_test
        self._rng.shuffle(train)
        self._rng.shuffle(val)
        self._rng.shuffle(test)

        return DatasetSplit(train=train, val=val, test=test)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skill_overlap(cv: CVData, job: JobData) -> float:
        """Fraction of job-required skills present in the CV."""
        if not job.skills:
            return 0.0
        cv_set = set(cv.skills)
        job_set = set(job.skills)
        return len(cv_set & job_set) / len(job_set)

    @staticmethod
    def _seniority_distance(cv: CVData, job: JobData) -> int:
        """Absolute difference in seniority level indices."""
        return abs(int(cv.seniority) - int(job.seniority))
