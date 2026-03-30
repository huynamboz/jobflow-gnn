"""Inference engine — input JD text, output ranked CVs with scores.

Design:
  1. CV embeddings are precomputed once at init (or load from checkpoint)
  2. New JD → extract skills → build temp node → encode → score against all CVs
  3. Hybrid scoring: α×GNN + β×skill_overlap + γ×seniority_match
  4. Return Top K results with eligible flag
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.crawler.skill_extractor import SkillExtractor
from ml_service.data.skill_graph import build_skill_cooccurrence
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding.base import EmbeddingProvider
from ml_service.graph.schema import CVData, JobData, SeniorityLevel
from ml_service.inference.checkpoint import load_checkpoint
from ml_service.inference.role_classifier import infer_role, role_match_penalty
from ml_service.models.gnn import HeteroGraphSAGE, prepare_data_for_gnn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """Single CV match result (JD → CVs)."""

    cv_id: int
    score: float
    eligible: bool
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    seniority_match: bool


@dataclass(frozen=True)
class JobMatchResult:
    """Single Job match result (CV → Jobs)."""

    job_id: int
    score: float
    eligible: bool
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    seniority_match: bool
    title: str = ""
    company: str = ""


class InferenceEngine:
    """Precompute CV embeddings once, score new JDs against all CVs.

    Usage:
        engine = InferenceEngine.from_checkpoint("checkpoints/latest", normalizer, provider)
        results = engine.match("Senior Python developer with Django and AWS experience", top_k=10)
    """

    def __init__(
        self,
        model: HeteroGraphSAGE,
        data: HeteroData,
        cvs: list[CVData],
        jobs: list[JobData],
        normalizer: SkillNormalizer,
        embedding_provider: EmbeddingProvider,
        *,
        alpha: float = 0.55,
        beta: float = 0.30,
        gamma: float = 0.15,
        threshold: float = 0.65,
    ) -> None:
        self._model = model
        self._data = data
        self._cvs = cvs
        self._jobs = jobs
        self._normalizer = normalizer
        self._embed = embedding_provider
        self._extractor = SkillExtractor(normalizer)
        self._skill_scorer = SkillOverlapScorer()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._threshold = threshold

        # Precompute CV embeddings from GNN
        self._cv_embeddings = self._precompute_cv_embeddings()

        # Build skill similarity lookup from co-occurrence graph
        self._skill_similarity = self._build_skill_similarity(cvs, jobs)

        logger.info(
            "InferenceEngine ready: %d CVs, %d jobs, %d skill-pairs, threshold=%.2f",
            len(cvs), len(jobs), len(self._skill_similarity), threshold,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        normalizer: SkillNormalizer,
        embedding_provider: EmbeddingProvider,
        **kwargs,
    ) -> InferenceEngine:
        """Load model from checkpoint and build engine."""
        model, data, cvs, jobs, meta = load_checkpoint(checkpoint_path)
        return cls(
            model=model,
            data=data,
            cvs=cvs,
            jobs=jobs,
            normalizer=normalizer,
            embedding_provider=embedding_provider,
            **kwargs,
        )

    def match(self, jd_text: str, top_k: int = 10) -> list[MatchResult]:
        """Match a JD text against all CVs. Returns Top K ranked results."""
        # Extract structured job info from text
        from ml_service.crawler.base import RawJob

        raw = RawJob(
            source="query",
            source_url="",
            title="",
            company="",
            location="",
            description=jd_text,
        )
        job_data = self._extractor.extract(raw, job_id=-1)

        if not job_data.skills:
            logger.warning("No skills extracted from JD text")
            return []

        # Score all CVs
        results: list[MatchResult] = []
        for cv in self._cvs:
            score = self._score_pair(cv, job_data)
            cv_skills = set(cv.skills)
            job_skills = set(job_data.skills)
            matched = tuple(sorted(cv_skills & job_skills))
            missing = tuple(sorted(job_skills - cv_skills))
            sen_match = abs(int(cv.seniority) - int(job_data.seniority)) <= 1

            results.append(
                MatchResult(
                    cv_id=cv.cv_id,
                    score=round(score, 4),
                    eligible=score >= self._threshold,
                    matched_skills=matched,
                    missing_skills=missing,
                    seniority_match=sen_match,
                )
            )

        # Sort by score descending, take top K
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def match_job_data(self, job: JobData, top_k: int = 10) -> list[MatchResult]:
        """Match a pre-parsed JobData against all CVs."""
        results: list[MatchResult] = []
        for cv in self._cvs:
            score = self._score_pair(cv, job)
            cv_skills = set(cv.skills)
            job_skills = set(job.skills)

            results.append(
                MatchResult(
                    cv_id=cv.cv_id,
                    score=round(score, 4),
                    eligible=score >= self._threshold,
                    matched_skills=tuple(sorted(cv_skills & job_skills)),
                    missing_skills=tuple(sorted(job_skills - cv_skills)),
                    seniority_match=abs(int(cv.seniority) - int(job.seniority)) <= 1,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def match_cv(self, cv: CVData, top_k: int = 10) -> list[JobMatchResult]:
        """Match a CV against all Jobs. Returns Top K ranked jobs."""
        results: list[JobMatchResult] = []
        for job in self._jobs:
            score = self._score_pair(cv, job)
            cv_skills = set(cv.skills)
            job_skills = set(job.skills)

            # Extract title from job text (first sentence)
            title = job.text.split(".")[0] if job.text else ""

            results.append(
                JobMatchResult(
                    job_id=job.job_id,
                    score=round(score, 4),
                    eligible=score >= self._threshold,
                    matched_skills=tuple(sorted(cv_skills & job_skills)),
                    missing_skills=tuple(sorted(job_skills - cv_skills)),
                    seniority_match=abs(int(cv.seniority) - int(job.seniority)) <= 1,
                    title=title,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def match_cv_text(self, cv_text: str, top_k: int = 10) -> list[JobMatchResult]:
        """Match raw CV text against all Jobs."""
        from ml_service.cv_parser import CVParser

        parser = CVParser(self._normalizer)
        cv = parser.parse_text(cv_text, cv_id=-1)

        if not cv.skills:
            logger.warning("No skills extracted from CV text")
            return []

        return self.match_cv(cv, top_k=top_k)

    @property
    def num_cvs(self) -> int:
        return len(self._cvs)

    @property
    def num_jobs(self) -> int:
        return len(self._jobs)

    @property
    def cv_pool(self) -> list[CVData]:
        return list(self._cvs)

    @property
    def job_pool(self) -> list[JobData]:
        return list(self._jobs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _precompute_cv_embeddings(self) -> torch.Tensor:
        """Run GNN encode once on the full graph, extract CV embeddings."""
        self._model.eval()
        data_prepared = prepare_data_for_gnn(self._data)

        # Strip label edges for clean inference
        from ml_service.training.trainer import _strip_label_edges

        data_clean = _strip_label_edges(data_prepared)
        data_clean = prepare_data_for_gnn(data_clean)

        with torch.no_grad():
            z_dict = self._model.encode(data_clean)
        return z_dict["cv"]  # [num_cvs, hidden_channels]

    def _score_pair(self, cv: CVData, job: JobData) -> float:
        """Hybrid score with semantic skills, must-have penalty, and role penalty.

        Components:
        1. Text similarity (α=0.55) — semantic matching
        2. Semantic skill overlap (β=0.30) — weighted + related skills from graph
        3. Seniority match (γ=0.15) — stricter penalty
        4. Must-have penalty — cap score if missing required skills
        5. Role penalty — mismatch between CV role and JD role
        """
        gnn_score = self._text_similarity(cv, job)

        # Semantic skill overlap: direct + related skills via graph
        skill_score = self._semantic_skill_overlap(cv, job)

        # Seniority: stricter penalty (0.4 per level)
        dist = abs(int(cv.seniority) - int(job.seniority))
        seniority_score = max(0.0, 1.0 - dist * 0.4)

        base_score = (
            self._alpha * gnn_score
            + self._beta * skill_score
            + self._gamma * seniority_score
        )

        # Role penalty
        cv_role = infer_role(cv.skills, cv.text)
        job_role = infer_role(job.skills, job.text)
        penalty = role_match_penalty(cv_role, job_role)

        score = base_score * penalty

        # Must-have penalty: cap score if missing high-importance skills
        score = self._apply_must_have_penalty(score, cv, job)

        return score

    def _semantic_skill_overlap(self, cv: CVData, job: JobData) -> float:
        """Skill overlap with semantic matching via skill graph.

        For each required JD skill:
        - Direct match in CV → full importance credit
        - Related skill in CV (via co-occurrence graph) → partial credit (0.3-0.6)
        - No match → 0
        """
        if not job.skills:
            return 0.0

        cv_set = set(cv.skills)
        total_weight = 0.0
        matched_weight = 0.0

        for skill, importance in zip(job.skills, job.skill_importances):
            total_weight += importance

            if skill in cv_set:
                # Direct match → full credit
                matched_weight += importance
            else:
                # Check related skills via graph
                best_sim = 0.0
                for cv_skill in cv_set:
                    pair = (min(skill, cv_skill), max(skill, cv_skill))
                    sim = self._skill_similarity.get(pair, 0.0)
                    best_sim = max(best_sim, sim)
                if best_sim > 0:
                    # Partial credit: 30-60% of importance based on similarity
                    matched_weight += importance * best_sim * 0.6

        if total_weight == 0:
            return 0.0
        return min(matched_weight / total_weight, 1.0)

    @staticmethod
    def _apply_must_have_penalty(score: float, cv: CVData, job: JobData) -> float:
        """Cap score if CV is missing high-importance (required) skills.

        Skills with importance >= 4 are considered "must-have".
        Missing 1 must-have → cap at 0.70
        Missing 2+ must-haves → cap at 0.55
        """
        if not job.skills:
            return score

        cv_set = set(cv.skills)
        missing_required = 0
        for skill, importance in zip(job.skills, job.skill_importances):
            if importance >= 4 and skill not in cv_set:
                missing_required += 1

        if missing_required >= 2:
            return min(score, 0.55)
        elif missing_required == 1:
            return min(score, 0.70)
        return score

    @staticmethod
    def _build_skill_similarity(
        cvs: list[CVData], jobs: list[JobData],
    ) -> dict[tuple[str, str], float]:
        """Build skill-pair similarity lookup from co-occurrence PMI.

        Returns dict of (skill_a, skill_b) → normalized similarity (0-1).
        """
        cooccurrence = build_skill_cooccurrence(cvs, jobs)
        if not cooccurrence:
            return {}

        # Normalize PMI to 0-1
        max_pmi = max(cooccurrence.values())
        min_pmi = min(cooccurrence.values())
        rng = max_pmi - min_pmi if max_pmi > min_pmi else 1.0

        return {
            pair: (pmi - min_pmi) / rng
            for pair, pmi in cooccurrence.items()
        }

    def _text_similarity(self, cv: CVData, job: JobData) -> float:
        """Cosine similarity between CV and JD text embeddings."""
        vecs = self._embed.encode([cv.text, job.text])
        cv_vec, job_vec = vecs[0], vecs[1]
        norm_cv = np.linalg.norm(cv_vec)
        norm_job = np.linalg.norm(job_vec)
        if norm_cv == 0 or norm_job == 0:
            return 0.0
        sim = float(np.dot(cv_vec, job_vec) / (norm_cv * norm_job))
        # Normalize from [-1,1] to [0,1]
        return (sim + 1.0) / 2.0
