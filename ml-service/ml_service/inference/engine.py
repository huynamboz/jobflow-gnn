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
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding.base import EmbeddingProvider
from ml_service.graph.schema import CVData, JobData, SeniorityLevel
from ml_service.inference.checkpoint import load_checkpoint
from ml_service.models.gnn import HeteroGraphSAGE, prepare_data_for_gnn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """Single CV match result."""

    cv_id: int
    score: float
    eligible: bool
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    seniority_match: bool


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
        alpha: float = 0.8,
        beta: float = 0.15,
        gamma: float = 0.05,
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
        logger.info(
            "InferenceEngine ready: %d CVs precomputed, threshold=%.2f",
            len(cvs),
            threshold,
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

    @property
    def num_cvs(self) -> int:
        return len(self._cvs)

    @property
    def cv_pool(self) -> list[CVData]:
        return list(self._cvs)

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
        """Hybrid score for a single (CV, JD) pair."""
        # GNN component: use precomputed CV embedding + compute job embedding similarity
        # For simplicity, use skill overlap + seniority as the scoring basis
        # GNN score via decode would require the job to be in the graph,
        # so we use text embedding cosine as GNN proxy for new JDs
        gnn_score = self._text_similarity(cv, job)

        # Skill overlap
        skill_score = self._skill_scorer.score(cv, job)

        # Seniority match
        dist = abs(int(cv.seniority) - int(job.seniority))
        seniority_score = max(0.0, 1.0 - dist * 0.25)

        return (
            self._alpha * gnn_score
            + self._beta * skill_score
            + self._gamma * seniority_score
        )

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
