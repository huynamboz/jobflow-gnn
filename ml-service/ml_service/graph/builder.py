from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ml_service.embedding.base import EmbeddingProvider
from ml_service.graph.schema import (
    CVData,
    JobData,
    LabeledPair,
    SeniorityLevel,
    SkillCategory,
)

# Salary normalization constant (max monthly salary in dataset)
_SALARY_NORM = 10_000.0


class GraphBuilder:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embed = embedding_provider

    def build(
        self,
        cvs: list[CVData],
        jobs: list[JobData],
        skill_catalog: dict[str, SkillCategory],
        pairs: list[LabeledPair],
    ) -> HeteroData:
        data = HeteroData()

        # Build index mappings
        cv_id_to_idx = {cv.cv_id: i for i, cv in enumerate(cvs)}
        job_id_to_idx = {job.job_id: i for i, job in enumerate(jobs)}
        skill_names = sorted(skill_catalog.keys())
        skill_to_idx = {s: i for i, s in enumerate(skill_names)}

        # --- CV nodes: embedding(384) + experience_years + education_level = 386 ---
        cv_texts = [cv.text for cv in cvs]
        cv_embeddings = self._embed.encode(cv_texts)  # (N, 384)
        cv_extra = np.array(
            [[cv.experience_years, float(cv.education)] for cv in cvs],
            dtype=np.float32,
        )
        data["cv"].x = torch.from_numpy(np.concatenate([cv_embeddings, cv_extra], axis=1))

        # --- Job nodes: embedding(384) + salary_min_norm + salary_max_norm = 386 ---
        job_texts = [job.text for job in jobs]
        job_embeddings = self._embed.encode(job_texts)  # (N, 384)
        job_extra = np.array(
            [[job.salary_min / _SALARY_NORM, job.salary_max / _SALARY_NORM] for job in jobs],
            dtype=np.float32,
        )
        data["job"].x = torch.from_numpy(np.concatenate([job_embeddings, job_extra], axis=1))

        # --- Skill nodes: embedding(384) + category = 385 ---
        skill_embeddings = self._embed.encode(skill_names)  # (S, 384)
        skill_cats = np.array(
            [[float(skill_catalog[s])] for s in skill_names],
            dtype=np.float32,
        )
        data["skill"].x = torch.from_numpy(np.concatenate([skill_embeddings, skill_cats], axis=1))

        # --- Seniority nodes: identity matrix (6, 6) ---
        data["seniority"].x = torch.eye(len(SeniorityLevel))

        # --- has_skill edges: CV -> Skill (with proficiency weight) ---
        hs_src, hs_dst, hs_attr = [], [], []
        for cv in cvs:
            cv_idx = cv_id_to_idx[cv.cv_id]
            for skill, prof in zip(cv.skills, cv.skill_proficiencies):
                if skill in skill_to_idx:
                    hs_src.append(cv_idx)
                    hs_dst.append(skill_to_idx[skill])
                    hs_attr.append(float(prof))
        data["cv", "has_skill", "skill"].edge_index = torch.tensor(
            [hs_src, hs_dst], dtype=torch.long
        )
        data["cv", "has_skill", "skill"].edge_attr = torch.tensor(hs_attr, dtype=torch.float)

        # --- requires_skill edges: Job -> Skill (with importance weight) ---
        rs_src, rs_dst, rs_attr = [], [], []
        for job in jobs:
            job_idx = job_id_to_idx[job.job_id]
            for skill, imp in zip(job.skills, job.skill_importances):
                if skill in skill_to_idx:
                    rs_src.append(job_idx)
                    rs_dst.append(skill_to_idx[skill])
                    rs_attr.append(float(imp))
        data["job", "requires_skill", "skill"].edge_index = torch.tensor(
            [rs_src, rs_dst], dtype=torch.long
        )
        data["job", "requires_skill", "skill"].edge_attr = torch.tensor(rs_attr, dtype=torch.float)

        # --- has_seniority edges: CV -> Seniority ---
        hsen_src, hsen_dst = [], []
        for cv in cvs:
            hsen_src.append(cv_id_to_idx[cv.cv_id])
            hsen_dst.append(int(cv.seniority))
        data["cv", "has_seniority", "seniority"].edge_index = torch.tensor(
            [hsen_src, hsen_dst], dtype=torch.long
        )

        # --- requires_seniority edges: Job -> Seniority ---
        rsen_src, rsen_dst = [], []
        for job in jobs:
            rsen_src.append(job_id_to_idx[job.job_id])
            rsen_dst.append(int(job.seniority))
        data["job", "requires_seniority", "seniority"].edge_index = torch.tensor(
            [rsen_src, rsen_dst], dtype=torch.long
        )

        # --- match / no_match label edges: CV -> Job ---
        match_src, match_dst = [], []
        nomatch_src, nomatch_dst = [], []
        for pair in pairs:
            cv_idx = cv_id_to_idx.get(pair.cv_id)
            job_idx = job_id_to_idx.get(pair.job_id)
            if cv_idx is None or job_idx is None:
                continue
            if pair.label == 1:
                match_src.append(cv_idx)
                match_dst.append(job_idx)
            else:
                nomatch_src.append(cv_idx)
                nomatch_dst.append(job_idx)

        data["cv", "match", "job"].edge_index = torch.tensor(
            [match_src, match_dst], dtype=torch.long
        )
        data["cv", "no_match", "job"].edge_index = torch.tensor(
            [nomatch_src, nomatch_dst], dtype=torch.long
        )

        return data
