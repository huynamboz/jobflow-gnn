"""FastAPI application for JobFlow-GNN matching service.

Endpoints:
    POST /match/jd        — Input JD text → Top K CVs
    POST /match/cv        — Input CV text → Top K Jobs
    POST /match/cv/upload — Upload CV PDF/DOCX → Top K Jobs
    GET  /health          — Service health + stats
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ml_service.cv_parser import CVParser
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.inference import InferenceEngine, JobMatchResult, MatchResult

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints/latest"
SKILL_ALIAS_PATH = "../roadmap/week1/skill-alias.json"

app = FastAPI(
    title="JobFlow-GNN",
    description="CV-Job matching powered by Graph Neural Networks",
    version="0.1.0",
)

# Global engine (loaded on startup)
_engine: InferenceEngine | None = None
_parser: CVParser | None = None


# ── Request/Response models ──────────────────────────────────────────────────


class JDMatchRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Job description text")
    top_k: int = Field(default=10, ge=1, le=100)


class CVMatchRequest(BaseModel):
    text: str = Field(..., min_length=10, description="CV/resume text")
    top_k: int = Field(default=10, ge=1, le=100)


class CVMatchResponse(BaseModel):
    cv_id: int
    score: float
    eligible: bool
    matched_skills: list[str]
    missing_skills: list[str]
    seniority_match: bool


class JobMatchResponse(BaseModel):
    job_id: int
    score: float
    eligible: bool
    matched_skills: list[str]
    missing_skills: list[str]
    seniority_match: bool
    title: str


class HealthResponse(BaseModel):
    status: str
    num_cvs: int
    num_jobs: int
    model_loaded: bool


# ── Startup ──────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    global _engine, _parser
    logger.info("Loading model from %s...", CHECKPOINT_DIR)

    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    provider = get_provider()
    _parser = CVParser(normalizer)

    try:
        _engine = InferenceEngine.from_checkpoint(
            CHECKPOINT_DIR,
            normalizer=normalizer,
            embedding_provider=provider,
        )
        logger.info("Engine ready: %d CVs, %d jobs", _engine.num_cvs, _engine.num_jobs)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        _engine = None


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _engine else "model not loaded",
        num_cvs=_engine.num_cvs if _engine else 0,
        num_jobs=_engine.num_jobs if _engine else 0,
        model_loaded=_engine is not None,
    )


@app.post("/match/jd", response_model=list[CVMatchResponse])
async def match_jd(req: JDMatchRequest):
    """Input JD text → Top K matching CVs."""
    if not _engine:
        raise HTTPException(503, "Model not loaded")

    results = _engine.match(req.text, top_k=req.top_k)
    return [
        CVMatchResponse(
            cv_id=r.cv_id,
            score=r.score,
            eligible=r.eligible,
            matched_skills=list(r.matched_skills),
            missing_skills=list(r.missing_skills),
            seniority_match=r.seniority_match,
        )
        for r in results
    ]


@app.post("/match/cv", response_model=list[JobMatchResponse])
async def match_cv(req: CVMatchRequest):
    """Input CV text → Top K matching Jobs."""
    if not _engine:
        raise HTTPException(503, "Model not loaded")

    results = _engine.match_cv_text(req.text, top_k=req.top_k)
    return [
        JobMatchResponse(
            job_id=r.job_id,
            score=r.score,
            eligible=r.eligible,
            matched_skills=list(r.matched_skills),
            missing_skills=list(r.missing_skills),
            seniority_match=r.seniority_match,
            title=r.title,
        )
        for r in results
    ]


@app.post("/match/cv/upload", response_model=list[JobMatchResponse])
async def match_cv_upload(file: UploadFile = File(...), top_k: int = 10):
    """Upload CV (PDF/DOCX) → Top K matching Jobs."""
    if not _engine or not _parser:
        raise HTTPException(503, "Model not loaded")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        cv_data = _parser.parse_file(tmp_path, cv_id=-1)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not cv_data.skills:
        raise HTTPException(422, "No skills could be extracted from the CV")

    results = _engine.match_cv(cv_data, top_k=top_k)
    return [
        JobMatchResponse(
            job_id=r.job_id,
            score=r.score,
            eligible=r.eligible,
            matched_skills=list(r.matched_skills),
            missing_skills=list(r.missing_skills),
            seniority_match=r.seniority_match,
            title=r.title,
        )
        for r in results
    ]
