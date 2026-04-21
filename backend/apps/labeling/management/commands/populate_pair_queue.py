"""
Management command: populate_pair_queue

Loads CVs and Jobs from files, runs smart stratified pair selection,
and populates LabelingCV, LabelingJob, PairQueue tables.

Usage:
    python manage.py populate_pair_queue \
        --cvs-path data/linkedin_cvs.json \
        --jobs-path data/raw_jobs.jsonl \
        --n-pairs 300 \
        --max-per-cv 8
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

from django.core.management.base import BaseCommand

# ensure backend/ on path
BASE = Path(__file__).resolve().parents[6]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


class Command(BaseCommand):
    help = "Populate PairQueue with smart-selected CV-Job pairs for labeling"

    def add_arguments(self, parser):
        parser.add_argument("--cvs-path",    default="data/linkedin_cvs.json", help="Path to linkedin_cvs.json")
        parser.add_argument("--jobs-path",  default="data/raw_jobs.jsonl",    help="Path to raw_jobs.jsonl")
        parser.add_argument("--dataset-dir", default=None,                    help="Path to Dataset/ folder for PDF paths")
        parser.add_argument("--n-pairs",    type=int, default=300,            help="Total pairs to generate")
        parser.add_argument("--max-per-cv", type=int, default=8,              help="Max pairs per CV")
        parser.add_argument("--seed",       type=int, default=42,             help="Random seed")
        parser.add_argument("--clear",      action="store_true",              help="Clear existing queue first")

    def handle(self, *args, **options):
        from apps.labeling.models import (
            LabelingCV, LabelingJob, PairQueue,
            SelectionReason, REASON_PRIORITY,
        )
        from ml_service.baselines.skill_overlap import SkillOverlapScorer
        from ml_service.crawler.storage import deduplicate, load_raw_jobs
        from ml_service.data.linkedin_cv_loader import load_linkedin_cvs_json
        from ml_service.data.skill_extractor import SkillExtractor
        from ml_service.data.skill_normalization import SkillNormalizer
        from ml_service.graph.schema import SeniorityLevel, EducationLevel

        random.seed(options["seed"])

        if options["clear"]:
            self.stdout.write("Clearing existing queue...")
            PairQueue.objects.all().delete()
            LabelingCV.objects.all().delete()
            LabelingJob.objects.all().delete()

        # --- Load CVs ---
        self.stdout.write("Loading CVs...")
        normalizer = SkillNormalizer("ml_service/data/skill-alias.json")
        cvs = load_linkedin_cvs_json(options["cvs_path"])
        cvs = [c for c in cvs if len(c.skills) >= 2]
        self.stdout.write(f"  Loaded {len(cvs)} CVs")

        # Build cv_id → pdf_path mapping from Dataset dir (same sort order as load_linkedin_cvs)
        cv_pdf_paths: dict[int, str] = {}
        if options["dataset_dir"]:
            dataset_dir = Path(options["dataset_dir"])
            idx = 0
            for category_dir in sorted(dataset_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                for pdf_path in sorted(category_dir.glob("*.pdf")):
                    cv_pdf_paths[idx] = str(pdf_path)
                    idx += 1
            self.stdout.write(f"  Mapped {len(cv_pdf_paths)} PDF paths")

        # --- Load Jobs ---
        self.stdout.write("Loading Jobs...")
        raw_jobs = deduplicate(load_raw_jobs(options["jobs_path"]))
        extractor = SkillExtractor(normalizer)
        extractor.fit(raw_jobs)
        from ml_service.graph.schema import JobData
        all_jobs = extractor.extract_batch(raw_jobs)
        jobs = [j for j in all_jobs if len(j.skills) >= 2]
        jobs = [
            JobData(job_id=i, seniority=j.seniority, skills=j.skills,
                    skill_importances=j.skill_importances,
                    salary_min=j.salary_min, salary_max=j.salary_max, text=j.text)
            for i, j in enumerate(jobs)
        ]
        self.stdout.write(f"  Loaded {len(jobs)} Jobs")

        # --- Upsert LabelingCV / LabelingJob ---
        self.stdout.write("Upserting LabelingCV records...")
        cv_objs = {}
        for cv in cvs:
            obj, _ = LabelingCV.objects.update_or_create(
                cv_id=cv.cv_id,
                defaults=dict(
                    source="linkedin",
                    skills=list(cv.skills),
                    seniority=cv.seniority.name,
                    experience_years=float(cv.experience_years),
                    education=cv.education.name,
                    text_summary=cv.text[:500] if cv.text else "",
                    pdf_path=cv_pdf_paths.get(cv.cv_id, ""),
                ),
            )
            cv_objs[cv.cv_id] = obj

        self.stdout.write("Upserting LabelingJob records...")
        job_objs = {}
        for job in jobs:
            obj, _ = LabelingJob.objects.update_or_create(
                job_id=job.job_id,
                defaults=dict(
                    title=job.text[:100].split("\n")[0] if job.text else f"Job {job.job_id}",
                    skills=list(job.skills),
                    seniority=job.seniority.name,
                    salary_min=job.salary_min if job.salary_min else None,
                    salary_max=job.salary_max if job.salary_max else None,
                    text_summary=job.text[:500] if job.text else "",
                ),
            )
            job_objs[job.job_id] = obj

        # --- Smart pair selection ---
        self.stdout.write("Computing skill overlaps and selecting pairs...")
        scorer = SkillOverlapScorer()
        n_pairs = options["n_pairs"]
        max_per_cv = options["max_per_cv"]

        buckets: dict[str, list] = {r: [] for r in SelectionReason.values}
        cv_count: dict[int, int] = defaultdict(int)

        for cv in cvs:
            if cv_count[cv.cv_id] >= max_per_cv:
                continue
            for job in jobs:
                if cv_count[cv.cv_id] >= max_per_cv:
                    break
                overlap = scorer.score(cv, job)
                seniority_gap = abs(int(cv.seniority) - int(job.seniority))

                if 0.15 <= overlap < 0.40:
                    reason = SelectionReason.MEDIUM_OVERLAP
                elif overlap >= 0.40:
                    reason = SelectionReason.HIGH_OVERLAP
                elif seniority_gap > 1:
                    reason = SelectionReason.HARD_NEGATIVE
                else:
                    reason = SelectionReason.RANDOM

                buckets[reason].append((cv, job, overlap, reason))
                cv_count[cv.cv_id] += 1

        # Shuffle each bucket
        for bucket in buckets.values():
            random.shuffle(bucket)

        # Sample according to ratio
        ratios = {
            SelectionReason.MEDIUM_OVERLAP: 0.40,
            SelectionReason.HIGH_OVERLAP:   0.30,
            SelectionReason.HARD_NEGATIVE:  0.20,
            SelectionReason.RANDOM:         0.10,
        }
        selected = []
        for reason, ratio in ratios.items():
            quota = int(n_pairs * ratio)
            selected.extend(buckets[reason][:quota])

        # If short, fill remainder from largest bucket
        if len(selected) < n_pairs:
            extras_needed = n_pairs - len(selected)
            used = set((cv.cv_id, job.job_id) for cv, job, _, _ in selected)
            for reason in ratios:
                for item in buckets[reason]:
                    cv, job, overlap, r = item
                    key = (cv.cv_id, job.job_id)
                    if key not in used and extras_needed > 0:
                        selected.append(item)
                        used.add(key)
                        extras_needed -= 1

        random.shuffle(selected)

        # --- Pre-assign splits (70/15/15) per bucket ---
        self.stdout.write(f"  Selected {len(selected)} pairs, assigning splits...")
        split_assigned = []
        for i, (cv, job, overlap, reason) in enumerate(selected):
            r = random.random()
            if r < 0.70:
                split = "train"
            elif r < 0.85:
                split = "val"
            else:
                split = "test"
            split_assigned.append((cv, job, overlap, reason, split))

        # --- Bulk insert PairQueue ---
        self.stdout.write("Inserting PairQueue entries...")
        existing = set(
            PairQueue.objects.values_list("cv__cv_id", "job__job_id")
        )
        to_create = []
        for cv, job, overlap, reason, split in split_assigned:
            if (cv.cv_id, job.job_id) in existing:
                continue
            to_create.append(PairQueue(
                cv=cv_objs[cv.cv_id],
                job=job_objs[job.job_id],
                skill_overlap_score=round(overlap, 4),
                selection_reason=reason,
                priority=REASON_PRIORITY[reason],
                split=split,
            ))

        PairQueue.objects.bulk_create(to_create, ignore_conflicts=True)

        # Summary
        total = PairQueue.objects.count()
        self.stdout.write(self.style.SUCCESS(
            f"\nDone! PairQueue has {total} pairs "
            f"({PairQueue.objects.filter(split='train').count()} train / "
            f"{PairQueue.objects.filter(split='val').count()} val / "
            f"{PairQueue.objects.filter(split='test').count()} test)"
        ))
