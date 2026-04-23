"""Re-parse raw_text for all linkedin_dataset CVs from original PDF files.

Usage:
    python manage.py reparse_cv_text --dataset-dir /path/to/Dataset
    python manage.py reparse_cv_text --dataset-dir /path/to/Dataset --min-chars 300 --dry-run
"""

from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from apps.cvs.models import CV


def _extract_text(pdf_path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # Try normal extract first
            text = page.extract_text()
            if text:
                pages.append(text)
            else:
                # Fallback: extract_text with layout=True for complex PDFs
                text2 = page.extract_text(layout=True)
                if text2:
                    pages.append(text2)
    return "\n".join(pages).strip()


class Command(BaseCommand):
    help = "Re-parse raw_text for linkedin_dataset CVs from original PDF files"

    def add_arguments(self, parser):
        parser.add_argument("--dataset-dir", required=True, help="Path to Dataset directory (e.g. /path/to/Dataset)")
        parser.add_argument("--min-chars", type=int, default=0, help="Only re-parse CVs with raw_text shorter than this. 0 = all.")
        parser.add_argument("--dry-run", action="store_true", help="Print what would happen without saving")
        parser.add_argument("--category", type=str, default="", help="Only process specific source_category")

    def handle(self, *args, **options):
        dataset_dir = Path(options["dataset_dir"])
        min_chars = options["min_chars"]
        dry_run = options["dry_run"]
        category_filter = options["category"].strip()

        if not dataset_dir.exists():
            self.stdout.write(self.style.ERROR(f"Dataset dir not found: {dataset_dir}"))
            return

        qs = CV.objects.filter(source="linkedin_dataset")
        if category_filter:
            qs = qs.filter(source_category=category_filter)
        if min_chars > 0:
            qs = qs.extra(where=[f"LENGTH(raw_text) < {min_chars}"])

        total = qs.count()
        self.stdout.write(f"CVs to process: {total} {'(dry-run)' if dry_run else ''}")

        updated = 0
        failed = 0
        improved = 0

        for cv in qs.order_by("id"):
            pdf_path = dataset_dir / cv.source_category / cv.file_name
            if not pdf_path.exists():
                self.stdout.write(self.style.WARNING(f"  NOT FOUND: {pdf_path}"))
                failed += 1
                continue

            try:
                new_text = _extract_text(pdf_path)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  FAIL CV #{cv.id} {cv.file_name}: {e}"))
                failed += 1
                continue

            old_len = len(cv.raw_text)
            new_len = len(new_text)
            gain = new_len - old_len

            if not new_text:
                self.stdout.write(self.style.WARNING(f"  EMPTY CV #{cv.id} {cv.file_name} (scanned PDF?)"))
                failed += 1
                continue

            status = f"CV #{cv.id:4d} {cv.file_name[:35]:35s} | {old_len:5d} → {new_len:5d} chars ({gain:+d})"
            if gain > 50:
                self.stdout.write(self.style.SUCCESS(f"  ↑ {status}"))
                improved += 1
            elif gain < -50:
                self.stdout.write(self.style.WARNING(f"  ↓ {status}"))
            else:
                self.stdout.write(f"  = {status}")

            if not dry_run and new_text != cv.raw_text:
                with transaction.atomic():
                    CV.objects.filter(id=cv.id).update(
                        raw_text=new_text,
                        parsed_text=new_text,
                    )
            updated += 1

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"Done: {updated} processed, {improved} improved, {failed} failed/missing"
        ))
        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — nothing was saved"))
