"""Import LinkedIn CV dataset (PDFs) to DB.

Usage:
    python manage.py import_cvs --dir /path/to/Dataset
    python manage.py import_cvs --dir /path/to/Dataset --categories "AI,Devops,Software Engineer"
"""

from pathlib import Path

from django.core.management.base import BaseCommand

from apps.cvs.services import CVService


class Command(BaseCommand):
    help = "Import LinkedIn CV PDFs from dataset directory to database"

    def add_arguments(self, parser):
        parser.add_argument("--dir", required=True, help="Path to Dataset directory")
        parser.add_argument("--categories", type=str, default="", help="Comma-separated category filter")
        parser.add_argument("--min-skills", type=int, default=2, help="Min skills to keep CV")

    def handle(self, *args, **options):
        dataset_dir = Path(options["dir"])
        categories = [c.strip() for c in options["categories"].split(",") if c.strip()] or None
        min_skills = options["min_skills"]

        if not dataset_dir.exists():
            self.stdout.write(self.style.ERROR(f"Directory not found: {dataset_dir}"))
            return

        service = CVService()
        created = 0
        skipped = 0
        failed = 0

        for category_dir in sorted(dataset_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category_name = category_dir.name
            if categories and category_name not in categories:
                continue

            cat_count = 0
            for pdf_path in sorted(category_dir.glob("*.pdf")):
                try:
                    cv_data = service._parser.parse_file(str(pdf_path))
                    if len(cv_data.skills) < min_skills:
                        skipped += 1
                        continue

                    service._save_cv(
                        cv_data,
                        file_path=str(pdf_path),
                        source="linkedin_dataset",
                        source_category=category_name,
                    )
                    created += 1
                    cat_count += 1
                except Exception as e:
                    failed += 1

            self.stdout.write(f"  {category_name}: {cat_count} imported")

        self.stdout.write(self.style.SUCCESS(
            f"Done: {created} created, {skipped} skipped (< {min_skills} skills), {failed} failed"
        ))
