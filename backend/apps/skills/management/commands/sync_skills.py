"""Sync skills from skill-alias.json to DB.

Usage: python manage.py sync_skills
"""

from django.core.management.base import BaseCommand

from apps.skills.services import SkillService


class Command(BaseCommand):
    help = "Sync skills from skill-alias.json to database"

    def handle(self, *args, **options):
        created = SkillService.sync_from_alias_file()
        self.stdout.write(self.style.SUCCESS(f"Synced skills: {created} created"))
