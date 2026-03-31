"""Skill service: manage canonical skills in DB."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from apps.skills.models import Skill

logger = logging.getLogger(__name__)


class SkillService:
    """Get or create skills in DB."""

    def get_or_create(self, canonical_name: str) -> Skill | None:
        """Get or create a skill by canonical name."""
        if not canonical_name:
            return None
        skill, _ = Skill.objects.get_or_create(
            canonical_name=canonical_name,
            defaults={"category": self._get_category(canonical_name)},
        )
        return skill

    def _get_category(self, canonical_name: str) -> int:
        """Lookup category from skill normalizer."""
        try:
            from ml_service.data.skill_normalization import SkillNormalizer
            normalizer = SkillNormalizer()
            catalog = normalizer.skill_catalog
            cat = catalog.get(canonical_name)
            return int(cat) if cat is not None else 0
        except Exception:
            return 0

    @staticmethod
    def sync_from_alias_file() -> int:
        """Sync all skills from skill-alias.json to DB. Returns count created."""
        from ml_service.data.skill_normalization import SkillNormalizer
        normalizer = SkillNormalizer()
        created = 0

        for canonical_name, category in normalizer.skill_catalog.items():
            _, was_created = Skill.objects.get_or_create(
                canonical_name=canonical_name,
                defaults={
                    "category": int(category),
                    "aliases": normalizer._alias_map.get(canonical_name.lower(), []),
                },
            )
            if was_created:
                created += 1

        logger.info("Synced skills: %d created, %d total", created, Skill.objects.count())
        return created
