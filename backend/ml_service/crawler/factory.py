"""Provider registry with auto-discovery.

Automatically scans ml_service/crawler/providers/ for CrawlProvider subclasses.
New providers only need to be placed in the providers/ directory.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path

from ml_service.crawler.base import CrawlProvider

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[CrawlProvider]] = {}
_discovered = False


def _discover_providers() -> None:
    """Auto-discover all CrawlProvider subclasses in providers/ package."""
    global _discovered
    if _discovered:
        return

    providers_path = Path(__file__).parent / "providers"
    if not providers_path.exists():
        _discovered = True
        return

    package_name = "ml_service.crawler.providers"
    for importer, module_name, _ in pkgutil.iter_modules([str(providers_path)]):
        try:
            module = importlib.import_module(f"{package_name}.{module_name}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, CrawlProvider) and obj is not CrawlProvider:
                    provider_name = obj.__dict__.get("_name", None)
                    # Instantiate to get name property
                    try:
                        instance = obj.__new__(obj)
                        provider_name = instance.name
                    except Exception:
                        provider_name = module_name.replace("_provider", "")
                    _REGISTRY[provider_name] = obj
                    logger.debug("Discovered provider: %s → %s", provider_name, obj.__name__)
        except Exception as e:
            logger.warning("Failed to load provider module %s: %s", module_name, e)

    _discovered = True
    logger.info("Discovered %d crawl providers: %s", len(_REGISTRY), list(_REGISTRY.keys()))


def register_provider(name: str, cls: type[CrawlProvider]) -> None:
    """Register a custom provider at runtime."""
    _REGISTRY[name] = cls


def get_provider(name: str = "jobspy", **kwargs) -> CrawlProvider:
    """Instantiate a crawl provider by name."""
    _discover_providers()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown crawl provider '{name}'. Available: {available}")
    return cls(**kwargs)


def list_providers() -> list[str]:
    """List all available provider names."""
    _discover_providers()
    return sorted(_REGISTRY.keys())
