"""Provider registry for crawl providers."""

from __future__ import annotations

from ml_service.crawler.base import CrawlProvider

_REGISTRY: dict[str, type[CrawlProvider]] = {}


def _ensure_registry() -> None:
    if _REGISTRY:
        return
    from ml_service.crawler.jobspy_provider import JobSpyProvider

    _REGISTRY["jobspy"] = JobSpyProvider


def register_provider(name: str, cls: type[CrawlProvider]) -> None:
    """Register a custom provider at runtime."""
    _REGISTRY[name] = cls


def get_provider(name: str = "jobspy", **kwargs) -> CrawlProvider:
    """Instantiate a crawl provider by name."""
    _ensure_registry()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown crawl provider '{name}'. Available: {available}")
    return cls(**kwargs)
