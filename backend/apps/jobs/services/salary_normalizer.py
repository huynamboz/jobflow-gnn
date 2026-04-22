"""Normalize raw salary values to a common unit: USD annual."""

from __future__ import annotations

# Approximate rates to USD (1 unit of currency = N USD)
_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "VND": 1 / 25_000,
    "EUR": 1.08,
    "GBP": 1.27,
    "SGD": 0.74,
    "JPY": 1 / 150,
    "KRW": 1 / 1_300,
    "AUD": 0.65,
    "CAD": 0.74,
    "INR": 1 / 83,
    "THB": 1 / 35,
    "MYR": 1 / 4.7,
    "PHP": 1 / 56,
    "IDR": 1 / 15_700,
}

# Multiplier to convert period → annual
_TO_ANNUAL: dict[str, float] = {
    "annual":  1.0,
    "monthly": 12.0,
    "hourly":  52 * 40,  # 52 weeks × 40 hrs
    "unknown": 12.0,     # assume monthly when unclear
}


def to_usd_annual(amount: int, currency: str, salary_type: str) -> int:
    """Convert a raw salary value to USD annual equivalent.

    Returns 0 if amount is 0 or conversion is not possible.
    """
    if amount <= 0:
        return 0
    rate = _TO_USD.get(currency.upper(), 1.0)
    multiplier = _TO_ANNUAL.get(salary_type.lower(), 12.0)
    return int(amount * rate * multiplier)


def normalize_salary_range(
    salary_min: int,
    salary_max: int,
    currency: str,
    salary_type: str,
) -> tuple[int, int]:
    """Return (usd_annual_min, usd_annual_max)."""
    usd_min = to_usd_annual(salary_min, currency, salary_type)
    usd_max = to_usd_annual(salary_max, currency, salary_type)
    # If only one bound given, mirror it
    if usd_min > 0 and usd_max == 0:
        usd_max = usd_min
    if usd_max > 0 and usd_min == 0:
        usd_min = usd_max
    return usd_min, usd_max
