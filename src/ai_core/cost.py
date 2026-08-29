"""Cost metadata for a completed generation.

Pricing is never guessed. A model without an explicit price stays unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

CostStatus = Literal["known", "unknown"]


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Caller-supplied list price. Units are USD per 1M tokens."""

    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def __post_init__(self) -> None:
        if self.input_usd_per_mtok < 0 or self.output_usd_per_mtok < 0:
            raise ValueError("prices must be at least 0")


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Estimated spend for one call, or an explicit unknown."""

    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    status: CostStatus
    currency: str = "USD"

    @property
    def known(self) -> bool:
        return self.status == "known" and self.estimated_cost_usd is not None


def _lookup(
    model: str,
    pricing: ModelPricing | Mapping[str, ModelPricing] | None,
) -> ModelPricing | None:
    if pricing is None:
        return None
    if isinstance(pricing, ModelPricing):
        return pricing if pricing.model == model else None
    return pricing.get(model)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing | Mapping[str, ModelPricing] | None,
) -> CostEstimate:
    """Return a cost estimate only when the caller supplied a price for `model`."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be at least 0")

    entry = _lookup(model, pricing)
    if entry is None:
        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,
            status="unknown",
        )

    usd = (
        input_tokens * entry.input_usd_per_mtok + output_tokens * entry.output_usd_per_mtok
    ) / 1_000_000
    return CostEstimate(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(usd, 6),
        status="known",
    )
