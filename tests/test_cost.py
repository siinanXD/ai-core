from __future__ import annotations

import pytest

from ai_core.cost import ModelPricing, estimate_cost

PRICES = {
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
}


def test_cost_calculation_uses_explicit_prices() -> None:
    estimate = estimate_cost("gpt-4o-mini", 1_000_000, 500_000, PRICES)

    assert estimate.status == "known"
    assert estimate.known is True
    assert estimate.estimated_cost_usd == 0.45
    assert estimate.input_tokens == 1_000_000
    assert estimate.output_tokens == 500_000
    assert estimate.currency == "USD"


def test_small_known_cost_is_not_rounded_to_zero() -> None:
    estimate = estimate_cost("gpt-4o-mini", 1, 1, PRICES)

    assert estimate.status == "known"
    assert estimate.estimated_cost_usd is not None
    assert estimate.estimated_cost_usd > 0


def test_cached_input_tokens_make_cost_unknown() -> None:
    estimate = estimate_cost("gpt-4o-mini", 1000, 100, PRICES, cached_input_tokens=500)

    assert estimate.status == "unknown"
    assert estimate.estimated_cost_usd is None


def test_unknown_pricing_stays_unknown() -> None:
    estimate = estimate_cost("not-a-priced-model", 1000, 1000, PRICES)

    assert estimate.status == "unknown"
    assert estimate.estimated_cost_usd is None
    assert estimate.known is False


def test_missing_pricing_table_is_unknown() -> None:
    estimate = estimate_cost("gpt-4o-mini", 10, 10, None)
    assert estimate.status == "unknown"
    assert estimate.estimated_cost_usd is None


def test_single_price_for_a_different_model_is_unknown() -> None:
    estimate = estimate_cost("other", 10, 10, PRICES["gpt-4o-mini"])
    assert estimate.status == "unknown"


def test_negative_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="token counts"):
        estimate_cost("gpt-4o-mini", -1, 0, PRICES)
