"""The device, and the invariants that make a support status mean something."""

import pytest

from drainlag.magnitude import SPAN_TOLERANCE, Magnitude, Range, tally, unquotable, violations


def m(**kw):
    base = {"rule": "R", "claimed_text": "2-3x", "claimed": Range(2, 3),
            "support": "folklore", "where": "docstring"}
    base.update(kw)
    return Magnitude(**base)


def test_measured_needs_a_dataset_and_an_observation():
    """A claim cannot call itself measured without pointing at the measurement."""
    with pytest.raises(ValueError, match="requires an observed"):
        m(support="measured", observed=Range(1, 2))          # no dataset
    with pytest.raises(ValueError, match="requires an observed"):
        m(support="measured", dataset="data/x.json")         # no observation


def test_folklore_may_not_carry_an_observation():
    """Once you measure it, it is measured or retracted -- not folklore."""
    with pytest.raises(ValueError, match="folklore with an observed"):
        m(support="folklore", observed=Range(1, 2), dataset="data/x.json")


def test_a_retraction_whose_measurement_agreed_is_refused():
    """The first version of this invariant was only this test, and it was wrong.

    Non-containment alone rejected both real retractions, because 5-50x sits
    inside an observed 0.98-595x. The spread test below is what fixed it; this
    one still has to hold for a claim that genuinely matches.
    """
    with pytest.raises(ValueError, match="theatre"):
        m(claimed=Range(2, 3), support="retracted",
          observed=Range(1.9, 3.2), dataset="data/x.json")


def test_a_claim_that_understates_the_spread_earns_a_retraction():
    """Wrong about being a range, rather than wrong about a value.

    This is BD001: 5-50x is inside 0.98-595x and still fails, because a
    factor-of-10 window was offered as the description of a factor-of-600
    spread.
    """
    r = m(claimed_text="5-50x", claimed=Range(5, 50), support="retracted",
          observed=Range(0.98, 594.6), dataset="data/sweep.json",
          knowable_from_source=False)
    assert r.observed.contains(r.claimed)
    assert r.understates_spread_by > SPAN_TOLERANCE
    assert r.understates_spread_by == pytest.approx(60.7, rel=0.02)


def test_a_claim_that_misses_entirely_also_earns_one():
    r = m(claimed=Range(2, 3), support="retracted",
          observed=Range(20, 30), dataset="data/x.json")
    assert r.wrong_by == pytest.approx(20 / 3, rel=1e-6)


def test_an_unknowable_magnitude_may_not_be_quoted_to_a_user():
    """The invariant that changed the code.

    However well measured, a factor the source cannot determine must not appear
    in a message. Say the mechanism instead.
    """
    with pytest.raises(ValueError, match="cannot be known from source"):
        m(support="witnessed", knowable_from_source=False,
          quoted_in_message=True, determinant="device time / enqueue time")


def test_a_knowable_magnitude_may_be_quoted():
    r = m(support="derived", knowable_from_source=True, quoted_in_message=True,
          determinant="the PCIe generation ratio")
    assert r.quoted_in_message


def test_span_is_a_ratio_not_a_difference():
    """These are all multiplicative factors, so 1-2x and 100-200x are the same
    width and a difference-based span would say otherwise."""
    assert Range(1, 2).span == 2.0
    assert Range(100, 200).span == 2.0


def test_range_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Range(5, 1)


def test_helpers_on_an_empty_registry():
    assert tally([]) == {"measured": 0, "witnessed": 0, "derived": 0,
                         "folklore": 0, "retracted": 0}
    assert unquotable([]) == []
    assert violations([]) == []
