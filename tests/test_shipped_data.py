"""Assertions about the measurements actually committed here.

These are the tests that fail if somebody re-runs the sweep on a different
machine and forgets that the prose quotes specific numbers, or edits a dataset
without re-deriving the conclusions from it.
"""

import pytest

from drainlag import data as data_mod
from drainlag import law as law_mod
from drainlag import report


@pytest.fixture(scope="module")
def doc():
    return data_mod.load()


def test_the_sweep_is_the_size_the_prose_says(doc):
    f = data_mod.fit(doc)
    assert len(f.cells) == 39
    assert {c.n for c in f.cells} == {128, 256, 512, 1024, 2048, 4096, 8192}


def test_the_factor_spans_three_orders_of_magnitude(doc):
    """The headline: it is not 5-50x and it is not one number."""
    r = data_mod.fit(doc).ratio_range
    assert r["min"] < 1.0
    assert r["max"] > 500
    assert not (5 <= r["min"] and r["max"] <= 50)


def test_the_identity_holds_where_there_is_an_error_to_predict(doc):
    f = data_mod.fit(doc)
    assert len(f.informative) >= 15
    assert f.worst_residual() < 0.20
    # and the median is far tighter than the worst
    resids = sorted(c.residual for c in f.informative)
    assert resids[len(resids) // 2] < 0.05


def test_the_identity_is_uninformative_where_there_is_no_error(doc):
    """Stated as a property, not hidden.

    Below a factor of 5 the residual grows to about half. Both sides are
    "about 1" there, and a fit statistic over that region measures noise.
    """
    f = data_mod.fit(doc)
    assert f.worst_residual(informative_only=False) > 0.30
    assert all(c.ratio < law_mod.INFORMATIVE_RATIO for c in f.uninformative)


def test_the_two_device_clocks_agree_on_measurements_long_enough_to_trust(doc):
    f = data_mod.fit(doc)
    assert f.crosscheck_cells() >= 20
    assert f.device_time_disagreement() < 0.05
    # and the number without the qualifier, so it is on the record
    assert f.device_time_disagreement(0) > 0.10


def test_there_is_a_floor_below_which_the_bug_does_not_exist(doc):
    """Kernels faster than the host enqueue cost cannot build a backlog."""
    enq = data_mod.enqueue_us(doc)
    assert 5 < enq < 20
    small = [c for c in data_mod.fit(doc).cells if c.kernel_us < 1.5 * enq]
    assert len(small) >= 15
    assert max(c.ratio for c in small) < 2.5


def test_a_longer_loop_makes_the_missing_drain_matter_less(doc):
    """The finding that reverses what a loop-triggered rule implies."""
    rows = sorted(data_mod.saturation_rows(doc), key=lambda r: r["k"])
    assert rows[0]["ratio"] > 5
    assert rows[-1]["ratio"] < 1.15
    assert rows[0]["k"] < rows[-1]["k"]
    # monotone once past the knee
    enq = data_mod.enqueue_us(doc)
    past = [r for r in rows if r["host_us_per_launch"] > 2 * enq]
    assert [r["ratio"] for r in past] == sorted((r["ratio"] for r in past),
                                                reverse=True)
    assert law_mod.knee(rows, enq) in (1024, 2048)


def test_the_warmup_claim_missed_in_both_directions(doc):
    warm = [x["ratio"] for x in data_mod.warmup(doc)["cublas_first_call_of_a_new_shape"]]
    cold = data_mod.context(doc)["first_matmul_over_steady"]
    assert max(warm) < 10        # below the claimed 10-100x band
    assert cold > 100            # above it
    assert cold > 1000


def test_contamination_is_visible_and_the_drain_removes_it(doc):
    c = data_mod.contamination(doc)
    assert c["first_vs_last_no_drain"] > 1.20
    assert c["first_vs_last_drained"] < 1.10


def test_the_torch_load_verdict_is_settled(doc):
    """corpusaudit's `unclear` on server.py:198 resolves to false_positive."""
    v = data_mod.verdicts(doc)["torch_load"]
    assert v["with_sync_over_no_sync"] < 1.05
    assert "already right" in v["verdict"]


def test_the_plan_shape_result_does_not_claim_to_settle_flashinfer(doc):
    """A reconstruction is not the thing, and the data says so in the file."""
    v = data_mod.verdicts(doc)["plan_shape"]
    assert 1.0 < v["ratio"] < 1.5
    assert "not flashinfer" in v["note"]


def test_the_report_leads_with_the_crossover_not_the_factor(doc):
    """Ordering is a claim about how a reader will misread the alternative."""
    lines = report.lines()
    kinds = [ln.split(":")[0] for ln in lines]
    assert kinds.index("crossover") < kinds.index("factor")


def test_the_report_states_the_environment_first(doc):
    assert report.lines()[0].startswith("machine:")
    assert "the ratios transfer, the absolute times do not" in report.lines()[0]


# ------------------------------------------------------------------- graphs
def test_the_speedup_relation_is_algebra_not_a_model(doc):
    """speedup = (kernel + gap) / kernel, to within floating point.

    Asserted so nobody mistakes it for a fitted result. The content of the
    finding is entirely in what `gap_us` turns out to be.
    """
    for c in data_mod.graph_cells(doc):
        assert c.identity == pytest.approx(c.speedup, rel=1e-9)


def test_graphs_are_worth_a_lot_below_the_crossover_and_nothing_above(doc):
    cells = data_mod.graph_cells(doc)
    tiny = [c for c in cells if c.true_kernel_us < 5]
    big = [c for c in cells if c.true_kernel_us > 300]
    assert min(c.speedup for c in tiny) > 2.5
    assert max(c.speedup for c in tiny) > 5.0
    assert max(c.speedup for c in big) < 1.05


def test_eager_event_timing_overstates_the_kernel(doc):
    """The actionable finding, and the reason the first model failed.

    CUDA events around an eager loop measure the pipeline rate. Using that
    number to decide whether launch overhead matters gives the wrong answer by
    a factor of five on exactly the kernels where it matters most.
    """
    cells = data_mod.graph_cells(doc)
    worst = max(c.event_timing_overstates_kernel_by for c in cells)
    assert worst > 5.0
    # and it is honest where the device really is the bottleneck
    big = [c for c in cells if c.true_kernel_us > 300]
    assert max(c.event_timing_overstates_kernel_by for c in big) < 1.10


def test_the_gap_bands_hold_except_where_the_data_says_they_do_not(doc):
    """18 of 22 in band, and the four exceptions are explained rather than fitted.

    Two are the crossover cell itself (kernel ~8.3 us against a ~10 us enqueue,
    so the regime label is at its boundary). Two are 2.9 ms kernels where the
    gap is 0.17% of the measurement -- below what a wall clock resolves.
    """
    cells = data_mod.graph_cells(doc)
    outside = [c for c in cells if not c.gap_in_band]
    assert len(cells) - len(outside) == 18
    for c in outside:
        near_crossover = 0.7 < c.true_kernel_us / c.host_us_per_launch < 1.3
        unresolvable = c.gap_fraction_of_measurement < 0.01
        assert near_crossover or unresolvable, (c.n, c.k)


def test_the_contamination_check_is_present_on_every_cell(doc):
    """Eager is measured before capture and again after, and both are shipped."""
    for c in data_mod.graph_cells(doc):
        assert c.eager_drift is not None
        assert 0.85 < c.eager_drift < 1.30


def test_report_ties_the_graph_result_to_the_same_constant(doc):
    text = report.text()
    assert "read a third way" in text
    assert "measure the pipeline rate, not the kernel" in text
