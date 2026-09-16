"""What an un-drained timer actually measures, and therefore how wrong it is.

## The identity

A timed region that launches K kernels and never drains the queue measures the
**host** side: how long it took to enqueue K launches. Add a `synchronize()`
and it measures the **device** side: how long the kernels took. So

    ratio = drained / undrained = device_us_per_launch / host_us_per_launch

is not a model with parameters. It is what those two clocks are for.

Measured across 39 cells on one consumer GPU, for every cell whose ratio
exceeds 5x the residual is under 19% and usually under 3%. Below that the
residual grows to 50% -- and that is not a failure, it is the identity being
uninformative exactly where there is no error to predict: 1.63 versus 1.14 is
a 43% residual between two numbers that both mean "about right".

## Why the factor is unknowable from source

`device_us_per_launch` is a property of the kernel and the card.
`host_us_per_launch` is a property of the framework, the OS, and -- past the
launch queue's depth -- of the kernel again, because the host starts blocking.

Neither appears in the text a linter reads. A rule can see that a region
launches work and never drains. It cannot see whether that costs you 2% or
59400%. Both were measured here.

## The two regimes of the host term

    enqueue-bound   K below the launch queue depth. `host_us_per_launch` is a
                    constant ~8-9 us: PyTorch dispatch plus the driver enqueue.
                    The backlog grows and the error is device/enqueue, which
                    ran from 1.0x to 595x.
    blocked         K past the depth (~1024-2048 launches, measured). The queue
                    is full, the host waits on the device for every further
                    launch, `host_us_per_launch` rises toward the device time,
                    and the ratio collapses toward 1.

The consequence is the one that reverses the intuition a linter encourages:
**a longer loop makes the missing sync matter less.** Same kernel, no sync,
64 iterations: 5.83x too fast. 16384 iterations: 1.06x. And benchdoctor's
BD001/BD002 only fire when there is a loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Ratios at or above this are where the identity is worth quoting. Below it both
# sides are "about 1" and the relative residual says more about noise than fit.
INFORMATIVE_RATIO = 5.0

# The drained wall clock and the CUDA-event clock are both supposed to be the
# device time, so their agreement is the cross-check that the comparison is
# measuring what it claims. Below this duration a `synchronize()` costs a few
# microseconds and event records land a few microseconds apart, which is most of
# a 50 us measurement -- the disagreement there is the instruments, not the
# quantity. Above it the two agree to 3.2% across 23 cells.
CROSSCHECK_MIN_MS = 1.0


@dataclass(frozen=True)
class Cell:
    """One (kernel size, loop count) measurement."""

    n: int
    k: int
    kernel_us: float           # device time per launch, from CUDA events
    undrained_ms: float
    drained_ms: float
    events_ms: float
    ratio: float

    @property
    def host_us_per_launch(self) -> float:
        """What the un-drained clock actually measured, per launch."""
        return self.undrained_ms * 1000.0 / self.k

    @property
    def predicted_ratio(self) -> float:
        return max(1.0, self.kernel_us / self.host_us_per_launch)

    @property
    def residual(self) -> float:
        return abs(self.predicted_ratio - self.ratio) / self.ratio

    @property
    def informative(self) -> bool:
        return self.ratio >= INFORMATIVE_RATIO

    @property
    def device_time_check(self) -> float:
        """drained vs CUDA events -- they should agree; a cross-check, not a result."""
        return self.drained_ms / self.events_ms if self.events_ms else float("nan")


@dataclass(frozen=True)
class Fit:
    cells: List[Cell]

    @property
    def informative(self) -> List[Cell]:
        return [c for c in self.cells if c.informative]

    @property
    def uninformative(self) -> List[Cell]:
        return [c for c in self.cells if not c.informative]

    def worst_residual(self, informative_only: bool = True) -> float:
        pool = self.informative if informative_only else self.cells
        return max((c.residual for c in pool), default=0.0)

    @property
    def ratio_range(self) -> Dict[str, float]:
        rs = [c.ratio for c in self.cells]
        return {"min": min(rs), "max": max(rs)}

    def device_time_disagreement(self, min_ms: float = CROSSCHECK_MIN_MS) -> float:
        """How far `drained` drifts from the CUDA-event time, on cells long
        enough for the answer to be about the quantity rather than the clocks.

        Both are supposed to be the device time. If they disagreed materially
        the whole comparison would be measuring something else, so this is
        reported before any ratio is.

        Passing `min_ms=0` gives 15.2%, and every contributor to that is a
        sub-0.2 ms cell where a few microseconds of synchronize overhead is a
        large fraction of the measurement. Quoting that number without the
        qualifier would be alarming and uninformative; hiding it would be
        worse, so both are available and the default is stated.
        """
        pool = [c for c in self.cells if c.drained_ms >= min_ms]
        if not pool:
            return 0.0
        return max(abs(c.device_time_check - 1.0) for c in pool)

    def crosscheck_cells(self, min_ms: float = CROSSCHECK_MIN_MS) -> int:
        return sum(1 for c in self.cells if c.drained_ms >= min_ms)


def host_regime(cell: Cell, enqueue_us: float, tol: float = 2.0) -> str:
    """`enqueue-bound` while the queue has room, `blocked` once it does not."""
    return "blocked" if cell.host_us_per_launch > tol * enqueue_us else "enqueue-bound"


def crossover_kernel_us(enqueue_us: float) -> float:
    """Kernel duration below which there is no error to make.

    If the device drains faster than the host can enqueue, no backlog forms and
    the un-drained number is already the right one. That is why BD001 is
    harmless on the small-kernel benchmarks that make up most of the corpus it
    was audited against.
    """
    return enqueue_us


def saturation_curve(rows: List[dict]) -> List[dict]:
    """The K-sweep past the queue depth, annotated with the collapse."""
    out = []
    for r in rows:
        out.append({**r, "excess_over_correct": r["ratio"] - 1.0})
    return out


def knee(rows: List[dict], enqueue_us: float, tol: float = 2.0) -> Optional[int]:
    """Smallest K in the sweep where the host is already blocking."""
    for r in sorted(rows, key=lambda r: r["k"]):
        if r["host_us_per_launch"] > tol * enqueue_us:
            return r["k"]
    return None


# --------------------------------------------------------------------------
# The same constant, read a third way: what the launch gap is worth
#
# `sweep.py` measures how wrong an un-drained timer is. The quantity that makes
# it wrong -- the host's cost to enqueue -- is not only a measurement artefact,
# it is time the device spends idle. CUDA graphs remove it.
#
# And measuring that turned up a flaw in this repository's own earlier numbers.
# `sweep.py` calls its event-measured quantity `kernel_us`, but events around an
# *eager* loop measure the pipeline rate: the kernel plus the idle the device
# spends waiting for the next submission. A graph replay submits once, so
# `graph_ms / K` is much closer to the kernel's own duration -- and at N=128 it
# is 1.9 us where the eager event timing said 9.6.
#
# The conclusions in sweep.py survive (for the device-bound cells the gap is
# ~2.4 us against 55-2880 us, so the two agree to a few percent). The *wording*
# did not: "kernels from 9 to 15 us" were kernels whose pipeline ran at that
# rate. They are 1.9-9.3 us kernels.

# Two measured bands, not a model. Where the kernel is shorter than the host's
# enqueue cost the device is left waiting and the gap tracks that cost; where it
# is longer the host stays ahead and the gap settles on a floor.
#
#   kernel < enqueue (1.9-4.0 us kernels)     gap 6.3-8.2 us
#   kernel > enqueue (8.3-372 us kernels)     gap 2.1-3.1 us
#
# A predictive form -- max(floor, enqueue - kernel) -- was written and then not
# shipped: its worst residual is 136%, concentrated in the K=8 cells where the
# host timing is 8 launches wide and unreliable, and in the 2.9 ms kernels where
# a 5 us gap is 0.17% of the measurement and at the edge of what a wall clock
# resolves. The bands are what was measured; a model with that residual would
# be decoration.
GAP_BAND_HOST_STARVED = (6.3, 8.2)
GAP_BAND_HOST_KEEPS_UP = (2.1, 3.1)


@dataclass(frozen=True)
class GraphCell:
    """One (kernel size, loop count) eager-vs-graph comparison."""

    n: int
    k: int
    host_us_per_launch: float
    event_us_per_launch: float      # eager event timing: the pipeline rate
    true_kernel_us: float           # graph replay per launch: the kernel
    gap_us: float                   # what the device idles between launches
    eager_ms: float
    graph_ms: float
    speedup: float
    eager_drift: float
    capture_ms: float
    breakeven_replays: Optional[float]

    @property
    def identity(self) -> float:
        """speedup = (kernel + gap) / kernel. Algebra, not a model."""
        return 1.0 + self.gap_us / self.true_kernel_us

    @property
    def regime(self) -> str:
        """`host-starved` while the kernel is shorter than the enqueue cost."""
        return ("host-starved" if self.true_kernel_us < self.host_us_per_launch
                else "host-keeps-up")

    @property
    def gap_in_band(self) -> bool:
        """Is the measured gap inside the band its regime was measured at?

        Not a prediction -- a consistency check on the two bands. The 2.9 ms
        cells sit outside on the high side (gap ~5 us against a 2.1-3.1 band)
        and that is reported rather than absorbed: there, the gap is 0.17% of
        the measurement and a wall clock is not the instrument for it.
        """
        lo, hi = (GAP_BAND_HOST_STARVED if self.regime == "host-starved"
                  else GAP_BAND_HOST_KEEPS_UP)
        return lo <= self.gap_us <= hi

    @property
    def gap_fraction_of_measurement(self) -> float:
        """How much of the eager per-launch time the gap is. Below ~1% a wall
        clock cannot resolve it and `gap_in_band` should not be read as a fault.
        """
        return self.gap_us / (self.true_kernel_us + self.gap_us)

    @property
    def event_timing_overstates_kernel_by(self) -> float:
        """How far eager event timing is from the kernel's own duration.

        1.0 means they agree. At N=128 it is about 5x, which is why the model
        this measurement was written with -- `max(1, host/event_us)` -- predicted
        no speedup where the truth was 5.4x.
        """
        return self.event_us_per_launch / self.true_kernel_us


def graph_worth_it(cells: List[GraphCell], threshold: float = 1.10) -> List[GraphCell]:
    return [c for c in cells if c.speedup >= threshold]
