"""drainlag -- how wrong is a timer that never drains the queue?

benchdoctor's BD001 says such a timer reads "usually 5-50x too fast". That
number came from my own experience, once, on one machine, and in twelve
projects of qualifying other people's figures nobody had ever checked it.

Measured across 39 cells on one consumer GPU: **0.98x to 595x.**

## What the measurement says

An un-drained timer measures the **host**: how long it took to enqueue the
launches. Add a `synchronize()` and it measures the **device**. So the error
factor is an identity, not a model:

    factor = device_us_per_launch / host_us_per_launch

Over the 20 cells whose factor exceeds 5x the worst residual is 18.2% and the
median is under 1%. Below 5x the residual grows -- which is the identity being
uninformative exactly where there is no error to predict.

Three consequences, none of which was in the rule:

**There is a floor below which the bug does not exist.** The host enqueues one
launch in 9.6 us. A kernel faster than that drains before the host can refill,
no backlog forms, and the un-drained number is *already correct*: 0.98-2.31x
across every cell whose per-launch pipeline ran at 9-15 us. Most benchmark
kernels in the corpus benchdoctor was audited against are in that regime.

(Those are not 9-15 us *kernels*, and this docstring said they were until the
graph measurement below showed otherwise: they are 1.9-9.3 us kernels whose
pipeline the host gated to 9-15 us. Same conclusion, wrong quantity named.)

**A longer loop is a safer loop.** The launch queue holds ~1024-2048 pending
launches; past that the host blocks and the measurement converges to correct.
Same kernel, no sync: K=64 is 5.83x too fast, K=16384 is 1.06x. And BD001 and
BD002 only fire when there is a loop.

**The ceiling is far above the claim.** A 23 ms kernel timed without a drain
read 595x too fast.

## The device: `magnitude`

Twelve tools qualified other people's numbers. This one qualifies the numbers
*my own diagnostics assert*. Six of them reach a reader; each now carries a
support status, and `Magnitude.__post_init__` refuses the dishonest
combinations -- `measured` without a dataset, `retracted` whose measurement
actually agreed, `folklore` that has been measured.

The invariant that changed the code: **`knowable_from_source` gates
`quoted_in_message`.** BD001's factor is device time over host enqueue time.
Neither appears in the text a linter reads, so the number may not appear in the
message however well measured it is.

Audit of all six:

    retracted   2   BD001's 5-50x (observed 0.98-595x) and BD002's 10-100x
                    (observed 1.03x warm, 1514x in a fresh process -- the band
                    contains neither)
    folklore    2   a rhetorical "3x off" and a blended "2-10x", both removed
    witnessed   1   BD010's 2-3x: n=1, my own incident, and the corpus audit
                    found the rule unreachable in public code
    derived     1   "Gen5 read as Gen1 is a 4x error" -- **wrong**. The spec
                    says 15.75x; 3.94x is Gen3 over Gen1. Checkable with
                    arithmetic, by anyone, at any time, and never checked

## And it closed one open verdict

corpusaudit left four findings `unclear`, each with the measurement that would
settle it. One of those measurements runs here:
`torch.load(mmap=True, map_location='cuda')` timed with and without a drain
came out at 0.991 -- the copy blocks the host, the un-drained number was
already right, and that verdict resolves to `false_positive`.

The flashinfer pair stays `unclear`. What ran here is the *shape* (small H2D
copies plus one small kernel, 1.17x), not flashinfer, and a reconstruction is
not the thing.

Zero dependencies for reading the results. A CUDA device for reproducing them.
"""

__version__ = "0.1.0"

from .claims import PCIE_MB_S_PER_LANE, REGISTRY, pcie_generation_error
from .law import INFORMATIVE_RATIO, Cell, Fit, crossover_kernel_us, knee
from .magnitude import SUPPORT, Magnitude, Range, tally, unquotable

__all__ = [
    "INFORMATIVE_RATIO",
    "PCIE_MB_S_PER_LANE",
    "REGISTRY",
    "SUPPORT",
    "Cell",
    "Fit",
    "Magnitude",
    "Range",
    "crossover_kernel_us",
    "knee",
    "pcie_generation_error",
    "tally",
    "unquotable",
]
