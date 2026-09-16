"""The registry: every numeric magnitude benchdoctor asserts, and its support.

Six numbers reach a reader, from messages and from the docstrings a maintainer
reads before changing a rule. This file is the audit of all six, and
`tests/test_registry.py` asserts the audit is complete: a number in a rule
module that is not registered here fails the build.

That ratchet is the point. It costs nothing to type "5-50x" into a diagnostic,
the sentence reads well, and nobody asks. Making it cost a registry entry with
a support status is the smallest change that makes the question unavoidable.

## The result

    retracted   2     measured for the first time, and both claims failed
    folklore    2     believed, never checked, now removed
    witnessed   1     n=1, my own incident, unlikely ever to be measurable
    derived     1     arithmetic -- and the arithmetic was wrong

Nothing is `measured`, and that is not an oversight: a magnitude only earns
`measured` if the claim survived the measurement. Both of the two that were
measured here failed, in opposite directions:

* BD001 claimed 5-50x. Observed **0.98x to 595x**, and the factor is
  `device_us_per_launch / host_us_per_launch`, neither of which a linter can
  see. Retracted from the message.
* BD002 claimed 10-100x. On a warm process the removable warmup cost is
  **1.03-2.36x**; in a fresh process the first kernel is **1514x**. The claimed
  band contains neither. Retracted.

And one was simply wrong on arithmetic, checkable from the PCIe specification
with no machine at all: "Gen5 read as Gen1 is a 4x error" is **15.75x**. Gen1
is 250 MB/s per lane (2.5 GT/s, 8b/10b); Gen5 is 3938 MB/s per lane (32 GT/s,
128b/130b). The 4x is Gen3-over-Gen1. That number had been sitting in a
docstring about *getting denominators wrong*.
"""

from __future__ import annotations

from typing import List

from .magnitude import Magnitude, Range

# PCIe per-lane throughput, MB/s, from the base specification. Encoded rather
# than asserted so the correction below is arithmetic a reader can redo.
PCIE_MB_S_PER_LANE = {1: 250.0, 2: 500.0, 3: 984.6, 4: 1969.2, 5: 3938.5,
                      6: 7877.0}


def pcie_generation_error(read_as: int, actually: int) -> float:
    """How far a denominator is off when the link generation is misread."""
    return PCIE_MB_S_PER_LANE[actually] / PCIE_MB_S_PER_LANE[read_as]


REGISTRY: List[Magnitude] = [
    Magnitude(
        rule="BD001",
        claimed_text="5-50x",
        claimed=Range(5, 50),
        support="retracted",
        where="docstring",
        determinant="device_us_per_launch / host_us_per_launch",
        knowable_from_source=False,
        quoted_in_message=False,
        observed=Range(0.98, 594.6),
        dataset="data/sweep.json",
        note="39 cells, per-launch pipeline from 9 us to 23 ms, loops from 1 to "
             "4096. Below the host enqueue cost (~9.6 us) there is no error to "
             "make: the device drains faster than the host can fill, so the "
             "un-drained number is already right (0.98-2.31x). Above it the "
             "factor is the "
             "ratio of the two per-launch costs and reached 595x. The claimed "
             "5-50x band corresponds to kernels of roughly 50-500 us -- one "
             "slice of the range, presented as the whole of it.",
    ),
    Magnitude(
        rule="BD002",
        claimed_text="10-100x",
        claimed=Range(10, 100),
        support="retracted",
        where="docstring",
        determinant="whether the CUDA context and cuBLAS are already loaded in "
                    "this process",
        knowable_from_source=False,
        quoted_in_message=False,
        observed=Range(1.03, 1514.0),
        dataset="data/claims.json, data/context.json",
        note="Two different costs wear one number. The cost a warmup loop inside "
             "a benchmark function can actually remove -- first call of a new "
             "shape, warm process -- is 1.03-2.36x. The cost that really is "
             "enormous is the first GPU kernel in a *fresh process*: 1514x a "
             "steady-state matmul, because cuBLAS load and kernel-module load "
             "are deferred until then. That one is per-process and is paid "
             "before the inspected function is ever called. So the folklore "
             "number is not invented, it is attached to the wrong thing -- and "
             "source cannot tell which of the two you are about to pay.",
    ),
    Magnitude(
        rule="BD010",
        claimed_text="2-3x",
        claimed=Range(2, 3),
        support="witnessed",
        where="message",
        determinant="the profiler's verification cost relative to the kernel's",
        knowable_from_source=False,
        quoted_in_message=False,
        observed=None,
        note="n=1. One afternoon, one GEMM sweep, `cutlass_profiler` without "
             "`--verification-enabled=false`. I did not record the numbers and "
             "cannot reproduce it here: there is no cutlass_profiler on this "
             "machine, and the corpus audit found the rule is `unreachable` -- "
             "zero `cutlass_profiler` command lines in 166 public shell "
             "scripts, including NVIDIA/cutlass's own. So this magnitude is "
             "unlikely to ever be measured from public code. The message now "
             "says verification runs inside the timed region and the reported "
             "GFLOPs come out low, without a factor.",
    ),
    Magnitude(
        rule="BD020",
        claimed_text="4x",
        claimed=Range(4, 4),
        support="derived",
        where="docstring",
        determinant="the ratio of per-lane throughput between the two PCIe "
                    "generations",
        knowable_from_source=True,
        quoted_in_message=False,
        observed=None,
        note="WRONG AS WRITTEN, and wrong in the direction that understates the "
             "trap. The docstring said 'Gen5 read as Gen1 is a 4x error'. From "
             "the base spec that is 3938.5/250 = 15.75x; 3.94x is Gen3 over "
             "Gen1. No machine was needed to catch this and none had. Corrected "
             "in place, and `pcie_generation_error()` here computes it so the "
             "next person can redo the arithmetic instead of trusting prose.",
    ),
    Magnitude(
        rule="BD003",
        claimed_text="3x",
        claimed=Range(3, 3),
        support="folklore",
        where="docstring",
        determinant="",
        knowable_from_source=False,
        quoted_in_message=False,
        observed=None,
        note="'this measurement, or a fluke 3x off, I cannot tell you which'. "
             "Purely rhetorical -- 3 was chosen because it sounds like a lot. "
             "The rule's point (one sample has no variance estimate) does not "
             "need a number and is stronger without one. Removed from the "
             "docstring; kept here so the removal is on the record rather than "
             "a silent edit.",
    ),
    Magnitude(
        rule="timing",
        claimed_text="2-10x",
        claimed=Range(2, 10),
        support="folklore",
        where="docstring",
        determinant="",
        knowable_from_source=False,
        quoted_in_message=False,
        observed=None,
        note="The timing module's own summary line: 'silently includes or "
             "excludes something that changes the number by 2-10x'. A blended "
             "guess across three rules whose real ranges are now known to be "
             "1.0-595x and 1.0-1514x. Superseded by the per-rule entries above "
             "and removed from the summary.",
    ),
]


# Numeric patterns in benchdoctor's rule text that are NOT magnitude claims.
# Named one by one with a reason, because an exemption list is exactly where a
# check like this quietly stops working. `drainlag claims` prints them.
EXEMPT = [
    ("BD021", "100%",
     "The value being clamped to, quoted from the code under analysis. BD021's "
     "whole subject is `min(x, 100)`; the 100 is the thing detected, not an "
     "assertion about how wrong anything is."),
]


def by_rule(rule: str) -> List[Magnitude]:
    return [m for m in REGISTRY if m.rule == rule]


def exempt_texts() -> List[str]:
    return sorted({text for _rule, text, _why in EXEMPT})


def claimed_texts() -> List[str]:
    """Every magnitude string the registry accounts for.

    `tests/test_registry.py` scans benchdoctor's rule modules for numeric
    magnitude patterns and requires each hit to appear here. That is the
    ratchet: a new number in a message needs an entry, and an entry needs a
    support status.
    """
    return sorted({m.claimed_text for m in REGISTRY})
