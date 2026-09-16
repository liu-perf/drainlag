"""★ The device: a claimed magnitude, and what entitles it to be quoted.

Twelve tools before this one qualified other people's numbers. Where a figure
came from, what arithmetic a column supports, how far a model sat from the
machine, whether an output had been validated, whether a *rule* had ever been
right. Meanwhile every one of those tools printed messages like this:

    "this times the launch queue, not the kernel -- usually 5-50x too fast"
    "one-time costs that can be 10-100x a steady-state iteration"
    "GFLOPs come out 2-3x low"
    "Gen5 read as Gen1 is a 4x error"

Four numbers a reader will act on. None of them had ever been measured, and one
of them is arithmetically wrong.

So each asserted magnitude carries a `Magnitude`: what was claimed, what
supports it, and -- the part that turned out to matter most -- **what the
magnitude actually depends on.**

## The five supports

    measured    a dataset in this repository backs it; carries the observed range
    witnessed   seen once, on one machine, in one incident that is named. n=1
    derived     follows from a published specification; checkable with arithmetic
    folklore    believed, widely repeated, no evidence here
    retracted   was asserted, then measured, and the measurement contradicted it

## The teeth

Three invariants, all raising:

1. `measured` and `retracted` require an observed range and a named dataset.
   A claim cannot describe itself as measured without pointing at the
   measurement.
2. `retracted` requires the claim to actually fail: either it does not overlap
   the observation, **or** it understates the observed spread by at least
   `SPAN_TOLERANCE`. The first version of this invariant tested only
   non-containment, and it rejected both of the real retractions in
   `claims.py` -- 5-50x sits comfortably inside an observed 0.98-595x, and
   10-100x inside 1.03-1514x. Neither claim is wrong about any single case.
   Both are wrong about being a range: a factor-of-10 window presented as the
   description of a factor-of-600 spread. Spread is compared as a ratio, not a
   difference, because every magnitude here is multiplicative.
3. `folklore` forbids an observed range. If you measured it, it is no longer
   folklore -- one way or the other.

And the one that changed the code: **`knowable_from_source` gates
`quoted_in_message`.** BD001's error factor is device time over host enqueue
time, neither of which appears in the source a linter reads. So the number may
not be in the message, no matter how well measured it is. A test enforces it.

That is the general result this repository exists for. A static checker can
identify a shape. It cannot know the magnitude, because the magnitude is a
property of the machine, and quoting one anyway is the most reasonable-looking
way to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SUPPORT = ("measured", "witnessed", "derived", "folklore", "retracted")
NEEDS_OBSERVATION = ("measured", "retracted")

# A claim earns `retracted` either by missing the observation entirely, or by
# understating its spread by at least this factor. The second test exists
# because the first one was not enough, and finding that out is the only
# interesting thing that happened while writing this module -- see the note on
# `retracted` below.
SPAN_TOLERANCE = 3.0


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"Range({self.lo}, {self.hi}): hi below lo")

    def contains(self, other: "Range") -> bool:
        return self.lo <= other.lo and other.hi <= self.hi

    def overlaps(self, other: "Range") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    @property
    def span(self) -> float:
        """hi/lo. A ratio, because these are all multiplicative factors."""
        return self.hi / self.lo if self.lo > 0 else float("inf")

    def __str__(self) -> str:
        if self.lo == self.hi:
            return f"{self.lo:g}x"
        return f"{self.lo:g}-{self.hi:g}x"


@dataclass(frozen=True)
class Magnitude:
    """One number a diagnostic asserts, and its warrant for asserting it."""

    rule: str
    claimed_text: str          # exactly as it appeared, e.g. "5-50x"
    claimed: Range
    support: str
    where: str                 # message | docstring
    # What the magnitude is actually a function of. Empty for `folklore`,
    # because not knowing this is what folklore means.
    determinant: str = ""
    # False when the determinant is not visible in the source a linter reads.
    knowable_from_source: bool = True
    # Whether the number appears in text a user of the tool sees.
    quoted_in_message: bool = False
    observed: Optional[Range] = None
    dataset: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.support not in SUPPORT:
            raise ValueError(f"{self.rule}: unknown support {self.support!r}")
        if self.support in NEEDS_OBSERVATION:
            if self.observed is None or not self.dataset:
                raise ValueError(
                    f"{self.rule}: support {self.support!r} requires an observed "
                    "range and a dataset. A claim cannot call itself measured "
                    "without pointing at the measurement.")
        if self.support == "retracted":
            missed = not self.observed.overlaps(self.claimed)
            narrow = self.understates_spread_by >= SPAN_TOLERANCE
            if not (missed or narrow):
                raise ValueError(
                    f"{self.rule}: marked retracted, but the claim "
                    f"{self.claimed} overlaps the observed {self.observed} and "
                    f"understates its spread by only "
                    f"{self.understates_spread_by:.1f}x. That is a "
                    "confirmation; calling it a retraction is theatre.")
        if self.support == "folklore" and self.observed is not None:
            raise ValueError(
                f"{self.rule}: folklore with an observed range. Once measured it "
                "is `measured` or `retracted`, not folklore.")
        if self.quoted_in_message and not self.knowable_from_source:
            raise ValueError(
                f"{self.rule}: quotes a magnitude in a user-visible message "
                f"that cannot be known from source ({self.determinant}). Say "
                "the mechanism, not the number.")

    @property
    def understates_spread_by(self) -> float:
        """How much narrower the claim is than what was observed.

        1.0 means the claim describes the spread. 60 means it offered a
        factor-of-10 window for something that ranged over a factor of 600 --
        which is BD001, and is the more interesting way for a magnitude in a
        diagnostic to be wrong than simply naming the wrong number.
        """
        if self.observed is None:
            return 1.0
        c = self.claimed.span
        return self.observed.span / c if c > 0 else float("inf")

    @property
    def wrong_by(self) -> Optional[float]:
        """How far the claim's nearest edge is from the observed range.

        `None` when nothing was observed. 1.0 means the claim touches the
        observed range; larger means it does not reach it.
        """
        if self.observed is None:
            return None
        if self.observed.overlaps(self.claimed):
            return 1.0
        if self.claimed.hi < self.observed.lo:
            return self.observed.lo / max(self.claimed.hi, 1e-12)
        return self.claimed.lo / max(self.observed.hi, 1e-12)

    def describe(self) -> str:
        head = f"{self.rule}: [{self.support}] claims {self.claimed_text}"
        if self.observed is not None:
            head += f", observed {self.observed}"
            w = self.wrong_by
            if w and w > 1.0:
                head += f" (claim misses by {w:.0f}x)"
            s = self.understates_spread_by
            if s >= SPAN_TOLERANCE:
                head += f" (understates the spread {s:.0f}x)"
        if self.determinant:
            head += f"; depends on {self.determinant}"
        if not self.knowable_from_source:
            head += " -- NOT knowable from source"
        return head


def tally(mags: List[Magnitude]) -> Dict[str, int]:
    return {s: sum(1 for m in mags if m.support == s) for s in SUPPORT}


def unquotable(mags: List[Magnitude]) -> List[Magnitude]:
    """Magnitudes a message must not contain, because source cannot know them."""
    return [m for m in mags if not m.knowable_from_source]


def violations(mags: List[Magnitude]) -> List[Tuple[str, str]]:
    """Registered magnitudes still quoted where they may not be.

    Empty here, and it has to stay empty: the constructor already refuses the
    combination, so a violation can only appear by editing the registry and the
    rule text out of step. This function exists so the CI step reads as a
    check rather than as a hope.
    """
    return [(m.rule, m.claimed_text) for m in mags
            if m.quoted_in_message and not m.knowable_from_source]
