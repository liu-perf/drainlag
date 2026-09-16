"""The ratchet: benchdoctor may not assert a magnitude that is not registered.

Two halves, and the second one is the inversion that makes this work.

**Messages.** Every numeric magnitude in a user-visible `Finding(message=...)`
must be registered here, must be marked `quoted_in_message`, and must be
`knowable_from_source`. On the current code that set is empty: all six
magnitudes turned out to depend on the machine rather than the source, so none
of them may be in a message, and none is.

**Docstrings.** Every `claimed_text` in the registry must still appear
*somewhere* in benchdoctor's source. That is deliberately the opposite of what
a normal test would assert. The retracted numbers are the record: deleting
"5-50x" along with the retraction note would erase the evidence that anyone
ever claimed it, and a silent edit is how a wrong number becomes a number
nobody remembers being wrong.

Both halves skip when benchdoctor is not importable, which is CI -- and
`test_registry_is_internally_consistent` runs regardless, because an audit
whose enforcement depends on the audited tool being on the path stops
enforcing the moment CI changes.
"""

import pathlib
import re

import pytest

from drainlag import claims as claims_mod
from drainlag.magnitude import SUPPORT, tally, unquotable, violations

MAGNITUDE = re.compile(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*x"
                       r"|\b\d+(?:\.\d+)?\s*x\b"
                       r"|\d+(?:\.\d+)?%")


def _benchdoctor_rules_dir():
    try:
        import benchdoctor.cli as cli
    except ImportError:
        pytest.skip("benchdoctor not importable; the registry stands alone")
    return pathlib.Path(cli.__file__).parent / "rules"


def _message_blocks(text):
    """The `message=(...)` payloads -- what a user of the tool reads."""
    return [m.group(1) for m in re.finditer(r"message=\((.*?)\n\s*\),", text, re.S)]


# --------------------------------------------------------------- always runs
def test_registry_is_internally_consistent():
    for m in claims_mod.REGISTRY:
        assert m.support in SUPPORT
        if m.support in ("measured", "retracted"):
            assert m.observed is not None and m.dataset


def test_no_registered_magnitude_is_quoted_where_it_may_not_be():
    assert violations(claims_mod.REGISTRY) == []


def test_every_magnitude_here_is_unknowable_except_the_pcie_one():
    """The general result, asserted.

    Five of six depend on the machine. The one exception is BD020's, which is a
    ratio between two published PCIe generations -- arithmetic, not a
    measurement, and therefore the only magnitude in this tool that a source
    reader could in principle verify.
    """
    knowable = [m.rule for m in claims_mod.REGISTRY if m.knowable_from_source]
    assert knowable == ["BD020"]
    assert len(unquotable(claims_mod.REGISTRY)) == 5


def test_the_pcie_correction_is_arithmetic_anyone_can_redo():
    assert claims_mod.pcie_generation_error(1, 5) == pytest.approx(15.754, abs=0.01)
    # and the number that was in the docstring is Gen3 over Gen1, not Gen5
    assert claims_mod.pcie_generation_error(1, 3) == pytest.approx(3.938, abs=0.01)


def test_the_two_retractions_understate_the_spread_not_just_the_value():
    """Why the retraction criterion had to be about spread.

    5-50x sits inside the observed 0.98-595x, so a non-containment test called
    it a confirmation. The claim is not wrong about any single case; it is
    wrong about being a range.
    """
    ret = [m for m in claims_mod.REGISTRY if m.support == "retracted"]
    assert len(ret) == 2
    for m in ret:
        assert m.observed.contains(m.claimed), m.rule
        assert m.understates_spread_by > 50, m.rule


def test_tally_of_the_audit():
    assert tally(claims_mod.REGISTRY) == {
        "measured": 0, "witnessed": 1, "derived": 1,
        "folklore": 2, "retracted": 2}


# ------------------------------------------------- needs benchdoctor on path
def test_no_unregistered_magnitude_reaches_a_user_visible_message():
    rules = _benchdoctor_rules_dir()
    known = set(claims_mod.claimed_texts()) | set(claims_mod.exempt_texts())
    unregistered = []
    for path in sorted(rules.rglob("*.py")):
        for blob in _message_blocks(path.read_text(encoding="utf-8")):
            for hit in MAGNITUDE.findall(blob):
                if hit.replace(" ", "") not in {k.replace(" ", "") for k in known}:
                    unregistered.append((path.name, hit))
    assert not unregistered, (
        f"magnitudes in messages with no registry entry: {unregistered}. "
        "Register it with a support status, or take it out of the message.")


def test_retraction_notes_are_still_in_the_source():
    """The inversion: the wrong numbers must NOT be gone.

    Each retracted or removed claim has to remain findable in benchdoctor's
    text, inside the note that retracts it. Deleting it would erase the record
    that it was ever asserted.
    """
    rules = _benchdoctor_rules_dir()
    corpus = "\n".join(p.read_text(encoding="utf-8")
                       for p in sorted(rules.rglob("*.py")))
    missing = [m.claimed_text for m in claims_mod.REGISTRY
               if m.support in ("retracted", "folklore", "witnessed")
               and m.claimed_text not in corpus]
    assert not missing, (
        f"{missing} no longer appears anywhere in benchdoctor. A retraction "
        "whose claim has been deleted is not a retraction, it is a quiet edit.")


def test_the_corrected_pcie_number_replaced_the_wrong_one():
    rules = _benchdoctor_rules_dir()
    pcie = (rules / "pcie.py").read_text(encoding="utf-8")
    assert "15.75x" in pcie
    # the old 4x must survive only as the explained mistake, i.e. next to Gen3
    assert "3.94x is Gen3 over Gen1" in pcie
