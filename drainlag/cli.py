"""`drainlag <subcommand>` -- read the measurements, not the machine.

    drainlag report            the whole thing as status lines
    drainlag law               the identity, cell by cell, with residuals
    drainlag saturation        why a longer loop is safer
    drainlag graphs            what the launch gap is worth
    drainlag claims            the six magnitudes and their support
    drainlag check             invariants, for CI

All of it reads `data/*.json`. Reproducing those needs a CUDA device and
`measure/*.py`; reading them needs nothing, which is deliberate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Optional

from . import claims as claims_mod
from . import data as data_mod
from . import law as law_mod
from . import report
from .magnitude import tally, unquotable, violations


def _root(args) -> Optional[pathlib.Path]:
    return pathlib.Path(args.root).resolve() if args.root else None


def cmd_report(args) -> int:
    if args.json:
        print(json.dumps(report.summary(_root(args)), indent=2,
                         ensure_ascii=False))
        return 0
    print(report.text(_root(args)))
    return 0


def cmd_law(args) -> int:
    doc = data_mod.load(_root(args))
    enq = data_mod.enqueue_us(doc)
    f = data_mod.fit(doc)
    print(f"host enqueue cost: {enq:.2f} us/launch")
    print(f"{'N':>6} {'kernel us':>10} {'K':>6} {'host us/L':>10} "
          f"{'measured':>9} {'identity':>9} {'resid':>7} {'regime':>14}")
    for c in sorted(f.cells, key=lambda c: (c.n, c.k)):
        mark = "" if c.informative else "  (uninformative)"
        print(f"{c.n:>6} {c.kernel_us:>10.1f} {c.k:>6} "
              f"{c.host_us_per_launch:>10.2f} {c.ratio:>9.2f} "
              f"{c.predicted_ratio:>9.2f} {c.residual * 100:>6.1f}% "
              f"{law_mod.host_regime(c, enq):>14}{mark}")
    print()
    print(f"worst residual where the factor exceeds "
          f"{law_mod.INFORMATIVE_RATIO:g}x: {f.worst_residual():.1%} "
          f"({len(f.informative)} cells)")
    print(f"worst residual over all {len(f.cells)} cells: "
          f"{f.worst_residual(informative_only=False):.1%}")
    return 0


def cmd_saturation(args) -> int:
    doc = data_mod.load(_root(args))
    enq = data_mod.enqueue_us(doc)
    s = doc["claims"]["saturation"]
    print(f"kernel: {s['device_us_per_launch']:.1f} us device time, "
          f"host enqueue {enq:.2f} us")
    print(f"{'K':>7} {'undrained ms':>13} {'drained ms':>11} {'factor':>8} "
          f"{'host us/L':>10} {'regime':>14}")
    for r in s["rows"]:
        reg = "blocked" if r["host_us_per_launch"] > 2 * enq else "enqueue-bound"
        print(f"{r['k']:>7} {r['undrained_ms']:>13.2f} {r['drained_ms']:>11.2f} "
              f"{r['ratio']:>8.2f} {r['host_us_per_launch']:>10.2f} {reg:>14}")
    print()
    print(f"the host starts blocking at K={law_mod.knee(s['rows'], enq)}; "
          "past that a longer loop is a *safer* loop")
    return 0


def cmd_graphs(args) -> int:
    doc = data_mod.load(_root(args))
    cells = data_mod.graph_cells(doc)
    print(f"{'N':>6} {'K':>5} {'host us':>8} {'event us':>9} {'true us':>8} "
          f"{'gap us':>7} {'speedup':>8} {'regime':>15} {'breakeven':>10}")
    for c in cells:
        be = c.breakeven_replays
        mark = "" if c.gap_in_band else "  *"
        print(f"{c.n:>6} {c.k:>5} {c.host_us_per_launch:>8.2f} "
              f"{c.event_us_per_launch:>9.1f} {c.true_kernel_us:>8.2f} "
              f"{c.gap_us:>7.2f} {c.speedup:>8.2f} {c.regime:>15} "
              f"{('%.1f' % be) if be else '-':>10}{mark}")
    print()
    print(f"gap bands: host-starved {law_mod.GAP_BAND_HOST_STARVED}, "
          f"host-keeps-up {law_mod.GAP_BAND_HOST_KEEPS_UP} us")
    outside = [c for c in cells if not c.gap_in_band]
    if outside:
        print(f"* {len(outside)} cell(s) outside their band:")
        for c in outside:
            print(f"    N={c.n} K={c.k}: gap is "
                  f"{c.gap_fraction_of_measurement:.2%} of the measurement")
    ov = max(cells, key=lambda c: c.event_timing_overstates_kernel_by)
    print()
    print(f"eager event timing overstates the kernel by up to "
          f"{ov.event_timing_overstates_kernel_by:.2f}x (N={ov.n}, K={ov.k}): "
          f"{ov.event_us_per_launch:.1f} us reported, "
          f"{ov.true_kernel_us:.2f} us actual")
    return 0


def cmd_claims(args) -> int:
    for m in claims_mod.REGISTRY:
        if args.support and m.support != args.support:
            continue
        print(m.describe())
        if m.dataset:
            print(f"    dataset: {m.dataset}")
        if m.note:
            for chunk in _wrap(m.note, 74):
                print(f"    {chunk}")
        print()
    t = tally(claims_mod.REGISTRY)
    print(", ".join(f"{k}={v}" for k, v in t.items() if v))
    return 0


def _wrap(text: str, width: int) -> List[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def cmd_check(args) -> int:
    root = _root(args)
    problems: List[str] = []
    try:
        doc = data_mod.load(root)
        print(f"data: ok ({len(doc)} files)")
    except Exception as exc:
        problems.append(f"data: {exc}")
        doc = None
    if doc is not None:
        f = data_mod.fit(doc)
        print(f"cells: ok ({len(f.cells)})")
        w = f.worst_residual()
        if w > 0.25:
            problems.append(f"identity: worst informative residual {w:.1%} > 25%")
        else:
            print(f"identity: ok (worst informative residual {w:.1%})")
        d = f.device_time_disagreement()
        if d > 0.08:
            problems.append(
                f"clocks: on cells over {law_mod.CROSSCHECK_MIN_MS:g} ms the "
                f"drained clock and the event clock disagree by {d:.1%}")
        else:
            print(f"clocks: ok ({d:.1%} over "
                  f"{f.crosscheck_cells()} cells above "
                  f"{law_mod.CROSSCHECK_MIN_MS:g} ms)")
        gc = data_mod.graph_cells(doc)
        ident = max(abs(c.identity - c.speedup) / c.speedup for c in gc)
        if ident > 1e-6:
            problems.append(f"graphs: speedup is supposed to be algebra and "
                            f"drifts by {ident:.2e}")
        else:
            print(f"graphs: ok ({len(gc)} cells, speedup identity exact)")
        ov = max(c.event_timing_overstates_kernel_by for c in gc)
        if ov < 2.0:
            problems.append(
                "graphs: eager event timing no longer overstates the kernel, "
                "which is the finding this repository reports")
        else:
            print(f"event-timing: ok (overstates the kernel by up to {ov:.2f}x)")
    bad = violations(claims_mod.REGISTRY)
    if bad:
        problems.append(f"unquotable magnitudes still quoted: {bad}")
    else:
        print(f"claims: ok ({len(claims_mod.REGISTRY)} registered, "
              f"{len(unquotable(claims_mod.REGISTRY))} unquotable, none quoted)")
    for p in problems:
        print("FAIL " + p, file=sys.stderr)
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="drainlag",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="repository root (default: found upwards)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("report", help="everything, as status lines")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("law", help="the identity, cell by cell")
    p.set_defaults(func=cmd_law)

    p = sub.add_parser("saturation", help="why a longer loop is safer")
    p.set_defaults(func=cmd_saturation)

    p = sub.add_parser("graphs", help="what the launch gap is worth")
    p.set_defaults(func=cmd_graphs)

    p = sub.add_parser("claims", help="the six magnitudes and their support")
    p.add_argument("--support", choices=("measured", "witnessed", "derived",
                                         "folklore", "retracted"))
    p.set_defaults(func=cmd_claims)

    p = sub.add_parser("check", help="invariants, for CI")
    p.set_defaults(func=cmd_check)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
