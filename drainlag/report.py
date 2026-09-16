"""Assemble the report: `{location}: [{status}] {message}`, same as the others.

Ordering decision, stated because it is a claim about how a reader will
misread the alternative: **the crossover comes before any factor.** A reader
who sees "up to 595x too fast" first will file it as "always catastrophic". A
reader who sees "below ~10 us per kernel there is no error at all, above it the
factor is device time over enqueue time" cannot.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

from . import claims as claims_mod
from . import data as data_mod
from . import law as law_mod
from .magnitude import tally, unquotable


def lines(root: Optional[pathlib.Path] = None) -> List[str]:
    doc = data_mod.load(root)
    env = data_mod.environment(doc)
    enq = data_mod.enqueue_us(doc)
    f = data_mod.fit(doc)
    out: List[str] = []

    out.append(f"machine: [info] {env['gpu']} (sm {env['sm']}, {env['sms']} SMs, "
               f"{env['vram_mib']} MiB), torch {env['torch']} / CUDA "
               f"{env['cuda_runtime']}, {env['os']}. One card, one OS, one "
               "build -- the ratios transfer, the absolute times do not")

    out.append(f"crossover: [info] the host enqueues one launch in {enq:.1f} us. "
               "A kernel faster than that drains before the host can refill, so "
               "no backlog forms and an un-drained timer is already correct")

    r = f.ratio_range
    out.append(f"factor: [warn] across {len(f.cells)} cells the un-drained "
               f"number was {r['min']:.2f}x to {r['max']:.0f}x too fast. It is "
               "not one number and it is not a property of the code")

    out.append(f"identity: [ok] ratio = device_us_per_launch / "
               f"host_us_per_launch, worst residual "
               f"{f.worst_residual():.1%} over the {len(f.informative)} cells "
               f"with a factor above {law_mod.INFORMATIVE_RATIO:g}x "
               f"({f.worst_residual(informative_only=False):.0%} over all "
               f"{len(f.cells)}, which is the identity being uninformative "
               "where there is no error to predict)")

    out.append(f"clocks: [ok] on the {f.crosscheck_cells()} cells longer than "
               f"{law_mod.CROSSCHECK_MIN_MS:g} ms, the drained wall clock and "
               f"the CUDA-event clock agree to "
               f"{f.device_time_disagreement():.1%}; both are supposed to be "
               "the device time. Include the short cells and it is "
               f"{f.device_time_disagreement(0):.0%}, all of it sub-0.2 ms "
               "measurements where synchronize overhead is most of the reading")

    sat = data_mod.saturation_rows(doc)
    lo = min(sat, key=lambda r: r["k"])
    hi = max(sat, key=lambda r: r["k"])
    kn = law_mod.knee(sat, enq)
    out.append(f"saturation: [warn] same kernel, no sync: K={lo['k']} is "
               f"{lo['ratio']:.2f}x too fast, K={hi['k']} is {hi['ratio']:.2f}x. "
               f"The host starts blocking at K={kn}, so a longer loop makes the "
               "missing drain matter LESS -- and the rules that look for this "
               "only fire when there is a loop")

    ctx = data_mod.context(doc)
    wu = data_mod.warmup(doc)
    warm = [x["ratio"] for x in wu["cublas_first_call_of_a_new_shape"]]
    out.append(f"warmup: [warn] two costs wear one number. Removable by a "
               f"warmup loop (first call of a new shape, warm process): "
               f"{min(warm):.2f}-{max(warm):.2f}x. Not removable and paid before "
               f"the function runs (first kernel in a fresh process): "
               f"{ctx['first_matmul_over_steady']:.0f}x")

    cont = data_mod.contamination(doc)
    out.append(f"contamination: [violation] repeating an un-drained measurement "
               f"without draining between rounds drifted "
               f"{cont['first_vs_last_no_drain']:.2f}x over "
               f"{len(cont['no_drain_between_rounds'])} rounds of identical "
               f"work; with a drain between rounds, "
               f"{cont['first_vs_last_drained']:.2f}x. A sweep that forgets "
               "this produces a smooth, plausible, wrong curve")

    v = data_mod.verdicts(doc)
    tl = v["torch_load"]
    out.append(f"verdict.torch_load: [ok] "
               f"torch.load(mmap=True, map_location='cuda') with a drain over "
               f"without: {tl['with_sync_over_no_sync']:.3f}. The copy blocks "
               "the host, so the un-drained number was already right -- "
               "corpusaudit's `unclear` on pytorch/benchmarks/inference/"
               "server.py:198 resolves to false_positive")
    ps = v["plan_shape"]
    out.append(f"verdict.plan_shape: [warn] the flashinfer shape (small H2D "
               f"copies plus one small kernel, measured once) understates by "
               f"{ps['ratio']:.2f}x. That bounds the effect but does not settle "
               "the verdict: this is a reconstruction, not flashinfer, and the "
               "two `unclear` findings stay `unclear`")

    gc = data_mod.graph_cells(doc)
    best = max(gc, key=lambda c: c.speedup)
    worst = min(gc, key=lambda c: c.speedup)
    out.append(f"graphs: [info] the same {enq:.1f} us, read a third way: it is "
               f"time the device spends idle. CUDA graphs remove it and are "
               f"worth {best.speedup:.2f}x on a {best.true_kernel_us:.1f} us "
               f"kernel and {worst.speedup:.2f}x on a "
               f"{worst.true_kernel_us / 1000:.1f} ms one. Where the un-drained "
               "timer lies most, a graph helps least -- the host was never the "
               "bottleneck there")
    ov = max(gc, key=lambda c: c.event_timing_overstates_kernel_by)
    out.append(f"event-timing: [violation] and you cannot tell which case you "
               f"are in from eager profiling. CUDA events around an eager loop "
               f"measure the pipeline rate, not the kernel: at N={ov.n} they "
               f"report {ov.event_us_per_launch:.1f} us for a kernel that a "
               f"graph replay runs in {ov.true_kernel_us:.2f} us -- "
               f"{ov.event_timing_overstates_kernel_by:.1f}x. The speedup model "
               "this repository was written with used that number and predicted "
               "1.00 where the truth was 5.36")
    drift = [c.eager_drift for c in gc]
    out.append(f"graph-pool: [warn] measuring eager after capturing a graph is "
               f"measuring a different allocator. Eager drifted "
               f"{min(drift):.2f}-{max(drift):.2f}x across capture, so every "
               "baseline here is taken before the capture, and the after value "
               "is reported next to it")

    for m in claims_mod.REGISTRY:
        sev = {"measured": "ok", "derived": "warn", "witnessed": "info",
               "folklore": "warn", "retracted": "violation"}[m.support]
        out.append(f"claim.{m.rule}: [{sev}] {m.describe()}")

    t = tally(claims_mod.REGISTRY)
    out.append("claims: [info] " + ", ".join(f"{k}={v}" for k, v in t.items()
                                             if v))
    uq = unquotable(claims_mod.REGISTRY)
    out.append(f"unquotable: [ok] {len(uq)} of {len(claims_mod.REGISTRY)} "
               "magnitudes depend on something the source does not contain, and "
               "none of them appears in a user-visible message")
    return out


def text(root: Optional[pathlib.Path] = None) -> str:
    return "\n".join(lines(root))


def summary(root: Optional[pathlib.Path] = None) -> Dict[str, object]:
    doc = data_mod.load(root)
    f = data_mod.fit(doc)
    return {
        "env": data_mod.environment(doc),
        "enqueue_us": data_mod.enqueue_us(doc),
        "cells": len(f.cells),
        "ratio_range": f.ratio_range,
        "worst_residual_informative": f.worst_residual(),
        "worst_residual_all": f.worst_residual(informative_only=False),
        "clock_disagreement": f.device_time_disagreement(),
        "clock_disagreement_all_cells": f.device_time_disagreement(0),
        "claims": tally(claims_mod.REGISTRY),
        "unquotable": [m.rule for m in unquotable(claims_mod.REGISTRY)],
        "pcie_gen5_over_gen1": claims_mod.pcie_generation_error(1, 5),
        "graph_speedup_range": {
            "min": min(c.speedup for c in data_mod.graph_cells(doc)),
            "max": max(c.speedup for c in data_mod.graph_cells(doc))},
        "event_timing_overstates_kernel_by_up_to":
            max(c.event_timing_overstates_kernel_by
                for c in data_mod.graph_cells(doc)),
    }
