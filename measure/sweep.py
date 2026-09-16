"""How wrong is a timed region that never drains the queue?

    python -X utf8 sweep.py <out.json>

benchdoctor's BD001 says such a region reads "usually 5-50x too fast". That
number came from my own experience, once, on one machine, and nothing has ever
checked it. This measures it.

## What is being measured

For a loop of K launches of a kernel whose device time is T:

    undrained   perf_counter around the loop, no sync -- what the buggy code reports
    drained     perf_counter around the loop plus one sync -- the honest number
    events      CUDA events around the loop -- device time, as a cross-check

`ratio = drained / undrained` is the factor the buggy number is too small by.

## The methodology trap this had to avoid

After an undrained measurement the queue still holds work. Start the next
measurement without draining and it inherits that backlog: the *next* number
is inflated by the previous cell's leftovers. Every measurement here is
preceded by an explicit `torch.cuda.synchronize()` for that reason, and
`--demo-contamination` shows what happens without it, because a sweep that
quietly self-contaminates produces a smooth, plausible, wrong curve.

## Two host-side constants that turn out to decide everything

`launch_us` -- how long one `torch.mm` takes to *enqueue* when nothing blocks.
`queue_depth` -- how many launches fit before the host blocks.

If T < launch_us the host cannot enqueue faster than the device drains, so no
backlog forms and the undrained number is *correct*. The error only exists
above that crossover. Nothing in BD001 knew this.
"""

import argparse
import json
import platform
import statistics
import sys
import time

import torch

DTYPE = torch.float16
BUDGET_S = 0.35          # cap device work per cell so the sweep finishes
REPS = 5                 # repetitions per cell; median reported, spread kept


def sync():
    torch.cuda.synchronize()


def kernel_time_us(a, b, c, reps=200):
    """Device time per launch, via CUDA events."""
    for _ in range(20):
        torch.mm(a, b, out=c)
    sync()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        torch.mm(a, b, out=c)
    e1.record()
    sync()
    return e0.elapsed_time(e1) * 1000.0 / reps


def launch_cost_us(n=64, reps=2000):
    """Host-side enqueue cost, measured where the device cannot be the bottleneck."""
    a = torch.randn(n, n, device="cuda", dtype=DTYPE)
    b = torch.randn(n, n, device="cuda", dtype=DTYPE)
    c = torch.empty(n, n, device="cuda", dtype=DTYPE)
    for _ in range(100):
        torch.mm(a, b, out=c)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.mm(a, b, out=c)
    t1 = time.perf_counter()
    sync()
    return (t1 - t0) * 1e6 / reps


def find_queue_depth(n=2048, hi=4096):
    """Smallest K at which the host blocks on launch.

    Enqueue K launches of a kernel far slower than the enqueue itself and time
    only the host side. While the queue has room the cost per launch is the
    enqueue cost; once it is full the host waits for the device, and the
    per-launch cost jumps to the kernel's device time. The knee is the depth.
    """
    a = torch.randn(n, n, device="cuda", dtype=DTYPE)
    b = torch.randn(n, n, device="cuda", dtype=DTYPE)
    c = torch.empty(n, n, device="cuda", dtype=DTYPE)
    t_dev = kernel_time_us(a, b, c, reps=30)
    lc = launch_cost_us()
    # per-launch host cost, as K grows
    curve = []
    k = 8
    while k <= hi:
        sync()
        t0 = time.perf_counter()
        for _ in range(k):
            torch.mm(a, b, out=c)
        t1 = time.perf_counter()
        sync()
        curve.append((k, (t1 - t0) * 1e6 / k))
        k *= 2
    # knee: first K where per-launch host cost exceeds 3x the enqueue cost
    knee = None
    for k, per in curve:
        if per > 3 * lc:
            knee = k
            break
    return {"kernel_us": t_dev, "launch_us": lc, "curve": curve,
            "knee_at_or_below": knee}


def cell(n, k):
    a = torch.randn(n, n, device="cuda", dtype=DTYPE)
    b = torch.randn(n, n, device="cuda", dtype=DTYPE)
    c = torch.empty(n, n, device="cuda", dtype=DTYPE)
    t_dev = kernel_time_us(a, b, c, reps=max(10, min(200, int(2000 / max(1, k)))))

    und, dra, evt = [], [], []
    for _ in range(REPS):
        # Every measurement starts from a drained queue. Without this the
        # previous cell's backlog lands in this one -- see --demo-contamination.
        sync()
        t0 = time.perf_counter()
        for _ in range(k):
            torch.mm(a, b, out=c)
        t1 = time.perf_counter()
        und.append((t1 - t0) * 1e3)

        sync()
        t0 = time.perf_counter()
        for _ in range(k):
            torch.mm(a, b, out=c)
        sync()
        t1 = time.perf_counter()
        dra.append((t1 - t0) * 1e3)

        sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(k):
            torch.mm(a, b, out=c)
        e1.record()
        sync()
        evt.append(e0.elapsed_time(e1))

    del a, b, c
    torch.cuda.empty_cache()
    med = statistics.median
    return {
        "n": n, "k": k, "kernel_us": t_dev,
        "undrained_ms": med(und), "drained_ms": med(dra), "events_ms": med(evt),
        "undrained_spread": max(und) / min(und) if min(und) > 0 else None,
        "drained_spread": max(dra) / min(dra) if min(dra) > 0 else None,
        "ratio": med(dra) / med(und) if med(und) > 0 else None,
        "reps": REPS,
    }


def demo_contamination(n=2048, k=64):
    """What a sweep looks like when it forgets to drain between cells.

    Runs the same undrained measurement twice in a row with no sync between
    them. The second inherits the first's backlog, so it reports a *larger*
    number for identical work -- and nothing about the output looks wrong.
    """
    a = torch.randn(n, n, device="cuda", dtype=DTYPE)
    b = torch.randn(n, n, device="cuda", dtype=DTYPE)
    c = torch.empty(n, n, device="cuda", dtype=DTYPE)
    for _ in range(20):
        torch.mm(a, b, out=c)
    sync()
    out = []
    for i in range(4):
        t0 = time.perf_counter()
        for _ in range(k):
            torch.mm(a, b, out=c)
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1e3)       # deliberately no sync
    sync()
    clean = []
    for i in range(4):
        sync()
        t0 = time.perf_counter()
        for _ in range(k):
            torch.mm(a, b, out=c)
        t1 = time.perf_counter()
        clean.append((t1 - t0) * 1e3)
    sync()
    return {"n": n, "k": k, "without_sync_between": out,
            "with_sync_between": clean,
            "without_spread": max(out) / min(out),
            "with_spread": max(clean) / min(clean)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--demo-contamination", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    p = torch.cuda.get_device_properties(0)
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "sm": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "vram_mib": round(p.total_memory / 2**20),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.system() + " " + platform.release(),
        "dtype": str(DTYPE),
        "note": "one consumer GPU, one OS, one torch build. Everything here is "
                "conditional on that and the ratios are the transferable part, "
                "not the absolute times.",
    }
    print(json.dumps(env, indent=1))

    doc = {"schema": "drainlag/sweep/1", "env": env}

    print("\n-- host-side constants --")
    doc["launch_us"] = launch_cost_us()
    print(f"launch cost: {doc['launch_us']:.2f} us/launch")
    doc["queue"] = find_queue_depth()
    print(f"queue knee at or below K={doc['queue']['knee_at_or_below']} "
          f"(kernel {doc['queue']['kernel_us']:.0f} us)")

    if args.demo_contamination:
        doc["contamination"] = demo_contamination()
        print("\n-- contamination demo --")
        print(json.dumps(doc["contamination"], indent=1))

    print("\n-- sweep --")
    print(f"{'N':>6} {'kernel us':>10} {'K':>6} {'undrained ms':>13} "
          f"{'drained ms':>11} {'events ms':>10} {'ratio':>8}")
    rows = []
    for n in (128, 256, 512, 1024, 2048, 4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=DTYPE)
        b = torch.randn(n, n, device="cuda", dtype=DTYPE)
        c = torch.empty(n, n, device="cuda", dtype=DTYPE)
        t_dev = kernel_time_us(a, b, c, reps=30)
        del a, b, c
        torch.cuda.empty_cache()
        k = 1
        while k <= 4096:
            if k * t_dev / 1e6 > BUDGET_S and k > 1:
                break
            r = cell(n, k)
            rows.append(r)
            print(f"{n:>6} {r['kernel_us']:>10.2f} {k:>6} "
                  f"{r['undrained_ms']:>13.3f} {r['drained_ms']:>11.3f} "
                  f"{r['events_ms']:>10.3f} {r['ratio']:>8.2f}")
            k *= 4
    doc["cells"] = rows
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write(chr(10))   # a text file ends with a newline; the bundle
                            # round-trip notices when it does not
    print(f"\nwrote {args.out}: {len(rows)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
