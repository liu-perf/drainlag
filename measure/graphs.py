"""The same number, read the other way: how much is launch overhead worth?

    python -X utf8 graphs.py <out.json>

`sweep.py` measures how wrong an un-drained timer is. The quantity that makes
it wrong is the host's cost to enqueue a launch -- 9.6 us on this machine --
and that cost is not only a measurement artefact. It is time.

CUDA graphs remove it: capture K launches once, replay them with a single
submission. So the *speedup a graph can give you* and the *error an un-drained
timer makes* are two readings of one ratio:

    un-drained timer is wrong by     device_us / host_us      (when device > host)
    a graph can speed you up by      host_us / device_us      (when host > device)

Which predicts something worth checking: **where the timer lies most, graphs
help least.** A 23 ms kernel makes an un-drained clock read 595x fast, and a
graph does nothing for it, because the host was never the bottleneck.

## The ordering this had to get right

The first attempt measured eager *after* capturing the graph and got 21.4 us
per launch for a kernel `sweep.py` had measured at 10.4 us. Graph capture
switches the allocator to a private memory pool and leaves it there, so the
eager path afterwards is not the eager path from before.

So each cell measures eager, then captures, then measures the graph, then
measures eager **again** -- and reports the drift. A speedup computed against
a contaminated baseline is the easiest way to publish a number twice as good
as the truth, and it would look completely ordinary.
"""

import argparse
import json
import platform
import statistics
import sys
import time

import torch

DT = torch.float16
BUDGET_S = 0.30
REPS = 5
med = statistics.median


def sync():
    torch.cuda.synchronize()


def drained_ms(fn, reps=REPS):
    out = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return med(out), min(out), max(out)


def device_ms(fn, reps=REPS):
    out = []
    for _ in range(reps):
        sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        sync()
        out.append(e0.elapsed_time(e1))
    return med(out)


def kernel_us(a, b, c, reps=100):
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


def host_us_per_launch(a, b, c, k):
    """What an un-drained clock measures: the host side, per launch."""
    sync()
    t0 = time.perf_counter()
    for _ in range(k):
        torch.mm(a, b, out=c)
    t1 = time.perf_counter()
    sync()
    return (t1 - t0) * 1e6 / k


def cell(n, k):
    a = torch.randn(n, n, device="cuda", dtype=DT)
    b = torch.randn(n, n, device="cuda", dtype=DT)
    c = torch.empty(n, n, device="cuda", dtype=DT)

    for _ in range(50):
        torch.mm(a, b, out=c)
    sync()
    dev_us = kernel_us(a, b, c, reps=max(20, min(200, int(4000 / max(1, k)))))
    host_us = host_us_per_launch(a, b, c, k)

    # Captures bound explicitly as defaults rather than closed over. In timing
    # code late binding is a real hazard -- a closure that reads `a` at call
    # time measures whatever `a` happens to be then, not what it was when the
    # baseline was taken -- and here it also lets the tensors be released at
    # the end of the cell without leaving the closure dangling.
    def eager(a=a, b=b, c=c, k=k):
        for _ in range(k):
            torch.mm(a, b, out=c)

    eager_before, eb_lo, eb_hi = drained_ms(eager)
    eager_dev = device_ms(eager)

    # Capture on a side stream, the way torch requires, and time the capture:
    # it is a one-time cost and the break-even point depends on it.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            torch.mm(a, b, out=c)
    torch.cuda.current_stream().wait_stream(s)
    sync()

    g = torch.cuda.CUDAGraph()
    t0 = time.perf_counter()
    with torch.cuda.graph(g):
        for _ in range(k):
            torch.mm(a, b, out=c)
    sync()
    capture_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(5):
        g.replay()
    sync()
    graph_ms, g_lo, g_hi = drained_ms(g.replay)
    graph_dev = device_ms(g.replay)

    # The contamination check. Graph capture moves the allocator to a private
    # pool; the eager path afterwards is not the one from before.
    eager_after, _, _ = drained_ms(eager)

    del g, a, b, c
    torch.cuda.empty_cache()

    speedup = eager_before / graph_ms if graph_ms else float("nan")
    gain_ms = eager_before - graph_ms
    # What a graph replay costs per launch is the closest thing to the kernel's
    # own duration available here: the submission happens once, so the device
    # runs the sequence back to back. Event timing around an *eager* loop does
    # NOT give this -- it includes the idle the device spends waiting for the
    # host, which is why `device_us_per_launch` above is 10.6 us for a kernel
    # that turns out to take 1.9 us.
    true_kernel_us = graph_ms * 1000.0 / k
    eager_us = eager_before * 1000.0 / k
    gap_us = eager_us - true_kernel_us
    return {
        "n": n, "k": k,
        "device_us_per_launch": dev_us,
        "host_us_per_launch": host_us,
        "eager_before_ms": eager_before,
        "eager_after_ms": eager_after,
        "eager_drift": eager_after / eager_before if eager_before else None,
        "eager_spread": eb_hi / eb_lo if eb_lo else None,
        "graph_ms": graph_ms,
        "graph_spread": g_hi / g_lo if g_lo else None,
        "eager_device_ms": eager_dev,
        "graph_device_ms": graph_dev,
        "capture_ms": capture_ms,
        "speedup": speedup,
        "true_kernel_us": true_kernel_us,
        "eager_us_per_launch": eager_us,
        "gap_us": gap_us,
        # Algebra, not a model: speedup = (kernel + gap) / kernel. Reported so
        # the reader can see that the content is entirely in `gap_us`.
        "speedup_identity": 1.0 + gap_us / true_kernel_us,
        # The model this script was written with, before it was run. Kept
        # because it FAILED: using event-measured device time it predicts 1.00
        # where the truth is 5.69, an 82% miss on the cells that matter. Event
        # timing of an eager loop measures the pipeline rate, not the kernel.
        "predicted_speedup_from_event_timing": max(1.0, host_us / dev_us),
        "breakeven_replays": (capture_ms / gain_ms) if gain_ms > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    p = torch.cuda.get_device_properties(0)
    doc = {"schema": "drainlag/graphs/1",
           "env": {"gpu": torch.cuda.get_device_name(0),
                   "sm": f"{p.major}.{p.minor}",
                   "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda,
                   "os": platform.system() + " " + platform.release()},
           "reading":
               "speedup = 1 + gap/kernel, which is algebra; the content is what "
               "`gap_us` turns out to be. The device idles between eager "
               "launches by roughly max(~2 us, host_enqueue - kernel), so a "
               "graph is worth 3-6x when the kernel is shorter than the host's "
               "enqueue cost and worth nothing when it is longer. Same ~9.6 us "
               "constant as the rest of this repository, read a third way.",
           "the_model_that_failed":
               "`predicted_speedup_from_event_timing` is the model this script "
               "was written with before it was run: max(1, host/device) using "
               "the event-measured device time. It predicts 1.00 where the "
               "truth is 5.69 -- an 82% miss -- because event timing around an "
               "eager loop measures the pipeline rate, not the kernel. The "
               "field is kept rather than deleted: you cannot tell from eager "
               "profiling how much a graph will buy you, and that is the "
               "finding."}

    print(f"{'N':>6} {'host us':>8} {'event us':>9} {'true us':>8} "
          f"{'gap us':>7} {'K':>5} {'eager ms':>9} {'graph ms':>9} "
          f"{'speedup':>8} {'drift':>6} {'breakeven':>10}")
    rows = []
    for n in (128, 256, 512, 1024, 2048, 4096):
        a = torch.randn(n, n, device="cuda", dtype=DT)
        b = torch.randn(n, n, device="cuda", dtype=DT)
        c = torch.empty(n, n, device="cuda", dtype=DT)
        t_dev = kernel_us(a, b, c, reps=30)
        del a, b, c
        torch.cuda.empty_cache()
        k = 8
        while k <= 1024:
            if k * t_dev / 1e6 > BUDGET_S and k > 8:
                break
            r = cell(n, k)
            rows.append(r)
            be = r["breakeven_replays"]
            print(f"{n:>6} {r['host_us_per_launch']:>8.2f} "
                  f"{r['device_us_per_launch']:>9.1f} "
                  f"{r['true_kernel_us']:>8.2f} {r['gap_us']:>7.2f} {k:>5} "
                  f"{r['eager_before_ms']:>9.3f} {r['graph_ms']:>9.3f} "
                  f"{r['speedup']:>8.2f} {r['eager_drift']:>6.2f} "
                  f"{('%.1f' % be) if be else '-':>10}")
            k *= 4
    doc["cells"] = rows
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write(chr(10))
    print(f"\nwrote {args.out}: {len(rows)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
