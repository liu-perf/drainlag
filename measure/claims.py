"""Measure the magnitudes benchdoctor's messages assert, plus two open verdicts.

    python -X utf8 claims.py <out.json>

Four experiments, each tied to a specific sentence somebody will read and act on.

1. `saturation`   -- BD001's error in the regime where the launch queue fills.
                     The sweep covers K below the queue depth, where the factor
                     is device_time/host_enqueue_time. Past the depth the host
                     blocks and the measurement should converge to correct --
                     meaning a *longer* loop makes the bug matter *less*, which
                     is the opposite of what a rule that fires on loops implies.

2. `contamination` -- what a sweep looks like when it forgets to drain between
                     cells. The first attempt at this failed to reproduce, at
                     K=64, and the reason is the same queue depth: 64 launches
                     of a 371 us kernel is 24 ms of backlog and the queue holds
                     ~1024 launches, so nothing spills. It needs K past the knee.

3. `warmup`       -- BD002 says first-iteration cost is "10-100x a steady-state
                     iteration" (context init, cuBLAS algorithm search, JIT).
                     Three separate causes, measured separately, because they
                     have wildly different sizes and the rule quotes one range
                     for all of them.

4. `verdicts`     -- the two corpusaudit `unclear` findings whose
                     `what_would_settle_it` is a measurement I can actually run:
                     `torch.load(mmap=True, map_location='cuda')` timed without
                     a drain, and the "plan()" shape (a few small H2D copies plus
                     one small kernel, measured once).
"""

import argparse
import json
import platform
import statistics
import sys
import time

import torch

DT = torch.float16
med = statistics.median


def sync():
    torch.cuda.synchronize()


def host_per_launch(a, b, c, k):
    """Host-side cost per launch, from an undrained loop of k launches."""
    sync()
    t0 = time.perf_counter()
    for _ in range(k):
        torch.mm(a, b, out=c)
    t1 = time.perf_counter()
    sync()
    return (t1 - t0) * 1e6 / k


def device_per_launch(a, b, c, reps):
    sync()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        torch.mm(a, b, out=c)
    e1.record()
    sync()
    return e0.elapsed_time(e1) * 1000.0 / reps


# ---------------------------------------------------------------- 1. saturation
def saturation(n=1024, ks=(64, 256, 1024, 2048, 4096, 8192, 16384)):
    a = torch.randn(n, n, device="cuda", dtype=DT)
    b = torch.randn(n, n, device="cuda", dtype=DT)
    c = torch.empty(n, n, device="cuda", dtype=DT)
    for _ in range(50):
        torch.mm(a, b, out=c)
    sync()
    dev_us = device_per_launch(a, b, c, 200)
    rows = []
    for k in ks:
        und, dra = [], []
        for _ in range(3):
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
        rows.append({"k": k, "undrained_ms": med(und), "drained_ms": med(dra),
                     "ratio": med(dra) / med(und),
                     "host_us_per_launch": med(und) * 1e3 / k})
    del a, b, c
    torch.cuda.empty_cache()
    return {"n": n, "device_us_per_launch": dev_us, "rows": rows}


# ------------------------------------------------------------- 2. contamination
def contamination(n=1024, k=4096, rounds=5):
    """Repeat an undrained measurement with and without a drain between rounds.

    K is chosen past the queue knee on purpose: below it the queue absorbs the
    backlog and nothing shows, which is what the first attempt at this
    experiment found out the hard way.
    """
    a = torch.randn(n, n, device="cuda", dtype=DT)
    b = torch.randn(n, n, device="cuda", dtype=DT)
    c = torch.empty(n, n, device="cuda", dtype=DT)
    for _ in range(50):
        torch.mm(a, b, out=c)
    sync()

    dirty = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(k):
            torch.mm(a, b, out=c)
        t1 = time.perf_counter()
        dirty.append((t1 - t0) * 1e3)     # deliberately no sync between rounds
    sync()

    clean = []
    for _ in range(rounds):
        sync()
        t0 = time.perf_counter()
        for _ in range(k):
            torch.mm(a, b, out=c)
        t1 = time.perf_counter()
        clean.append((t1 - t0) * 1e3)
    sync()
    del a, b, c
    torch.cuda.empty_cache()
    return {"n": n, "k": k,
            "no_drain_between_rounds": dirty,
            "drain_between_rounds": clean,
            "first_vs_last_no_drain": dirty[-1] / dirty[0],
            "first_vs_last_drained": clean[-1] / clean[0]}


# ------------------------------------------------------------------- 3. warmup
def warmup_costs():
    out = {}

    # (a) cuBLAS algorithm selection: first matmul of a *new shape*.
    per_shape = []
    for n in (1024, 1536, 2048, 3072):
        a = torch.randn(n, n, device="cuda", dtype=DT)
        b = torch.randn(n, n, device="cuda", dtype=DT)
        c = torch.empty(n, n, device="cuda", dtype=DT)
        sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); torch.mm(a, b, out=c); e1.record()
        sync()
        first = e0.elapsed_time(e1) * 1000.0
        steady = device_per_launch(a, b, c, 100)
        per_shape.append({"n": n, "first_us": first, "steady_us": steady,
                          "ratio": first / steady})
        del a, b, c
        torch.cuda.empty_cache()
    out["cublas_first_call_of_a_new_shape"] = per_shape

    # (b) host-side first call: what a wall clock sees, which is what BD002's
    #     victim measures. Includes any lazy module init, not just the kernel.
    host_first = []
    for n in (1024, 2048):
        a = torch.randn(n, n, device="cuda", dtype=DT)
        b = torch.randn(n, n, device="cuda", dtype=DT)
        c = torch.empty(n, n, device="cuda", dtype=DT)
        sync()
        t0 = time.perf_counter(); torch.mm(a, b, out=c); sync(); t1 = time.perf_counter()
        first = (t1 - t0) * 1e6
        ts = []
        for _ in range(30):
            sync()
            t0 = time.perf_counter(); torch.mm(a, b, out=c); sync(); t1 = time.perf_counter()
            ts.append((t1 - t0) * 1e6)
        host_first.append({"n": n, "first_us": first, "steady_us": med(ts),
                           "ratio": first / med(ts)})
        del a, b, c
        torch.cuda.empty_cache()
    out["host_timed_first_call"] = host_first

    # (c) torch.compile: the biggest of the three by orders of magnitude, and
    #     the one BD002's range does not remotely cover.
    try:
        def f(x):
            return (x @ x).relu().sum()

        g = torch.compile(f)
        x = torch.randn(512, 512, device="cuda", dtype=torch.float32)
        sync()
        t0 = time.perf_counter(); g(x); sync(); t1 = time.perf_counter()
        first = (t1 - t0) * 1e6
        ts = []
        for _ in range(20):
            sync()
            t0 = time.perf_counter(); g(x); sync(); t1 = time.perf_counter()
            ts.append((t1 - t0) * 1e6)
        out["torch_compile_first_call"] = {
            "first_us": first, "steady_us": med(ts), "ratio": first / med(ts)}
    except Exception as exc:
        out["torch_compile_first_call"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ----------------------------------------------------------------- 4. verdicts
def verdict_torch_load(tmpdir):
    """corpusaudit verdict: pytorch/benchmarks/inference/server.py:198.

    `torch.load(path, mmap=True, map_location=device)` timed with no drain. The
    question was whether the copies out of mmapped host memory block the host.
    """
    import pathlib
    p = pathlib.Path(tmpdir) / "weights.pt"
    # ~90 MB of parameters, comparable to a resnet18 checkpoint (~45 MB) but
    # large enough that a non-blocking copy would be visible.
    sd = {f"layer{i}.weight": torch.randn(1024, 1024, dtype=torch.float32)
          for i in range(22)}
    torch.save(sd, p)
    rows = []
    for label, do_sync in (("no_sync", False), ("with_sync", True)):
        ts = []
        for _ in range(5):
            sync()
            t0 = time.perf_counter()
            loaded = torch.load(p, mmap=True, map_location="cuda",
                                weights_only=True)
            # force the storages to materialise the way load_state_dict would
            total = sum(v.numel() for v in loaded.values())
            if do_sync:
                sync()
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1e3)
            del loaded
            torch.cuda.empty_cache()
        rows.append({"variant": label, "ms": ts, "median_ms": med(ts),
                     "elements": total})
    ratio = rows[1]["median_ms"] / rows[0]["median_ms"]
    return {"rows": rows, "with_sync_over_no_sync": ratio,
            "verdict": ("the copy blocks the host; the un-drained number is "
                        "already right" if ratio < 1.10 else
                        "the un-drained number understates the load")}


def verdict_plan_shape():
    """corpusaudit verdict: flashinfer bench_batch_attention.py:214/242.

    The shape is: a handful of small int32 tensors moved to the device, then one
    small kernel, timed exactly once with no drain, and the result *added into a
    published per-layer latency*. flashinfer itself is not installed here, so
    this measures the shape rather than the library -- stated plainly, because
    a reconstruction is not the thing.
    """
    idx = [torch.arange(0, 4096, dtype=torch.int32) for _ in range(4)]
    w = torch.randn(256, 256, device="cuda", dtype=DT)
    x = torch.randn(256, 256, device="cuda", dtype=DT)
    o = torch.empty(256, 256, device="cuda", dtype=DT)

    def plan_like():
        moved = [t.to("cuda", non_blocking=False) for t in idx]
        torch.mm(w, x, out=o)
        return moved

    for _ in range(20):
        plan_like()
    sync()

    und, dra = [], []
    for _ in range(20):
        sync()
        t0 = time.perf_counter(); plan_like(); t1 = time.perf_counter()
        und.append((t1 - t0) * 1e6)
        sync()
        t0 = time.perf_counter(); plan_like(); sync(); t1 = time.perf_counter()
        dra.append((t1 - t0) * 1e6)
    return {"undrained_us": med(und), "drained_us": med(dra),
            "ratio": med(dra) / med(und),
            "undrained_spread": max(und) / min(und),
            "note": "the shape, not flashinfer. NUM_LAYERS in that benchmark is "
                    "the divisor applied to this term, so the effect on the "
                    "headline is this gap divided by that."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    p = torch.cuda.get_device_properties(0)
    doc = {"schema": "drainlag/claims/1",
           "env": {"gpu": torch.cuda.get_device_name(0),
                   "sm": f"{p.major}.{p.minor}",
                   "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda,
                   "os": platform.system() + " " + platform.release()}}

    print("-- 1. saturation --")
    doc["saturation"] = saturation()
    for r in doc["saturation"]["rows"]:
        print(f"  K={r['k']:>6} undrained {r['undrained_ms']:>9.2f} ms  "
              f"drained {r['drained_ms']:>9.2f} ms  ratio {r['ratio']:>7.2f}  "
              f"host {r['host_us_per_launch']:>7.2f} us/launch")

    print("-- 2. contamination --")
    doc["contamination"] = contamination()
    c = doc["contamination"]
    print(f"  no drain between rounds: {[round(x, 1) for x in c['no_drain_between_rounds']]}")
    print(f"  drain between rounds:    {[round(x, 1) for x in c['drain_between_rounds']]}")
    print(f"  last/first no-drain {c['first_vs_last_no_drain']:.2f}x  "
          f"drained {c['first_vs_last_drained']:.2f}x")

    print("-- 3. warmup --")
    doc["warmup"] = warmup_costs()
    for r in doc["warmup"]["cublas_first_call_of_a_new_shape"]:
        print(f"  cuBLAS n={r['n']:>5} first {r['first_us']:>9.1f} us  "
              f"steady {r['steady_us']:>8.1f} us  ratio {r['ratio']:>7.2f}")
    for r in doc["warmup"]["host_timed_first_call"]:
        print(f"  host   n={r['n']:>5} first {r['first_us']:>9.1f} us  "
              f"steady {r['steady_us']:>8.1f} us  ratio {r['ratio']:>7.2f}")
    tc = doc["warmup"]["torch_compile_first_call"]
    print(f"  torch.compile: {tc}")

    print("-- 4. verdicts --")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        doc["verdict_torch_load"] = verdict_torch_load(td)
    v = doc["verdict_torch_load"]
    print(f"  torch.load  no_sync {v['rows'][0]['median_ms']:.2f} ms  "
          f"with_sync {v['rows'][1]['median_ms']:.2f} ms  "
          f"ratio {v['with_sync_over_no_sync']:.3f}")
    print(f"    -> {v['verdict']}")
    doc["verdict_plan_shape"] = verdict_plan_shape()
    v = doc["verdict_plan_shape"]
    print(f"  plan-shape  undrained {v['undrained_us']:.1f} us  "
          f"drained {v['drained_us']:.1f} us  ratio {v['ratio']:.2f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write(chr(10))   # a text file ends with a newline; the bundle
                            # round-trip notices when it does not
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
