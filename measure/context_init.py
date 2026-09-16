"""Where the "10-100x warmup" number actually lives.

    python -X utf8 context_init.py <out.json>

BD002 says first-iteration cost is "10-100x a steady-state iteration (context
init, cuBLAS algorithm search, JIT)". Measured on a warm context, the cuBLAS
part is 1.03-2.36x and the host-timed part is 1.67-2.06x -- an order of
magnitude below the claim.

That does not mean the claim came from nowhere. CUDA *context creation* really
is enormous. It is just a **per-process** cost, paid once, before any of the
code BD002 looks at runs -- so a warmup loop inside a benchmark function cannot
remove it and cannot be blamed for including it.

Distinguishing "the number is wrong" from "the number is right about something
else" needs a fresh process per sample, so this spawns subprocesses.
"""

import argparse
import json
import statistics
import subprocess
import sys
import textwrap

CHILD = textwrap.dedent('''
    import json, time
    t_import0 = time.perf_counter()
    import torch
    t_import1 = time.perf_counter()

    # First touch of the device creates the context. Time it host-side with a
    # sync, because that is what a wall clock around "the first iteration"
    # would see.
    t0 = time.perf_counter()
    x = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # First real kernel after the context exists.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = torch.empty(1024, 1024, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    ts = []
    for _ in range(50):
        torch.cuda.synchronize()
        s = time.perf_counter()
        torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - s)
    ts.sort()
    steady = ts[len(ts) // 2]

    print("RESULT " + json.dumps({
        "import_torch_s": t_import1 - t_import0,
        "context_create_s": t1 - t0,
        "first_matmul_s": t3 - t2,
        "steady_matmul_s": steady,
        "context_over_steady": (t1 - t0) / steady,
        "first_matmul_over_steady": (t3 - t2) / steady,
    }))
''')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for i in range(args.runs):
        proc = subprocess.run([sys.executable, "-X", "utf8", "-c", CHILD],
                              capture_output=True, text=True, encoding="utf-8")
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"run {i}: no result\n{proc.stderr[-500:]}", file=sys.stderr)
            continue
        r = json.loads(line[len("RESULT "):])
        rows.append(r)
        print(f"run {i}: import {r['import_torch_s']:.2f}s  "
              f"context {r['context_create_s'] * 1e3:>8.1f} ms  "
              f"first mm {r['first_matmul_s'] * 1e6:>8.1f} us  "
              f"steady {r['steady_matmul_s'] * 1e6:>7.1f} us  "
              f"context/steady {r['context_over_steady']:>8.0f}x")

    if not rows:
        return 1
    med = statistics.median
    doc = {
        "schema": "drainlag/context/1",
        "runs": rows,
        "median": {
            "import_torch_s": med(r["import_torch_s"] for r in rows),
            "context_create_ms": med(r["context_create_s"] for r in rows) * 1e3,
            "first_matmul_us": med(r["first_matmul_s"] for r in rows) * 1e6,
            "steady_matmul_us": med(r["steady_matmul_s"] for r in rows) * 1e6,
            "context_over_steady": med(r["context_over_steady"] for r in rows),
            "first_matmul_over_steady": med(r["first_matmul_over_steady"]
                                            for r in rows),
        },
        "reading":
            "CUDA context creation is thousands of times a steady-state matmul, "
            "so the folklore number is not invented -- it is just attached to "
            "the wrong thing. It is paid once per process, before the function "
            "BD002 inspects is ever called, and no warmup loop inside that "
            "function can remove it. The cost a warmup loop *can* remove is the "
            "first-call-of-a-new-shape cost, and that one is about 2x.",
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write(chr(10))   # a text file ends with a newline; the bundle
                            # round-trip notices when it does not
    m = doc["median"]
    print(f"\nmedian: context {m['context_create_ms']:.0f} ms = "
          f"{m['context_over_steady']:.0f}x a steady matmul; "
          f"first matmul {m['first_matmul_over_steady']:.2f}x")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
