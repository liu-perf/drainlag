
# drainlag

**My own linter told people a missing `synchronize()` makes a benchmark read "usually 5-50x too fast". Measured: 0.98x to 595x — and the factor is not a property of the code.**

```console
$ drainlag report
machine: [info] NVIDIA GeForce RTX 5060 Ti (sm 12.0, 36 SMs, 8151 MiB), torch 2.11.0+cu128 / CUDA 12.8, Windows 11. One card, one OS, one build -- the ratios transfer, the absolute times do not
crossover: [info] the host enqueues one launch in 9.6 us. A kernel faster than that drains before the host can refill, so no backlog forms and an un-drained timer is already correct
factor: [warn] across 39 cells the un-drained number was 0.98x to 595x too fast. It is not one number and it is not a property of the code
identity: [ok] ratio = device_us_per_launch / host_us_per_launch, worst residual 18.2% over the 17 cells with a factor above 5x
clocks: [ok] on the 23 cells longer than 1 ms, the drained wall clock and the CUDA-event clock agree to 3.2%
saturation: [warn] same kernel, no sync: K=64 is 5.83x too fast, K=16384 is 1.06x. The host starts blocking at K=2048, so a longer loop makes the missing drain matter LESS -- and the rules that look for this only fire when there is a loop
warmup: [warn] two costs wear one number. Removable by a warmup loop (first call of a new shape, warm process): 1.03-2.36x. Not removable and paid before the function runs (first kernel in a fresh process): 1514x
claim.BD001: [violation] claims 5-50x, observed 0.98-594.6x (understates the spread 61x); depends on device_us_per_launch / host_us_per_launch -- NOT knowable from source
claim.BD020: [warn] claims 4x; depends on the ratio of per-lane throughput between the two PCIe generations
```

Zero dependencies to read the results. A CUDA device and `measure/*.py` to reproduce them.

---

## Why this exists

Twelve projects spent on qualifying other people's numbers: where a figure came from, what arithmetic a telemetry column supports, how far a cost model sits from the machine, whether an output was ever validated, whether a *rule* had ever been right about anybody's code.

The twelfth one — a corpus audit of my own static analyser — ended with four findings marked `unclear`, each carrying the measurement that would settle it, and a sentence in its own demo script: *"that's the honest next step, not a PR."*

Reading back through the analyser to write that audit, something else was obvious. Its messages are full of numbers:

```
BD001  "this times the launch queue, not the kernel -- usually 5-50x too fast"
BD002  "one-time costs that can be 10-100x a steady-state iteration"
BD010  "verification runs inside the timed region and reports GFLOPs 2-3x low"
BD020  "Gen5 read as Gen1 is a 4x error"
```

Four numbers a reader will act on. **None had ever been measured. One is arithmetically wrong.** Twelve projects of demanding provenance for other people's figures, and the figures in my own diagnostics were things I believed.

## What was measured

39 cells on one consumer GPU: matmuls from 9 µs to 23 ms of device time, loops from 1 to 4096 iterations, three clocks on each — an un-drained wall clock, a drained wall clock, and CUDA events. Plus a saturation sweep to K=16384, five fresh processes for the context cost, and the two `unclear` verdicts the previous project left open.

### The factor is an identity, not a range

An un-drained timer measures the **host**: how long the launches took to enqueue. Add a `synchronize()` and it measures the **device**. So

```
factor = device_us_per_launch / host_us_per_launch
```

Over the 17 cells whose factor exceeds 5x the worst residual is 18.2% and the median is under 5%. Below 5x the residual grows to about half — which is the identity being uninformative exactly where there is no error to predict: 1.63 versus 1.14 is a large relative gap between two numbers that both mean "about right".

| N | kernel | K | host µs/launch | measured | identity |
|---|---|---|---|---|---|
| 2048 | 371 µs | 64 | 8.64 | **43.09x** | 43.06 |
| 4096 | 2.88 ms | 64 | 10.46 | **275.33x** | 275.39 |
| 8192 | 23.0 ms | 4 | 38.78 | **594.63x** | 593.66 |

### Three things the 5-50x version did not contain

**There is a floor below which the bug does not exist.** The host enqueues one launch in 9.6 µs. A kernel faster than that drains before the host can refill, no backlog forms, and the un-drained number is *already right*: **0.98–2.31x** across every cell whose per-launch pipeline ran at 9–15 µs. Most kernels in the corpus the analyser was audited against are in that regime — which is a large part of why it found nothing there.

> Those cells are *not* 9–15 µs kernels, and an earlier draft of this README said they were. The CUDA-graph measurement below shows the kernels are **1.9–9.3 µs**; the 9–15 µs is the rate the host gated them to. Same conclusion, wrong quantity named — see [the graph section](#the-same-constant-a-third-time-what-the-launch-gap-is-worth).

**A longer loop is a safer loop.** The launch queue holds ~1024–2048 pending launches; past that the host blocks on every further launch and the measurement converges to correct.

| K | 64 | 256 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| factor | 5.83 | 6.53 | 6.26 | **1.93** | 1.31 | 1.13 | **1.06** |

Same kernel, same missing sync. And BD001 and BD002 **only fire when there is a loop** — so the rule is most eager exactly where the bug is least harmful.

**The ceiling is far above the claim.** A 23 ms kernel timed without a drain read 595x too fast.

### The warmup claim missed in both directions

BD002 blends two costs under one number.

| cost | measured | removable by a warmup loop? |
|---|---|---|
| first call of a *new shape*, warm process | **1.03–2.36x** | yes |
| first GPU kernel in a *fresh process* | **1514x** | no — paid before the function runs |
| *claimed* | 10–100x | — |

cuBLAS load and kernel-module load are deferred until the first GEMM, so in a fresh process that call really is ~1500x a steady-state one. The folklore number was not invented; it was attached to the wrong thing. **The rule's remedy is still right. Only its magnitude was wrong.**

### One number was just wrong, and no machine was needed to catch it

> "Gen5 read as Gen1 is a 4x error"

Gen1 carries 250 MB/s per lane (2.5 GT/s, 8b/10b). Gen5 carries 3938.5 MB/s (32 GT/s, 128b/130b). The ratio is **15.75**. 3.94 is Gen3 over Gen1.

That had been sitting in a docstring about *getting denominators wrong*, for as long as the rule existed, checkable with a division by anybody at any time. `pcie_generation_error()` now computes it, so the next reader can redo the arithmetic instead of trusting prose.

### And a methodology trap this had to survive

Repeating an un-drained measurement without draining between rounds lets the previous round's backlog land in the next one. Five rounds of identical work:

```
no drain between rounds:  174.6  231.9  218.9  232.2  226.6   ms   (1.30x drift)
drain between rounds:     175.5  180.0  172.7  178.5  182.4   ms   (1.04x)
```

A 30% drift that looks like nothing. The first attempt at demonstrating this **failed to reproduce**, at K=64 — and the reason is the same queue depth that governs everything else: 64 launches of a 371 µs kernel is 24 ms of backlog and the queue holds a thousand launches, so nothing spilled. It needed K past the knee.

## It closed one of the previous project's open verdicts

corpusaudit left four findings `unclear`, each with the measurement that would settle it. One of those measurements runs here.

**`pytorch/benchmarks/inference/server.py:198`** — `torch.load(path, mmap=True, map_location='cuda')` timed with no drain. With a drain over without: **0.991**. The copies out of mmapped host memory block the host, the un-drained number was already right, and that verdict resolves to **`false_positive`**.

**The flashinfer pair stays `unclear`.** What ran here is the *shape* — a handful of small H2D copies plus one small kernel, measured once, understating by 1.17x — not flashinfer. A reconstruction is not the thing, and 1.17x divided by `NUM_LAYERS` bounds the effect on their headline without settling their verdict.

## The device: `magnitude`

Every number a diagnostic asserts carries a support status.

```python
SUPPORT = ("measured", "witnessed", "derived", "folklore", "retracted")
```

| status | means |
|---|---|
| `measured` | a dataset here backs it, and the claim survived |
| `witnessed` | seen once, on one machine, in a named incident. n=1 |
| `derived` | follows from a published spec; checkable with arithmetic |
| `folklore` | believed, widely repeated, no evidence here |
| `retracted` | asserted, then measured, and the measurement contradicted it |

**The teeth.** `Magnitude.__post_init__` refuses the dishonest combinations: `measured` without a dataset, `folklore` that has been measured, and a `retracted` whose measurement actually agreed.

That last one caught a real error in my first version of the invariant. It tested non-containment — and 5-50x sits comfortably *inside* an observed 0.98-595x, so it rejected both genuine retractions. The claim is not wrong about any single case. It is wrong about **being a range**: a factor-of-10 window offered as the description of a factor-of-600 spread. The criterion is now spread, compared as a ratio because these are all multiplicative:

```python
understates_spread_by = observed.span / claimed.span    # BD001: 61x
```

**And the invariant that changed the code:** `knowable_from_source` gates `quoted_in_message`. BD001's factor is device time over host enqueue time. Neither appears in the text a linter reads, so the number may not appear in the message however well measured it is. Five of the six magnitudes are in that category, and none of them is quoted any more.

## The audit of all six

| rule | claimed | support | observed |
|---|---|---|---|
| BD001 | 5-50x | **retracted** | 0.98–595x, understates the spread 61x |
| BD002 | 10-100x | **retracted** | 1.03x warm / 1514x cold, understates 147x |
| BD010 | 2-3x | witnessed | n=1; also `unreachable` in public code |
| BD020 | 4x | derived | **wrong**: the spec says 15.75x |
| BD003 | 3x | folklore | rhetorical; removed |
| timing | 2-10x | folklore | a blend of two ranges now known; removed |

**Nothing is `measured`.** That is not an oversight — a magnitude only earns `measured` if the claim survived the measurement, and both of the two that were measured here failed.

## The ratchet

`tests/test_registry.py` has two halves, and the second is the inversion that makes it work.

**Messages.** Every numeric magnitude in a user-visible `Finding(message=...)` must be registered, marked `quoted_in_message`, and `knowable_from_source`. That set is currently empty, so the test says: no unregistered number reaches a user. Adding one costs a registry entry with a support status.

**Docstrings.** Every `claimed_text` in the registry must still appear *somewhere* in benchdoctor's source — the opposite of what a normal test asserts. The retracted numbers are the record. Deleting "5-50x" along with its retraction note would erase the evidence that anyone ever claimed it, and a silent edit is how a wrong number becomes a number nobody remembers being wrong.

Exemptions are named one at a time with a reason and printed by `drainlag claims`, because an exemption list is exactly where a check like this quietly stops working. There is one: BD021's `100%` is the value being clamped to, quoted from the code under analysis.

## The same constant, a third time: what the launch gap is worth

The 9.6 µs is not only a measurement artefact. It is time the device spends idle, and CUDA graphs remove it — so the same number has a third reading, and this one is worth speed instead of costing credibility.

22 cells, eager loop against a captured graph replayed once:

| N | true kernel | gap | speedup |
|---|---|---|---|
| 128 | 1.87–2.69 µs | 7.6–8.2 µs | **4.02–5.36x** |
| 256 | 2.95–3.98 µs | 6.3–8.2 µs | **2.97–3.41x** |
| 512 | 8.25–9.32 µs | 2.1–2.9 µs | 1.25–1.32x |
| 1024 | 53.2–54.4 µs | 2.3–2.5 µs | 1.04–1.05x |
| 4096 | 2.88–2.89 ms | 4.9–5.3 µs | 1.00x |

**Where the un-drained timer lies most, a graph helps least** — a 2.9 ms kernel reads 275x fast on an un-drained clock and gains nothing from a graph, because the host was never the bottleneck.

`speedup = (kernel + gap) / kernel` is algebra, asserted to floating point by a test. The content is the gap, and it has two measured bands: **6.3–8.2 µs** while the kernel is shorter than the enqueue cost, **2.1–3.1 µs** once the host stays ahead. A predictive form was written and not shipped — 136% worst residual, so the bands are what gets published.

### And you cannot tell which case you are in from eager profiling

The model this measurement was written with, before running it, used the CUDA-event device time: `max(1, host/device)`. It predicts **1.00** where the truth is **5.36**.

Because **CUDA events around an eager loop measure the pipeline, not the kernel** — they include the idle the device spends waiting for the next submission. At N=128 they report 9.8 µs for a kernel a graph replays in **1.90 µs**: a 5.15x overstatement.

So *"we profiled it, the kernel takes 10 µs, launch overhead is about 9, so maybe 50%"* is wrong twice: the kernel is 1.9 µs and the overhead is 5x it. The failed model is still in the shipped data — deleting it would hide the finding.

**It also caught an error in this repository's own numbers.** `sweep.py`'s `kernel_us` is that same pipeline rate. For device-bound cells the difference is a few percent; for the host-bound cells it is 5x, and an earlier draft called them "kernels from 9 to 15 µs". They are 1.9–9.3 µs kernels. Corrected in place with the correction left visible.

Capture costs 7.3 ms the first time in a process and ~0.2 ms + 7.5 µs/launch after. Where graphs help, that pays back in **one or two replays**.

Full detail: [`docs/graphs.md`](docs/graphs.md).

## What this does not show

- **One card, one OS, one torch build.** The 9.6 µs enqueue cost is a PyTorch-on-Windows number and will be lower elsewhere; every factor scales inversely with it. The *identity* and the *regimes* transfer. The absolute times do not.
- **One kernel type.** All of it is `torch.mm` at various sizes. A kernel with different launch-argument sizes or a graph-captured replay would have a different host cost, and that changes every factor.
- **The `2-3x` will probably never be measured.** There is no `cutlass_profiler` here, and the corpus audit found none in 166 public shell scripts. Some magnitudes in a tool like this are simply unmeasurable from public evidence, and saying `witnessed` is the honest end state rather than a placeholder.
- **`torch.compile` was not measured.** Triton does not install on this platform, so the third warmup cause has no number — recorded as an error in the dataset rather than omitted.
- **The graph result is one kernel type.** All of it is `torch.mm`; a kernel PyTorch dispatches differently has a different host cost and a different crossover. Nothing here measures what a dynamic shape costs in re-capture, which is why real serving stacks bucket shapes.
- **The gap floor is described, not explained.** 2.1–3.1 µs survives even when the host is far ahead of the device, and this measurement says what it is rather than why.
- **The saturation sweep uses one kernel size.** The knee at K≈2048 is where *this* kernel fills the queue; the depth is a driver property but the K at which it matters depends on the kernel.

## Layout

```
drainlag/
  magnitude.py     ★ the device: a claimed magnitude and its support
  law.py             the identity, the host regimes, the crossover, the gap bands
  claims.py          the registry: all six magnitudes, plus the PCIe arithmetic
  data.py            loaders for the four committed datasets
  report.py  cli.py

data/sweep.json      39 cells, three clocks each
data/claims.json     saturation, contamination, warmup, the two verdicts
data/context.json    five fresh processes: context and first-kernel cost
data/graphs.json     22 eager-vs-CUDA-graph cells: what the gap is worth
measure/             the scripts that produced them (needs torch + a GPU)
docs/                magnitude.md, regimes.md, graphs.md
```

## Running it

```bash
pip install -e ".[dev]"
drainlag report
drainlag law               # the identity, cell by cell, with residuals
drainlag saturation        # why a longer loop is safer
drainlag graphs            # what the launch gap is worth
drainlag claims            # the six magnitudes and their support
drainlag check             # invariants; this is what CI runs
pytest -q                  # 38 tests, no GPU needed
```

Reproducing the data:

```bash
pip install -e ".[measure]"
python -X utf8 measure/sweep.py data/sweep.json --demo-contamination
python -X utf8 measure/claims.py data/claims.json
python -X utf8 measure/context_init.py data/context.json
python -X utf8 measure/graphs.py data/graphs.json
```

## Licence

MIT.
