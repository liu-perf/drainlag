# The same constant, a third time: what the launch gap is worth

The host takes **9.6 µs** to enqueue one launch. `sweep.py` treats that as the thing that makes an un-drained timer wrong. But it is not only a measurement artefact — it is time the device spends idle, and CUDA graphs remove it.

So the same number has a third reading, and this one is worth money instead of costing credibility.

## What was measured

22 cells: matmuls from 128² to 4096², loops from 8 to 512, comparing an eager loop against a captured CUDA graph replayed once. Each cell records the eager baseline **before** capture, the graph, and the eager baseline **again after** — see [the ordering trap](#the-ordering-this-had-to-get-right).

| N | true kernel | gap | speedup |
|---|---|---|---|
| 128 | 1.87–2.69 µs | 7.6–8.2 µs | **4.02–5.36x** |
| 256 | 2.95–3.98 µs | 6.3–8.2 µs | **2.97–3.41x** |
| 512 | 8.25–9.32 µs | 2.1–2.9 µs | 1.25–1.32x |
| 1024 | 53.2–54.4 µs | 2.3–2.5 µs | 1.04–1.05x |
| 2048 | 370–372 µs | 2.3–3.1 µs | 1.01x |
| 4096 | 2.88–2.89 ms | 4.9–5.3 µs | 1.00x |

**Where the un-drained timer lies most, a graph helps least.** A 2.9 ms kernel makes an un-drained clock read 275x fast and a graph buys nothing, because the host was never the bottleneck there.

## The relation is algebra; the content is the gap

```
speedup = eager_per_launch / graph_per_launch = (kernel + gap) / kernel
```

That is definitional, and `test_the_speedup_relation_is_algebra_not_a_model` asserts it holds to floating point. Nothing is being fitted. Everything interesting is in what `gap` turns out to be:

- **kernel shorter than the enqueue cost** → the device is left waiting, and the gap tracks that cost: **6.3–8.2 µs**.
- **kernel longer** → the host stays ahead and the gap settles on a floor: **2.1–3.1 µs**.

Two measured bands, not a model. A predictive form — `max(floor, enqueue − kernel)` — was written and then **not shipped**: its worst residual is 136%, concentrated in the K=8 cells where the host timing is eight launches wide, and in the 2.9 ms kernels where a 5 µs gap is 0.17% of the measurement and past what a wall clock resolves. 18 of 22 cells sit inside their band; `drainlag graphs` prints the four that do not, with the reason for each.

## The finding: you cannot tell from eager profiling

This is the part worth carrying away.

The model this measurement was written with, before it ran, was `max(1, host_us / device_us)` using the CUDA-event device time. It predicts **1.00** where the truth is **5.36** — an 82% miss, on exactly the cells where the answer matters.

Because **CUDA events around an eager loop do not measure the kernel.** They measure the pipeline: the kernel plus the idle the device spends waiting for the next submission. At N=128 they report 9.8 µs for a kernel that a graph replay runs in **1.90 µs** — a 5.15x overstatement.

So the natural inference — *"we profiled it, the kernel takes 10 µs, launch overhead is ~9 µs so it's maybe 50%, not worth chasing"* — is wrong twice over: the kernel is 1.9 µs and the overhead is 5x the kernel, not half of it.

There is no way around it from eager measurements. To know what a graph will buy you, you have to capture one, or profile the kernel in isolation.

`predicted_speedup_from_event_timing` is still in the shipped data, deliberately. Deleting a model that failed hides the finding.

### And it caught an error in this repository's own numbers

`sweep.py` calls its event-measured quantity `kernel_us`. For the device-bound cells that is fine — a 2.4 µs gap against 55–2880 µs of kernel is a few percent. For the host-bound cells it is not, and an earlier draft of the README and of `regimes.md` described them as "kernels from 9 to 15 µs".

They are **1.9–9.3 µs kernels** whose pipeline the host gated to 9–15 µs. The floor is real, the identity holds, the conclusion is unchanged — but the sentence named the wrong quantity, and it took building a different measurement to notice. Both pages now carry the correction rather than a quiet edit.

## The ordering this had to get right

The first attempt measured eager **after** capturing the graph and got 21.4 µs per launch for a kernel `sweep.py` had put at 10.4 µs. Graph capture switches the caching allocator to a private memory pool and leaves it there, so the eager path afterwards is not the eager path from before.

A speedup computed against a contaminated baseline is the easiest way to publish a number twice as good as the truth, and nothing about the output would look unusual.

So every cell measures eager, captures, measures the graph, then measures eager **again**, and ships the drift. Measured across the grid: **0.92–1.17x**. Real but small — under proper warmup the 21.4 µs anomaly did not reproduce, so the original cause was insufficient warmup rather than the pool. Both facts are in the record.

## The practical part: when is it worth capturing?

Capture is not free, and it has two components:

- a **one-time** cost the first time a graph is captured in a process: **7.3 ms**;
- a per-capture cost of roughly 0.2 ms + 7.5 µs per launch captured (0.28 ms at K=8, 3.7 ms at K=512).

`breakeven_replays` divides the capture cost by the per-replay saving:

| N | K | speedup | break-even replays |
|---|---|---|---|
| 128 | 128 | 5.13x | **1.3** |
| 256 | 512 | 3.33x | **1.0** |
| 512 | 128 | 1.31x | 4.3 |
| 1024 | 512 | 1.04x | 3.0 |
| 4096 | 8 | 1.00x | 10.3 |

For the kernels where graphs help, capture pays for itself in **one or two replays**. Where they do not help the break-even is meaningless — it is dividing by noise.

The 103.8 at N=128 K=8 is the one-time 7.3 ms initialisation landing on the first cell of the run. It is left in the data rather than amortised away, because "the first graph in your process costs 7 ms" is a real thing to know.

## What this does not show

- **One kernel type, one card.** All of it is `torch.mm`. A kernel with larger launch arguments, or one that PyTorch dispatches differently, has a different host cost and therefore a different crossover.
- **A graph is not free elsewhere.** It fixes shapes and addresses. Nothing here measures the cost of the re-capture a dynamic shape forces, which is the reason real serving stacks bucket shapes.
- **The gap floor is not explained.** 2.1–3.1 µs of it survives even when the host is far ahead, and this measurement says what it is, not why.
- **`speedup` is measured against eager PyTorch on Windows.** The dispatch cost is the largest single term and it is the least portable one.
