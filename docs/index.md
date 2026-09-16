# drainlag

**My own linter told people a missing `synchronize()` makes a benchmark read "usually 5-50x too fast". Measured: 0.98x to 595x — and the factor is not a property of the code.**

## Three numbers

**0.98x to 595x.** 39 cells on one consumer GPU. The claimed 5-50x band is one slice of that, presented as the whole of it.

**9.6 µs.** The host's cost to enqueue one launch. A kernel faster than that drains before the host can refill, so no backlog forms and **the un-drained number is already right**. There is a floor below which the bug does not exist, and the rule never mentioned it.

**1.06x at K=16384.** Same kernel, same missing sync: 64 iterations is 5.83x too fast, 16384 is 1.06x. Past the launch queue's depth the host blocks and the measurement converges to correct. **A longer loop is a safer loop** — and the rules that look for this only fire when there is a loop.

## The identity

An un-drained timer measures the *host*. A drained one measures the *device*. So the factor is

```
device_us_per_launch / host_us_per_launch
```

Worst residual 18.2% over the 17 cells with a factor above 5x; median under 5%. Neither term appears in the source a linter reads, which is why the number cannot be in the message.

## The device: `magnitude`

Every number a diagnostic asserts carries a support status — `measured`, `witnessed`, `derived`, `folklore`, `retracted` — and `knowable_from_source` gates `quoted_in_message`.

The audit of all six numbers in benchdoctor: **2 retracted, 2 folklore, 1 witnessed, 1 derived and arithmetically wrong** ("Gen5 read as Gen1 is a 4x error" — the spec says 15.75x). Nothing is `measured`, because a magnitude only earns that if the claim survives.

## It also closed an open verdict

The previous project left four findings `unclear`, each with the measurement that would settle it. One runs here: `torch.load(mmap=True, map_location='cuda')` timed with and without a drain came out at **0.991**, so the copy blocks the host and that verdict resolves to `false_positive`. The flashinfer pair stays `unclear` — what ran here is the shape, not flashinfer.

## A third reading: CUDA graphs

The 9.6 µs is time the device spends idle, and graphs remove it. 22 cells: **4.02–5.36x** on 1.9–2.7 µs kernels, **1.00x** on 2.9 ms ones. Where the un-drained timer lies most, a graph helps least.

`speedup = (kernel + gap)/kernel` is algebra; the content is that the gap is 6.3–8.2 µs while the kernel is shorter than the enqueue cost and 2.1–3.1 µs once it is longer.

**And you cannot predict it from eager profiling.** CUDA events around an eager loop measure the pipeline, not the kernel — 9.8 µs reported for a 1.90 µs kernel. The model this measurement was written with used that number and predicted no speedup where the truth was 5.36x. That failed model is still in the data.

## Read next

- [`graphs.md`](graphs.md) — what the launch gap is worth, and why eager profiling cannot tell you
- [`magnitude.md`](magnitude.md) — the device, and the invariant that caught my own first version of it
- [`regimes.md`](regimes.md) — the crossover, the queue knee, and why a longer loop is safer

## Sister projects

[nodebench](https://github.com/liu-perf/nodebench) · [benchdoctor](https://github.com/liu-perf/benchdoctor) · [tracedoctor](https://github.com/liu-perf/tracedoctor) · [regressiondoctor](https://github.com/liu-perf/regressiondoctor) · [fitdoctor](https://github.com/liu-perf/fitdoctor) · [telemetrydoctor](https://github.com/liu-perf/telemetrydoctor) · [servedoctor](https://github.com/liu-perf/servedoctor) · [kvsched](https://github.com/liu-perf/kvsched) · [prefixkv](https://github.com/liu-perf/prefixkv) · [blocktax](https://github.com/liu-perf/blocktax) · [queuebound](https://github.com/liu-perf/queuebound) · [corpusaudit](https://github.com/liu-perf/corpusaudit)

Twelve of them qualified other people's numbers. This one qualifies the numbers those tools assert.
