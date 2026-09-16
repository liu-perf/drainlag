# The regimes: where the bug lives, and where it does not

## The identity

```
factor = drained / undrained = device_us_per_launch / host_us_per_launch
```

Not a fitted model. An un-drained clock measures how long the host took to enqueue; a drained one measures how long the device took to run. The factor is the ratio of the two things those clocks are for.

Measured over 39 cells: worst residual **18.2%** across the 17 cells with a factor above 5x, median under 5%. Three examples, with the identity computed from the *measured* host cost rather than from a constant:

| N | kernel | K | host µs/launch | measured | identity | residual |
|---|---|---|---|---|---|---|
| 2048 | 371 µs | 64 | 8.64 | 43.09x | 43.06 | 0.1% |
| 4096 | 2.88 ms | 64 | 10.46 | 275.33x | 275.39 | 0.0% |
| 8192 | 23.0 ms | 4 | 38.78 | 594.63x | 593.66 | 0.2% |

Below a factor of 5 the residual grows to about half. That is not a failure of the identity, it is the identity being uninformative where there is no error to predict — 1.63 versus 1.14 is a 43% relative gap between two numbers that both mean "about right". `drainlag law` prints the split and `INFORMATIVE_RATIO` names the threshold.

## Two regimes, and both are about the host term

### enqueue-bound

While the launch queue has room, `host_us_per_launch` is a constant — PyTorch dispatch plus the driver enqueue, **9.6 µs** on this machine. The backlog grows, the un-drained clock never waits for it, and the factor is `device / 9.6 µs`.

This is where the error is real and can be enormous. It is also where the *floor* lives:

**A kernel faster than 9.6 µs cannot produce an error.** The device drains faster than the host can refill, no backlog forms at all, and the un-drained number is already the right one. Measured across every cell whose per-launch pipeline ran at 9–15 µs: **0.98x to 2.31x** — nothing.

> **Correction.** An earlier version of this page called those "kernels from 9 to 15 µs". They are not. `sweep.py` measures device time with CUDA events around an *eager* loop, which includes the idle the device spends waiting for the next submission — so it reports the pipeline rate, not the kernel. The CUDA-graph measurement in [`graphs.md`](graphs.md) submits once and shows those kernels are **1.9–9.3 µs**. The floor is real and the conclusion is unchanged; the sentence named the wrong quantity.

That floor is a large part of why BD001 was harmless on the public benchmark code it was audited against. The elementwise and attention-fragment kernels that fill a serving engine's `benchmarks/kernels/` directory sit under it.

### blocked

Past the queue's depth the host cannot enqueue without waiting. `host_us_per_launch` rises toward the device time and the factor collapses toward 1.

| K | 64 | 256 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| factor | 5.83 | 6.53 | 6.26 | **1.93** | 1.31 | 1.13 | **1.06** |
| host µs/launch | 9.55 | 8.51 | 8.87 | **28.68** | 42.13 | 48.93 | 52.19 |

The knee is between K=1024 and K=2048, consistent with the classic ~1024 pending-launch limit. The host cost jumping from 8.87 to 28.68 µs *is* the queue filling.

**So a longer loop makes a missing drain matter less.** Same kernel, same bug: 64 iterations is 483% wrong, 16384 iterations is 6% wrong.

And BD001 and BD002 only fire when there is a loop. The rule is most eager precisely where the bug is least harmful, and it never said so, because nobody had measured which way the curve went.

## The measurement trap

An un-drained measurement leaves the queue full. Run the next one without draining and it inherits the backlog. Five rounds of identical work, K=4096:

```
no drain between rounds:  174.6  231.9  218.9  232.2  226.6   ms   (1.30x drift)
drain between rounds:     175.5  180.0  172.7  178.5  182.4   ms   (1.04x)
```

A 30% drift, roughly monotone, with nothing about the output looking wrong. A sweep that forgets this produces a smooth, plausible, wrong curve — the exact failure mode this whole family of tools is about.

**The first attempt to demonstrate it failed.** At K=64 the contamination did not appear, and the reason is the queue depth again: 64 launches of a 371 µs kernel is 24 ms of backlog, the queue holds a thousand launches, and nothing spilled. The demo had to move past the knee. That failure is recorded in the measurement script's docstring rather than deleted, because "I predicted an effect, it did not show up, and here is why" is the part of a methodology note worth reading.

## What transfers and what does not

The 9.6 µs enqueue cost is PyTorch-on-Windows on one machine. Somewhere with a cheaper dispatch it will be lower, the crossover kernel size will move down, and every factor will scale up.

What transfers is the shape: a floor set by the host's enqueue cost, a ceiling set by the ratio of device time to it, and a collapse back to correctness past the queue depth. What does not transfer is any specific number — including all of the ones on this page.
