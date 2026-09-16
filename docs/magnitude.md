# `magnitude` — what entitles a diagnostic to quote a number

## The gap

Twelve tools before this one qualified other people's figures. Every one of them printed messages like these:

```
"this times the launch queue, not the kernel -- usually 5-50x too fast"
"one-time costs that can be 10-100x a steady-state iteration"
"verification runs inside the timed region and reports GFLOPs 2-3x low"
"Gen5 read as Gen1 is a 4x error"
```

Four numbers a reader will act on. It costs nothing to type one, the sentence reads better with it, and nobody asks where it came from. None of the four had ever been measured. One is arithmetically wrong.

Eleven generations of device, and this is the first pointed at the tool's own prose:

| gen | device | qualifies |
|---|---|---|
| 1 | `noise_basis` | the threshold |
| 2 | `basis` | each number's origin |
| 3 | aggregation contract | what arithmetic a column supports |
| 4 | evidence / closure | whether the input can answer the question |
| 5 | `fidelity` | how far a model sits from the machine |
| 6 | attribution | whose property a hit rate is |
| 7 | external authority | whether upstream agrees |
| 8 | twin control | whether the effect survives replication |
| 9 | `envelope` | whether *this output* was validated |
| 10 | `warrant` | whether the rule has ever been right |
| **11** | **`magnitude`** | **whether the number the rule quotes is supported** |

## Five supports

```python
SUPPORT = ("measured", "witnessed", "derived", "folklore", "retracted")
```

| status | meaning |
|---|---|
| `measured` | a dataset in this repository backs it, and the claim survived it |
| `witnessed` | seen once, on one machine, in an incident that is named. n=1 |
| `derived` | follows from a published specification; checkable with arithmetic |
| `folklore` | believed, widely repeated, no evidence here |
| `retracted` | asserted, then measured, and the measurement contradicted it |

`witnessed` exists because "I hit this bug and it was bad" is a real and common basis for writing a rule, and refusing it would just push those claims into `folklore`, where they do not belong. It says n=1 out loud instead.

`derived` exists because one of the six was neither measured nor believed — it was arithmetic, and the arithmetic was wrong. That is a distinct failure mode and it needs its own name, because it is the only one a reader could have caught without a machine.

## The invariant that caught my own first draft

The first version of the retraction check tested **non-containment**: a claim is retracted if the observed range does not contain it.

It rejected both real retractions. 5-50x sits comfortably inside an observed 0.98-595x, and 10-100x inside 1.03-1514x. By that test both claims were confirmations.

They are not. The claim is not wrong about any single case — pick a 200 µs kernel and 5-50x is fine. It is wrong about **being a range**: a factor-of-10 window offered as the description of a factor-of-600 spread. So the criterion is spread, compared as a ratio because every magnitude here is multiplicative:

```python
@property
def span(self) -> float:
    return self.hi / self.lo

understates_spread_by = observed.span / claimed.span
```

BD001: 607 / 10 = **61x**. BD002: 1470 / 10 = **147x**.

A claim now earns `retracted` by missing the observation *or* by understating its spread by at least `SPAN_TOLERANCE` (3x). And a `retracted` whose measurement genuinely agreed still raises, because labelling that a retraction is theatre.

## The invariant that changed the code

```python
if self.quoted_in_message and not self.knowable_from_source:
    raise ValueError(...)
```

BD001's factor is `device_us_per_launch / host_us_per_launch`. BD002's depends on whether this *process* has already loaded cuBLAS. BD010's is the profiler's verification cost relative to the kernel's. None of those appears in the text a linter reads.

So the number may not be in the message, however well measured it is. **A static checker can identify a shape. It cannot know the magnitude, because the magnitude is a property of the machine** — and quoting one anyway is the most reasonable-looking way to be wrong.

Five of the six are in that category. The exception is BD020's, a ratio between two published PCIe generations, and a test asserts it is the only one.

## The ratchet, and its inversion

`tests/test_registry.py`:

**Messages** — every numeric magnitude in a user-visible `Finding(message=...)` must be registered, `quoted_in_message`, and `knowable_from_source`. That set is empty today, so the test reads: no unregistered number reaches a user. Adding one costs a registry entry with a support status.

**Docstrings** — every `claimed_text` in the registry must still appear *somewhere* in benchdoctor's source. This is the opposite of what a normal test asserts, and it is the important half. The retracted numbers are the record; deleting "5-50x" along with its retraction note would erase the evidence that anyone ever claimed it. A silent edit is how a wrong number becomes a number nobody remembers being wrong.

**Exemptions** are named one at a time with a reason and printed by `drainlag claims`, because an exemption list is exactly where a check like this quietly stops working. There is one: BD021's `100%`, which is the value being clamped to — quoted from the code under analysis, not asserted about it.

## What a `measured` magnitude would look like

Nothing here has one, so it is worth saying what it would take: a claim, a dataset in this repository, and an observed range that the claim actually describes — same rough centre, same rough spread.

Both of the two magnitudes that got measured failed. `measured = 0` is a result, not a gap.
