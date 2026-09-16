"""Loaders for the three committed datasets.

Everything this package concludes is re-derivable from these three files with
no GPU. Reproducing them needs a CUDA device and `measure/*.py`; reading them
back needs nothing. The split matters for the same reason it did in the
previous project: a result that can only be re-checked by re-running it is a
result nobody re-checks.

    data/sweep.json     39 cells: kernel size x loop count, three clocks each
    data/claims.json    saturation past the queue depth, the contamination
                        demo, the warmup costs, and the two verdict experiments
    data/context.json   five fresh processes: CUDA context and first-kernel cost
    data/graphs.json    22 eager-vs-CUDA-graph cells: what the launch gap costs
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

from .law import Cell, Fit, GraphCell

FILES = ("sweep", "claims", "context", "graphs")


def find_root(start: Optional[pathlib.Path] = None) -> pathlib.Path:
    here = start or pathlib.Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / "data" / "sweep.json").is_file():
            return cand
    raise FileNotFoundError("no data/sweep.json in any parent directory")


def load(root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    root = root or find_root()
    out = {}
    for name in FILES:
        path = root / "data" / f"{name}.json"
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def cells(doc: Dict[str, Any]) -> List[Cell]:
    return [Cell(n=c["n"], k=c["k"], kernel_us=c["kernel_us"],
                 undrained_ms=c["undrained_ms"], drained_ms=c["drained_ms"],
                 events_ms=c["events_ms"], ratio=c["ratio"])
            for c in doc["sweep"]["cells"]]


def fit(doc: Dict[str, Any]) -> Fit:
    return Fit(cells(doc))


def enqueue_us(doc: Dict[str, Any]) -> float:
    return doc["sweep"]["launch_us"]


def environment(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc["sweep"]["env"]


def saturation_rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return doc["claims"]["saturation"]["rows"]


def warmup(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc["claims"]["warmup"]


def context(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc["context"]["median"]


def contamination(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc["claims"]["contamination"]


def graph_cells(doc: Dict[str, Any]) -> List[GraphCell]:
    return [GraphCell(n=c["n"], k=c["k"],
                      host_us_per_launch=c["host_us_per_launch"],
                      event_us_per_launch=c["device_us_per_launch"],
                      true_kernel_us=c["true_kernel_us"],
                      gap_us=c["gap_us"],
                      eager_ms=c["eager_before_ms"], graph_ms=c["graph_ms"],
                      speedup=c["speedup"], eager_drift=c["eager_drift"],
                      capture_ms=c["capture_ms"],
                      breakeven_replays=c["breakeven_replays"])
            for c in doc["graphs"]["cells"]]


def verdicts(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {"torch_load": doc["claims"]["verdict_torch_load"],
            "plan_shape": doc["claims"]["verdict_plan_shape"]}
