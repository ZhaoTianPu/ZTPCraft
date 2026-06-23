"""Host memory benchmarking helpers for JAX / dynamiqs workloads."""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar

import psutil

T = TypeVar("T")

_process = psutil.Process()


class MemoryStats(TypedDict):
    rss_before_mb: float
    rss_after_mb: float
    peak_rss_mb: float
    peak_delta_mb: float


def host_rss_mb() -> float:
    """Return current process RSS in megabytes."""
    return _process.memory_info().rss / 1e6


def sync_tree(x: Any) -> None:
    """Block until JAX / dynamiqs outputs in a pytree are ready."""
    if x is None:
        return
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
        return
    if hasattr(x, "propagators"):
        sync_tree(x.propagators)
    if hasattr(x, "states"):
        sync_tree(x.states)
    if isinstance(x, (tuple, list)):
        for item in x:
            sync_tree(item)


def array_bytes(x: Any) -> int:
    """Return nbytes of a JAX array, numpy array, or dynamiqs QArray."""
    if hasattr(x, "to_jax"):
        return int(x.to_jax().nbytes)

    import numpy as np

    return int(np.asarray(x).nbytes)


def theoretical_bytes(shape: tuple[int, ...], itemsize: int = 8) -> float:
    """Return dense array size in megabytes for the given shape and dtype width."""
    nbytes = itemsize
    for dim in shape:
        nbytes *= dim
    return nbytes / 1e6


class PeakRSSMonitor:
    """Poll process RSS in a background thread to capture peak usage."""

    def __init__(self, poll_interval_s: float = 5e-4):
        self.poll_interval_s = poll_interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            rss = host_rss_mb()
            if rss > self.peak_mb:
                self.peak_mb = rss
            time.sleep(self.poll_interval_s)

    def __enter__(self) -> PeakRSSMonitor:
        self.peak_mb = host_rss_mb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        rss = host_rss_mb()
        if rss > self.peak_mb:
            self.peak_mb = rss


def benchmark_memory(
    label: str,
    fn: Callable[[], T],
    *,
    warmup: bool = True,
    verbose: bool = True,
) -> tuple[T, MemoryStats]:
    """Measure peak host RSS while running ``fn``.

    Intended for CPU runs where process RSS is the relevant host-memory metric.
    """
    if warmup:
        sync_tree(fn())
        gc.collect()

    gc.collect()
    rss_before = host_rss_mb()

    with PeakRSSMonitor() as monitor:
        out = fn()
        sync_tree(out)

    gc.collect()
    rss_after = host_rss_mb()
    peak_rss = max(monitor.peak_mb, rss_after)
    peak_delta = peak_rss - rss_before

    stats: MemoryStats = {
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "peak_rss_mb": peak_rss,
        "peak_delta_mb": peak_delta,
    }

    if verbose:
        print(f"=== {label} ===")
        print(f"host RSS before     : {rss_before:8.1f} MB")
        print(f"host RSS after      : {rss_after:8.1f} MB")
        print(f"peak host RSS       : {peak_rss:8.1f} MB")
        print(f"peak above baseline : {peak_delta:8.1f} MB")

    return out, stats
