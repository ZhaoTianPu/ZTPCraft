from __future__ import annotations

import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[reportMissingTypeStubs]
import pynvml as _pynvml  # type: ignore[import-untyped]

nvmlInit = cast(Any, _pynvml.nvmlInit)
nvmlDeviceGetHandleByIndex = cast(Any, _pynvml.nvmlDeviceGetHandleByIndex)
nvmlDeviceGetMemoryInfo = cast(Any, _pynvml.nvmlDeviceGetMemoryInfo)

plt = cast(Any, plt)


class monitor_gpu:
    """
    Context manager for logging GPU memory usage.

    Example
    -------
    >>> with monitor_gpu("floquet"):
    ...     run_simulation()
    """

    name: str
    device: int
    interval: float
    output_dir: Path
    _records: list[tuple[float, float]]
    _running: bool
    _t0: float
    _thread: threading.Thread | None

    def __init__(
        self,
        name: str,
        *,
        device: int = 0,
        interval: float = 0.1,
        output_dir: str | Path = "gpu_logs",
    ):
        """Configure GPU memory logging for a named run.

        Parameters
        ----------
        name:
            Label used for output files (``{name}.csv`` and ``{name}.png``).
        device:
            CUDA device index to monitor.
        interval:
            Sampling period in seconds.
        output_dir:
            Directory where CSV and plot files are written.
        """
        self.name = name
        self.device = device
        self.interval = interval
        self.output_dir = Path(output_dir)
        self._records = []
        self._running = False
        self._t0 = 0.0
        self._thread = None

    def __enter__(self) -> monitor_gpu:
        nvmlInit()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._records = []
        self._running = True
        self._t0 = time.perf_counter()

        handle = nvmlDeviceGetHandleByIndex(self.device)

        def worker():
            while self._running:
                mem = nvmlDeviceGetMemoryInfo(handle)

                self._records.append(
                    (
                        time.perf_counter() - self._t0,
                        int(mem.used) / 1024**3,
                    )
                )

                time.sleep(self.interval)

        self._thread = threading.Thread(
            target=worker,
            daemon=True,
        )
        self._thread.start()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._running = False
        if self._thread is not None:
            self._thread.join()

        df = pd.DataFrame(
            self._records,
            columns=["time_s", "memory_gb"],  # type: ignore[arg-type]
        )

        csv_path = self.output_dir / f"{self.name}.csv"
        png_path = self.output_dir / f"{self.name}.png"

        df.to_csv(csv_path, index=False)  # type: ignore[reportUnknownMemberType]

        times = [t for t, _ in self._records]
        memory_gb = [m for _, m in self._records]

        plt.figure(figsize=(6, 3))  # type: ignore[reportUnknownMemberType]
        plt.plot(times, memory_gb)  # type: ignore[reportUnknownMemberType]
        plt.xlabel("Time (s)")  # type: ignore[reportUnknownMemberType]
        plt.ylabel("Memory (GB)")  # type: ignore[reportUnknownMemberType]
        plt.tight_layout()  # type: ignore[reportUnknownMemberType]
        plt.savefig(png_path, dpi=150)  # type: ignore[reportUnknownMemberType]
        plt.close()  # type: ignore[reportUnknownMemberType]

        peak = max(memory_gb, default=0.0)

        print(
            f"[GPU monitor] peak={peak:.2f} GB | "
            f"csv={csv_path} | "
            f"plot={png_path}"
        )

        return False