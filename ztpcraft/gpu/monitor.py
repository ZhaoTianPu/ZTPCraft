from __future__ import annotations

import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)


class monitor_gpu:
    """
    Context manager for logging GPU memory usage.

    Example
    -------
    >>> with monitor_gpu("floquet"):
    ...     run_simulation()
    """

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

    def __enter__(self):
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
                        mem.used / 1024**3,
                    )
                )

                time.sleep(self.interval)

        self._thread = threading.Thread(
            target=worker,
            daemon=True,
        )
        self._thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        self._thread.join()

        df = pd.DataFrame(
            self._records,
            columns=["time_s", "memory_gb"],
        )

        csv_path = self.output_dir / f"{self.name}.csv"
        png_path = self.output_dir / f"{self.name}.png"

        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(6, 3))
        plt.plot(df["time_s"], df["memory_gb"])
        plt.xlabel("Time (s)")
        plt.ylabel("Memory (GB)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        peak = df["memory_gb"].max()

        print(
            f"[GPU monitor] peak={peak:.2f} GB | "
            f"csv={csv_path} | "
            f"plot={png_path}"
        )

        return False