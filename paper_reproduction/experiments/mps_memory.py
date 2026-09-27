"""Process-local MPS memory sampling; peaks are sampled, not allocator peaks."""
import json
import math
import os
import threading
import time
from pathlib import Path


class MPSMemoryMonitor:
    def __init__(self, path, interval=0.5, backend=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("memory interval must be finite and positive")
        if backend is None:
            import torch
            backend = torch.mps
        self.backend = backend
        self.path = Path(path)
        self.interval = interval
        self.stop = threading.Event()
        self.error = None
        self.samples = 0
        self.peak_tensor = self.peak_driver = 0

    def sample(self):
        tensor = self.backend.current_allocated_memory()
        driver = self.backend.driver_allocated_memory()
        self.samples += 1
        self.peak_tensor = max(self.peak_tensor, tensor)
        self.peak_driver = max(self.peak_driver, driver)
        row = dict(time_unix=time.time(), elapsed_seconds=time.monotonic() - self.started,
                   pid=os.getpid(), device="mps", tensor_bytes=tensor, driver_bytes=driver,
                   recommended_max_bytes=self.recommended,
                   sampled_peak_tensor_bytes=self.peak_tensor,
                   sampled_peak_driver_bytes=self.peak_driver)
        self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()

    def loop(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except Exception as exc:
            self.error = exc

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x")
        self.started = time.monotonic()
        try:
            self.recommended = self.backend.recommended_max_memory()
            self.sample()
        except BaseException:
            self.handle.close()
            raise
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        try:
            self.sample()
        except Exception as error:
            self.error = error
        finally:
            self.handle.close()
        summary = dict(device="mps", pid=os.getpid(), samples=self.samples,
                       interval_seconds=self.interval,
                       sampled_peak_tensor_bytes=self.peak_tensor,
                       sampled_peak_driver_bytes=self.peak_driver,
                       recommended_max_bytes=self.recommended,
                       status="failed" if exc_type or self.error else "completed",
                       error=str(exc or self.error) if exc_type or self.error else None,
                       scope="worker process, all folds and SF prefix layers; includes loading/evaluation",
                       peak_kind="sampled lower bound; brief peaks may be missed")
        temporary = self.path.with_suffix(".summary.json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(self.path.with_suffix(".summary.json"))
        print("MPS memory: " + json.dumps(summary), flush=True)
        if self.error and exc_type is None:
            raise RuntimeError("MPS memory sampling failed") from self.error
