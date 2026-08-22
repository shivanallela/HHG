import time
from typing import Dict


class LatencyTracker:
    """Simple framework‑agnostic latency tracker.

    Usage example::
        tracker = LatencyTracker()
        tracker.start("embedding")
        # ... do embedding ...
        tracker.stop("embedding")
        print(tracker.elapsed("embedding"))
    """

    def __init__(self) -> None:
        self._start_times: Dict[str, float] = {}
        self._elapsed: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        if stage in self._start_times:
            raise RuntimeError(f"Stage '{stage}' has already been started.")
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        if stage not in self._start_times:
            raise RuntimeError(f"Cannot stop stage '{stage}' because it was never started.")
        start = self._start_times.pop(stage)
        elapsed = time.perf_counter() - start
        self._elapsed[stage] = self._elapsed.get(stage, 0.0) + elapsed

    def elapsed(self, stage: str) -> float:
        """Return elapsed time for *stage* in milliseconds.
        If the stage is still running, compute live elapsed time.
        If never started, returns 0.0.
        """
        if stage in self._start_times:
            return (time.perf_counter() - self._start_times[stage]) * 1000.0
        return self._elapsed.get(stage, 0.0) * 1000.0

    def total_elapsed(self) -> float:
        total = sum(self._elapsed.values())
        for start in self._start_times.values():
            total += time.perf_counter() - start
        return total * 1000.0

    def to_dict(self) -> Dict[str, float]:
        result = {stage: elapsed * 1000.0 for stage, elapsed in self._elapsed.items()}
        result["total_ms"] = self.total_elapsed()
        return result

    @property
    def stages(self) -> Dict[str, float]:
        return {stage: elapsed * 1000.0 for stage, elapsed in self._elapsed.items()}
