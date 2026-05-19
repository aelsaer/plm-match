from __future__ import annotations

import os
import resource
import statistics
import sys
import threading


def peak_rss_mb() -> float:
    """Return peak resident set size for the current process in MiB."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def current_rss_mb() -> float:
    """Return current resident set size for the process in MiB."""
    if sys.platform.startswith("linux"):
        try:
            statm = PathLikeProcStatm.read_text()
        except OSError:
            statm = ""
        parts = statm.split()
        if len(parts) >= 2:
            return float(int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0))
    return peak_rss_mb()


class _ProcStatm:
    def read_text(self) -> str:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            return f.read()


PathLikeProcStatm = _ProcStatm()


class ResourceSampler:
    """Sample process RSS while a scoped online/query block is running."""

    def __init__(self, *, interval_s: float = 0.25, scope: str = "query") -> None:
        self.interval_s = max(0.01, float(interval_s))
        self.scope = str(scope)
        self.samples_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceSampler":
        self.sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2.0))
        self.sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def sample(self) -> None:
        self.samples_mb.append(float(current_rss_mb()))

    def summary_fields(self, *, prefix: str = "query_process") -> dict[str, object]:
        if not self.samples_mb:
            self.sample()
        avg = float(statistics.fmean(self.samples_mb))
        peak = float(max(self.samples_mb))
        return {
            f"{prefix}_avg_rss_mb": avg,
            f"{prefix}_ram_mb": avg,
            f"{prefix}_peak_sampled_rss_mb": peak,
            f"{prefix}_ram_samples": int(len(self.samples_mb)),
            f"{prefix}_ram_sample_interval_s": float(self.interval_s),
            "ram_scope": "average sampled RSS of the online query process; excludes offline index/map construction",
            "runtime_scope": "query localization/matching after descriptors are available",
        }


def query_process_resource_fields() -> dict[str, object]:
    sampler = ResourceSampler()
    sampler.sample()
    return {
        **sampler.summary_fields(),
        "ram_scope": "current RSS sampled at summary time; use ResourceSampler for query-process averages",
    }
