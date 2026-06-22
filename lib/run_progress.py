"""Console progress + heartbeat for long parallel batch jobs."""

from __future__ import annotations

import threading
import time


class RunProgress:
    """Print fraction done, elapsed time, ETA, and periodic heartbeats."""

    def __init__(
        self,
        total: int,
        *,
        label: str = "Progress",
        heartbeat_sec: float = 30.0,
    ) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.heartbeat_sec = max(0.0, float(heartbeat_sec))
        self.done = 0
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._last_name = "(starting)"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        print(
            f"[PROGRESS] {self.label}: starting 0/{self.total}",
            flush=True,
        )
        if self.heartbeat_sec > 0 and self.total > 1:
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()

    def step(self, item_name: str = "") -> None:
        with self._lock:
            self.done += 1
            if item_name:
                self._last_name = item_name
        self._emit()

    def finish(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = time.monotonic() - self._started
        print(
            f"[PROGRESS] {self.label}: finished {self.done}/{self.total} in "
            f"{_format_duration(elapsed)}",
            flush=True,
        )

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_sec):
            with self._lock:
                if self.done >= self.total:
                    break
            self._emit(waiting=True)

    def _emit(self, *, waiting: bool = False) -> None:
        with self._lock:
            done = self.done
            total = self.total
            last = self._last_name

        if total <= 0:
            return

        pct = 100.0 * done / total
        elapsed = time.monotonic() - self._started
        eta = ""
        if 0 < done < total:
            rate = elapsed / done
            eta = f" | ETA ~{_format_duration(rate * (total - done))}"

        status = "waiting on workers…" if waiting and done < total else "running"
        bar = _bar(pct)
        print(
            f"[PROGRESS] {self.label}: {bar} {done}/{total} ({pct:.0f}%) | "
            f"elapsed {_format_duration(elapsed)}{eta} | last: {last} | {status}",
            flush=True,
        )


def _format_duration(seconds: float) -> str:
    total_sec = int(seconds)
    minutes, sec = divmod(total_sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _bar(pct: float, width: int = 20) -> str:
    filled = min(width, int(width * pct / 100))
    return "[" + "#" * filled + "-" * (width - filled) + "]"
