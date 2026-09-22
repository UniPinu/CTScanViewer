"""Background jobs for the service (CONTEXT.md §4 Stage 6).

Everything the viewer triggers is slow: a fetch is tens of seconds of network,
preprocessing is seconds of CPU, registration is a minute or two. None of that
can happen inside a request, so it runs here and the viewer polls.

**One heavy job at a time.** A single worker thread, not a pool. The work is
bounded by network and by the disk cache, and two concurrent fetches would race
each other for the same cache budget and evict each other's series (see
`ingest/stream.py`). The existing viewer already assumes one fetch at a time;
this makes that a property of the server rather than of the browser.

§4 marks this as the MVP shape and a real queue (Redis/RQ or Celery) plus a
worker pool as the later step. The `Job` record is deliberately serialisable so
that swap does not change the API.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["queued", "running", "done", "error", "cancelled"]

#: Jobs older than this are forgotten, so a long-lived server does not grow
#: without bound. Results live on disk and outlive the job record.
JOB_TTL_SECONDS = 60 * 60 * 6


@dataclass
class Job:
    """One unit of background work, and everything the UI needs to narrate it."""

    id: str
    kind: str
    status: Status = "queued"
    stage: str = ""
    message: str = ""
    progress: float | None = None  # 0..1, or None when indeterminate
    patient_id: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: Any = None
    log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "patient_id": self.patient_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 1),
            "error": self.error,
            "result": self.result,
            "log": self.log[-40:],
        }


class JobRegistry:
    """Tracks jobs and runs them one at a time."""

    def __init__(self, ttl_seconds: int = JOB_TTL_SECONDS) -> None:
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ct-job")
        self._ttl = ttl_seconds

    def submit(self, kind: str, work: Callable[[Job], Any], patient_id: str | None = None) -> Job:
        """Queue `work`, which is handed the `Job` so it can report progress."""
        self._reap()
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, patient_id=patient_id)
        with self._lock:
            self._jobs[job.id] = job
            self._futures[job.id] = self._pool.submit(self._run, job, work)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def active(self) -> Job | None:
        """The job currently running, if any."""
        with self._lock:
            for job in self._jobs.values():
                if job.status == "running":
                    return job
        return None

    def busy(self) -> bool:
        with self._lock:
            return any(j.status in ("queued", "running") for j in self._jobs.values())

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job, work: Callable[[Job], Any]) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            job.result = work(job)
            job.status = "done"
            job.progress = 1.0
        except Exception as e:  # noqa: BLE001 - any failure must reach the UI
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            # The full traceback is worth keeping for the log panel; the UI shows
            # `error`, and a developer can expand the log for the rest.
            job.log.extend(traceback.format_exc().strip().splitlines()[-12:])
        finally:
            job.finished_at = time.time()

    def _reap(self) -> None:
        """Drop finished jobs past their TTL."""
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [
                job_id for job_id, job in self._jobs.items()
                if job.finished_at and job.finished_at < cutoff
            ]
            for job_id in stale:
                self._jobs.pop(job_id, None)
                self._futures.pop(job_id, None)


def progress_reporter(job: Job) -> Callable[[str, str], None]:
    """Adapter matching `SeriesCache.on_progress`, so the cache narrates itself."""

    def report(stage: str, message: str) -> None:
        job.stage = stage
        if message:
            job.message = message
            job.log.append(message)

    return report


#: One registry per process. The API imports this rather than constructing its own.
registry = JobRegistry()
