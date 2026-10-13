"""Autonomous CFX Studio - Background Job and Task Management.
Provides threaded background execution, queue dispatch, cancellation, and progress telemetry.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from autonomous_cfx.core.logger import logger


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobRecord:
    """Metadata and execution tracking for a dispatched CFX job."""
    job_id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0.0 to 100.0
    current_stage: str = "Initialized"
    started_at: float = 0.0
    finished_at: float = 0.0
    error_message: Optional[str] = None
    result: Optional[Any] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


class JobRunner:
    """Executes asynchronous CFX jobs in background worker threads."""

    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit_job(
        self,
        name: str,
        target_fn: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[dict] = None,
        on_complete: Optional[Callable[[JobRecord], None]] = None,
    ) -> JobRecord:
        """Submit a task for background execution."""
        kwargs = kwargs or {}
        job_id = str(uuid.uuid4())[:8]

        job = JobRecord(
            job_id=job_id,
            name=name,
            status=JobStatus.PENDING,
        )
        cancel_event = threading.Event()

        with self._lock:
            self._jobs[job_id] = job
            self._cancel_flags[job_id] = cancel_event

        def worker():
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            logger.info(f"[JobRunner] Job '{name}' ({job_id}) started.")

            try:
                # Inject cancel flag if target supports it
                res = target_fn(*args, **kwargs)
                if cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.current_stage = "Cancelled by user"
                    logger.info(f"[JobRunner] Job '{name}' ({job_id}) was cancelled.")
                else:
                    job.status = JobStatus.COMPLETED
                    job.progress = 100.0
                    job.current_stage = "Completed"
                    job.result = res
                    logger.info(f"[JobRunner] Job '{name}' ({job_id}) completed successfully.")
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.current_stage = "Failed"
                logger.error(f"[JobRunner] Job '{name}' ({job_id}) failed: {e}", exc_info=True)
            finally:
                job.finished_at = time.time()
                if on_complete:
                    try:
                        on_complete(job)
                    except Exception as e_cb:
                        logger.error(f"[JobRunner] on_complete callback failed: {e_cb}")

        th = threading.Thread(target=worker, daemon=True, name=f"CFXJob_{job_id}")
        self._threads[job_id] = th
        th.start()
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Signal job cancellation."""
        with self._lock:
            if job_id in self._cancel_flags:
                self._cancel_flags[job_id].set()
                logger.info(f"[JobRunner] Cancellation signaled for job {job_id}")
                return True
        return False

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch current job state."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[JobRecord]:
        """List all tracked jobs."""
        with self._lock:
            return list(self._jobs.values())
