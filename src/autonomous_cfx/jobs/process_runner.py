"""Autonomous CFX Studio - Multi-Process Job Runner & Crash Recovery.
Executes CFX simulations in isolated OS child processes with clean cancellation and crash recovery.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from autonomous_cfx.core.logger import logger


class ProcessJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    CRASHED = "CRASHED"


@dataclass
class ProcessJobRecord:
    """Tracking record for an isolated worker process."""
    job_id: str
    name: str
    pid: Optional[int] = None
    status: ProcessJobStatus = ProcessJobStatus.PENDING
    progress: float = 0.0
    stage: str = "Initialized"
    started_at: float = 0.0
    finished_at: float = 0.0
    exitcode: Optional[int] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None


def _worker_bootstrap(
    target_fn: Callable,
    args: tuple,
    kwargs: dict,
    ipc_queue: mp.Queue,
    cancel_event: mp.Event,
):
    """Entrypoint running inside the child OS process."""
    pid = os.getpid()
    ipc_queue.put({"type": "started", "pid": pid})

    def progress_proxy(stage: str, cur: int, total: int, extra: dict = None):
        if cancel_event.is_set():
            return
        pct = float(cur) / float(max(1, total)) * 100.0
        ipc_queue.put({"type": "progress", "stage": stage, "percent": pct, "extra": extra or {}})

    try:
        # Inject progress_callback and cancel_check if target supports them
        import inspect
        try:
            sig = inspect.signature(target_fn)
            if "progress_callback" in sig.parameters:
                kwargs["progress_callback"] = progress_proxy
            if "cancel_check" in sig.parameters:
                kwargs["cancel_check"] = cancel_event.is_set
        except Exception:
            if hasattr(target_fn, "__code__"):
                if "progress_callback" in target_fn.__code__.co_varnames:
                    kwargs["progress_callback"] = progress_proxy
                if "cancel_check" in target_fn.__code__.co_varnames:
                    kwargs["cancel_check"] = cancel_event.is_set

        res = target_fn(*args, **kwargs)
        if cancel_event.is_set():
            ipc_queue.put({"type": "cancelled"})
        else:
            ipc_queue.put({"type": "success", "result": res})
    except Exception as e:
        tb = traceback.format_exc()
        ipc_queue.put({"type": "exception", "error": str(e), "traceback": tb})
    finally:
        ipc_queue.put({"type": "done"})


class MultiProcessJobRunner:
    """Manages process lifecycle, IPC queue polling, clean cancellation, and crash detection."""

    def __init__(self):
        self._jobs: Dict[str, ProcessJobRecord] = {}
        self._processes: Dict[str, mp.Process] = {}
        self._queues: Dict[str, mp.Queue] = {}
        self._cancel_flags: Dict[str, mp.Event] = {}

    def submit_process_job(
        self,
        name: str,
        target_fn: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> ProcessJobRecord:
        """Launch target function in an isolated child process."""
        kwargs = kwargs or {}
        job_id = str(uuid.uuid4())[:8]

        job = ProcessJobRecord(
            job_id=job_id,
            name=name,
            status=ProcessJobStatus.PENDING,
            started_at=time.time(),
        )

        ipc_queue = mp.Queue()
        cancel_event = mp.Event()

        proc = mp.Process(
            target=_worker_bootstrap,
            args=(target_fn, args, kwargs, ipc_queue, cancel_event),
            name=f"CFXProcess_{job_id}",
            daemon=True,
        )

        self._jobs[job_id] = job
        self._processes[job_id] = proc
        self._queues[job_id] = ipc_queue
        self._cancel_flags[job_id] = cancel_event

        proc.start()
        job.pid = proc.pid
        job.status = ProcessJobStatus.RUNNING
        logger.info(f"[MultiProcessJobRunner] Launched separate worker process '{name}' (PID: {proc.pid}, Job: {job_id})")
        return job

    def poll_job(self, job_id: str) -> ProcessJobRecord:
        """Poll child process queue, update telemetry, and detect worker crashes."""
        job = self._jobs.get(job_id)
        if not job or job.status in (
            ProcessJobStatus.COMPLETED,
            ProcessJobStatus.FAILED,
            ProcessJobStatus.CANCELLED,
            ProcessJobStatus.CRASHED,
        ):
            return job

        proc = self._processes.get(job_id)
        q = self._queues.get(job_id)

        # Read available IPC queue messages
        if q:
            while not q.empty():
                try:
                    msg = q.get_nowait()
                    m_type = msg.get("type")
                    if m_type == "started":
                        job.pid = msg.get("pid")
                    elif m_type == "progress":
                        job.stage = msg.get("stage", "")
                        job.progress = msg.get("percent", 0.0)
                    elif m_type == "success":
                        job.status = ProcessJobStatus.COMPLETED
                        job.result = msg.get("result")
                        job.progress = 100.0
                    elif m_type == "exception":
                        job.status = ProcessJobStatus.FAILED
                        job.error_message = msg.get("error")
                    elif m_type == "cancelled":
                        job.status = ProcessJobStatus.CANCELLED
                except Exception:
                    break

        # Check process liveness
        if proc and not proc.is_alive():
            job.finished_at = time.time()
            job.exitcode = proc.exitcode

            if job.status == ProcessJobStatus.RUNNING:
                if proc.exitcode == 0:
                    job.status = ProcessJobStatus.COMPLETED
                elif proc.exitcode is not None and proc.exitcode != 0:
                    # Child process crashed or was killed abnormally!
                    job.status = ProcessJobStatus.CRASHED
                    job.error_message = f"Worker process crashed with exit code {proc.exitcode}."
                    logger.error(f"[MultiProcessJobRunner] Worker crash detected for Job {job_id}: Exit code {proc.exitcode}")

        return job

    def cancel_process_job(self, job_id: str, timeout: float = 1.0) -> bool:
        """Clean cancellation of child process with grace period and fallback kill."""
        proc = self._processes.get(job_id)
        flag = self._cancel_flags.get(job_id)
        job = self._jobs.get(job_id)

        if not proc or not proc.is_alive():
            if job:
                job.status = ProcessJobStatus.CANCELLED
            return True

        logger.info(f"[MultiProcessJobRunner] Signalling clean cancellation to PID {proc.pid}...")
        if flag:
            flag.set()

        # Give process a moment to exit cleanly
        proc.join(timeout=timeout)

        if proc.is_alive():
            logger.warning(f"[MultiProcessJobRunner] Process PID {proc.pid} unresponsive; terminating...")
            proc.terminate()
            proc.join(timeout=0.5)

            if proc.is_alive():
                logger.warning(f"[MultiProcessJobRunner] Force killing PID {proc.pid}...")
                proc.kill()
                proc.join(timeout=0.5)

        if job:
            job.status = ProcessJobStatus.CANCELLED
            job.exitcode = proc.exitcode
            job.finished_at = time.time()
            logger.info(f"[MultiProcessJobRunner] Job {job_id} successfully cancelled.")
        return True
