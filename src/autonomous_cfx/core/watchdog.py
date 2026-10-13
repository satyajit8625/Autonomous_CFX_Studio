"""Autonomous CFX Studio - Process & Step Watchdog Runtime.
Guarantees deterministic timeout enforcement, child-process tree termination,
and structured incident reporting for hung or deadlocked tasks.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import psutil

from autonomous_cfx.core.logger import logger


@dataclass
class WatchdogTimeoutReport:
    """Diagnostic incident report authored when a task or process times out."""
    task_name: str
    timeout_seconds: float
    elapsed_seconds: float
    timed_out_at: float = field(default_factory=time.time)
    parent_pid: int = field(default_factory=os.getpid)
    terminated_child_pids: List[int] = field(default_factory=list)
    call_stack: List[str] = field(default_factory=list)
    incident_summary: str = ""

    def save(self, file_path: str) -> None:
        """Persist report as JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


class ProcessWatchdog:
    """Robust watchdog that halts execution and terminates child process trees on stall."""

    def __init__(
        self,
        task_name: str,
        timeout_seconds: float = 30.0,
        report_path: Optional[str] = None,
        exit_code: int = 124,
    ):
        self.task_name = task_name
        self.timeout = timeout_seconds
        self.report_path = report_path or os.path.join("evidence", "reports", "watchdog_timeout_report.json")
        self.exit_code = exit_code
        self.start_time: float = 0.0
        self._timer: Optional[threading.Timer] = None
        self._triggered: bool = False

    def __enter__(self) -> ProcessWatchdog:
        self.start_time = time.time()
        self._timer = threading.Timer(self.timeout, self._handle_timeout)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()
        return False

    def _handle_timeout(self) -> None:
        """Triggered upon timeout expiry. Kills child processes, saves report, and exits nonzero."""
        self._triggered = True
        elapsed = time.time() - self.start_time

        # Capture stack frames of all active threads
        stack_traces = []
        for th_id, frame in sys._current_frames().items():
            stack_traces.append(f"Thread {th_id}:\n" + "".join(traceback.format_stack(frame)))

        # Terminate child process tree recursively using psutil
        terminated_pids = []
        try:
            current_proc = psutil.Process(os.getpid())
            children = current_proc.children(recursive=True)
            for child in children:
                try:
                    terminated_pids.append(child.pid)
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.error(f"[Watchdog] Error discovering child processes: {e}")

        # Author incident report
        report = WatchdogTimeoutReport(
            task_name=self.task_name,
            timeout_seconds=self.timeout,
            elapsed_seconds=round(elapsed, 3),
            terminated_child_pids=terminated_pids,
            call_stack=stack_traces,
            incident_summary=(
                f"Watchdog timeout triggered after {elapsed:.2f}s (limit: {self.timeout}s). "
                f"Task '{self.task_name}' hung or deadlocked. Cleaned up {len(terminated_pids)} child processes."
            ),
        )

        try:
            report.save(self.report_path)
            print(f"\n[WATCHDOG TIMEOUT ERROR] Task '{self.task_name}' timed out after {self.timeout}s!", file=sys.stderr, flush=True)
            print(f"[WATCHDOG REPORT] Saved diagnostic incident report to: {self.report_path}", file=sys.stderr, flush=True)
            print(f"[WATCHDOG CLEANUP] Terminated {len(terminated_pids)} child processes: {terminated_pids}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WATCHDOG ERROR] Failed to save report: {e}", file=sys.stderr, flush=True)

        sys.stderr.flush()
        sys.stdout.flush()
        # Force hard exit with specified nonzero exit code
        os._exit(self.exit_code)
