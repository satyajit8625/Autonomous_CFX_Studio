"""Tests for separate-process execution, clean cancellation, and worker crash recovery."""

import os
import time
import pytest
from autonomous_cfx.jobs.process_runner import MultiProcessJobRunner, ProcessJobStatus


def sample_worker_add(x, y, progress_callback=None):
    if progress_callback:
        progress_callback("Calculating", 50, 100)
    time.sleep(0.1)
    if progress_callback:
        progress_callback("Done", 100, 100)
    return x + y


def sample_worker_long(progress_callback=None):
    for i in range(100):
        time.sleep(0.05)
        if progress_callback:
            progress_callback("Running", i, 100)
    return "done"


def sample_worker_crash():
    time.sleep(0.1)
    # Deliberate abnormal crash simulating segfault or unhandled child abort
    os._exit(42)


def test_process_runner_successful_execution():
    runner = MultiProcessJobRunner()
    job = runner.submit_process_job("AddJob", sample_worker_add, args=(15, 25))
    assert job.status == ProcessJobStatus.RUNNING
    assert job.pid is not None

    timeout = 4.0
    start = time.time()
    while job.status == ProcessJobStatus.RUNNING and (time.time() - start) < timeout:
        time.sleep(0.05)
        job = runner.poll_job(job.job_id)

    assert job.status == ProcessJobStatus.COMPLETED
    assert job.result == 40
    assert job.progress == 100.0


def test_process_runner_clean_cancellation():
    runner = MultiProcessJobRunner()
    job = runner.submit_process_job("LongJob", sample_worker_long)
    time.sleep(0.1)

    runner.cancel_process_job(job.job_id)
    assert job.status == ProcessJobStatus.CANCELLED


def test_process_runner_worker_crash_recovery():
    runner = MultiProcessJobRunner()
    job = runner.submit_process_job("CrashJob", sample_worker_crash)

    timeout = 4.0
    start = time.time()
    while job.status == ProcessJobStatus.RUNNING and (time.time() - start) < timeout:
        time.sleep(0.05)
        job = runner.poll_job(job.job_id)

    # Worker crashed with exit code 42! Host process must detect and record CRASHED without failing!
    assert job.status == ProcessJobStatus.CRASHED
    assert job.exitcode == 42
    assert "crashed with exit code 42" in (job.error_message or "")
