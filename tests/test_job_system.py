"""Tests for JobRunner background task manager."""

import time
import pytest
from autonomous_cfx.jobs.job_runner import JobRunner, JobStatus


def test_job_runner_execution():
    runner = JobRunner()

    def sample_task(x, y):
        return x + y

    job = runner.submit_job("Add_Task", sample_task, args=(10, 20))
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED)

    # Wait for completion
    timeout = 3.0
    start = time.time()
    while job.status != JobStatus.COMPLETED and (time.time() - start) < timeout:
        time.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    assert job.result == 30
    assert job.progress == 100.0


def test_job_runner_cancellation():
    runner = JobRunner()

    def long_task():
        time.sleep(0.5)
        return "done"

    job = runner.submit_job("Long_Task", long_task)
    success = runner.cancel_job(job.job_id)
    assert success == True
