"""Autonomous CFX Studio - Verification of Watchdog Hang Prevention & Process Cleanup.
Demonstrates:
1. Intentional subprocess / thread stall.
2. Watchdog timeout enforcement.
3. Clean recursive termination of child processes.
4. Nonzero process exit code (124).
5. Generation of structured JSON diagnostic failure report.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_hang_target():
    """Helper invoked in subprocess that intentionally hangs with child workers."""
    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from autonomous_cfx.core.watchdog import ProcessWatchdog

    report_path = str(ROOT_DIR / "evidence" / "reports" / "watchdog_timeout_report.json")

    # Spawn an intentional background child worker process that sleeps forever
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(9999)"])
    print(f"[Child Worker Spawned] PID={child.pid}", flush=True)

    # Wrap the hanging operation in ProcessWatchdog with a 1.5 second timeout
    with ProcessWatchdog(task_name="intentional_hang_simulation", timeout_seconds=1.5, report_path=report_path, exit_code=124):
        print("[Watchdog Active] Simulating uncooperative hang (sleeping 9999s)...", flush=True)
        time.sleep(9999)


def verify_watchdog():
    print("=" * 80)
    print("VERIFYING WATCHDOG TIMEOUT ENFORCEMENT & CHILD PROCESS CLEANUP")
    print("=" * 80)

    # Run the hang target in a subprocess and measure time to terminate
    start_t = time.time()
    proc = subprocess.Popen(
        [sys.executable, __file__, "--target"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(timeout=10.0)
    elapsed = time.time() - start_t
    exit_code = proc.returncode

    print(f"Subprocess terminated in {elapsed:.2f}s with exit code {exit_code}")
    print("Subprocess stdout:\n" + stdout.strip())
    print("Subprocess stderr:\n" + stderr.strip())

    # Assertions
    assert exit_code == 124, f"Expected exit code 124, got {exit_code}"
    assert elapsed < 5.0, f"Expected timeout under 5.0s, took {elapsed:.2f}s"

    # Verify diagnostic report
    report_path = Path("evidence/reports/watchdog_timeout_report.json")
    assert report_path.exists(), f"Expected report at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    print("\nVerified Incident Report JSON:")
    print(json.dumps(report_data, indent=2))

    assert report_data["task_name"] == "intentional_hang_simulation"
    assert report_data["timeout_seconds"] == 1.5
    assert len(report_data["terminated_child_pids"]) > 0, "Expected terminated child processes"

    print("\n[SUCCESS] Watchdog successfully stopped hanging run, killed child processes, and saved diagnostic report.")


if __name__ == "__main__":
    if "--target" in sys.argv:
        run_hang_target()
    else:
        verify_watchdog()
