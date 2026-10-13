"""Demonstration: Full Autonomous Workflow on External Disk Assets with OpenUSD Lighting Export.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to pythonpath
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.run_headless import run_pipeline


def main():
    print(">>> DEMO: EXTERNAL DISK ASSETS WORKFLOW + OPENUSD LIGHTING CACHE")
    res = run_pipeline(
        prompt="heavy leather jacket, stiff collar, low stretch, high friction with body, high damping",
        output_dir="output_external_demo",
        num_frames=24,
        fps=24.0,
        use_external_assets=True,
        artist_approved=True,
    )
    if res.success:
        print("\n>>> DEMO VERIFIED: External disk geometry ingested, simulated, evaluated, and exported to OpenUSD stage!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
