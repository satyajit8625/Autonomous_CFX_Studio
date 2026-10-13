"""Autonomous CFX Studio - Headless Batch Pipeline Runner.
Runs the complete milestone workflow headlessly:
Import -> Describe -> Simulate -> Evaluate -> Bounded Repair -> Auto-Approve -> Export Caches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from autonomous_cfx.agent.orchestrator import AutonomousCFXOrchestrator
from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.core.logger import logger
from autonomous_cfx.core.project import ProjectManager


def run_pipeline(
    prompt: str = "heavy leather jacket, stiff collar, low stretch, high friction with body, high damping",
    output_dir: str = "output",
    num_frames: int = 40,
    fps: float = 24.0,
    include_chair: bool = True,
):
    print("=" * 70)
    print("AUTONOMOUS CFX STUDIO - HEADLESS WORKFLOW PIPELINE")
    print("=" * 70)

    project_dir = Path(output_dir)
    pm = ProjectManager(project_dir)
    am = AssetManager(pm.assets_dir)
    orchestrator = AutonomousCFXOrchestrator(pm)

    # 1. Load milestone package (Jacket, Animated Body, Chair)
    print("\n[Step 1/5] Loading Milestone Assets (Jacket, Animated Body, Chair)...")
    jacket, body_col, chair_col = am.load_milestone_package(
        num_frames=num_frames,
        fps=fps,
        include_chair=include_chair,
    )
    colliders = [body_col]
    if chair_col:
        colliders.append(chair_col)

    collar_verts = list(jacket.vertex_groups.get("collar", []))

    # 2. Run Orchestrator Workflow
    print(f"\n[Step 2/5] Autonomous Agent Orchestration...")
    print(f"Prompt: \"{prompt}\"")

    def progress_cb(stage, cur, total, extra):
        sys.stdout.write(f"\r >> Progress: [{stage}] {cur}/{total}%")
        sys.stdout.flush()

    result = orchestrator.execute_workflow(
        cloth_geometry=jacket,
        colliders=colliders,
        behavior_description=prompt,
        pinned_vertices=collar_verts,
        shot_name="shot010",
        fps=fps,
        start_frame=1,
        end_frame=num_frames,
        auto_approve_if_passed=True,
        export_on_approval=True,
        progress_callback=progress_cb,
    )
    print("\n")

    # 3. Report Results
    print("\n[Step 3/5] Autonomous Evaluation & Diagnosis:")
    rep = result.final_report
    print(f"  • Status:               {rep.status.value}")
    print(f"  • Quality Score:        {rep.score} / 100")
    print(f"  • Max Penetration:      {rep.max_penetration_depth * 1000.0:.2f} mm")
    print(f"  • Max Stretch Ratio:    {rep.max_stretch_ratio * 100.0:.1f} %")
    print(f"  • Peak Velocity:        {rep.peak_velocity:.2f} m/s")
    print(f"  • Numerical Stability:  {rep.is_numerically_stable}")

    # 4. Repair History
    print("\n[Step 4/5] Bounded Repair Log:")
    if result.repair_history:
        for r in result.repair_history:
            print(f"  • Attempt {r.attempt_index}: Pre={r.pre_score} -> Post={r.post_score} (Resolved={r.resolved})")
            print(f"    Adjustments: {', '.join(r.adjustments_applied)}")
    else:
        print("  • No repairs needed; initial simulation passed tolerances!")

    # 5. Export Output
    print("\n[Step 5/5] Export Artifacts:")
    if result.artist_approved:
        print(f"  • Point Cache (.vpcache): {result.exported_cache_path}")
        print(f"  • Wavefront OBJ Sequence: {result.exported_obj_count} frames")
        print(f"  • Interchange Manifest:   {result.exported_manifest_path}")
    else:
        print("  • Export skipped (Quality threshold not met).")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous CFX Studio Headless Runner")
    parser.add_argument("--prompt", type=str, default="heavy leather jacket, stiff collar, low stretch, high friction with body, high damping")
    parser.add_argument("--output", type=str, default="test_output")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()

    res = run_pipeline(
        prompt=args.prompt,
        output_dir=args.output,
        num_frames=args.frames,
        fps=args.fps,
    )
    if not res.success:
        sys.exit(1)
