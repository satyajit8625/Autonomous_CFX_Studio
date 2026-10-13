"""Autonomous CFX Studio - Headless Batch Pipeline Runner.
Runs the complete milestone workflow headlessly with verified Local LLM intent parsing,
external disk geometry import, OpenUSD lighting export, and distinct human approval sign-off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from autonomous_cfx.agent.evaluator import HumanApprovalStatus, QualityStatus
from autonomous_cfx.agent.orchestrator import AutonomousCFXOrchestrator
from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.core.logger import logger
from autonomous_cfx.core.project import ProjectManager
from autonomous_cfx.core.usd_export import USDExporter


def run_pipeline(
    prompt: str = "heavy leather jacket, stiff collar, low stretch, high friction with body, high damping",
    output_dir: str = "output",
    num_frames: int = 30,
    fps: float = 24.0,
    use_external_assets: bool = False,
    artist_approved: bool = False,
    llm_backend: str = "embedded_local",
):
    print("=" * 75)
    print("AUTONOMOUS CFX STUDIO - PRODUCTION PIPELINE RUNNER")
    print("=" * 75)

    project_dir = Path(output_dir).resolve()
    pm = ProjectManager(project_dir)
    am = AssetManager(pm.assets_dir)
    orchestrator = AutonomousCFXOrchestrator(pm, llm_backend=llm_backend)

    # 1. Ingest Geometry Assets
    if use_external_assets:
        print("\n[Step 1/5] Ingesting REAL EXTERNAL DISK ASSETS (Garment OBJ, Animated Sequence)...")
        ext_files = am.prepare_external_disk_assets(num_frames=num_frames, fps=fps)
        jacket, body_col, chair_col = am.load_external_package(
            garment_file=ext_files["garment_file"],
            body_sequence_folder=ext_files["animated_body_folder"],
            obstacle_file=ext_files["obstacle_file"],
            fps=fps,
        )
    else:
        print("\n[Step 1/5] Ingesting Milestone Geometry (Jacket, Animated Body, Chair)...")
        jacket, body_col, chair_col = am.load_milestone_package(
            num_frames=num_frames,
            fps=fps,
            include_chair=True,
        )

    colliders = [body_col]
    if chair_col:
        colliders.append(chair_col)

    collar_verts = list(jacket.vertex_groups.get("collar", []))

    # 2. Agent Workflow Execution
    print(f"\n[Step 2/5] Autonomous Local LLM Intent Parsing & Orchestration...")
    print(f"Artist Prompt: \"{prompt}\"")

    def progress_cb(stage, cur, total, extra):
        sys.stdout.write(f"\r >> Progress: [{stage}] {cur}/{total}%")
        sys.stdout.flush()

    human_status = HumanApprovalStatus.ARTIST_APPROVED if artist_approved else HumanApprovalStatus.PENDING_REVIEW

    result = orchestrator.execute_workflow(
        cloth_geometry=jacket,
        colliders=colliders,
        behavior_description=prompt,
        pinned_vertices=collar_verts,
        shot_name="shot010",
        fps=fps,
        start_frame=1,
        end_frame=num_frames,
        human_approval=human_status,
        export_on_approval=artist_approved,
        progress_callback=progress_cb,
    )
    print("\n")

    # Clarification Check
    if result.clarification_requested:
        print("\n[AGENT CLARIFICATION NEEDED]")
        print(f"  Question: {result.clarification_requested.get('clarification_question')}")
        print("  Suggested Options:")
        for opt in result.clarification_requested.get("suggested_options", []):
            print(f"    • {opt}")
        return result

    # 3. Autonomous Diagnostic Report
    print("\n[Step 3/5] Autonomous Quality Assessment (Independent of Human Sign-off):")
    rep = result.final_report
    print(f"  • Automated Recommendation: {rep.status.value}")
    print(f"  • Quality Score:            {rep.score} / 100")
    print(f"  • Raw Penetration Depth:    {rep.raw_max_penetration_depth_mm:.2f} mm (Tolerance: < 1.0mm)")
    print(f"  • Penetrated Vertices:      {rep.raw_penetrated_vertex_count} vertices")
    print(f"  • Max Strain Ratio:         {rep.max_stretch_ratio * 100.0:.1f} % (Tolerance: < 15.0%)")
    print(f"  • Peak Velocity:            {rep.peak_velocity:.2f} m/s")
    print(f"  • Numerical Stability:      {rep.is_numerically_stable}")

    # 4. Repair History
    print("\n[Step 4/5] Bounded Self-Healing Trace:")
    if result.repair_history:
        for r in result.repair_history:
            print(f"  • Attempt {r.attempt_index}: Pre={r.pre_score} -> Post={r.post_score} (Resolved={r.resolved})")
            print(f"    Mode: {r.failure_mode}")
            print(f"    Adjustments: {', '.join(r.adjustments_applied)}")
    else:
        print("  • No repairs needed; initial simulation passed tolerances!")

    # 5. Human Approval and Lighting Cache Export
    print("\n[Step 5/5] Human Artist Authority & Lighting Cache Delivery:")
    print(f"  • Human Approval Status:    {result.human_approval_status.value}")

    if result.human_approval_status == HumanApprovalStatus.ARTIST_APPROVED:
        print("\n  [EXPORTED LIGHTING ASSETS]")
        print(f"  • Native OpenUSD Stage (.usda): {result.exported_usd_path}")
        print(f"  • Point Cache (.vpcache):       {result.exported_cache_path}")
        print(f"  • Wavefront OBJ Sequence:       {result.exported_obj_count} frames")
        print(f"  • Interchange Manifest:         {result.exported_manifest_path}")

        # Verify USD cache
        if result.exported_usd_path:
            is_valid = USDExporter.verify_usd_cache(result.exported_usd_path)
            print(f"  • OpenUSD Stage Verification:   {'VALID (Ready for Lighting / Houdini / Unreal)' if is_valid else 'INVALID'}")
    else:
        print("  • Lighting cache export withheld awaiting artist review.")

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 75)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous CFX Studio Pipeline Runner")
    parser.add_argument("--prompt", type=str, default="heavy leather jacket, stiff collar, low stretch, high friction with body, high damping")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--external-assets", action="store_true", help="Ingest real external disk OBJ files instead of procedural demo assets")
    parser.add_argument("--artist-approved", action="store_true", help="Explicit human artist sign-off approving delivery")
    parser.add_argument("--llm-backend", type=str, default="embedded_local", choices=["embedded_local", "ollama", "openai_compatible"])
    args = parser.parse_args()

    res = run_pipeline(
        prompt=args.prompt,
        output_dir=args.output,
        num_frames=args.frames,
        fps=args.fps,
        use_external_assets=args.external_assets,
        artist_approved=args.artist_approved,
        llm_backend=args.llm_backend,
    )
    if not res.success and not res.clarification_requested:
        sys.exit(1)
