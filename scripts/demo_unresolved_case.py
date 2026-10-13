"""Demonstration: Unresolvable Conflict where Agent Stops Honestly at Bound.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from autonomous_cfx.agent.evaluator import HumanApprovalStatus, QualityStatus, SimulationEvaluator
from autonomous_cfx.agent.repair import BoundedRepairStrategist
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.colliders import GroundPlaneCollider
from autonomous_cfx.simulation.properties import ClothMaterialProperties


def main():
    print("=" * 75)
    print("DEMO: UNRESOLVABLE GEOMETRIC CONFLICT (HONEST STOP AT BOUND)")
    print("=" * 75)

    # Pathological setup: vertex 0 is pinned deep below ground at y = -0.15m (150mm underground)
    verts = np.array([
        [0.0, -0.15, 0.0],  # Pinned inside collider!
        [0.2, 0.4, 0.0],
        [0.1, 0.6, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(name="pathological_mesh", vertices=verts, faces=faces)

    cache = SimulationCache(base_faces=faces)
    cache.add_frame(
        1,
        verts,
        metrics={
            "contact_events": 10,
            "max_margin_depth": 0.16,
            "raw_true_penetrations": 1,
            "raw_max_penetration_depth": 0.15,
            "raw_penetrated_vertex_ids": [0],
            "max_velocity": 0.0,
        },
    )

    initial_report = SimulationEvaluator.evaluate(cache, geo)
    print("\n[Initial Run Evaluation]")
    print(f"  • Status:            {initial_report.status.value}")
    print(f"  • Raw Penetration:   {initial_report.raw_max_penetration_depth_mm:.2f} mm")

    print("\n[Agent Bounded Repair Loop (Max 3 attempts)]")
    strategist = BoundedRepairStrategist(max_attempts=3)
    ground = GroundPlaneCollider(name="ground", height=0.0)
    props = ClothMaterialProperties(collision_thickness=0.01)

    repaired_cache, final_report, history = strategist.repair_and_rerun(
        current_report=initial_report,
        cloth_geometry=geo,
        properties=props,
        colliders=[ground],
        pinned_vertices=[0],  # Pinned constraint conflict
        fps=24.0,
        start_frame=1,
        end_frame=3,
    )

    print(f"\n[Bounded Repair Loop Terminated]")
    print(f"  • Attempts Executed: {len(history)} / 3")
    for r in history:
        print(f"  • Attempt {r.attempt_index}: Diagnosed Mode = {r.failure_mode}")
        print(f"    Diagnosis: {r.diagnosis[0]}")

    print(f"\n[Final Honest Assessment]")
    print(f"  • Status:            {final_report.status.value}")
    print(f"  • Human Approval:    {final_report.human_approval.value}")
    print(f"  • Action Taken:      HALTED HONESTLY at bounded limit.")
    print(f"  • Agent Issues:      {final_report.issues[-1]}")
    print(f"  • Recommendations:   {final_report.recommendations[-1]}")

    assert final_report.status == QualityStatus.UNRESOLVED_FAILED
    assert final_report.human_approval == HumanApprovalStatus.PENDING_REVIEW
    print("\n" + "=" * 75)
    print("DEMO VERIFIED: Agent stopped honestly without falsely claiming success!")
    print("=" * 75)


if __name__ == "__main__":
    main()
