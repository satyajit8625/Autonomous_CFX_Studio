"""Demonstration: Failed Simulation Diagnosed and Healed through Bounded Repairs.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from autonomous_cfx.agent.evaluator import SimulationEvaluator, QualityStatus
from autonomous_cfx.agent.repair import BoundedRepairStrategist
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.colliders import GroundPlaneCollider
from autonomous_cfx.simulation.properties import ClothMaterialProperties


def main():
    print("=" * 75)
    print("DEMO: FAILED SIMULATION DIAGNOSED AND HEALED BY AGENT")
    print("=" * 75)

    verts = np.array([
        [0.0, 0.4, 0.0],
        [0.2, 0.4, 0.0],
        [0.1, 0.6, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(name="test_patch", vertices=verts, faces=faces)

    # 1. Defective simulation state with 18.5mm ground penetration
    cache = SimulationCache(base_faces=faces)
    cache.add_frame(
        1,
        verts,
        metrics={
            "contact_events": 5,
            "max_margin_depth": 0.028,
            "raw_true_penetrations": 2,
            "raw_max_penetration_depth": 0.0185,
            "raw_penetrated_vertex_ids": [0, 1],
            "max_velocity": 4.0,
        },
    )

    initial_report = SimulationEvaluator.evaluate(cache, geo)
    print("\n[Initial Run Evaluation]")
    print(f"  • Status:            {initial_report.status.value}")
    print(f"  • Score:             {initial_report.score} / 100")
    print(f"  • Raw Penetration:   {initial_report.raw_max_penetration_depth_mm:.2f} mm")
    print(f"  • Issues Detected:   {initial_report.issues}")

    assert initial_report.status == QualityStatus.NEEDS_REPAIR

    # 2. Agent Bounded Self-Healing
    print("\n[Agent Bounded Repair Loop]")
    strategist = BoundedRepairStrategist(max_attempts=3)
    ground = GroundPlaneCollider(name="ground", height=0.0)
    props = ClothMaterialProperties(collision_thickness=0.004, solver_substeps=8)

    repaired_cache, final_report, history = strategist.repair_and_rerun(
        current_report=initial_report,
        cloth_geometry=geo,
        properties=props,
        colliders=[ground],
        pinned_vertices=[],
        fps=24.0,
        start_frame=1,
        end_frame=4,
    )

    print(f"\n[Repaired Run Evaluation] Completed after {len(history)} repair attempt(s):")
    for r in history:
        print(f"  • Attempt {r.attempt_index}: Pre={r.pre_score} -> Post={r.post_score} (Resolved: {r.resolved})")
        print(f"    Diagnosed Mode: {r.failure_mode}")
        print(f"    Adjustments:    {', '.join(r.adjustments_applied)}")

    print(f"\nFinal Assessment:")
    print(f"  • Status:            {final_report.status.value}")
    print(f"  • Final Score:       {final_report.score} / 100")
    print(f"  • Raw Penetration:   {final_report.raw_max_penetration_depth_mm:.2f} mm")

    assert final_report.status == QualityStatus.PASS_RECOMMENDED
    print("\n" + "=" * 75)
    print("DEMO VERIFIED: Defective simulation successfully diagnosed and repaired!")
    print("=" * 75)


if __name__ == "__main__":
    main()
