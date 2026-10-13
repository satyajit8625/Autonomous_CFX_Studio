"""Tests for raw penetration measurement detectors, diagnosed & healed simulations, and honest stopping."""

import numpy as np
import pytest
from autonomous_cfx.agent.evaluator import (
    HumanApprovalStatus,
    QualityMetricsReport,
    QualityStatus,
    SimulationEvaluator,
)
from autonomous_cfx.agent.repair import BoundedRepairStrategist
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.colliders import GroundPlaneCollider
from autonomous_cfx.simulation.properties import ClothMaterialProperties


def test_deliberate_penetration_detector():
    """Test raw penetration measurement against deliberately defective simulation positions."""
    verts = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 1.5, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(vertices=verts, faces=faces)

    cache = SimulationCache(base_faces=faces)
    cache.add_frame(1, verts)

    # Deliberate defect: Vertex 0 is plunged 25mm inside the ground plane (y = -0.025m)
    defective_verts = np.copy(verts)
    defective_verts[0, 1] = -0.025  # 25mm penetration

    cache.add_frame(
        2,
        defective_verts,
        metrics={
            "contact_events": 1,
            "max_margin_depth": 0.035,
            "raw_true_penetrations": 1,
            "raw_max_penetration_depth": 0.025,
            "raw_penetrated_vertex_ids": [0],
            "max_velocity": 2.0,
        },
    )

    report = SimulationEvaluator.evaluate(cache, geo)

    # Verifications: Raw measurements must strictly report the 25mm penetration and vertex 0!
    assert report.status == QualityStatus.NEEDS_REPAIR
    assert report.raw_max_penetration_depth_mm == 25.0
    assert report.raw_penetrated_vertex_count == 1
    assert report.raw_penetrated_vertex_ids == [0]
    assert report.human_approval == HumanApprovalStatus.PENDING_REVIEW


def test_diagnosed_and_healed_simulation():
    """Demonstrates a failing simulation that the agent diagnoses and heals through bounded repairs."""
    verts = np.array([
        [0.0, 0.5, 0.0],
        [0.2, 0.5, 0.0],
        [0.1, 0.7, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(vertices=verts, faces=faces)

    # Initial failing simulation state with 18.5mm penetration
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
    assert initial_report.status == QualityStatus.NEEDS_REPAIR
    assert initial_report.raw_max_penetration_depth_mm == 18.5

    # Run bounded repairs
    strategist = BoundedRepairStrategist(max_attempts=3)
    ground = GroundPlaneCollider(name="floor", height=0.0)
    props = ClothMaterialProperties(collision_thickness=0.004, solver_substeps=8)

    repaired_cache, final_rep, history = strategist.repair_and_rerun(
        current_report=initial_report,
        cloth_geometry=geo,
        properties=props,
        colliders=[ground],
        pinned_vertices=[],
        fps=24.0,
        start_frame=1,
        end_frame=4,
    )

    # Verified Improvement: Repaired simulation clears ground penetrations and passes tolerances!
    assert len(history) >= 1
    assert final_rep.status == QualityStatus.PASS_RECOMMENDED
    assert final_rep.score >= 85.0
    assert final_rep.raw_max_penetration_depth_mm <= 1.0


def test_unresolved_case_honest_stop():
    """Demonstrates an unresolvable geometric conflict where the agent stops honestly."""
    # Pinned vertex is pinned at y = -0.15m (150mm underground!)
    verts = np.array([
        [0.0, -0.15, 0.0],  # Pinned inside ground collider!
        [0.2, 0.5, 0.0],
        [0.1, 0.7, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(vertices=verts, faces=faces)

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
    assert initial_report.status == QualityStatus.NEEDS_REPAIR

    strategist = BoundedRepairStrategist(max_attempts=3)
    ground = GroundPlaneCollider(name="floor", height=0.0)
    props = ClothMaterialProperties(collision_thickness=0.01)

    # Vertex 0 is pinned (infinite mass, cannot be pushed out of ground!)
    repaired_cache, final_rep, history = strategist.repair_and_rerun(
        current_report=initial_report,
        cloth_geometry=geo,
        properties=props,
        colliders=[ground],
        pinned_vertices=[0],  # Pinned constraint conflict!
        fps=24.0,
        start_frame=1,
        end_frame=3,
    )

    # Honest Stop Verification:
    # 1. Stopped at bounded limit of 3 attempts
    assert len(history) == 3
    # 2. Honest status UNRESOLVED_FAILED
    assert final_rep.status == QualityStatus.UNRESOLVED_FAILED
    # 3. Explicit root cause diagnosis
    assert any("conflict" in rec.diagnosis[0].lower() or "pinned" in rec.diagnosis[0].lower() for rec in history)
    # 4. Human approval was NOT set automatically
    assert final_rep.human_approval == HumanApprovalStatus.PENDING_REVIEW
