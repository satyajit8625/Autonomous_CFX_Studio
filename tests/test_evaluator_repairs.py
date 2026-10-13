"""Tests for SimulationEvaluator and BoundedRepairStrategist."""

import numpy as np
import pytest
from autonomous_cfx.agent.evaluator import QualityStatus, SimulationEvaluator
from autonomous_cfx.agent.repair import BoundedRepairStrategist
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.colliders import GroundPlaneCollider
from autonomous_cfx.simulation.properties import ClothMaterialProperties


def test_evaluator_on_clean_simulation():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(vertices=verts, faces=faces)

    cache = SimulationCache(base_faces=faces)
    cache.add_frame(1, verts, metrics={"contact_events": 0, "max_depth": 0.0})
    cache.add_frame(2, verts + 0.001, metrics={"contact_events": 0, "max_depth": 0.0})

    report = SimulationEvaluator.evaluate(cache, geo)
    assert report.status == QualityStatus.APPROVED
    assert report.score >= 90.0
    assert report.is_numerically_stable == True


def test_evaluator_catches_excessive_penetration():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(vertices=verts, faces=faces)

    cache = SimulationCache(base_faces=faces)
    # Penetration depth 0.025m (above 0.012 limit)
    cache.add_frame(1, verts, metrics={"contact_events": 10, "max_depth": 0.025, "raw_max_penetration_depth": 0.025, "raw_true_penetrations": 10})

    report = SimulationEvaluator.evaluate(cache, geo)
    assert report.status == QualityStatus.NEEDS_REPAIR
    assert report.raw_max_penetration_depth_mm == 25.0
    assert len(report.issues) > 0


def test_bounded_repair_strategist():
    strategist = BoundedRepairStrategist(max_attempts=2)
    assert strategist.max_attempts == 2

    # Verify repair executes and terminates within bounded iterations
    verts = np.array([[0, 1, 0], [0.2, 1, 0], [0, 1.2, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    geo = MeshGeometry(vertices=verts, faces=faces)

    # Initial report with excessive penetration (30mm)
    cache = SimulationCache(base_faces=faces)
    cache.add_frame(1, verts, metrics={"contact_events": 5, "max_depth": 0.030, "raw_max_penetration_depth": 0.030, "raw_true_penetrations": 5})
    initial_report = SimulationEvaluator.evaluate(cache, geo)
    assert initial_report.status == QualityStatus.NEEDS_REPAIR

    ground = GroundPlaneCollider(name="floor", height=0.0)
    props = ClothMaterialProperties(collision_thickness=0.005)

    repaired_cache, final_rep, history = strategist.repair_and_rerun(
        current_report=initial_report,
        cloth_geometry=geo,
        properties=props,
        colliders=[ground],
        pinned_vertices=[],
        fps=24.0,
        start_frame=1,
        end_frame=3,
    )

    assert len(history) <= 2  # Bounded within 2 attempts
    assert final_rep is not None
    assert repaired_cache is not None
