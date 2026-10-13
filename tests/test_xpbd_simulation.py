"""Tests for XPBD solver, constraints, ground collision, and animated colliders."""

import numpy as np
import pytest
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.colliders import GroundPlaneCollider, AnimatedMeshCollider
from autonomous_cfx.simulation.engine import SimulationEngine
from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.simulation.xpbd_solver import XPBDSolver


def test_distance_constraint_preserves_length():
    # 2 vertices connected by an edge along X, pulled apart
    verts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 0]], dtype=np.int32)
    mesh = MeshGeometry(vertices=verts, faces=faces)

    props = ClothMaterialProperties(stretch_compliance=1e-7, solver_substeps=10, solver_iterations=5)
    solver = XPBDSolver(mesh, props)

    # Force displacement apart to 3.0m
    solver.positions[1, 0] = 3.0

    # Step solver without gravity to check edge recovery
    solver.gravity = np.zeros(3, dtype=np.float32)
    solver.step(dt=1.0 / 24.0, frame_num=1.0)

    # Rest length was 2.0. The solver should restore distance towards 2.0
    dist = np.linalg.norm(solver.positions[0] - solver.positions[1])
    assert dist < 2.5  # Significant correction towards rest length


def test_ground_plane_collision():
    # Single vertex falling under gravity towards ground at y = 0
    verts = np.array([[0.0, 0.5, 0.0], [0.1, 0.5, 0.0], [0.0, 0.6, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshGeometry(vertices=verts, faces=faces)

    ground = GroundPlaneCollider(name="floor", height=0.0)
    props = ClothMaterialProperties(collision_thickness=0.01, solver_substeps=10)
    solver = XPBDSolver(mesh, props)
    solver.add_collider(ground)

    # Run 10 steps of gravity fall
    for f in range(1, 15):
        solver.step(dt=1.0 / 24.0, frame_num=float(f))

    # All vertices must remain above ground (y >= 0.01)
    assert np.all(solver.positions[:, 1] >= 0.009)


def test_simulation_engine_run():
    verts = np.array([[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.0, 1.5, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshGeometry(vertices=verts, faces=faces)

    engine = SimulationEngine(
        cloth_geometry=mesh,
        fps=24.0,
        start_frame=1,
        end_frame=5,
    )
    cache = engine.run()
    assert cache.frame_count == 5
    assert cache.frame_numbers == [1, 2, 3, 4, 5]
