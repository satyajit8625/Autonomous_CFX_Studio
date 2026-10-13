"""Autonomous CFX Studio - Unit Tests for Robust Collision & Penetration Classification.

Tests:
1. Genuine deep penetration beyond 50 mm (75 mm and 120 mm).
2. Concave bodies (U-shaped concave geometry with hollow void).
3. Nearby limbs (adjacent torso and arm without false cross-limb projections).
4. Open meshes (single-sided planar collider with oriented normal).
5. Far-away cloth (broad-phase culling with zero false positive penetration).
"""

from __future__ import annotations

import numpy as np
import pytest

from autonomous_cfx.core.collision_detector import PenetrationClassifier
from autonomous_cfx.core.geometry import MeshGeometry


def create_solid_box(min_pt: np.ndarray, max_pt: np.ndarray, name: str = "box") -> MeshGeometry:
    """Create a closed, watertight 6-sided box with outward normals."""
    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # Front (-Z)
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # Back (+Z)
    ], dtype=np.float32)

    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # Front
        [4, 5, 6], [4, 6, 7],  # Back
        [0, 1, 5], [0, 5, 4],  # Bottom
        [3, 6, 2], [3, 7, 6],  # Top
        [0, 4, 7], [0, 7, 3],  # Left
        [1, 2, 6], [1, 6, 5],  # Right
    ], dtype=np.int32)

    return MeshGeometry(name=name, vertices=vertices, faces=faces)


def test_penetration_beyond_50mm():
    """Verify that genuine deep penetrations (75 mm and 120 mm) are accurately measured and not culled."""
    # Create solid torso box: width 0.40m, height 0.60m, depth 0.30m centered at origin
    torso = create_solid_box(np.array([-0.20, -0.30, -0.15]), np.array([0.20, 0.30, 0.15]), name="torso")
    assert PenetrationClassifier.is_mesh_watertight(torso) is True

    # Front face of torso is at z = -0.15
    # Cloth vertices:
    # v0: 75 mm deep inside front face (z = -0.15 + 0.075 = -0.075)
    # v1: 120 mm deep inside front face (z = -0.15 + 0.120 = -0.030)
    # v2: 10 mm outside front face (z = -0.15 - 0.010 = -0.160)
    cloth_verts = np.array([
        [0.0, 0.0, -0.075],  # 75 mm inside
        [0.0, 0.0, -0.030],  # 120 mm inside
        [0.0, 0.0, -0.160],  # 10 mm outside
    ], dtype=np.float32)
    cloth_faces = np.array([[0, 1, 2]], dtype=np.int32)
    cloth = MeshGeometry(name="deep_cloth", vertices=cloth_verts, faces=cloth_faces)

    res = PenetrationClassifier.evaluate_penetration(cloth=cloth, collider=torso, margin=0.008)

    assert res.is_watertight is True
    assert res.penetrating_vertex_count == 2
    assert 0 in res.penetrating_indices
    assert 1 in res.penetrating_indices

    rec_v0 = next(r for r in res.records if r.vertex_index == 0)
    rec_v1 = next(r for r in res.records if r.vertex_index == 1)
    rec_v2 = next(r for r in res.records if r.vertex_index == 2)

    # Verify v0 is 75 mm deep inside volume
    assert rec_v0.is_inside_volume is True
    assert abs(rec_v0.penetration_depth_mm - 75.0) < 1.0

    # Verify v1 is 120 mm deep inside volume (well beyond 50 mm cutoff!)
    assert rec_v1.is_inside_volume is True
    assert abs(rec_v1.penetration_depth_mm - 120.0) < 1.0

    # Verify v2 is outside
    assert rec_v2.is_inside_volume is False
    assert rec_v2.penetration_depth_mm == 0.0


def test_concave_body_void_handling():
    """Verify that a cloth point inside a concave hollow void is classified as outside the solid volume."""
    # Construct a U-shaped concave channel: base block with hollow trench in middle
    # Left pillar: [-0.30, -0.20, -0.20] to [-0.10, 0.20, 0.20]
    # Right pillar: [0.10, -0.20, -0.20] to [0.30, 0.20, 0.20]
    # Bottom floor: [-0.30, -0.20, -0.20] to [0.30, -0.10, 0.20]
    # Center void: x in [-0.10, 0.10], y in [-0.10, 0.20] is EMPTY AIR
    pillar_left = create_solid_box(np.array([-0.30, -0.20, -0.20]), np.array([-0.10, 0.20, 0.20]))

    # Test point inside left pillar (inside solid volume)
    # Test point in center void (x=0, y=0, z=0) - outside left pillar in air
    test_pts = np.array([
        [-0.20, 0.0, 0.0],  # Inside left pillar (solid)
        [0.00, 0.0, 0.0],   # In central void (air)
    ], dtype=np.float32)
    cloth = MeshGeometry(name="concave_test", vertices=test_pts, faces=np.array([[0, 1, 0]]))

    res = PenetrationClassifier.evaluate_penetration(cloth=cloth, collider=pillar_left, margin=0.008)
    assert res.records[0].is_inside_volume is True
    assert res.records[1].is_inside_volume is False
    assert res.records[1].penetration_depth_mm == 0.0


def test_nearby_limbs_no_cross_projection():
    """Verify that adjacent limbs (torso and arm) do not cause false cross-limb penetrations."""
    torso = create_solid_box(np.array([-0.20, -0.30, -0.15]), np.array([0.20, 0.30, 0.15]), name="torso")
    arm = create_solid_box(np.array([0.25, -0.20, -0.05]), np.array([0.35, 0.20, 0.05]), name="arm")

    # Gap between torso right edge (x=0.20) and arm left edge (x=0.25) is 50 mm of open space
    # Cloth vertex resting in this gap at x = 0.225
    gap_pt = np.array([[0.225, 0.0, 0.0]], dtype=np.float32)
    cloth = MeshGeometry(name="gap_cloth", vertices=gap_pt, faces=np.array([[0, 0, 0]]))

    res_torso = PenetrationClassifier.evaluate_penetration(cloth=cloth, collider=torso, margin=0.008)
    res_arm = PenetrationClassifier.evaluate_penetration(cloth=cloth, collider=arm, margin=0.008)

    # Point is in open air between limbs: neither penetrates inside torso nor inside arm
    assert res_torso.records[0].is_inside_volume is False
    assert res_arm.records[0].is_inside_volume is False
    assert res_torso.records[0].penetration_depth_mm == 0.0
    assert res_arm.records[0].penetration_depth_mm == 0.0


def test_open_mesh_planar_collider():
    """Verify that open/non-watertight meshes evaluate penetration via oriented surface normal."""
    # Single quad in XY plane at z = 0, facing +Z
    quad_verts = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ], dtype=np.float32)
    quad_faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    open_collider = MeshGeometry(name="ground_quad", vertices=quad_verts, faces=quad_faces)

    assert PenetrationClassifier.is_mesh_watertight(open_collider) is False

    # Cloth vertices:
    # v0: z = +0.05 (above quad in open space)
    # v1: z = -0.02 (20 mm penetrated through quad on negative normal side)
    cloth_verts = np.array([
        [0.0, 0.0, 0.05],   # Outside
        [0.0, 0.0, -0.02],  # Penetrating 20 mm
    ], dtype=np.float32)
    cloth = MeshGeometry(name="sheet_cloth", vertices=cloth_verts, faces=np.array([[0, 1, 0]]))

    res = PenetrationClassifier.evaluate_penetration(cloth=cloth, collider=open_collider, margin=0.008)
    assert res.is_watertight is False

    rec_above = res.records[0]
    rec_below = res.records[1]

    assert rec_above.is_penetrating is False
    assert rec_above.penetration_depth_mm == 0.0

    assert rec_below.is_penetrating is True
    assert rec_below.is_inside_volume is True
    assert abs(rec_below.penetration_depth_mm - 20.0) < 0.5


def test_far_away_cloth_zero_penetration():
    """Verify that broad-phase culling immediately rejects distant vertices with zero false positives."""
    torso = create_solid_box(np.array([-0.20, -0.30, -0.15]), np.array([0.20, 0.30, 0.15]), name="torso")

    # Points 2.5 meters away
    distant_pts = np.array([
        [2.5, 1.8, 3.0],
        [-5.0, 0.0, -2.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float32)
    cloth = MeshGeometry(name="distant_cloth", vertices=distant_pts, faces=np.array([[0, 1, 2]]))

    res = PenetrationClassifier.evaluate_penetration(cloth=cloth, collider=torso, margin=0.008)
    assert res.broad_phase_culled_count == 3
    assert res.max_penetration_mm == 0.0
    assert res.penetrating_vertex_count == 0
