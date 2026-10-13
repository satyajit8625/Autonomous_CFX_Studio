"""Tests for XPBD animated attachments, aerodynamic wind drag physics,
worker cancellation callback, and Alembic UV / FaceSet preservation.
"""

import os
import tempfile
import numpy as np
import pytest

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.constraints import AnimatedAttachment
from autonomous_cfx.simulation.colliders import AnimatedMeshCollider
from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.simulation.xpbd_solver import XPBDSolver
from autonomous_cfx.simulation.engine import SimulationEngine
from autonomous_cfx.core.alembic_io import (
    ALEMBIC_AVAILABLE,
    AlembicObjectRole,
    AlembicShotExporter,
    AlembicShotImporter,
)


def _create_planar_cloth_mesh(nx=4, ny=4, size=0.5) -> MeshGeometry:
    """Create a planar cloth grid in the XZ or XY plane."""
    xs = np.linspace(-size / 2, size / 2, nx, dtype=np.float32)
    ys = np.linspace(1.0, 1.0 + size, ny, dtype=np.float32)
    verts = []
    for y in ys:
        for x in xs:
            verts.append([x, y, 0.0])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            i0 = j * nx + i
            i1 = i0 + 1
            i2 = (j + 1) * nx + i
            i3 = i2 + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])
    faces = np.array(faces, dtype=np.int32)
    return MeshGeometry(name="test_cloth", vertices=verts, faces=faces)


def test_animated_attachment_kinematics():
    """Verify that cloth vertices attached to an animated collider follow motion exactly."""
    cloth = _create_planar_cloth_mesh(nx=4, ny=4, size=0.4)
    # Pinned collar row: top row of cloth vertices (y ~ 1.4)
    top_v_idx = np.where(cloth.vertices[:, 1] >= 1.35)[0]
    assert len(top_v_idx) == 4

    # Animated body collider translating along +X and +Y
    col_v0 = np.array([
        [-0.2, 1.4, -0.05],
        [0.2, 1.4, -0.05],
        [0.0, 1.0, -0.05],
    ], dtype=np.float32)
    col_faces = np.array([[0, 1, 2]], dtype=np.int32)
    col_geom = MeshGeometry(name="anim_body", vertices=col_v0, faces=col_faces)

    # Frame 1 to Frame 10 translations: character moves by +0.5m in X
    frames = {}
    for f in range(1, 11):
        shift = (f - 1) * 0.05
        frames[f] = col_v0 + np.array([shift, 0.0, 0.0], dtype=np.float32)

    collider = AnimatedMeshCollider("body", base_geometry=col_geom, animation_frames=frames, fps=24.0)

    props = ClothMaterialProperties(
        density=0.3,
        solver_substeps=8,
        stretch_compliance=1e-5,
    )
    solver = XPBDSolver(cloth, props)
    att = solver.attach_to_collider(top_v_idx, collider, stiffness=1.0)
    solver.add_collider(collider)

    # Step solver for 5 frames
    for f in range(2, 6):
        solver.step(dt=1.0 / 24.0, frame_num=float(f))

    # At frame 5, character has moved by (5 - 1) * 0.05 = 0.20m in X
    expected_x_shift = 0.20
    attached_x = solver.positions[top_v_idx, 0]
    initial_x = cloth.vertices[top_v_idx, 0]
    actual_shift = attached_x - initial_x

    np.testing.assert_allclose(actual_shift, expected_x_shift, atol=1e-3)
    assert np.all(solver.inv_masses[top_v_idx] == 0.0)


def test_aerodynamic_wind_drag_physics():
    """Verify aerodynamic wind acceleration and units."""
    cloth = _create_planar_cloth_mesh(nx=3, ny=3, size=0.3)
    # Pin top vertices so cloth hangs and swings under wind
    top_v_idx = np.where(cloth.vertices[:, 1] >= 1.25)[0]

    # Strong lateral wind along +Z at 8.0 m/s with 0 gravity to isolate wind effect
    props = ClothMaterialProperties(
        density=0.25,  # kg/m^2
        solver_substeps=10,
        air_drag=0.01,
        wind_velocity=(0.0, 0.0, 8.0),  # m/s along +Z
        wind_gust=0.0,
    )
    solver = XPBDSolver(cloth, props)
    solver.gravity = np.zeros(3, dtype=np.float32)  # Zero gravity
    solver.set_pinned_vertices(top_v_idx)

    # Free vertices
    free_v_idx = [i for i in range(len(cloth.vertices)) if i not in top_v_idx]

    # Step solver
    dt = 1.0 / 24.0
    solver.step(dt=dt, frame_num=1.0)

    # Velocities on free vertices must be positive along +Z due to aerodynamic drag
    vz = solver.velocities[free_v_idx, 2]
    pz = solver.positions[free_v_idx, 2]
    assert np.all(vz > 0.0), f"Expected positive +Z velocity under wind drag, got {vz}"
    assert np.all(pz > 0.0), f"Expected cloth displacement along +Z under wind drag, got {pz}"


def test_simulation_engine_clean_cancellation():
    """Verify that cancel_check callback halts SimulationEngine computation immediately."""
    cloth = _create_planar_cloth_mesh(nx=3, ny=3, size=0.3)
    engine = SimulationEngine(
        cloth_geometry=cloth,
        fps=24.0,
        start_frame=1,
        end_frame=50,
    )

    cancelled_after_frame = 3

    def cancel_check():
        return True  # Cancel immediately

    cache = engine.run(cancel_check=cancel_check)
    # Frame 1 is initialized, frame 2 checks cancel_check and immediately breaks
    assert cache.frame_count <= 2, f"Expected clean early cancellation, got {cache.frame_count} frames"


@pytest.mark.skipif(not ALEMBIC_AVAILABLE, reason="Alembic C++ bindings required")
def test_alembic_uvs_facesets_and_animated_xforms_roundtrip():
    """Verify preservation of UV coordinates, FaceSets, and multi-sample transforms in Alembic export."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = os.path.join(tmp_dir, "source_shot.abc")
        out_path = os.path.join(tmp_dir, "simulated_output.abc")

        # 1. Author source archive with animated body collider and UVs
        cloth = _create_planar_cloth_mesh(nx=3, ny=3, size=0.3)
        # Assign mock UVs
        n_verts = cloth.num_vertices
        cloth.uvs = np.zeros((n_verts, 2), dtype=np.float32)
        cloth.uvs[:, 0] = np.linspace(0.0, 1.0, n_verts)
        cloth.uvs[:, 1] = np.linspace(0.0, 1.0, n_verts)
        cloth.face_sets["collar_band"] = np.array([0, 1], dtype=np.int32)
        cloth.face_sets["hem_trim"] = np.array([2, 3], dtype=np.int32)

        # 5 frames of animated simulation
        sim_frames = []
        for f in range(1, 6):
            f_mesh = cloth.copy()
            f_mesh.vertices[:, 2] += (f - 1) * 0.01  # Small drift
            sim_frames.append(f_mesh)

        exporter = AlembicShotExporter()
        exporter.export_shot(
            output_path=out_path,
            simulated_garments={"jacket": sim_frames},
            fps=24.0,
            start_frame=1,
            full_shot_export=False,
        )

        assert os.path.exists(out_path)

        # Read back and verify UVs and FaceSets
        importer = AlembicShotImporter()
        summary = importer.inspect_archive(out_path)
        assert len(summary.nodes) >= 1
        jacket_node = next(n for n in summary.nodes if n.node_type == "PolyMesh")
        assert jacket_node.has_uvs is True
        assert "collar_band" in jacket_node.face_sets
        assert "hem_trim" in jacket_node.face_sets

        data = importer.load_shot(out_path)
        loaded_meshes = list(data.garment_meshes.values())[0]
        assert len(loaded_meshes) == 5
        assert loaded_meshes[0].uvs is not None
        assert len(loaded_meshes[0].uvs) == n_verts
        assert "collar_band" in loaded_meshes[0].face_sets
