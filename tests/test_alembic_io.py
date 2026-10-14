"""Tests for standalone Animated Alembic (.abc) I/O and Maya DCC compatibility."""

import os
import tempfile
import numpy as np
import pytest

from autonomous_cfx.core.alembic_io import (
    ALEMBIC_AVAILABLE,
    AlembicObjectRole,
    AlembicShotExporter,
    AlembicShotImporter,
)
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.assets.procedural import (
    generate_procedural_animated_body,
    generate_procedural_jacket,
)


@pytest.mark.skipif(not ALEMBIC_AVAILABLE, reason="Alembic C++ bindings not available")
def test_alembic_export_and_import_roundtrip():
    """Verify writing animated cloth cache to .abc and inspecting/reading back."""
    tmp_dir = tempfile.mkdtemp(prefix="cfx_abc_test_")
    abc_path = os.path.join(tmp_dir, "test_cloth_sim.abc")

    # Generate 5 frames of animated cloth
    base_mesh = generate_procedural_jacket()
    animated_frames = []
    for f in range(5):
        # Apply a progressive gravity/flutter displacement
        displaced_verts = base_mesh.vertices.copy()
        displaced_verts[:, 1] -= f * 0.02
        displaced_verts[:, 0] += np.sin(f * 0.5) * 0.01
        frame_geom = MeshGeometry(
            vertices=displaced_verts,
            faces=base_mesh.faces,
            name=f"jacket_f{f+1}"
        )
        animated_frames.append(frame_geom)

    exporter = AlembicShotExporter()
    out_file = exporter.export_shot(
        output_path=abc_path,
        simulated_garments={"/shot010/garments/hero_jacket": animated_frames},
        fps=24.0,
        start_frame=1,
        full_shot_export=False
    )
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000

    # Inspect archive
    importer = AlembicShotImporter()
    summary = importer.inspect_archive(out_file)

    assert summary.fps == 24.0
    assert summary.total_samples == 5
    assert summary.frame_range == (1, 5)
    assert len(summary.nodes) >= 1

    # Locate mesh node
    mesh_nodes = [n for n in summary.nodes if n.node_type == "PolyMesh"]
    assert len(mesh_nodes) == 1
    jacket_node = mesh_nodes[0]
    assert jacket_node.name == "hero_jacket"
    assert jacket_node.num_samples == 5
    assert jacket_node.num_vertices == len(base_mesh.vertices)
    assert jacket_node.num_faces == len(base_mesh.faces)
    assert jacket_node.is_topology_constant is True
    assert jacket_node.assigned_role == AlembicObjectRole.GARMENT

    # Load shot
    shot_data = importer.load_shot(out_file)
    assert len(shot_data.garment_meshes) == 1
    loaded_frames = list(shot_data.garment_meshes.values())[0]
    assert len(loaded_frames) == 5

    # Check frame 0 and frame 4 vertex positions match input
    np.testing.assert_allclose(loaded_frames[0].vertices, animated_frames[0].vertices, atol=1e-5)
    np.testing.assert_allclose(loaded_frames[4].vertices, animated_frames[4].vertices, atol=1e-5)


@pytest.mark.skipif(not ALEMBIC_AVAILABLE, reason="Alembic C++ bindings not available")
def test_alembic_multi_object_shot_with_colliders():
    """Verify archive with both garment and animated body collider."""
    tmp_dir = tempfile.mkdtemp(prefix="cfx_abc_shot_")
    abc_path = os.path.join(tmp_dir, "shot010_full.abc")

    jacket = generate_procedural_jacket()
    body_base, body_dict = generate_procedural_animated_body(num_frames=6)
    body_frames = [
        MeshGeometry(vertices=body_dict[f], faces=body_base.faces, name=f"body_{f}")
        for f in range(1, 7)
    ]

    exporter = AlembicShotExporter()
    out_file = exporter.export_shot(
        output_path=abc_path,
        simulated_garments={
            "/shot/jacket": [jacket] * 6,
            "/shot/body_collider": body_frames
        },
        fps=24.0,
        start_frame=1
    )

    importer = AlembicShotImporter()
    summary = importer.inspect_archive(out_file)
    assert summary.total_samples == 6

    # Verify semantic role auto-assignment
    roles = {n.name: n.assigned_role for n in summary.nodes if n.node_type == "PolyMesh"}
    assert roles.get("jacket") == AlembicObjectRole.GARMENT
    assert roles.get("body_collider") == AlembicObjectRole.BODY

    shot_data = importer.load_shot(out_file)
    assert len(shot_data.garment_meshes) == 1
    assert len(shot_data.body_colliders) == 1


def test_alembic_maya_ingest():
    """Verify that a clean receiving Autodesk Maya session successfully opens exported .abc."""
    import subprocess
    maya_exe = r"C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe"
    if not os.path.exists(maya_exe):
        pytest.skip("Maya 2023 not installed on this host")

    tmp_dir = tempfile.mkdtemp(prefix="cfx_maya_abc_")
    abc_path = os.path.join(tmp_dir, "cloth_for_maya.abc")

    jacket = generate_procedural_jacket()
    exporter = AlembicShotExporter()
    exporter.export_shot(
        output_path=abc_path,
        simulated_garments={"/shot010/garment_mesh": [jacket] * 3},
        fps=24.0,
        start_frame=1
    )

    py_script = f"""
import maya.standalone
maya.standalone.initialize()
import maya.cmds as cmds
cmds.loadPlugin('AbcImport')
cmds.file(r'{abc_path}', i=True, type='Alembic')
meshes = cmds.ls(type='mesh')
assert len(meshes) >= 1, f"Expected mesh in Maya, found {{meshes}}"
num_verts = cmds.polyEvaluate(meshes[0], vertex=True)
print(f"MAYA_INGEST_OK: mesh={{meshes[0]}}, verts={{num_verts}}")
maya.standalone.uninitialize()
"""
    cmd = [maya_exe, "-c", py_script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Maya failed: {result.stderr}"
    assert "MAYA_INGEST_OK" in result.stdout, f"Maya output did not confirm ingest: {result.stdout}"


@pytest.mark.skipif(not ALEMBIC_AVAILABLE, reason="Alembic C++ bindings not available")
def test_lazy_alembic_loading_and_streaming():
    """Verify on-demand sample loading, caching, prefetching, and memory release."""
    tmp_dir = tempfile.mkdtemp(prefix="cfx_lazy_abc_")
    abc_path = os.path.join(tmp_dir, "stream_test.abc")

    jacket = generate_procedural_jacket()
    animated_frames = []
    for f in range(10):
        m = jacket.copy()
        m.vertices[:, 1] -= f * 0.01
        animated_frames.append(m)

    exporter = AlembicShotExporter()
    exporter.export_shot(
        output_path=abc_path,
        simulated_garments={"/shot/garments/jacket": animated_frames},
        fps=24.0,
        start_frame=1,
    )

    importer = AlembicShotImporter()
    with importer.load_shot(abc_path, lazy=True) as shot_data:
        garment_seq = list(shot_data.garment_meshes.values())[0]

        # 1. Verify sequence length matches samples
        assert len(garment_seq) == 10

        # 2. Cache should be initially empty before access
        assert len(garment_seq._cache) == 0

        # 3. Access sample 0
        f0 = garment_seq[0]
        assert f0.num_vertices == jacket.num_vertices
        assert len(garment_seq._cache) == 1
        np.testing.assert_allclose(f0.vertices, animated_frames[0].vertices, atol=1e-5)

        # 4. Access sample 9
        f9 = garment_seq[9]
        assert len(garment_seq._cache) == 2
        np.testing.assert_allclose(f9.vertices, animated_frames[9].vertices, atol=1e-5)

        # 5. Prefetch samples 2, 3, 4
        garment_seq.prefetch([2, 3, 4])
        assert len(garment_seq._cache) == 5

        # 6. Clear cache
        garment_seq.clear_cache()
        assert len(garment_seq._cache) == 0

