"""Tests for SimulationCache, .vpcache binary serialization, OBJ sequence, and manifests."""

import numpy as np
import pytest
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry


def test_simulation_cache_serialization(tmp_path):
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    cache = SimulationCache(name="test_sim", fps=24.0, start_frame=1, base_faces=faces)

    f1_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    f2_pos = np.array([[0, 0.1, 0], [1, 0.1, 0], [0, 1.1, 0]], dtype=np.float32)

    cache.add_frame(1, f1_pos, metrics={"depth": 0.0})
    cache.add_frame(2, f2_pos, metrics={"depth": 0.002})

    assert cache.frame_count == 2
    assert cache.frame_numbers == [1, 2]

    # Save point cache
    cache_file = tmp_path / "test_sim.vpcache"
    cache.save_point_cache(cache_file)
    assert cache_file.exists()

    # Load point cache
    loaded = SimulationCache.load_point_cache(cache_file)
    assert loaded.name == "test_sim"
    assert loaded.fps == 24.0
    assert loaded.frame_count == 2
    assert np.allclose(loaded.frames[1], f1_pos)
    assert np.allclose(loaded.frames[2], f2_pos)
    assert np.allclose(loaded.base_faces, faces)


def test_obj_sequence_and_manifest_export(tmp_path):
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    cache = SimulationCache(name="cloth_take", fps=24.0, start_frame=1, base_faces=faces)
    f1_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cache.add_frame(1, f1_pos)

    # OBJ export
    obj_dir = tmp_path / "obj_out"
    exported_files = cache.export_obj_sequence(obj_dir, prefix="cloth")
    assert len(exported_files) == 1
    assert (obj_dir / "cloth_0001.obj").exists()

    # Manifest export
    manifest_file = cache.export_interchange_manifest(tmp_path)
    assert (tmp_path / "cloth_take_manifest.json").exists()
