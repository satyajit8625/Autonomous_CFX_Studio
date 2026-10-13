"""Tests for Pixar OpenUSD (.usda / .usd) lighting-consumable cache export."""

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.usd_export import USDExporter


def test_usd_cache_export_and_verification(tmp_path):
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cache = SimulationCache(name="cloth_usd_test", fps=24.0, start_frame=1, base_faces=faces)

    f1 = np.array([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], dtype=np.float32)
    f2 = np.array([[0, 0.9, 0], [1, 0.9, 0], [1, 1.9, 0], [0, 1.9, 0]], dtype=np.float32)

    cache.add_frame(1, f1)
    cache.add_frame(2, f2)

    usd_file = tmp_path / "cloth_sim.usda"
    out_path = cache.export_usd_cache(usd_file)

    assert usd_file.exists()
    assert USDExporter.verify_usd_cache(usd_file) == True

    # Open stage via OpenUSD library and verify attributes
    stage = Usd.Stage.Open(str(usd_file))
    assert stage is not None
    assert stage.GetStartTimeCode() == 1.0
    assert stage.GetEndTimeCode() == 2.0
    assert stage.GetTimeCodesPerSecond() == 24.0

    prim = stage.GetDefaultPrim()
    assert prim.IsValid()
    mesh = UsdGeom.Mesh(prim)
    assert mesh is not None

    # Verify time-sampled points at frame 1 and 2
    pts_f1 = mesh.GetPointsAttr().Get(Usd.TimeCode(1.0))
    pts_f2 = mesh.GetPointsAttr().Get(Usd.TimeCode(2.0))

    assert len(pts_f1) == 4
    assert len(pts_f2) == 4
    assert np.allclose(pts_f1[0], [0, 1, 0])
    assert np.allclose(pts_f2[0], [0, 0.9, 0])
