"""Autonomous CFX Studio - Lighting-Consumable OpenUSD Cache Exporter.
Exports production-grade, animated Pixar OpenUSD (.usda / .usdc / .usd) geometry caches
fully compliant with USD lighting workflows (Houdini Solaris, Maya USD, Katana, Unreal Engine 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.logger import logger


class USDExporter:
    """Exports simulated CFX caches to native Pixar OpenUSD stages."""

    @classmethod
    def export_cache(
        cls,
        cache: SimulationCache,
        output_path: str | Path,
        prim_path: str = "/World/ClothMesh",
    ) -> str:
        """Create and write a lighting-compliant OpenUSD stage with time-sampled geometry."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames = cache.frame_numbers
        if not frames:
            raise ValueError("Simulation cache has no frames to export.")

        # If file exists, remove it so CreateNew succeeds
        if output_path.exists():
            output_path.unlink()

        stage = Usd.Stage.CreateNew(str(output_path))
        if not stage:
            raise RuntimeError(f"Failed to create USD stage at: {output_path}")

        # Stage Metadata
        start_frame = float(min(frames))
        end_frame = float(max(frames))
        stage.SetStartTimeCode(start_frame)
        stage.SetEndTimeCode(end_frame)
        stage.SetTimeCodesPerSecond(cache.fps)
        stage.SetFramesPerSecond(cache.fps)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Define Mesh Prim
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

        # Topology (constant across time)
        num_faces = len(cache.base_faces)
        face_counts = np.full(num_faces, 3, dtype=np.int32)
        flat_indices = cache.base_faces.flatten().astype(np.int32)

        vt_face_counts = Vt.IntArray.FromNumpy(face_counts)
        vt_face_indices = Vt.IntArray.FromNumpy(flat_indices)

        usd_mesh.GetFaceVertexCountsAttr().Set(vt_face_counts)
        usd_mesh.GetFaceVertexIndicesAttr().Set(vt_face_indices)

        # Subdivision scheme: None for polygonal mesh
        usd_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        # Time-Sampled Geometry Attributes
        points_attr = usd_mesh.GetPointsAttr()
        extent_attr = usd_mesh.GetExtentAttr()

        for f in frames:
            pos = cache.frames[f].astype(np.float32)
            tc = Usd.TimeCode(float(f))

            # Convert numpy (N, 3) to Vt.Vec3fArray
            vt_points = Vt.Vec3fArray.FromNumpy(pos)
            points_attr.Set(vt_points, tc)

            # Extent (Bounding Box)
            min_pt = np.min(pos, axis=0)
            max_pt = np.max(pos, axis=0)
            extent_arr = np.array([min_pt, max_pt], dtype=np.float32)
            vt_extent = Vt.Vec3fArray.FromNumpy(extent_arr)
            extent_attr.Set(vt_extent, tc)

        # Set default prim
        stage.SetDefaultPrim(usd_mesh.GetPrim())

        # Commit to disk
        stage.Save()
        logger.info(
            f"Successfully exported lighting-consumable OpenUSD cache: {output_path} "
            f"({len(frames)} frames, {len(cache.frames[frames[0]])} points)"
        )
        return str(output_path)

    @classmethod
    def verify_usd_cache(cls, usd_file_path: str | Path) -> bool:
        """Verify that a USD cache file can be opened and queried by OpenUSD."""
        path = Path(usd_file_path).resolve()
        if not path.exists():
            return False

        try:
            stage = Usd.Stage.Open(str(path))
            if not stage:
                return False

            prim = stage.GetDefaultPrim()
            if not prim or not prim.IsA(UsdGeom.Mesh):
                return False

            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get(Usd.TimeCode(stage.GetStartTimeCode()))
            return len(pts) > 0
        except Exception as e:
            logger.error(f"USD cache verification failed: {e}")
            return False
