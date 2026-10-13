"""Autonomous CFX Studio - Cache I/O Subsystem.
Handles exporting and importing simulation caches across OBJ sequences, binary point caches (.vpcache),
and standard interchange manifests (Alembic/USD readiness).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.logger import logger


class SimulationCache:
    """In-memory and on-disk representation of simulated geometry cache across frames."""

    def __init__(
        self,
        name: str = "cloth_sim",
        fps: float = 24.0,
        start_frame: int = 1,
        base_faces: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.fps = fps
        self.start_frame = start_frame
        self.base_faces: np.ndarray = (
            base_faces if base_faces is not None else np.zeros((0, 3), dtype=np.int32)
        )
        # Dictionary mapping frame_number -> vertex_positions (N, 3)
        self.frames: Dict[int, np.ndarray] = {}
        # Dictionary mapping frame_number -> vertex_velocities (N, 3)
        self.velocities: Dict[int, np.ndarray] = {}
        # Per-frame diagnostic metrics
        self.frame_metrics: Dict[int, Dict[str, Any]] = {}
        # High-level metadata
        self.metadata: Dict[str, Any] = {
            "name": name,
            "fps": fps,
            "version": "1.0.0",
            "generator": "Autonomous CFX Studio",
        }

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def frame_numbers(self) -> List[int]:
        return sorted(self.frames.keys())

    def add_frame(
        self,
        frame_num: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a frame of vertex positions to the cache."""
        self.frames[frame_num] = np.array(positions, dtype=np.float32)
        if velocities is not None:
            self.velocities[frame_num] = np.array(velocities, dtype=np.float32)
        if metrics is not None:
            self.frame_metrics[frame_num] = metrics

    def get_frame_geometry(self, frame_num: int) -> Optional[MeshGeometry]:
        """Reconstruct MeshGeometry for a specific frame."""
        if frame_num not in self.frames:
            return None
        return MeshGeometry(
            name=f"{self.name}_f{frame_num:04d}",
            vertices=self.frames[frame_num],
            faces=self.base_faces,
        )

    def save_point_cache(self, file_path: str | Path) -> str:
        """Save point cache as compressed .vpcache (npz binary with metadata)."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        frames_list = self.frame_numbers
        if not frames_list:
            raise ValueError("Simulation cache contains no frames to save.")

        pos_array = np.stack([self.frames[f] for f in frames_list], axis=0)  # (F, N, 3)
        vel_array = (
            np.stack([self.velocities[f] for f in frames_list], axis=0)
            if self.velocities
            else np.zeros_like(pos_array)
        )

        meta_json = json.dumps(
            {
                "name": self.name,
                "fps": self.fps,
                "start_frame": self.start_frame,
                "frames": frames_list,
                "metadata": self.metadata,
                "frame_metrics": self.frame_metrics,
            }
        )

        with open(file_path, "wb") as f_out:
            np.savez_compressed(
                f_out,
                positions=pos_array,
                velocities=vel_array,
                faces=self.base_faces,
                meta=np.array(meta_json),
            )
        logger.info(f"Saved point cache: {file_path} ({len(frames_list)} frames)")
        return str(file_path)

    @classmethod
    def load_point_cache(cls, file_path: str | Path) -> SimulationCache:
        """Load .vpcache binary cache file."""
        file_path = Path(file_path)
        with open(file_path, "rb") as f_in:
            data = np.load(f_in)
            meta_dict = json.loads(str(data["meta"]))
            cache = cls(
                name=meta_dict.get("name", "cloth_sim"),
                fps=meta_dict.get("fps", 24.0),
                start_frame=meta_dict.get("start_frame", 1),
                base_faces=data["faces"],
            )
            cache.metadata = meta_dict.get("metadata", {})
            cache.frame_metrics = {
                int(k): v for k, v in meta_dict.get("frame_metrics", {}).items()
            }

            frames_list = meta_dict.get("frames", [])
            positions = data["positions"]
            velocities = data["velocities"]

            for idx, f in enumerate(frames_list):
                cache.frames[int(f)] = positions[idx]
                if velocities is not None and len(velocities) > idx:
                    cache.velocities[int(f)] = velocities[idx]
        logger.info(f"Loaded point cache: {file_path} ({len(frames_list)} frames)")
        return cache

    def export_obj_sequence(self, output_dir: str | Path, prefix: str = "cloth") -> List[str]:
        """Export sequence of Wavefront OBJ files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []

        for frame_num in self.frame_numbers:
            verts = self.frames[frame_num]
            obj_path = output_dir / f"{prefix}_{frame_num:04d}.obj"
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write(f"# Autonomous CFX Studio - Frame {frame_num}\n")
                f.write(f"o {prefix}\n")
                for v in verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                # 1-based indexing for OBJ faces
                for face in self.base_faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            exported_files.append(str(obj_path))

        logger.info(f"Exported {len(exported_files)} OBJ files to {output_dir}")
        return exported_files

    def export_interchange_manifest(
        self,
        output_dir: str | Path,
        alembic_equivalent_name: str = "cloth_sim.abc",
    ) -> str:
        """Create a standard Alembic/USD interchange manifest to bridge with Maya, Houdini, or Unreal."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{self.name}_manifest.json"

        frames = self.frame_numbers
        manifest_data = {
            "format": "Autonomous_CFX_Interchange_Manifest_v1",
            "name": self.name,
            "fps": self.fps,
            "frame_range": [min(frames), max(frames)] if frames else [0, 0],
            "total_frames": len(frames),
            "num_vertices": int(self.frames[frames[0]].shape[0]) if frames else 0,
            "num_faces": int(len(self.base_faces)),
            "alembic_target": alembic_equivalent_name,
            "point_cache_file": f"{self.name}.vpcache",
            "metadata": self.metadata,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Exported interchange manifest: {manifest_path}")
        return str(manifest_path)

    def export_usd_cache(
        self,
        output_file_path: str | Path,
        prim_path: str = "/World/ClothMesh",
    ) -> str:
        """Export native lighting-consumable OpenUSD (.usda / .usd) geometry cache."""
        from autonomous_cfx.core.usd_export import USDExporter
        return USDExporter.export_cache(self, output_file_path, prim_path=prim_path)
