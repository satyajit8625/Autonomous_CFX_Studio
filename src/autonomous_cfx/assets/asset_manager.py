"""Autonomous CFX Studio - Asset Manager.
Loads, saves, and provisions milestone assets (Jacket, Body collider, Chair collider).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import trimesh

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.assets.procedural import (
    generate_procedural_jacket,
    generate_procedural_animated_body,
    generate_procedural_chair,
)
from autonomous_cfx.simulation.colliders import AnimatedMeshCollider
from autonomous_cfx.core.logger import logger


class AssetManager:
    """Manages geometric assets and collider generation."""

    def __init__(self, asset_dir: Optional[str | Path] = None):
        self.asset_dir = Path(asset_dir) if asset_dir else Path("assets")
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def load_mesh_file(self, file_path: str | Path) -> MeshGeometry:
        """Load external 3D mesh (OBJ, STL, PLY) into MeshGeometry."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        tm = trimesh.load(str(path), process=False)
        if isinstance(tm, trimesh.Scene):
            tm = trimesh.util.concatenate(tm.dump())

        mesh = MeshGeometry.from_trimesh(tm, name=path.stem)
        logger.info(f"Loaded mesh '{mesh.name}' ({mesh.num_vertices} vertices, {mesh.num_faces} faces)")
        return mesh

    def save_mesh_obj(self, mesh: MeshGeometry, output_path: str | Path) -> str:
        """Save MeshGeometry to Wavefront OBJ."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tm = mesh.to_trimesh()
        tm.export(str(output_path))
        logger.info(f"Saved mesh '{mesh.name}' to {output_path}")
        return str(output_path)

    def load_milestone_package(
        self,
        num_frames: int = 40,
        fps: float = 24.0,
        include_chair: bool = True,
        sit_animation: bool = False,
    ) -> Tuple[MeshGeometry, AnimatedMeshCollider, Optional[AnimatedMeshCollider]]:
        """Load or generate the canonical milestone setup: Jacket cloth, Animated Body, Chair collider."""
        jacket = generate_procedural_jacket()

        body_geo, body_anim = generate_procedural_animated_body(
            num_frames=num_frames,
            fps=fps,
            sit_on_chair=sit_animation,
        )
        body_collider = AnimatedMeshCollider(
            name="body_collider",
            base_geometry=body_geo,
            animation_frames=body_anim,
            fps=fps,
        )

        chair_collider = None
        if include_chair:
            chair_geo = generate_procedural_chair()
            # Chair is static across all frames
            chair_anim = {f: chair_geo.vertices for f in range(1, num_frames + 1)}
            chair_collider = AnimatedMeshCollider(
                name="chair_collider",
                base_geometry=chair_geo,
                animation_frames=chair_anim,
                fps=fps,
            )

        logger.info(
            f"Prepared milestone assets: Jacket ({jacket.num_vertices} verts), "
            f"Body ({body_geo.num_vertices} verts, {num_frames} frames), "
            f"Chair ({'enabled' if chair_collider else 'disabled'})"
        )
        return jacket, body_collider, chair_collider
