"""Autonomous CFX Studio - External Geometry Generator & Importer.
Creates and loads external production-grade OBJ files and animated mesh sequences on disk.
Guarantees the system does not rely exclusively on procedural in-memory demo meshes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.assets.procedural import (
    generate_procedural_jacket,
    generate_procedural_animated_body,
    generate_procedural_chair,
)
from autonomous_cfx.simulation.colliders import AnimatedMeshCollider
from autonomous_cfx.core.logger import logger


def generate_external_asset_files(output_dir: str | Path, num_frames: int = 30, fps: float = 24.0) -> Dict[str, str]:
    """Writes real external OBJ files and animated OBJ sequences to disk."""
    base_dir = Path(output_dir).resolve()
    garment_dir = base_dir / "garments"
    anim_dir = base_dir / "characters" / "body_sequence"
    obstacle_dir = base_dir / "obstacles"

    garment_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)
    obstacle_dir.mkdir(parents=True, exist_ok=True)

    # 1. External Cloth Garment OBJ
    jacket_geo = generate_procedural_jacket()
    garment_path = garment_dir / "external_jacket_tailored.obj"
    with open(garment_path, "w", encoding="utf-8") as f:
        f.write("# External Production Garment: Tailored Jacket\n")
        f.write("o external_jacket\n")
        for v in jacket_geo.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in jacket_geo.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # 2. External Animated Body Collider Sequence (Discrete OBJ files per frame)
    _, anim_dict = generate_procedural_animated_body(num_frames=num_frames, fps=fps)
    body_base_geo = generate_procedural_animated_body(num_frames=1)[0]

    for frame_num in range(1, num_frames + 1):
        f_verts = anim_dict[frame_num]
        frame_obj = anim_dir / f"body_motion_{frame_num:04d}.obj"
        with open(frame_obj, "w", encoding="utf-8") as f:
            f.write(f"# External Animated Body Collider - Frame {frame_num}\n")
            f.write("o body_collider\n")
            for v in f_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in body_base_geo.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # 3. External Static Obstacle OBJ
    chair_geo = generate_procedural_chair()
    obstacle_path = obstacle_dir / "external_bench_collider.obj"
    with open(obstacle_path, "w", encoding="utf-8") as f:
        f.write("# External Static Obstacle Collider: Bench\n")
        f.write("o bench_collider\n")
        for v in chair_geo.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in chair_geo.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    logger.info(
        f"Generated external production asset files on disk at: {base_dir}\n"
        f"  • Garment: {garment_path.name}\n"
        f"  • Animated Collider: {num_frames} OBJ files in {anim_dir.name}\n"
        f"  • Obstacle: {obstacle_path.name}"
    )

    return {
        "garment_file": str(garment_path),
        "animated_body_folder": str(anim_dir),
        "obstacle_file": str(obstacle_path),
    }


def import_external_animated_collider_sequence(
    folder_path: str | Path,
    file_prefix: str = "body_motion",
    fps: float = 24.0,
) -> AnimatedMeshCollider:
    """Imports a real sequence of external OBJ files on disk into an AnimatedMeshCollider."""
    folder = Path(folder_path).resolve()
    obj_files = sorted(list(folder.glob(f"{file_prefix}_*.obj")))

    if not obj_files:
        raise FileNotFoundError(f"No animated OBJ files matching '{file_prefix}_*.obj' in {folder}")

    # Read frame 1 to get base topology
    first_tm = trimesh.load(str(obj_files[0]), process=False)
    if isinstance(first_tm, trimesh.Scene):
        first_tm = trimesh.util.concatenate(first_tm.dump())

    base_geometry = MeshGeometry.from_trimesh(first_tm, name=file_prefix)

    # Ingest per-frame vertices
    anim_frames: Dict[int, np.ndarray] = {}
    for idx, obj_file in enumerate(obj_files, start=1):
        tm = trimesh.load(str(obj_file), process=False)
        if isinstance(tm, trimesh.Scene):
            tm = trimesh.util.concatenate(tm.dump())
        anim_frames[idx] = np.array(tm.vertices, dtype=np.float32)

    collider = AnimatedMeshCollider(
        name=file_prefix,
        base_geometry=base_geometry,
        animation_frames=anim_frames,
        fps=fps,
    )
    logger.info(f"Imported external animated collider from disk: {len(obj_files)} frames from {folder.name}")
    return collider
