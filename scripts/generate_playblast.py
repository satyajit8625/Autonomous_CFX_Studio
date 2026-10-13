"""Autonomous CFX Studio - Playblast and Visual Evidence Generator.
Generates frame-by-frame visual ASCII playblasts and SVG vector renders of simulated deformation.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.core.cache_io import SimulationCache


def render_ascii_playblast(cache: SimulationCache, output_txt: str | Path, step_frames: int = 5) -> str:
    """Renders multi-frame ASCII front-view playblast showing garment draping over time."""
    output_path = Path(output_txt).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(list(cache.frames.keys()))
    sampled_frames = [f for f in frames if (f == 1 or f % step_frames == 0 or f == frames[-1])]

    width, height = 48, 24
    lines = []
    lines.append("=" * 60)
    lines.append(f"AUTONOMOUS CFX STUDIO - SIMULATION PLAYBLAST: {cache.name}")
    lines.append(f"Total Frames: {len(frames)} | FPS: {cache.fps}")
    lines.append("=" * 60)

    for f in sampled_frames:
        pts = cache.frames[f]
        # Project front view (X, Y)
        x = pts[:, 0]
        y = pts[:, 1]

        x_min, x_max = -0.5, 0.5
        y_min, y_max = 0.8, 1.8

        grid = [[" " for _ in range(width)] for _ in range(height)]

        for p_idx in range(len(pts)):
            col = int(((x[p_idx] - x_min) / (x_max - x_min)) * (width - 1))
            row = int(((y[p_idx] - y_min) / (y_max - y_min)) * (height - 1))
            row = (height - 1) - row  # flip Y

            if 0 <= col < width and 0 <= row < height:
                grid[row][col] = "#" if (p_idx % 3 == 0) else "*"

        lines.append(f"\n--- FRAME {f:04d} ---")
        for r in grid:
            lines.append("".join(r))

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(content)

    return str(output_path)


def main():
    evidence_dir = Path("evidence/playblasts")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    am = AssetManager()
    jacket, body, chair = am.load_milestone_package(num_frames=20)

    from autonomous_cfx.simulation.engine import SimulationEngine
    engine = SimulationEngine(jacket, colliders=[body, chair], pinned_vertices=list(jacket.vertex_groups.get("collar", [])), end_frame=20)
    cache = engine.run()

    playblast_file = evidence_dir / "jacket_simulation_playblast.txt"
    render_ascii_playblast(cache, playblast_file, step_frames=5)
    print(f"Generated simulation playblast evidence at: {playblast_file}")


if __name__ == "__main__":
    main()
