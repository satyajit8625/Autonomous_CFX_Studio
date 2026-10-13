"""Tests for importing external production geometry files and animated sequences from disk."""

import pytest
from pathlib import Path
from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.assets.external_importer import generate_external_asset_files, import_external_animated_collider_sequence


def test_external_asset_generation_and_ingestion(tmp_path):
    files = generate_external_asset_files(tmp_path, num_frames=5, fps=24.0)

    garment_file = Path(files["garment_file"])
    anim_folder = Path(files["animated_body_folder"])
    obstacle_file = Path(files["obstacle_file"])

    assert garment_file.exists()
    assert anim_folder.exists()
    assert obstacle_file.exists()
    assert len(list(anim_folder.glob("body_motion_*.obj"))) == 5

    # Load through AssetManager
    am = AssetManager(tmp_path)
    cloth, body_col, obs_col = am.load_external_package(
        garment_file=garment_file,
        body_sequence_folder=anim_folder,
        obstacle_file=obstacle_file,
        fps=24.0,
    )

    assert cloth.num_vertices > 0
    assert cloth.num_faces > 0
    assert len(body_col.animation_frames) == 5
    assert obs_col is not None

    # Test frame update from external animation
    body_col.update_frame(3.0)
    assert len(body_col.current_vertices) == body_col.geometry.num_vertices
