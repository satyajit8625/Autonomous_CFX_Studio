"""Tests for procedural milestone assets (Jacket, Animated Body, Chair)."""

import pytest
from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.assets.procedural import (
    generate_procedural_animated_body,
    generate_procedural_chair,
    generate_procedural_jacket,
)


def test_procedural_jacket():
    jacket = generate_procedural_jacket()
    assert jacket.num_vertices > 100
    assert jacket.num_faces > 100
    assert "collar" in jacket.vertex_groups
    assert len(jacket.vertex_groups["collar"]) > 0

    bbox = jacket.get_bounding_box()
    # Check dimensions match human jacket scale
    extents = bbox.extents
    assert 0.3 < extents[0] < 2.0  # Width across sleeves
    assert 0.4 < extents[1] < 1.2  # Height


def test_procedural_animated_body():
    body_geo, anim_dict = generate_procedural_animated_body(num_frames=10, fps=24.0)
    assert body_geo.num_vertices > 50
    assert len(anim_dict) == 10
    assert 1 in anim_dict
    assert 10 in anim_dict
    assert anim_dict[1].shape == body_geo.vertices.shape


def test_procedural_chair():
    chair = generate_procedural_chair()
    assert chair.num_vertices > 0
    assert chair.num_faces > 0
    bbox = chair.get_bounding_box()
    assert bbox.extents[1] > 0.4  # Height of chair


def test_asset_manager_milestone_package(tmp_path):
    mgr = AssetManager(tmp_path)
    jacket, body_col, chair_col = mgr.load_milestone_package(
        num_frames=10,
        fps=24.0,
        include_chair=True,
    )
    assert jacket is not None
    assert body_col is not None
    assert chair_col is not None
