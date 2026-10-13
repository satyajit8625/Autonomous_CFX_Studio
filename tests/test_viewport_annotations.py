"""Tests for Viewport annotation canvas, drawing tools, undo, and bundle capture."""

import os
import tempfile
import pytest
from PySide6.QtWidgets import QApplication

from autonomous_cfx.assets.procedural import generate_procedural_jacket
from autonomous_cfx.core.surface_mapping import SurfaceSelection
from autonomous_cfx.ui.viewport_widget import Viewport3DWidget


@pytest.fixture(scope="session")
def qapp():
    """Headless QApplication instance for UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["", "-platform", "offscreen"])
    return app


def test_viewport_annotation_tools_and_undo(qapp):
    """Verify tool selection, stroke recording, undo, and clear in viewport."""
    vp = Viewport3DWidget()
    vp.resize(400, 400)
    mesh = generate_procedural_jacket()
    vp.set_cloth_geometry(mesh)

    assert vp.annotation_mode is False
    vp.set_annotation_mode(True)
    assert vp.annotation_mode is True

    # Draw brush stroke
    vp.set_annotation_tool("brush")
    vp.annotation_strokes.append({
        "tool": "brush",
        "points": [(200.0, 200.0), (205.0, 210.0)],
        "color": vp.active_color,
        "label": "Left sleeve penetration",
        "radius": 14.0
    })

    # Draw outline stroke
    vp.set_annotation_tool("outline")
    vp.annotation_strokes.append({
        "tool": "outline",
        "points": [(100.0, 100.0), (150.0, 100.0), (150.0, 150.0), (100.0, 100.0)],
        "color": vp.active_color,
        "label": "Protected cuff region",
        "radius": 2.0
    })

    assert len(vp.annotation_strokes) == 2

    # Undo
    vp.undo_annotation()
    assert len(vp.annotation_strokes) == 1
    assert vp.annotation_strokes[0]["label"] == "Left sleeve penetration"

    # Clear
    vp.clear_annotations()
    assert len(vp.annotation_strokes) == 0


def test_viewport_bundle_capture_and_3d_preview(qapp):
    """Verify capturing versioned bundle with clean render, mask, depth, face ID, and camera."""
    tmp_dir = tempfile.mkdtemp(prefix="cfx_vp_bundle_")
    vp = Viewport3DWidget()
    vp.resize(300, 300)
    mesh = generate_procedural_jacket()
    vp.set_cloth_geometry(mesh)

    vp.set_annotation_mode(True)
    vp.annotation_strokes.append({
        "tool": "brush",
        "points": [(150.0, 150.0)],
        "color": vp.active_color,
        "label": "Sleeve penetration",
        "radius": 14.0
    })

    bundle = vp.capture_annotation_bundle(
        bundle_dir=tmp_dir,
        shot_name="shot010",
        garment_name="jacket",
        version="v001",
        note_text="Remove body penetration here during frames 1045–1060."
    )

    assert os.path.exists(bundle.clean_image_path)
    assert os.path.exists(bundle.annotation_overlay_path)
    assert os.path.exists(bundle.depth_buffer_path)
    assert os.path.exists(bundle.face_id_buffer_path)
    assert len(bundle.regions) == 1
    assert bundle.regions[0].label == "Sleeve penetration"
    assert "1045" in bundle.regions[0].intent_description

    # Test 3D selection preview
    selection = SurfaceSelection(
        selected_faces=[1, 2],
        selected_vertices=[10, 11, 12],
        vertex_weights={10: 1.0, 11: 0.6, 12: 0.2},
        falloff_radius=0.05
    )
    vp.set_surface_selection_preview(selection)
    assert vp.surface_selection_preview == selection
