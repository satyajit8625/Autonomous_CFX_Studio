"""Tests for Image-to-Geometry Surface Mapping, occlusion filtering, and spatial falloff."""

import numpy as np
import pytest

from autonomous_cfx.assets.procedural import generate_procedural_jacket
from autonomous_cfx.core.annotation_bundle import CameraParameters
from autonomous_cfx.core.surface_mapping import ImageToSurfaceMapper


def test_surface_mapping_rasterization_and_selection():
    """Verify rasterizing mesh buffers and unprojecting 2D mask to surface geometry."""
    mesh = generate_procedural_jacket()

    # Camera looking at jacket from front
    camera = CameraParameters(
        yaw=0.0,
        pitch=0.0,
        distance=2.5,
        target=(0.0, 1.0, 0.0),
        fov=1.8,
        width=200,
        height=200,
        view_matrix=[],
        projection_matrix=[]
    )

    depth_buf, face_id_buf = ImageToSurfaceMapper.rasterize_viewport_buffers(mesh, camera)

    assert depth_buf.shape == (200, 200)
    assert face_id_buf.shape == (200, 200)

    # There should be rasterized pixels in center
    valid_pixels = face_id_buf >= 0
    num_valid = np.sum(valid_pixels)
    assert num_valid > 100, f"Expected rasterized faces, got {num_valid}"

    # Create a 2D mask in the center
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[90:110, 90:110] = 255

    selection = ImageToSurfaceMapper.map_mask_to_surface(
        mask=mask,
        face_id_buffer=face_id_buf,
        mesh=mesh,
        falloff_radius=0.08
    )

    assert len(selection.selected_faces) > 0
    assert len(selection.selected_vertices) > 0
    assert len(selection.vertex_weights) >= len(selection.selected_vertices)

    # Check falloff weights: core vertices must be 1.0, outer falloff in (0, 1)
    weights = list(selection.vertex_weights.values())
    assert 1.0 in weights
    assert any(0.0 < w < 1.0 for w in weights)


def test_surface_mapping_occlusion_and_backface_exclusion():
    """Verify that faces on the back of the jacket are NOT selected by front-facing mask."""
    mesh = generate_procedural_jacket()

    # Front camera
    camera = CameraParameters(
        yaw=0.0,
        pitch=0.0,
        distance=2.5,
        target=(0.0, 1.0, 0.0),
        fov=1.8,
        width=150,
        height=150,
        view_matrix=[],
        projection_matrix=[]
    )

    depth_buf, face_id_buf = ImageToSurfaceMapper.rasterize_viewport_buffers(mesh, camera)

    # Face IDs visible from front
    front_faces = set(np.unique(face_id_buf[face_id_buf >= 0]))

    # Vertices of visible faces must be on the front side of the jacket (closer to camera along +Z)
    front_face_list = list(front_faces)
    v_ids = np.unique(mesh.faces[front_face_list])
    visible_verts = mesh.vertices[v_ids]

    assert np.mean(visible_verts[:, 2]) > 0.0, "Visible surface must be on front side of jacket (Z > 0)"
    # Back-most vertices at Z ~ -0.225 must not be visible
    assert np.min(visible_verts[:, 2]) > -0.15, "Deep back-facing vertices must be occluded"


def test_external_screenshot_ambiguity_flag():
    """Verify that externally uploaded screenshots are honestly flagged as ambiguous."""
    mesh = generate_procedural_jacket()

    selection = ImageToSurfaceMapper.handle_external_screenshot(
        mesh=mesh,
        approx_region_hint="sleeve",
        confidence=0.4
    )

    assert selection.is_ambiguous is True
    assert selection.confidence_score == 0.4
    assert "tentative/ambiguous" in selection.notes
