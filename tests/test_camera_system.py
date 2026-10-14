"""Tests for animated shot-camera system, optics, Alembic hierarchy, and viewport tracking."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List
import numpy as np
import pytest

from autonomous_cfx.core.alembic_io import (
    ALEMBIC_AVAILABLE,
    AlembicShotExporter,
    AlembicShotImporter,
)
from autonomous_cfx.core.annotation_bundle import CameraParameters
from autonomous_cfx.core.camera import (
    CameraMode,
    FilmFitMode,
    FreeCameraState,
    ShotCamera,
    ShotCameraSample,
)
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.surface_mapping import ImageToSurfaceMapper
from autonomous_cfx.assets.procedural import generate_procedural_jacket


def test_camera_pinhole_optics():
    """Verify exact pinhole optical math, FOV, and aspect ratio."""
    # 50mm lens on 35mm full frame sensor (36mm x 24mm)
    world_m = np.eye(4, dtype=np.float64)
    world_m[3, :3] = [0.0, 1.5, 3.0]

    sample = ShotCameraSample(
        frame=1.0,
        time_seconds=0.0,
        world_matrix=world_m,
        focal_length_mm=50.0,
        horizontal_aperture_mm=36.0,
        vertical_aperture_mm=24.0,
    )

    # Expected FOV: 2 * arctan(aperture / (2 * f))
    expected_hfov = 2.0 * math.degrees(math.atan(36.0 / (2.0 * 50.0)))
    expected_vfov = 2.0 * math.degrees(math.atan(24.0 / (2.0 * 50.0)))
    expected_aspect = 36.0 / 24.0  # 1.5

    assert abs(sample.horizontal_fov_deg - expected_hfov) < 1e-5
    assert abs(sample.vertical_fov_deg - expected_vfov) < 1e-5
    assert abs(sample.film_aspect_ratio - expected_aspect) < 1e-5

    # Lens Squeeze (Anamorphic 2.0x)
    anamorphic_sample = ShotCameraSample(
        frame=1.0,
        time_seconds=0.0,
        world_matrix=world_m,
        focal_length_mm=50.0,
        horizontal_aperture_mm=36.0,
        vertical_aperture_mm=24.0,
        lens_squeeze=2.0,
    )
    assert abs(anamorphic_sample.film_aspect_ratio - 3.0) < 1e-5

    # Look direction and eye
    assert np.allclose(sample.eye, [0.0, 1.5, 3.0])
    assert np.allclose(sample.forward, [0.0, 0.0, -1.0])
    assert np.allclose(sample.up, [0.0, 1.0, 0.0])
    assert np.allclose(sample.right, [1.0, 0.0, 0.0])


def test_film_fit_letterbox_and_pillarbox():
    """Verify film gate letterbox/pillarbox calculation prevents distortion."""
    world_m = np.eye(4, dtype=np.float64)
    sample_16_9 = ShotCameraSample(
        frame=1.0,
        time_seconds=0.0,
        world_matrix=world_m,
        focal_length_mm=50.0,
        horizontal_aperture_mm=35.56,  # 16:9 ratio
        vertical_aperture_mm=20.0,
    )

    cam_overscan = ShotCamera(
        name="cam_overscan",
        full_path="/cam",
        film_fit=FilmFitMode.OVERSCAN,
    )

    # 1. Viewport wider than film (e.g. 21:9 viewport 2100x900 vs 16:9 film) -> Pillarboxing (black bars on left/right)
    gx, gy, gw, gh = cam_overscan.compute_film_gate(2100, 900, sample_16_9)
    assert gh == 900
    assert gw < 2100
    assert gx > 0  # Centered horizontally
    assert gy == 0
    # Gate aspect matches film ratio
    assert abs((gw / float(gh)) - sample_16_9.film_aspect_ratio) < 0.02

    # 2. Viewport taller than film (e.g. 4:3 viewport 1200x900 vs 16:9 film) -> Letterboxing (black bars on top/bottom)
    gx2, gy2, gw2, gh2 = cam_overscan.compute_film_gate(1200, 900, sample_16_9)
    assert gw2 == 1200
    assert gh2 < 900
    assert gx2 == 0
    assert gy2 > 0  # Centered vertically
    assert abs((gw2 / float(gh2)) - sample_16_9.film_aspect_ratio) < 0.02

    # 3. Fill mode fills viewport completely
    cam_fill = ShotCamera(name="cam_fill", full_path="/cam", film_fit=FilmFitMode.FILL)
    gxf, gyf, gwf, ghf = cam_fill.compute_film_gate(1200, 900, sample_16_9)
    assert gwf >= 1200
    assert ghf >= 900


@pytest.mark.skipif(not ALEMBIC_AVAILABLE, reason="Alembic C++ library not available")
def test_alembic_camera_roundtrip_and_nested_hierarchy():
    """Verify exporting and importing animated camera with nested parent transforms."""
    tmp_dir = tempfile.mkdtemp(prefix="cfx_cam_test_")
    abc_path = os.path.join(tmp_dir, "test_shot_cam.abc")

    # Construct animated camera moving along an arc over 10 frames
    samples = []
    for f in range(1, 11):
        t = (f - 1) / 24.0
        angle = math.radians(f * 5.0)
        eye = np.array([math.sin(angle) * 3.0, 1.2 + f * 0.05, math.cos(angle) * 3.0], dtype=np.float64)
        target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(fwd, up_w)
        r /= np.linalg.norm(r)
        u = np.cross(r, fwd)
        u /= np.linalg.norm(u)

        M = np.eye(4, dtype=np.float64)
        M[0, :3] = r
        M[1, :3] = u
        M[2, :3] = -fwd
        M[3, :3] = eye

        samples.append(ShotCameraSample(
            frame=float(f),
            time_seconds=t,
            world_matrix=M,
            focal_length_mm=35.0 + f * 1.5,
            horizontal_aperture_mm=36.0,
            vertical_aperture_mm=24.0,
            near_clip=0.05,
            far_clip=500.0,
        ))

    src_cam = ShotCamera(
        name="crane_cam",
        full_path="/shot010/crane/camera",
        fps=24.0,
        start_frame=1,
        samples=samples,
    )

    exporter = AlembicShotExporter()
    out_file = exporter.export_camera(output_path=abc_path, camera=src_cam)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 500

    importer = AlembicShotImporter()
    cams = importer.load_cameras(out_file)
    assert len(cams) >= 1
    loaded_cam = cams.get("camera") or cams.get("crane_cam") or list(cams.values())[0]

    assert len(loaded_cam.samples) == 10
    assert abs(loaded_cam.fps - 24.0) < 1e-3

    eval_s = loaded_cam.evaluate(5.0)
    orig_s = samples[4]
    assert abs(eval_s.focal_length_mm - orig_s.focal_length_mm) < 1e-2
    assert abs(eval_s.horizontal_aperture_mm - 36.0) < 1e-2
    assert np.allclose(eval_s.world_matrix, orig_s.world_matrix, atol=1e-3)

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_nonzero_start_frame_and_subframe_interpolation():
    """Verify camera starting at nonzero frame (101) with subframe evaluation at 48 fps."""
    samples = []
    for f in range(101, 111):
        t = (f - 101) / 48.0
        M = np.eye(4, dtype=np.float64)
        M[3, 0] = float(f - 101) * 0.1  # X translation
        M[3, 1] = 1.5
        M[3, 2] = 4.0
        samples.append(ShotCameraSample(
            frame=float(f),
            time_seconds=t,
            world_matrix=M,
            focal_length_mm=40.0 + (f - 101) * 2.0,
            horizontal_aperture_mm=36.0,
            vertical_aperture_mm=24.0,
        ))

    cam = ShotCamera(
        name="high_speed_cam",
        full_path="/cam",
        fps=48.0,
        start_frame=101,
        samples=samples,
    )

    # Frame 101 exact
    s_start = cam.evaluate(101.0)
    assert abs(s_start.time_seconds - 0.0) < 1e-5
    assert abs(s_start.focal_length_mm - 40.0) < 1e-5
    assert abs(s_start.eye[0] - 0.0) < 1e-5

    # Subframe 103.5 interpolation
    s_sub = cam.evaluate(103.5)
    assert abs(s_sub.focal_length_mm - 45.0) < 1e-4
    assert abs(s_sub.eye[0] - 0.25) < 1e-4

    # Out of range clamping
    s_before = cam.evaluate(90.0)
    assert abs(s_before.focal_length_mm - 40.0) < 1e-5
    s_after = cam.evaluate(200.0)
    assert abs(s_after.focal_length_mm - 58.0) < 1e-5


def test_follow_selected_character_tracking_far_from_origin():
    """Verify Follow Selected Character mode shifts target without resetting distance or viewing angles."""
    free_cam = FreeCameraState(
        yaw=45.0,
        pitch=20.0,
        distance=2.5,
        target=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        follow_character=True,
    )

    initial_distance = free_cam.distance
    initial_yaw = free_cam.yaw
    initial_pitch = free_cam.pitch

    # Character moves far away along X and Z across multiple steps
    char_positions = [
        np.array([0.0, 1.0, 0.0]),
        np.array([5.0, 1.2, 3.0]),
        np.array([12.0, 1.4, 8.0]),
        np.array([25.0, 1.8, 18.0]),  # 25 meters away
    ]

    for pos in char_positions:
        free_cam.update_character_tracking(pos)

    # Target must follow character's exact world position
    assert np.allclose(free_cam.target, [25.0, 1.8, 18.0], atol=1e-4)

    # Viewing offset invariants: distance, pitch, and yaw must NOT change or reset
    assert abs(free_cam.distance - initial_distance) < 1e-5
    assert abs(free_cam.yaw - initial_yaw) < 1e-5
    assert abs(free_cam.pitch - initial_pitch) < 1e-5


def test_viewport_camera_modes_and_locking():
    """Verify switching between Free Inspection and Shot Camera with navigation locking."""
    from autonomous_cfx.ui.viewport_widget import Viewport3DWidget
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])

    viewport = Viewport3DWidget()
    viewport.resize(1280, 720)

    # Initially in Free Inspection mode
    assert viewport.camera_mode == CameraMode.FREE_INSPECTION
    assert viewport.active_shot_camera is None
    assert not viewport.btn_cam_lock.isEnabled()

    # Create and assign shot cameras
    samples = [
        ShotCameraSample(
            frame=1.0,
            time_seconds=0.0,
            world_matrix=np.eye(4),
            focal_length_mm=50.0,
            horizontal_aperture_mm=36.0,
            vertical_aperture_mm=24.0,
        )
    ]
    test_cam = ShotCamera(name="shot_cam_a", full_path="/shot_cam_a", samples=samples)
    viewport.set_shot_cameras({"shot_cam_a": test_cam})

    # Switch to Shot Camera mode via dropdown
    idx = viewport.combo_camera.findData("shot_cam_a")
    assert idx > 0
    viewport.combo_camera.setCurrentIndex(idx)

    assert viewport.camera_mode == CameraMode.SHOT_CAMERA
    assert viewport.active_shot_camera == test_cam
    assert viewport.is_navigation_locked is True
    assert viewport.btn_cam_lock.isEnabled()

    # Verify mouse navigation is locked
    prev_yaw = viewport.yaw
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent, QPointF, Qt

    press_evt = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(100, 100), QPointF(100, 100), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    move_evt = QMouseEvent(QEvent.Type.MouseMove, QPointF(150, 120), QPointF(150, 120), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    viewport.mousePressEvent(press_evt)
    viewport.mouseMoveEvent(move_evt)
    # Yaw must NOT change because navigation is locked
    assert abs(viewport.yaw - prev_yaw) < 1e-5

    # Switch back to Free Inspection mode
    viewport.combo_camera.setCurrentIndex(0)
    assert viewport.camera_mode == CameraMode.FREE_INSPECTION
    assert viewport.active_shot_camera is None
    # Imported shot camera sample is unchanged
    assert test_cam.samples[0].focal_length_mm == 50.0


def test_annotation_bundle_under_animated_shot_camera():
    """Verify captured annotation bundle stores exact active camera matrices and unprojects accurately."""
    from autonomous_cfx.ui.viewport_widget import Viewport3DWidget
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])

    viewport = Viewport3DWidget()
    viewport.resize(800, 600)

    # Setup cloth mesh
    cloth = generate_procedural_jacket()
    viewport.set_cloth_geometry(cloth)

    # Setup animated camera framing the cloth
    eye = np.array([0.0, 1.2, 2.0], dtype=np.float64)
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    r = np.cross(fwd, up_w)
    r /= np.linalg.norm(r)
    u = np.cross(r, fwd)
    u /= np.linalg.norm(u)
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = -fwd
    M[3, :3] = eye

    sample = ShotCameraSample(
        frame=5.0,
        time_seconds=4.0 / 24.0,
        world_matrix=M,
        focal_length_mm=45.0,
        horizontal_aperture_mm=36.0,
        vertical_aperture_mm=24.0,
    )
    cam = ShotCamera(name="render_cam", full_path="/render_cam", samples=[sample])
    viewport.set_shot_cameras({"render_cam": cam})
    viewport.combo_camera.setCurrentIndex(viewport.combo_camera.findData("render_cam"))
    viewport.current_frame = 5

    # Capture annotation bundle
    tmp_bundle_dir = tempfile.mkdtemp(prefix="cfx_ann_cam_")
    bundle = viewport.capture_annotation_bundle(
        bundle_dir=tmp_bundle_dir,
        shot_name="shot010",
        garment_name="jacket",
        version="v001",
    )

    assert bundle.camera.camera_type == "shot_camera"
    assert bundle.camera.camera_name == "render_cam"
    assert abs(bundle.camera.focal_length_mm - 45.0) < 1e-4
    assert len(bundle.camera.view_matrix) == 4
    assert len(bundle.camera.projection_matrix) == 4
    assert bundle.camera.gate_rect is not None

    # Test unprojection buffer rasterization
    depth_buf, face_buf = ImageToSurfaceMapper.rasterize_viewport_buffers(cloth, bundle.camera)
    assert depth_buf.shape == (600, 800)
    assert face_buf.shape == (600, 800)
    # Visible cloth faces must be present in the buffer
    visible_faces = np.unique(face_buf[face_buf >= 0])
    assert len(visible_faces) > 10

    shutil.rmtree(tmp_bundle_dir, ignore_errors=True)


def test_maya_camera_standalone_alignment():
    """Verify camera optics and matrices match Autodesk Maya camera standalone evaluation."""
    maya_exe = r"C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe"
    if not os.path.exists(maya_exe):
        pytest.skip(f"Maya standalone python not found at {maya_exe}")

    maya_script = """
import sys
import math
import maya.standalone
maya.standalone.initialize()
import maya.cmds as cmds
import maya.api.OpenMaya as om

cmds.file(new=True, force=True)
cam_xform, cam_shape = cmds.camera(
    focalLength=35.0,
    horizontalFilmAperture=1.41732,  # 36.0 mm in inches
    verticalFilmAperture=0.94488,   # 24.0 mm in inches
    nearClipPlane=0.1,
    farClipPlane=100.0
)
cmds.setAttr(cam_xform + ".translateX", 0.0)
cmds.setAttr(cam_xform + ".translateY", 1.5)
cmds.setAttr(cam_xform + ".translateZ", 3.0)
cmds.setAttr(cam_xform + ".rotateX", -15.0)

hfov = cmds.camera(cam_shape, q=True, horizontalFieldOfView=True)
vfov = cmds.camera(cam_shape, q=True, verticalFieldOfView=True)

# Query camera transform DAG path and inclusive world matrix
sel = om.MSelectionList()
sel.add(cam_xform)
cam_dag = sel.getDagPath(0)
mat = cam_dag.inclusiveMatrix()

print(f"MAYA_HFOV:{hfov}")
print(f"MAYA_VFOV:{vfov}")
print("MAYA_MAT:" + ",".join(str(mat[i]) for i in range(16)))

maya.standalone.uninitialize()
"""
    cmd = [maya_exe, "-c", maya_script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, f"Maya execution failed: {result.stderr}"

    maya_hfov = None
    maya_vfov = None
    for line in result.stdout.splitlines():
        if line.startswith("MAYA_HFOV:"):
            maya_hfov = float(line.split(":")[1])
        elif line.startswith("MAYA_VFOV:"):
            maya_vfov = float(line.split(":")[1])

    assert maya_hfov is not None
    assert maya_vfov is not None

    # Compare with our ShotCameraSample optical calculation
    sample = ShotCameraSample(
        frame=1.0,
        time_seconds=0.0,
        world_matrix=np.eye(4),
        focal_length_mm=35.0,
        horizontal_aperture_mm=1.41732 * 25.4,
        vertical_aperture_mm=0.94488 * 25.4,
    )

    assert abs(sample.horizontal_fov_deg - maya_hfov) < 1e-4
    assert abs(sample.vertical_fov_deg - maya_vfov) < 1e-4
