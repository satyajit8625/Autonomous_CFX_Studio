"""Autonomous CFX Studio - Interactive 3D Viewport.
Software-rasterized/accelerated 3D projection viewport supporting Orbit, Pan, Zoom, wireframe, and shaded rendering.
Includes automated bounding box framing (Frame All, Frame Selected), before/after comparison, quality defect heatmap,
and full 2D annotation capture canvas.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    OPENGL_WIDGET_AVAILABLE = True
except ImportError:
    from PySide6.QtWidgets import QWidget as QOpenGLWidget  # type: ignore
    OPENGL_WIDGET_AVAILABLE = False

try:
    import OpenGL.GL as gl
    from OpenGL.GLU import gluLookAt, gluPerspective
    PYOPENGL_AVAILABLE = True
except ImportError:
    gl = None
    gluLookAt = None
    gluPerspective = None
    PYOPENGL_AVAILABLE = False

from autonomous_cfx.core.camera import (
    CameraMode,
    FilmFitMode,
    FreeCameraState,
    ShotCamera,
    ShotCameraSample,
)
from autonomous_cfx.core.geometry import MeshGeometry


class Viewport3DWidget(QOpenGLWidget):
    """Interactive 3D preview viewport with mouse orbit, pan, and zoom.
    Supports hardware-accelerated OpenGL rasterization with automatic CPU QPainter fallback."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self._gl_initialized: bool = False

        # Camera parameters
        self.yaw: float = 30.0    # degrees
        self.pitch: float = 12.0  # degrees
        self.distance: float = 1.8  # Closer initial distance for character focus
        self.target: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Mouse interaction tracking
        self.last_mouse_pos: Optional[QPointF] = None
        self.is_orbiting: bool = False
        self.is_panning: bool = False

        # Displayed geometries
        self.cloth_geometry: Optional[MeshGeometry] = None
        self.collider_geometries: List[MeshGeometry] = []
        self.contact_points: Optional[np.ndarray] = None

        # Display toggles
        self.show_wireframe: bool = True
        self.show_colliders: bool = True
        self.show_grid: bool = True
        self.show_hud: bool = False  # Keep clean character focus by default
        self.current_frame: int = 1

        # Comparison State (Before / After)
        self.pre_correction_geometry: Optional[MeshGeometry] = None
        self.show_comparison_before: bool = False

        # Quality Heatmap Overlay
        self.show_quality_heatmap: bool = False
        self.vertex_penetrations_mm: Optional[np.ndarray] = None

        # Annotation Canvas State
        self.annotation_mode: bool = False
        self.current_tool: str = "brush"  # 'brush', 'outline', 'arrow', 'text'
        self.brush_radius: float = 14.0
        self.active_stroke: List[Tuple[float, float]] = []
        self.annotation_strokes: List[Dict[str, Any]] = []
        self.undo_stack: List[List[Dict[str, Any]]] = []
        self.current_region_label: str = "Defect Region"
        self.active_color: QColor = QColor(239, 68, 68, 200)

        # 3D Surface Selection Preview
        self.surface_selection_preview: Any = None

        # Camera modes & animated shot cameras
        self.camera_mode: CameraMode = CameraMode.FREE_INSPECTION
        self.free_camera_state: FreeCameraState = FreeCameraState(
            yaw=self.yaw, pitch=self.pitch, distance=self.distance, target=self.target
        )
        self.active_shot_camera: Optional[ShotCamera] = None
        self.shot_cameras: Dict[str, ShotCamera] = {}
        self.is_navigation_locked: bool = True
        self.active_gate_rect: Tuple[int, int, int, int] = (0, 0, 400, 400)

        self._build_overlay_toolbar()

    def _build_overlay_toolbar(self):
        """Top overlay bar for quick viewport actions (Camera, Frame All, Wireframe, Heatmap)."""
        layout_main = QVBoxLayout(self)
        layout_main.setContentsMargins(10, 10, 10, 10)

        bar_frame = QWidget(self)
        bar_layout = QHBoxLayout(bar_frame)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(6)

        # Camera Mode Selector
        self.combo_camera = QComboBox(self)
        self.combo_camera.setStyleSheet(self._combo_style())
        self.combo_camera.addItem("🎥 Free Inspection", "free")
        self.combo_camera.currentIndexChanged.connect(self._on_camera_selection_changed)
        bar_layout.addWidget(self.combo_camera)

        # Navigation Lock Toggle (for Shot Camera)
        self.btn_cam_lock = QPushButton("🔒 Locked")
        self.btn_cam_lock.setCheckable(True)
        self.btn_cam_lock.setChecked(True)
        self.btn_cam_lock.setEnabled(False)  # only relevant in shot camera mode
        self.btn_cam_lock.setToolTip("Lock viewport navigation to animated shot camera")
        self.btn_cam_lock.setStyleSheet(self._button_style())
        self.btn_cam_lock.clicked.connect(self._toggle_camera_lock)
        bar_layout.addWidget(self.btn_cam_lock)

        # Follow Selected Character Toggle (for Free Inspection)
        self.btn_follow_char = QPushButton("🏃 Follow")
        self.btn_follow_char.setCheckable(True)
        self.btn_follow_char.setChecked(False)
        self.btn_follow_char.setToolTip("Follow moving character in Free Inspection mode without resetting zoom/framing")
        self.btn_follow_char.setStyleSheet(self._button_style())
        self.btn_follow_char.clicked.connect(self._toggle_follow_character)
        bar_layout.addWidget(self.btn_follow_char)

        btn_frame_all = QPushButton("🎯 Frame All")
        btn_frame_all.setStyleSheet(self._button_style())
        btn_frame_all.clicked.connect(self.frame_all)
        bar_layout.addWidget(btn_frame_all)

        self.btn_wire = QPushButton("🌐 Wire")
        self.btn_wire.setCheckable(True)
        self.btn_wire.setChecked(self.show_wireframe)
        self.btn_wire.setStyleSheet(self._button_style())
        self.btn_wire.clicked.connect(self._toggle_wireframe)
        bar_layout.addWidget(self.btn_wire)

        self.btn_cols = QPushButton("🧍 Colliders")
        self.btn_cols.setCheckable(True)
        self.btn_cols.setChecked(self.show_colliders)
        self.btn_cols.setStyleSheet(self._button_style())
        self.btn_cols.clicked.connect(self._toggle_colliders)
        bar_layout.addWidget(self.btn_cols)

        self.btn_heatmap = QPushButton("🔥 Defect Heatmap")
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setStyleSheet(self._button_style())
        self.btn_heatmap.clicked.connect(self._toggle_heatmap)
        bar_layout.addWidget(self.btn_heatmap)

        self.btn_hud = QPushButton("ℹ️ HUD")
        self.btn_hud.setCheckable(True)
        self.btn_hud.setChecked(self.show_hud)
        self.btn_hud.setStyleSheet(self._button_style())
        self.btn_hud.clicked.connect(self._toggle_hud)
        bar_layout.addWidget(self.btn_hud)

        bar_layout.addStretch(1)
        layout_main.addWidget(bar_frame)
        layout_main.addStretch(1)

    def _button_style(self) -> str:
        return (
            "QPushButton { background-color: rgba(30, 41, 59, 180); color: #cbd5e1; font-size: 10px; "
            "font-weight: 600; padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(71, 85, 105, 180); }"
            "QPushButton:hover { background-color: rgba(51, 65, 85, 220); color: #ffffff; }"
            "QPushButton:checked { background-color: rgba(37, 99, 235, 220); color: #ffffff; border-color: #3b82f6; }"
        )

    def _combo_style(self) -> str:
        return (
            "QComboBox { background-color: rgba(30, 41, 59, 210); color: #cbd5e1; font-size: 10px; "
            "font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(71, 85, 105, 180); min-width: 130px; }"
            "QComboBox:hover { background-color: rgba(51, 65, 85, 230); color: #ffffff; }"
            "QComboBox::drop-down { border: none; width: 14px; }"
            "QComboBox QAbstractItemView { background-color: #0f172a; color: #cbd5e1; selection-background-color: #2563eb; selection-color: #ffffff; }"
        )

    def set_shot_cameras(self, cameras: Dict[str, ShotCamera]) -> None:
        """Register imported animated shot cameras and populate the viewport camera selector."""
        self.shot_cameras = dict(cameras)
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()
        self.combo_camera.addItem("🎥 Free Inspection", "free")

        unique_cams: Dict[str, ShotCamera] = {}
        for key, cam in cameras.items():
            if cam.name not in unique_cams:
                unique_cams[cam.name] = cam

        for cam_name in sorted(unique_cams.keys()):
            self.combo_camera.addItem(f"🎬 {cam_name}", cam_name)

        self.combo_camera.blockSignals(False)

        # Restore active shot camera or default to free inspection
        if self.active_shot_camera and self.active_shot_camera.name in unique_cams:
            idx = self.combo_camera.findData(self.active_shot_camera.name)
            if idx >= 0:
                self.combo_camera.setCurrentIndex(idx)
        else:
            self.combo_camera.setCurrentIndex(0)
            self.camera_mode = CameraMode.FREE_INSPECTION
            self.active_shot_camera = None
            self.btn_cam_lock.setEnabled(False)
            self.btn_follow_char.setEnabled(True)
        self.update()

    def _on_camera_selection_changed(self, index: int) -> None:
        """Switch between Free Inspection and animated Shot Camera."""
        data = self.combo_camera.currentData()
        if data == "free" or data is None or index <= 0:
            self.camera_mode = CameraMode.FREE_INSPECTION
            self.active_shot_camera = None
            self.btn_cam_lock.setEnabled(False)
            self.btn_follow_char.setEnabled(True)
            # Restore free camera parameters without modifying imported camera
            self.yaw = self.free_camera_state.yaw
            self.pitch = self.free_camera_state.pitch
            self.distance = self.free_camera_state.distance
            self.target = self.free_camera_state.target.copy()
        else:
            cam_name = str(data)
            for key, cam in self.shot_cameras.items():
                if cam.name == cam_name:
                    self.active_shot_camera = cam
                    break
            self.camera_mode = CameraMode.SHOT_CAMERA
            self.is_navigation_locked = True
            self.btn_cam_lock.setChecked(True)
            self.btn_cam_lock.setEnabled(True)
            self.btn_cam_lock.setText("🔒 Locked")
            self.btn_follow_char.setEnabled(False)
        self.update()

    def _toggle_camera_lock(self, checked: bool) -> None:
        self.is_navigation_locked = checked
        self.btn_cam_lock.setText("🔒 Locked" if checked else "🔓 Unlocked")
        self.update()

    def _toggle_follow_character(self, checked: bool) -> None:
        self.free_camera_state.follow_character = checked
        if checked:
            target_mesh = self.cloth_geometry
            if target_mesh and len(target_mesh.vertices) > 0:
                self.free_camera_state.last_character_pos = np.mean(target_mesh.vertices, axis=0)
        self.update()

    def update_character_follow(self, mesh: Optional[MeshGeometry] = None) -> None:
        """Follow character movement in Free Inspection mode while preserving viewing offset and zoom."""
        if not self.free_camera_state.follow_character or self.camera_mode != CameraMode.FREE_INSPECTION:
            return
        target_mesh = mesh or self.cloth_geometry
        if target_mesh and len(target_mesh.vertices) > 0:
            center = np.mean(target_mesh.vertices, axis=0)
            self.free_camera_state.update_character_tracking(center)
            self.target = self.free_camera_state.target.copy()
            self.update()

    def _toggle_wireframe(self, checked: bool):
        self.show_wireframe = checked
        self.update()

    def _toggle_colliders(self, checked: bool):
        self.show_colliders = checked
        self.update()

    def _toggle_heatmap(self, checked: bool):
        self.show_quality_heatmap = checked
        self.update()

    def _toggle_hud(self, checked: bool):
        self.show_hud = checked
        self.update()

    # Camera Framing Math
    def frame_all(self):
        """Fit all visible geometries comfortably in the viewport."""
        all_verts = []
        if self.cloth_geometry and len(self.cloth_geometry.vertices) > 0:
            all_verts.append(self.cloth_geometry.vertices)
        if self.show_colliders:
            for c in self.collider_geometries:
                if len(c.vertices) > 0:
                    all_verts.append(c.vertices)

        if not all_verts:
            self.free_camera_state.target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.free_camera_state.distance = 1.8
        else:
            pts = np.vstack(all_verts)
            box_min = np.min(pts, axis=0)
            box_max = np.max(pts, axis=0)
            self.free_camera_state.target = (box_min + box_max) * 0.5
            extent = box_max - box_min
            max_dim = float(np.max(extent))
            # Distance that fits the character in 70-80% of screen
            self.free_camera_state.distance = max(0.5, max_dim * 1.55)

        if self.camera_mode == CameraMode.SHOT_CAMERA:
            self.combo_camera.setCurrentIndex(0)
        else:
            self.target = self.free_camera_state.target.copy()
            self.distance = self.free_camera_state.distance
            self.update()

    def frame_selected(self, mesh: Optional[MeshGeometry] = None):
        """Fit a specific selected mesh in the viewport."""
        target_mesh = mesh or self.cloth_geometry
        if target_mesh and len(target_mesh.vertices) > 0:
            box_min = np.min(target_mesh.vertices, axis=0)
            box_max = np.max(target_mesh.vertices, axis=0)
            self.free_camera_state.target = (box_min + box_max) * 0.5
            extent = box_max - box_min
            self.free_camera_state.distance = max(0.4, float(np.max(extent)) * 1.6)

            if self.camera_mode == CameraMode.SHOT_CAMERA:
                self.combo_camera.setCurrentIndex(0)
            else:
                self.target = self.free_camera_state.target.copy()
                self.distance = self.free_camera_state.distance
                self.update()

    def set_before_after_comparison(self, pre_geom: Optional[MeshGeometry], show_before: bool):
        """Toggle between pre-correction and post-correction geometry."""
        self.pre_correction_geometry = pre_geom
        self.show_comparison_before = show_before
        self.update()

    def set_quality_overlay(self, enabled: bool, penetrations_mm: Optional[np.ndarray] = None):
        """Enable or disable defect heatmap visualization."""
        self.show_quality_heatmap = enabled
        self.vertex_penetrations_mm = penetrations_mm
        self.btn_heatmap.setChecked(enabled)
        self.update()

    def set_annotation_mode(self, enabled: bool) -> None:
        """Toggle between camera navigation and 2D annotation drawing."""
        self.annotation_mode = enabled
        self.update()

    def set_annotation_tool(self, tool: str) -> None:
        """Select drawing tool: 'brush', 'outline', 'arrow', 'text'."""
        self.current_tool = tool
        self.update()

    def set_surface_selection_preview(self, selection: Any) -> None:
        """Preview selected 3D surface vertices and falloff in viewport."""
        self.surface_selection_preview = selection
        self.update()

    def clear_annotations(self) -> None:
        """Clear all annotation strokes on current frame."""
        self.undo_stack.append(list(self.annotation_strokes))
        self.annotation_strokes.clear()
        self.active_stroke.clear()
        self.update()

    def undo_annotation(self) -> None:
        """Undo last annotation stroke."""
        if self.annotation_strokes:
            self.undo_stack.append(list(self.annotation_strokes))
            self.annotation_strokes.pop()
            self.update()
        elif self.undo_stack:
            self.annotation_strokes = self.undo_stack.pop()
            self.update()

    def set_cloth_geometry(self, mesh: Optional[MeshGeometry], auto_frame: bool = False) -> None:
        """Update cloth geometry to display."""
        self.cloth_geometry = mesh
        if auto_frame:
            self.frame_all()
        self.update()

    def set_colliders(self, colliders: List[MeshGeometry]) -> None:
        """Update collider geometries to display."""
        self.collider_geometries = colliders
        self.update()

    def set_contact_points(self, points: Optional[np.ndarray]) -> None:
        """Highlight active collision/contact points."""
        self.contact_points = points
        self.update()

    # Camera transformation math
    def _get_camera_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute camera eye position and rotation basis."""
        rad_yaw = np.radians(self.yaw)
        rad_pitch = np.radians(self.pitch)

        eye_x = self.target[0] + self.distance * np.cos(rad_pitch) * np.sin(rad_yaw)
        eye_y = self.target[1] + self.distance * np.sin(rad_pitch)
        eye_z = self.target[2] + self.distance * np.cos(rad_pitch) * np.cos(rad_yaw)
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)

        forward = self.target - eye
        forward /= (np.linalg.norm(forward) + 1e-7)

        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_world)
        right /= (np.linalg.norm(right) + 1e-7)

        up = np.cross(right, forward)
        up /= (np.linalg.norm(up) + 1e-7)

        return eye, right, up

    def project_3d_to_2d(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D world coordinates to screen (X, Y) pixels and depth Z."""
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        if self.camera_mode == CameraMode.SHOT_CAMERA and self.active_shot_camera is not None:
            screen_coords, z_view, gate = self.active_shot_camera.project_world_to_screen(
                points=points,
                frame=float(self.current_frame),
                viewport_width=self.width(),
                viewport_height=self.height(),
            )
            self.active_gate_rect = gate
            return screen_coords, z_view

        # Free inspection camera projection
        eye, right, up = self._get_camera_matrix()
        forward = np.cross(up, right)

        # View space translation
        rel = points - eye

        # Dot product with view axes
        x_view = np.dot(rel, right)
        y_view = np.dot(rel, up)
        z_view = np.dot(rel, forward)

        # Screen perspective projection
        fov = 1.8
        w, h = max(1, self.width()), max(1, self.height())
        aspect = w / float(h)

        z_view_clamped = np.maximum(z_view, 0.1)

        x_screen = (x_view / (z_view_clamped * fov * aspect)) * (w * 0.5) + (w * 0.5)
        y_screen = -(y_view / (z_view_clamped * fov)) * (h * 0.5) + (h * 0.5)

        screen_coords = np.column_stack([x_screen, y_screen])
        self.active_gate_rect = (0, 0, w, h)
        return screen_coords, z_view

    # Mouse navigation & annotation events
    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()
        if self.annotation_mode:
            if event.button() == Qt.LeftButton:
                pt = (float(event.position().x()), float(event.position().y()))
                self.active_stroke = [pt]
                self.update()
            return

        # Lock navigation when viewing animated shot camera
        if self.camera_mode == CameraMode.SHOT_CAMERA and self.is_navigation_locked:
            return

        if event.button() == Qt.LeftButton:
            self.is_orbiting = True
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.is_panning = True

    def mouseReleaseEvent(self, event):
        if self.annotation_mode:
            if self.active_stroke:
                stroke_entry = {
                    "tool": self.current_tool,
                    "points": list(self.active_stroke),
                    "color": self.active_color,
                    "label": self.current_region_label,
                    "radius": self.brush_radius,
                }
                self.annotation_strokes.append(stroke_entry)
                self.active_stroke.clear()
                self.update()
            return

        self.is_orbiting = False
        self.is_panning = False
        self.last_mouse_pos = None

    def mouseMoveEvent(self, event):
        if self.annotation_mode:
            if event.buttons() & Qt.LeftButton:
                pt = (float(event.position().x()), float(event.position().y()))
                self.active_stroke.append(pt)
                self.update()
            return

        if self.camera_mode == CameraMode.SHOT_CAMERA and self.is_navigation_locked:
            return

        if self.last_mouse_pos is None:
            return

        delta = event.position() - self.last_mouse_pos
        self.last_mouse_pos = event.position()

        if self.is_orbiting:
            self.yaw = (self.yaw + delta.x() * 0.5) % 360.0
            self.pitch = np.clip(self.pitch + delta.y() * 0.5, -85.0, 85.0)
            self.free_camera_state.yaw = self.yaw
            self.free_camera_state.pitch = self.pitch
            self.update()
        elif self.is_panning:
            eye, right, up = self._get_camera_matrix()
            pan_speed = self.distance * 0.002
            self.target += (-right * delta.x() + up * delta.y()) * pan_speed
            self.free_camera_state.target = self.target.copy()
            self.update()

    def wheelEvent(self, event):
        if self.annotation_mode:
            delta = event.angleDelta().y()
            if delta > 0:
                self.brush_radius = min(50.0, self.brush_radius + 2.0)
            else:
                self.brush_radius = max(4.0, self.brush_radius - 2.0)
            self.update()
            return

        if self.camera_mode == CameraMode.SHOT_CAMERA and self.is_navigation_locked:
            return

        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.distance = np.clip(self.distance * zoom_factor, 0.2, 25.0)
        self.free_camera_state.distance = self.distance
        self.update()

    # --- OpenGL Lifecycle & GPU-Accelerated Rendering ---
    def initializeGL(self):
        """Set up OpenGL context state."""
        if not PYOPENGL_AVAILABLE:
            return
        try:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glClearColor(0.06, 0.07, 0.09, 1.0)
            self._gl_initialized = True
        except Exception:
            self._gl_initialized = False

    def resizeGL(self, w: int, h: int):
        """Handle OpenGL viewport resizing."""
        if not PYOPENGL_AVAILABLE:
            return
        try:
            gl.glViewport(0, 0, max(1, w), max(1, h))
        except Exception:
            pass

    def _draw_grid_gl(self):
        """Draw 3D ground plane grid via OpenGL lines."""
        if not PYOPENGL_AVAILABLE:
            return
        extent = 1.6
        steps = 13
        gl.glColor4f(0.18, 0.24, 0.33, 0.55)
        gl.glBegin(gl.GL_LINES)
        for val in np.linspace(-extent, extent, steps):
            gl.glVertex3f(-extent, 0.0, val)
            gl.glVertex3f(extent, 0.0, val)
            gl.glVertex3f(val, 0.0, -extent)
            gl.glVertex3f(val, 0.0, extent)
        gl.glEnd()

    def _draw_mesh_gl(
        self,
        mesh: MeshGeometry,
        face_color: QColor,
        wire_color: Optional[QColor],
    ):
        """Fast GPU rendering of shaded mesh and wireframe using vertex arrays."""
        if not PYOPENGL_AVAILABLE or mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return

        verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32).flatten()

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, verts)

        # 1. Shaded surface with polygon offset so wireframe sits cleanly on top
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(1.0, 1.0)
        gl.glColor4f(face_color.redF(), face_color.greenF(), face_color.blueF(), face_color.alphaF())
        gl.glDrawElements(gl.GL_TRIANGLES, len(faces), gl.GL_UNSIGNED_INT, faces)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

        # 2. Wireframe lines
        if wire_color is not None:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glColor4f(wire_color.redF(), wire_color.greenF(), wire_color.blueF(), wire_color.alphaF())
            gl.glDrawElements(gl.GL_TRIANGLES, len(faces), gl.GL_UNSIGNED_INT, faces)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def _draw_mesh_heatmap_gl(self, mesh: MeshGeometry):
        """GPU rendering of defect penetration heatmap with per-vertex color buffer."""
        if not PYOPENGL_AVAILABLE or mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return

        verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32).flatten()
        pens = self.vertex_penetrations_mm if self.vertex_penetrations_mm is not None else np.zeros(len(mesh.vertices))

        # Compute per-vertex colors: Green (0mm) -> Red (>=5mm)
        t = np.clip(pens / 5.0, 0.0, 1.0)
        colors = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
        colors[:, 0] = (34.0 * (1.0 - t) + 239.0 * t) / 255.0
        colors[:, 1] = (197.0 * (1.0 - t) + 68.0 * t) / 255.0
        colors[:, 2] = (94.0 * (1.0 - t) + 68.0 * t) / 255.0
        colors[:, 3] = 0.82

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, verts)
        gl.glColorPointer(4, gl.GL_FLOAT, 0, colors)

        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(1.0, 1.0)
        gl.glDrawElements(gl.GL_TRIANGLES, len(faces), gl.GL_UNSIGNED_INT, faces)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

        if self.show_wireframe:
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glColor4f(1.0, 0.9, 0.3, 0.7)
            gl.glDrawElements(gl.GL_TRIANGLES, len(faces), gl.GL_UNSIGNED_INT, faces)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        else:
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def _draw_contacts_gl(self):
        """Draw contact/collision points as glowing points."""
        if not PYOPENGL_AVAILABLE or self.contact_points is None or len(self.contact_points) == 0:
            return
        gl.glPointSize(6.0)
        gl.glColor4f(0.94, 0.27, 0.27, 0.9)
        gl.glBegin(gl.GL_POINTS)
        for pt in self.contact_points:
            gl.glVertex3f(pt[0], pt[1], pt[2])
        gl.glEnd()

    def _draw_framing_gate_overlay(self, painter: QPainter):
        """Draw letterbox/pillarbox masks and film gate border with camera metadata banner."""
        if self.active_shot_camera is None:
            return

        w_view = self.width()
        h_view = self.height()
        gx, gy, gw, gh = self.active_gate_rect
        sample = self.active_shot_camera.evaluate(float(self.current_frame))

        # Dark letterbox / pillarbox bars (rgba: 10, 12, 16, 235)
        mask_brush = QBrush(QColor(10, 12, 16, 235))
        painter.setPen(Qt.NoPen)

        # Top bar
        if gy > 0:
            painter.fillRect(0, 0, w_view, gy, mask_brush)
        # Bottom bar
        if gy + gh < h_view:
            painter.fillRect(0, gy + gh, w_view, h_view - (gy + gh), mask_brush)
        # Left bar
        if gx > 0:
            painter.fillRect(0, gy, gx, gh, mask_brush)
        # Right bar
        if gx + gw < w_view:
            painter.fillRect(gx + gw, gy, w_view - (gx + gw), gh, mask_brush)

        # Framing Gate outline
        gate_pen = QPen(QColor(245, 158, 11, 200), 1.5)  # Amber gold film gate outline
        painter.setPen(gate_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(gx, gy, gw, gh)

        # Camera metadata banner (top of viewport)
        banner_y = max(gy + 20, 24)
        banner_x = max(gx + 12, 16)

        lock_status = "🔒 LOCKED" if self.is_navigation_locked else "🔓 UNLOCKED"
        meta_text = (
            f"🎬 {self.active_shot_camera.name}  |  "
            f"f={sample.focal_length_mm:.1f}mm  |  "
            f"Film: {sample.horizontal_aperture_mm:.1f}x{sample.vertical_aperture_mm:.1f}mm ({sample.film_aspect_ratio:.2f}:1)  |  "
            f"Frame {int(round(self.current_frame))}  |  {lock_status}"
        )

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        font_metrics = painter.fontMetrics()
        text_w = font_metrics.horizontalAdvance(meta_text) + 20
        text_h = font_metrics.height() + 8

        pill_rect = QPolygonF([
            QPointF(banner_x - 6, banner_y - 14),
            QPointF(banner_x + text_w, banner_y - 14),
            QPointF(banner_x + text_w, banner_y + text_h - 14),
            QPointF(banner_x - 6, banner_y + text_h - 14),
        ])
        painter.setPen(QPen(QColor(71, 85, 105, 180), 1))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawPolygon(pill_rect)

        painter.setPen(QPen(QColor(241, 245, 249), 1))
        painter.drawText(banner_x + 4, banner_y + 2, meta_text)

        # Display warning pill if camera has unsupported features
        if self.active_shot_camera.unsupported_features:
            warn_text = f"⚠️ Unsupported: {', '.join(self.active_shot_camera.unsupported_features)}"
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QPen(QColor(251, 146, 60), 1))
            painter.drawText(banner_x + 4, banner_y + 20, warn_text)

    def paintGL(self):
        """Hardware-accelerated 3D OpenGL render pass with 2D QPainter overlay."""
        if not PYOPENGL_AVAILABLE or not self._gl_initialized:
            return

        try:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            w, h = max(1, self.width()), max(1, self.height())

            is_shot_cam = (self.camera_mode == CameraMode.SHOT_CAMERA and self.active_shot_camera is not None)

            if is_shot_cam:
                sample = self.active_shot_camera.evaluate(float(self.current_frame))
                gate_x, gate_y, gate_w, gate_h = self.active_shot_camera.compute_film_gate(w, h, sample)
                self.active_gate_rect = (gate_x, gate_y, gate_w, gate_h)

                # Restrict 3D rasterization to film gate rect (OpenGL bottom-left origin)
                gl_y = h - (gate_y + gate_h)
                gl.glViewport(gate_x, gl_y, max(1, gate_w), max(1, gate_h))

                # Perspective projection matching film aperture and focal length
                proj_mat = self.active_shot_camera.compute_projection_matrix(sample)
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                gl.glLoadMatrixd(np.ascontiguousarray(proj_mat.T, dtype=np.float64))

                # View transform
                view_mat = self.active_shot_camera.compute_view_matrix(sample)
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()
                gl.glLoadMatrixd(np.ascontiguousarray(view_mat.T, dtype=np.float64))
            else:
                self.active_gate_rect = (0, 0, w, h)
                gl.glViewport(0, 0, w, h)
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                aspect = float(w) / float(h)
                gluPerspective(45.0, aspect, 0.05, 100.0)

                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()
                eye, right, up = self._get_camera_matrix()
                gluLookAt(
                    float(eye[0]), float(eye[1]), float(eye[2]),
                    float(self.target[0]), float(self.target[1]), float(self.target[2]),
                    0.0, 1.0, 0.0,
                )

            # 1. 3D Ground Grid
            if self.show_grid:
                self._draw_grid_gl()

            # 2. Colliders
            if self.show_colliders:
                col_face = QColor(51, 65, 85, 120)
                col_wire = QColor(71, 85, 105, 160) if self.show_wireframe else None
                for col_mesh in self.collider_geometries:
                    self._draw_mesh_gl(col_mesh, col_face, col_wire)

            # 3. Active Cloth Mesh (Before vs After)
            active_mesh = self.cloth_geometry
            mesh_color = QColor(245, 158, 11, 210)
            wire_color = QColor(251, 191, 36, 170) if self.show_wireframe else None

            if self.show_comparison_before and self.pre_correction_geometry:
                active_mesh = self.pre_correction_geometry
                mesh_color = QColor(239, 68, 68, 200)
                wire_color = QColor(248, 113, 113, 160) if self.show_wireframe else None

            if active_mesh and len(active_mesh.vertices) > 0:
                if self.show_quality_heatmap and self.vertex_penetrations_mm is not None:
                    self._draw_mesh_heatmap_gl(active_mesh)
                else:
                    self._draw_mesh_gl(active_mesh, mesh_color, wire_color)

            # 4. Contact Highlights
            if self.contact_points is not None and len(self.contact_points) > 0:
                self._draw_contacts_gl()

            # Restore full viewport for 2D overlays
            gl.glViewport(0, 0, w, h)

            # 5. 2D QPainter Overlays (Framing Gate, Surface preview, Annotations, HUD)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            if is_shot_cam:
                self._draw_framing_gate_overlay(painter)

            if self.surface_selection_preview and active_mesh:
                self._draw_surface_selection_preview(painter)

            self._draw_annotation_overlay(painter)

            if self.show_hud:
                self._draw_hud(painter)

            painter.end()
        except Exception:
            pass

    # --- Paint Event (Hybrid GPU / CPU Dispatcher) ---
    def paintEvent(self, event):
        """Dispatch to hardware-accelerated OpenGL when valid, or fallback to software QPainter."""
        if (
            OPENGL_WIDGET_AVAILABLE
            and PYOPENGL_AVAILABLE
            and hasattr(self, "isValid")
            and self.isValid()
            and self._gl_initialized
        ):
            super().paintEvent(event)
        else:
            self._paint_software(event)

    def _paint_software(self, event):
        """Software-rasterized QPainter fallback when OpenGL context is inactive."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background clean studio dark gradient
        painter.fillRect(self.rect(), QColor(15, 17, 23))

        is_shot_cam = (self.camera_mode == CameraMode.SHOT_CAMERA and self.active_shot_camera is not None)

        # Draw 3D Ground Grid
        if self.show_grid:
            self._draw_grid(painter)

        # Draw Colliders
        if self.show_colliders:
            for col_mesh in self.collider_geometries:
                self._draw_mesh(
                    painter,
                    col_mesh,
                    face_color=QColor(51, 65, 85, 120),
                    wire_color=QColor(71, 85, 105, 160),
                )

        # Determine Cloth Mesh (Before vs After)
        active_mesh = self.cloth_geometry
        mesh_color = QColor(245, 158, 11, 210)  # Amber gold
        wire_color = QColor(251, 191, 36, 170) if self.show_wireframe else None

        if self.show_comparison_before and self.pre_correction_geometry:
            active_mesh = self.pre_correction_geometry
            mesh_color = QColor(239, 68, 68, 200)  # Red for pre-correction
            wire_color = QColor(248, 113, 113, 160) if self.show_wireframe else None

        # Draw Cloth Geometry
        if active_mesh and len(active_mesh.vertices) > 0:
            if self.show_quality_heatmap and self.vertex_penetrations_mm is not None:
                self._draw_mesh_heatmap(painter, active_mesh)
            else:
                self._draw_mesh(
                    painter,
                    active_mesh,
                    face_color=mesh_color,
                    wire_color=wire_color,
                )

        # Draw Contact Highlights
        if self.contact_points is not None and len(self.contact_points) > 0:
            pts2d, z_d = self.project_3d_to_2d(self.contact_points)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(239, 68, 68, 220)))
            for pt in pts2d:
                painter.drawEllipse(QPointF(pt[0], pt[1]), 3.5, 3.5)

        # Film Framing Gate Overlay (Letterboxing/Pillarboxing)
        if is_shot_cam:
            self._draw_framing_gate_overlay(painter)

        # Draw 3D Surface Selection Preview with Falloff
        if self.surface_selection_preview and active_mesh:
            self._draw_surface_selection_preview(painter)

        # Draw 2D Annotation Strokes
        self._draw_annotation_overlay(painter)

        # Optional Viewport HUD Overlay
        if self.show_hud:
            self._draw_hud(painter)

        painter.end()

    def _draw_surface_selection_preview(self, painter: QPainter):
        """Highlight active 3D surface vertices with glowing manifold falloff colors."""
        sel = self.surface_selection_preview
        if not hasattr(sel, "vertex_weights") or not sel.vertex_weights:
            return

        verts = self.cloth_geometry.vertices if self.cloth_geometry else np.zeros((0, 3))
        pts2d, z_vals = self.project_3d_to_2d(verts)

        for v_idx, w in sel.vertex_weights.items():
            if v_idx >= len(pts2d) or z_vals[v_idx] < 0.1:
                continue
            p = pts2d[v_idx]
            alpha = int(np.clip(w * 240, 40, 255))
            radius = 3.0 + w * 4.0

            painter.setPen(QPen(QColor(254, 240, 138, alpha), 1))
            painter.setBrush(QBrush(QColor(249, 115, 22, alpha)))
            painter.drawEllipse(QPointF(p[0], p[1]), radius, radius)

    def _draw_mesh_heatmap(self, painter: QPainter, mesh: MeshGeometry):
        """Colorize mesh faces according to defect penetration depths."""
        pts2d, z_vals = self.project_3d_to_2d(mesh.vertices)
        pens = self.vertex_penetrations_mm if self.vertex_penetrations_mm is not None else np.zeros(len(mesh.vertices))

        face_centers_z = np.mean(z_vals[mesh.faces], axis=1)
        sort_indices = np.argsort(-face_centers_z)

        for f_idx in sort_indices:
            if face_centers_z[f_idx] < 0.1:
                continue
            face = mesh.faces[f_idx]
            avg_pen = np.mean(pens[face])

            # Interpolate color from green (0mm) to red (>5mm)
            t = np.clip(avg_pen / 5.0, 0.0, 1.0)
            col = QColor(
                int(34 * (1 - t) + 239 * t),
                int(197 * (1 - t) + 68 * t),
                int(94 * (1 - t) + 68 * t),
                200
            )

            p0 = pts2d[face[0]]
            p1 = pts2d[face[1]]
            p2 = pts2d[face[2]]

            poly = QPolygonF([QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1])])
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(col.lighter(120), 1) if self.show_wireframe else Qt.NoPen)
            painter.drawPolygon(poly)

    def _draw_annotation_overlay(self, painter: QPainter):
        """Draw 2D annotation layers, brush paths, arrows, and outlines."""
        for stroke in self.annotation_strokes:
            tool = stroke["tool"]
            pts = stroke["points"]
            color = stroke.get("color") or self.active_color
            radius = stroke.get("radius", 14.0)

            if tool == "brush" and pts:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                for pt in pts:
                    painter.drawEllipse(QPointF(pt[0], pt[1]), radius, radius)

            elif tool == "outline" and len(pts) > 1:
                pen = QPen(color, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
                poly = QPolygonF([QPointF(x, y) for x, y in pts])
                painter.drawPolygon(poly)

            elif tool == "arrow" and len(pts) >= 2:
                pen = QPen(color, 3)
                painter.setPen(pen)
                p_start = QPointF(pts[0][0], pts[0][1])
                p_end = QPointF(pts[-1][0], pts[-1][1])
                painter.drawLine(p_start, p_end)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(p_end, 5.0, 5.0)

            elif tool == "text" and pts:
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
                painter.drawText(int(pts[0][0]) + 5, int(pts[0][1]) - 5, stroke.get("label", "Note"))

        # Draw active stroke currently being drawn
        if self.active_stroke:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(self.active_color))
            for pt in self.active_stroke:
                painter.drawEllipse(QPointF(pt[0], pt[1]), self.brush_radius, self.brush_radius)

    def get_camera_parameters(self) -> Any:
        """Construct a versioned CameraParameters dataclass corresponding to active camera state."""
        if self.camera_mode == CameraMode.SHOT_CAMERA and self.active_shot_camera is not None:
            return self.active_shot_camera.to_camera_parameters(
                frame=float(self.current_frame),
                width=self.width(),
                height=self.height(),
            )
        return self.free_camera_state.to_camera_parameters(
            width=self.width(),
            height=self.height(),
        )

    def capture_annotation_bundle(
        self,
        bundle_dir: str,
        shot_name: str = "shot010",
        garment_name: str = "jacket",
        version: str = "v001",
        note_text: str = ""
    ) -> Any:
        import os
        from PySide6.QtGui import QImage
        from autonomous_cfx.core.annotation_bundle import AnnotationBundle, AnnotationRegion
        from autonomous_cfx.core.surface_mapping import ImageToSurfaceMapper

        os.makedirs(bundle_dir, exist_ok=True)
        w, h = max(100, self.width()), max(100, self.height())
        cam_params = self.get_camera_parameters()

        # 1. Clean Render
        clean_img = QImage(w, h, QImage.Format_ARGB32)
        clean_img.fill(QColor(15, 17, 23))
        p_clean = QPainter(clean_img)
        p_clean.setRenderHint(QPainter.Antialiasing)
        if self.cloth_geometry:
            self._draw_mesh(p_clean, self.cloth_geometry, QColor(245, 158, 11, 210), QColor(251, 191, 36, 170))
        p_clean.end()
        clean_path = os.path.join(bundle_dir, "clean_render.png")
        clean_img.save(clean_path)

        # 2. Annotation Overlay Image
        overlay_img = QImage(w, h, QImage.Format_ARGB32)
        overlay_img.fill(QColor(0, 0, 0, 0))
        p_over = QPainter(overlay_img)
        p_over.setRenderHint(QPainter.Antialiasing)
        self._draw_annotation_overlay(p_over)
        p_over.end()
        overlay_path = os.path.join(bundle_dir, "annotation_overlay.png")
        overlay_img.save(overlay_path)

        # 3. Rasterize Depth Buffer & Face ID Buffer
        depth_path = os.path.join(bundle_dir, "depth_buffer.npy")
        face_id_path = os.path.join(bundle_dir, "face_id_buffer.npy")
        if self.cloth_geometry:
            depth_buf, face_id_buf = ImageToSurfaceMapper.rasterize_viewport_buffers(self.cloth_geometry, cam_params)
            np.save(depth_path, depth_buf)
            np.save(face_id_path, face_id_buf)

        # 4. Create Annotation Regions
        regions: List[AnnotationRegion] = []
        for i, s in enumerate(self.annotation_strokes):
            reg = AnnotationRegion(
                region_id=f"reg_{i+1:02d}",
                label=s.get("label", "Defect Region"),
                intent_description=note_text if i == 0 and note_text else s.get("label", ""),
                drawing_tool=s.get("tool", "brush"),
                stroke_points=s.get("points", []),
                is_protected="protect" in s.get("label", "").lower() or "cuff" in s.get("label", "").lower()
            )
            regions.append(reg)

        bundle = AnnotationBundle(
            bundle_id=f"bundle_{int(time.time())}",
            shot_name=shot_name,
            garment_name=garment_name,
            geometry_version=version,
            frame=self.current_frame,
            time_seconds=float(self.current_frame - 1) / 24.0,
            camera=cam_params,
            regions=regions,
            clean_image_path=clean_path,
            annotation_overlay_path=overlay_path,
            depth_buffer_path=depth_path,
            face_id_buffer_path=face_id_path
        )
        bundle.save(bundle_dir)
        return bundle

    def _draw_grid(self, painter: QPainter):
        grid_lines = []
        extent = 1.6
        steps = 13
        for val in np.linspace(-extent, extent, steps):
            grid_lines.append(np.array([[-extent, 0.0, val], [extent, 0.0, val]], dtype=np.float32))
            grid_lines.append(np.array([[val, 0.0, -extent], [val, 0.0, extent]], dtype=np.float32))

        pen = QPen(QColor(30, 41, 59, 140), 1)
        painter.setPen(pen)

        for line3d in grid_lines:
            pts2d, z_d = self.project_3d_to_2d(line3d)
            if z_d[0] > 0.1 and z_d[1] > 0.1:
                painter.drawLine(
                    QPointF(pts2d[0, 0], pts2d[0, 1]),
                    QPointF(pts2d[1, 0], pts2d[1, 1]),
                )

    def _draw_mesh(
        self,
        painter: QPainter,
        mesh: MeshGeometry,
        face_color: QColor,
        wire_color: Optional[QColor],
    ):
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return

        pts2d, z_vals = self.project_3d_to_2d(mesh.vertices)

        brush = QBrush(face_color)
        pen = QPen(wire_color, 1) if wire_color else QPen(Qt.NoPen)

        painter.setBrush(brush)
        painter.setPen(pen)

        face_centers_z = np.mean(z_vals[mesh.faces], axis=1)
        sort_indices = np.argsort(-face_centers_z)

        for f_idx in sort_indices:
            if face_centers_z[f_idx] < 0.1:
                continue
            face = mesh.faces[f_idx]
            p0 = pts2d[face[0]]
            p1 = pts2d[face[1]]
            p2 = pts2d[face[2]]

            poly = QPolygonF([QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1])])
            painter.drawPolygon(poly)

    def _draw_hud(self, painter: QPainter):
        painter.setPen(QPen(QColor(148, 163, 184), 1))
        painter.setFont(QFont("Segoe UI", 9))
        hud_text = (
            f"Frame: {self.current_frame}  |  "
            f"Cloth Verts: {self.cloth_geometry.num_vertices if self.cloth_geometry else 0}  |  "
            f"Colliders: {len(self.collider_geometries)}  |  "
            f"Distance: {self.distance:.2f}m"
        )
        painter.drawText(15, self.height() - 15, hud_text)
