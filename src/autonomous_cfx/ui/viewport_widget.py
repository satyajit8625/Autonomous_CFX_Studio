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
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autonomous_cfx.core.geometry import MeshGeometry


class Viewport3DWidget(QWidget):
    """Interactive 3D preview viewport with mouse orbit, pan, and zoom."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

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

        self._build_overlay_toolbar()

    def _build_overlay_toolbar(self):
        """Top overlay bar for quick viewport actions (Frame All, Wireframe, Heatmap)."""
        layout_main = QVBoxLayout(self)
        layout_main.setContentsMargins(10, 10, 10, 10)

        bar_frame = QWidget(self)
        bar_layout = QHBoxLayout(bar_frame)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(6)

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
            self.target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.distance = 1.8
            self.update()
            return

        pts = np.vstack(all_verts)
        box_min = np.min(pts, axis=0)
        box_max = np.max(pts, axis=0)
        self.target = (box_min + box_max) * 0.5
        extent = box_max - box_min
        max_dim = float(np.max(extent))
        # Distance that fits the character in 70-80% of screen
        self.distance = max(0.5, max_dim * 1.55)
        self.update()

    def frame_selected(self, mesh: Optional[MeshGeometry] = None):
        """Fit a specific selected mesh in the viewport."""
        target_mesh = mesh or self.cloth_geometry
        if target_mesh and len(target_mesh.vertices) > 0:
            box_min = np.min(target_mesh.vertices, axis=0)
            box_max = np.max(target_mesh.vertices, axis=0)
            self.target = (box_min + box_max) * 0.5
            extent = box_max - box_min
            self.distance = max(0.4, float(np.max(extent)) * 1.6)
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
            return np.zeros((0, 2)), np.zeros((0,))

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
        w, h = self.width(), self.height()
        aspect = w / float(h) if h > 0 else 1.0

        z_view_clamped = np.maximum(z_view, 0.1)

        x_screen = (x_view / (z_view_clamped * fov * aspect)) * (w * 0.5) + (w * 0.5)
        y_screen = -(y_view / (z_view_clamped * fov)) * (h * 0.5) + (h * 0.5)

        screen_coords = np.column_stack([x_screen, y_screen])
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

        if self.last_mouse_pos is None:
            return

        delta = event.position() - self.last_mouse_pos
        self.last_mouse_pos = event.position()

        if self.is_orbiting:
            self.yaw = (self.yaw + delta.x() * 0.5) % 360.0
            self.pitch = np.clip(self.pitch + delta.y() * 0.5, -85.0, 85.0)
            self.update()
        elif self.is_panning:
            eye, right, up = self._get_camera_matrix()
            pan_speed = self.distance * 0.002
            self.target += (-right * delta.x() + up * delta.y()) * pan_speed
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

        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.distance = np.clip(self.distance * zoom_factor, 0.2, 25.0)
        self.update()

    # Paint event
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background clean studio dark gradient
        painter.fillRect(self.rect(), QColor(15, 17, 23))

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

        # Draw 3D Surface Selection Preview with Falloff
        if self.surface_selection_preview and active_mesh:
            self._draw_surface_selection_preview(painter)

        # Draw 2D Annotation Strokes
        self._draw_annotation_overlay(painter)

        # Optional Viewport HUD Overlay
        if self.show_hud:
            self._draw_hud(painter)

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
        from autonomous_cfx.core.annotation_bundle import CameraParameters
        return CameraParameters(
            yaw=self.yaw,
            pitch=self.pitch,
            distance=self.distance,
            target=(float(self.target[0]), float(self.target[1]), float(self.target[2])),
            fov=1.8,
            width=self.width(),
            height=self.height(),
            view_matrix=[],
            projection_matrix=[]
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
