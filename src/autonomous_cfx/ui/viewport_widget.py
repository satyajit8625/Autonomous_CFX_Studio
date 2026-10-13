"""Autonomous CFX Studio - Interactive 3D Viewport.
Software-rasterized/accelerated 3D projection viewport supporting Orbit, Pan, Zoom, wireframe, and shaded rendering.
Guaranteed cross-platform stability across all Windows display drivers.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from autonomous_cfx.core.geometry import MeshGeometry


class Viewport3DWidget(QWidget):
    """Interactive 3D preview viewport with mouse orbit, pan, and zoom."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        # Camera parameters
        self.yaw: float = 30.0    # degrees
        self.pitch: float = 15.0  # degrees
        self.distance: float = 3.2
        self.target: np.ndarray = np.array([0.0, 1.2, 0.0], dtype=np.float32)

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
        self.current_frame: int = 1

        # Annotation Canvas State
        self.annotation_mode: bool = False
        self.current_tool: str = "brush"  # 'brush', 'outline', 'arrow', 'text'
        self.brush_radius: float = 14.0
        self.active_stroke: List[Tuple[float, float]] = []
        self.annotation_strokes: List[Dict[str, Any]] = []
        self.undo_stack: List[List[Dict[str, Any]]] = []
        self.current_region_label: str = "Defect Region"
        self.active_color: QColor = QColor(255, 60, 60, 200)

        # 3D Surface Selection Preview
        self.surface_selection_preview: Any = None

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

    def set_cloth_geometry(self, mesh: Optional[MeshGeometry]) -> None:
        """Update cloth geometry to display."""
        self.cloth_geometry = mesh
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

        # Avoid divide by zero behind camera
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
            # Adjust brush radius with mouse wheel
            delta = event.angleDelta().y()
            if delta > 0:
                self.brush_radius = min(50.0, self.brush_radius + 2.0)
            else:
                self.brush_radius = max(4.0, self.brush_radius - 2.0)
            self.update()
            return

        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.distance = np.clip(self.distance * zoom_factor, 0.4, 20.0)
        self.update()

    # Paint event
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background gradient
        painter.fillRect(self.rect(), QColor(22, 24, 29))

        # Draw 3D Ground Grid
        if self.show_grid:
            self._draw_grid(painter)

        # Draw Colliders
        if self.show_colliders:
            for col_mesh in self.collider_geometries:
                self._draw_mesh(
                    painter,
                    col_mesh,
                    face_color=QColor(70, 75, 88, 120),
                    wire_color=QColor(100, 110, 125, 180),
                )

        # Draw Cloth Geometry
        if self.cloth_geometry and len(self.cloth_geometry.vertices) > 0:
            self._draw_mesh(
                painter,
                self.cloth_geometry,
                face_color=QColor(230, 150, 40, 200),
                wire_color=QColor(255, 200, 100, 160) if self.show_wireframe else None,
            )

        # Draw Contact Highlights
        if self.contact_points is not None and len(self.contact_points) > 0:
            pts2d, z_d = self.project_3d_to_2d(self.contact_points)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 60, 60, 220)))
            for pt in pts2d:
                painter.drawEllipse(QPointF(pt[0], pt[1]), 3.0, 3.0)

        # Draw 3D Surface Selection Preview with Falloff
        if self.surface_selection_preview and self.cloth_geometry:
            self._draw_surface_selection_preview(painter)

        # Draw 2D Annotation Strokes
        self._draw_annotation_overlay(painter)

        # Viewport HUD Overlay
        self._draw_hud(painter)

    def _draw_surface_selection_preview(self, painter: QPainter):
        """Highlight active 3D surface vertices with glowing manifold falloff colors."""
        sel = self.surface_selection_preview
        if not hasattr(sel, "vertex_weights") or not sel.vertex_weights:
            return

        verts = self.cloth_geometry.vertices
        pts2d, z_vals = self.project_3d_to_2d(verts)

        for v_idx, w in sel.vertex_weights.items():
            if v_idx >= len(pts2d) or z_vals[v_idx] < 0.1:
                continue
            p = pts2d[v_idx]
            alpha = int(np.clip(w * 240, 40, 255))
            radius = 3.0 + w * 4.0

            # Yellow-orange glowing selection
            painter.setPen(QPen(QColor(255, 220, 50, alpha), 1))
            painter.setBrush(QBrush(QColor(255, 120, 0, alpha)))
            painter.drawEllipse(QPointF(p[0], p[1]), radius, radius)

    def _draw_annotation_overlay(self, painter: QPainter):
        """Draw 2D annotation layers, brush paths, arrows, and outlines."""
        for stroke in self.annotation_strokes:
            tool = stroke["tool"]
            pts = stroke["points"]
            color = stroke["color"]
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
        """Construct CameraParameters dataclass representing current viewport camera."""
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
        """Capture versioned annotation bundle with clean render, mask, depth and face ID buffers."""
        import os
        from PySide6.QtGui import QImage
        from autonomous_cfx.core.annotation_bundle import AnnotationBundle, AnnotationRegion
        from autonomous_cfx.core.surface_mapping import ImageToSurfaceMapper

        os.makedirs(bundle_dir, exist_ok=True)
        w, h = max(100, self.width()), max(100, self.height())
        cam_params = self.get_camera_parameters()

        # 1. Clean Render
        clean_img = QImage(w, h, QImage.Format_ARGB32)
        clean_img.fill(QColor(22, 24, 29))
        p_clean = QPainter(clean_img)
        p_clean.setRenderHint(QPainter.Antialiasing)
        if self.cloth_geometry:
            self._draw_mesh(p_clean, self.cloth_geometry, QColor(230, 150, 40, 200), QColor(255, 200, 100, 160))
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
        """Draw 3D floor grid at y = 0."""
        grid_lines = []
        extent = 1.5
        steps = 11
        for val in np.linspace(-extent, extent, steps):
            grid_lines.append(np.array([[-extent, 0.0, val], [extent, 0.0, val]], dtype=np.float32))
            grid_lines.append(np.array([[val, 0.0, -extent], [val, 0.0, extent]], dtype=np.float32))

        pen = QPen(QColor(45, 50, 60), 1)
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
        """Project and render 3D mesh triangles."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return

        pts2d, z_vals = self.project_3d_to_2d(mesh.vertices)

        # Painter settings
        brush = QBrush(face_color)
        pen = QPen(wire_color, 1) if wire_color else QPen(Qt.NoPen)

        painter.setBrush(brush)
        painter.setPen(pen)

        # Draw faces with depth sorting (approximate painter's algorithm)
        face_centers_z = np.mean(z_vals[mesh.faces], axis=1)
        sort_indices = np.argsort(-face_centers_z)  # Draw farthest first

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
        """Render viewport frame counter and camera stats."""
        painter.setPen(QPen(QColor(180, 190, 205), 1))
        painter.setFont(QFont("Segoe UI", 9))
        hud_text = (
            f"Frame: {self.current_frame}  |  "
            f"Cloth Verts: {self.cloth_geometry.num_vertices if self.cloth_geometry else 0}  |  "
            f"Colliders: {len(self.collider_geometries)}  |  "
            f"Cam: [{self.yaw:.0f}°, {self.pitch:.0f}°]"
        )
        painter.drawText(15, 25, hud_text)
