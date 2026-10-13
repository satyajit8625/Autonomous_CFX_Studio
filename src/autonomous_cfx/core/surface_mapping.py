"""Autonomous CFX Studio - Image-to-Geometry Surface Mapping.

Maps 2D viewport annotation masks to stable 3D surface vertices and faces using
depth-buffered screen unprojection, manifold geodesic falloff, and strict occlusion
filtering (preventing accidental backface / hidden surface selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from autonomous_cfx.core.annotation_bundle import CameraParameters
from autonomous_cfx.core.geometry import MeshGeometry


@dataclass
class SurfaceSelection:
    """Stable surface selection mapped from viewport annotations."""
    selected_faces: List[int]
    selected_vertices: List[int]
    vertex_weights: Dict[int, float]  # vertex_idx -> weight in [0.0, 1.0]
    falloff_radius: float
    is_ambiguous: bool = False
    confidence_score: float = 1.0
    garment_name: str = ""
    notes: str = ""


class ImageToSurfaceMapper:
    """Unprojects screen annotation masks to 3D surface geometry with occlusion checking."""

    @staticmethod
    def rasterize_viewport_buffers(
        mesh: MeshGeometry,
        camera: CameraParameters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Software rasterizer generating high-precision per-pixel depth and face-ID buffers.

        Guarantees that occluded back faces or hidden surfaces are not visible in the face-ID buffer.
        """
        w, h = camera.width, camera.height
        depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
        face_id_buffer = np.full((h, w), -1, dtype=np.int32)

        if mesh.num_vertices == 0 or mesh.num_faces == 0:
            return depth_buffer, face_id_buffer

        # Camera basis vectors
        rad_yaw = np.radians(camera.yaw)
        rad_pitch = np.radians(camera.pitch)
        eye_x = camera.target[0] + camera.distance * np.cos(rad_pitch) * np.sin(rad_yaw)
        eye_y = camera.target[1] + camera.distance * np.sin(rad_pitch)
        eye_z = camera.target[2] + camera.distance * np.cos(rad_pitch) * np.cos(rad_yaw)
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)

        target = np.array(camera.target, dtype=np.float32)
        forward = target - eye
        forward /= (np.linalg.norm(forward) + 1e-7)

        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_world)
        right /= (np.linalg.norm(right) + 1e-7)
        up = np.cross(right, forward)
        up /= (np.linalg.norm(up) + 1e-7)

        # Transform vertices to view space
        rel = mesh.vertices - eye
        x_view = np.dot(rel, right)
        y_view = np.dot(rel, up)
        z_view = np.dot(rel, forward)

        aspect = w / float(h) if h > 0 else 1.0
        fov = camera.fov

        # Perspective projection
        z_clamped = np.maximum(z_view, camera.near_clip)
        x_screen = (x_view / (z_clamped * fov * aspect)) * (w * 0.5) + (w * 0.5)
        y_screen = -(y_view / (z_clamped * fov)) * (h * 0.5) + (h * 0.5)
        screen_pts = np.column_stack([x_screen, y_screen])

        # Rasterize faces with backface culling
        faces = mesh.faces
        for f_idx, (v0, v1, v2) in enumerate(faces):
            p0 = screen_pts[v0]
            p1 = screen_pts[v1]
            p2 = screen_pts[v2]
            z0 = z_view[v0]
            z1 = z_view[v1]
            z2 = z_view[v2]

            # Skip if any vertex is behind camera
            if z0 < camera.near_clip or z1 < camera.near_clip or z2 < camera.near_clip:
                continue

            # 2D cross product for front-facing check (screen space winding)
            edge1 = p1 - p0
            edge2 = p2 - p0
            cross_2d = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            # Cull back-facing triangles
            if cross_2d <= 0.0:
                continue

            # Bounding box in pixels
            min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
            max_x = min(w - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
            min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
            max_y = min(h - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

            if min_x > max_x or min_y > max_y:
                continue

            # Vectorized pixel test within bounding box
            xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
            px = xs.flatten()
            py = ys.flatten()

            denom = cross_2d
            w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
            w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
            w2 = 1.0 - w0 - w1

            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            if not np.any(inside):
                continue

            inside_px = px[inside]
            inside_py = py[inside]
            z_interp = w0[inside] * z0 + w1[inside] * z1 + w2[inside] * z2

            current_depth = depth_buffer[inside_py, inside_px]
            closer = z_interp < current_depth

            if np.any(closer):
                closer_px = inside_px[closer]
                closer_py = inside_py[closer]
                depth_buffer[closer_py, closer_px] = z_interp[closer]
                face_id_buffer[closer_py, closer_px] = f_idx

        return depth_buffer, face_id_buffer

    @classmethod
    def map_mask_to_surface(
        cls,
        mask: np.ndarray,
        face_id_buffer: np.ndarray,
        mesh: MeshGeometry,
        falloff_radius: float = 0.06
    ) -> SurfaceSelection:
        """Map 2D binary mask onto 3D visible surface faces and compute manifold falloff."""
        if mask.shape != face_id_buffer.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match face ID buffer shape {face_id_buffer.shape}"
            )

        # Identify hit visible faces
        marked_pixels = mask > 0
        hit_face_ids = np.unique(face_id_buffer[marked_pixels])
        valid_faces = [int(f) for f in hit_face_ids if f >= 0 and f < mesh.num_faces]

        if not valid_faces:
            return SurfaceSelection(
                selected_faces=[],
                selected_vertices=[],
                vertex_weights={},
                falloff_radius=falloff_radius,
                notes="No visible surface faces intersected by annotation mask."
            )

        # Core selected vertices
        faces_array = mesh.faces[valid_faces]
        core_vertices = set(int(v) for v in np.unique(faces_array))

        # Compute geodesic/topological falloff weights on mesh graph
        weights = cls._compute_manifold_falloff(mesh, core_vertices, falloff_radius)

        selected_vertices = sorted(list(weights.keys()))

        return SurfaceSelection(
            selected_faces=valid_faces,
            selected_vertices=selected_vertices,
            vertex_weights=weights,
            falloff_radius=falloff_radius,
            confidence_score=1.0,
            is_ambiguous=False
        )

    @staticmethod
    def _compute_manifold_falloff(
        mesh: MeshGeometry,
        core_vertices: Set[int],
        falloff_radius: float
    ) -> Dict[int, float]:
        """Compute smooth spatial falloff weights on mesh topology from core selection."""
        # Build adjacency graph
        edges, _ = mesh.compute_edges()
        adj: Dict[int, List[int]] = {i: [] for i in range(mesh.num_vertices)}
        for v1, v2 in edges:
            adj[v1].append(v2)
            adj[v2].append(v1)

        weights: Dict[int, float] = {v: 1.0 for v in core_vertices}
        if falloff_radius <= 1e-5:
            return weights

        # Breadth-first search for Euclidean boundary distance
        core_positions = mesh.vertices[list(core_vertices)]
        from scipy.spatial import KDTree
        kdtree = KDTree(core_positions)

        visited: Set[int] = set(core_vertices)
        frontier: Set[int] = set()

        # Seed frontier with direct neighbors of core vertices
        for cv in core_vertices:
            for neighbor in adj[cv]:
                if neighbor not in visited:
                    frontier.add(neighbor)

        # Expand outwards up to falloff radius
        while frontier:
            next_frontier: Set[int] = set()
            for v_idx in frontier:
                visited.add(v_idx)
                pos = mesh.vertices[v_idx]
                dist, _ = kdtree.query(pos)

                if dist <= falloff_radius:
                    # Smoothstep falloff: 1 - 3*(d/R)^2 + 2*(d/R)^3
                    s = dist / falloff_radius
                    weight = 1.0 - 3.0 * (s ** 2) + 2.0 * (s ** 3)
                    weights[v_idx] = float(np.clip(weight, 0.0, 1.0))

                    for neighbor in adj[v_idx]:
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
            frontier = next_frontier

        return weights

    @classmethod
    def handle_external_screenshot(
        cls,
        mesh: MeshGeometry,
        approx_region_hint: str,
        confidence: float = 0.5
    ) -> SurfaceSelection:
        """Handle externally uploaded screenshots where camera matrices/depth are missing.

        Marks selection as ambiguous and requires explicit confirmation.
        """
        # Find matching vertex groups if hint matches (e.g. 'sleeve', 'cuff', 'collar')
        matched_verts: Set[int] = set()
        hint_lower = approx_region_hint.lower()
        for group_name, v_indices in mesh.vertex_groups.items():
            if group_name.lower() in hint_lower or hint_lower in group_name.lower():
                matched_verts.update(v_indices)

        if not matched_verts:
            # Fallback to rough bounding box sector
            matched_verts = set(range(min(20, mesh.num_vertices)))

        weights = {v: 1.0 for v in matched_verts}
        return SurfaceSelection(
            selected_faces=[],
            selected_vertices=sorted(list(matched_verts)),
            vertex_weights=weights,
            falloff_radius=0.05,
            is_ambiguous=True,
            confidence_score=confidence,
            notes=(
                f"External screenshot mapping is tentative/ambiguous for hint '{approx_region_hint}'. "
                "Artist camera alignment or region confirmation is required before simulation."
            )
        )
