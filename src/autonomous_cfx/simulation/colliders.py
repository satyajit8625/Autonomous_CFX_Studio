"""Autonomous CFX Studio - Collider Architecture.
Supports animated meshes, continuous motion interpolation, and analytical obstacle shapes.
Separates collision margin from true penetration tolerances and reports raw physical measurements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree

from autonomous_cfx.core.geometry import MeshGeometry, BoundingBox
from autonomous_cfx.core.collision_detector import point_triangle_closest_point
from autonomous_cfx.core.logger import logger

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _compute_vertex_normals_numba(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        F = faces.shape[0]
        V = vertices.shape[0]
        vert_normals = np.zeros((V, 3), dtype=np.float32)

        for f in range(F):
            i0 = faces[f, 0]
            i1 = faces[f, 1]
            i2 = faces[f, 2]

            ax = vertices[i1, 0] - vertices[i0, 0]
            ay = vertices[i1, 1] - vertices[i0, 1]
            az = vertices[i1, 2] - vertices[i0, 2]

            bx = vertices[i2, 0] - vertices[i0, 0]
            by = vertices[i2, 1] - vertices[i0, 1]
            bz = vertices[i2, 2] - vertices[i0, 2]

            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx

            vert_normals[i0, 0] += cx
            vert_normals[i0, 1] += cy
            vert_normals[i0, 2] += cz

            vert_normals[i1, 0] += cx
            vert_normals[i1, 1] += cy
            vert_normals[i1, 2] += cz

            vert_normals[i2, 0] += cx
            vert_normals[i2, 1] += cy
            vert_normals[i2, 2] += cz

        for v in range(V):
            nx = vert_normals[v, 0]
            ny = vert_normals[v, 1]
            nz = vert_normals[v, 2]
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            if norm > 1e-7:
                inv_n = 1.0 / norm
                vert_normals[v, 0] = nx * inv_n
                vert_normals[v, 1] = ny * inv_n
                vert_normals[v, 2] = nz * inv_n
            else:
                vert_normals[v, 0] = 0.0
                vert_normals[v, 1] = 1.0
                vert_normals[v, 2] = 0.0

        return vert_normals

    @njit(fastmath=True)
    def _project_proximity_numba(
        candidate_pts: np.ndarray,
        closest_pts: np.ndarray,
        target_normals: np.ndarray,
        margin: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        M = candidate_pts.shape[0]
        contact_flags = np.zeros(M, dtype=np.bool_)
        true_pen_flags = np.zeros(M, dtype=np.bool_)
        projected_pts = np.copy(candidate_pts)
        contact_normals = np.zeros_like(candidate_pts)
        signed_dists = np.zeros(M, dtype=np.float32)

        for i in range(M):
            dx = candidate_pts[i, 0] - closest_pts[i, 0]
            dy = candidate_pts[i, 1] - closest_pts[i, 1]
            dz = candidate_pts[i, 2] - closest_pts[i, 2]

            nx = target_normals[i, 0]
            ny = target_normals[i, 1]
            nz = target_normals[i, 2]

            s_dist = dx * nx + dy * ny + dz * nz
            signed_dists[i] = s_dist

            if s_dist < margin:
                contact_flags[i] = True
                pushout = margin - s_dist
                projected_pts[i, 0] = candidate_pts[i, 0] + pushout * nx
                projected_pts[i, 1] = candidate_pts[i, 1] + pushout * ny
                projected_pts[i, 2] = candidate_pts[i, 2] + pushout * nz
                contact_normals[i, 0] = nx
                contact_normals[i, 1] = ny
                contact_normals[i, 2] = nz

            if s_dist < 0.0:
                true_pen_flags[i] = True

        return contact_flags, true_pen_flags, projected_pts, contact_normals, signed_dists
else:
    _compute_vertex_normals_numba = None
    _project_proximity_numba = None


@dataclass
class CollisionMeasurement:
    """Raw physical measurements separating margin buffer contact from true interior penetration."""
    contact_count: int = 0
    margin_intrusion_depth: float = 0.0     # meters inside margin buffer [0, margin]
    true_penetration_count: int = 0         # vertices strictly inside collider body (signed_dist < 0)
    raw_max_penetration_depth: float = 0.0  # meters inside body (< 0 signed distance magnitude)
    penetrating_vertex_ids: List[int] = field(default_factory=list)


class Collider(ABC):
    """Abstract base class for simulation colliders."""

    def __init__(self, name: str = "collider"):
        self.name = name
        self.last_measurement = CollisionMeasurement()

    @abstractmethod
    def update_frame(self, frame: float) -> None:
        """Update collider transform / geometry for current time/frame."""
        pass

    @abstractmethod
    def project_points(
        self,
        points: np.ndarray,
        margin: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Test and project points against the collider.
        Returns:
            contact_mask: (N,) bool mask of points colliding/within margin
            projected_points: (N, 3) corrected surface positions (surface + margin)
            contact_normals: (N, 3) surface normals at contact
            collider_velocities: (N, 3) velocity of contact points
        """
        pass


class GroundPlaneCollider(Collider):
    """Infinite ground plane collider at y = height."""

    def __init__(self, name: str = "ground", height: float = 0.0):
        super().__init__(name)
        self.height = height

    def update_frame(self, frame: float) -> None:
        pass

    def project_points(
        self,
        points: np.ndarray,
        margin: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        signed_dist = points[:, 1] - self.height
        contact_mask = signed_dist < margin
        true_pen_mask = signed_dist < 0.0

        projected = np.copy(points)
        projected[contact_mask, 1] = self.height + margin

        normals = np.zeros_like(points)
        normals[:, 1] = 1.0

        velocities = np.zeros_like(points)

        # Record raw physical measurements
        true_pen_count = int(np.sum(true_pen_mask))
        raw_depths = np.maximum(0.0, -signed_dist[true_pen_mask]) if true_pen_count > 0 else np.zeros(0)
        max_raw_depth = float(np.max(raw_depths)) if len(raw_depths) > 0 else 0.0

        pen_ids = list(np.where(true_pen_mask)[0])

        self.last_measurement = CollisionMeasurement(
            contact_count=int(np.sum(contact_mask)),
            margin_intrusion_depth=float(np.max(np.maximum(0.0, margin - signed_dist))) if np.any(contact_mask) else 0.0,
            true_penetration_count=true_pen_count,
            raw_max_penetration_depth=max_raw_depth,
            penetrating_vertex_ids=pen_ids,
        )

        return contact_mask, projected, normals, velocities


class AnimatedMeshCollider(Collider):
    """Animated triangular mesh collider with temporal interpolation and raw penetration reporting."""

    def __init__(
        self,
        name: str = "animated_collider",
        base_geometry: Optional[MeshGeometry] = None,
        animation_frames: Optional[Dict[int, np.ndarray]] = None,
        fps: float = 24.0,
    ):
        super().__init__(name)
        self.geometry = base_geometry or MeshGeometry(name=name)
        self.animation_frames: Dict[int, np.ndarray] = animation_frames or {}
        self.fps = fps
        self.current_frame: float = 1.0
        self.current_vertices: np.ndarray = (
            np.copy(self.geometry.vertices) if len(self.geometry.vertices) > 0 else np.zeros((0, 3))
        )
        self.prev_vertices: np.ndarray = np.copy(self.current_vertices)
        self._trimesh_cache: Optional[trimesh.Trimesh] = None
        self._aabb: Optional[BoundingBox] = None

        if len(self.current_vertices) > 0:
            self._update_mesh_cache()

    def set_animation_data(self, frames_dict: Dict[int, np.ndarray]) -> None:
        """Provide per-frame vertex coordinates."""
        self.animation_frames = frames_dict
        if 1 in self.animation_frames:
            self.current_vertices = np.copy(self.animation_frames[1])
            self.prev_vertices = np.copy(self.current_vertices)
            self._update_mesh_cache()

    def update_frame(self, frame: float) -> None:
        """Interpolate vertices for frame (supporting sub-frame fractions)."""
        if not self.animation_frames:
            return

        if hasattr(self, "_last_evaluated_frame") and abs(self._last_evaluated_frame - frame) < 1e-5:
            return
        self._last_evaluated_frame = frame

        self.prev_vertices = np.copy(self.current_vertices)
        self.current_frame = frame

        f_floor = int(np.floor(frame))
        f_ceil = int(np.ceil(frame))
        alpha = frame - f_floor

        # Clamp to available frames
        sorted_keys = sorted(self.animation_frames.keys())
        k_min, k_max = sorted_keys[0], sorted_keys[-1]
        f_floor = max(k_min, min(k_max, f_floor))
        f_ceil = max(k_min, min(k_max, f_ceil))

        v0 = self.animation_frames.get(f_floor, self.current_vertices)
        v1 = self.animation_frames.get(f_ceil, self.current_vertices)

        if f_floor == f_ceil or alpha == 0:
            self.current_vertices = np.copy(v0)
        else:
            self.current_vertices = (1.0 - alpha) * v0 + alpha * v1

        self._update_mesh_cache()

    def _update_mesh_cache(self) -> None:
        """Update spatial tree, vertex normals, and bounding box."""
        if len(self.current_vertices) == 0 or len(self.geometry.faces) == 0:
            return

        self._aabb = BoundingBox(
            min_point=np.min(self.current_vertices, axis=0),
            max_point=np.max(self.current_vertices, axis=0),
        )

        if NUMBA_AVAILABLE:
            self._vertex_normals = _compute_vertex_normals_numba(
                np.ascontiguousarray(self.current_vertices, dtype=np.float32),
                np.ascontiguousarray(self.geometry.faces, dtype=np.int32),
            )
        else:
            faces = self.geometry.faces
            v0 = self.current_vertices[faces[:, 0]]
            v1 = self.current_vertices[faces[:, 1]]
            v2 = self.current_vertices[faces[:, 2]]
            face_cross = np.cross(v1 - v0, v2 - v0)

            vert_normals = np.zeros_like(self.current_vertices, dtype=np.float32)
            for i in range(3):
                np.add.at(vert_normals, faces[:, i], face_cross)

            norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vertex_normals = (vert_normals / norms).astype(np.float32)

        self._vertex_tree = KDTree(self.current_vertices)

    def project_points(
        self,
        points: np.ndarray,
        margin: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fast proximity projection separating margin buffer contact from raw true interior penetrations."""
        N = len(points)
        contact_mask = np.zeros(N, dtype=bool)
        projected = np.copy(points)
        contact_normals = np.zeros_like(points)
        collider_velocities = np.zeros_like(points)

        self.last_measurement = CollisionMeasurement()

        if not hasattr(self, "_vertex_tree") or self._vertex_tree is None or self._aabb is None or N == 0:
            return contact_mask, projected, contact_normals, collider_velocities

        # Step 1: Broadphase AABB query
        candidate_mask = self._aabb.contains(points, margin=margin * 3.0)
        if not np.any(candidate_mask):
            return contact_mask, projected, contact_normals, collider_velocities

        candidate_indices = np.where(candidate_mask)[0]
        candidate_pts = points[candidate_indices]

        # Step 2: Proximity query using vertex KDTree
        _, closest_v_idx = self._vertex_tree.query(candidate_pts)
        closest_pts = self.current_vertices[closest_v_idx]
        target_normals = self._vertex_normals[closest_v_idx]

        if NUMBA_AVAILABLE:
            (
                c_mask_local,
                t_mask_local,
                proj_local,
                normals_local,
                s_dist_local,
            ) = _project_proximity_numba(
                np.ascontiguousarray(candidate_pts, dtype=np.float32),
                np.ascontiguousarray(closest_pts, dtype=np.float32),
                np.ascontiguousarray(target_normals, dtype=np.float32),
                float(margin),
            )

            if np.any(c_mask_local):
                contact_cand_idx = candidate_indices[c_mask_local]
                contact_mask[contact_cand_idx] = True
                projected[contact_cand_idx] = proj_local[c_mask_local]
                contact_normals[contact_cand_idx] = normals_local[c_mask_local]

                if len(self.prev_vertices) == len(self.current_vertices):
                    active_v_idx = closest_v_idx[c_mask_local]
                    tri_v_curr = self.current_vertices[active_v_idx]
                    tri_v_prev = self.prev_vertices[active_v_idx]
                    collider_velocities[contact_cand_idx] = (tri_v_curr - tri_v_prev) * self.fps

            true_pen_indices = candidate_indices[t_mask_local]
            raw_depths = -s_dist_local[t_mask_local] if np.any(t_mask_local) else np.zeros(0, dtype=np.float32)
            max_raw = float(np.max(raw_depths)) if len(raw_depths) > 0 else 0.0

            self.last_measurement = CollisionMeasurement(
                contact_count=int(np.sum(c_mask_local)),
                margin_intrusion_depth=float(np.max(np.maximum(0.0, margin - s_dist_local))) if np.any(c_mask_local) else 0.0,
                true_penetration_count=int(len(true_pen_indices)),
                raw_max_penetration_depth=max_raw,
                penetrating_vertex_ids=list(true_pen_indices),
            )
            return contact_mask, projected, contact_normals, collider_velocities

        # Fallback NumPy path
        diff = candidate_pts - closest_pts
        signed_dist = np.sum(diff * target_normals, axis=1)

        is_contact = signed_dist < margin
        is_true_pen = signed_dist < 0.0

        if np.any(is_contact):
            contact_cand_idx = candidate_indices[is_contact]
            contact_mask[contact_cand_idx] = True

            active_normals = target_normals[is_contact]
            pushout = (margin - signed_dist[is_contact])[:, np.newaxis] * active_normals
            corrected_pts = candidate_pts[is_contact] + pushout
            projected[contact_cand_idx] = corrected_pts
            contact_normals[contact_cand_idx] = active_normals

            if len(self.prev_vertices) == len(self.current_vertices):
                active_v_idx = closest_v_idx[is_contact]
                tri_v_curr = self.current_vertices[active_v_idx]
                tri_v_prev = self.prev_vertices[active_v_idx]
                collider_velocities[contact_cand_idx] = (tri_v_curr - tri_v_prev) * self.fps

        true_pen_indices = candidate_indices[is_true_pen]
        raw_depths = -signed_dist[is_true_pen] if np.any(is_true_pen) else np.zeros(0)
        max_raw = float(np.max(raw_depths)) if len(raw_depths) > 0 else 0.0

        self.last_measurement = CollisionMeasurement(
            contact_count=int(np.sum(is_contact)),
            margin_intrusion_depth=float(np.max(np.maximum(0.0, margin - signed_dist))) if np.any(is_contact) else 0.0,
            true_penetration_count=int(len(true_pen_indices)),
            raw_max_penetration_depth=max_raw,
            penetrating_vertex_ids=list(true_pen_indices),
        )

        return contact_mask, projected, contact_normals, collider_velocities
