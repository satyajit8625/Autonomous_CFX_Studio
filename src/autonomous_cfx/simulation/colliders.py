"""Autonomous CFX Studio - Collider Architecture.
Supports animated meshes, continuous motion interpolation, and analytical obstacle shapes.
Separates collision margin from true penetration tolerances and reports raw physical measurements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh

from autonomous_cfx.core.geometry import MeshGeometry, BoundingBox
from autonomous_cfx.core.logger import logger


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
        self.prev_vertices = np.copy(self.current_vertices)
        self.current_frame = frame

        if not self.animation_frames:
            return

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
        """Update Trimesh instance and bounding box."""
        if len(self.current_vertices) == 0 or len(self.geometry.faces) == 0:
            return

        self._trimesh_cache = trimesh.Trimesh(
            vertices=self.current_vertices,
            faces=self.geometry.faces,
            process=False,
        )
        self._aabb = BoundingBox(
            min_point=np.min(self.current_vertices, axis=0),
            max_point=np.max(self.current_vertices, axis=0),
        )

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

        if self._trimesh_cache is None or self._aabb is None or N == 0:
            return contact_mask, projected, contact_normals, collider_velocities

        # Step 1: Broadphase AABB query
        candidate_mask = self._aabb.contains(points, margin=margin * 3.0)
        if not np.any(candidate_mask):
            return contact_mask, projected, contact_normals, collider_velocities

        candidate_indices = np.where(candidate_mask)[0]
        candidate_pts = points[candidate_indices]

        # Step 2: Proximity query on candidates using Trimesh
        try:
            closest_pts, distances, triangle_ids = trimesh.proximity.closest_point(
                self._trimesh_cache, candidate_pts
            )

            # Triangle normals
            tri_normals = self._trimesh_cache.face_normals[triangle_ids]

            # Vector from closest surface point to cloth point
            diff = candidate_pts - closest_pts
            # Signed distance based on face normal dot product (< 0 means inside collider mesh)
            signed_dist = np.sum(diff * tri_normals, axis=1)

            # Contact is anything within margin
            is_contact = signed_dist < margin
            # True penetration is strictly inside collider surface (signed_dist < 0.0)
            is_true_pen = signed_dist < 0.0

            if np.any(is_contact):
                contact_cand_idx = candidate_indices[is_contact]
                contact_mask[contact_cand_idx] = True

                target_normals = tri_normals[is_contact]
                corrected_pts = closest_pts[is_contact] + target_normals * margin
                projected[contact_cand_idx] = corrected_pts
                contact_normals[contact_cand_idx] = target_normals

                # Collider motion velocity
                if len(self.prev_vertices) == len(self.current_vertices):
                    triangles = self.geometry.faces[triangle_ids[is_contact]]
                    tri_v_curr = np.mean(self.current_vertices[triangles], axis=1)
                    tri_v_prev = np.mean(self.prev_vertices[triangles], axis=1)
                    collider_velocities[contact_cand_idx] = (tri_v_curr - tri_v_prev) * self.fps

            # Record raw measurements
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
        except Exception as e:
            logger.debug(f"Proximity test fallback: {e}")

        return contact_mask, projected, contact_normals, collider_velocities
