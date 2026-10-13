"""Autonomous CFX Studio - Collider Architecture.
Supports animated meshes, continuous motion interpolation, and analytical obstacle shapes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh

from autonomous_cfx.core.geometry import MeshGeometry, BoundingBox
from autonomous_cfx.core.logger import logger


class Collider(ABC):
    """Abstract base class for simulation colliders."""

    def __init__(self, name: str = "collider"):
        self.name = name

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
            penetrated_mask: (N,) bool mask of points colliding/penetrating
            projected_points: (N, 3) corrected surface positions
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
        penetrated_mask = points[:, 1] < (self.height + margin)
        projected = np.copy(points)
        projected[penetrated_mask, 1] = self.height + margin

        normals = np.zeros_like(points)
        normals[:, 1] = 1.0

        velocities = np.zeros_like(points)
        return penetrated_mask, projected, normals, velocities


class AnimatedMeshCollider(Collider):
    """Animated triangular mesh collider with temporal interpolation."""

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
        """Fast proximity projection using AABB pruning and triangle mesh distance."""
        N = len(points)
        penetrated_mask = np.zeros(N, dtype=bool)
        projected = np.copy(points)
        contact_normals = np.zeros_like(points)
        collider_velocities = np.zeros_like(points)

        if self._trimesh_cache is None or self._aabb is None or N == 0:
            return penetrated_mask, projected, contact_normals, collider_velocities

        # Step 1: Broadphase AABB query
        candidate_mask = self._aabb.contains(points, margin=margin * 3.0)
        if not np.any(candidate_mask):
            return penetrated_mask, projected, contact_normals, collider_velocities

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
            # Signed distance based on face normal dot product
            signed_dist = np.sum(diff * tri_normals, axis=1)

            # A point is colliding if signed distance < margin
            is_penetrating = signed_dist < margin

            if np.any(is_penetrating):
                pen_candidate_idx = candidate_indices[is_penetrating]
                penetrated_mask[pen_candidate_idx] = True

                # Target position: closest surface point + normal * margin
                target_normals = tri_normals[is_penetrating]
                corrected_pts = closest_pts[is_penetrating] + target_normals * margin
                projected[pen_candidate_idx] = corrected_pts
                contact_normals[pen_candidate_idx] = target_normals

                # Compute collider motion velocity for contact points
                # v = (v_current - v_prev) * fps
                if len(self.prev_vertices) == len(self.current_vertices):
                    # Estimate surface velocity from closest triangle barycentric or average
                    triangles = self.geometry.faces[triangle_ids[is_penetrating]]
                    tri_v_curr = np.mean(self.current_vertices[triangles], axis=1)
                    tri_v_prev = np.mean(self.prev_vertices[triangles], axis=1)
                    collider_velocities[pen_candidate_idx] = (tri_v_curr - tri_v_prev) * self.fps
        except Exception as e:
            logger.debug(f"Proximity test fallback: {e}")

        return penetrated_mask, projected, contact_normals, collider_velocities
