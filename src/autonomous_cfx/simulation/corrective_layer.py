"""Autonomous CFX Studio - Non-Destructive Corrective Layer Engine.

Applies bounded, time-consistent surface corrections (penetration de-projection,
Laplacian defect smoothing, protected feature freezing, and smooth temporal bell-curve
fades) non-destructively to simulation caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

from autonomous_cfx.agent.correction_agent import StructuredCorrectionRequest
from autonomous_cfx.core.geometry import MeshGeometry


@dataclass
class CorrectionTelemetry:
    """Quantitative metrics measuring precision, temporal smoothness, and locality."""
    max_penetration_before_mm: float
    max_penetration_after_mm: float
    max_displacement_outside_region_mm: float
    max_temporal_velocity_jump: float
    corrected_frames_count: int
    protected_vertices_preserved: bool
    is_flicker_free: bool


class CorrectiveLayerEngine:
    """Non-destructive temporal corrective layer applicator."""

    @classmethod
    def apply_correction(
        cls,
        sim_cache: List[MeshGeometry],
        request: StructuredCorrectionRequest,
        colliders_cache: Optional[List[MeshGeometry]] = None,
        collision_margin: float = 0.008,
        smooth_iterations: int = 3
    ) -> Tuple[List[MeshGeometry], CorrectionTelemetry]:
        """Apply bounded multimodal correction to simulation frames non-destructively."""
        num_frames = len(sim_cache)
        if num_frames == 0:
            raise ValueError("Simulation cache is empty.")

        # Protected vertex set
        protected_verts: Set[int] = set()
        for prot_sel in request.protected_regions:
            protected_verts.update(prot_sel.selected_vertices)

        # Spatial weights
        spatial_weights: Dict[int, float] = {}
        for v_idx, w in request.target_region.vertex_weights.items():
            if v_idx in protected_verts:
                spatial_weights[v_idx] = 0.0
            else:
                spatial_weights[v_idx] = float(w)

        start_f, end_f = request.frame_range
        fade_in, fade_out = request.fade_frames

        # Measure baseline penetration in target region
        max_pen_before_mm = 0.0
        target_indices = request.target_region.selected_vertices
        if colliders_cache:
            for f_idx in range(num_frames):
                frame_num = f_idx + 1
                if start_f - fade_in <= frame_num <= end_f + fade_out:
                    cloth = sim_cache[f_idx]
                    col = colliders_cache[min(f_idx, len(colliders_cache) - 1)]
                    pen_mm = cls._measure_penetration_depth(cloth, col, collision_margin, target_indices=target_indices)
                    max_pen_before_mm = max(max_pen_before_mm, pen_mm)

        corrected_frames: List[MeshGeometry] = []
        max_outside_disp = 0.0
        corrected_frames_count = 0

        # Create adjacency graph for Laplacian smoothing
        ref_mesh = sim_cache[0]
        edges, _ = ref_mesh.compute_edges()
        adj: Dict[int, List[int]] = {i: [] for i in range(ref_mesh.num_vertices)}
        for v1, v2 in edges:
            adj[v1].append(v2)
            adj[v2].append(v1)

        for f_idx in range(num_frames):
            frame_num = f_idx + 1
            src_mesh = sim_cache[f_idx]
            verts_new = np.copy(src_mesh.vertices)

            # Compute temporal bell-curve weight W_temporal(t)
            w_time = cls._compute_temporal_weight(frame_num, start_f, end_f, fade_in, fade_out)

            if w_time > 1e-4:
                corrected_frames_count += 1
                collider = colliders_cache[min(f_idx, len(colliders_cache) - 1)] if colliders_cache else None

                # Compute per-vertex displacements Delta x
                delta_x = np.zeros_like(verts_new, dtype=np.float32)

                if collider is not None:
                    col_kdtree = KDTree(collider.vertices)
                    col_normals = collider.compute_normals()

                    for v_idx, w_spatial in spatial_weights.items():
                        if w_spatial <= 1e-4 or v_idx in protected_verts:
                            continue

                        p = verts_new[v_idx]
                        dist, nearest_idx = col_kdtree.query(p)
                        n = col_normals[nearest_idx]
                        col_pos = collider.vertices[nearest_idx]

                        # Check if inside collider or violating margin
                        to_cloth = p - col_pos
                        signed_dist = float(np.dot(to_cloth, n))

                        if signed_dist < collision_margin:
                            if signed_dist < 0.0:
                                needed_push = (-signed_dist + collision_margin * w_spatial) * w_time + 0.001
                            else:
                                needed_push = (collision_margin - signed_dist) * (w_spatial * w_time)
                            disp_vec = n * needed_push
                            delta_x[v_idx] = disp_vec

                # Laplacian smoothing of displacements to preserve natural folds and avoid crinkling
                for _ in range(smooth_iterations):
                    delta_smooth = np.copy(delta_x)
                    for v_idx, w_spatial in spatial_weights.items():
                        if v_idx in protected_verts or w_spatial <= 1e-4:
                            continue
                        neighbors = adj[v_idx]
                        if neighbors:
                            avg_delta = np.mean(delta_x[neighbors], axis=0)
                            delta_smooth[v_idx] = 0.5 * delta_x[v_idx] + 0.5 * avg_delta
                    delta_x = delta_smooth

                # Ensure penetrating vertices retain full required clearance along normal
                if collider is not None:
                    for v_idx, w_spatial in spatial_weights.items():
                        if w_spatial <= 1e-4 or v_idx in protected_verts:
                            continue
                        p = src_mesh.vertices[v_idx]
                        dist, nearest_idx = col_kdtree.query(p)
                        n = col_normals[nearest_idx]
                        col_pos = collider.vertices[nearest_idx]
                        signed_dist = float(np.dot(p - col_pos, n))
                        if signed_dist < collision_margin:
                            if signed_dist < 0.0:
                                target_push = (-signed_dist + collision_margin * w_spatial) * w_time + 0.001
                            else:
                                target_push = (collision_margin - signed_dist) * (w_spatial * w_time)
                            proj = float(np.dot(delta_x[v_idx], n))
                            if proj < target_push:
                                delta_x[v_idx] += n * (target_push - proj)

                # Apply displacements
                verts_new += delta_x

                # Verification: measure displacement on unaffected vertices
                for v_idx in range(ref_mesh.num_vertices):
                    if v_idx not in spatial_weights or spatial_weights[v_idx] <= 1e-4:
                        disp_norm = float(np.linalg.norm(verts_new[v_idx] - src_mesh.vertices[v_idx]))
                        max_outside_disp = max(max_outside_disp, disp_norm)

            frame_copy = MeshGeometry(
                name=f"{src_mesh.name}_corrected",
                vertices=verts_new,
                faces=src_mesh.faces,
                uvs=src_mesh.uvs,
                normals=None,
                vertex_groups=src_mesh.vertex_groups
            )
            corrected_frames.append(frame_copy)

        # Measure post-correction penetration
        max_pen_after_mm = 0.0
        if colliders_cache:
            for f_idx in range(num_frames):
                frame_num = f_idx + 1
                if start_f <= frame_num <= end_f:
                    cloth = corrected_frames[f_idx]
                    col = colliders_cache[min(f_idx, len(colliders_cache) - 1)]
                    pen_mm = cls._measure_penetration_depth(cloth, col, collision_margin, target_indices=target_indices)
                    max_pen_after_mm = max(max_pen_after_mm, pen_mm)

        # Measure temporal acceleration of the corrective layer itself (second difference)
        max_layer_accel = 0.0
        for f_idx in range(1, num_frames - 1):
            d_prev = corrected_frames[f_idx - 1].vertices - sim_cache[f_idx - 1].vertices
            d_curr = corrected_frames[f_idx].vertices - sim_cache[f_idx].vertices
            d_next = corrected_frames[f_idx + 1].vertices - sim_cache[f_idx + 1].vertices
            accel = np.linalg.norm(d_next - 2.0 * d_curr + d_prev, axis=1)
            max_layer_accel = max(max_layer_accel, float(np.max(accel)))

        # Protected vertices preserved check
        protected_preserved = True
        if protected_verts:
            for f_idx in range(num_frames):
                diff = np.linalg.norm(
                    corrected_frames[f_idx].vertices[list(protected_verts)] -
                    sim_cache[f_idx].vertices[list(protected_verts)]
                )
                if diff > 1e-6:
                    protected_preserved = False
                    break

        telemetry = CorrectionTelemetry(
            max_penetration_before_mm=round(max_pen_before_mm, 2),
            max_penetration_after_mm=round(max_pen_after_mm, 2),
            max_displacement_outside_region_mm=round(max_outside_disp * 1000.0, 4),
            max_temporal_velocity_jump=round(max_layer_accel, 4),
            corrected_frames_count=corrected_frames_count,
            protected_vertices_preserved=protected_preserved,
            is_flicker_free=max_layer_accel < 0.2
        )

        return corrected_frames, telemetry

    @staticmethod
    def _compute_temporal_weight(
        frame: int,
        start_f: int,
        end_f: int,
        fade_in: int,
        fade_out: int
    ) -> float:
        """Compute smooth sin^2 temporal bell-curve weight in [0.0, 1.0]."""
        if start_f <= frame <= end_f:
            return 1.0

        # Fade in: [start_f - fade_in, start_f)
        if start_f - fade_in <= frame < start_f:
            if fade_in <= 0:
                return 0.0
            u = (frame - (start_f - fade_in)) / float(fade_in)
            return float(np.sin(0.5 * np.pi * u) ** 2)

        # Fade out: (end_f, end_f + fade_out]
        if end_f < frame <= end_f + fade_out:
            if fade_out <= 0:
                return 0.0
            v = ((end_f + fade_out) - frame) / float(fade_out)
            return float(np.sin(0.5 * np.pi * v) ** 2)

        return 0.0

    @staticmethod
    def _measure_penetration_depth(
        cloth: MeshGeometry,
        collider: MeshGeometry,
        margin: float,
        target_indices: Optional[List[int]] = None
    ) -> float:
        """Calculate max true penetration depth inside collider volume in millimeters."""
        test_verts = cloth.vertices[target_indices] if target_indices is not None else cloth.vertices
        if len(test_verts) == 0:
            return 0.0

        kdtree = KDTree(collider.vertices)
        normals = collider.compute_normals()

        dists, indices = kdtree.query(test_verts)
        col_pts = collider.vertices[indices]
        col_n = normals[indices]

        to_cloth = test_verts - col_pts
        signed_dists = np.sum(to_cloth * col_n, axis=1)

        penetrating = signed_dists < 0.0
        if np.any(penetrating):
            max_pen = float(np.max(-signed_dists[penetrating]))
            return max_pen * 1000.0  # mm
        return 0.0
