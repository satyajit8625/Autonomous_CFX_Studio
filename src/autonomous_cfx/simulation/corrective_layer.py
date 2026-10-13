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
    fps: float = 24.0
    max_temporal_accel_m_s2: float = 0.0


class CorrectiveLayerEngine:
    """Non-destructive temporal corrective layer applicator."""

    @classmethod
    def apply_correction(
        cls,
        sim_cache: List[MeshGeometry],
        request: StructuredCorrectionRequest,
        colliders_cache: Optional[List[MeshGeometry]] = None,
        collision_margin: float = 0.008,
        smooth_iterations: int = 3,
        fps: float = 24.0
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
                    from autonomous_cfx.core.collision_detector import PenetrationClassifier
                    target_v_list = [v for v, w in spatial_weights.items() if w > 1e-4 and v not in protected_verts]

                    for _pass in range(2):
                        temp_geom = MeshGeometry(name="temp", vertices=verts_new, faces=src_mesh.faces)
                        pen_res = PenetrationClassifier.evaluate_penetration(
                            temp_geom, collider, margin=collision_margin, target_indices=target_v_list
                        )
                        pen_dict = {rec.vertex_index: rec for rec in pen_res.records if rec.is_penetrating}
                        if not pen_dict:
                            break

                        for v_idx, rec in pen_dict.items():
                            w_spatial = spatial_weights.get(v_idx, 0.0)
                            target_pt = rec.closest_surface_point + rec.surface_normal * (collision_margin * 1.05)
                            disp = (target_pt - verts_new[v_idx]) * (w_spatial * w_time)
                            delta_x[v_idx] += disp
                            verts_new[v_idx] += disp

                    # Laplacian smoothing on displacements to preserve natural folds and avoid crinkling
                    for _ in range(smooth_iterations):
                        delta_smooth = np.copy(delta_x)
                        for v_idx in target_v_list:
                            neighbors = adj[v_idx]
                            if neighbors:
                                avg_delta = np.mean(delta_x[neighbors], axis=0)
                                delta_smooth[v_idx] = 0.7 * delta_x[v_idx] + 0.3 * avg_delta
                        delta_x = delta_smooth

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

        # Physical acceleration scaled by FPS^2 (m/s^2)
        max_physical_accel = max_layer_accel * (float(fps) ** 2)

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
            is_flicker_free=bool(max_layer_accel < 0.15),
            fps=float(fps),
            max_temporal_accel_m_s2=round(max_physical_accel, 2)
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
        target_indices: Optional[List[int]] = None,
        max_proximity_threshold: Optional[float] = None
    ) -> float:
        """Calculate max true penetration depth inside collider volume in millimeters.

        Uses PenetrationClassifier separating contact acceleration from volumetric classification:
        - Accurately captures deep penetrations beyond 50 mm without artificial cutoffs.
        - Robust against concave bodies, nearby limbs, open meshes, and far-away cloth.
        """
        from autonomous_cfx.core.collision_detector import PenetrationClassifier
        result = PenetrationClassifier.evaluate_penetration(
            cloth=cloth,
            collider=collider,
            margin=margin,
            target_indices=target_indices
        )
        return float(result.max_penetration_mm)
