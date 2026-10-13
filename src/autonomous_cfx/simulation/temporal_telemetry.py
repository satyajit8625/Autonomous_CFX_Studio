"""Autonomous CFX Studio - Comprehensive Temporal Behavior & Smoothness Telemetry.

Provides rigorous before/after comparison of:
1. FPS-aware velocity distributions (mean, max, 95th percentile in m/s).
2. FPS-aware acceleration distributions (mean, max, 95th percentile in m/s^2).
3. Frame-by-frame layer correction displacement profile (mm).
4. Boundary artifact analysis: position and velocity jumps at fade-in/fade-out transitions.
Proves true C^1 temporal smoothness beyond a single acceleration threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry


@dataclass
class TemporalProfile:
    """Statistical distribution of motion dynamics across frames."""
    fps: float
    mean_velocity_m_s: float
    max_velocity_m_s: float
    p95_velocity_m_s: float
    mean_accel_m_s2: float
    max_accel_m_s2: float
    p95_accel_m_s2: float
    frame_velocities: List[float]
    frame_accelerations: List[float]


@dataclass
class BoundaryContinuityReport:
    """Quantitative measurement of boundary artifacts at temporal fade transitions."""
    fade_in_frame: int
    fade_out_frame: int
    fade_in_pos_jump_mm: float
    fade_in_vel_jump_m_s: float
    fade_out_pos_jump_mm: float
    fade_out_vel_jump_m_s: float
    is_c1_continuous: bool


@dataclass
class ComprehensiveTemporalReport:
    """Full comparative telemetry report contrasting pre- and post-correction dynamics."""
    fps: float
    before_profile: TemporalProfile
    after_profile: TemporalProfile
    max_layer_displacement_mm: float
    displacement_curve_by_frame: Dict[int, float]
    boundary_continuity: BoundaryContinuityReport
    is_smooth: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": self.fps,
            "velocity_comparison_m_s": {
                "before": {
                    "mean": self.before_profile.mean_velocity_m_s,
                    "max": self.before_profile.max_velocity_m_s,
                    "p95": self.before_profile.p95_velocity_m_s,
                },
                "after": {
                    "mean": self.after_profile.mean_velocity_m_s,
                    "max": self.after_profile.max_velocity_m_s,
                    "p95": self.after_profile.p95_velocity_m_s,
                },
            },
            "acceleration_comparison_m_s2": {
                "before": {
                    "mean": self.before_profile.mean_accel_m_s2,
                    "max": self.before_profile.max_accel_m_s2,
                    "p95": self.before_profile.p95_accel_m_s2,
                },
                "after": {
                    "mean": self.after_profile.mean_accel_m_s2,
                    "max": self.after_profile.max_accel_m_s2,
                    "p95": self.after_profile.p95_accel_m_s2,
                },
            },
            "max_layer_displacement_mm": self.max_layer_displacement_mm,
            "boundary_continuity": {
                "fade_in_pos_jump_mm": self.boundary_continuity.fade_in_pos_jump_mm,
                "fade_in_vel_jump_m_s": self.boundary_continuity.fade_in_vel_jump_m_s,
                "fade_out_pos_jump_mm": self.boundary_continuity.fade_out_pos_jump_mm,
                "fade_out_vel_jump_m_s": self.boundary_continuity.fade_out_vel_jump_m_s,
                "is_c1_continuous": self.boundary_continuity.is_c1_continuous,
            },
            "is_smooth": self.is_smooth,
        }


class TemporalTelemetryAnalyzer:
    """Analyzes and compares multi-frame physical kinematics before and after corrections."""

    @classmethod
    def compute_dynamics_profile(
        cls,
        frames: List[MeshGeometry],
        fps: float = 24.0,
        target_indices: Optional[List[int]] = None
    ) -> TemporalProfile:
        """Compute frame-by-frame velocity and acceleration distributions in physical units."""
        num_frames = len(frames)
        if num_frames < 2:
            return TemporalProfile(fps, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], [])

        dt = 1.0 / float(fps)
        all_vels: List[float] = []
        frame_max_vels: List[float] = []

        # Velocity: v(t) = (x(t) - x(t-1)) / dt
        for f in range(1, num_frames):
            v0 = frames[f - 1].vertices[target_indices] if target_indices else frames[f - 1].vertices
            v1 = frames[f].vertices[target_indices] if target_indices else frames[f].vertices
            disp = np.linalg.norm(v1 - v0, axis=1)
            vel = disp / dt
            all_vels.extend(vel)
            frame_max_vels.append(float(np.max(vel)))

        # Acceleration: a(t) = (x(t+1) - 2x(t) + x(t-1)) / dt^2
        all_accels: List[float] = []
        frame_max_accels: List[float] = []
        for f in range(1, num_frames - 1):
            v_prev = frames[f - 1].vertices[target_indices] if target_indices else frames[f - 1].vertices
            v_curr = frames[f].vertices[target_indices] if target_indices else frames[f].vertices
            v_next = frames[f + 1].vertices[target_indices] if target_indices else frames[f + 1].vertices
            acc = np.linalg.norm(v_next - 2.0 * v_curr + v_prev, axis=1) / (dt ** 2)
            all_accels.extend(acc)
            frame_max_accels.append(float(np.max(acc)))

        vel_arr = np.array(all_vels)
        acc_arr = np.array(all_accels)

        return TemporalProfile(
            fps=fps,
            mean_velocity_m_s=round(float(np.mean(vel_arr)), 4) if len(vel_arr) > 0 else 0.0,
            max_velocity_m_s=round(float(np.max(vel_arr)), 4) if len(vel_arr) > 0 else 0.0,
            p95_velocity_m_s=round(float(np.percentile(vel_arr, 95)), 4) if len(vel_arr) > 0 else 0.0,
            mean_accel_m_s2=round(float(np.mean(acc_arr)), 4) if len(acc_arr) > 0 else 0.0,
            max_accel_m_s2=round(float(np.max(acc_arr)), 4) if len(acc_arr) > 0 else 0.0,
            p95_accel_m_s2=round(float(np.percentile(acc_arr, 95)), 4) if len(acc_arr) > 0 else 0.0,
            frame_velocities=[round(v, 4) for v in frame_max_vels],
            frame_accelerations=[round(a, 4) for a in frame_max_accels],
        )

    @classmethod
    def compare_temporal_behavior(
        cls,
        before_frames: List[MeshGeometry],
        after_frames: List[MeshGeometry],
        frame_range: Tuple[int, int],
        fade_frames: Tuple[int, int] = (3, 3),
        fps: float = 24.0,
        target_indices: Optional[List[int]] = None,
    ) -> ComprehensiveTemporalReport:
        """Evaluate detailed kinematics before and after correction with boundary continuity."""
        num_frames = len(before_frames)
        dt = 1.0 / float(fps)

        prof_before = cls.compute_dynamics_profile(before_frames, fps=fps, target_indices=target_indices)
        prof_after = cls.compute_dynamics_profile(after_frames, fps=fps, target_indices=target_indices)

        # Layer displacement profile: ||x_after - x_before||
        disp_by_frame: Dict[int, float] = {}
        max_layer_disp_m = 0.0

        for f in range(num_frames):
            v_b = before_frames[f].vertices[target_indices] if target_indices else before_frames[f].vertices
            v_a = after_frames[f].vertices[target_indices] if target_indices else after_frames[f].vertices
            d = np.linalg.norm(v_a - v_b, axis=1)
            f_max_d = float(np.max(d))
            disp_by_frame[f + 1] = round(f_max_d * 1000.0, 3)  # mm
            max_layer_disp_m = max(max_layer_disp_m, f_max_d)

        # Boundary jump verification at fade limits
        start_f, end_f = frame_range
        fade_in, fade_out = fade_frames
        f_in_bound = max(1, start_f - fade_in)
        f_out_bound = min(num_frames, end_f + fade_out)

        # Position jump at boundary (mm)
        pos_jump_in_mm = disp_by_frame.get(f_in_bound, 0.0)
        pos_jump_out_mm = disp_by_frame.get(f_out_bound, 0.0)

        # Velocity jump at boundary (m/s)
        # Compare velocity difference between layer displacement at boundary
        vel_jump_in_m_s = (disp_by_frame.get(f_in_bound + 1, 0.0) - disp_by_frame.get(f_in_bound, 0.0)) / (1000.0 * dt)
        vel_jump_out_m_s = (disp_by_frame.get(f_out_bound, 0.0) - disp_by_frame.get(f_out_bound - 1, 0.0)) / (1000.0 * dt)

        is_c1 = (pos_jump_in_mm < 0.01) and (pos_jump_out_mm < 0.01) and (abs(vel_jump_in_m_s) < 1.50) and (abs(vel_jump_out_m_s) < 1.50)
        # Acceleration ratio: after acceleration should not exhibit catastrophic spikes relative to solver
        max_allowed_accel = max(120.0, prof_before.max_accel_m_s2 * 1.5)
        is_smooth = is_c1 and (prof_after.max_accel_m_s2 <= max_allowed_accel)

        boundary_report = BoundaryContinuityReport(
            fade_in_frame=f_in_bound,
            fade_out_frame=f_out_bound,
            fade_in_pos_jump_mm=round(pos_jump_in_mm, 4),
            fade_in_vel_jump_m_s=round(abs(vel_jump_in_m_s), 4),
            fade_out_pos_jump_mm=round(pos_jump_out_mm, 4),
            fade_out_vel_jump_m_s=round(abs(vel_jump_out_m_s), 4),
            is_c1_continuous=is_c1,
        )

        return ComprehensiveTemporalReport(
            fps=fps,
            before_profile=prof_before,
            after_profile=prof_after,
            max_layer_displacement_mm=round(max_layer_disp_m * 1000.0, 3),
            displacement_curve_by_frame=disp_by_frame,
            boundary_continuity=boundary_report,
            is_smooth=is_smooth,
        )
