"""Autonomous CFX Studio - Simulation Quality Evaluator.
Performs automated multi-metric diagnostics on simulation caches: penetration, stretch, and numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.logger import logger


class QualityStatus(str, Enum):
    APPROVED = "APPROVED"
    NEEDS_REPAIR = "NEEDS_REPAIR"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


@dataclass
class QualityMetricsReport:
    """Comprehensive production CFX diagnostic report."""
    status: QualityStatus
    score: float  # 0.0 to 100.0
    total_frames: int
    frames_with_penetration: int
    max_penetration_depth: float
    total_penetration_events: int
    max_stretch_ratio: float
    mean_stretch_ratio: float
    overstretched_edge_count: int
    peak_velocity: float
    is_numerically_stable: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class SimulationEvaluator:
    """Evaluates simulation results against physical realism and production CFX standards."""

    # Tolerances
    MAX_ALLOWABLE_PENETRATION_DEPTH = 0.012  # 12mm
    MAX_ALLOWABLE_STRETCH_RATIO = 0.15       # 15% strain
    VELOCITY_EXPLOSION_THRESHOLD = 30.0      # 30 m/s

    @classmethod
    def evaluate(
        cls,
        cache: SimulationCache,
        base_geometry: MeshGeometry,
    ) -> QualityMetricsReport:
        """Run comprehensive multi-frame evaluation."""
        issues: List[str] = []
        recommendations: List[str] = []

        frames = cache.frame_numbers
        total_frames = len(frames)
        if total_frames == 0:
            return QualityMetricsReport(
                status=QualityStatus.CRITICAL_FAILURE,
                score=0.0,
                total_frames=0,
                frames_with_penetration=0,
                max_penetration_depth=0.0,
                total_penetration_events=0,
                max_stretch_ratio=0.0,
                mean_stretch_ratio=0.0,
                overstretched_edge_count=0,
                peak_velocity=0.0,
                is_numerically_stable=False,
                issues=["Simulation cache contains zero frames."],
                recommendations=["Verify solver initialization and step execution."],
            )

        # 1. Edge stretch analysis
        edges, rest_lens = base_geometry.compute_edges()
        max_stretch_overall = 0.0
        sum_stretch_overall = 0.0
        total_edge_samples = 0
        overstretched_edges = 0

        # 2. Velocity and stability analysis
        peak_velocity = 0.0
        is_stable = True

        # 3. Penetration aggregation from frame metrics
        total_pen_events = 0
        max_pen_depth = 0.0
        frames_pen_count = 0

        for frame_num in frames:
            pos = cache.frames[frame_num]

            # Check for NaN / Inf
            if np.isnan(pos).any() or np.isinf(pos).any():
                is_stable = False
                issues.append(f"Frame {frame_num}: NaN or Inf detected in vertex positions.")
                break

            # Velocities
            if frame_num in cache.velocities:
                vels = cache.velocities[frame_num]
                v_mag = np.linalg.norm(vels, axis=1)
                max_v = float(np.max(v_mag)) if len(v_mag) > 0 else 0.0
                peak_velocity = max(peak_velocity, max_v)
                if max_v > cls.VELOCITY_EXPLOSION_THRESHOLD:
                    is_stable = False
                    issues.append(f"Frame {frame_num}: Velocity explosion detected ({max_v:.1f} m/s).")

            # Strains
            if len(edges) > 0:
                p0 = pos[edges[:, 0]]
                p1 = pos[edges[:, 1]]
                cur_lens = np.linalg.norm(p0 - p1, axis=1)
                strains = np.abs(cur_lens - rest_lens) / (rest_lens + 1e-7)

                max_strain = float(np.max(strains))
                max_stretch_overall = max(max_stretch_overall, max_strain)
                sum_stretch_overall += float(np.sum(strains))
                total_edge_samples += len(strains)

                over = np.sum(strains > cls.MAX_ALLOWABLE_STRETCH_RATIO)
                overstretched_edges = max(overstretched_edges, int(over))

            # Frame metrics
            f_meta = cache.frame_metrics.get(frame_num, {})
            events = f_meta.get("contact_events", 0)
            depth = f_meta.get("max_depth", 0.0)
            if events > 0:
                frames_pen_count += 1
                total_pen_events += events
                max_pen_depth = max(max_pen_depth, depth)

        mean_stretch = (
            sum_stretch_overall / float(total_edge_samples) if total_edge_samples > 0 else 0.0
        )

        # 4. Scoring Algorithm (starts at 100)
        score = 100.0

        if not is_stable:
            score = 10.0
            status = QualityStatus.CRITICAL_FAILURE
            recommendations.append("Increase solver substeps and add velocity damping to stabilize.")
        else:
            # Penetration penalty
            if max_pen_depth > cls.MAX_ALLOWABLE_PENETRATION_DEPTH:
                pen_penalty = min(35.0, (max_pen_depth - cls.MAX_ALLOWABLE_PENETRATION_DEPTH) * 1500.0)
                score -= pen_penalty
                issues.append(f"Excessive penetration depth: {max_pen_depth*1000.0:.1f}mm (limit: {cls.MAX_ALLOWABLE_PENETRATION_DEPTH*1000.0:.1f}mm).")
                recommendations.append("Expand collider thickness margin and enable pre-simulation relaxation.")

            # Stretch penalty
            if max_stretch_overall > cls.MAX_ALLOWABLE_STRETCH_RATIO:
                stretch_penalty = min(30.0, (max_stretch_overall - cls.MAX_ALLOWABLE_STRETCH_RATIO) * 100.0)
                score -= stretch_penalty
                issues.append(f"Cloth overstretching: {max_stretch_overall*100.0:.1f}% strain detected.")
                recommendations.append("Decrease stretch compliance (increase stiffness) and raise solver iterations.")

            # Frequency of contact penalty
            if total_pen_events > 50:
                score -= min(15.0, (total_pen_events - 50) * 0.1)

            score = max(0.0, min(100.0, score))

            if score >= 85.0:
                status = QualityStatus.APPROVED
            else:
                status = QualityStatus.NEEDS_REPAIR

        report = QualityMetricsReport(
            status=status,
            score=round(score, 1),
            total_frames=total_frames,
            frames_with_penetration=frames_pen_count,
            max_penetration_depth=round(max_pen_depth, 4),
            total_penetration_events=total_pen_events,
            max_stretch_ratio=round(max_stretch_overall, 3),
            mean_stretch_ratio=round(mean_stretch, 4),
            overstretched_edge_count=overstretched_edges,
            peak_velocity=round(peak_velocity, 2),
            is_numerically_stable=is_stable,
            issues=issues,
            recommendations=recommendations,
        )

        logger.info(
            f"Simulation evaluated: Status={report.status.value}, Score={report.score}/100, "
            f"PenDepth={report.max_penetration_depth*1000:.1f}mm, MaxStrain={report.max_stretch_ratio*100:.1f}%"
        )
        return report
