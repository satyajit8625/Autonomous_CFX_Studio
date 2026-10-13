"""Autonomous CFX Studio - Simulation Quality Evaluator.
Performs automated multi-metric diagnostics on simulation caches:
strictly separates physical collision margins from true penetration tolerances,
reports raw penetration depth measurements and vertex indices,
and keeps automated quality recommendations separate from human artist approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.logger import logger


class QualityStatus(str, Enum):
    PASS_RECOMMENDED = "PASS_RECOMMENDED"
    NEEDS_REPAIR = "NEEDS_REPAIR"
    UNRESOLVED_FAILED = "UNRESOLVED_FAILED"
    # Legacy alias for backward compatibility
    APPROVED = "PASS_RECOMMENDED"
    CRITICAL_FAILURE = "UNRESOLVED_FAILED"


class HumanApprovalStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    ARTIST_APPROVED = "ARTIST_APPROVED"
    ARTIST_REJECTED = "ARTIST_REJECTED"


@dataclass
class QualityMetricsReport:
    """Comprehensive production CFX diagnostic report."""
    status: QualityStatus
    score: float  # 0.0 to 100.0
    total_frames: int

    # Raw physical penetration telemetry (distinct from collision margin)
    raw_max_penetration_depth_mm: float
    raw_penetrated_vertex_count: int
    raw_penetrated_vertex_ids: List[int]
    raw_frames_with_true_penetration: int
    margin_contact_events: int

    # Legacy fields
    frames_with_penetration: int
    max_penetration_depth: float  # meters
    total_penetration_events: int

    # Strain & Stability
    max_stretch_ratio: float
    mean_stretch_ratio: float
    overstretched_edge_count: int
    peak_velocity: float
    is_numerically_stable: bool

    # Human sign-off authority (kept strictly independent of automated checks)
    human_approval: HumanApprovalStatus = HumanApprovalStatus.PENDING_REVIEW
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[str] = None

    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class SimulationEvaluator:
    """Evaluates simulation results against physical realism and production CFX standards."""

    # Tolerances
    TRUE_PENETRATION_TOLERANCE_M = 0.001   # 1.0mm allowable interior penetration
    MAX_ALLOWABLE_PENETRATION_DEPTH = 0.012 # 12mm allowable margin intrusion buffer
    MAX_ALLOWABLE_STRETCH_RATIO = 0.15     # 15% strain
    VELOCITY_EXPLOSION_THRESHOLD = 30.0    # 30 m/s

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
                status=QualityStatus.UNRESOLVED_FAILED,
                score=0.0,
                total_frames=0,
                raw_max_penetration_depth_mm=0.0,
                raw_penetrated_vertex_count=0,
                raw_penetrated_vertex_ids=[],
                raw_frames_with_true_penetration=0,
                margin_contact_events=0,
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

        # 3. Raw Penetration Aggregation
        total_margin_contacts = 0
        max_margin_depth = 0.0
        margin_pen_frames = 0

        raw_true_pen_count = 0
        raw_max_true_pen_depth = 0.0
        raw_pen_frames = 0
        all_pen_vert_ids: Set[int] = set()

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

            # Margin contacts
            c_events = f_meta.get("contact_events", 0)
            m_depth = f_meta.get("max_margin_depth", f_meta.get("max_depth", 0.0))
            if c_events > 0 or m_depth > 0:
                margin_pen_frames += 1
                total_margin_contacts += c_events
                max_margin_depth = max(max_margin_depth, m_depth)

            # Raw true penetrations (interior to collider body)
            true_pens = f_meta.get("raw_true_penetrations", 0)
            raw_depth = f_meta.get("raw_max_penetration_depth", 0.0)
            pen_ids = f_meta.get("raw_penetrated_vertex_ids", [])
            if true_pens > 0 or raw_depth > 0:
                raw_pen_frames += 1
                raw_true_pen_count += true_pens
                raw_max_true_pen_depth = max(raw_max_true_pen_depth, raw_depth)
                all_pen_vert_ids.update(pen_ids)

        mean_stretch = (
            sum_stretch_overall / float(total_edge_samples) if total_edge_samples > 0 else 0.0
        )

        # 4. Strict Production Quality Scoring
        score = 100.0

        if not is_stable:
            score = 10.0
            status = QualityStatus.UNRESOLVED_FAILED
            recommendations.append("Increase solver substeps and add velocity damping to stabilize.")
        else:
            # A. True Penetration Penalty (Strict zero-tolerance in production)
            if raw_max_true_pen_depth > cls.TRUE_PENETRATION_TOLERANCE_M:
                # Severe penalty for actual collider intersection
                true_pen_penalty = min(50.0, 20.0 + (raw_max_true_pen_depth * 1000.0) * 2.0)
                score -= true_pen_penalty
                issues.append(
                    f"True collider penetration detected: {raw_max_true_pen_depth*1000.0:.2f}mm "
                    f"on {len(all_pen_vert_ids)} vertices (limit: {cls.TRUE_PENETRATION_TOLERANCE_M*1000.0:.1f}mm)."
                )
                recommendations.append("Expand collision margin and increase solver substeps to resolve body intrusion.")
            elif max_margin_depth > cls.MAX_ALLOWABLE_PENETRATION_DEPTH:
                # Minor penalty for deep margin intrusion without true penetration
                score -= min(15.0, (max_margin_depth - cls.MAX_ALLOWABLE_PENETRATION_DEPTH) * 1000.0)

            # B. Stretch Distortion Penalty
            if max_stretch_overall > cls.MAX_ALLOWABLE_STRETCH_RATIO:
                stretch_penalty = min(30.0, (max_stretch_overall - cls.MAX_ALLOWABLE_STRETCH_RATIO) * 100.0)
                score -= stretch_penalty
                issues.append(f"Cloth overstretching: {max_stretch_overall*100.0:.1f}% strain detected.")
                recommendations.append("Decrease stretch compliance (increase stiffness) and raise solver iterations.")

            score = max(0.0, min(100.0, score))

            if score >= 85.0 and raw_max_true_pen_depth <= cls.TRUE_PENETRATION_TOLERANCE_M:
                status = QualityStatus.PASS_RECOMMENDED
            else:
                status = QualityStatus.NEEDS_REPAIR

        report = QualityMetricsReport(
            status=status,
            score=round(score, 1),
            total_frames=total_frames,
            raw_max_penetration_depth_mm=round(raw_max_true_pen_depth * 1000.0, 3),
            raw_penetrated_vertex_count=len(all_pen_vert_ids),
            raw_penetrated_vertex_ids=sorted(list(all_pen_vert_ids)),
            raw_frames_with_true_penetration=raw_pen_frames,
            margin_contact_events=total_margin_contacts,
            frames_with_penetration=margin_pen_frames,
            max_penetration_depth=round(max_margin_depth, 4),
            total_penetration_events=total_margin_contacts,
            max_stretch_ratio=round(max_stretch_overall, 3),
            mean_stretch_ratio=round(mean_stretch, 4),
            overstretched_edge_count=overstretched_edges,
            peak_velocity=round(peak_velocity, 2),
            is_numerically_stable=is_stable,
            human_approval=HumanApprovalStatus.PENDING_REVIEW,
            issues=issues,
            recommendations=recommendations,
        )

        logger.info(
            f"Quality Evaluated: RecStatus={report.status.value}, Score={report.score}/100, "
            f"RawPen={report.raw_max_penetration_depth_mm:.2f}mm ({report.raw_penetrated_vertex_count} verts), "
            f"MaxStrain={report.max_stretch_ratio*100:.1f}%, HumanStatus={report.human_approval.value}"
        )
        return report
