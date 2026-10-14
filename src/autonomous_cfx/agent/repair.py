"""Autonomous CFX Studio - Bounded Self-Healing Repair System.
Diagnoses quality evaluation failures, applies targeted physical repair recipes,
and stops honestly when bounded limits or geometric rigging conflicts are reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from autonomous_cfx.agent.evaluator import QualityMetricsReport, QualityStatus, SimulationEvaluator
from autonomous_cfx.agent.tools import FailureMode, validate_tool_call
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.colliders import Collider
from autonomous_cfx.simulation.engine import SimulationEngine
from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.core.logger import logger


@dataclass
class RepairRecord:
    """Historical trace of a single bounded repair cycle."""
    attempt_index: int
    diagnosis: List[str]
    adjustments_applied: List[str]
    pre_score: float
    post_score: float
    resolved: bool
    failure_mode: Optional[str] = None


class BoundedRepairStrategist:
    """Bounded automated diagnosis and self-healing engine for cloth simulations."""

    DEFAULT_MAX_ATTEMPTS = 3

    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        self.max_attempts = max_attempts
        self.repair_history: List[RepairRecord] = []

    def repair_and_rerun(
        self,
        current_report: QualityMetricsReport,
        cloth_geometry: MeshGeometry,
        properties: ClothMaterialProperties,
        colliders: List[Collider],
        pinned_vertices: List[int],
        fps: float,
        start_frame: int,
        end_frame: int,
        current_cache: Optional[SimulationCache] = None,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[SimulationCache, QualityMetricsReport, List[RepairRecord]]:
        """Executes bounded repair loop until quality passes or bound is reached."""
        self.repair_history = []
        active_props = ClothMaterialProperties(**properties.__dict__)
        last_report = current_report
        best_cache: Optional[SimulationCache] = current_cache

        # If already passing, return immediately
        if last_report.status == QualityStatus.PASS_RECOMMENDED:
            return best_cache, last_report, self.repair_history

        attempt = 1
        while attempt <= self.max_attempts and last_report.status != QualityStatus.PASS_RECOMMENDED:
            diagnosis: List[str] = []
            adjustments: List[str] = []
            f_mode = FailureMode.COLLIDER_TUNNELING.value

            # Check for Pinned Vertex Rigging Conflict
            pinned_conflicts = set(pinned_vertices).intersection(set(last_report.raw_penetrated_vertex_ids))
            if pinned_conflicts:
                f_mode = FailureMode.PINNED_COLLIDER_CONFLICT.value
                diagnosis.append(
                    f"Geometric constraint conflict: Pinned vertices {list(pinned_conflicts)} "
                    f"are embedded inside the collider surface. Solver cannot de-penetrate infinite-mass vertices."
                )

            # Check for Numerical Instability
            if not last_report.is_numerically_stable:
                f_mode = FailureMode.NUMERICAL_EXPLOSION.value
                diagnosis.append("Numerical instability / velocity explosion detected.")
                active_props.solver_substeps = int(active_props.solver_substeps * 1.5)
                active_props.damping = min(0.08, active_props.damping * 1.8)
                adjustments.append(f"Elevated substeps to {active_props.solver_substeps} and damping to {active_props.damping:.3f}.")

            # Check for True Penetration / Tunneling
            if last_report.raw_max_penetration_depth_mm > SimulationEvaluator.TRUE_PENETRATION_TOLERANCE_M * 1000.0 or last_report.raw_penetrated_vertex_count > 0:
                if not pinned_conflicts:
                    f_mode = FailureMode.COLLIDER_TUNNELING.value
                diagnosis.append(
                    f"True collider penetration detected: {last_report.raw_max_penetration_depth_mm:.2f}mm "
                    f"across {last_report.raw_penetrated_vertex_count} vertices."
                )
                active_props.collision_thickness = round(active_props.collision_thickness + 0.004, 4)
                active_props.solver_substeps = max(active_props.solver_substeps + 6, 20)
                adjustments.append(
                    f"Expanded collision margin to {active_props.collision_thickness*1000.0:.1f}mm "
                    f"and increased substeps to {active_props.solver_substeps}."
                )

            # Check for Overstretching
            if last_report.max_stretch_ratio > SimulationEvaluator.MAX_ALLOWABLE_STRETCH_RATIO:
                f_mode = FailureMode.OVERSTRETCHING.value
                diagnosis.append(f"Excessive cloth strain ({last_report.max_stretch_ratio*100.0:.1f}%).")
                active_props.stretch_compliance = max(1e-7, active_props.stretch_compliance * 0.4)
                active_props.solver_iterations += 2
                adjustments.append(
                    f"Tightened stretch compliance to {active_props.stretch_compliance:.2e} "
                    f"and raised solver iterations to {active_props.solver_iterations}."
                )

            if not adjustments:
                active_props.solver_iterations += 1
                adjustments.append("Increased solver iterations by 1 for convergence polish.")

            # Validate tool schema for diagnostic record
            diag_args = {
                "failure_mode": f_mode,
                "raw_penetration_depth_mm": float(last_report.raw_max_penetration_depth_mm),
                "violating_frames": list(range(1, min(10, end_frame + 1))),
                "root_cause": "; ".join(diagnosis),
                "recommended_repair_strategy": "; ".join(adjustments),
            }
            try:
                validate_tool_call("diagnose_simulation_failure", diag_args)
            except Exception as e_v:
                logger.debug(f"Diagnosis schema validation notice: {e_v}")

            if cancel_check and cancel_check():
                logger.warning(f"[Bounded Repair] Repair loop cancelled at attempt {attempt}.")
                break

            logger.info(
                f"[Bounded Repair Attempt {attempt}/{self.max_attempts}] Mode={f_mode}. "
                f"Applying recipe: {', '.join(adjustments)}"
            )

            # 2. Re-run simulation
            engine = SimulationEngine(
                cloth_geometry=cloth_geometry,
                properties=active_props,
                colliders=colliders,
                pinned_vertices=pinned_vertices,
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            new_cache = engine.run(progress_callback=progress_callback, cancel_check=cancel_check)
            best_cache = new_cache

            # 3. Re-evaluate
            new_report = SimulationEvaluator.evaluate(new_cache, cloth_geometry)
            is_resolved = new_report.status == QualityStatus.PASS_RECOMMENDED

            record = RepairRecord(
                attempt_index=attempt,
                diagnosis=diagnosis,
                adjustments_applied=adjustments,
                pre_score=last_report.score,
                post_score=new_report.score,
                resolved=is_resolved,
                failure_mode=f_mode,
            )
            self.repair_history.append(record)

            logger.info(
                f"[Repair Attempt {attempt} Result] Score: {last_report.score} -> {new_report.score} "
                f"(RawPen: {new_report.raw_max_penetration_depth_mm:.2f}mm, Status: {new_report.status.value})"
            )

            last_report = new_report
            if is_resolved:
                break

            attempt += 1

        # Honest Stop when repairs cannot resolve the defect
        if last_report.status != QualityStatus.PASS_RECOMMENDED:
            last_report.status = QualityStatus.UNRESOLVED_FAILED
            last_report.issues.append(
                f"Autonomous repair bound reached ({self.max_attempts}/{self.max_attempts} attempts). "
                "The simulation cannot be cleared automatically due to persistent physical/geometric conflicts."
            )
            last_report.recommendations.append(
                "Escalating to artist: manual intervention required on character collider animation or tailoring."
            )
            logger.warning(
                f"[Bounded Repair] Halting honestly: {self.max_attempts} attempts exhausted without meeting tolerances."
            )

        return best_cache, last_report, self.repair_history
