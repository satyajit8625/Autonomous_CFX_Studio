"""Autonomous CFX Studio - Bounded Self-Healing Repair System.
Diagnoses quality evaluation failures and applies targeted, bounded physical repair recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from autonomous_cfx.agent.evaluator import QualityMetricsReport, QualityStatus, SimulationEvaluator
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
    ) -> Tuple[SimulationCache, QualityMetricsReport, List[RepairRecord]]:
        """Executes bounded repair loop until quality passes or bound is reached."""
        self.repair_history = []
        active_props = ClothMaterialProperties(**properties.__dict__)
        last_report = current_report
        best_cache: Optional[SimulationCache] = current_cache

        attempt = 1
        while attempt <= self.max_attempts and last_report.status != QualityStatus.APPROVED:
            diagnosis: List[str] = []
            adjustments: List[str] = []

            # 1. Diagnose and formulate recipe
            if not last_report.is_numerically_stable:
                diagnosis.append("Numerical instability / velocity explosion detected.")
                active_props.solver_substeps = int(active_props.solver_substeps * 1.5)
                active_props.damping = min(0.08, active_props.damping * 1.8)
                adjustments.append(f"Elevated substeps to {active_props.solver_substeps} and damping to {active_props.damping:.3f}.")

            if last_report.max_penetration_depth > SimulationEvaluator.MAX_ALLOWABLE_PENETRATION_DEPTH or last_report.frames_with_penetration > 0:
                diagnosis.append(f"Collider penetration detected (max depth {last_report.max_penetration_depth*1000.0:.1f}mm).")
                active_props.collision_thickness = round(active_props.collision_thickness * 1.25 + 0.003, 4)
                active_props.solver_substeps = max(active_props.solver_substeps, 18)
                adjustments.append(f"Expanded collision thickness to {active_props.collision_thickness*1000.0:.1f}mm and substeps to {active_props.solver_substeps}.")

            if last_report.max_stretch_ratio > SimulationEvaluator.MAX_ALLOWABLE_STRETCH_RATIO:
                diagnosis.append(f"Excessive cloth strain ({last_report.max_stretch_ratio*100.0:.1f}%).")
                active_props.stretch_compliance = max(1e-7, active_props.stretch_compliance * 0.4)
                active_props.solver_iterations += 2
                adjustments.append(f"Tightened stretch compliance to {active_props.stretch_compliance:.2e} and increased solver iterations to {active_props.solver_iterations}.")

            if not adjustments:
                # Minor default polish if score is marginally below threshold
                active_props.solver_iterations += 1
                adjustments.append("Increased solver iterations by 1 for convergence polish.")

            logger.info(
                f"[Bounded Repair Attempt {attempt}/{self.max_attempts}] "
                f"Applying recipe: {', '.join(adjustments)}"
            )

            # 2. Re-run simulation with adjusted parameters
            engine = SimulationEngine(
                cloth_geometry=cloth_geometry,
                properties=active_props,
                colliders=colliders,
                pinned_vertices=pinned_vertices,
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            new_cache = engine.run(progress_callback=progress_callback)
            best_cache = new_cache

            # 3. Re-evaluate
            new_report = SimulationEvaluator.evaluate(new_cache, cloth_geometry)

            is_resolved = new_report.status == QualityStatus.APPROVED
            record = RepairRecord(
                attempt_index=attempt,
                diagnosis=diagnosis,
                adjustments_applied=adjustments,
                pre_score=last_report.score,
                post_score=new_report.score,
                resolved=is_resolved,
            )
            self.repair_history.append(record)

            logger.info(
                f"[Repair Attempt {attempt} Result] Score changed from {last_report.score} -> {new_report.score} "
                f"({ 'PASSED' if is_resolved else 'ITERATING' })"
            )

            last_report = new_report
            if is_resolved:
                break

            attempt += 1

        return best_cache, last_report, self.repair_history
