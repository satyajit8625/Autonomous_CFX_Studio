"""Autonomous CFX Studio - Master Agent Orchestrator.
Coordinates the end-to-end milestone workflow:
Import -> Describe Behavior -> Prepare & Run -> Evaluate -> Bounded Repair -> Review -> Export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from autonomous_cfx.agent.behavior_parser import BehaviorParser, BehaviorParseResult
from autonomous_cfx.agent.evaluator import QualityMetricsReport, QualityStatus, SimulationEvaluator
from autonomous_cfx.agent.repair import BoundedRepairStrategist, RepairRecord
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.logger import logger
from autonomous_cfx.core.project import ProjectManager
from autonomous_cfx.simulation.colliders import Collider
from autonomous_cfx.simulation.engine import SimulationEngine
from autonomous_cfx.simulation.properties import ClothMaterialProperties


@dataclass
class WorkflowExecutionResult:
    """Consolidated outcome of the autonomous CFX milestone execution."""
    success: bool
    behavior_parse: BehaviorParseResult
    initial_report: QualityMetricsReport
    final_report: QualityMetricsReport
    repair_history: List[RepairRecord]
    simulation_cache: SimulationCache
    exported_cache_path: Optional[str] = None
    exported_obj_count: int = 0
    exported_manifest_path: Optional[str] = None
    artist_approved: bool = False


class AutonomousCFXOrchestrator:
    """Master autonomous orchestrator managing the CFX lifecycle."""

    def __init__(self, project_manager: Optional[ProjectManager] = None):
        self.project_manager = project_manager or ProjectManager(Path("."))
        self.repair_strategist = BoundedRepairStrategist(max_attempts=3)

    def execute_workflow(
        self,
        cloth_geometry: MeshGeometry,
        colliders: List[Collider],
        behavior_description: str,
        pinned_vertices: Optional[List[int]] = None,
        shot_name: str = "shot010",
        fps: float = 24.0,
        start_frame: int = 1,
        end_frame: int = 40,
        auto_approve_if_passed: bool = True,
        export_on_approval: bool = True,
        progress_callback: Optional[Callable[[str, int, int, dict], None]] = None,
    ) -> WorkflowExecutionResult:
        """Execute complete milestone autonomous workflow."""
        logger.info(f"=== Starting Autonomous CFX Workflow: {cloth_geometry.name} ===")

        # 1. Parse behavior description
        def notify(stage: str, cur: int, total: int, extra: dict = None):
            if progress_callback:
                progress_callback(stage, cur, total, extra or {})

        notify("Parsing Behavior", 0, 100, {"prompt": behavior_description})
        parse_result = BehaviorParser.parse(behavior_description)
        props = parse_result.properties

        # 2. Prepare & Run Initial Simulation
        notify("Simulating Initial Run", 10, 100, {"material": parse_result.detected_material})
        engine = SimulationEngine(
            cloth_geometry=cloth_geometry,
            properties=props,
            colliders=colliders,
            pinned_vertices=pinned_vertices or [],
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        def engine_progress(curr, total, metrics):
            pct = 10 + int((curr / float(total)) * 40)
            notify("Simulating Initial Run", pct, 100, metrics)

        sim_cache = engine.run(progress_callback=engine_progress)

        # 3. Evaluate Simulation Quality
        notify("Evaluating Quality", 55, 100)
        initial_report = SimulationEvaluator.evaluate(sim_cache, cloth_geometry)
        final_report = initial_report
        repair_records: List[RepairRecord] = []

        # 4. Attempt Bounded Self-Healing Repairs if needed
        if initial_report.status != QualityStatus.APPROVED:
            notify("Executing Bounded Repairs", 60, 100, {"pre_score": initial_report.score})
            logger.info("Simulation did not meet quality threshold. Initiating bounded repairs...")

            def repair_progress(curr, total, metrics):
                pct = 60 + int((curr / float(total)) * 25)
                notify("Repairing Simulation", pct, 100, metrics)

            repaired_cache, repaired_report, records = self.repair_strategist.repair_and_rerun(
                current_report=initial_report,
                cloth_geometry=cloth_geometry,
                properties=props,
                colliders=colliders,
                pinned_vertices=pinned_vertices or [],
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                current_cache=sim_cache,
                progress_callback=repair_progress,
            )
            sim_cache = repaired_cache
            final_report = repaired_report
            repair_records = records

        # 5. Artist Review & Approval
        notify("Artist Review", 90, 100, {"score": final_report.score, "status": final_report.status.value})
        is_approved = False
        if final_report.status == QualityStatus.APPROVED and auto_approve_if_passed:
            is_approved = True
            logger.info("Quality criteria passed automatically: Artist approved cache.")

        # 6. Export Approved Caches
        exported_cache_file = None
        exported_obj_count = 0
        exported_manifest = None

        if is_approved and export_on_approval:
            notify("Exporting Approved Caches", 95, 100)
            next_version = self.project_manager.get_next_cache_version(shot_name, cloth_geometry.name)
            cache_dir = self.project_manager.get_cache_version_dir(shot_name, cloth_geometry.name, next_version)

            # A. Save point cache (.vpcache)
            vpcache_path = cache_dir / f"{cloth_geometry.name}_v{next_version:03d}.vpcache"
            exported_cache_file = sim_cache.save_point_cache(vpcache_path)

            # B. Export OBJ Sequence
            obj_dir = cache_dir / "obj_sequence"
            obj_files = sim_cache.export_obj_sequence(obj_dir, prefix=cloth_geometry.name)
            exported_obj_count = len(obj_files)

            # C. Export Interchange Manifest (Alembic/USD readiness)
            exported_manifest = sim_cache.export_interchange_manifest(
                output_dir=cache_dir,
                alembic_equivalent_name=f"{cloth_geometry.name}_v{next_version:03d}.abc",
            )

        notify("Completed", 100, 100, {"approved": is_approved})
        logger.info(f"=== Autonomous CFX Workflow Completed (Approved: {is_approved}) ===")

        return WorkflowExecutionResult(
            success=(final_report.status == QualityStatus.APPROVED),
            behavior_parse=parse_result,
            initial_report=initial_report,
            final_report=final_report,
            repair_history=repair_records,
            simulation_cache=sim_cache,
            exported_cache_path=exported_cache_file,
            exported_obj_count=exported_obj_count,
            exported_manifest_path=exported_manifest,
            artist_approved=is_approved,
        )
