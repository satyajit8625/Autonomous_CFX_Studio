"""Autonomous CFX Studio - Master Agent Orchestrator.
Coordinates the end-to-end milestone workflow:
Import -> Describe (Local LLM Intent Parsing) -> Prepare & Run -> Evaluate -> Bounded Repair -> Human Review -> Export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from autonomous_cfx.agent.evaluator import (
    HumanApprovalStatus,
    QualityMetricsReport,
    QualityStatus,
    SimulationEvaluator,
)
from autonomous_cfx.agent.behavior_parser import BehaviorParser
from autonomous_cfx.agent.llm_client import AgentMessage, LocalLLMClient, ToolCallRecord
from autonomous_cfx.agent.repair import BoundedRepairStrategist, RepairRecord
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.logger import logger
from autonomous_cfx.core.project import ProjectManager
from autonomous_cfx.core.usd_export import USDExporter
from autonomous_cfx.simulation.colliders import Collider
from autonomous_cfx.simulation.engine import SimulationEngine
from autonomous_cfx.simulation.properties import ClothMaterialProperties


@dataclass
class WorkflowExecutionResult:
    """Consolidated outcome of the autonomous CFX milestone execution."""
    success: bool
    initial_report: QualityMetricsReport
    final_report: QualityMetricsReport
    repair_history: List[RepairRecord]
    simulation_cache: SimulationCache
    conversation_history: List[AgentMessage] = field(default_factory=list)
    clarification_requested: Optional[Dict] = None

    # Exported lighting assets
    exported_usd_path: Optional[str] = None
    exported_cache_path: Optional[str] = None
    exported_obj_count: int = 0
    exported_manifest_path: Optional[str] = None

    # Independent human review status
    human_approval_status: HumanApprovalStatus = HumanApprovalStatus.PENDING_REVIEW
    behavior_parse: Optional[Any] = None

    @property
    def artist_approved(self) -> bool:
        return self.human_approval_status == HumanApprovalStatus.ARTIST_APPROVED

    @property
    def configured_material(self) -> Optional[str]:
        if self.behavior_parse and hasattr(self.behavior_parse, "detected_material"):
            return self.behavior_parse.detected_material
        for msg in self.conversation_history:
            for tc in msg.tool_calls:
                if tc.tool_name == "configure_simulation_parameters":
                    return tc.arguments.get("material_name")
        return None


class AutonomousCFXOrchestrator:
    """Master autonomous orchestrator managing the CFX lifecycle."""

    def __init__(
        self,
        project_manager: Optional[ProjectManager] = None,
        llm_backend: str = "embedded_local",
    ):
        self.project_manager = project_manager or ProjectManager(Path("."))
        self.repair_strategist = BoundedRepairStrategist(max_attempts=3)
        self.llm_client = LocalLLMClient(backend=llm_backend)

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
        human_approval: HumanApprovalStatus = HumanApprovalStatus.PENDING_REVIEW,
        auto_approve_if_passed: bool = False,
        export_on_approval: bool = False,
        progress_callback: Optional[Callable[[str, int, int, dict], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> WorkflowExecutionResult:
        """Execute complete milestone autonomous workflow."""
        if auto_approve_if_passed and human_approval == HumanApprovalStatus.PENDING_REVIEW:
            human_approval = HumanApprovalStatus.ARTIST_APPROVED
        logger.info(f"=== Starting Autonomous CFX Workflow: {cloth_geometry.name} ===")

        def notify(stage: str, cur: int, total: int, extra: dict = None):
            if cancel_check and cancel_check():
                return
            if progress_callback:
                progress_callback(stage, cur, total, extra or {})

        # 1. Local Language-Model Intent Parsing with Validated Tool Calling
        notify("Local LLM Intent Parsing", 0, 100, {"prompt": behavior_description})
        assistant_text, tool_calls = self.llm_client.generate_response(behavior_description)

        clarification = None
        props = ClothMaterialProperties()

        for tc in tool_calls:
            if tc.tool_name == "request_clarification":
                clarification = tc.arguments
                logger.warning(f"[Agent] Clarification requested: {clarification.get('clarification_question')}")
                return WorkflowExecutionResult(
                    success=False,
                    initial_report=QualityMetricsReport(
                        status=QualityStatus.NEEDS_REPAIR,
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
                        is_numerically_stable=True,
                        issues=[f"Clarification required: {clarification.get('clarification_question')}"],
                    ),
                    final_report=QualityMetricsReport(
                        status=QualityStatus.NEEDS_REPAIR,
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
                        is_numerically_stable=True,
                    ),
                    repair_history=[],
                    simulation_cache=SimulationCache(name=cloth_geometry.name),
                    conversation_history=self.llm_client.conversation_history,
                    clarification_requested=clarification,
                )
            elif tc.tool_name == "configure_simulation_parameters":
                args = tc.arguments
                props = ClothMaterialProperties(
                    name=args.get("material_name", "custom"),
                    density=args.get("density", 0.3),
                    stretch_compliance=args.get("stretch_compliance", 1e-5),
                    bend_compliance=args.get("bend_compliance", 1e-3),
                    damping=args.get("damping", 0.02),
                    friction=args.get("friction", 0.35),
                    collision_thickness=args.get("collision_margin", 0.010),
                    solver_substeps=args.get("solver_substeps", 16),
                    solver_iterations=args.get("solver_iterations", 4),
                )
                logger.info(f"[Agent] Configured physical properties via tool call: {args.get('reasoning')}")

        # 2. Prepare & Run Initial Simulation
        notify("Simulating Initial Run", 10, 100, {"material": props.name})
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

        sim_cache = engine.run(progress_callback=engine_progress, cancel_check=cancel_check)

        # 3. Evaluate Simulation Quality (Strictly Automated Recommendation)
        notify("Evaluating Quality", 55, 100)
        initial_report = SimulationEvaluator.evaluate(sim_cache, cloth_geometry)
        final_report = initial_report
        repair_records: List[RepairRecord] = []

        # 4. Attempt Bounded Self-Healing Repairs if needed
        if initial_report.status != QualityStatus.PASS_RECOMMENDED and not (cancel_check and cancel_check()):
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
                cancel_check=cancel_check,
            )
            sim_cache = repaired_cache
            final_report = repaired_report
            repair_records = records

        # 5. Distinct Human Review State
        final_report.human_approval = human_approval
        notify("Artist Review", 90, 100, {
            "score": final_report.score,
            "automated_status": final_report.status.value,
            "human_approval": final_report.human_approval.value,
        })

        # 6. Export Approved Lighting Caches (Only when artist approved or explicit flag)
        exported_usd_file = None
        exported_cache_file = None
        exported_obj_count = 0
        exported_manifest = None

        if (final_report.human_approval == HumanApprovalStatus.ARTIST_APPROVED or export_on_approval) and final_report.status == QualityStatus.PASS_RECOMMENDED:
            notify("Exporting Approved Caches", 95, 100)
            next_version = self.project_manager.get_next_cache_version(shot_name, cloth_geometry.name)
            cache_dir = self.project_manager.get_cache_version_dir(shot_name, cloth_geometry.name, next_version)

            # A. Export Lighting-Consumable OpenUSD Cache (.usda / .usd)
            usd_path = cache_dir / f"{cloth_geometry.name}_v{next_version:03d}.usda"
            exported_usd_file = USDExporter.export_cache(sim_cache, usd_path)

            # B. Save binary point cache (.vpcache)
            vpcache_path = cache_dir / f"{cloth_geometry.name}_v{next_version:03d}.vpcache"
            exported_cache_file = sim_cache.save_point_cache(vpcache_path)

            # C. Export OBJ Sequence
            obj_dir = cache_dir / "obj_sequence"
            obj_files = sim_cache.export_obj_sequence(obj_dir, prefix=cloth_geometry.name)
            exported_obj_count = len(obj_files)

            # D. Export Interchange Manifest
            exported_manifest = sim_cache.export_interchange_manifest(
                output_dir=cache_dir,
                alembic_equivalent_name=f"{cloth_geometry.name}_v{next_version:03d}.abc",
            )

        notify("Completed", 100, 100, {
            "automated_status": final_report.status.value,
            "human_approval": final_report.human_approval.value,
        })
        logger.info(
            f"=== Autonomous CFX Workflow Completed (Status: {final_report.status.value}, "
            f"HumanApproval: {final_report.human_approval.value}) ==="
        )

        return WorkflowExecutionResult(
            success=(final_report.status == QualityStatus.PASS_RECOMMENDED),
            initial_report=initial_report,
            final_report=final_report,
            repair_history=repair_records,
            simulation_cache=sim_cache,
            conversation_history=self.llm_client.conversation_history,
            clarification_requested=clarification,
            exported_usd_path=exported_usd_file,
            exported_cache_path=exported_cache_file,
            exported_obj_count=exported_obj_count,
            exported_manifest_path=exported_manifest,
            human_approval_status=final_report.human_approval,
            behavior_parse=BehaviorParser.parse(behavior_description),
        )
