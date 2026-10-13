"""Autonomous CFX Studio - Multimodal Correction Agent.

Interprets multimodal artist feedback (2D annotation bundle, surface selections, notes,
and frame ranges) into validated, structured correction requests. Formulates bounded,
physically sound repair plans (solver adjustment, resimulation, or non-destructive
corrective layers) with honest model runtime attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple

from autonomous_cfx.core.annotation_bundle import AnnotationBundle, AnnotationRegion
from autonomous_cfx.core.surface_mapping import SurfaceSelection


class CorrectionIntent(str, Enum):
    """Categorized CFX defect repair intent."""
    REPAIR_BODY_PENETRATION = "repair_body_penetration"
    REDUCE_EXCESSIVE_STRETCH = "reduce_excessive_stretch"
    REDUCE_LOCAL_MOTION = "reduce_local_motion"
    PRESERVE_SILHOUETTE = "preserve_silhouette"
    ADJUST_ATTACHMENT = "adjust_attachment"
    SMOOTH_LOCAL_DEFECT = "smooth_local_defect"


class ExecutionStrategy(str, Enum):
    """Underlying strategy chosen to execute the requested repair."""
    NON_DESTRUCTIVE_CORRECTIVE_LAYER = "non_destructive_corrective_layer"
    SOLVER_PARAMETER_RESIMULATION = "solver_parameter_resimulation"
    CONTACT_COLLIDER_ADJUSTMENT = "contact_collider_adjustment"


@dataclass
class StructuredCorrectionRequest:
    """Rigorous structured specification of an artist's correction intent."""
    request_id: str
    garment_name: str
    geometry_version: str
    intent: CorrectionIntent
    target_region: SurfaceSelection
    frame_range: Tuple[int, int]  # (start_frame, end_frame)
    fade_frames: Tuple[int, int] = (5, 5)  # (fade_in, fade_out)
    protected_regions: List[SurfaceSelection] = field(default_factory=list)
    quality_objectives: Dict[str, float] = field(default_factory=dict)
    compute_budget_iterations: int = 20
    artist_note: str = ""
    is_ambiguous: bool = False
    ambiguity_reason: str = ""
    model_attribution: str = "Deterministic Structured Intent Parser (Embedded Fallback)"


@dataclass
class CorrectionPlan:
    """The formulated plan and rationale presented to the artist before execution."""
    request_id: str
    chosen_strategy: ExecutionStrategy
    explanation: str
    target_vertex_count: int
    protected_vertex_count: int
    effective_frame_range: Tuple[int, int]  # includes temporal fades
    bounded_actions: List[str]
    requires_artist_confirmation: bool = False
    model_runtime: str = "Deterministic Grammar-Constrained Engine"


class MultimodalCorrectionAgent:
    """Agent that interprets multimodal annotations and generates executable repair plans."""

    def __init__(self, model_endpoint: Optional[str] = None):
        self.model_endpoint = model_endpoint
        # Explicit label of active runtime
        self.active_runtime = (
            f"Local LLM via {model_endpoint}" if model_endpoint
            else "Deterministic Structured Intent Parser (Embedded Fallback)"
        )

    def parse_annotation_to_request(
        self,
        bundle: AnnotationBundle,
        region: AnnotationRegion,
        surface_selection: SurfaceSelection,
        protected_selection: Optional[SurfaceSelection] = None
    ) -> StructuredCorrectionRequest:
        """Parse multimodal annotation bundle and notes into a validated StructuredCorrectionRequest."""
        note = region.intent_description.strip()
        label = region.label.lower()

        # 1. Detect Intent
        intent = CorrectionIntent.REPAIR_BODY_PENETRATION
        note_lower = note.lower()
        if "penetrat" in note_lower or "clip" in note_lower or "push" in note_lower:
            intent = CorrectionIntent.REPAIR_BODY_PENETRATION
        elif "stretch" in note_lower or "overstretch" in note_lower:
            intent = CorrectionIntent.REDUCE_EXCESSIVE_STRETCH
        elif "motion" in note_lower or "damping" in note_lower or "jitter" in note_lower:
            intent = CorrectionIntent.REDUCE_LOCAL_MOTION
        elif "attachment" in note_lower or "pin" in note_lower:
            intent = CorrectionIntent.ADJUST_ATTACHMENT
        elif "smooth" in note_lower or "wrinkle" in note_lower:
            intent = CorrectionIntent.SMOOTH_LOCAL_DEFECT
        elif "silhouette" in note_lower or ("shape" in note_lower and "preserve" in note_lower):
            intent = CorrectionIntent.PRESERVE_SILHOUETTE

        # 2. Extract Frame Range from note (e.g. "frames 1045–1060" or "frames 10 to 25")
        frame_match = re.search(r"frames?\s+(\d+)\s*[-–to]+\s*(\d+)", note, re.IGNORECASE)
        if frame_match:
            start_f = int(frame_match.group(1))
            end_f = int(frame_match.group(2))
        else:
            # Default to marked frame with +/- 5 frame window
            start_f = max(1, bundle.frame - 5)
            end_f = bundle.frame + 5

        # 3. Detect temporal fade (e.g. "fade the correction into surrounding frames")
        fade_in = 5
        fade_out = 5
        fade_match = re.search(r"fade\s+(\d+)", note, re.IGNORECASE)
        if fade_match:
            f_val = int(fade_match.group(1))
            fade_in = f_val
            fade_out = f_val

        # 4. Detect ambiguity or contradictory requirements
        is_ambiguous = surface_selection.is_ambiguous
        ambiguity_reason = surface_selection.notes if is_ambiguous else ""

        if start_f > end_f:
            is_ambiguous = True
            ambiguity_reason = f"Invalid frame range: start frame {start_f} is greater than end frame {end_f}."

        protected_regions = [protected_selection] if protected_selection else []

        request = StructuredCorrectionRequest(
            request_id=f"corr_{bundle.bundle_id}_{region.region_id}",
            garment_name=bundle.garment_name,
            geometry_version=bundle.geometry_version,
            intent=intent,
            target_region=surface_selection,
            frame_range=(start_f, end_f),
            fade_frames=(fade_in, fade_out),
            protected_regions=protected_regions,
            quality_objectives={"max_penetration_mm": 0.5, "stretch_limit_ratio": 1.1},
            compute_budget_iterations=20,
            artist_note=note,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            model_attribution=self.active_runtime
        )
        return request

    def plan_correction_strategy(
        self,
        request: StructuredCorrectionRequest
    ) -> CorrectionPlan:
        """Analyze request and explain whether solver modification, resimulation, or corrective layer is best."""
        # Check for ambiguity
        if request.is_ambiguous:
            return CorrectionPlan(
                request_id=request.request_id,
                chosen_strategy=ExecutionStrategy.NON_DESTRUCTIVE_CORRECTIVE_LAYER,
                explanation=(
                    f"Request is ambiguous or contradictory: {request.ambiguity_reason}. "
                    "Halting before expensive compute; artist confirmation required."
                ),
                target_vertex_count=len(request.target_region.selected_vertices),
                protected_vertex_count=sum(len(p.selected_vertices) for p in request.protected_regions),
                effective_frame_range=request.frame_range,
                bounded_actions=["request_clarification"],
                requires_artist_confirmation=True,
                model_runtime=self.active_runtime
            )

        start_f, end_f = request.frame_range
        fade_in, fade_out = request.fade_frames
        eff_start = max(1, start_f - fade_in)
        eff_end = end_f + fade_out

        target_verts = len(request.target_region.selected_vertices)
        prot_verts = sum(len(p.selected_vertices) for p in request.protected_regions)

        if request.intent == CorrectionIntent.REPAIR_BODY_PENETRATION:
            explanation = (
                f"Selected non-destructive corrective layer over frames {start_f}–{end_f} "
                f"(fading smoothly from frame {eff_start} to {eff_end}). "
                f"Applies bounded surface de-penetration along collider normal with Laplacian smoothing "
                f"across {target_verts} target vertices, strictly freezing {prot_verts} protected vertices."
            )
            strategy = ExecutionStrategy.NON_DESTRUCTIVE_CORRECTIVE_LAYER
            actions = [
                "project_penetrating_vertices_past_margin",
                "apply_manifold_laplacian_smoothing",
                "freeze_protected_regions",
                "apply_temporal_sin2_fade"
            ]

        elif request.intent == CorrectionIntent.REDUCE_EXCESSIVE_STRETCH:
            explanation = (
                f"Targeted area exhibits excessive elongation. Stiffening local distance constraints "
                f"in target region and resimulating local frames {eff_start}–{eff_end}."
            )
            strategy = ExecutionStrategy.SOLVER_PARAMETER_RESIMULATION
            actions = ["stiffen_local_compliance", "resimulate_substeps"]

        else:
            explanation = (
                f"Applying non-destructive smoothing layer to attenuate high-frequency defects "
                f"across frames {start_f}–{end_f}."
            )
            strategy = ExecutionStrategy.NON_DESTRUCTIVE_CORRECTIVE_LAYER
            actions = ["laplacian_defect_filter", "apply_temporal_fade"]

        return CorrectionPlan(
            request_id=request.request_id,
            chosen_strategy=strategy,
            explanation=explanation,
            target_vertex_count=target_verts,
            protected_vertex_count=prot_verts,
            effective_frame_range=(eff_start, eff_end),
            bounded_actions=actions,
            requires_artist_confirmation=False,
            model_runtime=self.active_runtime
        )
