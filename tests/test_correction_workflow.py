"""Tests for multimodal correction agent, temporal consistency, and non-destructive corrective layers."""

import numpy as np
import pytest

from autonomous_cfx.agent.correction_agent import (
    CorrectionIntent,
    ExecutionStrategy,
    MultimodalCorrectionAgent,
)
from autonomous_cfx.assets.procedural import (
    generate_procedural_animated_body,
    generate_procedural_jacket,
)
from autonomous_cfx.core.annotation_bundle import (
    AnnotationBundle,
    AnnotationRegion,
    CameraParameters,
)
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.surface_mapping import SurfaceSelection
from autonomous_cfx.simulation.corrective_layer import CorrectiveLayerEngine


def test_multimodal_intent_parsing_and_planning():
    """Verify natural-language note parsing into structured intent, frames, and bounded actions."""
    agent = MultimodalCorrectionAgent()

    bundle = AnnotationBundle(
        bundle_id="b_001",
        shot_name="shot010",
        garment_name="hero_jacket",
        geometry_version="v001",
        frame=1050,
        time_seconds=43.75,
        camera=CameraParameters(
            yaw=30.0, pitch=15.0, distance=3.0, target=(0.0, 1.2, 0.0),
            fov=1.8, width=512, height=512, view_matrix=[], projection_matrix=[]
        )
    )

    region = AnnotationRegion(
        region_id="r_01",
        label="Left Sleeve",
        intent_description=(
            "Remove body penetration here during frames 1045–1060. "
            "Preserve the cuff shape and existing folds. Fade the correction into surrounding frames."
        ),
        drawing_tool="brush"
    )

    # Sleeve selection
    target_selection = SurfaceSelection(
        selected_faces=[10, 11, 12],
        selected_vertices=[50, 51, 52, 53, 54],
        vertex_weights={50: 1.0, 51: 1.0, 52: 0.8, 53: 0.5, 54: 0.2},
        falloff_radius=0.06
    )

    # Protected cuff
    cuff_selection = SurfaceSelection(
        selected_faces=[20],
        selected_vertices=[60, 61],
        vertex_weights={60: 1.0, 61: 1.0},
        falloff_radius=0.0
    )

    request = agent.parse_annotation_to_request(
        bundle=bundle,
        region=region,
        surface_selection=target_selection,
        protected_selection=cuff_selection
    )

    assert request.intent == CorrectionIntent.REPAIR_BODY_PENETRATION
    assert request.frame_range == (1045, 1060)
    assert request.fade_frames == (5, 5)
    assert len(request.protected_regions) == 1
    assert "Deterministic" in request.model_attribution

    plan = agent.plan_correction_strategy(request)
    assert plan.chosen_strategy == ExecutionStrategy.NON_DESTRUCTIVE_CORRECTIVE_LAYER
    assert plan.effective_frame_range == (1040, 1065)
    assert plan.target_vertex_count == 5
    assert plan.protected_vertex_count == 2
    assert "Laplacian smoothing" in plan.explanation


def test_temporal_corrective_layer_execution():
    """Verify de-penetration, protected feature preservation, and temporal continuity."""
    jacket = generate_procedural_jacket()
    body_base, body_dict = generate_procedural_animated_body(num_frames=20)
    body_frames = [
        MeshGeometry(vertices=body_dict[f], faces=body_base.faces, name=f"body_{f}")
        for f in range(1, 21)
    ]

    # Create a 20-frame cloth cache where frames 8 to 14 have deliberate body penetration
    cloth_frames = []
    penetrating_v_idx = 42
    protected_v_idx = 100

    for f in range(1, 21):
        v_copy = np.copy(jacket.vertices)
        if 8 <= f <= 14:
            # Push vertex 42 deep inside the body collider at that frame along normal
            b_geom_f = body_frames[f - 1]
            b_normals = b_geom_f.compute_normals()
            nearest_body_v = b_geom_f.vertices[10]
            b_n = b_normals[10]
            v_copy[penetrating_v_idx] = nearest_body_v - b_n * 0.025
        cloth_frames.append(MeshGeometry(name=f"cloth_{f}", vertices=v_copy, faces=jacket.faces))

    # Formulate structured request
    agent = MultimodalCorrectionAgent()
    bundle = AnnotationBundle(
        bundle_id="b_002",
        shot_name="shot010",
        garment_name="hero_jacket",
        geometry_version="v001",
        frame=10,
        time_seconds=0.416,
        camera=CameraParameters(
            yaw=0, pitch=0, distance=2, target=(0, 1, 0),
            fov=1.8, width=200, height=200, view_matrix=[], projection_matrix=[]
        )
    )

    region = AnnotationRegion(
        region_id="r_pen",
        label="Penetration Region",
        intent_description="Remove body penetration here during frames 8–14. Fade the correction into surrounding frames.",
        drawing_tool="brush"
    )

    target_sel = SurfaceSelection(
        selected_faces=[5],
        selected_vertices=[penetrating_v_idx],
        vertex_weights={penetrating_v_idx: 1.0},
        falloff_radius=0.05
    )

    cuff_sel = SurfaceSelection(
        selected_faces=[1],
        selected_vertices=[protected_v_idx],
        vertex_weights={protected_v_idx: 1.0},
        falloff_radius=0.0
    )

    request = agent.parse_annotation_to_request(bundle, region, target_sel, cuff_sel)

    # Execute non-destructive layer
    corrected_cache, telemetry = CorrectiveLayerEngine.apply_correction(
        sim_cache=cloth_frames,
        request=request,
        colliders_cache=body_frames,
        collision_margin=0.008
    )

    assert len(corrected_cache) == 20
    assert telemetry.max_penetration_before_mm > 10.0, f"Expected initial defect, got {telemetry.max_penetration_before_mm}"
    assert telemetry.max_penetration_after_mm < 1.0, f"Penetration not resolved: {telemetry.max_penetration_after_mm} mm"
    assert telemetry.protected_vertices_preserved is True
    assert telemetry.max_displacement_outside_region_mm < 0.001
    assert telemetry.is_flicker_free is True
