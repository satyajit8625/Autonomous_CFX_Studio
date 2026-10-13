"""Tests for master AutonomousCFXOrchestrator end-to-end milestone workflow."""

import pytest
from autonomous_cfx.agent.orchestrator import AutonomousCFXOrchestrator
from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.core.project import ProjectManager


def test_orchestrator_end_to_end_workflow(tmp_path):
    pm = ProjectManager(tmp_path)
    am = AssetManager(pm.assets_dir)
    orchestrator = AutonomousCFXOrchestrator(pm)

    jacket, body_col, chair_col = am.load_milestone_package(
        num_frames=5,
        fps=24.0,
        include_chair=True,
    )
    colliders = [body_col, chair_col]
    pinned = list(jacket.vertex_groups.get("collar", []))

    result = orchestrator.execute_workflow(
        cloth_geometry=jacket,
        colliders=colliders,
        behavior_description="heavy leather jacket, stiff collar, low stretch",
        pinned_vertices=pinned,
        shot_name="test_shot",
        fps=24.0,
        start_frame=1,
        end_frame=5,
        auto_approve_if_passed=True,
        export_on_approval=True,
    )

    assert result.success == True
    assert result.behavior_parse.detected_material == "leather"
    assert result.simulation_cache.frame_count == 5
    assert result.artist_approved == True
    assert result.exported_cache_path is not None
    assert result.exported_obj_count == 5
    assert result.exported_manifest_path is not None
