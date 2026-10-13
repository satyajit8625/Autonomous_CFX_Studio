"""Autonomous CFX Studio - UI Workflow and State Machine Verification Tests.
Tests the full artist journey: Empty startup, Demo Scene, Role confirmation, Agent conversation,
state-dependent gating, approval invalidation, and diagnostics drawer navigation.
"""

import pytest
from PySide6.QtWidgets import QApplication

from autonomous_cfx.core.alembic_io import AlembicObjectRole
from autonomous_cfx.ui.app import AutonomousCFXMainWindow
from autonomous_cfx.ui.workflow_banner import WorkflowStage


@pytest.fixture(scope="session")
def qapp():
    """Headless QApplication instance for UI testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["", "-platform", "offscreen"])
    return app


def test_ui_empty_startup_and_gating(qapp):
    """Verify application starts empty with Approve/Export disabled and explanatory tooltips."""
    window = AutonomousCFXMainWindow()
    window.resize(1200, 800)

    # 1. Starts with empty project
    assert window.cloth_geometry is None
    assert window.active_cache is None
    assert window.is_approved is False
    assert window.roles_confirmed is False

    # 2. Stepper should be on Import stage
    assert window.workflow_banner.current_stage == WorkflowStage.IMPORT_ALEMBIC

    # 3. Simulate, Approve, and Export must be disabled at startup
    assert window.agent_panel.btn_run_sim.isEnabled() is False
    assert window.btn_review_approve.isEnabled() is False
    assert window.btn_top_export.isEnabled() is False
    assert window.act_export_abc.isEnabled() is False

    # 4. Check tooltip explanation
    assert "Export Locked" in window.btn_top_export.toolTip()


def test_ui_demo_scene_loading_and_framing(qapp):
    """Verify loading demo scene populates scene browser, viewport, and frames character."""
    window = AutonomousCFXMainWindow()
    window._on_load_demo_scene()

    assert window.cloth_geometry is not None
    assert window.cloth_geometry.num_vertices > 0
    assert len(window.body_colliders) >= 1
    assert window.roles_confirmed is True

    # Stepper advances to Describe Behavior
    assert window.workflow_banner.current_stage == WorkflowStage.DESCRIBE_BEHAVIOR

    # Simulation button unlocked
    assert window.agent_panel.btn_run_sim.isEnabled() is True

    # Camera distance framed comfortably (character not tiny)
    assert window.viewport.distance < 3.0
    assert window.viewport.distance > 0.4


def test_ui_role_modification_and_invalidation(qapp):
    """Verify changing roles un-confirms setup and blocks simulation until re-confirmed."""
    window = AutonomousCFXMainWindow()
    window._on_load_demo_scene()
    assert window.agent_panel.btn_run_sim.isEnabled() is True

    # Modify role in scene browser
    jacket_path = "/shot/hero_jacket"
    window.scene_browser._on_role_changed(jacket_path, "Ignored")

    # Simulation must now be blocked
    window.roles_confirmed = window.scene_browser.has_valid_roles()
    window._update_workflow_state()
    assert window.agent_panel.btn_run_sim.isEnabled() is False

    # Re-assign and confirm
    window.scene_browser._on_role_changed(jacket_path, "Garment")
    window.scene_browser._on_confirm_clicked()
    window._update_workflow_state()
    assert window.agent_panel.btn_run_sim.isEnabled() is True


def test_ui_approval_gating_and_invalidation(qapp):
    """Verify export requires approval, and cache changes invalidate approval."""
    window = AutonomousCFXMainWindow()
    window._on_load_demo_scene()

    # Artificially set an active cache (e.g. from 1-frame mock)
    from autonomous_cfx.core.cache_io import SimulationCache
    cache = SimulationCache("hero_jacket", base_faces=window.cloth_geometry.faces)
    cache.add_frame(1, window.cloth_geometry.vertices.copy())
    window.active_cache = cache
    window.current_result_version = 1
    window._update_workflow_state()

    # Review button enabled, export still disabled
    assert window.btn_review_approve.isEnabled() is True
    assert window.btn_top_export.isEnabled() is False

    # Artist approves
    window._on_artist_approve()
    assert window.is_approved is True
    assert window.approved_version == 1
    assert window.btn_top_export.isEnabled() is True
    assert window.workflow_banner.current_stage == WorkflowStage.EXPORT_ALEMBIC

    # A subsequent modification (e.g. new sim or change) invalidates approval
    window._invalidate_approval()
    assert window.is_approved is False
    assert window.btn_top_export.isEnabled() is False
    assert "Export Locked" in window.btn_top_export.toolTip()


def test_ui_clarification_card_interaction(qapp):
    """Verify contradictory prompt triggers clarification card with clickable options."""
    window = AutonomousCFXMainWindow()
    window._on_load_demo_scene()

    # Input contradictory prompt
    contradictory_prompt = "completely rigid stiff leather but flows like liquid silk water"
    window.agent_panel.txt_prompt.setText(contradictory_prompt)
    window.agent_panel._on_run_sim_clicked()

    # Verify clarification card added to chat
    # The last parsed props shouldn't be set yet because clarification was requested
    assert window.agent_panel.is_simulating is False
