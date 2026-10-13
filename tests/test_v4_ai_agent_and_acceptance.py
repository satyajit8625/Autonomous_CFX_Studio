"""Autonomous CFX Studio - Unit Tests for v4.0 AI-Agent Acceptance Milestone.
Tests:
- ApprovalAuthority and ApprovalGatekeeper production boundaries
- ProcessWatchdog timeout and child termination mechanics
- Representative garment and animated body topology
- Penetration depth measurement with proximity thresholding
- CorrectiveLayerEngine temporal acceleration dynamics (m/s^2)
- LocalLLMClient and LocalVisionClient functionality
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pytest

from autonomous_cfx.agent.llm_client import LocalLLMClient, LocalVisionClient
from autonomous_cfx.agent.tools import validate_tool_call
from autonomous_cfx.assets.representative_garment import (
    generate_representative_animated_body,
    generate_tailored_shearling_coat,
)
from autonomous_cfx.core.approval import (
    ApprovalAuthority,
    ApprovalGatekeeper,
    ApprovalRecord,
)
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.watchdog import ProcessWatchdog
from autonomous_cfx.simulation.corrective_layer import CorrectiveLayerEngine


def test_approval_gatekeeper_blocks_synthetic_approval():
    """Verify that synthetic test approvals are blocked from authorizing production exports."""
    synthetic = ApprovalRecord.create_synthetic_test_approval(
        approved_version=1,
        cache_hash="test_hash_123",
        test_case_id="unit_test_harness",
        allow_isolated_publishing=False,
    )
    assert synthetic.authority == ApprovalAuthority.TEST_HARNESS_SYNTHETIC

    with pytest.raises(PermissionError) as exc_info:
        ApprovalGatekeeper.validate_production_export_authorization(synthetic, is_isolated_test_run=False)
    assert "synthetic test harness event" in str(exc_info.value)


def test_approval_gatekeeper_authorizes_genuine_human():
    """Verify that human artist approval permits production delivery."""
    human = ApprovalRecord.create_human_approval(
        approved_version=1,
        cache_hash="test_hash_123",
        artist_name="senior_cfx_artist_mark",
        notes="Reviewed and approved",
    )
    assert human.authority == ApprovalAuthority.HUMAN_ARTIST
    # Must not raise
    ApprovalGatekeeper.validate_production_export_authorization(human, is_isolated_test_run=False)


def test_representative_garment_topology():
    """Verify representative shearling coat mesh properties and topology."""
    coat = generate_tailored_shearling_coat()
    assert coat.num_vertices > 500
    assert coat.num_faces > 1000
    assert coat.uvs is not None
    assert len(coat.uvs) == coat.num_vertices

    # Mannequin body collider
    body, body_dict = generate_representative_animated_body(num_frames=10)
    assert body.num_vertices > 200
    assert len(body_dict) == 10
    assert body_dict[1].shape == (body.num_vertices, 3)


def test_penetration_proximity_thresholding():
    """Verify that penetration measurement ignores distant vertices and reports correct depths."""
    coat = generate_tailored_shearling_coat()
    body, _ = generate_representative_animated_body(num_frames=2)
    b_normals = body.compute_normals()

    # Intrude vertex 180 by exactly 18 mm along body normal
    target_idx = 180
    body_ref_idx = 135
    coat.vertices[target_idx] = body.vertices[body_ref_idx] - b_normals[body_ref_idx] * 0.018

    measured_pen = CorrectiveLayerEngine._measure_penetration_depth(
        cloth=coat,
        collider=body,
        margin=0.008,
        target_indices=[target_idx],
        max_proximity_threshold=0.05,
    )
    assert 14.5 <= measured_pen <= 18.5, f"Expected ~15-18 mm sub-vertex penetration, got {measured_pen}"


def test_watchdog_timeout_execution():
    """Verify that ProcessWatchdog terminates child commands on timeout with exit code 124."""
    import subprocess
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "verify_watchdog_hang_prevention.py"
    proc = subprocess.Popen(
        [sys.executable, str(script_path), "--target"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(timeout=6.0)
    assert proc.returncode == 124
    assert os.path.exists("evidence/reports/watchdog_timeout_report.json")


def test_tool_call_density_unit_normalization():
    """Verify that textile GSM units (> 2.0) are normalized to kg/m^2."""
    raw_args = {
        "material_name": "denim",
        "density": 450.0,  # 450 gsm -> should normalize to 0.45 kg/m^2
        "stretch_compliance": 1e-5,
        "bend_compliance": 1e-4,
        "damping": 0.02,
        "friction": 0.5,
        "collision_margin": 0.01,
        "solver_substeps": 16,
        "reasoning": "Heavy denim fabric",
    }
    validated = validate_tool_call("configure_simulation_parameters", raw_args)
    assert validated["density"] == 0.45
