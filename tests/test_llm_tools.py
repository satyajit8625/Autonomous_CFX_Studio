"""Tests for Local Language Model integration, tool schemas, and clarification handling."""

import pytest
import jsonschema
from autonomous_cfx.agent.tools import (
    TOOL_SCHEMAS,
    AmbiguityType,
    FailureMode,
    RecommendedStatus,
    validate_tool_call,
)
from autonomous_cfx.agent.llm_client import LocalLLMClient


def test_tool_schemas_validation():
    # Valid configure parameters
    valid_cfg = {
        "material_name": "leather",
        "density": 0.75,
        "stretch_compliance": 1e-6,
        "bend_compliance": 5e-4,
        "damping": 0.04,
        "friction": 0.55,
        "collision_margin": 0.012,
        "self_collision_thickness": 0.005,
        "solver_substeps": 16,
        "solver_iterations": 4,
        "reasoning": "Standard heavy leather.",
    }
    validate_tool_call("configure_simulation_parameters", valid_cfg)

    # Invalid parameter (density below minimum)
    invalid_cfg = dict(valid_cfg)
    invalid_cfg["density"] = 0.001
    with pytest.raises(jsonschema.ValidationError):
        validate_tool_call("configure_simulation_parameters", invalid_cfg)


def test_clarification_tool_schema():
    valid_clarif = {
        "ambiguity_type": AmbiguityType.CONTRADICTORY_REQUIREMENTS.value,
        "clarification_question": "Do you want rigid leather or fluid silk?",
        "suggested_options": ["Option A: Leather", "Option B: Silk"],
    }
    validate_tool_call("request_clarification", valid_clarif)


def test_local_llm_client_normal_prompt():
    client = LocalLLMClient(backend="embedded_local")
    text, tool_calls = client.generate_response("heavy leather jacket, stiff collar, low stretch")

    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call.tool_name == "configure_simulation_parameters"
    assert call.validation_passed == True
    assert call.arguments["material_name"] == "leather"
    assert call.arguments["stretch_compliance"] < 1e-5

    # Check conversation history recorded
    assert len(client.conversation_history) >= 2


def test_local_llm_client_contradictory_prompt_triggers_clarification():
    client = LocalLLMClient(backend="embedded_local")
    # Contradictory: rigid and flowing like water
    text, tool_calls = client.generate_response("make it completely rigid but flow like water")

    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call.tool_name == "request_clarification"
    assert call.validation_passed == True
    assert call.arguments["ambiguity_type"] == AmbiguityType.CONTRADICTORY_REQUIREMENTS.value
    assert len(call.arguments["suggested_options"]) >= 2
