"""Autonomous CFX Studio - Agent Tool Definitions and Schemas.
Defines strictly validated tool interfaces for local language model execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import jsonschema


class AmbiguityType(str, Enum):
    CONTRADICTORY_REQUIREMENTS = "contradictory_requirements"
    UNDERSPECIFIED_BEHAVIOR = "underspecified_behavior"
    EXTREME_PARAMETER = "extreme_parameter"


class FailureMode(str, Enum):
    COLLIDER_TUNNELING = "collider_tunneling"
    DEEP_PENETRATION = "deep_penetration"
    OVERSTRETCHING = "overstretching"
    NUMERICAL_EXPLOSION = "numerical_explosion"
    PINNED_COLLIDER_CONFLICT = "pinned_collider_conflict"


class RecommendedStatus(str, Enum):
    PASS_RECOMMENDED = "PASS_RECOMMENDED"
    NEEDS_REPAIR = "NEEDS_REPAIR"
    UNRESOLVED_FAILED = "UNRESOLVED_FAILED"


# -------------------------------------------------------------------------
# Tool JSON Schemas
# -------------------------------------------------------------------------

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "configure_simulation_parameters": {
        "type": "object",
        "properties": {
            "material_name": {
                "type": "string",
                "description": "Primary material classification (e.g. leather, silk, denim, wool, cotton, custom).",
            },
            "density": {
                "type": "number",
                "minimum": 0.04,
                "maximum": 2.0,
                "description": "Surface density in kg/m^2.",
            },
            "stretch_compliance": {
                "type": "number",
                "minimum": 1e-8,
                "maximum": 1e-2,
                "description": "XPBD inverse stretch stiffness alpha in m/N. Lower = stiffer.",
            },
            "bend_compliance": {
                "type": "number",
                "minimum": 1e-6,
                "maximum": 0.5,
                "description": "XPBD inverse bending stiffness alpha in rad/N. Lower = stiffer folds.",
            },
            "damping": {
                "type": "number",
                "minimum": 0.001,
                "maximum": 0.20,
                "description": "Velocity damping coefficient [0, 1].",
            },
            "friction": {
                "type": "number",
                "minimum": 0.02,
                "maximum": 0.95,
                "description": "Kinetic/static friction coefficient with colliders [0, 1].",
            },
            "collision_margin": {
                "type": "number",
                "minimum": 0.002,
                "maximum": 0.050,
                "description": "Physical collision buffer margin from collider surface in meters.",
            },
            "self_collision_thickness": {
                "type": "number",
                "minimum": 0.001,
                "maximum": 0.020,
                "description": "Thickness margin for cloth-cloth self-collision in meters.",
            },
            "solver_substeps": {
                "type": "integer",
                "minimum": 4,
                "maximum": 64,
                "description": "XPBD substeps per simulation frame.",
            },
            "solver_iterations": {
                "type": "integer",
                "minimum": 1,
                "maximum": 12,
                "description": "XPBD constraint iterations per substep.",
            },
            "reasoning": {
                "type": "string",
                "description": "Technical rationale for selected physical parameters.",
            },
        },
        "required": [
            "material_name",
            "density",
            "stretch_compliance",
            "bend_compliance",
            "damping",
            "friction",
            "collision_margin",
            "solver_substeps",
            "reasoning",
        ],
        "additionalProperties": False,
    },
    "request_clarification": {
        "type": "object",
        "properties": {
            "ambiguity_type": {
                "type": "string",
                "enum": [e.value for e in AmbiguityType],
                "description": "Type of detected requirement ambiguity.",
            },
            "clarification_question": {
                "type": "string",
                "description": "Direct question clarifying the underspecified or contradictory requirement.",
            },
            "suggested_options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Actionable options presented to the artist.",
            },
            "context": {
                "type": "string",
                "description": "Detailed explanation of why clarification is needed.",
            },
        },
        "required": ["ambiguity_type", "clarification_question", "suggested_options"],
        "additionalProperties": False,
    },
    "diagnose_simulation_failure": {
        "type": "object",
        "properties": {
            "failure_mode": {
                "type": "string",
                "enum": [e.value for e in FailureMode],
                "description": "Primary diagnosed failure mode.",
            },
            "raw_penetration_depth_mm": {
                "type": "number",
                "description": "Measured maximum penetration depth in millimeters.",
            },
            "violating_frames": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Frames with detected tolerance violations.",
            },
            "root_cause": {
                "type": "string",
                "description": "Physical diagnosis of why the simulation failed tolerances.",
            },
            "recommended_repair_strategy": {
                "type": "string",
                "description": "Prescribed repair strategy to address root cause.",
            },
        },
        "required": ["failure_mode", "root_cause", "recommended_repair_strategy"],
        "additionalProperties": False,
    },
    "apply_repair_action": {
        "type": "object",
        "properties": {
            "repair_attempt_index": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Bounded repair iteration index (1-based).",
            },
            "substep_multiplier": {
                "type": "number",
                "minimum": 1.0,
                "maximum": 3.0,
                "description": "Factor to increase solver substeps.",
            },
            "margin_expansion_mm": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 15.0,
                "description": "Additional collision margin in millimeters.",
            },
            "compliance_stiffening_factor": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 1.0,
                "description": "Factor to multiply stretch compliance by (e.g. 0.5 = 2x stiffer).",
            },
            "enable_pre_relaxation": {
                "type": "boolean",
                "description": "Whether to run geometric de-penetration push-out before frame 0 dynamics.",
            },
            "rationale": {
                "type": "string",
                "description": "Technical justification for this repair adjustment.",
            },
        },
        "required": ["repair_attempt_index", "rationale"],
        "additionalProperties": False,
    },
    "submit_quality_recommendation": {
        "type": "object",
        "properties": {
            "recommended_status": {
                "type": "string",
                "enum": [e.value for e in RecommendedStatus],
                "description": "Automated quality recommendation. Note: Human approval is tracked separately.",
            },
            "score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "Quantitative quality score (0 to 100).",
            },
            "tolerance_checks": {
                "type": "object",
                "properties": {
                    "penetration_passed": {"type": "boolean"},
                    "strain_passed": {"type": "boolean"},
                    "stability_passed": {"type": "boolean"},
                },
                "required": ["penetration_passed", "strain_passed", "stability_passed"],
            },
            "summary": {
                "type": "string",
                "description": "Summary of simulation quality assessment.",
            },
        },
        "required": ["recommended_status", "score", "tolerance_checks", "summary"],
        "additionalProperties": False,
    },
}


def validate_tool_call(tool_name: str, arguments: Dict[str, Any]) -> None:
    """Validate tool call arguments against schema. Raises jsonschema.ValidationError on mismatch."""
    if tool_name not in TOOL_SCHEMAS:
        raise ValueError(f"Unknown tool: '{tool_name}'. Available tools: {list(TOOL_SCHEMAS.keys())}")
    schema = TOOL_SCHEMAS[tool_name]
    jsonschema.validate(instance=arguments, schema=schema)
