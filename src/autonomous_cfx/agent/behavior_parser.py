"""Autonomous CFX Studio - Natural Language Behavior Parser.
Translates artist cloth behavior prompts into physical simulation parameters and solver configurations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.core.logger import logger


@dataclass
class BehaviorParseResult:
    """Detailed parsing result including physical properties, solver recommendations, and rationale."""
    prompt: str
    properties: ClothMaterialProperties
    detected_material: str
    detected_traits: List[str] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)


class BehaviorParser:
    """Parses natural language descriptions into verified physical cloth parameters."""

    MATERIAL_KEYWORDS = {
        "leather": "leather",
        "silk": "silk",
        "denim": "denim",
        "wool": "wool",
        "cotton": "cotton",
    }

    @classmethod
    def parse(cls, prompt: str) -> BehaviorParseResult:
        """Parse natural language cloth behavior prompt."""
        text = prompt.lower()
        traits: List[str] = []
        rationale: List[str] = []

        # 1. Detect base material
        detected_mat = "cotton"  # Default base
        for kw, mat_name in cls.MATERIAL_KEYWORDS.items():
            if re.search(r"\b" + kw + r"\b", text):
                detected_mat = mat_name
                traits.append(f"Base material: {kw.capitalize()}")
                rationale.append(f"Identified primary cloth material as '{kw.capitalize()}'.")
                break

        props = ClothMaterialProperties.from_preset(detected_mat)

        # 2. Analyze weight / density
        if re.search(r"\b(heavy|thick|dense|weighted)\b", text):
            props.density *= 1.4
            props.collision_thickness = max(props.collision_thickness, 0.012)
            traits.append("High Density")
            rationale.append("Increased surface density and collision margin due to heavy/thick descriptor.")
        elif re.search(r"\b(light|lightweight|thin|sheer|airy)\b", text):
            props.density *= 0.6
            props.air_drag *= 1.5
            traits.append("Low Density")
            rationale.append("Reduced density and increased aerodynamic drag for lightweight cloth.")

        # 3. Analyze stretch resistance / stiffness
        if re.search(r"\b(stiff|rigid|low stretch|non-stretch|tight|firm)\b", text):
            props.stretch_compliance = max(1e-7, props.stretch_compliance * 0.2)
            props.solver_substeps = max(props.solver_substeps, 16)
            props.solver_iterations = max(props.solver_iterations, 5)
            traits.append("High Stretch Resistance")
            rationale.append("Tightened stretch compliance (alpha) and elevated solver substeps for low stretch.")
        elif re.search(r"\b(stretchy|elastic|loose|flexible|giving)\b", text):
            props.stretch_compliance *= 4.0
            traits.append("Compliant Stretch")
            rationale.append("Relaxed stretch compliance for elastic behavior.")

        # 4. Analyze bending / folds / collar stiffness
        if re.search(r"\b(stiff collar|crisp|structured|heavy folds|creased)\b", text):
            props.bend_compliance = max(1e-5, props.bend_compliance * 0.3)
            traits.append("High Bend Stiffness")
            rationale.append("Decreased bending compliance to support stiff collar and structured folds.")
        elif re.search(r"\b(soft|floppy|flowing|fluid|draping|loose drape)\b", text):
            props.bend_compliance *= 3.0
            traits.append("Soft Bend Drape")
            rationale.append("Increased bend compliance for soft, fluid drapery.")

        # 5. Analyze friction & contact interaction
        if re.search(r"\b(high friction|grippy|non-slip|sticky|rough)\b", text):
            props.friction = min(0.85, props.friction + 0.25)
            traits.append("High Friction")
            rationale.append("Enhanced friction coefficient for firm contact holding.")
        elif re.search(r"\b(slippery|low friction|smooth|slick|glossy)\b", text):
            props.friction = max(0.05, props.friction * 0.5)
            traits.append("Low Friction")
            rationale.append("Lowered friction coefficient for smooth gliding over body colliders.")

        # 6. Analyze damping / energy dissipation
        if re.search(r"\b(high damping|settled|calm|deadened|damped)\b", text):
            props.damping = min(0.08, props.damping * 2.0)
            traits.append("High Damping")
            rationale.append("Elevated velocity damping to suppress excessive fluttering.")
        elif re.search(r"\b(bouncy|lively|fluttering|energetic)\b", text):
            props.damping = max(0.005, props.damping * 0.5)
            traits.append("Low Damping")
            rationale.append("Reduced damping for lively, dynamic motion.")

        # 7. Analyze aerodynamic forces & wind
        if re.search(r"\b(wind|breeze|gust|mountain air|gale)\b", text):
            if "mountain" in text or "mountain air" in text:
                props.wind_velocity = (2.5, 0.0, 1.2)
            elif "strong" in text or "gale" in text:
                props.wind_velocity = (5.0, 0.0, 2.0)
            elif "breeze" in text:
                props.wind_velocity = (1.5, 0.0, 0.5)
            else:
                props.wind_velocity = (2.0, 0.0, 1.0)

            if "gust" in text or "gusty" in text:
                props.wind_gust = 0.40
                traits.append("Gusty Aerodynamic Wind")
                rationale.append("Enabled aerodynamic drag with gust oscillation amplitude 0.40.")
            else:
                traits.append("Aerodynamic Wind")
                rationale.append(f"Configured 3D wind velocity vector {props.wind_velocity} m/s.")

        logger.info(f"Parsed behavior prompt: '{prompt}' -> {detected_mat} ({len(traits)} traits)")
        return BehaviorParseResult(
            prompt=prompt,
            properties=props,
            detected_material=detected_mat,
            detected_traits=traits,
            rationale=rationale,
        )
