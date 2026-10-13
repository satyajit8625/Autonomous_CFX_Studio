"""Autonomous CFX Studio - Physical Cloth Material Properties and Presets.
Implements real-world physical material parameters with XPBD compliance values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ClothMaterialProperties:
    """Physical material properties for XPBD cloth simulation."""
    name: str = "Standard Cloth"
    density: float = 0.3              # Surface density (kg/m^2)
    stretch_compliance: float = 1e-5  # Inverse stiffness alpha (m/N). Lower = stiffer
    bend_compliance: float = 1e-3     # Inverse bending stiffness. Lower = stiffer folds
    damping: float = 0.02             # Velocity damping coefficient [0, 1]
    friction: float = 0.35            # Surface friction with colliders [0, 1]
    collision_thickness: float = 0.01 # Collision margin/thickness (meters)
    air_drag: float = 0.01            # Aerodynamic resistance
    solver_substeps: int = 15         # Recommended XPBD substeps per simulation frame
    solver_iterations: int = 4        # XPBD constraint iterations per substep
    wind_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D wind velocity in m/s
    wind_gust: float = 0.0            # Gust oscillation amplitude factor [0, 1]
    wind_frequency: float = 1.2       # Gust oscillation frequency in Hz
    air_density: float = 1.225        # Ambient air density in kg/m^3
    drag_coefficient: float = 1.0     # Dimensionless cloth aerodynamic drag coefficient Cd

    @classmethod
    def get_presets(cls) -> Dict[str, ClothMaterialProperties]:
        """Catalog of standard industry cloth presets."""
        return {
            "leather": cls(
                name="Heavy Leather",
                density=0.75,
                stretch_compliance=1e-6,
                bend_compliance=5e-4,
                damping=0.04,
                friction=0.55,
                collision_thickness=0.012,
                air_drag=0.005,
                solver_substeps=16,
                solver_iterations=5,
            ),
            "silk": cls(
                name="Lightweight Silk",
                density=0.08,
                stretch_compliance=5e-5,
                bend_compliance=0.05,
                damping=0.01,
                friction=0.15,
                collision_thickness=0.006,
                air_drag=0.03,
                solver_substeps=12,
                solver_iterations=3,
            ),
            "denim": cls(
                name="Rugged Denim",
                density=0.45,
                stretch_compliance=3e-6,
                bend_compliance=2e-3,
                damping=0.03,
                friction=0.45,
                collision_thickness=0.010,
                air_drag=0.01,
                solver_substeps=15,
                solver_iterations=4,
            ),
            "wool": cls(
                name="Heavy Wool Overcoat",
                density=0.60,
                stretch_compliance=5e-6,
                bend_compliance=1e-3,
                damping=0.05,
                friction=0.50,
                collision_thickness=0.015,
                air_drag=0.02,
                solver_substeps=16,
                solver_iterations=4,
            ),
            "cotton": cls(
                name="Medium Cotton",
                density=0.25,
                stretch_compliance=1e-5,
                bend_compliance=5e-3,
                damping=0.02,
                friction=0.30,
                collision_thickness=0.008,
                air_drag=0.015,
                solver_substeps=14,
                solver_iterations=4,
            ),
        }

    @classmethod
    def from_preset(cls, preset_name: str) -> ClothMaterialProperties:
        """Instantiate properties from a named preset."""
        presets = cls.get_presets()
        key = preset_name.lower().strip()
        for k, v in presets.items():
            if k in key or key in k:
                return v
        return cls()
