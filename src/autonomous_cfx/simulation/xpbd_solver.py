"""Autonomous CFX Studio - XPBD Physics Solver.
Extended Position Based Dynamics solver with unconditional stability and realistic physical compliance.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.constraints import BendingConstraintBatch, DistanceConstraintBatch
from autonomous_cfx.simulation.colliders import Collider
from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.core.logger import logger


class XPBDSolver:
    """XPBD Cloth Solver managing vertex positions, velocities, constraints, and contacts."""

    def __init__(
        self,
        geometry: MeshGeometry,
        properties: Optional[ClothMaterialProperties] = None,
    ):
        self.geometry = geometry
        self.props = properties or ClothMaterialProperties()

        # Initialize physical state
        self.positions: np.ndarray = np.copy(geometry.vertices)
        self.prev_positions: np.ndarray = np.copy(geometry.vertices)
        self.velocities: np.ndarray = np.zeros_like(self.positions)
        self.masses: np.ndarray = geometry.compute_vertex_masses(self.props.density)
        self.inv_masses: np.ndarray = 1.0 / self.masses

        # Pinning mask (True for pinned vertices)
        self.pinned_mask: np.ndarray = np.zeros(len(self.positions), dtype=bool)

        # Global forces
        self.gravity: np.ndarray = np.array([0.0, -9.81, 0.0], dtype=np.float32)
        self.wind: np.ndarray = np.zeros(3, dtype=np.float32)

        # Build constraints
        edges, rest_lens = geometry.compute_edges()
        self.distance_constraints = DistanceConstraintBatch(
            edges=edges,
            rest_lengths=rest_lens,
            compliance=self.props.stretch_compliance,
            num_vertices=len(self.positions),
        )

        quads, rest_bends = geometry.compute_bending_pairs()
        self.bending_constraints = BendingConstraintBatch(
            quads=quads,
            rest_distances=rest_bends,
            compliance=self.props.bend_compliance,
            num_vertices=len(self.positions),
        )

        # Colliders
        self.colliders: List[Collider] = []

    def set_pinned_vertices(self, vertex_indices: List[int] | np.ndarray) -> None:
        """Pin specific vertices to prevent them from moving."""
        idx = np.asarray(vertex_indices, dtype=np.int32)
        self.pinned_mask[idx] = True
        self.inv_masses[idx] = 0.0

    def add_collider(self, collider: Collider) -> None:
        """Register a collider to the solver."""
        self.colliders.append(collider)

    def step(self, dt: float, frame_num: float) -> Tuple[int, float]:
        """Perform one simulation step subdivided into XPBD substeps.
        Returns:
            total_penetrations: count of penetrated vertices resolved during step
            max_penetration_depth: max penetration distance detected (meters)
        """
        substeps = max(1, self.props.solver_substeps)
        dt_sub = dt / float(substeps)
        iterations = max(1, self.props.solver_iterations)

        total_contact_events = 0
        max_contact_depth = 0.0

        for s in range(substeps):
            sub_frame = frame_num + (s / float(substeps))

            # 1. Update collider states
            for collider in self.colliders:
                collider.update_frame(sub_frame)

            # 2. External forces and velocity damping
            damping_factor = max(0.0, 1.0 - (self.props.damping * dt_sub * 20.0))
            # Aerodynamic resistance
            drag_factor = max(0.0, 1.0 - (self.props.air_drag * dt_sub * 10.0))

            active_mask = ~self.pinned_mask
            self.velocities[active_mask] += (self.gravity + self.wind) * dt_sub
            self.velocities[active_mask] *= (damping_factor * drag_factor)

            # 3. Predict positions
            self.prev_positions = np.copy(self.positions)
            predicted_positions = np.copy(self.positions)
            predicted_positions[active_mask] += self.velocities[active_mask] * dt_sub

            # 4. Reset constraint Lagrange multipliers
            self.distance_constraints.reset_multipliers()
            self.bending_constraints.reset_multipliers()

            # 5. Solver iterations
            for _ in range(iterations):
                # Distance / stretch constraints
                self.distance_constraints.project(
                    predicted_positions, self.inv_masses, dt_sub
                )
                # Bending constraints
                self.bending_constraints.project(
                    predicted_positions, self.inv_masses, dt_sub
                )

                # Collision resolution
                for collider in self.colliders:
                    pen_mask, proj_pts, normals, col_vels = collider.project_points(
                        predicted_positions, margin=self.props.collision_thickness
                    )
                    if np.any(pen_mask):
                        # Calculate penetration depth
                        depths = np.linalg.norm(predicted_positions[pen_mask] - proj_pts[pen_mask], axis=1)
                        if len(depths) > 0:
                            max_contact_depth = max(max_contact_depth, float(np.max(depths)))
                        total_contact_events += int(np.sum(pen_mask))

                        # Apply projection to non-pinned vertices
                        valid_pen = pen_mask & active_mask
                        predicted_positions[valid_pen] = proj_pts[valid_pen]

            # 6. Velocity update & Contact friction
            self.velocities[active_mask] = (
                predicted_positions[active_mask] - self.prev_positions[active_mask]
            ) / dt_sub

            # Apply friction at contact points
            for collider in self.colliders:
                pen_mask, _, normals, col_vels = collider.project_points(
                    predicted_positions, margin=self.props.collision_thickness * 1.05
                )
                contact_active = pen_mask & active_mask
                if np.any(contact_active):
                    # Relative velocity
                    rel_vel = self.velocities[contact_active] - col_vels[contact_active]
                    n = normals[contact_active]
                    v_norm = np.sum(rel_vel * n, axis=1, keepdims=True) * n
                    v_tan = rel_vel - v_norm

                    # Friction dampens tangential relative velocity
                    fric = min(1.0, max(0.0, self.props.friction))
                    v_tan_damped = v_tan * (1.0 - fric)

                    self.velocities[contact_active] = (
                        col_vels[contact_active] + v_norm + v_tan_damped
                    )

            # 7. Commit positions
            self.positions = predicted_positions

        return total_contact_events, max_contact_depth
