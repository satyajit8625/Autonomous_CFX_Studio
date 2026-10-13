"""Autonomous CFX Studio - XPBD Physics Solver.
Extended Position Based Dynamics solver with unconditional stability, physical compliance,
cloth self-collision, and raw penetration measurement telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.simulation.constraints import (
    AnimatedAttachment,
    BendingConstraintBatch,
    DistanceConstraintBatch,
)
from autonomous_cfx.simulation.colliders import Collider, CollisionMeasurement
from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.simulation.spatial_hash import SpatialHashSelfCollision
from autonomous_cfx.core.logger import logger


@dataclass
class StepTelemetry:
    """Step telemetry reporting raw physical measurements."""
    margin_contact_events: int = 0
    max_margin_intrusion: float = 0.0
    raw_true_penetrations: int = 0
    raw_max_penetration_depth: float = 0.0
    raw_penetrated_vertex_ids: List[int] = field(default_factory=list)
    self_collision_events: int = 0


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
        self.prev_positions: np.copy(geometry.vertices)
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

        # Cloth self-collision
        self.self_collision = SpatialHashSelfCollision(
            thickness=getattr(self.props, "self_collision_thickness", 0.005),
            edges=edges,
            faces=geometry.faces,
            num_vertices=len(self.positions),
        )

        # Colliders & Attachments
        self.colliders: List[Collider] = []
        self.attachments: List[AnimatedAttachment] = []

    def set_pinned_vertices(self, vertex_indices: List[int] | np.ndarray) -> None:
        """Pin specific vertices to prevent them from moving."""
        idx = np.asarray(vertex_indices, dtype=np.int32)
        self.pinned_mask[idx] = True
        self.inv_masses[idx] = 0.0

    def add_collider(self, collider: Collider) -> None:
        """Register a collider to the solver."""
        self.colliders.append(collider)

    def add_attachment(self, attachment: AnimatedAttachment) -> None:
        """Register an animated attachment binding cloth vertices to a collider or trajectory."""
        self.attachments.append(attachment)
        if attachment.stiffness >= 1.0:
            self.pinned_mask[attachment.cloth_indices] = True
            self.inv_masses[attachment.cloth_indices] = 0.0

    def attach_to_collider(
        self,
        cloth_indices: List[int] | np.ndarray,
        collider: Collider,
        collider_indices: Optional[List[int] | np.ndarray] = None,
        stiffness: float = 1.0,
    ) -> AnimatedAttachment:
        """Bind cloth vertices to an animated collider.
        If collider_indices is None, automatically matches each cloth vertex to the closest
        collider vertex at rest pose and records relative rest offset vectors.
        """
        c_idx = np.asarray(cloth_indices, dtype=np.int32)
        if len(c_idx) == 0:
            att = AnimatedAttachment(cloth_indices=c_idx, collider=collider, stiffness=stiffness)
            self.add_attachment(att)
            return att

        cloth_pts = self.positions[c_idx]

        if collider_indices is None:
            if hasattr(collider, "current_vertices") and len(collider.current_vertices) > 0:
                col_verts = collider.current_vertices
            elif hasattr(collider, "geometry") and len(collider.geometry.vertices) > 0:
                col_verts = collider.geometry.vertices
            else:
                col_verts = np.zeros((1, 3), dtype=np.float32)

            from scipy.spatial import KDTree
            kdt = KDTree(col_verts)
            _, matched_col_idx = kdt.query(cloth_pts)
            matched_col_idx = np.asarray(matched_col_idx, dtype=np.int32)
        else:
            matched_col_idx = np.asarray(collider_indices, dtype=np.int32)

        if hasattr(collider, "current_vertices") and len(collider.current_vertices) > 0:
            col_target_pts = collider.current_vertices[matched_col_idx]
        elif hasattr(collider, "geometry") and len(collider.geometry.vertices) > 0:
            col_target_pts = collider.geometry.vertices[matched_col_idx]
        else:
            col_target_pts = np.zeros_like(cloth_pts)

        rest_offsets = (cloth_pts - col_target_pts).astype(np.float32)

        attachment = AnimatedAttachment(
            cloth_indices=c_idx,
            collider=collider,
            collider_indices=matched_col_idx,
            rest_offsets=rest_offsets,
            stiffness=stiffness,
        )
        self.add_attachment(attachment)
        return attachment

    def step(self, dt: float, frame_num: float) -> Tuple[int, float, StepTelemetry]:
        """Perform one simulation step subdivided into XPBD substeps.
        Returns:
            total_contacts: count of margin contacts resolved
            max_contact_depth: max margin intrusion depth
            telemetry: StepTelemetry containing raw penetration depth and penetrated vertex IDs
        """
        substeps = max(1, self.props.solver_substeps)
        dt_sub = dt / float(substeps)
        iterations = max(1, self.props.solver_iterations)

        total_contacts = 0
        max_contact_depth = 0.0
        total_raw_true_pens = 0
        max_raw_pen_depth = 0.0
        all_pen_vert_ids: Set[int] = set()
        total_self_collisions = 0

        for s in range(substeps):
            sub_frame = (frame_num - 1.0) + ((s + 1) / float(substeps))

            # 1. Update collider states
            for collider in self.colliders:
                collider.update_frame(sub_frame)

            # 2. Update animated attachments kinematically
            for att in self.attachments:
                if att.collider is not None and hasattr(att.collider, "current_vertices"):
                    if att.collider_indices is not None and att.rest_offsets is not None:
                        col_pts = att.collider.current_vertices[att.collider_indices]
                        target_pts = col_pts + att.rest_offsets
                        if att.stiffness >= 1.0:
                            # Kinematic lock: update cloth positions directly and velocity accordingly
                            old_p = np.copy(self.positions[att.cloth_indices])
                            self.positions[att.cloth_indices] = target_pts
                            self.velocities[att.cloth_indices] = (target_pts - old_p) / dt_sub
                        else:
                            p_curr = self.positions[att.cloth_indices]
                            spring_force = (target_pts - p_curr) * att.stiffness / dt_sub
                            self.velocities[att.cloth_indices] += spring_force * dt_sub

            # 3. External forces: Gravity and Aerodynamic Wind Drag
            damping_factor = max(0.0, 1.0 - (self.props.damping * dt_sub * 20.0))
            active_mask = ~self.pinned_mask

            # Physical Aerodynamic Wind Field (in m/s) with temporal gust oscillation
            t_sec = sub_frame * dt
            gust_amp = float(getattr(self.props, "wind_gust", 0.0))
            gust_freq = float(getattr(self.props, "wind_frequency", 1.2))
            gust_mult = 1.0 + gust_amp * float(np.sin(2.0 * np.pi * gust_freq * t_sec))

            wind_base = getattr(self.props, "wind_velocity", self.wind)
            wind_base = np.asarray(wind_base, dtype=np.float32)
            v_wind = wind_base * gust_mult  # 3D wind velocity vector in m/s

            if np.any(active_mask):
                v_cloth = self.velocities[active_mask]
                v_rel = v_wind - v_cloth
                v_rel_speed = np.linalg.norm(v_rel, axis=1, keepdims=True)

                air_density = float(getattr(self.props, "air_density", 1.225))  # kg/m^3
                cd = float(getattr(self.props, "drag_coefficient", 1.0))
                sigma = max(0.01, float(self.props.density))  # surface density sigma in kg/m^2
                aero_factor = 0.5 * air_density * cd / sigma

                # Aerodynamic drag acceleration: a_aero = 0.5 * rho * Cd * (1 / sigma) * |v_rel| * v_rel (m/s^2)
                a_aero = aero_factor * v_rel_speed * v_rel
                a_drag = - (float(self.props.air_drag) * 10.0) * v_cloth

                delta_v = (self.gravity + a_aero + a_drag) * dt_sub

                # Stability clamp: drag impulse cannot reverse/overshoot relative air velocity in one substep
                dv_norm = np.linalg.norm(delta_v, axis=1, keepdims=True)
                max_allowable_dv = v_rel_speed + np.linalg.norm(self.gravity) * dt_sub + 1e-4
                overshoot = dv_norm > max_allowable_dv
                if np.any(overshoot):
                    delta_v = np.where(overshoot, delta_v * (max_allowable_dv / (dv_norm + 1e-7)), delta_v)

                self.velocities[active_mask] += delta_v
                self.velocities[active_mask] *= damping_factor

            # 4. Predict positions
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
                # Isometric Bending constraints
                self.bending_constraints.project(
                    predicted_positions, self.inv_masses, dt_sub
                )

                # Cloth self-collision
                self_col_count = self.self_collision.resolve_self_collisions(
                    predicted_positions, self.inv_masses, iterations=1
                )
                total_self_collisions += self_col_count

                # Moving collider contacts & penetration resolution
                for collider in self.colliders:
                    contact_mask, proj_pts, normals, col_vels = collider.project_points(
                        predicted_positions, margin=self.props.collision_thickness
                    )
                    meas = collider.last_measurement
                    if meas.contact_count > 0:
                        total_contacts += meas.contact_count
                        max_contact_depth = max(max_contact_depth, meas.margin_intrusion_depth)

                    if meas.true_penetration_count > 0:
                        total_raw_true_pens += meas.true_penetration_count
                        max_raw_pen_depth = max(max_raw_pen_depth, meas.raw_max_penetration_depth)
                        all_pen_vert_ids.update(meas.penetrating_vertex_ids)

                    # Project non-pinned cloth vertices out
                    valid_contact = contact_mask & active_mask
                    if np.any(valid_contact):
                        predicted_positions[valid_contact] = proj_pts[valid_contact]

            # 6. Velocity update & Contact friction
            self.velocities[active_mask] = (
                predicted_positions[active_mask] - self.prev_positions[active_mask]
            ) / dt_sub

            # Apply friction at contact points
            for collider in self.colliders:
                contact_mask, _, normals, col_vels = collider.project_points(
                    predicted_positions, margin=self.props.collision_thickness * 1.05
                )
                contact_active = contact_mask & active_mask
                if np.any(contact_active):
                    rel_vel = self.velocities[contact_active] - col_vels[contact_active]
                    n = normals[contact_active]
                    # Inelastic contact: prevent artificial normal velocity explosion from discrete projection pushout
                    v_norm_scalar = np.sum(rel_vel * n, axis=1, keepdims=True)
                    v_norm = np.minimum(0.0, v_norm_scalar) * n
                    v_tan = rel_vel - v_norm_scalar * n

                    fric = min(1.0, max(0.0, self.props.friction))
                    v_tan_damped = v_tan * (1.0 - fric)

                    self.velocities[contact_active] = (
                        col_vels[contact_active] + v_norm + v_tan_damped
                    )

            # 7. Commit positions
            self.positions = predicted_positions

        # Measure committed unresolved residual penetrations at the destination frame
        committed_pens = 0
        committed_max_depth = 0.0
        committed_pen_ids: Set[int] = set()

        for collider in self.colliders:
            collider.update_frame(float(frame_num))
            _, _, _, _ = collider.project_points(
                self.positions, margin=self.props.collision_thickness
            )
            meas = collider.last_measurement
            if meas.true_penetration_count > 0:
                active_pens = [v for v in meas.penetrating_vertex_ids if not self.pinned_mask[v]]
                if active_pens:
                    committed_pens += len(active_pens)
                    committed_pen_ids.update(active_pens)
                    committed_max_depth = max(committed_max_depth, meas.raw_max_penetration_depth)

        telemetry = StepTelemetry(
            margin_contact_events=total_contacts,
            max_margin_intrusion=max_contact_depth,
            raw_true_penetrations=committed_pens,
            raw_max_penetration_depth=committed_max_depth,
            raw_penetrated_vertex_ids=sorted(list(committed_pen_ids)),
            self_collision_events=total_self_collisions,
        )
        return total_contacts, max_contact_depth, telemetry
