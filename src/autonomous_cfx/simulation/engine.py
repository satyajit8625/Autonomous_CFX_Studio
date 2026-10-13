"""Autonomous CFX Studio - High-Level Simulation Engine.
Orchestrates multi-frame solver progression, initial penetration healing, and cache generation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.simulation.xpbd_solver import XPBDSolver
from autonomous_cfx.simulation.constraints import AnimatedAttachment
from autonomous_cfx.simulation.colliders import Collider, AnimatedMeshCollider
from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.core.logger import logger


class SimulationEngine:
    """Manages full simulation execution pipeline."""

    def __init__(
        self,
        cloth_geometry: MeshGeometry,
        properties: Optional[ClothMaterialProperties] = None,
        colliders: Optional[List[Collider]] = None,
        pinned_vertices: Optional[List[int]] = None,
        fps: float = 24.0,
        start_frame: int = 1,
        end_frame: int = 40,
        attachments: Optional[List[AnimatedAttachment]] = None,
    ):
        self.cloth_geometry = cloth_geometry.copy()
        self.props = properties or ClothMaterialProperties()
        self.colliders = colliders or []
        self.pinned_vertices = pinned_vertices or []
        self.fps = fps
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.attachments = attachments or []
        self.cancel_requested: bool = False

    def pre_relax_initial_penetrations(self, solver: XPBDSolver, iterations: int = 10) -> int:
        """Resolve any initial cloth-collider overlap before dynamic simulation starts."""
        total_pushed = 0
        for _ in range(iterations):
            for collider in solver.colliders:
                collider.update_frame(float(self.start_frame))
                pen_mask, proj_pts, _, _ = collider.project_points(
                    solver.positions, margin=self.props.collision_thickness * 1.1
                )
                valid_mask = pen_mask & (~solver.pinned_mask)
                if np.any(valid_mask):
                    solver.positions[valid_mask] = proj_pts[valid_mask]
                    total_pushed += int(np.sum(valid_mask))
            # Relax distance constraints so de-penetration preserves cloth edge rest lengths
            solver.distance_constraints.project(solver.positions, solver.inv_masses, dt=0.01)
        if total_pushed > 0:
            logger.info(f"Pre-relaxed {total_pushed} initial cloth-collider penetrations at frame {self.start_frame}")
        return total_pushed

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> SimulationCache:
        """Execute simulation run over the specified frame range."""
        self.cancel_requested = False
        dt = 1.0 / self.fps
        total_frames = (self.end_frame - self.start_frame) + 1

        solver = XPBDSolver(self.cloth_geometry, self.props)

        for col in self.colliders:
            solver.add_collider(col)

        # Register explicit attachments
        for att in self.attachments:
            solver.add_attachment(att)

        # If pinned vertices were specified without explicit attachment:
        # If there is an animated mesh collider, attach them kinematically so they move with character
        if self.pinned_vertices and not self.attachments:
            anim_colliders = [c for c in self.colliders if isinstance(c, AnimatedMeshCollider)]
            if anim_colliders:
                # Kinematically bind to primary animated collider
                solver.attach_to_collider(self.pinned_vertices, anim_colliders[0], stiffness=1.0)
            else:
                solver.set_pinned_vertices(self.pinned_vertices)
        elif self.pinned_vertices and self.attachments:
            solver.set_pinned_vertices(self.pinned_vertices)


        # Pre-resolve initial penetrations at frame 0/start
        self.pre_relax_initial_penetrations(solver)

        cache = SimulationCache(
            name=self.cloth_geometry.name,
            fps=self.fps,
            start_frame=self.start_frame,
            base_faces=self.cloth_geometry.faces,
        )
        cache.metadata["properties"] = self.props.__dict__

        # Record initial frame
        cache.add_frame(
            frame_num=self.start_frame,
            positions=solver.positions,
            velocities=solver.velocities,
            metrics={"contact_events": 0, "max_depth": 0.0, "max_velocity": 0.0},
        )

        logger.info(
            f"Starting simulation: frames {self.start_frame}..{self.end_frame} ({total_frames} frames) "
            f"with substeps={self.props.solver_substeps}, compliance={self.props.stretch_compliance}"
        )

        for frame_idx, frame_num in enumerate(range(self.start_frame + 1, self.end_frame + 1), start=2):
            if self.cancel_requested or (cancel_check is not None and cancel_check()):
                logger.warning(f"Simulation cancelled by user at frame {frame_num}")
                break

            # Solver step
            contacts, max_depth, telemetry = solver.step(dt=dt, frame_num=float(frame_num))

            # Compute diagnostic metrics for frame
            vel_mags = np.linalg.norm(solver.velocities, axis=1)
            max_vel = float(np.max(vel_mags)) if len(vel_mags) > 0 else 0.0
            mean_vel = float(np.mean(vel_mags)) if len(vel_mags) > 0 else 0.0

            metrics = {
                "contact_events": contacts,
                "max_margin_depth": max_depth,
                "raw_true_penetrations": telemetry.raw_true_penetrations,
                "raw_max_penetration_depth": telemetry.raw_max_penetration_depth,
                "raw_penetrated_vertex_ids": telemetry.raw_penetrated_vertex_ids,
                "self_collision_events": telemetry.self_collision_events,
                "max_velocity": max_vel,
                "mean_velocity": mean_vel,
            }

            cache.add_frame(
                frame_num=frame_num,
                positions=solver.positions,
                velocities=solver.velocities,
                metrics=metrics,
            )

            if progress_callback:
                progress_callback(frame_idx, total_frames, metrics)

        logger.info(f"Simulation completed: {cache.frame_count} frames recorded.")
        return cache
