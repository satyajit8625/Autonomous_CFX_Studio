"""Autonomous CFX Studio - XPBD Constraints Architecture.
Implements vectorized distance, bending, attachment/pinning, and contact constraints with Jacobi valence normalization.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class DistanceConstraintBatch:
    """Vectorized XPBD distance (stretch and shear) constraint batch with Jacobi stabilization."""

    def __init__(
        self,
        edges: np.ndarray,
        rest_lengths: np.ndarray,
        compliance: float = 1e-5,
        num_vertices: Optional[int] = None,
    ):
        self.edges: np.ndarray = np.asarray(edges, dtype=np.int32)  # (E, 2)
        self.rest_lengths: np.asarray = np.asarray(rest_lengths, dtype=np.float32)  # (E,)
        self.compliance: float = compliance
        self.lambdas: np.ndarray = np.zeros(len(self.edges), dtype=np.float32)

        # Precompute vertex valence for stable Jacobi parallel iteration
        n_verts = num_vertices if num_vertices is not None else (int(np.max(edges)) + 1 if len(edges) > 0 else 0)
        self.inv_deg: np.ndarray = np.ones(n_verts, dtype=np.float32)
        if len(self.edges) > 0:
            deg = np.zeros(n_verts, dtype=np.float32)
            np.add.at(deg, self.edges[:, 0], 1.0)
            np.add.at(deg, self.edges[:, 1], 1.0)
            self.inv_deg = 1.0 / np.maximum(deg, 1.0)

    def reset_multipliers(self) -> None:
        """Reset Lagrange multipliers at start of time step."""
        self.lambdas.fill(0.0)

    def project(
        self,
        positions: np.ndarray,
        inv_masses: np.ndarray,
        dt: float,
    ) -> None:
        """Vectorized XPBD distance projection with Jacobi under-relaxation."""
        if len(self.edges) == 0:
            return

        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]

        p0 = positions[i0]
        p1 = positions[i1]
        w0 = inv_masses[i0]
        w1 = inv_masses[i1]
        w_sum = w0 + w1

        active = w_sum > 1e-7
        if not np.any(active):
            return

        p0 = p0[active]
        p1 = p1[active]
        w0 = w0[active]
        w1 = w1[active]
        w_sum = w_sum[active]
        i0_act = i0[active]
        i1_act = i1[active]
        rest = self.rest_lengths[active]
        lams = self.lambdas[active]

        diff = p0 - p1
        dist = np.linalg.norm(diff, axis=1)
        zero_dist = dist < 1e-7
        dist_clamped = np.where(zero_dist, 1.0, dist)
        n = np.where(zero_dist[:, None], np.array([0.0, 1.0, 0.0]), diff / dist_clamped[:, None])

        c = dist - rest
        alpha_tilde = self.compliance / (dt * dt)

        delta_lambda = (-c - alpha_tilde * lams) / (w_sum + alpha_tilde)
        lams += delta_lambda
        self.lambdas[active] = lams

        corr0 = (w0[:, None] * delta_lambda[:, None]) * n
        corr1 = (-w1[:, None] * delta_lambda[:, None]) * n

        # Accumulate displacements using Jacobi valence averaging
        delta_pos = np.zeros_like(positions)
        np.add.at(delta_pos, i0_act, corr0)
        np.add.at(delta_pos, i1_act, corr1)

        # Apply valence normalization to prevent Jacobi overshoot
        if len(self.inv_deg) == len(positions):
            delta_pos *= self.inv_deg[:, None]

        # Clamp max displacement per iteration for numerical safety
        d_norm = np.linalg.norm(delta_pos, axis=1, keepdims=True)
        max_d = 0.05  # 5cm max shift per iteration
        scale = np.where(d_norm > max_d, max_d / (d_norm + 1e-7), 1.0)
        delta_pos *= scale

        positions += delta_pos


class BendingConstraintBatch:
    """Vectorized XPBD isometric bending constraint batch with Jacobi stabilization."""

    def __init__(
        self,
        quads: np.ndarray,
        rest_distances: np.ndarray,
        compliance: float = 1e-3,
        num_vertices: Optional[int] = None,
    ):
        self.quads: np.ndarray = np.asarray(quads, dtype=np.int32)  # (B, 4): [u, v, o1, o2]
        self.rest_distances: np.ndarray = np.asarray(rest_distances, dtype=np.float32)  # (B,)
        self.compliance: float = compliance
        self.lambdas: np.ndarray = np.zeros(len(self.quads), dtype=np.float32)

        n_verts = num_vertices if num_vertices is not None else (int(np.max(quads)) + 1 if len(quads) > 0 else 0)
        self.inv_deg: np.ndarray = np.ones(n_verts, dtype=np.float32)
        if len(self.quads) > 0:
            deg = np.zeros(n_verts, dtype=np.float32)
            np.add.at(deg, self.quads[:, 2], 1.0)
            np.add.at(deg, self.quads[:, 3], 1.0)
            self.inv_deg = 1.0 / np.maximum(deg, 1.0)

    def reset_multipliers(self) -> None:
        self.lambdas.fill(0.0)

    def project(
        self,
        positions: np.ndarray,
        inv_masses: np.ndarray,
        dt: float,
    ) -> None:
        """Vectorized bending projection on opposite vertices o1 and o2 with Jacobi stabilization."""
        if len(self.quads) == 0:
            return

        o1 = self.quads[:, 2]
        o2 = self.quads[:, 3]

        p1 = positions[o1]
        p2 = positions[o2]
        w1 = inv_masses[o1]
        w2 = inv_masses[o2]
        w_sum = w1 + w2

        active = w_sum > 1e-7
        if not np.any(active):
            return

        p1 = p1[active]
        p2 = p2[active]
        w1 = w1[active]
        w2 = w2[active]
        w_sum = w_sum[active]
        o1_act = o1[active]
        o2_act = o2[active]
        rest = self.rest_distances[active]
        lams = self.lambdas[active]

        diff = p1 - p2
        dist = np.linalg.norm(diff, axis=1)
        zero_dist = dist < 1e-7
        dist_clamped = np.where(zero_dist, 1.0, dist)
        n = np.where(zero_dist[:, None], np.array([0.0, 1.0, 0.0]), diff / dist_clamped[:, None])

        c = dist - rest
        alpha_tilde = self.compliance / (dt * dt)

        delta_lambda = (-c - alpha_tilde * lams) / (w_sum + alpha_tilde)
        lams += delta_lambda
        self.lambdas[active] = lams

        corr1 = (w1[:, None] * delta_lambda[:, None]) * n
        corr2 = (-w2[:, None] * delta_lambda[:, None]) * n

        delta_pos = np.zeros_like(positions)
        np.add.at(delta_pos, o1_act, corr1)
        np.add.at(delta_pos, o2_act, corr2)

        if len(self.inv_deg) == len(positions):
            delta_pos *= self.inv_deg[:, None]

        d_norm = np.linalg.norm(delta_pos, axis=1, keepdims=True)
        max_d = 0.05
        scale = np.where(d_norm > max_d, max_d / (d_norm + 1e-7), 1.0)
        delta_pos *= scale

        positions += delta_pos
