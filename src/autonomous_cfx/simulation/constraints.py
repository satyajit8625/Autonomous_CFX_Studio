"""Autonomous CFX Studio - XPBD Constraints Architecture.
Implements vectorized distance, bending, attachment/pinning, and contact constraints with Jacobi valence normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _project_distance_numba(
        positions: np.ndarray,
        inv_masses: np.ndarray,
        edges: np.ndarray,
        rest_lengths: np.ndarray,
        lambdas: np.ndarray,
        inv_deg: np.ndarray,
        compliance: float,
        dt: float,
    ) -> None:
        E = edges.shape[0]
        alpha_tilde = compliance / (dt * dt)
        delta_pos = np.zeros_like(positions)

        for e in range(E):
            i0 = edges[e, 0]
            i1 = edges[e, 1]
            w0 = inv_masses[i0]
            w1 = inv_masses[i1]
            w_sum = w0 + w1
            if w_sum < 1e-7:
                continue

            dx = positions[i0, 0] - positions[i1, 0]
            dy = positions[i0, 1] - positions[i1, 1]
            dz = positions[i0, 2] - positions[i1, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 1e-7:
                nx = 0.0
                ny = 1.0
                nz = 0.0
            else:
                inv_dist = 1.0 / dist
                nx = dx * inv_dist
                ny = dy * inv_dist
                nz = dz * inv_dist

            c = dist - rest_lengths[e]
            lam = lambdas[e]
            delta_lambda = (-c - alpha_tilde * lam) / (w_sum + alpha_tilde)
            lambdas[e] = lam + delta_lambda

            factor0 = w0 * delta_lambda
            delta_pos[i0, 0] += factor0 * nx
            delta_pos[i0, 1] += factor0 * ny
            delta_pos[i0, 2] += factor0 * nz

            factor1 = -w1 * delta_lambda
            delta_pos[i1, 0] += factor1 * nx
            delta_pos[i1, 1] += factor1 * ny
            delta_pos[i1, 2] += factor1 * nz

        N = positions.shape[0]
        has_deg = len(inv_deg) == N
        max_d = 0.05
        for i in range(N):
            scale = inv_deg[i] if has_deg else 1.0
            dx = delta_pos[i, 0] * scale
            dy = delta_pos[i, 1] * scale
            dz = delta_pos[i, 2] * scale
            d_len = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d_len > max_d:
                clamp_scale = max_d / (d_len + 1e-7)
                dx *= clamp_scale
                dy *= clamp_scale
                dz *= clamp_scale
            positions[i, 0] += dx
            positions[i, 1] += dy
            positions[i, 2] += dz

    @njit(fastmath=True)
    def _project_bending_numba(
        positions: np.ndarray,
        inv_masses: np.ndarray,
        quads: np.ndarray,
        rest_distances: np.ndarray,
        lambdas: np.ndarray,
        inv_deg: np.ndarray,
        compliance: float,
        dt: float,
    ) -> None:
        B = quads.shape[0]
        alpha_tilde = compliance / (dt * dt)
        delta_pos = np.zeros_like(positions)

        for b in range(B):
            o1 = quads[b, 2]
            o2 = quads[b, 3]
            w1 = inv_masses[o1]
            w2 = inv_masses[o2]
            w_sum = w1 + w2
            if w_sum < 1e-7:
                continue

            dx = positions[o1, 0] - positions[o2, 0]
            dy = positions[o1, 1] - positions[o2, 1]
            dz = positions[o1, 2] - positions[o2, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 1e-7:
                nx = 0.0
                ny = 1.0
                nz = 0.0
            else:
                inv_dist = 1.0 / dist
                nx = dx * inv_dist
                ny = dy * inv_dist
                nz = dz * inv_dist

            c = dist - rest_distances[b]
            lam = lambdas[b]
            delta_lambda = (-c - alpha_tilde * lam) / (w_sum + alpha_tilde)
            lambdas[b] = lam + delta_lambda

            factor1 = w1 * delta_lambda
            delta_pos[o1, 0] += factor1 * nx
            delta_pos[o1, 1] += factor1 * ny
            delta_pos[o1, 2] += factor1 * nz

            factor2 = -w2 * delta_lambda
            delta_pos[o2, 0] += factor2 * nx
            delta_pos[o2, 1] += factor2 * ny
            delta_pos[o2, 2] += factor2 * nz

        N = positions.shape[0]
        has_deg = len(inv_deg) == N
        max_d = 0.05
        for i in range(N):
            scale = inv_deg[i] if has_deg else 1.0
            dx = delta_pos[i, 0] * scale
            dy = delta_pos[i, 1] * scale
            dz = delta_pos[i, 2] * scale
            d_len = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d_len > max_d:
                clamp_scale = max_d / (d_len + 1e-7)
                dx *= clamp_scale
                dy *= clamp_scale
                dz *= clamp_scale
            positions[i, 0] += dx
            positions[i, 1] += dy
            positions[i, 2] += dz
else:
    _project_distance_numba = None
    _project_bending_numba = None


@dataclass
class AnimatedAttachment:
    """Kinematic or compliant attachment binding cloth vertices to an animated collider or trajectory.

    Attributes:
        cloth_indices: (K,) array of garment vertex indices to attach.
        collider: Animated collider instance to attach to (e.g. AnimatedMeshCollider).
        collider_indices: (K,) array of matching collider vertex indices.
        rest_offsets: (K, 3) offset vector from collider vertex to cloth vertex at rest pose.
        stiffness: 1.0 for hard kinematic lock (infinite mass), < 1.0 for compliant attachment.
    """
    cloth_indices: np.ndarray
    collider: Optional[Any] = None
    collider_indices: Optional[np.ndarray] = None
    rest_offsets: Optional[np.ndarray] = None
    stiffness: float = 1.0

    def __post_init__(self):
        self.cloth_indices = np.asarray(self.cloth_indices, dtype=np.int32)
        if self.collider_indices is not None:
            self.collider_indices = np.asarray(self.collider_indices, dtype=np.int32)
        if self.rest_offsets is not None:
            self.rest_offsets = np.asarray(self.rest_offsets, dtype=np.float32)



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

        if NUMBA_AVAILABLE:
            _project_distance_numba(
                positions,
                inv_masses,
                self.edges,
                self.rest_lengths,
                self.lambdas,
                self.inv_deg,
                float(self.compliance),
                float(dt),
            )
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

        if NUMBA_AVAILABLE:
            _project_bending_numba(
                positions,
                inv_masses,
                self.quads,
                self.rest_distances,
                self.lambdas,
                self.inv_deg,
                float(self.compliance),
                float(dt),
            )
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
