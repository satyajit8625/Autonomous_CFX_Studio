"""Autonomous CFX Studio - Spatial Hashing Cloth Self-Collision Subsystem.
Resolves cloth-to-cloth self-intersections and folding contact in O(N) linear time.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _hash_cell(cx: int, cy: int, cz: int, table_size: int) -> int:
        # Large prime spatial hash
        h = (cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)
        return int(np.abs(h)) % table_size

    @njit(fastmath=True)
    def _resolve_self_collision_numba(
        positions: np.ndarray,
        inv_masses: np.ndarray,
        adj_matrix: np.ndarray,
        cell_size: float,
        min_dist: float,
        iterations: int,
    ) -> int:
        N = positions.shape[0]
        table_size = max(N * 4, 1024)
        min_dist_sq = min_dist * min_dist
        inv_cell = 1.0 / cell_size
        head = np.full(table_size, -1, dtype=np.int32)
        next_node = np.zeros(N, dtype=np.int32)
        coords = np.zeros((N, 3), dtype=np.int32)

        total_resolved = 0

        for _iter in range(iterations):
            head.fill(-1)
            for i in range(N):
                cx = int(np.floor(positions[i, 0] * inv_cell))
                cy = int(np.floor(positions[i, 1] * inv_cell))
                cz = int(np.floor(positions[i, 2] * inv_cell))
                coords[i, 0] = cx
                coords[i, 1] = cy
                coords[i, 2] = cz
                h = _hash_cell(cx, cy, cz, table_size)
                next_node[i] = head[h]
                head[h] = i

            displacements = np.zeros_like(positions)
            counts = np.zeros(N, dtype=np.int32)

            for i in range(N):
                w_i = inv_masses[i]
                cx = coords[i, 0]
                cy = coords[i, 1]
                cz = coords[i, 2]

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            h = _hash_cell(cx + dx, cy + dy, cz + dz, table_size)
                            j = head[h]
                            while j != -1:
                                if i < j and not adj_matrix[i, j]:
                                    w_j = inv_masses[j]
                                    w_sum = w_i + w_j
                                    if w_sum > 1e-7:
                                        rx = positions[i, 0] - positions[j, 0]
                                        ry = positions[i, 1] - positions[j, 1]
                                        rz = positions[i, 2] - positions[j, 2]
                                        d_sq = rx * rx + ry * ry + rz * rz
                                        if d_sq < min_dist_sq and d_sq > 1e-12:
                                            d = np.sqrt(d_sq)
                                            penetration = min_dist - d
                                            inv_d = 1.0 / d
                                            nx = rx * inv_d
                                            ny = ry * inv_d
                                            nz = rz * inv_d

                                            corr_i = (w_i / w_sum) * penetration * 0.5
                                            corr_j = (w_j / w_sum) * penetration * 0.5

                                            displacements[i, 0] += nx * corr_i
                                            displacements[i, 1] += ny * corr_i
                                            displacements[i, 2] += nz * corr_i
                                            counts[i] += 1

                                            displacements[j, 0] -= nx * corr_j
                                            displacements[j, 1] -= ny * corr_j
                                            displacements[j, 2] -= nz * corr_j
                                            counts[j] += 1
                                            total_resolved += 1
                                j = next_node[j]

            for i in range(N):
                if counts[i] > 0:
                    scale = 1.0 / counts[i]
                    positions[i, 0] += displacements[i, 0] * scale
                    positions[i, 1] += displacements[i, 1] * scale
                    positions[i, 2] += displacements[i, 2] * scale

        return total_resolved
else:
    _resolve_self_collision_numba = None


class SpatialHashSelfCollision:
    """Fast spatial hashing grid for vertex-vertex and fold self-collision resolution."""

    def __init__(
        self,
        thickness: float = 0.005,  # 5mm collision thickness
        edges: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        num_vertices: int = 0,
    ):
        self.thickness = thickness
        self.min_dist = 2.0 * thickness
        self.cell_size = max(self.min_dist * 1.5, 0.01)

        # Precompute topologically adjacent vertex pairs to exclude from self-collision
        self.topological_neighbors: List[Set[int]] = [set() for _ in range(num_vertices)]
        self.adj_matrix: np.ndarray = np.zeros((num_vertices, num_vertices), dtype=np.bool_)

        if edges is not None and len(edges) > 0:
            for u, v in edges:
                if u < num_vertices and v < num_vertices:
                    self.topological_neighbors[u].add(v)
                    self.topological_neighbors[v].add(u)
                    self.adj_matrix[u, v] = True
                    self.adj_matrix[v, u] = True

        if faces is not None and len(faces) > 0:
            for f in faces:
                for i in range(3):
                    u = f[i]
                    v = f[(i + 1) % 3]
                    if u < num_vertices and v < num_vertices:
                        self.topological_neighbors[u].add(v)
                        self.topological_neighbors[v].add(u)
                        self.adj_matrix[u, v] = True
                        self.adj_matrix[v, u] = True

    def resolve_self_collisions(
        self,
        positions: np.ndarray,
        inv_masses: np.ndarray,
        iterations: int = 2,
    ) -> int:
        """Detect and resolve self-intersections. Returns total collision pairs resolved."""
        N = len(positions)
        if N == 0:
            return 0

        if NUMBA_AVAILABLE and hasattr(self, "adj_matrix") and len(self.adj_matrix) == N:
            return _resolve_self_collision_numba(
                positions,
                inv_masses,
                self.adj_matrix,
                float(self.cell_size),
                float(self.min_dist),
                int(iterations),
            )

        total_resolved = 0
        min_dist_sq = self.min_dist * self.min_dist

        for _ in range(iterations):
            # 1. Build grid
            grid: Dict[Tuple[int, int, int], List[int]] = {}
            coords = np.floor(positions / self.cell_size).astype(np.int32)

            for i in range(N):
                key = (int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))
                if key not in grid:
                    grid[key] = []
                grid[key].append(i)

            # 2. Check 27 adjacent cells for neighbor vertices
            displacements = np.zeros_like(positions)
            counts = np.zeros(N, dtype=np.int32)

            neighbor_offsets = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            ]

            checked_pairs: Set[Tuple[int, int]] = set()

            for key, v_indices in grid.items():
                for dx, dy, dz in neighbor_offsets:
                    n_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if n_key not in grid:
                        continue
                    n_indices = grid[n_key]

                    for i in v_indices:
                        w_i = inv_masses[i]
                        for j in n_indices:
                            if i >= j:
                                continue
                            pair = (i, j)
                            if pair in checked_pairs:
                                continue
                            checked_pairs.add(pair)

                            # Skip connected vertices in mesh topology
                            if i < len(self.topological_neighbors) and j in self.topological_neighbors[i]:
                                continue

                            w_j = inv_masses[j]
                            w_sum = w_i + w_j
                            if w_sum < 1e-7:
                                continue

                            diff = positions[i] - positions[j]
                            dist_sq = float(np.sum(diff * diff))

                            if dist_sq < min_dist_sq and dist_sq > 1e-9:
                                dist = np.sqrt(dist_sq)
                                overlap = self.min_dist - dist
                                n = diff / dist

                                corr_i = (w_i / w_sum) * overlap * n
                                corr_j = -(w_j / w_sum) * overlap * n

                                displacements[i] += corr_i
                                displacements[j] += corr_j
                                counts[i] += 1
                                counts[j] += 1
                                total_resolved += 1

            # Apply normalized self-collision impulses
            active = counts > 0
            if np.any(active):
                displacements[active] /= counts[active, None]
                positions += displacements

        return total_resolved
