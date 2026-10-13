"""Autonomous CFX Studio - Spatial Hashing Cloth Self-Collision Subsystem.
Resolves cloth-to-cloth self-intersections and folding contact in O(N) linear time.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np


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
        if edges is not None and len(edges) > 0:
            for u, v in edges:
                if u < num_vertices and v < num_vertices:
                    self.topological_neighbors[u].add(v)
                    self.topological_neighbors[v].add(u)

        if faces is not None and len(faces) > 0:
            for f in faces:
                for i in range(3):
                    u = f[i]
                    v = f[(i + 1) % 3]
                    if u < num_vertices and v < num_vertices:
                        self.topological_neighbors[u].add(v)
                        self.topological_neighbors[v].add(u)

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
