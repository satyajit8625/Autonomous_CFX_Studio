"""Autonomous CFX Studio - Geometry and Mesh Representation.
Provides high-performance vector math, normals, edge graph extraction, and bounding box computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh


@dataclass
class BoundingBox:
    """Axis-Aligned Bounding Box (AABB)."""
    min_point: np.ndarray
    max_point: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.min_point + self.max_point) * 0.5

    @property
    def extents(self) -> np.ndarray:
        return self.max_point - self.min_point

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Returns boolean mask for points inside the bounding box."""
        return np.all(
            (points >= (self.min_point - margin)) & (points <= (self.max_point + margin)),
            axis=-1,
        )


@dataclass
class MeshGeometry:
    """Manifold or open surface mesh representation for CFX simulation."""
    name: str = "mesh"
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    uvs: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    vertex_groups: Dict[str, np.ndarray] = field(default_factory=dict)
    face_sets: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.vertices, np.ndarray):
            self.vertices = np.asarray(self.vertices, dtype=np.float32)
        else:
            self.vertices = self.vertices.astype(np.float32)

        if not isinstance(self.faces, np.ndarray):
            self.faces = np.asarray(self.faces, dtype=np.int32)
        else:
            self.faces = self.faces.astype(np.int32)

        if self.normals is None and len(self.vertices) > 0 and len(self.faces) > 0:
            self.compute_normals()

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def copy(self) -> MeshGeometry:
        """Return a deep copy of the mesh geometry."""
        return MeshGeometry(
            name=self.name,
            vertices=np.copy(self.vertices),
            faces=np.copy(self.faces),
            uvs=np.copy(self.uvs) if self.uvs is not None else None,
            normals=np.copy(self.normals) if self.normals is not None else None,
            vertex_groups={k: np.copy(v) for k, v in self.vertex_groups.items()},
            face_sets={k: np.copy(v) for k, v in self.face_sets.items()},
        )

    def get_bounding_box(self) -> BoundingBox:
        """Compute the axis-aligned bounding box."""
        if len(self.vertices) == 0:
            return BoundingBox(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        return BoundingBox(
            min_point=np.min(self.vertices, axis=0),
            max_point=np.max(self.vertices, axis=0),
        )

    def compute_normals(self) -> np.ndarray:
        """Vectorized computation of smooth vertex normals."""
        if len(self.vertices) == 0 or len(self.faces) == 0:
            self.normals = np.zeros_like(self.vertices)
            return self.normals

        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Face normals weighted by face area (cross product magnitude)
        face_cross = np.cross(v1 - v0, v2 - v0)
        
        vert_normals = np.zeros_like(self.vertices, dtype=np.float32)
        for i in range(3):
            np.add.at(vert_normals, self.faces[:, i], face_cross)

        # Normalize
        norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normals = (vert_normals / norms).astype(np.float32)
        return self.normals

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract unique undirected edges and their rest lengths.
        Returns:
            edges: (E, 2) int array of vertex index pairs.
            rest_lengths: (E,) float array of rest edge lengths.
        """
        if len(self.faces) == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        # 3 edges per triangle
        e1 = self.faces[:, [0, 1]]
        e2 = self.faces[:, [1, 2]]
        e3 = self.faces[:, [2, 0]]
        all_edges = np.vstack([e1, e2, e3])

        # Sort vertices per edge so (i, j) == (j, i)
        sorted_edges = np.sort(all_edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)

        # Rest lengths
        diffs = self.vertices[unique_edges[:, 0]] - self.vertices[unique_edges[:, 1]]
        rest_lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)

        return unique_edges.astype(np.int32), rest_lengths

    def compute_bending_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract adjacent face pairs sharing an edge for dihedral/isometric bending.
        For each shared interior edge between triangle (v0, v1, v2) and (v1, v0, v3),
        returns quadruplet (v0, v1, v2, v3) or opposite vertex pairs (v2, v3) with rest distance.
        """
        if len(self.faces) == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        edge_dict: Dict[Tuple[int, int], List[int]] = {}
        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                u, v = int(face[i]), int(face[(i + 1) % 3])
                opp = int(face[(i + 2) % 3])
                key = (min(u, v), max(u, v))
                if key not in edge_dict:
                    edge_dict[key] = []
                edge_dict[key].append(opp)

        quads = []
        rest_cross_lengths = []
        for (u, v), opp_list in edge_dict.items():
            if len(opp_list) == 2:
                # Interior shared edge: shared vertices u, v and opposite vertices o1, o2
                o1, o2 = opp_list[0], opp_list[1]
                quads.append([u, v, o1, o2])
                dist = np.linalg.norm(self.vertices[o1] - self.vertices[o2])
                rest_cross_lengths.append(dist)

        if not quads:
            return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        return (
            np.array(quads, dtype=np.int32),
            np.array(rest_cross_lengths, dtype=np.float32),
        )

    def compute_vertex_masses(self, surface_density: float = 0.3) -> np.ndarray:
        """Compute lumped vertex masses based on face areas and surface density (kg/m^2).
        Returns:
            masses: (N,) float array.
        """
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return np.ones(len(self.vertices), dtype=np.float32) * 0.1

        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)

        masses = np.zeros(len(self.vertices), dtype=np.float32)
        # Each vertex gets 1/3 of the area of incident faces
        one_third_area = face_areas * (surface_density / 3.0)
        for i in range(3):
            np.add.at(masses, self.faces[:, i], one_third_area)

        # Minimum clamp to prevent zero mass
        masses = np.maximum(masses, 1e-4)
        return masses.astype(np.float32)

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert to a trimesh.Trimesh object."""
        return trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            process=False,
        )

    @classmethod
    def from_trimesh(cls, tm: trimesh.Trimesh, name: str = "imported_mesh") -> MeshGeometry:
        """Create MeshGeometry from a trimesh.Trimesh object."""
        return cls(
            name=name,
            vertices=np.array(tm.vertices, dtype=np.float32),
            faces=np.array(tm.faces, dtype=np.int32),
            normals=np.array(tm.vertex_normals, dtype=np.float32) if tm.vertex_normals is not None else None,
        )
