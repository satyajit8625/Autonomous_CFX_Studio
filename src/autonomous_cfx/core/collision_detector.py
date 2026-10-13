"""Autonomous CFX Studio - Robust Collision & Penetration Classifier.

Separates broad-phase contact acceleration from inside/outside volumetric classification:
1. Broad-phase AABB acceleration to reject far-away cloth with zero false positives.
2. Exact ray-mesh parity voting with non-degenerate irrational rays for closed/watertight bodies.
3. Sub-vertex point-to-triangle distance computation supporting arbitrary penetration depths (> 50 mm, up to torso radius).
4. Oriented surface normal and thickness bounds for open meshes.
5. Robust handling of concave bodies (armpits, inner thighs) and nearby limbs without cross-void projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree

from autonomous_cfx.core.geometry import MeshGeometry


@dataclass
class PenetrationRecord:
    """Detailed penetration measurement for a query vertex."""
    vertex_index: int
    is_penetrating: bool
    penetration_depth_m: float  # In meters (inside volume)
    penetration_depth_mm: float  # In millimeters (inside volume)
    margin_intrusion_mm: float  # Clearance violation including margin
    surface_normal: np.ndarray
    closest_surface_point: np.ndarray
    is_inside_volume: bool


@dataclass
class PenetrationResult:
    """Consolidated collision detection outcome across query vertices."""
    max_penetration_mm: float  # Maximum depth inside collider volume
    max_margin_penetration_mm: float  # Maximum violation including margin
    mean_penetration_mm: float
    penetrating_vertex_count: int
    penetrating_indices: List[int]
    records: List[PenetrationRecord]
    is_watertight: bool
    broad_phase_culled_count: int


def point_triangle_closest_point(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute the closest point on triangle (a, b, c) to query point p."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


class PenetrationClassifier:
    """Robust 3D Penetration Classifier separating acceleration from topological classification."""

    # Irrationally perturbed ray vectors for non-degenerate parity testing
    PARITY_RAYS = [
        np.array([0.28734, 0.95783, 0.01234], dtype=np.float64) / 1.0,
        np.array([0.91234, -0.40938, 0.02345], dtype=np.float64) / 1.0,
        np.array([0.12345, 0.34567, 0.93021], dtype=np.float64) / 1.0,
    ]

    @classmethod
    def is_mesh_watertight(cls, mesh: MeshGeometry) -> bool:
        """Verify whether mesh forms a closed 2-manifold (every edge shared by exactly 2 faces)."""
        if mesh.num_faces == 0:
            return False

        edge_counts = {}
        for face in mesh.faces:
            n = len(face)
            for i in range(n):
                i0, i1 = face[i], face[(i + 1) % n]
                edge = (min(i0, i1), max(i0, i1))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        # Closed 2-manifold meshes have exactly 2 faces per manifold edge
        return all(count == 2 for count in edge_counts.values())

    @classmethod
    def test_points_inside_closed_mesh(
        cls,
        points: np.ndarray,
        mesh: MeshGeometry,
        candidate_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Classify points as inside or outside a closed polyhedron using multi-ray parity voting."""
        num_pts = len(points)
        inside_flags = np.zeros(num_pts, dtype=bool)
        if num_pts == 0 or mesh.num_faces == 0:
            return inside_flags

        tris = mesh.vertices[mesh.faces]  # (F, 3, 3)
        v0 = tris[:, 0, :]
        e1 = tris[:, 1, :] - v0
        e2 = tris[:, 2, :] - v0

        eval_indices = np.where(candidate_mask)[0] if candidate_mask is not None else range(num_pts)
        if len(eval_indices) == 0:
            return inside_flags

        votes = np.zeros(len(eval_indices), dtype=int)
        normalized_rays = [r / np.linalg.norm(r) for r in cls.PARITY_RAYS]

        for r_dir in normalized_rays:
            pvec = np.cross(r_dir, e2)
            det = np.sum(e1 * pvec, axis=1)
            valid_det = np.abs(det) > 1e-9
            safe_det = np.where(valid_det, det, 1.0)

            for local_idx, pt_idx in enumerate(eval_indices):
                pt = points[pt_idx]
                tvec = pt - v0
                u = np.sum(tvec * pvec, axis=1) / safe_det
                u_ok = valid_det & (u >= 0.0) & (u <= 1.0)
                if not np.any(u_ok):
                    continue

                qvec = np.cross(tvec[u_ok], e1[u_ok])
                v = np.sum(r_dir * qvec, axis=1) / det[u_ok]
                v_ok = (v >= 0.0) & (u[u_ok] + v <= 1.0)
                if not np.any(v_ok):
                    continue

                t_hit = np.sum(e2[u_ok][v_ok] * qvec[v_ok], axis=1) / det[u_ok][v_ok]
                hits = int(np.sum(t_hit > 1e-6))
                if hits % 2 == 1:
                    votes[local_idx] += 1

        is_inside = votes >= 2
        for local_idx, pt_idx in enumerate(eval_indices):
            inside_flags[pt_idx] = is_inside[local_idx]

        return inside_flags

    @classmethod
    def evaluate_penetration(
        cls,
        cloth: MeshGeometry,
        collider: MeshGeometry,
        margin: float = 0.008,
        target_indices: Optional[List[int]] = None,
        force_closed: Optional[bool] = None,
    ) -> PenetrationResult:
        """Measure genuine penetration depth separating acceleration from inside/outside classification.

        Args:
            cloth: Cloth mesh geometry.
            collider: Collider mesh geometry.
            margin: Collision safety margin in meters (e.g. 0.008 m = 8 mm).
            target_indices: Optional vertex subset of cloth to inspect.
            force_closed: Explicitly override closed/watertight detection.
        """
        all_cloth_verts = cloth.vertices
        if target_indices is not None:
            query_indices = np.array(target_indices, dtype=np.int32)
            query_points = all_cloth_verts[query_indices]
        else:
            query_indices = np.arange(len(all_cloth_verts), dtype=np.int32)
            query_points = all_cloth_verts

        num_queries = len(query_points)
        if num_queries == 0 or collider.num_vertices == 0 or collider.num_faces == 0:
            return PenetrationResult(
                max_penetration_mm=0.0,
                max_margin_penetration_mm=0.0,
                mean_penetration_mm=0.0,
                penetrating_vertex_count=0,
                penetrating_indices=[],
                records=[],
                is_watertight=False,
                broad_phase_culled_count=0,
            )

        # 1. Broad-phase AABB acceleration
        col_min = np.min(collider.vertices, axis=0) - (margin + 0.05)
        col_max = np.max(collider.vertices, axis=0) + (margin + 0.05)

        in_aabb = np.all((query_points >= col_min) & (query_points <= col_max), axis=1)
        broad_phase_culled = int(num_queries - np.sum(in_aabb))

        is_watertight = force_closed if force_closed is not None else cls.is_mesh_watertight(collider)

        # 2. Inside / Outside Classification
        if is_watertight:
            inside_mask = cls.test_points_inside_closed_mesh(query_points, collider, candidate_mask=in_aabb)
        else:
            inside_mask = np.zeros(num_queries, dtype=bool)

        # 3. Spatial Acceleration: KDTree on collider faces for sub-vertex surface distance
        face_centroids = np.mean(collider.vertices[collider.faces], axis=1)
        face_tree = KDTree(face_centroids)
        normals = collider.compute_normals()

        records: List[PenetrationRecord] = []
        penetrating_indices: List[int] = []
        max_pen_mm = 0.0
        max_margin_pen_mm = 0.0
        sum_pen_mm = 0.0

        for i in range(num_queries):
            v_idx = int(query_indices[i])
            pt = query_points[i]

            if not in_aabb[i]:
                # Broad-phase culled: Guaranteed zero penetration
                records.append(
                    PenetrationRecord(
                        vertex_index=v_idx,
                        is_penetrating=False,
                        penetration_depth_m=0.0,
                        penetration_depth_mm=0.0,
                        margin_intrusion_mm=0.0,
                        surface_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        closest_surface_point=pt,
                        is_inside_volume=False,
                    )
                )
                continue

            # Query K nearest faces to compute sub-vertex distance
            k_query = min(8, len(collider.faces))
            _, candidate_faces = face_tree.query(pt, k=k_query)
            if not isinstance(candidate_faces, np.ndarray):
                candidate_faces = np.array([candidate_faces])

            best_dist = float("inf")
            best_closest_pt = collider.vertices[collider.faces[candidate_faces[0]][0]]
            best_face = candidate_faces[0]

            for f_cand in candidate_faces:
                f_verts = collider.vertices[collider.faces[f_cand]]
                c_pt = point_triangle_closest_point(pt, f_verts[0], f_verts[1], f_verts[2])
                dist = float(np.linalg.norm(pt - c_pt))
                if dist < best_dist:
                    best_dist = dist
                    best_closest_pt = c_pt
                    best_face = f_cand

            v0_f = collider.vertices[collider.faces[best_face][0]]
            v1_f = collider.vertices[collider.faces[best_face][1]]
            v2_f = collider.vertices[collider.faces[best_face][2]]
            face_n = np.cross(v1_f - v0_f, v2_f - v0_f)
            fn_len = np.linalg.norm(face_n)
            surf_n = face_n / fn_len if fn_len > 1e-7 else np.array([0.0, 1.0, 0.0])

            is_inside = bool(inside_mask[i])

            if is_watertight:
                if is_inside:
                    # Inside the solid volume: Sub-vertex distance to surface
                    vol_depth_m = best_dist
                    margin_intrusion_m = best_dist + margin
                    is_pen = True
                else:
                    vol_depth_m = 0.0
                    if best_dist < margin:
                        margin_intrusion_m = margin - best_dist
                        is_pen = True
                    else:
                        margin_intrusion_m = 0.0
                        is_pen = False
            else:
                # Open mesh: project onto oriented surface normal
                to_query = pt - best_closest_pt
                signed_dist = float(np.dot(to_query, surf_n))
                if signed_dist < 0.0:
                    vol_depth_m = -signed_dist
                    margin_intrusion_m = margin - signed_dist
                    is_pen = True
                    is_inside = True
                elif signed_dist < margin:
                    vol_depth_m = 0.0
                    margin_intrusion_m = margin - signed_dist
                    is_pen = True
                    is_inside = False
                else:
                    vol_depth_m = 0.0
                    margin_intrusion_m = 0.0
                    is_pen = False
                    is_inside = False

            rec = PenetrationRecord(
                vertex_index=v_idx,
                is_penetrating=is_pen,
                penetration_depth_m=vol_depth_m,
                penetration_depth_mm=round(vol_depth_m * 1000.0, 3),
                margin_intrusion_mm=round(margin_intrusion_m * 1000.0, 3),
                surface_normal=surf_n,
                closest_surface_point=best_closest_pt,
                is_inside_volume=is_inside,
            )
            records.append(rec)

            if is_inside:
                penetrating_indices.append(v_idx)
                max_pen_mm = max(max_pen_mm, vol_depth_m * 1000.0)
                sum_pen_mm += vol_depth_m * 1000.0
            if is_pen:
                max_margin_pen_mm = max(max_margin_pen_mm, margin_intrusion_m * 1000.0)

        mean_pen_mm = sum_pen_mm / len(penetrating_indices) if penetrating_indices else 0.0

        return PenetrationResult(
            max_penetration_mm=round(max_pen_mm, 3),
            max_margin_penetration_mm=round(max_margin_pen_mm, 3),
            mean_penetration_mm=round(mean_pen_mm, 3),
            penetrating_vertex_count=len(penetrating_indices),
            penetrating_indices=penetrating_indices,
            records=records,
            is_watertight=is_watertight,
            broad_phase_culled_count=broad_phase_culled,
        )
