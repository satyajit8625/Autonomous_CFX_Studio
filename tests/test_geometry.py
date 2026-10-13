"""Tests for MeshGeometry, Normals, Edges, and Bounding Box."""

import numpy as np
import pytest
from autonomous_cfx.core.geometry import MeshGeometry, BoundingBox


def test_bounding_box():
    min_pt = np.array([-1.0, 0.0, -2.0], dtype=np.float32)
    max_pt = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    bbox = BoundingBox(min_pt, max_pt)

    assert np.allclose(bbox.center, [0.0, 1.0, 0.0])
    assert np.allclose(bbox.extents, [2.0, 2.0, 4.0])

    test_pts = np.array([
        [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0],
    ], dtype=np.float32)
    mask = bbox.contains(test_pts)
    assert mask[0] == True
    assert mask[1] == False


def test_mesh_normals_and_edges():
    # Simple quad with 2 triangles
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    mesh = MeshGeometry(name="quad", vertices=verts, faces=faces)
    assert mesh.num_vertices == 4
    assert mesh.num_faces == 2

    # Normals should point along +Z
    normals = mesh.compute_normals()
    assert normals.shape == (4, 3)
    for n in normals:
        assert np.isclose(n[2], 1.0, atol=1e-3)

    # Unique edges (4 outer + 1 diagonal = 5)
    edges, rest_lens = mesh.compute_edges()
    assert len(edges) == 5
    assert len(rest_lens) == 5

    # Bending pairs (1 interior shared edge between the 2 triangles)
    quads, bend_rests = mesh.compute_bending_pairs()
    assert len(quads) == 1
    assert len(bend_rests) == 1


def test_mesh_vertex_masses():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshGeometry(vertices=verts, faces=faces)

    masses = mesh.compute_vertex_masses(surface_density=1.0)
    assert len(masses) == 3
    assert np.all(masses > 0.0)
    assert np.isclose(np.sum(masses), 0.5, atol=1e-3)  # triangle area is 0.5
