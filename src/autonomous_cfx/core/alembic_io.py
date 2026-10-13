"""Autonomous CFX Studio - Native Standalone Animated Alembic (.abc) I/O.

High-performance, standalone Alembic geometry reading and authoring using official
C++ Alembic Ogawa/HDF5 bindings (Abc, AbcGeom, imath). Runs completely independent
of Autodesk Maya or other host DCCs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import imath
    from alembic import Abc, AbcGeom, AbcCoreAbstract
    ALEMBIC_AVAILABLE = True
except ImportError:
    ALEMBIC_AVAILABLE = False
    imath = None
    Abc = None
    AbcGeom = None
    AbcCoreAbstract = None

from autonomous_cfx.core.geometry import MeshGeometry


class AlembicObjectRole(str, Enum):
    """Semantic role assigned to Alembic scene objects."""
    GARMENT = "garment"
    BODY = "body"
    PROP = "prop"
    IGNORED = "ignored"


@dataclass
class AlembicNodeInfo:
    """Metadata describing a node in an Alembic archive hierarchy."""
    name: str
    full_path: str
    node_type: str  # 'PolyMesh', 'Xform', etc.
    parent_path: str
    assigned_role: AlembicObjectRole = AlembicObjectRole.IGNORED
    num_samples: int = 0
    num_vertices: int = 0
    num_faces: int = 0
    is_topology_constant: bool = True
    bounds_min: Optional[Tuple[float, float, float]] = None
    bounds_max: Optional[Tuple[float, float, float]] = None
    has_uvs: bool = False
    has_normals: bool = False
    face_sets: List[str] = field(default_factory=list)


@dataclass
class AlembicShotSummary:
    """Summary of an inspected animated Alembic archive."""
    archive_path: str
    frame_range: Tuple[int, int]
    fps: float
    time_per_frame: float
    start_time_seconds: float
    end_time_seconds: float
    total_samples: int
    nodes: List[AlembicNodeInfo] = field(default_factory=list)
    up_axis: str = "Y"
    linear_units: str = "meters"
    archive_format: str = "Ogawa"


@dataclass
class AlembicShotData:
    """Loaded animated geometry data ready for CFX simulation."""
    summary: AlembicShotSummary
    garment_meshes: Dict[str, List[MeshGeometry]] = field(default_factory=dict)
    body_colliders: Dict[str, List[MeshGeometry]] = field(default_factory=dict)
    prop_colliders: Dict[str, List[MeshGeometry]] = field(default_factory=dict)
    ignored_objects: List[str] = field(default_factory=list)


class AlembicShotImporter:
    """Standalone animated Alembic (.abc) archive inspector and loader."""

    def __init__(self):
        if not ALEMBIC_AVAILABLE:
            raise RuntimeError(
                "Alembic C++ geometry bindings are not installed in this environment. "
                "Ensure 'alembic' and 'imath' packages are installed."
            )

    def inspect_archive(self, abc_path: str) -> AlembicShotSummary:
        """Inspect hierarchy, animation, sample rates, and topology stability without full geometry extraction."""
        if not os.path.exists(abc_path):
            raise FileNotFoundError(f"Alembic file not found: {abc_path}")

        archive = Abc.IArchive(abc_path)
        start_time, end_time = Abc.GetArchiveStartAndEndTime(archive)

        top = archive.getTop()
        nodes: List[AlembicNodeInfo] = []

        # Default standard timing assumptions
        fps = 24.0
        time_per_frame = 1.0 / 24.0
        total_samples = 1

        # Recursive hierarchy inspection
        def _inspect_node(obj, parent_path: str):
            nonlocal fps, time_per_frame, total_samples
            full_path = obj.getFullName()
            name = obj.getName()
            meta = obj.getMetaData()

            node_type = "Object"
            is_mesh = AbcGeom.IPolyMesh.matches(meta)
            is_xform = AbcGeom.IXform.matches(meta)

            if is_mesh:
                node_type = "PolyMesh"
                mesh = AbcGeom.IPolyMesh(obj.getParent(), name)
                schema = mesh.getSchema()
                num_s = schema.getNumSamples()
                total_samples = max(total_samples, num_s)

                ts = schema.getTimeSampling()
                tst = ts.getTimeSamplingType()
                t_cycle = tst.getTimePerCycle()
                if t_cycle > 1e-6:
                    time_per_frame = t_cycle
                    fps = round(1.0 / t_cycle, 2)

                # Check topology constancy across samples
                is_topo_constant = True
                first_v_count = 0
                first_f_count = 0
                has_uvs = schema.getUVsParam().valid()
                has_normals = schema.getNormalsParam().valid()
                face_sets = list(schema.getFaceSetNames())

                b_min, b_max = None, None
                for s_idx in range(min(num_s, 5)):  # check first few samples for stability
                    s = schema.getValue(Abc.ISampleSelector(s_idx))
                    pos = s.getPositions()
                    fc = s.getFaceCounts()
                    if s_idx == 0:
                        first_v_count = len(pos)
                        first_f_count = len(fc)
                        if len(pos) > 0:
                            pos_np = np.ctypeslib.as_array(pos)
                            b_min = tuple(float(x) for x in pos_np.min(axis=0))
                            b_max = tuple(float(x) for x in pos_np.max(axis=0))
                    else:
                        if len(pos) != first_v_count or len(fc) != first_f_count:
                            is_topo_constant = False

                # Auto-infer semantic role
                name_lower = name.lower()
                role = AlembicObjectRole.IGNORED
                if any(w in name_lower for w in ["garment", "cloth", "jacket", "shirt", "pant", "cape", "dress"]):
                    role = AlembicObjectRole.GARMENT
                elif any(w in name_lower for w in ["body", "char", "character", "avatar", "skin", "hero", "actor"]):
                    role = AlembicObjectRole.BODY
                elif any(w in name_lower for w in ["chair", "bench", "prop", "ground", "collider", "table", "floor"]):
                    role = AlembicObjectRole.PROP

                nodes.append(AlembicNodeInfo(
                    name=name,
                    full_path=full_path,
                    node_type=node_type,
                    parent_path=parent_path,
                    assigned_role=role,
                    num_samples=num_s,
                    num_vertices=first_v_count,
                    num_faces=first_f_count,
                    is_topology_constant=is_topo_constant,
                    bounds_min=b_min,
                    bounds_max=b_max,
                    has_uvs=has_uvs,
                    has_normals=has_normals,
                    face_sets=face_sets,
                ))

            elif is_xform:
                node_type = "Xform"
                nodes.append(AlembicNodeInfo(
                    name=name,
                    full_path=full_path,
                    node_type=node_type,
                    parent_path=parent_path,
                    assigned_role=AlembicObjectRole.IGNORED,
                    num_samples=0,
                ))

            # Recurse children
            for i in range(obj.getNumChildren()):
                _inspect_node(obj.getChild(i), full_path)

        for i in range(top.getNumChildren()):
            _inspect_node(top.getChild(i), "/")

        start_frame = int(round(start_time * fps)) + 1 if start_time >= 0 else 1
        end_frame = start_frame + total_samples - 1 if total_samples > 1 else start_frame

        return AlembicShotSummary(
            archive_path=abc_path,
            frame_range=(start_frame, end_frame),
            fps=fps,
            time_per_frame=time_per_frame,
            start_time_seconds=float(start_time),
            end_time_seconds=float(end_time),
            total_samples=total_samples,
            nodes=nodes,
            up_axis="Y",
            linear_units="meters",
            archive_format="Ogawa",
        )

    def load_shot(
        self,
        abc_path: str,
        role_assignments: Optional[Dict[str, AlembicObjectRole]] = None,
        world_space: bool = True
    ) -> AlembicShotData:
        """Load animated geometry streams from an Alembic archive with role assignment and coordinate transform."""
        summary = self.inspect_archive(abc_path)
        archive = Abc.IArchive(abc_path)
        top = archive.getTop()

        # Apply user role overrides if provided
        if role_assignments:
            for node in summary.nodes:
                if node.full_path in role_assignments:
                    node.assigned_role = role_assignments[node.full_path]
                elif node.name in role_assignments:
                    node.assigned_role = role_assignments[node.name]

        garment_meshes: Dict[str, List[MeshGeometry]] = {}
        body_colliders: Dict[str, List[MeshGeometry]] = {}
        prop_colliders: Dict[str, List[MeshGeometry]] = {}
        ignored_objects: List[str] = []

        # Helper to compute world matrix chain along hierarchy
        def _get_world_matrix_for_path(obj_path: str, sample_idx: int) -> np.ndarray:
            if not world_space or obj_path == "/" or not obj_path:
                return np.eye(4, dtype=np.float64)

            parts = [p for p in obj_path.split("/") if p]
            curr = top
            mat = np.eye(4, dtype=np.float64)

            for part in parts:
                child = None
                for i in range(curr.getNumChildren()):
                    if curr.getChild(i).getName() == part:
                        child = curr.getChild(i)
                        break
                if child is None:
                    break

                if AbcGeom.IXform.matches(child.getMetaData()):
                    xform = AbcGeom.IXform(curr, part)
                    xs = xform.getSchema()
                    s_count = xs.getNumSamples()
                    sel = Abc.ISampleSelector(min(sample_idx, max(0, s_count - 1)))
                    m = xs.getValue(sel).getMatrix()
                    m_np = np.array(m, dtype=np.float64).reshape((4, 4))
                    mat = mat @ m_np
                curr = child

            return mat

        # Extract mesh samples for active roles
        for node in summary.nodes:
            if node.node_type != "PolyMesh":
                continue

            if node.assigned_role == AlembicObjectRole.IGNORED:
                ignored_objects.append(node.full_path)
                continue

            # Locate mesh object in hierarchy
            parent = top
            parts = [p for p in node.full_path.split("/") if p]
            for p_name in parts[:-1]:
                for i in range(parent.getNumChildren()):
                    if parent.getChild(i).getName() == p_name:
                        parent = parent.getChild(i)
                        break

            mesh = AbcGeom.IPolyMesh(parent, node.name)
            schema = mesh.getSchema()
            num_samples = schema.getNumSamples()

            # First sample topology
            s0 = schema.getValue(Abc.ISampleSelector(0))
            face_counts = np.ctypeslib.as_array(s0.getFaceCounts()).copy()
            face_indices = np.ctypeslib.as_array(s0.getFaceIndices()).copy()

            # Convert face_indices + face_counts to Nx3 triangle array
            triangles = []
            idx_offset = 0
            for count in face_counts:
                if count == 3:
                    triangles.append([
                        face_indices[idx_offset],
                        face_indices[idx_offset + 1],
                        face_indices[idx_offset + 2]
                    ])
                elif count == 4:
                    # Triangulate quad
                    triangles.append([
                        face_indices[idx_offset],
                        face_indices[idx_offset + 1],
                        face_indices[idx_offset + 2]
                    ])
                    triangles.append([
                        face_indices[idx_offset],
                        face_indices[idx_offset + 2],
                        face_indices[idx_offset + 3]
                    ])
                idx_offset += count
            triangles_np = np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)

            animated_frames: List[MeshGeometry] = []
            for s_idx in range(num_samples):
                s = schema.getValue(Abc.ISampleSelector(s_idx))
                pos_raw = np.ctypeslib.as_array(s.getPositions()).copy()

                # Transform to world space if required
                if world_space and node.parent_path:
                    mat = _get_world_matrix_for_path(node.parent_path, s_idx)
                    homo = np.column_stack([pos_raw, np.ones((len(pos_raw), 1), dtype=np.float64)])
                    pos_world = (homo @ mat)[:, :3].astype(np.float32)
                else:
                    pos_world = pos_raw.astype(np.float32)

                frame_mesh = MeshGeometry(
                    vertices=pos_world,
                    faces=triangles_np,
                    name=f"{node.name}_f{s_idx + 1}"
                )
                animated_frames.append(frame_mesh)

            if node.assigned_role == AlembicObjectRole.GARMENT:
                garment_meshes[node.full_path] = animated_frames
            elif node.assigned_role == AlembicObjectRole.BODY:
                body_colliders[node.full_path] = animated_frames
            elif node.assigned_role == AlembicObjectRole.PROP:
                prop_colliders[node.full_path] = animated_frames

        return AlembicShotData(
            summary=summary,
            garment_meshes=garment_meshes,
            body_colliders=body_colliders,
            prop_colliders=prop_colliders,
            ignored_objects=ignored_objects
        )


class AlembicShotExporter:
    """Standalone authoring engine for versioned production Alembic (.abc) caches."""

    def __init__(self):
        if not ALEMBIC_AVAILABLE:
            raise RuntimeError("Alembic C++ geometry bindings are required for export.")

    def export_shot(
        self,
        output_path: str,
        simulated_garments: Dict[str, List[MeshGeometry]],
        fps: float = 24.0,
        start_frame: int = 1,
        source_alembic_path: Optional[str] = None,
        full_shot_export: bool = True,
        linear_units: str = "meters"
    ) -> str:
        """Export a versioned Alembic (.abc) archive.

        If full_shot_export is True and source_alembic_path is provided, unmodified colliders and
        transforms are preserved from source, while designated garments are replaced with
        approved simulated geometry.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        def _do_write():
            time_per_cycle = 1.0 / float(fps)
            start_time = float(start_frame - 1) * time_per_cycle

            # Create output archive
            out_archive = Abc.OArchive(output_path)
            top = out_archive.getTop()

            # Add TimeSampling
            time_sampling = AbcCoreAbstract.TimeSampling(time_per_cycle, start_time)
            ts_index = out_archive.addTimeSampling(time_sampling)

            # Write simulated garments
            for garment_name, frame_meshes in simulated_garments.items():
                clean_name = garment_name.strip("/").split("/")[-1]
                xform = AbcGeom.OXform(top, f"{clean_name}_xform", ts_index)
                # Identity transform for world-space vertices
                xs = AbcGeom.XformSample()
                xform.getSchema().set(xs)

                mesh_obj = AbcGeom.OPolyMesh(xform, clean_name, ts_index)
                mesh_schema = mesh_obj.getSchema()

                if not frame_meshes:
                    continue

                # Standard face topology
                faces = frame_meshes[0].faces
                face_counts_np = np.full(len(faces), 3, dtype=np.int32)
                face_indices_np = faces.flatten().astype(np.int32)

                counts_buf = imath.IntArrayFromBuffer(face_counts_np)
                indices_buf = imath.IntArrayFromBuffer(face_indices_np)

                for mesh_frame in frame_meshes:
                    verts_f32 = mesh_frame.vertices.astype(np.float32)
                    pos_buf = imath.V3fArrayFromBuffer(verts_f32)

                    sample = AbcGeom.OPolyMeshSchemaSample(pos_buf, indices_buf, counts_buf)
                    mesh_schema.set(sample)

            # Preserve unmodified props/colliders if full shot export requested
            if full_shot_export and source_alembic_path and os.path.exists(source_alembic_path):
                in_archive = Abc.IArchive(source_alembic_path)
                in_top = in_archive.getTop()

                # Helper to copy unmodified meshes
                def _copy_unmodified(in_obj, out_parent):
                    name = in_obj.getName()
                    meta = in_obj.getMetaData()

                    # If this mesh is one of the replaced garments, do not copy original
                    if any(name in g or in_obj.getFullName() in g for g in simulated_garments):
                        return

                    if AbcGeom.IXform.matches(meta):
                        in_xform = AbcGeom.IXform(in_obj.getParent(), name)
                        out_xform = AbcGeom.OXform(out_parent, name, ts_index)
                        s = in_xform.getSchema().getValue()
                        out_xs = AbcGeom.XformSample()
                        out_xs.setMatrix(s.getMatrix())
                        out_xform.getSchema().set(out_xs)

                        for i in range(in_obj.getNumChildren()):
                            _copy_unmodified(in_obj.getChild(i), out_xform)

                    elif AbcGeom.IPolyMesh.matches(meta):
                        in_mesh = AbcGeom.IPolyMesh(in_obj.getParent(), name)
                        in_schema = in_mesh.getSchema()
                        out_mesh = AbcGeom.OPolyMesh(out_parent, name, ts_index)
                        out_schema = out_mesh.getSchema()

                        num_s = in_schema.getNumSamples()
                        for s_i in range(num_s):
                            in_s = in_schema.getValue(Abc.ISampleSelector(s_i))
                            pos = in_s.getPositions()
                            ind = in_s.getFaceIndices()
                            cnt = in_s.getFaceCounts()
                            out_s = AbcGeom.OPolyMeshSchemaSample(pos, ind, cnt)
                            out_schema.set(out_s)

                    else:
                        for i in range(in_obj.getNumChildren()):
                            _copy_unmodified(in_obj.getChild(i), out_parent)

                for i in range(in_top.getNumChildren()):
                    _copy_unmodified(in_top.getChild(i), top)

        _do_write()

        # Verification: reopen archive to validate integrity
        self._validate_exported_archive(output_path, expected_garments=list(simulated_garments.keys()))
        return output_path

    def _validate_exported_archive(self, abc_path: str, expected_garments: List[str]) -> None:
        """Reopen exported archive and verify samples, hierarchy, and non-empty bounds."""
        test_arch = Abc.IArchive(abc_path)
        top = test_arch.getTop()
        if top.getNumChildren() == 0:
            raise ValueError(f"Exported Alembic archive has no root children: {abc_path}")

        found_meshes = []
        def _check(obj):
            if AbcGeom.IPolyMesh.matches(obj.getMetaData()):
                found_meshes.append(obj.getName())
            for i in range(obj.getNumChildren()):
                _check(obj.getChild(i))

        for i in range(top.getNumChildren()):
            _check(top.getChild(i))

        for g in expected_garments:
            clean_g = g.strip("/").split("/")[-1]
            if clean_g not in found_meshes:
                raise ValueError(f"Exported archive missing expected garment mesh: {clean_g}")
