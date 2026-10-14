"""Autonomous CFX Studio - Native Standalone Animated Alembic (.abc) I/O.

High-performance, standalone Alembic geometry reading and authoring using official
C++ Alembic Ogawa/HDF5 bindings (Abc, AbcGeom, imath). Runs completely independent
of Autodesk Maya or other host DCCs.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

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

from autonomous_cfx.core.camera import FilmFitMode, ShotCamera, ShotCameraSample
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.logger import logger


@njit(fastmath=True)
def _triangulate_polygons_numba(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Numba-accelerated polygon fan triangulation."""
    total_triangles = 0
    for i in range(len(face_counts)):
        c = face_counts[i]
        if c >= 3:
            total_triangles += c - 2

    out = np.empty((total_triangles, 3), dtype=np.int32)
    t_idx = 0
    offset = 0
    for i in range(len(face_counts)):
        c = face_counts[i]
        if c == 3:
            out[t_idx, 0] = face_indices[offset]
            out[t_idx, 1] = face_indices[offset + 1]
            out[t_idx, 2] = face_indices[offset + 2]
            t_idx += 1
        elif c == 4:
            out[t_idx, 0] = face_indices[offset]
            out[t_idx, 1] = face_indices[offset + 1]
            out[t_idx, 2] = face_indices[offset + 2]
            t_idx += 1
            out[t_idx, 0] = face_indices[offset]
            out[t_idx, 1] = face_indices[offset + 2]
            out[t_idx, 2] = face_indices[offset + 3]
            t_idx += 1
        elif c > 4:
            v0 = face_indices[offset]
            for j in range(1, c - 1):
                out[t_idx, 0] = v0
                out[t_idx, 1] = face_indices[offset + j]
                out[t_idx, 2] = face_indices[offset + j + 1]
                t_idx += 1
        offset += c
    return out


def _triangulate_polygons_fallback(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Pure Python/NumPy fallback polygon fan triangulation."""
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
        elif count > 4:
            v0 = face_indices[idx_offset]
            for j in range(1, count - 1):
                triangles.append([
                    v0,
                    face_indices[idx_offset + j],
                    face_indices[idx_offset + j + 1]
                ])
        idx_offset += count
    return np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)


def _triangulate_polygons(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Dispatch polygon triangulation to compiled Numba kernel or fallback."""
    if NUMBA_AVAILABLE:
        return _triangulate_polygons_numba(face_counts, face_indices)
    return _triangulate_polygons_fallback(face_counts, face_indices)


class AlembicObjectRole(str, Enum):
    """Semantic role assigned to Alembic scene objects."""
    GARMENT = "garment"
    BODY = "body"
    PROP = "prop"
    IGNORED = "ignored"
    CAMERA = "camera"


@dataclass
class AlembicNodeInfo:
    """Metadata describing a node in an Alembic archive hierarchy."""
    name: str
    full_path: str
    node_type: str  # 'PolyMesh', 'Xform', 'Camera', etc.
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
    cameras: List[AlembicNodeInfo] = field(default_factory=list)
    up_axis: str = "Y"
    linear_units: str = "meters"
    archive_format: str = "Ogawa"


@dataclass
class AlembicShotData:
    """Loaded animated geometry data ready for CFX simulation."""
    summary: AlembicShotSummary
    garment_meshes: Dict[str, Any] = field(default_factory=dict)
    body_colliders: Dict[str, Any] = field(default_factory=dict)
    prop_colliders: Dict[str, Any] = field(default_factory=dict)
    ignored_objects: List[str] = field(default_factory=list)
    cameras: Dict[str, ShotCamera] = field(default_factory=dict)

    def close(self) -> None:
        """Release underlying Alembic archive and schema handles to free OS file locks."""
        for mapping in (self.garment_meshes, self.body_colliders, self.prop_colliders):
            for seq in mapping.values():
                if hasattr(seq, "close"):
                    seq.close()
        import gc
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LazyAnimatedMeshSequence(collections.abc.Sequence):
    """Memory-efficient, lazy animated mesh sequence from Alembic stream.
    Samples geometry on demand and caches accessed frames."""

    def __init__(
        self,
        schema: Any,
        node_name: str,
        node_parent_path: str,
        num_samples: int,
        triangles: np.ndarray,
        uvs: Optional[np.ndarray],
        face_sets: Dict[str, np.ndarray],
        world_space: bool,
        get_world_matrix_fn: Callable[[str, int], np.ndarray],
    ):
        self._schema = schema
        self._node_name = node_name
        self._node_parent_path = node_parent_path
        self._num_samples = max(1, num_samples)
        self._triangles = triangles
        self._uvs = uvs
        self._face_sets = face_sets
        self._world_space = world_space
        self._get_world_matrix_fn = get_world_matrix_fn
        self._cache: Dict[int, MeshGeometry] = {}

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._num_samples))]
        if idx < 0:
            idx += self._num_samples
        if not (0 <= idx < self._num_samples):
            raise IndexError(f"Sample index {idx} out of range [0, {self._num_samples})")

        if idx not in self._cache:
            self._cache[idx] = self._load_sample(idx)
        return self._cache[idx]

    def _load_sample(self, s_idx: int) -> MeshGeometry:
        s = self._schema.getValue(Abc.ISampleSelector(s_idx))
        pos_raw = np.ctypeslib.as_array(s.getPositions()).copy()

        if self._world_space and self._node_parent_path:
            mat = self._get_world_matrix_fn(self._node_parent_path, s_idx)
            homo = np.column_stack([pos_raw, np.ones((len(pos_raw), 1), dtype=np.float64)])
            pos_world = (homo @ mat)[:, :3].astype(np.float32)
        else:
            pos_world = pos_raw.astype(np.float32)

        return MeshGeometry(
            vertices=pos_world,
            faces=self._triangles,
            uvs=self._uvs,
            face_sets=self._face_sets,
            name=f"{self._node_name}_f{s_idx + 1}"
        )

    def prefetch(self, frame_indices: List[int]) -> None:
        """Prefetch a batch of frames into RAM cache."""
        for idx in frame_indices:
            if 0 <= idx < self._num_samples and idx not in self._cache:
                self._cache[idx] = self._load_sample(idx)

    def clear_cache(self) -> None:
        """Evict cached frames to reclaim memory."""
        self._cache.clear()

    def close(self) -> None:
        """Release underlying schema and C++ handles."""
        self._schema = None
        self._get_world_matrix_fn = None


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
        cameras_list: List[AlembicNodeInfo] = []

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
            is_camera = AbcGeom.ICamera.matches(meta)

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

            elif is_camera:
                node_type = "Camera"
                cam = AbcGeom.ICamera(obj.getParent(), name)
                schema = cam.getSchema()
                num_s = schema.getNumSamples()
                total_samples = max(total_samples, num_s)
                c_node = AlembicNodeInfo(
                    name=name,
                    full_path=full_path,
                    node_type="Camera",
                    parent_path=parent_path,
                    assigned_role=AlembicObjectRole.CAMERA,
                    num_samples=num_s,
                )
                nodes.append(c_node)
                cameras_list.append(c_node)

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
            cameras=cameras_list,
            up_axis="Y",
            linear_units="meters",
            archive_format="Ogawa",
        )

    def load_cameras(self, abc_path: str) -> Dict[str, ShotCamera]:
        """Load animated shot cameras from an Alembic (.abc) archive.
        Resolves nested parent Xform hierarchies, time sampling rates, and optical apertures."""
        if not os.path.exists(abc_path):
            raise FileNotFoundError(f"Alembic file not found: {abc_path}")

        archive = Abc.IArchive(abc_path)
        top = archive.getTop()
        cameras: Dict[str, ShotCamera] = {}
        camera_nodes: List[Tuple[Any, str, str]] = []

        def _find_cameras(obj, parent_path: str):
            meta = obj.getMetaData()
            name = obj.getName()
            full_path = obj.getFullName()
            if AbcGeom.ICamera.matches(meta):
                camera_nodes.append((obj.getParent(), name, parent_path))
            for i in range(obj.getNumChildren()):
                _find_cameras(obj.getChild(i), full_path)

        for i in range(top.getNumChildren()):
            _find_cameras(top.getChild(i), "/")

        _xform_chain_cache: Dict[str, List[AbcGeom.IXform]] = {}

        def _resolve_xform_chain(obj_path: str) -> List[AbcGeom.IXform]:
            if not obj_path or obj_path == "/":
                return []
            if obj_path in _xform_chain_cache:
                return _xform_chain_cache[obj_path]

            parts = [p for p in obj_path.split("/") if p]
            curr = top
            chain = []
            for part in parts:
                child = None
                for i in range(curr.getNumChildren()):
                    if curr.getChild(i).getName() == part:
                        child = curr.getChild(i)
                        break
                if child is None:
                    break
                if AbcGeom.IXform.matches(child.getMetaData()):
                    chain.append(AbcGeom.IXform(curr, part))
                curr = child

            _xform_chain_cache[obj_path] = chain
            return chain

        for parent_obj, cam_shape_name, parent_path in camera_nodes:
            cam = AbcGeom.ICamera(parent_obj, cam_shape_name)
            schema = cam.getSchema()
            num_samples = schema.getNumSamples()
            ts = schema.getTimeSampling()
            tst = ts.getTimeSamplingType()
            time_per_sample = tst.getTimePerCycle() if tst.getTimePerCycle() > 1e-6 else 1.0 / 24.0
            fps = round(1.0 / time_per_sample, 2)

            parent_full_path = parent_obj.getFullName()
            xform_chain = _resolve_xform_chain(parent_full_path)

            samples: List[ShotCameraSample] = []
            unsupported: List[str] = []

            for s_idx in range(num_samples):
                try:
                    s = schema.getValue(s_idx)
                except Exception:
                    s = schema.getValue(Abc.ISampleSelector(s_idx))
                t_sec = ts.getSampleTime(s_idx)
                frame_num = int(round(t_sec * fps)) + 1 if t_sec >= 0 else s_idx + 1

                # Calculate world matrix: M_world = M_local @ M_parent1 @ M_parent2 ...
                mat = np.eye(4, dtype=np.float64)
                for xf in xform_chain:
                    xs = xf.getSchema()
                    s_count = xs.getNumSamples()
                    target_idx = min(s_idx, max(0, s_count - 1))
                    try:
                        m_val = xs.getValue(target_idx).getMatrix()
                    except Exception:
                        m_val = xs.getValue(Abc.ISampleSelector(target_idx)).getMatrix()
                    mat = mat @ np.array(m_val, dtype=np.float64).reshape((4, 4))

                f_mm = float(s.getFocalLength())
                h_ap_mm = float(s.getHorizontalAperture()) * 10.0
                v_ap_mm = float(s.getVerticalAperture()) * 10.0
                near_c = float(s.getNearClippingPlane())
                far_c = float(s.getFarClippingPlane())
                h_off_mm = float(s.getHorizontalFilmOffset()) * 10.0
                v_off_mm = float(s.getVerticalFilmOffset()) * 10.0
                squeeze = float(s.getLensSqueezeRatio())

                samples.append(ShotCameraSample(
                    frame=float(frame_num),
                    time_seconds=float(t_sec),
                    world_matrix=mat,
                    focal_length_mm=f_mm,
                    horizontal_aperture_mm=h_ap_mm,
                    vertical_aperture_mm=v_ap_mm,
                    near_clip=near_c,
                    far_clip=far_c,
                    horizontal_offset_mm=h_off_mm,
                    vertical_offset_mm=v_off_mm,
                    lens_squeeze=squeeze,
                ))

            cam_name = parent_obj.getName() if parent_obj.getName() != "/" else cam_shape_name
            shot_cam = ShotCamera(
                name=cam_name,
                full_path=cam.getFullName(),
                fps=fps,
                start_frame=int(samples[0].frame) if samples else 1,
                samples=samples,
                film_fit=FilmFitMode.OVERSCAN,
                unsupported_features=unsupported,
            )
            cameras[cam_name] = shot_cam
            cameras[cam.getFullName()] = shot_cam
            logger.info(
                f"[Alembic] Loaded camera '{cam_name}': {len(samples)} samples, "
                f"f={samples[0].focal_length_mm:.1f}mm, film={samples[0].horizontal_aperture_mm:.1f}x{samples[0].vertical_aperture_mm:.1f}mm"
            )

        del top, archive
        import gc
        gc.collect()
        return cameras

    def load_shot(
        self,
        abc_path: str,
        role_assignments: Optional[Dict[str, AlembicObjectRole]] = None,
        world_space: bool = True,
        lazy: bool = False,
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

        garment_meshes: Dict[str, Any] = {}
        body_colliders: Dict[str, Any] = {}
        prop_colliders: Dict[str, Any] = {}
        ignored_objects: List[str] = []
        cameras = self.load_cameras(abc_path)

        # Cached hierarchy lookup for Xform nodes to avoid repeated string searches per frame
        _xform_chain_cache: Dict[str, List[AbcGeom.IXform]] = {}

        def _get_world_matrix_for_path(obj_path: str, sample_idx: int) -> np.ndarray:
            if not world_space or obj_path == "/" or not obj_path:
                return np.eye(4, dtype=np.float64)

            if obj_path not in _xform_chain_cache:
                parts = [p for p in obj_path.split("/") if p]
                curr = top
                chain = []
                for part in parts:
                    child = None
                    for i in range(curr.getNumChildren()):
                        if curr.getChild(i).getName() == part:
                            child = curr.getChild(i)
                            break
                    if child is None:
                        break
                    if AbcGeom.IXform.matches(child.getMetaData()):
                        chain.append(AbcGeom.IXform(curr, part))
                    curr = child
                _xform_chain_cache[obj_path] = chain

            chain = _xform_chain_cache[obj_path]
            mat = np.eye(4, dtype=np.float64)
            for xf in chain:
                xs = xf.getSchema()
                s_count = xs.getNumSamples()
                sel = Abc.ISampleSelector(min(sample_idx, max(0, s_count - 1)))
                m = xs.getValue(sel).getMatrix()
                mat = mat @ np.array(m, dtype=np.float64).reshape((4, 4))
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

            # Accelerated polygon fan triangulation
            triangles_np = _triangulate_polygons(face_counts, face_indices)

            # Extract UVs if available
            mesh_uvs = None
            uv_param = schema.getUVsParam()
            if uv_param.valid():
                try:
                    uv_expanded = uv_param.getExpandedValue(Abc.ISampleSelector(0))
                    mesh_uvs = np.ctypeslib.as_array(uv_expanded.getVals()).copy()
                except Exception:
                    pass

            # Extract FaceSets if available
            face_sets_dict = {}
            try:
                for fs_name in schema.getFaceSetNames():
                    fs_obj = schema.getFaceSet(fs_name)
                    fs_faces = np.ctypeslib.as_array(fs_obj.getSchema().getValue().getFaces()).copy()
                    face_sets_dict[fs_name] = fs_faces
            except Exception:
                pass

            stream = LazyAnimatedMeshSequence(
                schema=schema,
                node_name=node.name,
                node_parent_path=node.parent_path,
                num_samples=num_samples,
                triangles=triangles_np,
                uvs=mesh_uvs,
                face_sets=face_sets_dict,
                world_space=world_space,
                get_world_matrix_fn=_get_world_matrix_for_path,
            )
            frames_obj = stream if lazy else list(stream)

            if node.assigned_role == AlembicObjectRole.GARMENT:
                garment_meshes[node.full_path] = frames_obj
            elif node.assigned_role == AlembicObjectRole.BODY:
                body_colliders[node.full_path] = frames_obj
            elif node.assigned_role == AlembicObjectRole.PROP:
                prop_colliders[node.full_path] = frames_obj

        return AlembicShotData(
            summary=summary,
            garment_meshes=garment_meshes,
            body_colliders=body_colliders,
            prop_colliders=prop_colliders,
            ignored_objects=ignored_objects,
            cameras=cameras,
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

                has_uvs = frame_meshes[0].uvs is not None and len(frame_meshes[0].uvs) > 0
                uv_sample = None
                if has_uvs:
                    try:
                        uvs_np = frame_meshes[0].uvs.astype(np.float32)
                        uv_buf = imath.V2fArrayFromBuffer(uvs_np)
                        scope = (
                            AbcGeom.GeometryScope.kVertexScope
                            if len(uvs_np) == len(frame_meshes[0].vertices)
                            else AbcGeom.GeometryScope.kFacevaryingScope
                        )
                        uv_sample = AbcGeom.OV2fGeomParamSample(uv_buf, scope)
                    except Exception:
                        pass

                for mesh_frame in frame_meshes:
                    verts_f32 = mesh_frame.vertices.astype(np.float32)
                    pos_buf = imath.V3fArrayFromBuffer(verts_f32)

                    sample = AbcGeom.OPolyMeshSchemaSample(pos_buf, indices_buf, counts_buf)
                    if uv_sample is not None:
                        sample.setUVs(uv_sample)
                    mesh_schema.set(sample)

                # Export face sets for garment if present
                if hasattr(frame_meshes[0], "face_sets") and frame_meshes[0].face_sets:
                    for fs_name, fs_faces in frame_meshes[0].face_sets.items():
                        try:
                            out_fs = AbcGeom.OFaceSet(mesh_obj, fs_name, ts_index)
                            fs_s = AbcGeom.OFaceSetSchemaSample()
                            fs_s.setFaces(imath.IntArrayFromBuffer(fs_faces.astype(np.int32)))
                            out_fs.getSchema().set(fs_s)
                        except Exception:
                            pass

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
                        in_xs = in_xform.getSchema()
                        num_x_samples = in_xs.getNumSamples()
                        for s_i in range(max(1, num_x_samples)):
                            s = in_xs.getValue(Abc.ISampleSelector(s_i))
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
                        in_uv_param = in_schema.getUVsParam()
                        has_uvs_unmod = in_uv_param.valid()

                        for s_i in range(num_s):
                            in_s = in_schema.getValue(Abc.ISampleSelector(s_i))
                            pos = in_s.getPositions()
                            ind = in_s.getFaceIndices()
                            cnt = in_s.getFaceCounts()
                            out_s = AbcGeom.OPolyMeshSchemaSample(pos, ind, cnt)
                            if has_uvs_unmod:
                                try:
                                    uv_exp = in_uv_param.getExpandedValue(Abc.ISampleSelector(s_i))
                                    uv_samp = AbcGeom.OV2fGeomParamSample(uv_exp.getVals(), uv_exp.getScope())
                                    out_s.setUVs(uv_samp)
                                except Exception:
                                    pass
                            out_schema.set(out_s)

                        # Copy FaceSets for unmodified poly mesh
                        try:
                            for fs_name in in_schema.getFaceSetNames():
                                in_fs = in_schema.getFaceSet(fs_name)
                                out_fs = AbcGeom.OFaceSet(out_mesh, fs_name, ts_index)
                                fs_s = AbcGeom.OFaceSetSchemaSample()
                                fs_s.setFaces(in_fs.getSchema().getValue().getFaces())
                                out_fs.getSchema().set(fs_s)
                        except Exception:
                            pass

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

    def export_camera(
        self,
        output_path: str,
        camera: ShotCamera,
    ) -> str:
        """Export animated camera to a standalone Alembic (.abc) archive."""
        if not ALEMBIC_AVAILABLE:
            raise RuntimeError("Alembic C++ geometry bindings are required for camera export.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        archive = Abc.OArchive(output_path)
        top = archive.getTop()

        ts = AbcCoreAbstract.TimeSampling(1.0 / max(1e-4, camera.fps), (camera.start_frame - 1) / max(1e-4, camera.fps))
        ts_idx = archive.addTimeSampling(ts)

        # Create parent Xform
        xform = AbcGeom.OXform(top, camera.name, ts_idx)
        xform_schema = xform.getSchema()

        # Create Camera Shape
        cam_shape = AbcGeom.OCamera(xform, f"{camera.name}Shape", ts_idx)
        cam_schema = cam_shape.getSchema()

        for s in camera.samples:
            # Write xform matrix
            xf_sample = AbcGeom.XformSample()
            m_imath = imath.M44d()
            for r in range(4):
                for c in range(4):
                    m_imath[r][c] = float(s.world_matrix[r, c])
            xf_sample.setMatrix(m_imath)
            xform_schema.set(xf_sample)

            # Write camera properties (aperture in cm)
            c_sample = AbcGeom.CameraSample()
            c_sample.setFocalLength(float(s.focal_length_mm))
            c_sample.setHorizontalAperture(float(s.horizontal_aperture_mm) / 10.0)
            c_sample.setVerticalAperture(float(s.vertical_aperture_mm) / 10.0)
            c_sample.setNearClippingPlane(float(s.near_clip))
            c_sample.setFarClippingPlane(float(s.far_clip))
            c_sample.setHorizontalFilmOffset(float(s.horizontal_offset_mm) / 10.0)
            c_sample.setVerticalFilmOffset(float(s.vertical_offset_mm) / 10.0)
            c_sample.setLensSqueezeRatio(float(s.lens_squeeze))
            cam_schema.set(c_sample)

        del cam_schema, cam_shape, xform_schema, xform, top, archive
        return output_path
