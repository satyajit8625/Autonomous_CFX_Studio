"""Autonomous CFX Studio - Maya Interchange Connector (Optional).
Connects Autonomous CFX Studio with Autodesk Maya for asset and cache exchange.
Guaranteed 100% standalone-safe: core application runs seamlessly without Maya installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomous_cfx.core.logger import logger

# Safe detection of Maya runtime environment
MAYA_AVAILABLE = False
try:
    import maya.cmds as cmds
    import maya.OpenMaya as OpenMaya
    MAYA_AVAILABLE = True
except ImportError:
    cmds = None
    OpenMaya = None


class MayaConnector:
    """Optional bridge for Autodesk Maya geometry import and cache loading."""

    @classmethod
    def is_available(cls) -> bool:
        """Check whether Maya Python environment is loaded."""
        return MAYA_AVAILABLE

    @classmethod
    def export_selection_to_obj(cls, output_path: str | Path) -> bool:
        """Export active Maya selection as OBJ geometry."""
        if not cls.is_available():
            logger.info("Maya environment not present. Maya export skipped.")
            return False

        try:
            sel = cmds.ls(selection=True)
            if not sel:
                logger.warning("No geometry selected in Maya to export.")
                return False

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmds.file(
                str(output_path),
                force=True,
                options="groups=1;ptgroups=1;materials=0;smoothing=1;normals=1",
                type="OBJexport",
                exportSelected=True,
            )
            logger.info(f"Exported Maya selection '{sel[0]}' to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting from Maya: {e}")
            return False

    @classmethod
    def generate_maya_import_script(
        cls,
        cache_manifest_path: str | Path,
        script_output_path: str | Path,
    ) -> str:
        """Generates a standalone Maya Python script that artists can execute in Maya Script Editor."""
        manifest_path = Path(cache_manifest_path).resolve()
        script_path = Path(script_output_path).resolve()
        script_path.parent.mkdir(parents=True, exist_ok=True)

        py_script = f'''# Autonomous CFX Studio - Maya Cache Loader Script
# Generated automatically by Autonomous CFX Studio
import json
import maya.cmds as cmds

manifest_path = r"{str(manifest_path)}"
print(f"[Autonomous CFX] Loading cache manifest: {{manifest_path}}")

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"[Autonomous CFX] Asset: {{manifest.get('name')}}, Frames: {{manifest.get('frame_range')}}")
print("[Autonomous CFX] Connect this cache to your target cloth rig in Maya.")
'''
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(py_script)

        logger.info(f"Generated Maya import script: {script_path}")
        return str(script_path)
