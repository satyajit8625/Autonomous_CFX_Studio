"""Tests for MayaConnector standalone safety and script generation."""

from pathlib import Path
import pytest
from autonomous_cfx.connectors.maya_connector import MayaConnector


def test_maya_connector_standalone_safety(tmp_path):
    # Standalone environments should report Maya availability accurately
    avail = MayaConnector.is_available()
    assert isinstance(avail, bool)

    # Calling export outside Maya must return False safely without throwing exceptions
    res = MayaConnector.export_selection_to_obj(tmp_path / "test.obj")
    if not avail:
        assert res == False

    # Standalone script generation should always succeed
    script_file = tmp_path / "load_cache.py"
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{}", encoding="utf-8")

    out_script = MayaConnector.generate_maya_import_script(manifest_file, script_file)
    assert Path(out_script).exists()
    content = Path(out_script).read_text(encoding="utf-8")
    assert "Autonomous CFX" in content
