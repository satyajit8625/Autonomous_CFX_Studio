"""Autonomous CFX Studio - Alembic Import and Export Configuration Dialogs.
Exposes standalone .abc inspection, metadata/units confirmation, and versioned export modes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from autonomous_cfx.core.alembic_io import AlembicShotSummary


class AlembicImportDialog(QDialog):
    """Dialog for inspecting Alembic shot metadata and confirming FPS, units, and axes."""

    def __init__(self, summary: AlembicShotSummary, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Import Animated Shot - {Path(summary.archive_path).name}")
        self.resize(650, 480)
        self.summary = summary
        self.confirmed_fps: float = summary.fps
        self.confirmed_units: str = summary.linear_units
        self.confirmed_up_axis: str = summary.up_axis
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Archive Overview Box
        grp_overview = QGroupBox("Alembic Archive Metadata")
        vbox_o = QVBoxLayout(grp_overview)

        info = (
            f"• Archive Path: {self.summary.archive_path}\n"
            f"• Storage Backend: {self.summary.archive_format} (Official C++ geometry Alembic)\n"
            f"• Sample Count: {self.summary.total_samples} samples  |  "
            f"Detected Frame Range: {self.summary.frame_range[0]} – {self.summary.frame_range[1]}\n"
            f"• Total Scene Nodes: {len(self.summary.nodes)}"
        )
        lbl_info = QLabel(info)
        lbl_info.setStyleSheet("color: #cbd5e1; font-size: 11px; font-family: Consolas;")
        vbox_o.addWidget(lbl_info)
        layout.addWidget(grp_overview)

        # 2. Hierarchy & Nodes Table
        grp_nodes = QGroupBox("Hierarchy & Geometry Nodes")
        vbox_n = QVBoxLayout(grp_nodes)

        table = QTableWidget(len(self.summary.nodes), 5)
        table.setHorizontalHeaderLabels(["Node Path", "Type", "Verts", "Samples", "Constant Topology"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setStyleSheet("background-color: #11141a; color: #e2e8f0; font-size: 11px;")

        for row, node in enumerate(self.summary.nodes):
            table.setItem(row, 0, QTableWidgetItem(node.full_path))
            table.setItem(row, 1, QTableWidgetItem(node.node_type))
            table.setItem(row, 2, QTableWidgetItem(str(node.num_vertices)))
            table.setItem(row, 3, QTableWidgetItem(str(node.num_samples)))
            topo_str = "✔ Yes" if node.is_topology_constant else "⚠️ Changing (unsupported)"
            table.setItem(row, 4, QTableWidgetItem(topo_str))

        vbox_n.addWidget(table)
        layout.addWidget(grp_nodes, stretch=1)

        # 3. Units, FPS, and Coordinate Space Confirmation
        grp_settings = QGroupBox("Coordinate Space & Timing Confirmation")
        vbox_s = QVBoxLayout(grp_settings)

        row_s = QHBoxLayout()
        row_s.addWidget(QLabel("Frame Rate (FPS):"))
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["24.0", "25.0", "29.97", "30.0", "60.0"])
        self.combo_fps.setCurrentText(f"{self.summary.fps:.1f}" if self.summary.fps in (24.0, 25.0, 30.0, 60.0) else "24.0")
        row_s.addWidget(self.combo_fps)

        row_s.addWidget(QLabel("Linear Units:"))
        self.combo_units = QComboBox()
        self.combo_units.addItems(["meters", "centimeters", "millimeters", "inches"])
        self.combo_units.setCurrentText(self.summary.linear_units)
        row_s.addWidget(self.combo_units)

        row_s.addWidget(QLabel("Up Axis:"))
        self.combo_axis = QComboBox()
        self.combo_axis.addItems(["Y", "Z"])
        self.combo_axis.setCurrentText(self.summary.up_axis)
        row_s.addWidget(self.combo_axis)

        vbox_s.addLayout(row_s)
        layout.addWidget(grp_settings)

        # Buttons
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_confirm = QPushButton("📥 Import Shot with Settings")
        btn_confirm.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;"
        )
        btn_confirm.clicked.connect(self._on_confirm)
        btn_row.addWidget(btn_confirm)
        layout.addLayout(btn_row)

    def _on_confirm(self):
        self.confirmed_fps = float(self.combo_fps.currentText())
        self.confirmed_units = self.combo_units.currentText()
        self.confirmed_up_axis = self.combo_axis.currentText()
        self.accept()


class AlembicExportDialog(QDialog):
    """Dialog for configuring versioned Alembic export (Garment-only vs Full-shot)."""

    def __init__(
        self,
        default_path: str,
        garment_name: str,
        frame_count: int,
        source_archive: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Approved Alembic Archive (.abc)")
        self.resize(550, 360)
        self.default_path = default_path
        self.garment_name = garment_name
        self.frame_count = frame_count
        self.source_archive = source_archive

        self.selected_export_path: str = default_path
        self.is_full_shot: bool = True
        self.validate_after_export: bool = True
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Output Path Row
        grp_dest = QGroupBox("Destination Archive")
        vbox_d = QVBoxLayout(grp_dest)

        row_p = QHBoxLayout()
        self.txt_path = QLineEdit(self.default_path)
        row_p.addWidget(self.txt_path, stretch=1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse)
        row_p.addWidget(btn_browse)
        vbox_d.addLayout(row_p)
        layout.addWidget(grp_dest)

        # 2. Export Mode Radio Group
        grp_mode = QGroupBox("Export Scope")
        vbox_m = QVBoxLayout(grp_mode)

        self.rad_full = QRadioButton("Full Shot Export (Replaces garment, preserves unmodified colliders and props)")
        self.rad_full.setChecked(True)
        vbox_m.addWidget(self.rad_full)

        self.rad_garment = QRadioButton("Garment-Only Export (Writes only simulated cloth geometry)")
        vbox_m.addWidget(self.rad_garment)

        layout.addWidget(grp_mode)

        # 3. Pre-Export Quality Notice
        grp_summary = QGroupBox("Export Manifest")
        vbox_s = QVBoxLayout(grp_summary)
        summary_text = (
            f"• Target Garment: {self.garment_name}\n"
            f"• Frame Count: {self.frame_count} animated geometry samples\n"
            f"• Source Reference: {Path(self.source_archive).name if self.source_archive else 'None'}\n"
            f"• Format: C++ Alembic Ogawa (.abc) compatible with Autodesk Maya / Houdini"
        )
        lbl_sum = QLabel(summary_text)
        lbl_sum.setStyleSheet("color: #cbd5e1; font-size: 11px; font-family: Consolas;")
        vbox_s.addWidget(lbl_sum)
        layout.addWidget(grp_summary)

        # Buttons
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_export = QPushButton("💾 Export Approved Alembic")
        btn_export.setStyleSheet(
            "background-color: #059669; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;"
        )
        btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

    def _on_browse(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export Alembic", self.txt_path.text(), "Alembic (*.abc)")
        if f:
            self.txt_path.setText(f)

    def _on_export(self):
        p = self.txt_path.text().strip()
        if not p:
            QMessageBox.warning(self, "Export", "Please specify a destination file path.")
            return
        self.selected_export_path = p
        self.is_full_shot = self.rad_full.isChecked()
        self.accept()
