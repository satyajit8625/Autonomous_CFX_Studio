"""Autonomous CFX Studio - Dedicated Image-Guided Correction Panel.
Implements the 10-step correction workflow:
1. Choose a frame
2. Capture the viewport
3. Highlight the problem region (Brush, Outline, Arrow)
4. Optionally mark a protected region (e.g. cuff freezing)
5. Enter an artist instruction
6. Set affected frame range and fades
7. Preview the mapped 3D selection
8. Review interpreted correction strategy
9. Apply bounded corrective attempt
10. Compare and accept or reject
Handles external screenshots with camera/region confirmation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CorrectionPanel(QWidget):
    """Dedicated step-by-step panel for viewport capture, region marking, and non-destructive repairs."""

    capture_requested = Signal()                   # Step 2: Capture viewport
    tool_changed = Signal(str)                     # Step 3: tool ('brush', 'outline', 'arrow')
    brush_radius_changed = Signal(float)           # Step 3: radius
    region_type_changed = Signal(str)              # Step 4: 'defect' or 'protected'
    undo_stroke_requested = Signal()               # Step 3: undo
    clear_strokes_requested = Signal()             # Step 3: clear
    preview_mapping_requested = Signal(dict)       # Step 7: Preview 3D mapping with params
    execute_correction_requested = Signal(dict)    # Step 9: Execute corrective layer
    toggle_compare_requested = Signal(bool)        # Step 10: Toggle before/after comparison
    accept_correction_requested = Signal()         # Step 10: Accept
    reject_correction_requested = Signal()         # Step 10: Reject
    external_screenshot_imported = Signal(str, int) # External screenshot file, frame

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(360)
        self.current_frame: int = 1
        self.has_active_result: bool = False
        self.has_preview_selection: bool = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Title
        lbl_title = QLabel("Image-Guided Correction Studio")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(lbl_title)

        # Step 1: Choose Frame & Capture Viewport
        grp_step1 = QGroupBox("1 & 2. Frame & Viewport Capture")
        vbox_1 = QVBoxLayout(grp_step1)
        vbox_1.setSpacing(6)

        row_frame = QHBoxLayout()
        row_frame.addWidget(QLabel("Target Frame:"))
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(1, 9999)
        self.spin_frame.setValue(1)
        row_frame.addWidget(self.spin_frame)

        self.btn_capture = QPushButton("📸 Capture Viewport")
        self.btn_capture.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_capture.clicked.connect(self.capture_requested.emit)
        row_frame.addWidget(self.btn_capture)
        vbox_1.addLayout(row_frame)

        # External screenshot option
        btn_ext = QPushButton("📁 Import External Screenshot...")
        btn_ext.setStyleSheet(
            "QPushButton { background-color: #1e293b; color: #94a3b8; font-size: 10px; padding: 4px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #334155; color: #f8fafc; }"
        )
        btn_ext.clicked.connect(self._on_import_external_screenshot)
        vbox_1.addWidget(btn_ext)

        layout.addWidget(grp_step1)

        # Step 3 & 4: Drawing Tools & Region Type
        grp_step2 = QGroupBox("3 & 4. Highlight Defect & Protected Regions")
        vbox_2 = QVBoxLayout(grp_step2)
        vbox_2.setSpacing(6)

        tool_row = QHBoxLayout()
        self.btn_brush = QPushButton("🖌️ Brush")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setChecked(True)
        self.btn_brush.clicked.connect(lambda: self._select_tool("brush"))
        tool_row.addWidget(self.btn_brush)

        self.btn_outline = QPushButton("⭕ Outline")
        self.btn_outline.setCheckable(True)
        self.btn_outline.clicked.connect(lambda: self._select_tool("outline"))
        tool_row.addWidget(self.btn_outline)

        self.btn_arrow = QPushButton("↗ Arrow")
        self.btn_arrow.setCheckable(True)
        self.btn_arrow.clicked.connect(lambda: self._select_tool("arrow"))
        tool_row.addWidget(self.btn_arrow)

        btn_undo = QPushButton("↩️")
        btn_undo.setToolTip("Undo last mark")
        btn_undo.clicked.connect(self.undo_stroke_requested.emit)
        tool_row.addWidget(btn_undo)

        btn_clear = QPushButton("🗑️")
        btn_clear.setToolTip("Clear all annotations")
        btn_clear.clicked.connect(self.clear_strokes_requested.emit)
        tool_row.addWidget(btn_clear)

        vbox_2.addLayout(tool_row)

        # Brush Radius slider
        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Brush Size:"))
        self.slider_radius = QSlider(Qt.Horizontal)
        self.slider_radius.setRange(4, 50)
        self.slider_radius.setValue(14)
        self.slider_radius.valueChanged.connect(lambda v: self.brush_radius_changed.emit(float(v)))
        radius_row.addWidget(self.slider_radius)
        self.lbl_radius_val = QLabel("14px")
        self.slider_radius.valueChanged.connect(lambda v: self.lbl_radius_val.setText(f"{v}px"))
        radius_row.addWidget(self.lbl_radius_val)
        vbox_2.addLayout(radius_row)

        # Region Type toggle
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Region Purpose:"))
        self.btn_type_defect = QPushButton("🔴 Defect (Repair)")
        self.btn_type_defect.setCheckable(True)
        self.btn_type_defect.setChecked(True)
        self.btn_type_defect.clicked.connect(lambda: self._select_region_type("defect"))
        reg_row.addWidget(self.btn_type_defect)

        self.btn_type_protect = QPushButton("🛡️ Protected (Freeze)")
        self.btn_type_protect.setCheckable(True)
        self.btn_type_protect.clicked.connect(lambda: self._select_region_type("protected"))
        reg_row.addWidget(self.btn_type_protect)
        vbox_2.addLayout(reg_row)

        layout.addWidget(grp_step2)

        # Step 5 & 6: Instruction & Temporal Range
        grp_step3 = QGroupBox("5 & 6. Correction Instruction & Temporal Scope")
        vbox_3 = QVBoxLayout(grp_step3)
        vbox_3.setSpacing(6)

        self.txt_instruction = QLineEdit()
        self.txt_instruction.setPlaceholderText("e.g. Remove body penetration here during frames 10-25. Preserve cuff shape.")
        self.txt_instruction.setText("Remove body penetration here during frames 10-25. Preserve cuff shape.")
        vbox_3.addWidget(self.txt_instruction)

        # Range & Fade spinboxes
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Frames:"))
        self.spin_start_frame = QSpinBox()
        self.spin_start_frame.setRange(1, 9999)
        self.spin_start_frame.setValue(10)
        time_row.addWidget(self.spin_start_frame)

        time_row.addWidget(QLabel("to"))
        self.spin_end_frame = QSpinBox()
        self.spin_end_frame.setRange(1, 9999)
        self.spin_end_frame.setValue(25)
        time_row.addWidget(self.spin_end_frame)

        time_row.addWidget(QLabel("Fade (±f):"))
        self.spin_fade = QSpinBox()
        self.spin_fade.setRange(0, 20)
        self.spin_fade.setValue(3)
        time_row.addWidget(self.spin_fade)

        vbox_3.addLayout(time_row)
        layout.addWidget(grp_step3)

        # Step 7 & 8: Preview 3D & Agent Interpretation
        grp_step4 = QGroupBox("7 & 8. 3D Preview & Agent Strategy")
        vbox_4 = QVBoxLayout(grp_step4)
        vbox_4.setSpacing(6)

        self.btn_preview_3d = QPushButton("🔍 Preview Mapped 3D Selection")
        self.btn_preview_3d.setStyleSheet(
            "QPushButton { background-color: #4338ca; color: white; font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3730a3; }"
        )
        self.btn_preview_3d.clicked.connect(self._on_preview_clicked)
        vbox_4.addWidget(self.btn_preview_3d)

        self.lbl_strategy = QLabel("Agent Strategy: Awaiting 3D surface mapping preview.")
        self.lbl_strategy.setWordWrap(True)
        self.lbl_strategy.setStyleSheet(
            "background-color: #171d28; color: #94a3b8; font-size: 11px; padding: 6px; "
            "border-radius: 4px; border: 1px solid #2d3748;"
        )
        vbox_4.addWidget(self.lbl_strategy)

        layout.addWidget(grp_step4)

        # Step 9 & 10: Apply Bounded Correction & Review
        grp_step5 = QGroupBox("9 & 10. Execute, Compare & Accept")
        vbox_5 = QVBoxLayout(grp_step5)
        vbox_5.setSpacing(6)

        self.btn_execute = QPushButton("⚡ Apply Bounded Correction")
        self.btn_execute.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; font-size: 12px; "
            "padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #047857; }"
            "QPushButton:disabled { background-color: #1e293b; color: #64748b; }"
        )
        self.btn_execute.setEnabled(False)
        self.btn_execute.clicked.connect(self._on_execute_clicked)
        vbox_5.addWidget(self.btn_execute)

        # Before / After Compare button
        compare_row = QHBoxLayout()
        self.btn_compare = QPushButton("👁 Toggle Before / After")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setEnabled(False)
        self.btn_compare.setStyleSheet("padding: 6px;")
        self.btn_compare.clicked.connect(lambda c: self.toggle_compare_requested.emit(c))
        compare_row.addWidget(self.btn_compare)
        vbox_5.addLayout(compare_row)

        # Decision Row
        decision_row = QHBoxLayout()
        self.btn_accept = QPushButton("✔ Accept Correction")
        self.btn_accept.setEnabled(False)
        self.btn_accept.setStyleSheet(
            "background-color: #15803d; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.btn_accept.clicked.connect(self.accept_correction_requested.emit)
        decision_row.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("✖ Discard")
        self.btn_reject.setEnabled(False)
        self.btn_reject.setStyleSheet(
            "background-color: #991b1b; color: white; padding: 6px; border-radius: 4px;"
        )
        self.btn_reject.clicked.connect(self.reject_correction_requested.emit)
        decision_row.addWidget(self.btn_reject)
        vbox_5.addLayout(decision_row)

        # Telemetry readout
        self.lbl_telemetry = QLabel("No correction applied.")
        self.lbl_telemetry.setWordWrap(True)
        self.lbl_telemetry.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic;")
        vbox_5.addWidget(self.lbl_telemetry)

        layout.addWidget(grp_step5)
        layout.addStretch(1)

    def _select_tool(self, tool_name: str):
        self.btn_brush.setChecked(tool_name == "brush")
        self.btn_outline.setChecked(tool_name == "outline")
        self.btn_arrow.setChecked(tool_name == "arrow")
        self.tool_changed.emit(tool_name)

    def _select_region_type(self, r_type: str):
        self.btn_type_defect.setChecked(r_type == "defect")
        self.btn_type_protect.setChecked(r_type == "protected")
        self.region_type_changed.emit(r_type)

    def _on_preview_clicked(self):
        params = {
            "instruction": self.txt_instruction.text().strip(),
            "start_frame": self.spin_start_frame.value(),
            "end_frame": self.spin_end_frame.value(),
            "fade": self.spin_fade.value(),
            "falloff_radius": 0.08,
        }
        self.preview_mapping_requested.emit(params)

    def _on_execute_clicked(self):
        params = {
            "instruction": self.txt_instruction.text().strip(),
            "start_frame": self.spin_start_frame.value(),
            "end_frame": self.spin_end_frame.value(),
            "fade": self.spin_fade.value(),
            "falloff_radius": 0.08,
        }
        self.execute_correction_requested.emit(params)

    def update_strategy_display(self, strategy_text: str):
        self.lbl_strategy.setText(f"Agent Strategy:\n{strategy_text}")
        self.btn_execute.setEnabled(True)

    def update_telemetry_results(self, pen_before: float, pen_after: float, outside_disp: float, is_flicker_free: bool):
        self.lbl_telemetry.setText(
            f"Correction Telemetry:\n"
            f"• Penetration: {pen_before:.2f} mm → {pen_after:.2f} mm\n"
            f"• Outside area displacement: {outside_disp:.2f} mm\n"
            f"• Smoothness: Flicker-Free = {is_flicker_free}"
        )
        self.lbl_telemetry.setStyleSheet("color: #34d399; font-size: 11px; font-weight: 500;")
        self.btn_compare.setEnabled(True)
        self.btn_accept.setEnabled(True)
        self.btn_reject.setEnabled(True)

    def _on_import_external_screenshot(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import External Defect Screenshot", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            # External screenshot confirmation
            msg = QMessageBox(self)
            msg.setWindowTitle("External Screenshot Calibration")
            msg.setText(
                f"Importing external screenshot: {file_path}\n\n"
                "Precise camera and depth projection are ambiguous without calibration.\n"
                "Please confirm target frame number and verify camera alignment before mapping."
            )
            msg.setIcon(QMessageBox.Information)
            msg.exec()
            self.external_screenshot_imported.emit(file_path, self.spin_frame.value())
