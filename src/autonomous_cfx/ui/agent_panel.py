"""Autonomous CFX Studio - Agent Orchestration Control Panel.
Provides prompt input, presets, live agent logs, and workflow dispatch controls.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AgentPanel(QWidget):
    """Panel for prompt-driven cloth behavior specification and agent execution."""

    run_requested = Signal(str)     # Emitted with behavior prompt
    cancel_requested = Signal()     # Emitted when user cancels

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(340)
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # 1. Behavior Description Group
        grp_behavior = QGroupBox("Cloth Behavior Specification")
        vbox_b = QVBoxLayout(grp_behavior)

        lbl_instruct = QLabel("Describe cloth physics in natural language:")
        lbl_instruct.setStyleSheet("color: #a0a8b4; font-size: 11px;")
        vbox_b.addWidget(lbl_instruct)

        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("e.g. heavy leather jacket, stiff collar, low stretch, high friction with body, high damping")
        self.txt_prompt.setText("heavy leather jacket, stiff collar, low stretch, high friction with body, high damping")
        self.txt_prompt.setMaximumHeight(70)
        vbox_b.addWidget(self.txt_prompt)

        # Quick preset buttons
        lbl_presets = QLabel("Quick Behavior Presets:")
        lbl_presets.setStyleSheet("color: #8c93a0; font-size: 10px; margin-top: 4px;")
        vbox_b.addWidget(lbl_presets)

        preset_layout = QHBoxLayout()
        presets = [
            ("Leather", "heavy leather jacket, stiff collar, low stretch, high friction with body, high damping"),
            ("Silk", "lightweight silk windbreaker, loose drape, soft folds, low friction, fluid drape"),
            ("Denim", "rugged denim jacket, tight fit, high stretch resistance, medium weight"),
            ("Wool", "heavy wool overcoat, thick folds, high damping, structured drape"),
        ]
        for name, p_text in presets:
            btn = QPushButton(name)
            btn.setStyleSheet("font-size: 11px; padding: 3px 6px;")
            btn.clicked.connect(lambda checked=False, t=p_text: self.txt_prompt.setText(t))
            preset_layout.addWidget(btn)
        vbox_b.addLayout(preset_layout)

        main_layout.addWidget(grp_behavior)

        # 2. Execution Controls
        grp_controls = QGroupBox("Autonomous Execution")
        vbox_c = QVBoxLayout(grp_controls)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("⚡ Run Autonomous Simulation")
        self.btn_run.setStyleSheet(
            "background-color: #2b7de9; color: white; font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        self.btn_run.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.btn_run, stretch=2)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("padding: 8px;")
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        btn_row.addWidget(self.btn_cancel, stretch=1)
        vbox_c.addLayout(btn_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        vbox_c.addWidget(self.progress_bar)

        self.lbl_stage = QLabel("Ready")
        self.lbl_stage.setStyleSheet("color: #728299; font-style: italic; font-size: 11px;")
        vbox_c.addWidget(self.lbl_stage)

        main_layout.addWidget(grp_controls)

        # 3. Live Agent Telemetry & Reasoning Log
        grp_logs = QGroupBox("Agent Decision & Repair Stream")
        vbox_l = QVBoxLayout(grp_logs)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color: #15181e; color: #d0d5dd; font-family: Consolas; font-size: 11px;")
        vbox_l.addWidget(self.log_view)

        main_layout.addWidget(grp_logs, stretch=1)

    def _on_run_clicked(self):
        prompt = self.txt_prompt.toPlainText().strip()
        if prompt:
            self.btn_run.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.append_log(f"▶ Initiating workflow with prompt: '{prompt}'")
            self.run_requested.emit(prompt)

    def set_running_state(self, is_running: bool):
        self.btn_run.setEnabled(not is_running)
        self.btn_cancel.setEnabled(is_running)

    def update_progress(self, percent: int, stage_text: str):
        self.progress_bar.setValue(percent)
        self.lbl_stage.setText(stage_text)

    def append_log(self, text: str):
        self.log_view.append(text)
        # Scroll to bottom
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_view.setTextCursor(cursor)

    def clear_logs(self):
        self.log_view.clear()
