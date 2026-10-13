"""Autonomous CFX Studio - Timeline and Playback Scrubber Widget.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TimelineWidget(QWidget):
    """Timeline controller for scrubbing and animating simulation cache frames."""

    frame_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fps: float = 24.0
        self.is_playing: bool = False

        # Playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        # Step Back
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(36)
        self.btn_prev.clicked.connect(self.step_backward)
        layout.addWidget(self.btn_prev)

        # Play/Pause
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(70)
        self.btn_play.clicked.connect(self.toggle_play)
        layout.addWidget(self.btn_play)

        # Step Forward
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(36)
        self.btn_next.clicked.connect(self.step_forward)
        layout.addWidget(self.btn_next)

        # Frame Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 40)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, stretch=1)

        # Frame SpinBox
        self.spinbox = QSpinBox()
        self.spinbox.setRange(1, 40)
        self.spinbox.setValue(1)
        self.spinbox.valueChanged.connect(self.set_frame)
        layout.addWidget(self.spinbox)

        # Range label
        self.lbl_range = QLabel("/ 40")
        layout.addWidget(self.lbl_range)

    def set_range(self, start_frame: int, end_frame: int):
        """Configure frame limits."""
        self.slider.setRange(start_frame, end_frame)
        self.spinbox.setRange(start_frame, end_frame)
        self.lbl_range.setText(f"/ {end_frame}")

    def get_current_frame(self) -> int:
        return self.slider.value()

    def set_frame(self, frame_num: int):
        if self.slider.value() != frame_num:
            self.slider.setValue(frame_num)
        if self.spinbox.value() != frame_num:
            self.spinbox.setValue(frame_num)
        self.frame_changed.emit(frame_num)

    def _on_slider_changed(self, val: int):
        self.spinbox.setValue(val)
        self.frame_changed.emit(val)

    def toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        self.is_playing = True
        self.btn_play.setText("❚❚ Pause")
        interval_ms = int(1000.0 / self.fps)
        self.timer.start(interval_ms)

    def pause(self):
        self.is_playing = False
        self.btn_play.setText("▶ Play")
        self.timer.stop()

    def step_forward(self):
        cur = self.slider.value()
        if cur < self.slider.maximum():
            self.set_frame(cur + 1)
        else:
            self.set_frame(self.slider.minimum())

    def step_backward(self):
        cur = self.slider.value()
        if cur > self.slider.minimum():
            self.set_frame(cur - 1)
        else:
            self.set_frame(self.slider.maximum())

    def _on_tick(self):
        self.step_forward()
