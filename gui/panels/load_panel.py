"""
load_panel.py — Left sidebar: file paths, parameters, Load button.
"""
from __future__ import annotations
from typing import Optional
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog,
)
from qtpy.QtCore import Signal


class LoadPanel(QWidget):
    """Sidebar widget for configuring and triggering data load."""

    load_requested = Signal(dict)  # dict of all params
    sorter_load_requested = Signal(str)  # path to KS dir or LH HDF5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Recording ──────────────────────────────────────────────
        rec_grp = QGroupBox("Recording")
        rec_layout = QVBoxLayout()

        self._dat_path, self._dat_btn = self._file_row(
            rec_layout, ".dat/.bin", ".dat"
        )
        self._n_channels = QSpinBox()
        self._n_channels.setRange(1, 10000)
        self._n_channels.setValue(512)
        self._add_row(rec_layout, "n_channels", self._n_channels)

        self._fs = QSpinBox()
        self._fs.setRange(1000, 100000)
        self._fs.setValue(20000)
        self._add_row(rec_layout, "fs (Hz)", self._fs)

        self._start_min = QDoubleSpinBox()
        self._start_min.setRange(0, 10000)
        self._start_min.setValue(0.0)
        self._add_row(rec_layout, "start (min)", self._start_min)

        self._duration_min = QDoubleSpinBox()
        self._duration_min.setRange(0, 10000)
        self._duration_min.setSpecialValueText("Full file")
        self._duration_min.setValue(0.0)
        self._add_row(rec_layout, "duration (min)", self._duration_min)

        rec_grp.setLayout(rec_layout)
        layout.addWidget(rec_grp)

        # ── Channel Map ────────────────────────────────────────────
        map_grp = QGroupBox("Channel Map")
        map_layout = QVBoxLayout()
        self._map_path, self._map_btn = self._file_row(
            map_layout, ".npy", ".npy"
        )
        map_layout.addWidget(
            QLabel("(optional — fallback: linear layout)")
        )
        map_grp.setLayout(map_layout)
        layout.addWidget(map_grp)

        # ── Sorter Output (optional) ───────────────────────────────
        sorter_grp = QGroupBox("Sorter Output (optional)")
        sorter_layout = QVBoxLayout()
        self._sorter_path, self._sorter_btn = self._file_row(
            sorter_layout, "KS dir / LH .h5", ""
        )
        self._sorter_load_btn = QPushButton("Load Sorter")
        self._sorter_load_btn.clicked.connect(self._on_sorter_load)
        sorter_layout.addWidget(self._sorter_load_btn)
        sorter_layout.addWidget(QLabel("(for miss-rate display)"))
        sorter_grp.setLayout(sorter_layout)
        layout.addWidget(sorter_grp)

        # ── LH Params ──────────────────────────────────────────────
        param_grp = QGroupBox("LH Params")
        param_layout = QVBoxLayout()

        self._min_valid_count = QSpinBox()
        self._min_valid_count.setRange(10, 100000)
        self._min_valid_count.setValue(300)
        self._add_row(param_layout, "min_valid_count", self._min_valid_count)

        self._min_bl_bulk = QDoubleSpinBox()
        self._min_bl_bulk.setRange(0.0, 1.0)
        self._min_bl_bulk.setSingleStep(0.05)
        self._min_bl_bulk.setValue(0.70)
        self._add_row(param_layout, "min_bl_bulk", self._min_bl_bulk)

        self._max_snippets = QSpinBox()
        self._max_snippets.setRange(100, 100000)
        self._max_snippets.setValue(5000)
        self._add_row(param_layout, "max_snippets", self._max_snippets)

        param_grp.setLayout(param_layout)
        layout.addWidget(param_grp)

        layout.addStretch()

        # ── Load Button ────────────────────────────────────────────
        self._load_btn = QPushButton("Load Recording")
        self._load_btn.setMinimumHeight(36)
        self._load_btn.clicked.connect(self._on_load)
        layout.addWidget(self._load_btn)

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _file_row(parent_layout: QVBoxLayout, label: str, ext: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        line = QLineEdit()
        btn = QPushButton("Browse")
        btn.clicked.connect(
            lambda: LoadPanel._browse(line, ext)
        )
        row.addWidget(line)
        row.addWidget(btn)
        parent_layout.addLayout(row)
        return line, btn

    @staticmethod
    def _browse(line_edit: QLineEdit, ext: str):
        if ext:
            # Allow both .dat and .bin files
            path, _ = QFileDialog.getOpenFileName(
                None, "Select File", "", f"Data Files (*.dat *.bin);;All Files (*)"
            )
        else:
            path = QFileDialog.getExistingDirectory(
                None, "Select Directory"
            )
        if path:
            line_edit.setText(path)

    @staticmethod
    def _add_row(parent_layout: QVBoxLayout, label: str, widget):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(100)
        row.addWidget(lbl)
        row.addWidget(widget)
        parent_layout.addLayout(row)

    def _on_load(self):
        self.load_requested.emit(self.get_params())

    def _on_sorter_load(self):
        path = self._sorter_path.text().strip()
        if path:
            self.sorter_load_requested.emit(path)

    def get_params(self) -> dict:
        """Read all widget values and return as a params dict."""
        dur = self._duration_min.value()
        return {
            "dat_path": self._dat_path.text().strip(),
            "n_channels": self._n_channels.value(),
            "dtype": "int16",  # fixed default
            "fs": self._fs.value(),
            "start_min": self._start_min.value(),
            "duration_min": dur if dur > 0 else None,
            "map_path": self._map_path.text().strip() or None,
            "sorter_path": self._sorter_path.text().strip() or None,
            # LH params merged with DEFAULT_PARAMS later
            "min_valid_count": self._min_valid_count.value(),
            "min_bl_bulk": self._min_bl_bulk.value(),
            "max_snippets": self._max_snippets.value(),
        }

    def set_loading_state(self, loading: bool):
        """Disable/enable the Load button during I/O."""
        self._load_btn.setEnabled(not loading)
        self._load_btn.setText("Loading…" if loading else "Load Recording")

    def set_defaults(self, dat_path: Optional[str], n_channels: Optional[int]):
        """Pre-fill from CLI defaults."""
        if dat_path:
            self._dat_path.setText(dat_path)
        if n_channels:
            self._n_channels.setValue(n_channels)
