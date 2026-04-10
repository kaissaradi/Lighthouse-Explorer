"""
load_panel.py — Left sidebar: file paths, parameters, Load button.
Now supports both single .dat/.bin files (Kilosort) and Litke bin folders.
"""
from __future__ import annotations
from typing import Optional
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QFileDialog, QRadioButton,
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

        # ── Source type selection ──────────────────────────────────
        src_grp = QGroupBox("Source type")
        src_layout = QHBoxLayout()
        self.file_radio = QRadioButton("Single .dat/.bin file (Kilosort)")
        self.folder_radio = QRadioButton("Litke bin folder")
        self.file_radio.setChecked(True)
        src_layout.addWidget(self.file_radio)
        src_layout.addWidget(self.folder_radio)
        src_grp.setLayout(src_layout)
        layout.addWidget(src_grp)

        # ── Recording ──────────────────────────────────────────────
        rec_grp = QGroupBox("Recording")
        rec_layout = QVBoxLayout()

        # File path (for single file mode)
        self._dat_path, self._dat_btn = self._file_row(
            rec_layout, ".dat/.bin", ".dat"
        )
        # Folder path (for Litke mode) – initially hidden
        self._folder_path, self._folder_btn = self._folder_row(
            rec_layout, "Litke folder"
        )
        self._folder_path.setVisible(False)
        self._folder_btn.setVisible(False)

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

        # ── Sorter Output (optional) ───────────────────────────────
        sorter_grp = QGroupBox("Sorter Output (optional)")
        sorter_layout = QVBoxLayout()
        self._sorter_path, self._sorter_btn = self._folder_row(
            sorter_layout, "KS dir / LH .h5"
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

        self._min_trough = QSpinBox()
        self._min_trough.setRange(-5000, 0)
        self._min_trough.setValue(-2000)
        self._add_row(param_layout, "min_trough", self._min_trough)

        self._bin_width = QDoubleSpinBox()
        self._bin_width.setRange(1.0, 50.0)
        self._bin_width.setValue(10.0)
        self._add_row(param_layout, "bin_width", self._bin_width)

        param_grp.setLayout(param_layout)
        layout.addWidget(param_grp)

        layout.addStretch()

        # ── Load Button ────────────────────────────────────────────
        self._load_btn = QPushButton("Load Recording")
        self._load_btn.setMinimumHeight(36)
        self._load_btn.clicked.connect(self._on_load)
        layout.addWidget(self._load_btn)

        # Connect radio buttons to toggle visibility
        self.file_radio.toggled.connect(self._on_source_type_changed)

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _file_row(parent_layout: QVBoxLayout, label: str, ext: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        line = QLineEdit()
        btn = QPushButton("Browse")
        btn.clicked.connect(
            lambda: LoadPanel._browse_file(line, ext)
        )
        row.addWidget(line)
        row.addWidget(btn)
        parent_layout.addLayout(row)
        return line, btn

    @staticmethod
    def _folder_row(parent_layout: QVBoxLayout, label: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        line = QLineEdit()
        btn = QPushButton("Browse Folder")
        btn.clicked.connect(
            lambda: LoadPanel._browse_folder(line)
        )
        row.addWidget(line)
        row.addWidget(btn)
        parent_layout.addLayout(row)
        return line, btn

    @staticmethod
    def _browse_file(line_edit: QLineEdit, ext: str):
        if ext == ".npy":
            path, _ = QFileDialog.getOpenFileName(
                None, "Select File", "", "Numpy Files (*.npy);;All Files (*)"
            )
        elif ext:
            path, _ = QFileDialog.getOpenFileName(
                None, "Select File", "", "Data Files (*.dat *.bin);;All Files (*)"
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, "Select File", "", "All Files (*)"
            )
        if path:
            line_edit.setText(path)

    @staticmethod
    def _browse_folder(line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(None, "Select Litke Bin Folder")
        if folder:
            line_edit.setText(folder)

    @staticmethod
    def _add_row(parent_layout: QVBoxLayout, label: str, widget):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(100)
        row.addWidget(lbl)
        row.addWidget(widget)
        parent_layout.addLayout(row)

    def _on_source_type_changed(self):
        """Show/hide the appropriate path widget based on radio button selection."""
        is_file_mode = self.file_radio.isChecked()
        self._dat_path.setVisible(is_file_mode)
        self._dat_btn.setVisible(is_file_mode)
        self._folder_path.setVisible(not is_file_mode)
        self._folder_btn.setVisible(not is_file_mode)

    def _on_load(self):
        self.load_requested.emit(self.get_params())

    def _on_sorter_load(self):
        path = self._sorter_path.text().strip()
        if path:
            self.sorter_load_requested.emit(path)

    def get_params(self) -> dict:
        """Read all widget values and return as a params dict."""
        dur = self._duration_min.value()
        is_litke = self.folder_radio.isChecked()

        if is_litke:
            dat_path = self._folder_path.text().strip()
        else:
            dat_path = self._dat_path.text().strip()

        params = {
            "dat_path": dat_path,
            "is_litke_folder": is_litke,
            "n_channels": self._n_channels.value(),
            "dtype": "int16",  # fixed default
            "fs": self._fs.value(),
            "start_min": self._start_min.value(),
            "duration_min": dur if dur > 0 else None,
            "sorter_path": self._sorter_path.text().strip() or None,
            "min_valid_count": self._min_valid_count.value(),
            "support_min_bl_bulk": self._min_bl_bulk.value(),
            "min_trough": self._min_trough.value(),
            "bin_width": self._bin_width.value(),
        }
        return params

    def set_loading_state(self, loading: bool):
        """Disable/enable the Load button during I/O."""
        self._load_btn.setEnabled(not loading)
        self._load_btn.setText("Loading…" if loading else "Load Recording")

    def set_defaults(self, dat_path: Optional[str], n_channels: Optional[int]):
        """Pre-fill from CLI defaults."""
        if dat_path:
            # If the default path is a directory, assume Litke folder and switch mode
            import os
            if os.path.isdir(dat_path):
                self.folder_radio.setChecked(True)
                self._folder_path.setText(dat_path)
            else:
                self.file_radio.setChecked(True)
                self._dat_path.setText(dat_path)
        if n_channels:
            self._n_channels.setValue(n_channels)
        # Ensure correct visibility after defaults
        self._on_source_type_changed()