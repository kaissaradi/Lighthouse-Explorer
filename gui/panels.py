from __future__ import annotations
from .main_window import COLORS

"""
load_panel.py — Left sidebar: file paths, parameters, Load button.
Now supports both single .dat/.bin files (Kilosort) and Litke bin folders.
"""
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
            "batch_max_workers": 1,
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

"""
array_map_panel.py — Channel selector with grouped views and progress bar.
"""
from typing import Optional
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor, QBrush
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QComboBox,
)


class _ChannelListWidget(QListWidget):
    """QListWidget that emits on arrow-key navigation in addition to clicks."""

    channel_activated = Signal(int)  # emitted on arrow key move or Enter

    def keyPressEvent(self, event):
        super().keyPressEvent(event)  # let Qt move the selection first
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Return, Qt.Key_Enter):
            item = self.currentItem()
            if item is not None:
                ch = item.data(Qt.UserRole)
                if ch is not None:
                    self.channel_activated.emit(int(ch))


class ArrayMapPanel(QWidget):
    """Channel selector with groups: All / LH Found / Uncertain / No LH."""

    channel_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_channels: int = 0
        self._selected_ch: Optional[int] = None
        self._qc_results: dict = {}  # {ch: QCResult}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["All", "LH Found", "Uncertain", "No LH"])
        self._view_combo.currentTextChanged.connect(self._rebuild_list)
        toolbar.addWidget(self._view_combo)
        toolbar.addWidget(QLabel("CH:"))
        self._ch_input = QLineEdit()
        self._ch_input.setPlaceholderText("#")
        self._ch_input.returnPressed.connect(self._on_go)
        self._ch_input.setMaximumWidth(50)
        toolbar.addWidget(self._ch_input)
        self._go_btn = QPushButton("Go")
        self._go_btn.clicked.connect(self._on_go)
        toolbar.addWidget(self._go_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumHeight(14)
        layout.addWidget(self._progress_bar)

        # Progress label
        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._progress_lbl.setVisible(False)
        layout.addWidget(self._progress_lbl)

        # Channel list — custom subclass handles keyboard
        self._channel_list = _ChannelListWidget()
        self._channel_list.setFocusPolicy(Qt.StrongFocus)
        self._channel_list.enterEvent = lambda e: self._channel_list.setFocus()
        self._channel_list.itemClicked.connect(self._on_item_clicked)
        self._channel_list.channel_activated.connect(self.channel_selected)
        layout.addWidget(self._channel_list)

        # Status label
        self._status_lbl = QLabel("No recording loaded.")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_lbl)

    # ── public API ─────────────────────────────────────────────────

    def set_array(self, ei_positions):
        import numpy as np
        positions = np.asarray(ei_positions)
        self._n_channels = positions.shape[0]
        self._selected_ch = None
        self._qc_results.clear()
        self._status_lbl.setText(f"{self._n_channels} channels loaded.")
        self._rebuild_list()

    def set_progress(self, current: int, total: int, message: str = ""):
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_lbl.setVisible(True)
        self._progress_lbl.setText(message or f"CH {current}/{total}")

    def hide_progress(self):
        self._progress_bar.setVisible(False)
        self._progress_lbl.setVisible(False)

    def update_channel_result(self, ch: int, result):
        """Update a single channel's label+color without rebuilding the whole list."""
        self._qc_results[ch] = result
        view = self._view_combo.currentText()

        # Check if this channel should be visible in the current view
        visible = self._ch_passes_filter(ch, result, view)

        # Find existing item for this channel
        existing_row = None
        for i in range(self._channel_list.count()):
            if self._channel_list.item(i).data(Qt.UserRole) == ch:
                existing_row = i
                break

        if visible:
            label, color = self._make_label_color(ch, result)
            if existing_row is not None:
                # Update in place — no clear(), selection preserved
                item = self._channel_list.item(existing_row)
                item.setText(label)
                item.setForeground(QBrush(color))
            else:
                # New item — insert in channel-number order
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, ch)
                item.setForeground(QBrush(color))
                insert_pos = self._find_insert_pos(ch)
                self._channel_list.insertItem(insert_pos, item)
        else:
            # Channel no longer passes filter — remove it
            if existing_row is not None:
                self._channel_list.takeItem(existing_row)

        self._update_status()

    def set_qc_result_color(self, ch: int, miss_rate: Optional[float]):
        result = self._qc_results.get(ch)
        if result:
            self.update_channel_result(ch, result)

    def set_selected_channel(self, ch: int):
        self._selected_ch = ch
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.data(Qt.UserRole) == ch:
                self._channel_list.setCurrentItem(item)
                self._channel_list.scrollToItem(item)
                break

    def clear(self):
        self._n_channels = 0
        self._selected_ch = None
        self._qc_results.clear()
        self._channel_list.clear()
        self._status_lbl.setText("No recording loaded.")
        self.hide_progress()

    # ── internals ──────────────────────────────────────────────────

    def _ch_passes_filter(self, ch: int, result, view: str) -> bool:
        if view == "All":
            return True
        if result is None:
            return False
        if view == "LH Found":
            return result.n_lh > 0
        if view == "Uncertain":
            return result.n_uncertain > 0
        if view == "No LH":
            return result.n_lh == 0 and result.n_total > 0
        return True

    def _make_label_color(self, ch: int, result):
        if result:
            label = f"CH {ch} — {result.n_lh} LH, {result.n_soup} soup, {result.n_uncertain} unc"
            lh_ratio = result.n_lh / result.n_total if result.n_total > 0 else 0
            color = QColor(int(255 * (1.0 - lh_ratio)), int(255 * lh_ratio), 40)
        else:
            label = f"CH {ch} — pending"
            color = QColor(136, 136, 136)
        return label, color

    def _find_insert_pos(self, ch: int) -> int:
        """Binary-search for insertion index maintaining channel order."""
        lo, hi = 0, self._channel_list.count()
        while lo < hi:
            mid = (lo + hi) // 2
            if self._channel_list.item(mid).data(Qt.UserRole) < ch:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _get_group_channels(self) -> list[int]:
        view = self._view_combo.currentText()
        if view == "All":
            return list(range(self._n_channels))
        channels = []
        for ch in range(self._n_channels):
            result = self._qc_results.get(ch)
            if self._ch_passes_filter(ch, result, view):
                channels.append(ch)
        return channels

    def _rebuild_list(self):
        """Full rebuild — only called on view-filter change or initial load."""
        # Remember current item so we can restore it
        cur_item = self._channel_list.currentItem()
        cur_ch = cur_item.data(Qt.UserRole) if cur_item else self._selected_ch

        self._channel_list.blockSignals(True)
        self._channel_list.clear()
        channels = self._get_group_channels()
        for ch in channels:
            result = self._qc_results.get(ch)
            label, color = self._make_label_color(ch, result)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, ch)
            item.setForeground(QBrush(color))
            self._channel_list.addItem(item)
        self._channel_list.blockSignals(False)

        self._update_status()

        # Restore selection
        if cur_ch is not None:
            self.set_selected_channel(cur_ch)

    def _update_status(self):
        shown = self._channel_list.count()
        self._status_lbl.setText(f"Showing {shown}/{self._n_channels} channels.")

    def _on_item_clicked(self, item: QListWidgetItem):
        ch = item.data(Qt.UserRole)
        if ch is not None:
            self.channel_selected.emit(int(ch))

    def _on_go(self):
        text = self._ch_input.text().strip()
        if not text:
            return
        try:
            ch = int(text)
        except ValueError:
            return
        if 0 <= ch < self._n_channels:
            self._view_combo.setCurrentText("All")
            self._select_channel_by_index(ch)
            self._ch_input.clear()

    def _select_channel_by_index(self, ch: int):
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.data(Qt.UserRole) == ch:
                self._channel_list.setCurrentItem(item)
                self._channel_list.scrollToItem(item)
                self.channel_selected.emit(ch)
                break
"""
qc_view_panel.py — Right side: 4 pyqtgraph plots (2×2) + summary stats bar.
"""
from typing import Optional
import pyqtgraph as pg
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from qtpy.QtCore import Qt
from core.lh_qc_pipeline import QCResult

# Re-export with the keys this module historically used (uppercase "LH")
COLORS = {
    "LH": COLORS["lh"],
    "soup": COLORS["soup"],
    "uncertain_boundary": COLORS["uncertain_boundary"],
    "uncertain_lowBL": COLORS["uncertain_lowBL"],
    "cluster0": COLORS["cluster0"],
    "cluster1": COLORS["cluster1"],
}


class QCViewPanel(QWidget):
    """QC visualization panel with 4 plots and summary bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_result: Optional[QCResult] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Summary bar ──────────────────────────────────────────
        self._summary_bar = QHBoxLayout()
        self._summary_labels: list[QLabel] = []
        for _ in range(8):
            lbl = QLabel("—")
            lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #F0F0F2;")
            self._summary_labels.append(lbl)
            self._summary_bar.addWidget(lbl)
        self._summary_bar.addStretch()
        layout.addLayout(self._summary_bar)

        # ── 2×2 Plot Grid ────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(4)

        self._plot_hist = pg.PlotWidget()
        self._plot_pca = pg.PlotWidget()
        self._plot_waveforms = pg.PlotWidget()

        self._plot_hist.showGrid(x=True, y=True, alpha=0.15)
        self._plot_pca.showGrid(x=True, y=True, alpha=0.15)
        self._plot_waveforms.showGrid(x=True, y=True, alpha=0.15)

        # Bottom-left: stacked KS fragmentation bar + FR overlay
        self._fr_layout = pg.GraphicsLayoutWidget()
        self._plot_fr   = self._fr_layout.addPlot(row=0, col=0)   # KS fragmentation
        self._plot_fr_time = self._fr_layout.addPlot(row=1, col=0) # FR over time
        self._fr_layout.ci.layout.setRowStretchFactor(0, 3)
        self._fr_layout.ci.layout.setRowStretchFactor(1, 2)
        self._plot_fr.showGrid(x=True, y=True, alpha=0.15)
        self._plot_fr_time.showGrid(x=True, y=True, alpha=0.15)

        grid.addWidget(self._plot_hist, 0, 0)
        grid.addWidget(self._plot_pca, 0, 1)
        grid.addWidget(self._fr_layout, 1, 0)
        grid.addWidget(self._plot_waveforms, 1, 1)

        # Set stretch — give equal weight
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        layout.addLayout(grid)

        # ── Placeholder label ─────────────────────────────────────
        self._placeholder = QLabel(
            "Select a channel and run QC to see results here."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #5A5C65; font-size: 14px;"
        )
        layout.addWidget(self._placeholder)

    def show_result(self, result: QCResult):
        """Main entry point. Update all 4 plots and summary bar."""
        self._current_result = result
        
        # Intercept rejected channels
        if result.reject_reason:
            self._placeholder.setText(f"Channel {result.channel} Rejected: {result.reject_reason}")
            self._placeholder.setStyleSheet("color: #F08080; font-size: 16px; font-weight: bold;")
            self._placeholder.show()
            self._update_summary_bar(result)
            self._update_amp_histogram(result)
            self._update_pca_scatter(result)
            self._update_fr_plot(result)
            self._update_fr_time_plot(result)
            self._update_waveforms(result)
            self._update_summary_bar(result)
            return
            
        # Standard display for valid channels
        self._placeholder.hide()
        self._update_summary_bar(result)
        self._update_amp_histogram(result)
        self._update_pca_scatter(result)
        self._update_fr_plot(result)
        self._update_fr_time_plot(result)
        self._update_waveforms(result)

    def show_loading(self, channel: int):
        """Show 'Running QC on CH X...' placeholder."""
        self._placeholder.setText(f"Running QC on CH {channel}…")
        self._placeholder.show()
        self._clear_plots()

    def show_error(self, msg: str):
        """Show error message."""
        self._placeholder.setText(f"Error: {msg}")
        self._placeholder.setStyleSheet("color: #F08080; font-size: 14px;")
        self._placeholder.show()
        self._clear_plots()

    def clear(self):
        """Reset all plots to empty state."""
        self._current_result = None
        self._placeholder.setText(
            "Select a channel and run QC to see results here."
        )
        self._placeholder.setStyleSheet("color: #5A5C65; font-size: 14px;")
        self._placeholder.show()
        self._clear_plots()
        for lbl in self._summary_labels:
            lbl.setText("—")

    # ── plot updaters ────────────────────────────────────────────

    def _clear_plots(self):
        for p in [
            self._plot_hist, self._plot_pca,
            self._plot_fr, self._plot_fr_time,
            self._plot_waveforms,
        ]:
            p.clear()

    def _update_summary_bar(self, result: QCResult):
        fields = [
            f"CH: {result.channel}",
            f"Total: {result.n_total}",
            f"LH: {result.n_lh} ({result.n_lh/max(1,result.n_total)*100:.0f}%)",
            f"Soup: {result.n_soup} ({result.n_soup/max(1,result.n_total)*100:.0f}%)",
            f"Uncertain: {result.n_uncertain}",
            f"Sorter: {result.n_sorter_spikes}",
            f"Miss: {result.miss_rate:.1%}" if result.miss_rate is not None else "Miss: N/A",
            f"Valley {'OK' if result.valley.accepted else 'N/A'}",
        ]
        for i, lbl in enumerate(self._summary_labels):
            lbl.setText(fields[i] if i < len(fields) else "—")

    def _update_amp_histogram(self, result: QCResult):
        p = self._plot_hist
        p.clear()

        vals = result.valley.all_vals
        if vals.size == 0:
            p.setTitle("No crossings")
            return

        counts, edges = result.valley.amp_hist_counts, result.valley.amp_hist_edges

        # pyqtgraph with stepMode=True requires len(x) == len(y) + 1
        p.plot(
            edges,
            counts,
            stepMode=True,
            fillLevel=0,
            brush="#2E6DD4",
            pen=pg.mkPen("#4A8BEF"),
        )

        # Valley threshold line
        if result.valley.valley_low is not None:
            vline = pg.InfiniteLine(
                pos=result.valley.valley_low,
                angle=90,
                pen=pg.mkPen("#FF9800", width=1.5, style=Qt.DashLine),
            )
            p.addItem(vline)

        p.setTitle(f"Amplitude Histogram (CH {result.channel})")
        p.setLabel("bottom", "ADC amplitude")
        p.setLabel("left", "Count")
        
        # --- NEW CODE: Clamp the visual zoom ---
        # Find the absolute minimum edge, but don't let the view zoom out further than -800
        view_min = max(np.min(edges), -800)
        p.setXRange(view_min, 0, padding=0)

        p.setYRange(0, 2000, padding=0)
        
    def _update_pca_scatter(self, result: QCResult):
        p = self._plot_pca
        p.clear()

        coords = result.pca_km.pca_coords
        labels = result.pca_km.km_labels
        if coords.shape[0] == 0:
            p.setTitle("No spikes")
            return

        evr = result.pca_km.explained_variance_ratio

        for cluster_id, color in [(0, COLORS["cluster0"]), (1, COLORS["cluster1"])]:
            mask = labels == cluster_id
            if mask.any():
                x = coords[mask, 0]
                y = coords[mask, 1]
                p.plot(
                    x, y,
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolBrush=pg.mkBrush(color),
                    symbolPen=None,
                )

        p.setTitle(f"PCA: PC1 vs PC2 (CH {result.channel})")
        p.setLabel(
            "bottom",
            f"PC1 ({evr[0]*100:.0f}%)" if evr.size > 0 else "PC1",
        )
        p.setLabel(
            "left",
            f"PC2 ({evr[1]*100:.0f}%)" if evr.size > 1 else "PC2",
        )

    def _update_fr_plot(self, result):
        """Venn Diagram replacing KS fragmentation bar chart."""
        p = self._plot_fr
        p.clear()

        fs = getattr(result, "fs", 20_000)
        coincidence_samp = int(0.001 * fs)  # ±1 ms window

        # 1. Gather all LH spikes
        lh_times = np.array([], dtype=np.int64)
        if hasattr(result, "valley"):
            parts = [t for t in [result.valley.left_times, result.valley.rightk_times] if t.size]
            if parts:
                lh_times = np.sort(np.concatenate(parts))

        # 2. Gather all KS spikes for this channel
        sorter_unit_map = getattr(result, "sorter_unit_map", {})
        if sorter_unit_map:
            all_t = np.concatenate(list(sorter_unit_map.values()))
            ks_times = np.sort(all_t)
        else:
            ks_times = np.array([], dtype=np.int64)

        if not sorter_unit_map:
            p.setTitle(f"LH vs KS Venn Diagram — CH {result.channel} (no sorter)")
            return
        if lh_times.size == 0:
            p.setTitle(f"LH vs KS Venn Diagram — CH {result.channel} (no LH spikes)")
            return

        # 3. Fast two-pointer matching (shared utility)
        from core.loader import match_spikes
        matched, lh_only, ks_only, _ = match_spikes(lh_times, ks_times, coincidence_samp)

        # 4. Draw the Venn Diagram 
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1.0
        
        # Circle boundaries
        x_lh = -0.5 + r * np.cos(theta)
        y_lh = r * np.sin(theta)
        
        x_ks = 0.5 + r * np.cos(theta)
        y_ks = r * np.sin(theta)
        
        # Add outlines
        p.addItem(pg.PlotCurveItem(x_lh, y_lh, pen=pg.mkPen(COLORS["LH"], width=2))) # Green LH
        p.addItem(pg.PlotCurveItem(x_ks, y_ks, pen=pg.mkPen(COLORS["cluster0"], width=2))) # Blue KS
        
        # Add Text Labels
        text_lh = pg.TextItem(f"LH Only\n{lh_only}", color=COLORS["LH"], anchor=(0.5, 0.5))
        text_lh.setPos(-0.9, 0)
        p.addItem(text_lh)
        
        text_both = pg.TextItem(f"Matched\n{matched}", color="#FFFFFF", anchor=(0.5, 0.5))
        text_both.setPos(0, 0)
        p.addItem(text_both)
        
        text_ks = pg.TextItem(f"KS Only\n{ks_only}", color=COLORS["cluster0"], anchor=(0.5, 0.5))
        text_ks.setPos(0.9, 0)
        p.addItem(text_ks)

        # 5. Clean up plot aesthetics
        p.hideAxis('bottom')
        p.hideAxis('left')
        p.setAspectLocked(True) # Keep circles round
        p.setXRange(-2, 2, padding=0)
        p.setYRange(-1.5, 1.5, padding=0)
        
        win_ms = coincidence_samp * 1000 / fs
        p.setTitle(f"LH vs KS Venn Diagram — CH {result.channel} (±{win_ms:.0f} ms)")

    def _update_fr_time_plot(self, result):
        """FR over time: LH (green) vs KS sorter (blue) — bottom sub-panel."""
        p = self._plot_fr_time  # PlotItem
        p.clear()
        p.addLegend(offset=(10, 10), labelTextSize="8pt")

        fs = getattr(result, "fs", 20_000)
        bin_s = 1.0
        lh_times = np.array([], dtype=np.int64)
        if hasattr(result, "valley"):
            parts = [t for t in [result.valley.left_times, result.valley.rightk_times] if t.size]
            if parts:
                lh_times = np.sort(np.concatenate(parts))

        sorter_times = getattr(result, "sorter_times", None)

        all_times = lh_times.copy()
        if sorter_times is not None and sorter_times.size:
            all_times = np.concatenate([all_times, sorter_times])
        if all_times.size == 0:
            p.setTitle("FR over time (no spikes)")
            return

        n_bins = max(1, int(all_times.max() / fs / bin_s) + 1)
        bins = np.arange(n_bins + 1, dtype=np.float64) * bin_s

        if lh_times.size:
            lh_counts, _ = np.histogram(lh_times / fs, bins=bins)
            p.plot(bins[:-1], lh_counts.astype(np.float64),
                   pen=pg.mkPen(COLORS["LH"], width=1.2), name="LH")
        if sorter_times is not None and sorter_times.size:
            s_counts, _ = np.histogram(sorter_times / fs, bins=bins)
            p.plot(bins[:-1], s_counts.astype(np.float64),
                   pen=pg.mkPen(COLORS["cluster0"], width=1.2), name="KS")

        has_ks = sorter_times is not None and sorter_times.size > 0
        p.setTitle(f"FR over time — CH {result.channel}" + ("" if has_ks else " (no KS)"))
        p.setLabel("bottom", "Time (s)")
        p.setLabel("left", "Spikes/s")

    def _update_waveforms(self, result: QCResult):
        p = self._plot_waveforms
        p.clear()

        means = result.pca_km.cluster_mean_waveforms
        if len(means) < 2:
            p.setTitle("No waveforms")
            return

        L = means[0].shape[0]
        x = np.arange(L, dtype=np.float64)

        for k, (color, label) in enumerate([
            (COLORS["cluster0"], "Cluster 0"),
            (COLORS["cluster1"], "Cluster 1"),
        ]):
            p.plot(
                x, means[k],
                pen=pg.mkPen(color, width=1.5),
                name=label,
            )

        p.setTitle(f"Mean Waveforms on CH {result.channel}")
        p.setLabel("bottom", "Samples")
        p.setLabel("left", "ADC amplitude")
        p.addLegend(offset=(10, 10))
"""
qc_summary_dialog.py — Recording-level QC summary across all channels.

Shows a 2×2 pyqtgraph panel:
  [0,0] Miss rate distribution (histogram across LH channels)
  [0,1] KS fragmentation index (bar: % channels with 1 / 2 / 3+ / missed-only units)
  [1,0] LH spike count vs KS spike count scatter (one dot per channel)
  [1,1] Array map coloured by miss rate (or fragmentation if no sorter)

Open via:  QCSummaryDialog(qc_results, sorter_spike_units, fs, parent).exec_()
"""
from typing import Optional
import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
)
from qtpy.QtCore import Qt


def _fragmentation_index(sorter_unit_map: dict, lh_times: np.ndarray, fs: float) -> dict:
    """
    Given {unit_id: times} for one channel and its LH spike times,
    return a dict with keys: n_units_matched, n_missed, pct_missed, dominant_frac.
    Returns None if no sorter data.
    """
    if not sorter_unit_map or lh_times.size == 0:
        return None

    from core.loader import match_spikes

    coincidence_samp = int(0.001 * fs)

    # Overall match to get miss count
    all_ks_times = np.sort(np.concatenate(list(sorter_unit_map.values())))
    n_matched_total, n_missed, _, _ = match_spikes(lh_times, all_ks_times, coincidence_samp)

    # Per-unit match to determine which units actually participate
    match_counts: dict[int, int] = {}
    for uid, unit_times in sorter_unit_map.items():
        unit_times_sorted = np.sort(np.asarray(unit_times, dtype=np.int64))
        n_m, _, _, _ = match_spikes(lh_times, unit_times_sorted, coincidence_samp)
        if n_m > 0:
            match_counts[uid] = n_m

    n_total = lh_times.size
    dominant_frac = (max(match_counts.values()) / n_total) if match_counts else 0.0
    return dict(
        n_units_matched=len(match_counts),
        n_missed=n_missed,
        pct_missed=n_missed / max(1, n_total),
        dominant_frac=dominant_frac,
    )


class QCSummaryDialog(QDialog):
    """Pop-up recording-level QC summary panel."""

    def __init__(self, qc_results: dict, sorter_spike_units: dict,
                 fs: float = 20_000.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording QC Summary")
        self.resize(1100, 700)
        self._qc_results = qc_results
        self._sorter_spike_units = sorter_spike_units
        self._fs = fs
        self._build_ui()
        self._populate()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(6, 6, 6, 6)

        self._title = QLabel("Recording-level QC Summary")
        self._title.setStyleSheet("font-size: 14px; font-weight: bold; color: #E0E0E0;")
        vlay.addWidget(self._title)

        self._glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self._glw)

        # 2×2 plot grid
        self._p_miss   = self._glw.addPlot(row=0, col=0, title="Miss Rate Distribution")
        self._p_frag   = self._glw.addPlot(row=0, col=1, title="KS Fragmentation Index")
        self._p_scatter= self._glw.addPlot(row=1, col=0, title="LH vs KS Spike Count")
        self._p_map    = self._glw.addPlot(row=1, col=1, title="Array Map (colour = miss rate)")

        for p in [self._p_miss, self._p_frag, self._p_scatter, self._p_map]:
            p.showGrid(x=True, y=True, alpha=0.15)

        # Close button
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        hlay = QHBoxLayout()
        hlay.addStretch()
        hlay.addWidget(btn)
        vlay.addLayout(hlay)

    # ── Data ─────────────────────────────────────────────────────────────────

    def _populate(self):
        results = self._qc_results
        units   = self._sorter_spike_units
        fs      = self._fs

        # Collect per-channel stats — only LH channels
        lh_channels = [ch for ch, r in results.items() if r.n_lh > 0]
        if not lh_channels:
            self._title.setText("No LH channels found — run batch QC first.")
            return

        miss_rates       = []   # float, NaN if no sorter
        frag_n_units     = []   # int: 0=missed-only, 1=clean, 2=split, 3+=bad
        lh_counts        = []
        ks_counts        = []
        channels_ordered = sorted(lh_channels)

        for ch in channels_ordered:
            r = results[ch]
            lh_times = np.sort(np.concatenate([
                t for t in [r.valley.left_times, r.valley.rightk_times] if t.size
            ])) if (r.valley.left_times.size + r.valley.rightk_times.size) else np.array([], dtype=np.int64)

            lh_counts.append(lh_times.size)
            ks_counts.append(r.n_sorter_spikes if r.n_sorter_spikes >= 0 else 0)

            unit_map = units.get(ch, {})
            fi = _fragmentation_index(unit_map, lh_times, fs)

            if fi is None:
                miss_rates.append(np.nan)
                frag_n_units.append(-1)
            else:
                miss_rates.append(fi["pct_missed"])
                frag_n_units.append(fi["n_units_matched"])

        miss_arr = np.array(miss_rates)
        frag_arr = np.array(frag_n_units)
        lh_arr   = np.array(lh_counts,  dtype=np.float64)
        ks_arr   = np.array(ks_counts,  dtype=np.float64)
        has_sorter = not np.all(np.isnan(miss_arr))

        self._title.setText(
            f"Recording QC Summary — {len(lh_channels)} LH channels"
            + (f"  |  median miss rate: {np.nanmedian(miss_arr)*100:.1f}%" if has_sorter else "  |  (no sorter loaded)")
        )

        # ── [0,0] Miss rate histogram ─────────────────────────────────────────
        p = self._p_miss
        if has_sorter:
            valid = miss_arr[~np.isnan(miss_arr)] * 100  # pct
            counts, edges = np.histogram(valid, bins=np.linspace(0, 100, 21))
            p.addItem(pg.BarGraphItem(
                x=edges[:-1], height=counts, width=(edges[1]-edges[0])*0.9,
                brush=pg.mkBrush(COLORS["sorter"]), pen=pg.mkPen("#1a1a1a"),
            ))
            p.setLabel("bottom", "Miss rate (%)")
            p.setLabel("left", "# channels")
            # Median line
            med = float(np.nanmedian(valid))
            p.addItem(pg.InfiniteLine(
                pos=med, angle=90,
                pen=pg.mkPen(COLORS["soup"], width=1.5, style=Qt.DashLine),
                label=f"med={med:.1f}%",
                labelOpts={"color": COLORS["soup"], "position": 0.85},
            ))
        else:
            lbl = pg.TextItem("No sorter loaded", color=COLORS["muted"], anchor=(0.5, 0.5))
            p.addItem(lbl)
            lbl.setPos(0.5, 0.5)

        # ── [0,1] Fragmentation index bar chart ───────────────────────────────
        p = self._p_frag
        if has_sorter:
            bins_frag = {"Missed\nonly": 0, "1 unit\n(clean)": 1,
                         "2 units\n(split)": 2, "3+ units\n(bad)": 3}
            frag_counts = {}
            for label, lo in [("Missed\nonly", 0), ("1 unit\n(clean)", 1),
                               ("2 units\n(split)", 2), ("3+ units\n(bad)", 3)]:
                if lo == 0:
                    frag_counts[label] = int(np.sum(frag_arr == 0))
                elif lo == 3:
                    frag_counts[label] = int(np.sum(frag_arr >= 3))
                else:
                    frag_counts[label] = int(np.sum(frag_arr == lo))

            colors = [COLORS["frag_missed"], COLORS["frag_clean"], COLORS["frag_split"], COLORS["frag_bad"]]
            labels = list(frag_counts.keys())
            vals   = list(frag_counts.values())
            xs = np.arange(len(labels), dtype=np.float64)
            for i, (lbl, val, col) in enumerate(zip(labels, vals, colors)):
                p.addItem(pg.BarGraphItem(
                    x=[xs[i]], height=[val], width=0.7,
                    brush=pg.mkBrush(col), pen=pg.mkPen("#1a1a1a"),
                ))
            ax = p.getAxis("bottom")
            ax.setTicks([[(xi, lbl) for xi, lbl in zip(xs, labels)]])
            p.setLabel("left", "# LH channels")
        else:
            lbl = pg.TextItem("No sorter loaded", color=COLORS["muted"], anchor=(0.5, 0.5))
            p.addItem(lbl)
            lbl.setPos(0.5, 0.5)

        # ── [1,0] LH vs KS scatter ───────────────────────────────────────────
        p = self._p_scatter
        if has_sorter and ks_arr.max() > 0:
            # Colour by miss rate: green=low, red=high
            norm_miss = np.nan_to_num(miss_arr, nan=0.0)
            colors_scatter = [
                pg.mkBrush(
                    int(255 * m), int(255 * (1 - m)), 60, 200
                ) for m in norm_miss
            ]
            scatter = pg.ScatterPlotItem(
                x=lh_arr, y=ks_arr,
                size=7, pen=pg.mkPen(None), brush=colors_scatter,
            )
            p.addItem(scatter)
            # Identity line
            mx = max(lh_arr.max(), ks_arr.max()) * 1.05
            p.plot([0, mx], [0, mx], pen=pg.mkPen("#555", width=1, style=Qt.DashLine))
            p.setLabel("bottom", "LH spike count")
            p.setLabel("left", "KS spike count")
        else:
            p.plot(lh_arr, ks_arr, pen=None, symbol="o",
                   symbolSize=5, symbolBrush=pg.mkBrush(COLORS["lh"]))
            p.setLabel("bottom", "LH spike count")
            p.setLabel("left", "KS spike count (0 = no sorter)")

        # ── [1,1] Array map coloured by miss rate ────────────────────────────
        p = self._p_map
        p.setAspectLocked(True)

        # Use channel index as a proxy for position if no geometry available
        # Arrange channels in a sqrt(C) × sqrt(C) grid
        all_chs = sorted(results.keys())
        C = len(all_chs)
        ncols = max(1, int(np.ceil(np.sqrt(C))))

        xs_map, ys_map, brushes = [], [], []
        for i, ch in enumerate(all_chs):
            xs_map.append(i % ncols)
            ys_map.append(i // ncols)
            r = results[ch]
            if r.n_lh == 0:
                # Not an LH channel — grey
                brushes.append(pg.mkBrush(80, 80, 80, 180))
            elif not has_sorter:
                # LH but no sorter — green shade by spike count
                brushes.append(pg.mkBrush(COLORS["lh"]))
            else:
                miss = miss_arr[channels_ordered.index(ch)] if ch in channels_ordered else np.nan
                if np.isnan(miss):
                    brushes.append(pg.mkBrush(COLORS["lh"]))
                else:
                    r_val = int(255 * miss)
                    g_val = int(255 * (1 - miss))
                    brushes.append(pg.mkBrush(r_val, g_val, 60, 220))

        sc = pg.ScatterPlotItem(
            x=xs_map, y=ys_map,
            size=9, pen=pg.mkPen(None), brush=brushes,
        )
        p.addItem(sc)
        p.setLabel("bottom", "col")
        p.setLabel("left", "row")
        p.setTitle("Array map  (grey=no LH, green=low miss, red=high miss)")