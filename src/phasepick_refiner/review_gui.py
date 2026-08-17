"""Qt controls for reviewing master events across station pages."""

import sys

import pandas as pd
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from .configuration import ProjectConfiguration
from .data import PickDataset
from .review_plot import WaveformPagePlotter
from .review_session import MasterReviewSession
from .waveforms import WaveformArchive


class MasterReviewWindow(QMainWindow):
    """Responsive master-event viewer with controls in a right sidebar."""

    def __init__(
        self,
        configuration: ProjectConfiguration,
        dataset: PickDataset,
        waveform_archive: WaveformArchive,
    ):
        super().__init__()
        self.configuration = configuration
        self.dataset = dataset
        self.waveform_archive = waveform_archive
        self.session = MasterReviewSession(
            configuration, dataset, waveform_archive
        )
        self.plotter = WaveformPagePlotter(
            configuration, dataset, self.session
        )
        self.active_pick_phase = ""
        self.station_y_ranges: list[tuple[float, float, str]] = []
        self._changing_controls = False

        self.setWindowTitle("PhasePick Refiner - Master Review")
        self.setMinimumSize(1050, 700)
        self.resize(1320, 900)
        self.setStatusBar(QStatusBar())
        self._build_interface()
        self._populate_cluster_selector()
        self.application = QApplication.instance()
        if self.application is not None:
            # Spinbox editors consume letter keys before a normal window
            # shortcut sees them. An application filter receives those keys
            # first and applies them only while this window is active.
            self.application.installEventFilter(self)

    def _build_interface(self) -> None:
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.master_information = QLabel()
        self.master_information.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        # Explicit margins are more stable than constrained_layout for the
        # dense 15-row station plot and avoid its collapsed-axes warning.
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.master_information)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=1)

        controls_widget = QWidget()
        controls_widget.setFixedWidth(255)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(8, 4, 8, 4)
        controls_layout.setSpacing(7)

        controls_layout.addWidget(QLabel("Cluster"))
        self.cluster_selector = QComboBox()
        controls_layout.addWidget(self.cluster_selector)

        # Cluster navigation is the common review workflow. The master-event
        # dropdown remains available for clusters with several local masters.
        cluster_buttons = QHBoxLayout()
        self.previous_cluster_button = QPushButton("Previous")
        self.next_cluster_button = QPushButton("Next")
        cluster_buttons.addWidget(self.previous_cluster_button)
        cluster_buttons.addWidget(self.next_cluster_button)
        controls_layout.addLayout(cluster_buttons)

        controls_layout.addWidget(QLabel("Master event"))
        self.event_selector = QComboBox()
        controls_layout.addWidget(self.event_selector)

        controls_layout.addWidget(QLabel("Station page"))
        page_buttons = QHBoxLayout()
        self.previous_page_button = QPushButton("Previous")
        self.next_page_button = QPushButton("Next")
        page_buttons.addWidget(self.previous_page_button)
        page_buttons.addWidget(self.next_page_button)
        controls_layout.addLayout(page_buttons)
        self.page_information = QLabel()
        controls_layout.addWidget(self.page_information)

        self.selected_station_information = QLabel(
            "Selected station: none"
        )
        self.selected_station_information.setWordWrap(True)
        controls_layout.addWidget(self.selected_station_information)

        controls_layout.addWidget(QLabel("Manual arrival"))
        pick_buttons = QHBoxLayout()
        self.p_pick_button = QPushButton("Pick P")
        self.s_pick_button = QPushButton("Pick S")
        pick_buttons.addWidget(self.p_pick_button)
        pick_buttons.addWidget(self.s_pick_button)
        controls_layout.addLayout(pick_buttons)
        self.cancel_pick_button = QPushButton("Cancel pick")
        controls_layout.addWidget(self.cancel_pick_button)

        self.reset_button = QPushButton("Reset selected station")
        self.save_button = QPushButton("Save masters")
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.save_button)

        display_form = QFormLayout()
        display_form.setHorizontalSpacing(7)
        self.low_filter_input = self._number_input(
            configuration_value=(
                self.configuration.correlation_settings
                .filter_frequency_hz[0]
            ),
            minimum=0.01,
            maximum=100.0,
            step=0.5,
        )
        self.high_filter_input = self._number_input(
            configuration_value=(
                self.configuration.correlation_settings
                .filter_frequency_hz[1]
            ),
            minimum=0.01,
            maximum=100.0,
            step=0.5,
        )
        view_start, view_end = (
            self.configuration.viewer_settings.x_limits_seconds
        )
        self.x_minimum_input = self._number_input(
            view_start, -1000.0, 1000.0, 1.0
        )
        self.x_maximum_input = self._number_input(
            view_end, -1000.0, 1000.0, 1.0
        )
        self.gain_input = self._number_input(
            self.configuration.viewer_settings.default_gain,
            0.1,
            10.0,
            0.1,
        )
        display_form.addRow("Filter low", self.low_filter_input)
        display_form.addRow("Filter high", self.high_filter_input)
        display_form.addRow("X minimum", self.x_minimum_input)
        display_form.addRow("X maximum", self.x_maximum_input)
        display_form.addRow("Gain", self.gain_input)
        controls_layout.addLayout(display_form)

        self.apply_display_button = QPushButton("Apply display")
        controls_layout.addWidget(self.apply_display_button)
        controls_layout.addStretch(1)

        shortcut_information = QLabel(
            "P/S: pick mode\nEsc: listening\n"
            "Left/Right: cluster\nPage Up/Down: station page"
        )
        controls_layout.addWidget(shortcut_information)

        main_layout.addWidget(plot_widget, stretch=1)
        main_layout.addWidget(controls_widget)
        self.setCentralWidget(central_widget)
        self._connect_controls()
        self._create_shortcuts()

    def _connect_controls(self) -> None:
        self.cluster_selector.currentTextChanged.connect(
            self._cluster_changed
        )
        self.event_selector.currentTextChanged.connect(
            self._event_changed
        )
        self.previous_cluster_button.clicked.connect(
            lambda: self._change_cluster(-1)
        )
        self.next_cluster_button.clicked.connect(
            lambda: self._change_cluster(1)
        )
        self.previous_page_button.clicked.connect(
            lambda: self._change_page(-1)
        )
        self.next_page_button.clicked.connect(
            lambda: self._change_page(1)
        )
        self.p_pick_button.clicked.connect(
            lambda: self._start_pick_mode("P")
        )
        self.s_pick_button.clicked.connect(
            lambda: self._start_pick_mode("S")
        )
        self.cancel_pick_button.clicked.connect(self._cancel_pick_mode)
        self.reset_button.clicked.connect(self._reset_selected_station)
        self.save_button.clicked.connect(self._save_master_table)
        self.apply_display_button.clicked.connect(
            self._apply_display_settings
        )
        self.canvas.mpl_connect("button_press_event", self._plot_clicked)
        self.canvas.mpl_connect("scroll_event", self._plot_scrolled)

    def _create_shortcuts(self) -> None:
        """Keep picking shortcuts active when the canvas owns keyboard focus."""
        self.p_pick_shortcut = QShortcut(QKeySequence("P"), self)
        self.s_pick_shortcut = QShortcut(QKeySequence("S"), self)
        self.cancel_pick_shortcut = QShortcut(
            QKeySequence(Qt.Key_Escape), self
        )
        self.p_pick_shortcut.setContext(Qt.WindowShortcut)
        self.s_pick_shortcut.setContext(Qt.WindowShortcut)
        self.cancel_pick_shortcut.setContext(Qt.WindowShortcut)
        self.p_pick_shortcut.activated.connect(
            lambda: self._start_pick_mode("P")
        )
        self.s_pick_shortcut.activated.connect(
            lambda: self._start_pick_mode("S")
        )
        self.cancel_pick_shortcut.activated.connect(
            self._cancel_pick_mode
        )

    @staticmethod
    def _number_input(
        configuration_value: float,
        minimum: float,
        maximum: float,
        step: float,
    ) -> QDoubleSpinBox:
        input_widget = QDoubleSpinBox()
        input_widget.setRange(minimum, maximum)
        input_widget.setDecimals(2)
        input_widget.setSingleStep(step)
        input_widget.setValue(float(configuration_value))
        return input_widget

    def _populate_cluster_selector(self) -> None:
        cluster_ids = self.session.cluster_ids()
        self._changing_controls = True
        self.cluster_selector.clear()
        self.cluster_selector.addItems(cluster_ids)
        self._changing_controls = False
        if cluster_ids:
            self._cluster_changed(cluster_ids[0])
        else:
            self.statusBar().showMessage(
                "No selected station-cluster masters are available"
            )

    def _cluster_changed(self, cluster_id: str) -> None:
        if self._changing_controls or not cluster_id:
            return
        event_ids = self.session.master_events(cluster_id)
        self._changing_controls = True
        self.event_selector.clear()
        self.event_selector.addItems(event_ids)
        self._changing_controls = False
        if event_ids:
            self._load_selected_event()

    def _event_changed(self, event_id: str) -> None:
        if self._changing_controls or not event_id:
            return
        self._load_selected_event()

    def _load_selected_event(self) -> None:
        cluster_id = self.cluster_selector.currentText().strip()
        event_id = self.event_selector.currentText().strip()
        if not cluster_id or not event_id:
            return
        self.session.load_event(cluster_id, event_id)
        self._draw_waveforms()

    def _draw_waveforms(self, preserve_x_limits: bool = False) -> None:
        if self._changing_controls or not self.session.event_id:
            return
        if self.x_maximum_input.value() <= self.x_minimum_input.value():
            self.statusBar().showMessage(
                "X maximum must be greater than X minimum"
            )
            return
        if self.high_filter_input.value() <= self.low_filter_input.value():
            self.statusBar().showMessage(
                "High filter frequency must be above low frequency"
            )
            return

        # A click redraws arrival markers. Preserve the user's current zoom
        # during that redraw; page/event/display changes intentionally reset it.
        current_x_limits = None
        if preserve_x_limits and self.figure.axes:
            current_x_limits = self.figure.axes[0].get_xlim()

        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self.figure.subplots_adjust(
            left=0.12,
            right=0.985,
            bottom=0.075,
            top=0.91,
        )
        result = self.plotter.draw(
            axis,
            low_frequency=self.low_filter_input.value(),
            high_frequency=self.high_filter_input.value(),
            gain=self.gain_input.value(),
            x_minimum=self.x_minimum_input.value(),
            x_maximum=self.x_maximum_input.value(),
        )
        if current_x_limits is not None:
            axis.set_xlim(current_x_limits)
        self.station_y_ranges = result.station_y_ranges
        self.master_information.setText(
            f"{len(self.session.station_ids)} waveform stations | "
            f"{self.session.stations_per_page} stations per page | "
            "* = this event is the station's local CC master"
        )
        self.page_information.setText(
            f"Page {self.session.current_page + 1} "
            f"of {self.session.page_count()}"
        )
        self._update_selected_station_information()
        self.canvas.draw_idle()

    def _update_selected_station_information(self) -> None:
        station_id = self.session.selected_station_id
        if not station_id:
            self.selected_station_information.setText(
                "Selected station: none"
            )
            return
        status = self.session.station_status(station_id)
        distance_km = self.session.station_distance_km(station_id)
        distance_text = (
            f"{distance_km:.1f} km"
            if distance_km is not None
            else "unavailable"
        )
        self.selected_station_information.setText(
            f"Selected station:\n{station_id}\n"
            f"Epicentral distance: {distance_text}\n{status}"
        )

    def _change_cluster(self, step: int) -> None:
        cluster_count = self.cluster_selector.count()
        if cluster_count == 0:
            return
        next_index = (
            self.cluster_selector.currentIndex() + step
        ) % cluster_count
        self.cluster_selector.setCurrentIndex(next_index)

    def _change_page(self, step: int) -> None:
        self.session.change_page(step)
        self._draw_waveforms()

    def _apply_display_settings(self) -> None:
        self._draw_waveforms()
        self.canvas.setFocus(Qt.OtherFocusReason)

    def _start_pick_mode(self, phase: str) -> None:
        self.active_pick_phase = phase
        self.statusBar().showMessage(
            f"{phase}-pick mode: click any station waveform row"
        )

    def _cancel_pick_mode(self) -> None:
        self.active_pick_phase = ""
        self.statusBar().showMessage("Listening mode")

    def _plot_clicked(self, event: object) -> None:
        if (
            event.inaxes is None
            or event.xdata is None
            or event.ydata is None
            or event.button != 1
            or self.toolbar.mode
        ):
            return
        station_id = self._station_at_y(float(event.ydata))
        if not station_id:
            return
        self.session.selected_station_id = station_id

        if not self.active_pick_phase:
            self._draw_waveforms(preserve_x_limits=True)
            return
        origin_time = self.dataset.origin_time(self.session.event_id)
        if origin_time is None:
            return
        reviewed_time = origin_time + pd.Timedelta(
            seconds=float(event.xdata)
        )
        phase = self.active_pick_phase
        self.active_pick_phase = ""
        message = self.session.record_manual_pick(
            station_id, phase, reviewed_time
        )
        self.statusBar().showMessage(message)
        self._draw_waveforms(preserve_x_limits=True)

    def _station_at_y(self, y_position: float) -> str:
        for lower_y, upper_y, station_id in self.station_y_ranges:
            if lower_y <= y_position <= upper_y:
                return station_id
        return ""

    def _reset_selected_station(self) -> None:
        message = self.session.reset_selected_station()
        self.statusBar().showMessage(message)
        self._draw_waveforms()

    def _save_master_table(self) -> None:
        output_file, edited_count, pending_count = self.session.save()
        message = (
            f"Saved {edited_count} reviewed phase picks to {output_file}"
        )
        if pending_count:
            message += (
                f"; {pending_count} incomplete manual phases remain unsaved"
            )
        self.statusBar().showMessage(message)

    def _plot_scrolled(self, event: object) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        left_limit, right_limit = event.inaxes.get_xlim()
        scale = 0.8 if event.button == "up" else 1.25
        new_left = event.xdata - (event.xdata - left_limit) * scale
        new_right = event.xdata + (right_limit - event.xdata) * scale
        event.inaxes.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def keyPressEvent(self, event: object) -> None:
        if event.key() == Qt.Key_P:
            self._start_pick_mode("P")
        elif event.key() == Qt.Key_S:
            self._start_pick_mode("S")
        elif event.key() == Qt.Key_Escape:
            self._cancel_pick_mode()
        elif event.key() == Qt.Key_Right:
            self._change_cluster(1)
        elif event.key() == Qt.Key_Left:
            self._change_cluster(-1)
        elif event.key() == Qt.Key_PageDown:
            self._change_page(1)
        elif event.key() == Qt.Key_PageUp:
            self._change_page(-1)
        else:
            super().keyPressEvent(event)

    def eventFilter(self, watched: object, event: object) -> bool:
        """Handle picking keys even when a numeric editor has focus."""
        if self.isActiveWindow() and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_P:
                self._start_pick_mode("P")
                return True
            if event.key() == Qt.Key_S:
                self._start_pick_mode("S")
                return True
            if event.key() == Qt.Key_Escape:
                self._cancel_pick_mode()
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event: object) -> None:
        if self.session.pending_manual_picks:
            decision = QMessageBox.question(
                self,
                "Incomplete manual picks",
                "Some stations have only P or only S selected. Incomplete "
                "pairs cannot become CC masters and will not be saved. "
                "Close anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if decision != QMessageBox.Yes:
                event.ignore()
                return
        self._save_master_table()
        if self.application is not None:
            self.application.removeEventFilter(self)
        event.accept()


def open_master_review_window(
    configuration: ProjectConfiguration,
    dataset: PickDataset,
    waveform_archive: WaveformArchive,
) -> None:
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)

    window = MasterReviewWindow(
        configuration, dataset, waveform_archive
    )
    window.show()
    if owns_application:
        application.exec_()
