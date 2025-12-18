#################################################################################
"""
Copyright: Homi Bhabha Centre for Science Education (TIFR)

Developed for the International Olympiad on Astronomy and Astrophysics (IOAA)
Primary Development: Shirish Pathare - HBCSE, TIFR

Based on original code by: Ashish Mhaske - IUCAA, Pune

With guidance from:
    Prof. Avinash Deshpande, RRI, Bengaluru and Prof. Surhud More, IUCAA, Pune

With thanks to: IOAA Academic Committee

#-----------------------------------------------------
"""
#################################################################################
import sys
import os
import re
import traceback
from datetime import datetime, timedelta
import subprocess
import time
from collections import deque
import math
from shutil import which

# --- Core Qt/GUI Imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QPushButton, QLabel, QLineEdit, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox, QInputDialog, QTextEdit,
    QDateEdit, QTimeEdit, QDoubleSpinBox
)
from PySide6.QtCore import QObject, QThread, Signal, QTimer, Qt, QDate, QTime, QEvent
from PySide6.QtGui import QFont, QPixmap, QGuiApplication

# --- Scientific/Plotting Imports ---
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.transforms import blended_transform_factory

# --- Data and Calculation Imports ---
from rtlsdr import RtlSdr
from astropy.coordinates import SkyCoord, Galactic, ICRS, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time
from PyAstronomy import pyasl
from scipy.signal import medfilt, find_peaks, iirnotch, filtfilt
from scipy.interpolate import interp1d

# --- Default Parameters & Constants ---
DEFAULT_SKY_TEMP = 5
DEFAULT_GROUND_TEMP = 300
DEFAULT_CENTER_FREQ = 1420.40575
DEFAULT_SAMPLE_RATE = 2.048
DEFAULT_NF = 512
DEFAULT_INT_TIME = 60
DEFAULT_GAIN = 50
MUMBAI_LATITUDE = 19.0760 * u.deg
MUMBAI_LONGITUDE = 72.8777 * u.deg
MUMBAI_ALTITUDE = 58 * u.m
mumbai_location = EarthLocation(lat=MUMBAI_LATITUDE, lon=MUMBAI_LONGITUDE, height=MUMBAI_ALTITUDE)
IST_OFFSET_HOURS = 5.5
SUN_APEX_RA_DEG = 270.2
SUN_APEX_DEC_DEG = 28.7
V_SUN_LSR_SPEED = 20.5
COLOR_BLUE = "#00008B"
COLOR_GREEN = "#27AE60"

# ====================================================================================
# WORKER THREAD FOR SDR TASKS
# ====================================================================================
class SdrWorker(QObject):
    plot_data = Signal(object, object, str)
    progress = Signal(str)
    finished = Signal()
    final_data_for_save = Signal(object, object)
    noise_freq_detected = Signal(float)

    def __init__(self, params, mode='acquire'):
        super().__init__()
        self.params = params
        self.mode = mode
        self.is_running = True
        self.sdr_instance = None

    def run(self):
        if self.mode == 'acquire':
            self.run_sdr_acquisition()
        elif self.mode == 'quick_look':
            self.run_quick_look()

    def run_sdr_acquisition(self):
        try:
            self.progress.emit("Initializing SDR...")
            sdr = RtlSdr()
            sdr.center_freq = self.params['center_freq']
            sdr.sample_rate = self.params['sample_rate']
            sdr.gain = self.params['gain']

            nfft = self.params['nfft']
            duration = self.params['duration']
            samples_per_read = int(sdr.sample_rate)
            num_reads = int(duration)
            frequencies = np.linspace(
                sdr.center_freq / 1e6 - (sdr.sample_rate / 1e6) / 2,
                sdr.center_freq / 1e6 + (sdr.sample_rate / 1e6) / 2,
                nfft
            )

            total_segments = 0
            psd_sum = np.zeros(nfft)

            for i in range(num_reads):
                if not self.is_running:
                    self.progress.emit("Recording cancelled.")
                    break

                self.progress.emit(f"Reading samples: {i+1}/{num_reads} seconds...")

                samples = sdr.read_samples(samples_per_read)

                samples -= np.mean(samples)

                num_segments_in_read = len(samples) // nfft
                for j in range(num_segments_in_read):
                    segment = samples[j*nfft:(j+1)*nfft]
                    windowed = segment * np.hanning(nfft)
                    fft = np.fft.fftshift(np.fft.fft(windowed, nfft))
                    psd_sum += np.abs(fft)**2

                total_segments += num_segments_in_read

                if total_segments > 0:
                    current_psd = (psd_sum / total_segments) * (sdr.sample_rate / nfft)
                    title = f'Processed {i+1}/{num_reads} seconds'
                    self.plot_data.emit(frequencies, 10 * np.log10(current_psd), title)

            sdr.close()

            if total_segments > 0:
                final_psd_linear = (psd_sum / total_segments) * (sdr.sample_rate / nfft)

                avg_fft = psd_sum / total_segments
                max_noise_index = np.argmax(avg_fft)
                noise_freq_mhz = frequencies[max_noise_index]

                self.progress.emit(f"Detected noise frequency: {noise_freq_mhz:.3f} MHz")
                self.noise_freq_detected.emit(noise_freq_mhz)
            else:
                final_psd_linear = np.zeros(nfft)
                self.noise_freq_detected.emit(-1)

            self.progress.emit("Finished recording.")
            self.final_data_for_save.emit(frequencies, final_psd_linear)

        except Exception as e:
            self.progress.emit(f"SDR Error: {e}")
        finally:
            self.finished.emit()

    def run_quick_look(self):
        try:
            self.progress.emit("Initializing SDR for Quick Look...")
            self.sdr_instance = RtlSdr()
            sdr = self.sdr_instance
            sdr.center_freq, sdr.sample_rate, sdr.gain = self.params['center_freq'], self.params['sample_rate'], self.params['gain']
            nfft = self.params['nfft']
            frequencies = np.linspace(
                sdr.center_freq / 1e6 - (sdr.sample_rate / 1e6) / 2,
                sdr.center_freq / 1e6 + (sdr.sample_rate / 1e6) / 2,
                nfft
            )

            samples_per_chunk = nfft * 1024
            psd_buffer = deque(maxlen=4)
            self.start_time = time.time()

            def process_samples_callback(samples, context):
                samples -= np.mean(samples)
                psd_sum = np.zeros(nfft)
                num_segments_in_read = len(samples) // nfft

                if num_segments_in_read > 0:
                    for j in range(num_segments_in_read):
                        segment = samples[j*nfft:(j+1)*nfft]
                        windowed_segment = segment * np.hanning(nfft)
                        fft = np.fft.fftshift(np.fft.fft(windowed_segment, nfft))
                        psd_sum += np.abs(fft)**2

                    current_psd = (psd_sum / num_segments_in_read) * (sdr.sample_rate / nfft)
                    psd_buffer.append(current_psd)
                    averaged_psd = np.mean(np.array(psd_buffer), axis=0)
                    elapsed = int(time.time() - self.start_time)
                    title = f'Quick Look ({elapsed}s)'
                    self.plot_data.emit(frequencies, 10 * np.log10(averaged_psd), title)

            sdr.read_samples_async(process_samples_callback, num_samples=samples_per_chunk)
            while self.is_running:
                time.sleep(0.1)
        except Exception as e:
            self.progress.emit(f"SDR Error: {e}")
        finally:
            if self.sdr_instance:
                try:
                    self.sdr_instance.cancel_read_async()
                except Exception:
                    pass
                self.sdr_instance.close()
            self.progress.emit("Quick Look stopped.")
            self.finished.emit()

    def stop(self):
        self.progress.emit("Stopping...")
        if self.sdr_instance and self.mode == 'quick_look':
            try:
                self.sdr_instance.cancel_read_async()
            except Exception:
                pass
        self.is_running = False

# ====================================================================================
# CUSTOM WIDGETS
# ====================================================================================
class LogoTabPageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        logo_label = QLabel()

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(script_dir, 'IOAA.png')

            logo_pixmap = QPixmap(logo_path)

            if logo_pixmap.isNull():
                print(f"Warning: Could not load logo from the absolute path: {logo_path}")
                logo_pixmap = QPixmap(128, 128)
                logo_pixmap.fill(Qt.transparent)

        except Exception as e:
            print(f"An error occurred while setting up the logo path: {e}")
            logo_pixmap = QPixmap(128, 128)
            logo_pixmap.fill(Qt.transparent)

        logo_label.setPixmap(logo_pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignTop | Qt.AlignRight)

        grid_layout = QGridLayout(self)
        grid_layout.setContentsMargins(10, 10, 10, 10)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        grid_layout.addLayout(self.content_layout, 0, 0)
        grid_layout.addWidget(logo_label, 0, 1, Qt.AlignTop | Qt.AlignRight)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 0)

# ====================================================================================
# MAIN APPLICATION WINDOW
# ====================================================================================
class RadioAstronomyApp(QMainWindow):
    def __init__(self, group_code, group_folder):
        super().__init__()
        self.group_code, self.group_folder = group_code, group_folder
        self.detected_noise_freq = None
        self.thread, self.worker = None, None
        self.is_sdr_busy = False
        self.plot_data_dict = {}
        self.final_data_to_save = None
        self.point1, self.point2 = None, None

        self.brightness_tab_created = False
        self.brightness_tab = None
        self._last_freq_axis = None
        self.last_Ts_final = None
        self._ts_interp_func = None

        self.setWindowTitle("Radio Astronomy Suite")
        self._create_fonts()
        self._apply_stylesheet()
        self._create_widgets()
        self._create_status_bar()
        self._add_floating_close_button()
        self._add_copyright_label()
        self.status_bar.showMessage("Application started.")
        QTimer.singleShot(0, lambda: self.setWindowState(Qt.WindowMaximized | Qt.WindowActive))
        self._max_try = 0
        for delay in (0, 120, 350):
            QTimer.singleShot(delay, self._ensure_maximized)

    def _ensure_maximized(self):
        self.setWindowState(self.windowState() | Qt.WindowMaximized | Qt.WindowActive)

        if not self.isMaximized():
            w = self.windowHandle()
            if w is not None:
                w.setWindowState(Qt.WindowMaximized)

        if not self.isMaximized():
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen:
                self.setGeometry(screen.availableGeometry())

        self._max_try = getattr(self, "_max_try", 0) + 1
        if self.isMaximized() or self._max_try >= 3:
            return

    def _create_fonts(self):
        self.title_font = QFont("DejaVu Sans", 16, QFont.Weight.Bold)
        self.heading_font = QFont("DejaVu Sans", 14, QFont.Weight.Bold)
        self.label_font = QFont("DejaVu Sans", 12, QFont.Weight.Normal)
        self.input_font = QFont("DejaVu Sans", 14, QFont.Weight.Normal)
        self.bold_label_font = QFont("DejaVu Sans", 12, QFont.Weight.Bold)
        self.display_font = QFont("DejaVu Sans", 30, QFont.Weight.Bold)

    def _apply_stylesheet(self):
        stylesheet = """
            QPushButton {
                background-color: #5A5A5A; color: #FFFFFF; border: 1px solid #6A6A6A;
                padding: 8px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #6A6A6A; }
            QPushButton:disabled { background-color: #4A4A4A; color: #888888; }
            QPushButton#AccentButton {
                background-color: #007BFF; color: #FFFFFF; font-weight: bold;
                border: none; padding: 10px;
            }
            QPushButton#AccentButton:hover { background-color: #0056b3; }
        """
        self.setStyleSheet(stylesheet)

    def _create_widgets(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        self.quick_look_tab_widget = LogoTabPageWidget()
        self.pointing_tab_widget = LogoTabPageWidget()
        self.acquisition_tab_widget = LogoTabPageWidget()
        self.noise_temp_cal_tab_widget = LogoTabPageWidget()
        self.line_temp_tab_widget = LogoTabPageWidget()
        self.messages_tab_widget = LogoTabPageWidget()

        self.quick_look_tab = QWidget()
        self.pointing_tab = QWidget()
        self.acquisition_tab = QWidget()
        self.noise_temp_cal_tab = QWidget()
        self.line_temp_tab = QWidget()
        self.messages_tab = QWidget()

        self.quick_look_tab_widget.content_layout.addWidget(self.quick_look_tab)
        self.pointing_tab_widget.content_layout.addWidget(self.pointing_tab)
        self.acquisition_tab_widget.content_layout.addWidget(self.acquisition_tab)
        self.noise_temp_cal_tab_widget.content_layout.addWidget(self.noise_temp_cal_tab)
        self.line_temp_tab_widget.content_layout.addWidget(self.line_temp_tab)
        self.messages_tab_widget.content_layout.addWidget(self.messages_tab)

        self.tab_widget.addTab(self.quick_look_tab_widget, "Quick Look")
        self.tab_widget.addTab(self.pointing_tab_widget, "Pointing && Velocity Corrections")
        self.tab_widget.addTab(self.acquisition_tab_widget, "Data Acquisition")
        self.tab_widget.addTab(self.noise_temp_cal_tab_widget, "Calibration")
        self.tab_widget.addTab(self.line_temp_tab_widget, "HI Line Analysis")
        self.tab_widget.addTab(self.messages_tab_widget, "Messages Log")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self._setup_quick_look_tab(self.quick_look_tab)
        self._setup_pointing_tab(self.pointing_tab)
        self._setup_acquisition_tab(self.acquisition_tab)
        self._setup_noise_temp_calibration_tab(self.noise_temp_cal_tab)
        self._setup_line_temp_profile_tab(self.line_temp_tab)
        self._setup_messages_tab(self.messages_tab)

    def _create_status_bar(self):
        self.status_bar = self.statusBar()

    def _create_plot_widget(self, parent):
        plot_widget = QWidget(parent)
        plot_layout = QVBoxLayout(plot_widget)
        figure = Figure(figsize=(10, 7), dpi=100)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, plot_widget)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)
        return plot_widget, figure, canvas, toolbar

    def _setup_plot_formatting(self, ax, title="", xlabel="", ylabel=""):
        ax.clear()
        ax.set_title(title, fontdict={'size': 16, 'weight': 'bold'})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.figure.tight_layout()
        ax.figure.canvas.draw()

    def _setup_noise_temp_calibration_tab(self, tab_widget):
        main_layout = QHBoxLayout(tab_widget)

        left_container = QGroupBox("1. Load Calibration Data")
        left_layout = QVBoxLayout(left_container)
        plot_widget_left, self.ntc_left_fig, self.ntc_left_canvas, _ = self._create_plot_widget(left_container)
        self.ntc_left_ax = self.ntc_left_fig.add_subplot(111)
        self._setup_plot_formatting(self.ntc_left_ax, "Loaded Data", "Frequency (MHz)", "Relative Power (dB)")

        left_button_layout = QHBoxLayout()
        ground_button = QPushButton("Load Ground Data")
        sky_button = QPushButton("Load Sky Data")
        clear_button = QPushButton("Clear Plot")
        left_button_layout.addWidget(ground_button)
        left_button_layout.addWidget(sky_button)
        left_button_layout.addWidget(clear_button)
        left_layout.addWidget(plot_widget_left)
        left_layout.addLayout(left_button_layout)
        ground_button.clicked.connect(lambda: self._open_file_and_plot_ntc('Ground'))
        sky_button.clicked.connect(lambda: self._open_file_and_plot_ntc('Sky'))
        clear_button.clicked.connect(lambda: self._clr_plot(self.ntc_left_ax, "Loaded Data", "Frequency (MHz)", "Relative Power (dB)"))

        right_container = QGroupBox("2. Calculate Receiver Temperature (Tr)")
        right_layout = QVBoxLayout(right_container)
        plot_widget_right, self.ntc_right_fig, self.ntc_right_canvas, _ = self._create_plot_widget(right_container)
        self.ntc_right_ax = self.ntc_right_fig.add_subplot(111)
        self._setup_plot_formatting(self.ntc_right_ax, "", "Frequency (MHz)", "Temperature (K)")
        calc_tr_button = QPushButton("Calculate Receiver Temperature")
        calc_tr_button.setObjectName("AccentButton")
        right_layout.addWidget(plot_widget_right)
        right_layout.addWidget(calc_tr_button)
        calc_tr_button.clicked.connect(self._calculate_receiver_temp_ntc)
        self.ntc_right_canvas.mpl_connect('button_press_event', self._on_tr_click_ntc)

        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(right_container, 1)

    def _open_file_and_plot_ntc(self, file_type):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Open {file_type} Data File", self.group_folder, "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            x_data, y_data = df.iloc[:, 0].values, df.iloc[:, 1].values
            file_name = os.path.basename(file_path)
            y_data[y_data <= 0] = 1e-12
            self.plot_data_dict[file_type] = {'x_data': x_data, 'y_data': y_data, 'file_name': file_name}
            y_data_db = 10 * np.log10(y_data)
            self.ntc_left_ax.plot(x_data, y_data_db, label=f"{file_name} ({file_type})")
            self.ntc_left_ax.legend()
            self.ntc_left_canvas.draw()
            self._add_message(f"Loaded and plotted {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Could not read or plot file:\n{e}")

    def _calculate_receiver_temp_ntc(self):
        self.point1, self.point2 = None, None
        if 'Ground' not in self.plot_data_dict or 'Sky' not in self.plot_data_dict:
            QMessageBox.warning(self, "Missing Data", "Both Ground and Sky data must be plotted first.")
            return

        self._setup_plot_formatting(self.ntc_right_ax, "", "Frequency (MHz)", "Temperature (K)")

        p_ground = self.plot_data_dict['Ground']['y_data'].copy()
        p_sky = self.plot_data_dict['Sky']['y_data'].copy()
        x_data = self.plot_data_dict['Ground']['x_data']

        p_sky[p_sky <= 0] = 1e-12
        p_ground[p_ground <= 0] = 1e-12

        spike_start = 1419.975
        spike_end = 1419.990
        spike_mask = (x_data >= spike_start) & (x_data <= spike_end)

        def interpolate_spike(y_data, mask):
            if not np.any(mask):
                return y_data
            y_clean = y_data.copy()
            valid_indices = ~mask
            if np.sum(valid_indices) < 2:
                return y_data

            interp_func = interp1d(
                x_data[valid_indices], y_data[valid_indices],
                kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            y_clean[mask] = interp_func(x_data[mask])
            return y_clean

        p_ground = interpolate_spike(p_ground, spike_mask)
        p_sky = interpolate_spike(p_sky, spike_mask)

        safe_indices = p_sky > 0
        y_factor = np.full_like(p_ground, np.nan)
        y_factor[safe_indices] = p_ground[safe_indices] / p_sky[safe_indices]
        Tr_original = (DEFAULT_GROUND_TEMP - y_factor * DEFAULT_SKY_TEMP) / (y_factor - 1)

        self.ntc_right_ax.plot(x_data, Tr_original, label='Tr (Uncorrected)')
        self.ntc_right_ax.legend()

        try:
            freq_mask = (x_data >= 1420.20) & (x_data <= 1420.60)
            if np.any(freq_mask):
                tr_in_window = Tr_original[freq_mask]
                if np.all(np.isnan(tr_in_window)):
                    self._add_message("Info: Data in the priority window is all NaN. Skipping annotation.")
                else:
                    local_peak_index = np.nanargmax(tr_in_window)
                    original_indices = np.where(freq_mask)[0]
                    final_peak_index = original_indices[local_peak_index]
                    peak_freq, peak_temp = x_data[final_peak_index], Tr_original[final_peak_index]
                    self._add_message(f"Annotating highest peak found in priority window at {peak_freq:.2f} MHz.")
                    freq_midpoint = (x_data[0] + x_data[-1]) / 2
                    offset = (-200, -15) if peak_freq > freq_midpoint else (60, -15)
                    self.ntc_right_ax.annotate(
                        'Possible Contaminating HI line',
                        xy=(peak_freq, peak_temp),
                        xytext=offset,
                        textcoords='offset points',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                        fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8)
                    )
            else:
                self._add_message("Info: No data points found in the priority window [1420.20, 1420.60] MHz.")
        except Exception as e:
            self._add_message(f"Info: Could not auto-annotate peak due to an error: {e}")

        try:
            min_temp, max_temp = np.nanmin(Tr_original), np.nanmax(Tr_original)
            text_y_position = min_temp + 0.25 * (max_temp - min_temp)
            trans = blended_transform_factory(self.ntc_right_ax.transAxes, self.ntc_right_ax.transData)
            self.ntc_right_ax.text(
                0.5, text_y_position,
                "Click on two points on either side of the contaminating line.",
                ha='center', fontsize=11, color='navy',
                bbox=dict(boxstyle='square,pad=0.4', facecolor='lightblue', alpha=0.8),
                transform=trans
            )
        except Exception as e:
            self._add_message(f"Info: Could not place instructional text: {e}")

        self.ntc_right_canvas.draw()

        self.plot_data_dict['Tr_original'] = {'x_data': x_data, 'y_data': Tr_original}
        self.plot_data_dict['Tr'] = {'x_data': x_data, 'y_data': Tr_original.copy()}

    def _on_tr_click_ntc(self, event):
        if event.inaxes != self.ntc_right_ax or self.plot_data_dict.get('Tr_original') is None:
            return

        x_val, y_val = event.xdata, event.ydata
        if x_val is None or y_val is None:
            return

        x_data = self.plot_data_dict['Tr_original']['x_data']
        Tr_original = self.plot_data_dict['Tr_original']['y_data']

        if self.point1 is None:
            self.point1 = (x_val, y_val)
            self._setup_plot_formatting(self.ntc_right_ax, "", "Frequency (MHz)", "Temperature (K)")
            self.ntc_right_ax.plot(x_data, Tr_original, label='Original Tr')
            self.ntc_right_ax.plot(x_val, y_val, 'ro', markersize=10)
            self.ntc_right_ax.legend()
            self.ntc_right_canvas.draw()
            self._add_message("Point 1 selected. Select a second point.")
        elif self.point2 is None:
            self.point2 = (x_val, y_val)
            x_min_masked, x_max_masked = min(self.point1[0], self.point2[0]), max(self.point1[0], self.point2[0])
            masked_indices = (x_data >= x_min_masked) & (x_data <= x_max_masked)
            unmasked_indices = ~masked_indices & np.isfinite(Tr_original)
            x_fit_data, y_fit_data = x_data[unmasked_indices], Tr_original[unmasked_indices]
            poly_degree = 2
            if len(x_fit_data) <= poly_degree:
                QMessageBox.critical(self, "Fit Error", "Not enough data points outside the selected region to perform fit.")
                self.point1, self.point2 = None, None
                return
            try:
                x_mean, x_std = x_fit_data.mean(), x_fit_data.std()
                x_fit_normalized = (x_fit_data - x_mean) / x_std if x_std > 0 else x_fit_data
                coeffs = np.polyfit(x_fit_normalized, y_fit_data, poly_degree)
                p = np.poly1d(coeffs)
                x_data_normalized = (x_data - x_mean) / x_std if x_std > 0 else x_data
                Tr_best_fit_baseline = p(x_data_normalized)
            except Exception as e:
                QMessageBox.critical(self, "Fit Error", f"Could not perform the polynomial fit.\nError: {e}")
                self.point1, self.point2 = None, None
                return

            self.plot_data_dict['Tr']['y_data'] = Tr_best_fit_baseline

            self._setup_plot_formatting(self.ntc_right_ax, "", "Frequency (MHz)", "Temperature (K)")
            y_unmasked = np.copy(Tr_original)
            y_unmasked[masked_indices] = np.nan
            y_masked = np.copy(Tr_original)
            y_masked[unmasked_indices] = np.nan

            self.ntc_right_ax.axvspan(x_min_masked, x_max_masked, color='darkgrey', alpha=0.3, label='Excluded Region')
            self.ntc_right_ax.plot(x_data, y_unmasked, color='blue', label='Receiver Temperature')
            self.ntc_right_ax.plot(x_data, y_masked, color='lightgrey', label='Excluded Spectrum')
            self.ntc_right_ax.plot(x_data, Tr_best_fit_baseline, 'g--', linewidth=2, label=f'Poly Fit (deg={poly_degree})')
            self.ntc_right_ax.plot(
                [self.point1[0], self.point2[0]],
                [self.point1[1], self.point2[1]],
                'ro', markersize=10,
                label='Fit Boundary Points'
            )
            self.ntc_right_ax.text(
                0.98, 0.98,
                "Calibration Complete - Proceed to next tab",
                transform=self.ntc_right_ax.transAxes,
                fontsize=12, fontweight='bold', color='darkgreen',
                ha='right', va='top',
                bbox=dict(boxstyle='square,pad=0.5', facecolor='lightgreen', alpha=0.8)
            )
            self.ntc_right_ax.legend(loc='upper left')
            self.ntc_right_canvas.draw()
            self._add_message("Calibration Complete. Plot saved. Proceed to the next tab.")
            self.save_baseline_fit_plot()

            try:
                p_ground = self.plot_data_dict['Ground']['y_data']
                SMOOTHING_WINDOW_SIZE = 17
                if SMOOTHING_WINDOW_SIZE % 2 == 0:
                    SMOOTHING_WINDOW_SIZE += 1
                p_ground_smoothed = pd.Series(p_ground).rolling(
                    window=SMOOTHING_WINDOW_SIZE, center=True
                ).mean().values
                p_ground_smoothed = pd.Series(p_ground_smoothed).fillna(method='bfill').fillna(method='ffill').values
                p_ground_smoothed[p_ground_smoothed <= 0] = 1e-12

                Tr_corrected = self.plot_data_dict['Tr']['y_data']
                G = p_ground_smoothed / (DEFAULT_GROUND_TEMP + Tr_corrected)
                G[G <= 0] = 1e-12
                self.plot_data_dict['Gain'] = {'x_data': x_data, 'y_data': G}
                self.plot_data_dict['SmoothedGround'] = {'x_data': x_data, 'y_data': p_ground_smoothed}

                gain_csv_path = os.path.join(self.group_folder, f"gain_vs_freq_{self.group_code}.csv")
                pd.DataFrame({'Frequency (MHz)': x_data, 'Gain': G}).to_csv(gain_csv_path, index=False)
                self._add_message(f"Gain saved: {os.path.basename(gain_csv_path)}")

                gain_png_path = os.path.join(self.group_folder, f"gain_plot_{self.group_code}.png")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x_data, G, color='blue', linewidth=1.5)
                ax.set_title("Gain vs Frequency")
                ax.set_xlabel("Frequency (MHz)")
                ax.set_ylabel("Gain")
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(gain_png_path, dpi=150)
                plt.close(fig)
                self._add_message(f"Gain plot saved: {os.path.basename(gain_png_path)}")
            except Exception as e:
                self._add_message(f"ERROR: Failed to compute/save Gain: {e}")

            self.point1, self.point2 = None, None

    def save_baseline_fit_plot(self):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"baseline_fit_{self.group_code}_{timestamp}.png"
            file_path = os.path.join(self.group_folder, filename)
            self.ntc_right_fig.savefig(file_path, dpi=150, bbox_inches='tight')
            self._add_message(f"SUCCESS: Baseline fit plot saved to {filename}")
        except Exception as e:
            error_message = f"Failed to save baseline fit screenshot: {e}"
            self._add_message(f"ERROR: {error_message}")
            QMessageBox.critical(self, "Save Error", error_message)

    def _setup_quick_look_tab(self, tab_widget):
        layout = QHBoxLayout(tab_widget)
        plot_widget, self.ql_fig, self.ql_canvas, _ = self._create_plot_widget(tab_widget)
        self.ql_ax = self.ql_fig.add_subplot(111)
        self._setup_plot_formatting(self.ql_ax, "Quick Look", "Frequency (MHz)", "Relative Power (dB)")

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        self.ql_start_button = QPushButton("Start Visualization")
        self.ql_start_button.setObjectName("AccentButton")
        self.ql_stop_button = QPushButton("Stop Visualization")
        self.ql_stop_button.setEnabled(False)
        controls_layout.addWidget(self.ql_start_button)
        controls_layout.addWidget(self.ql_stop_button)
        controls_layout.addStretch()
        layout.addWidget(plot_widget, 3)
        layout.addWidget(controls_group, 1)
        self.ql_start_button.clicked.connect(self.start_quick_look)
        self.ql_stop_button.clicked.connect(self.stop_sdr_task)

    def _setup_pointing_tab(self, tab_widget):
        main_layout = QGridLayout(tab_widget)
        main_layout.setSpacing(20)

        output_group = QGroupBox("Live Pointing Data")
        output_group.setFont(self.heading_font)
        output_layout = QHBoxLayout(output_group)

        az_group = QGroupBox("Azimuth")
        az_group.setFont(self.heading_font)
        self.azimuth_label = QLabel("N/A")
        self.azimuth_label.setFont(self.display_font)
        self.azimuth_label.setStyleSheet(f"color: {COLOR_BLUE};")
        self.azimuth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        az_box = QVBoxLayout()
        az_box.addWidget(self.azimuth_label)
        az_group.setLayout(az_box)

        alt_group = QGroupBox("Altitude")
        alt_group.setFont(self.heading_font)
        self.altitude_label = QLabel("N/A")
        self.altitude_label.setFont(self.display_font)
        self.altitude_label.setStyleSheet(f"color: {COLOR_BLUE};")
        self.altitude_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        alt_box = QVBoxLayout()
        alt_box.addWidget(self.altitude_label)
        alt_group.setLayout(alt_box)

        lsr_group = QGroupBox("Velocity Correction")
        lsr_group.setFont(self.heading_font)
        self.lsr_label = QLabel("N/A")
        self.lsr_label.setFont(self.display_font)
        self.lsr_label.setStyleSheet(f"color: {COLOR_GREEN};")
        self.lsr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lsr_box = QVBoxLayout()
        lsr_box.addWidget(self.lsr_label)
        lsr_box.addWidget(QLabel("km/s"))
        lsr_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lsr_group.setLayout(lsr_box)

        output_layout.addWidget(az_group)
        output_layout.addWidget(alt_group)
        output_layout.addWidget(lsr_group)

        controls_group = QGroupBox("Calculation Controls")
        controls_group.setFont(self.heading_font)
        controls_layout = QFormLayout(controls_group)
        controls_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        controls_layout.setVerticalSpacing(15)

        loc_label = QLabel(
            f"Observer Location (Mumbai - Fixed): Latitude: {MUMBAI_LATITUDE.value:.4f}, "
            f"Longitude: {MUMBAI_LONGITUDE.value:.4f}"
        )
        loc_label.setFont(self.bold_label_font)

        self.date_edit = QDateEdit(self)
        self.date_edit.setFont(self.input_font)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setCalendarPopup(True)

        self.time_edit = QTimeEdit(self)
        self.time_edit.setFont(self.input_font)
        self.time_edit.setDisplayFormat("HH:mm:ss")

        self.override_dt_checkbox = QCheckBox("Set Time Manually")
        self.override_dt_checkbox.toggled.connect(self._toggle_datetime_override)

        datetime_input_layout = QHBoxLayout()
        datetime_input_layout.addWidget(self.date_edit)
        datetime_input_layout.addWidget(self.time_edit)
        datetime_input_layout.addSpacing(20)
        datetime_input_layout.addWidget(self.override_dt_checkbox)

        self.l_spinbox = QDoubleSpinBox()
        self.l_spinbox.setFont(self.input_font)
        self.l_spinbox.setRange(0, 360)
        self.l_spinbox.setDecimals(1)
        self.l_spinbox.setSingleStep(0.5)
        self.l_spinbox.setSuffix(" °")

        self.b_spinbox = QDoubleSpinBox()
        self.b_spinbox.setFont(self.input_font)
        self.b_spinbox.setRange(-90, 90)
        self.b_spinbox.setDecimals(1)
        self.b_spinbox.setSingleStep(0.5)
        self.b_spinbox.setSuffix(" °")

        self.calc_button = QPushButton("Calculate Pointing and Velocity Correction")
        self.calc_button.setObjectName("AccentButton")
        self.calc_button.setFont(self.heading_font)

        time_label = QLabel("Observation Time (IST):")
        l_label = QLabel("Galactic Longitude (l):")
        b_label = QLabel("Galactic Latitude (b):")
        time_label.setFont(self.bold_label_font)
        l_label.setFont(self.bold_label_font)
        b_label.setFont(self.bold_label_font)

        controls_layout.addRow(loc_label)
        controls_layout.addRow(time_label, datetime_input_layout)
        controls_layout.addRow(l_label, self.l_spinbox)
        controls_layout.addRow(b_label, self.b_spinbox)
        controls_layout.addRow("", self.calc_button)

        main_layout.addWidget(output_group, 0, 0)
        main_layout.addWidget(controls_group, 1, 0)
        main_layout.setRowStretch(2, 1)

        self.calc_button.clicked.connect(self._calculate_pointing_and_lsr)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_datetime_display)
        self._toggle_datetime_override(False)

    def _setup_acquisition_tab(self, tab_widget):
        layout = QHBoxLayout(tab_widget)

        controls_group = QGroupBox("Parameters & Controls")
        controls_layout = QVBoxLayout(controls_group)
        form_layout = QFormLayout()

        self.acq_center_freq_entry = QLineEdit(str(DEFAULT_CENTER_FREQ))
        self.acq_center_freq_entry.setReadOnly(True)
        self.acq_sample_rate_entry = QLineEdit(str(DEFAULT_SAMPLE_RATE))
        self.acq_sample_rate_entry.setReadOnly(True)
        self.acq_nf_entry = QLineEdit(str(DEFAULT_NF))
        self.acq_nf_entry.setReadOnly(True)
        self.acq_gain_entry = QLineEdit(str(DEFAULT_GAIN))
        self.acq_gain_entry.setReadOnly(True)
        self.acq_int_time_entry = QLineEdit(str(DEFAULT_INT_TIME))

        form_layout.addRow("Center Freq (MHz):", self.acq_center_freq_entry)
        form_layout.addRow("Sample Rate (MHz):", self.acq_sample_rate_entry)
        form_layout.addRow("Number of Bins:", self.acq_nf_entry)
        form_layout.addRow("Gain:", self.acq_gain_entry)
        form_layout.addRow("Integration Time (s):", self.acq_int_time_entry)

        controls_layout.addLayout(form_layout)

        self.record_button = QPushButton("Record Data")
        self.record_button.setObjectName("AccentButton")
        self.record_button.clicked.connect(self.start_recording)

        self.stop_acq_button = QPushButton("Stop Recording")
        self.stop_acq_button.clicked.connect(self.stop_sdr_task)
        self.stop_acq_button.setEnabled(False)

        self.clear_acq_button = QPushButton("Clear Plot")
        self.clear_acq_button.clicked.connect(
            lambda: self._clr_plot(self.acq_ax, "Data Acquisition", "Frequency (MHz)", "Power (dBm)")
        )

        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.stop_acq_button)
        controls_layout.addWidget(self.clear_acq_button)
        controls_layout.addStretch()

        layout.addWidget(controls_group)

        plot_widget, self.acq_fig, self.acq_canvas, self.acq_toolbar = self._create_plot_widget(tab_widget)
        self.acq_ax = self.acq_fig.add_subplot(111)
        self._setup_plot_formatting(self.acq_ax, title="Data Acquisition", xlabel="Frequency (MHz)", ylabel="Power (dBm)")
        layout.addWidget(plot_widget, stretch=3)

    def _setup_line_temp_profile_tab(self, tab_widget):
        layout = QVBoxLayout(tab_widget)
        plot_widget, self.lt_fig, self.lt_canvas, _ = self._create_plot_widget(tab_widget)
        self.lt_ax = self.lt_fig.add_subplot(111)
        self._setup_plot_formatting(self.lt_ax, "Source Data for Processing", "Frequency (MHz)", "Power (dB)")

        def open_source_for_lt():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Source Data File", self.group_folder, "CSV Files (*.csv)"
            )
            if not file_path:
                return
            try:
                df = pd.read_csv(file_path)
                x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
                y[y <= 0] = 1e-12
                self.plot_data_dict['Source'] = {
                    'x_data': x,
                    'y_data': y,
                    'file_name': os.path.basename(file_path)
                }
                y_db = 10 * np.log10(y)
                self._setup_plot_formatting(
                    self.lt_ax,
                    f"Loaded Source Data: {os.path.basename(file_path)}",
                    "Frequency (MHz)",
                    "Power (dB)"
                )
                self.lt_ax.plot(x, y_db, label=os.path.basename(file_path))
                self.lt_ax.legend()
                self.lt_canvas.draw()
                self._add_message(f"Loaded {os.path.basename(file_path)} for analysis.")

                try:
                    window_size = 31
                    if len(y_db) < window_size:
                        self._add_message("Data too short for ripple analysis.")
                        return
                    ripple = y_db - medfilt(y_db, kernel_size=window_size)
                    N = len(ripple)
                    yf = np.abs(np.fft.fft(ripple))
                    dfreq = x[1] - x[0]
                    xf = np.fft.fftfreq(N, d=dfreq)

                    fig_ripple, ax_ripple = plt.subplots(figsize=(12, 7))
                    ax_ripple.plot(xf[:N//2], (2.0/N * yf[:N//2]), color='purple')
                    ax_ripple.set_title(
                        f"FFT of Signal Ripple/Noise\n(from {os.path.basename(file_path)})"
                    )
                    ax_ripple.set_xlabel("Ripple Frequency (cycles / MHz)")
                    ax_ripple.set_ylabel("Amplitude")
                    ax_ripple.grid(True, linestyle='--')

                    ripple_peaks, _ = find_peaks(
                        yf[:N//2],
                        height=np.mean(yf[1:N//2]) + np.std(yf[1:N//2]),
                        distance=5
                    )
                    for peak_idx in ripple_peaks:
                        if peak_idx == 0:
                            continue
                        peak_freq, peak_amp = xf[peak_idx], (2.0/N * yf[peak_idx])
                        ax_ripple.annotate(
                            f"{peak_freq:.2f}",
                            xy=(peak_freq, peak_amp),
                            xytext=(peak_freq, peak_amp + 0.05 * np.max(2.0/N * yf[1:N//2])),
                            ha='center',
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4)
                        )

                    base_filename = os.path.splitext(os.path.basename(file_path))[0]
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    screenshot_path = os.path.join(
                        self.group_folder,
                        f"ripple_FFT_{base_filename}_{timestamp}.png"
                    )
                    fig_ripple.tight_layout()
                    fig_ripple.savefig(screenshot_path, dpi=150)
                    plt.close(fig_ripple)
                    self._add_message(
                        f"SUCCESS: Ripple FFT analysis saved to {os.path.basename(screenshot_path)}"
                    )
                except Exception as e:
                    self._add_message(f"ERROR: Could not perform ripple FFT analysis: {e}")
            except Exception as e:
                QMessageBox.critical(self, "File Error", f"Could not process source file:\n{e}")

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        load_button = QPushButton("Load Source Data")
        calc_button = QPushButton("Calculate HI Line Temperature")
        calc_button.setObjectName("AccentButton")
        button_layout.addWidget(load_button)
        button_layout.addWidget(calc_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        layout.addWidget(plot_widget)

        load_button.clicked.connect(open_source_for_lt)
        calc_button.clicked.connect(self._calculate_brightness_temperature)

    def _setup_messages_tab(self, tab_widget):
        layout = QVBoxLayout(tab_widget)
        self.messages_textbox = QTextEdit()
        self.messages_textbox.setReadOnly(True)
        layout.addWidget(self.messages_textbox)

    def _save_message_log(self):
        try:
            log_content = self.messages_textbox.toPlainText()
            if not log_content.strip():
                return

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"message_log_{self.group_code}_{timestamp}.txt"
            file_path = os.path.join(self.group_folder, filename)

            with open(file_path, 'w') as f:
                f.write(log_content)

            self.messages_textbox.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Message log saved to {filename}"
            )
        except Exception as e:
            error_message = f"Failed to save message log: {e}"
            self._add_message(f"ERROR: {error_message}")

    def _clr_plot(self, ax_to_clear, title, xlabel, ylabel):
        ax_to_clear.clear()
        self._setup_plot_formatting(ax_to_clear, title, xlabel, ylabel)
        ax_to_clear.figure.canvas.draw()

    def _add_message(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_message = f"{timestamp} {message}"
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(full_message)
        if hasattr(self, 'messages_textbox'):
            self.messages_textbox.append(full_message)

    def _toggle_datetime_override(self, checked):
        if checked:
            self.date_edit.setReadOnly(False)
            self.time_edit.setReadOnly(False)
            self.clock_timer.stop()
        else:
            self.date_edit.setReadOnly(True)
            self.time_edit.setReadOnly(True)
            self._update_datetime_display()
            if not self.clock_timer.isActive():
                self.clock_timer.start(1000)

    def _update_datetime_display(self):
        current_dt = datetime.now()
        self.date_edit.setDate(QDate(current_dt.date()))
        self.time_edit.setTime(QTime(current_dt.time()))

    def _on_tab_changed(self, index):
        current_tab_widget = self.tab_widget.widget(index)
        if current_tab_widget == self.acquisition_tab_widget:
            if self.is_sdr_busy and self.worker is not None:
                if self.worker.mode == 'quick_look':
                    self._add_message(
                        "Switched to Data Acquisition tab. "
                        "Stopping Quick Look to allow recording..."
                    )
                    self.stop_sdr_task()

    def _calculate_pointing_and_lsr(self):
        try:
            date_val = self.date_edit.date().toPython()
            time_val = self.time_edit.time().toPython()
            local_dt_obj = datetime.combine(date_val, time_val)
            gal_l_val, gal_b_val = self.l_spinbox.value(), self.b_spinbox.value()
            utc_dt_obj = local_dt_obj - timedelta(hours=IST_OFFSET_HOURS)
            obs_time = Time(utc_dt_obj, format='datetime', scale='utc')

            source_galactic_coords = Galactic(l=gal_l_val * u.deg, b=gal_b_val * u.deg)
            altaz_frame = AltAz(obstime=obs_time, location=mumbai_location)
            source_altaz_coords = source_galactic_coords.transform_to(altaz_frame)
            azimuth, altitude = source_altaz_coords.az.to_value(u.deg), source_altaz_coords.alt.to_value(u.deg)

            self.azimuth_label.setText(f"{azimuth:.1f}°")
            self.altitude_label.setText(f"{altitude:.1f}°")
            self._add_message(f"Pointing calculated: Az={azimuth:.1f}°, Alt={altitude:.1f}°")

            jd = pyasl.jdcnv(local_dt_obj)
            source_icrs_coords = source_galactic_coords.transform_to(ICRS())
            ra_2000, dec_2000 = source_icrs_coords.ra.deg, source_icrs_coords.dec.deg
            corr, _ = pyasl.helcorr(
                MUMBAI_LONGITUDE.value,
                MUMBAI_LATITUDE.value,
                MUMBAI_ALTITUDE.value,
                ra_2000,
                dec_2000,
                jd
            )

            v_sun = V_SUN_LSR_SPEED
            sun_ra, sun_dec = math.radians(SUN_APEX_RA_DEG), math.radians(SUN_APEX_DEC_DEG)
            obs_ra, obs_dec = math.radians(ra_2000), math.radians(dec_2000)

            a = math.cos(sun_dec) * math.cos(obs_dec)
            b = (math.cos(sun_ra) * math.cos(obs_ra)) + (math.sin(sun_ra) * math.sin(obs_ra))
            c = math.sin(sun_dec) * math.sin(obs_dec)
            v_rs = v_sun * ((a * b) + c)

            v_lsr_total_correction = -1 * (corr + v_rs)
            self.lsr_label.setText(f"{v_lsr_total_correction:.3f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during calculation:\n{e}")
            self._add_message(f"Calculation Error: {e}")

    def _calculate_brightness_temperature(self):
        required_keys = ['Source', 'Tr', 'Gain']
        if not all(key in self.plot_data_dict for key in required_keys):
            QMessageBox.warning(
                self,
                "Missing Data",
                "Source, corrected Tr, and Gain data must all be available."
            )
            return

        try:
            P_source = self.plot_data_dict['Source']['y_data'].copy()
            Tr = self.plot_data_dict['Tr']['y_data']
            G = self.plot_data_dict['Gain']['y_data'].copy()
            freq_axis = self.plot_data_dict['Gain']['x_data']
            self._last_freq_axis = freq_axis

            P_source[P_source <= 0] = 1e-12
            G[G <= 0] = 1e-12
        except (KeyError, IndexError, AttributeError) as e:
            self._add_message(
                f"ERROR: Could not prepare data for calculation. "
                f"Malformed data structure? Details: {e}"
            )
            QMessageBox.critical(
                self,
                "Data Error",
                f"Failed to read necessary data arrays. Error: {e}"
            )
            return

        try:
            spike_start = 1419.975
            spike_end = 1419.990
            spike_mask = (freq_axis >= spike_start) & (freq_axis <= spike_end)

            if np.any(spike_mask):
                valid_indices = ~spike_mask
                if np.sum(valid_indices) > 1:
                    interp_func = interp1d(
                        freq_axis[valid_indices],
                        P_source[valid_indices],
                        kind='linear',
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    P_source[spike_mask] = interp_func(freq_axis[spike_mask])
                    self._add_message(
                        f"Removed SDR spike from {spike_start:.3f} to {spike_end:.3f} MHz using interpolation."
                    )
                else:
                    self._add_message("WARNING: Not enough valid points to interpolate SDR spike.")
        except Exception as e:
            self._add_message(f"WARNING: Failed to remove SDR spike due to error: {e}")

        try:
            self._add_message("Attempting automatic ripple filtering...")
            y_db = 10 * np.log10(P_source)
            ripple = y_db - medfilt(y_db, kernel_size=5)

            N = len(ripple)
            df = freq_axis[1] - freq_axis[0]
            xf = np.fft.fftfreq(N, d=df)
            yf = np.abs(np.fft.fft(ripple))

            positive_freq_mask = (xf > 0)
            peak_threshold = np.mean(yf[positive_freq_mask]) + 3 * np.std(yf[positive_freq_mask])
            peak_indices, _ = find_peaks(
                yf[:N//2], height=peak_threshold, distance=10
            )

            if len(peak_indices) > 0:
                self._add_message(
                    f"Detected {len(peak_indices)} significant ripple components. "
                    f"Applying notch filters..."
                )
                for peak_idx in peak_indices:
                    if peak_idx == 0:
                        continue
                    target_freq = xf[peak_idx]
                    nyquist = (1/df) / 2.0
                    w0 = abs(target_freq) / nyquist

                    if w0 >= 1.0 or w0 <= 0:
                        self._add_message(
                            f"  - Skipping invalid normalized frequency {w0:.3f} for notch filter."
                        )
                        continue

                    self._add_message(
                        f"  - Removing ripple at frequency {target_freq:.2f} cycles/MHz."
                    )
                    b, a = iirnotch(w0, Q=30.0)
                    P_source = filtfilt(b, a, P_source)
                self._add_message("Ripple filtering complete.")
            else:
                self._add_message("No significant ripple frequencies detected.")
        except Exception as e:
            self._add_message(f"WARNING: Could not perform auto ripple filtering. Error: {e}")

        Ts_raw = (P_source / G) - Tr - DEFAULT_SKY_TEMP
        Ts_raw = np.nan_to_num(Ts_raw, nan=0.0, posinf=0.0, neginf=0.0)

        hi_peak_center = 1420.40575
        mask_fit = (freq_axis < (hi_peak_center - 0.5)) | (freq_axis > (hi_peak_center + 0.5))

        if np.sum(mask_fit) < 2:
            self._add_message("WARNING: Not enough data points to perform baseline fit. Skipping.")
            Ts_baseline_subtracted = Ts_raw
        else:
            poly_coeffs = np.polyfit(freq_axis[mask_fit], Ts_raw[mask_fit], deg=1)
            baseline_fit = np.polyval(poly_coeffs, freq_axis)
            Ts_baseline_subtracted = Ts_raw - baseline_fit

        n = min(len(Ts_raw) // 10, 100)
        edge_vals = np.concatenate(
            [Ts_baseline_subtracted[:n], Ts_baseline_subtracted[-n:]]
        )
        edge_median_offset = np.median(edge_vals)
        self.last_Ts_final = Ts_baseline_subtracted - edge_median_offset

        self._ts_interp_func = interp1d(
            self._last_freq_axis,
            self.last_Ts_final,
            bounds_error=False,
            fill_value=np.nan
        )
        self._add_message("Brightness temperature calculation complete.")
        self._create_or_update_brightness_tab()

    def _create_or_update_brightness_tab(self):
        """Creates the HI Line Temperature tab if it doesn't exist, then updates the plot."""
        try:
            if not self.brightness_tab_created:
                self.brightness_tab = LogoTabPageWidget()
                page_layout = self.brightness_tab.content_layout

                coord_layout = QHBoxLayout()
                self.bt_freq_label = QLabel("Frequency: N/A")
                self.bt_temp_label = QLabel("Temperature: N/A")
                font = self.bt_freq_label.font()
                font.setPointSize(14)
                self.bt_freq_label.setFont(font)
                self.bt_temp_label.setFont(font)
                coord_layout.addStretch()
                coord_layout.addWidget(self.bt_freq_label)
                coord_layout.addStretch()
                coord_layout.addWidget(self.bt_temp_label)
                coord_layout.addStretch()
                page_layout.addLayout(coord_layout)

                plot_widget, self.bt_fig, self.bt_canvas, _ = self._create_plot_widget(self)
                self.bt_ax = self.bt_fig.add_subplot(111)
                page_layout.addWidget(plot_widget)

                self.bt_canvas.mpl_connect('motion_notify_event', self.on_bt_plot_hover)

                self.tab_widget.addTab(self.brightness_tab, "HI Line Temperature")
                self.brightness_tab_created = True
                self._add_message("HI Line Temperature tab created successfully.")

            self._update_brightness_plot()

            idx = self.tab_widget.indexOf(self.brightness_tab)
            if idx != -1:
                self.tab_widget.setCurrentIndex(idx)

        except Exception as e:
            self._add_message(f"FATAL: Could not create or update the GUI tab. Error: {e}")
            QMessageBox.critical(
                self,
                "GUI Error",
                f"A critical error occurred while building the display tab: {e}\n\n{traceback.format_exc()}"
            )

    def _update_brightness_plot(self):
        """Updates the brightness temperature plot with the latest calculated data."""
        if self.last_Ts_final is None or self._last_freq_axis is None:
            self._add_message("ERROR: Cannot update plot, no final data available.")
            return

        n_points = len(self._last_freq_axis)
        chop_count = int(n_points * 0.05)
        if chop_count * 2 >= n_points:
            chop_count = 0

        freq_axis_chopped = self._last_freq_axis[chop_count:-chop_count]
        Ts_to_plot = self.last_Ts_final[chop_count:-chop_count]

        self.bt_ax.clear()
        self._setup_plot_formatting(
            self.bt_ax,
            "HI Line Temperature Profile",
            "Frequency (MHz)",
            "Line Temperature (K)"
        )

        self.bt_ax.plot(
            freq_axis_chopped,
            Ts_to_plot,
            label="Final $T_{line}$",
            linewidth=1.5,
            color='green'
        )
        self.bt_ax.axhline(0, color='black', linestyle='-.', linewidth=1, label='Zero Baseline')

        self.bt_ax.legend(loc='upper left')
        self.bt_ax.grid(True, linestyle='--', alpha=0.6)
        self.bt_canvas.draw()

    def on_bt_plot_hover(self, event):
        if event.inaxes == self.bt_ax and event.xdata is not None and event.ydata is not None:
            self.bt_freq_label.setText(f"Frequency: {event.xdata:.4f} MHz")
            self.bt_temp_label.setText(f"Temperature: {event.ydata:.2f} K")
        else:
            self.bt_freq_label.setText("Frequency: N/A")
            self.bt_temp_label.setText("Temperature: N/A")

    # =================================================================================
    # SDR CONTROL
    # =================================================================================
    def start_sdr_task(self, mode, button_to_disable, button_to_enable, params):
        if self.is_sdr_busy:
            QMessageBox.warning(self, "Busy", "Another SDR task is already running.")
            return
        self.is_sdr_busy = True
        button_to_disable.setEnabled(False)
        button_to_enable.setEnabled(True)
        self.thread = QThread()
        self.worker = SdrWorker(params, mode=mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_sdr_finished)
        self.worker.progress.connect(self.status_bar.showMessage)
        if mode == 'acquire':
            self.worker.plot_data.connect(self.update_acq_plot)
            self.worker.final_data_for_save.connect(self.set_final_data)
            self.worker.noise_freq_detected.connect(self.set_noise_freq)
        elif mode == 'quick_look':
            self.worker.plot_data.connect(self.update_ql_plot)
        self.thread.start()

    def set_noise_freq(self, freq_mhz):
        if freq_mhz > 0:
            self.detected_noise_freq = freq_mhz
            self._add_message(f"Detected dominant RFI: {freq_mhz:.2f} MHz")
        else:
            self.detected_noise_freq = None
            self._add_message("No dominant RFI detected.")

    def start_quick_look(self):
        params = {
            'center_freq': DEFAULT_CENTER_FREQ * 1e6,
            'sample_rate': DEFAULT_SAMPLE_RATE * 1e6,
            'gain': DEFAULT_GAIN,
            'nfft': DEFAULT_NF
        }
        self.start_sdr_task('quick_look', self.ql_start_button, self.ql_stop_button, params)

    def start_recording(self):
        params = {
            'center_freq': float(self.acq_center_freq_entry.text()) * 1e6,
            'sample_rate': float(self.acq_sample_rate_entry.text()) * 1e6,
            'gain': float(self.acq_gain_entry.text()),
            'nfft': int(self.acq_nf_entry.text()),
            'duration': float(self.acq_int_time_entry.text()),
        }
        self.start_sdr_task('acquire', self.record_button, self.stop_acq_button, params)

    def stop_sdr_task(self):
        if self.worker:
            self.worker.stop()
        self.ql_stop_button.setEnabled(False)
        self.stop_acq_button.setEnabled(False)

    def on_sdr_finished(self):
        if self.thread is None:
            return
        self.thread.quit()
        self.thread.wait()
        self.thread, self.worker, self.is_sdr_busy = None, None, False
        self.ql_start_button.setEnabled(True)
        self.ql_stop_button.setEnabled(False)
        self.record_button.setEnabled(True)
        self.stop_acq_button.setEnabled(False)
        if self.final_data_to_save:
            self._add_message("Recording finished. Ready to save.")
            self.save_recorded_data()
            self.final_data_to_save = None

    def update_acq_plot(self, freqs, power_db, title):
        self._setup_plot_formatting(self.acq_ax, title, "Frequency (MHz)", "Power (dB)")
        self.acq_ax.plot(freqs, power_db)
        self.acq_canvas.draw()

    def update_ql_plot(self, freqs, power_db, title):
        self._setup_plot_formatting(self.ql_ax, title, "Frequency (MHz)", "Relative Power (dB)")
        self.ql_ax.plot(freqs, power_db)
        self.ql_canvas.draw()

    def set_final_data(self, freqs, power_linear):
        self.final_data_to_save = (freqs, power_linear)

    def save_recorded_data(self):
        if not self.final_data_to_save:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data File", self.group_folder, "CSV Files (*.csv)"
        )
        if not file_path:
            self._add_message("Save operation cancelled.")
            return
        try:
            freqs, power_linear = self.final_data_to_save
            df = pd.DataFrame({'Frequency (MHz)': freqs, 'Power (linear units)': power_linear})
            df.to_csv(file_path, index=False)
            self._add_message(f"Data successfully saved to {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")

    # =================================================================================
    # FLOATING CLOSE BUTTON + COPYRIGHT LABEL
    # =================================================================================
    def _add_floating_close_button(self):
        self._close_btn = QPushButton("C", self)
        self._close_btn.setToolTip("Turn OFF LNAB and close the app")
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.setFixedSize(30, 30)

        self._close_btn.setStyleSheet("""
            QPushButton {
                padding: 0;
                font-size: 10px;
                border-radius: 6px;
                border: 0px;
                background-color: rgba(255, 0, 0, 10);
                color: white;
            }
            QPushButton:hover   { background-color: rgba(255, 0, 0, 110); }
            QPushButton:pressed { background-color: rgba(200, 0, 0, 160); }
        """)

        self._close_btn.clicked.connect(self._on_close_and_poweroff_clicked)
        self._position_floating_button()
        self.installEventFilter(self)

    def _position_floating_button(self):
        if not hasattr(self, "_close_btn") or self._close_btn is None:
            return
        margin_side = 34
        margin_bottom = 52
        btn_w = self._close_btn.sizeHint().width()
        btn_h = self._close_btn.sizeHint().height()
        x = self.width() - btn_w - margin_side
        y = self.height() - btn_h - margin_bottom
        self._close_btn.move(max(0, x), max(0, y))
        self._close_btn.raise_()


    def _add_copyright_label(self):
        """Create a small '© HBCSE, TIFR' label at the bottom-right corner."""
        self._copyright_label = QLabel("© HBCSE, TIFR", self)
        font = QFont("DejaVu Sans", 11, QFont.Weight.Bold)  # +2 points, bold
        self._copyright_label.setFont(font)
        self._copyright_label.setStyleSheet(
            "color: rgba(255, 255, 255, 255);"      # fully white, more visible
            "background-color: rgba(0, 0, 0, 160);" # slightly darker background
            "padding: 3px 8px;"
        )
        self._copyright_label.adjustSize()
        self._position_copyright_label()


    def _position_copyright_label(self):
        """Reposition the copyright label to stay at bottom-right."""
        if not hasattr(self, "_copyright_label") or self._copyright_label is None:
            return

        margin_side = 10
        margin_bottom = 10
        label_w = self._copyright_label.sizeHint().width()
        label_h = self._copyright_label.sizeHint().height()
        x = self.width() - label_w - margin_side
        y = self.height() - label_h - margin_bottom
        self._copyright_label.move(max(0, x), max(0, y))
        self._copyright_label.raise_()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            self._position_floating_button()
            self._position_copyright_label()
        return super().eventFilter(obj, event)

    # =================================================================================
    # LNA POWER CONTROL & CLOSE EVENTS
    # =================================================================================
    def _power_off_lna(self):
        """Run rtl_biast -b 0 to turn OFF bias-T (LNA). Safe to call even if not present."""
        try:
            if which("rtl_biast") is None:
                self._add_message("rtl_biast not found; skipping LNA power OFF.")
                return
            subprocess.run(["rtl_biast", "-b", "0"], check=True, capture_output=True, text=True)
            self._add_message("LNA powered OFF (rtl_biast -b 0).")
        except subprocess.CalledProcessError as e:
            self._add_message(
                f"Warning: Failed to power OFF LNA. stderr: {e.stderr.strip()}"
            )
        except Exception as e:
            self._add_message(f"Warning: Unexpected error while powering OFF LNA: {e}")

    def closeEvent(self, event):
        reply = QMessageBox.warning(
            self,
            "Exit Confirmation",
            "Are you sure you want to exit? This will power off the LNAB.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                self._add_message("Application closing. Automatically saving log...")
                self._save_message_log()
                self._power_off_lna()
            finally:
                event.accept()
        else:
            event.ignore()

    def _on_close_and_poweroff_clicked(self):
        self.close()

# ====================================================================================
# MAIN
# ====================================================================================
if __name__ == "__main__":
    try:
        print("Powering LNA ON...")
        result = subprocess.run(
            ['rtl_biast', '-b', '1'],
            check=True,
            capture_output=True,
            text=True
        )
        print("LNA powered ON successfully.")
    except FileNotFoundError:
        print(
            "ERROR: 'rtl_biast' command not found. "
            "Make sure rtl-sdr tools are installed and in your system's PATH."
        )
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: Failed to power on LNA. 'rtl_biast' returned an error.\n"
            f"Standard Error: {e.stderr}"
        )
    except Exception as e:
        print(f"An unexpected error occurred while trying to power on LNA: {e}")

    app = QApplication(sys.argv)

    while True:
        dialog = QInputDialog()
        dialog.setWindowTitle("Group Code")
        dialog.setLabelText("Enter your group code:")
        dialog.resize(400, 150)
        ok = dialog.exec()
        group_code = dialog.textValue()
        if not ok or not group_code:
            sys.exit()

        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        group_folder = os.path.join(desktop_path, group_code)

        if os.path.exists(group_folder):
            reply = QMessageBox.question(
                None,
                "Folder Already Exists",
                f"The folder '{group_code}' already exists on Desktop.\n"
                f"Do you want to proceed and use this folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                break
            else:
                continue
        else:
            os.makedirs(group_folder)
            break

    window = RadioAstronomyApp(group_code=group_code, group_folder=group_folder)
    window.setWindowState(Qt.WindowMaximized | Qt.WindowActive)
    window.show()
    sys.exit(app.exec())
