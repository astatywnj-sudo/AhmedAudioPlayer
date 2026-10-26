# -*- coding: utf-8 -*-
"""
AhmedAudioPlayer V5.6.2
- Root cause fix for "speeds up after ~1s"
- Anti-crackle (fire crackling) fix
- Fast parameter smoothing so sliders actually respond (~3s settle)
- Volume / Speed / Pitch / Echo work live while playing
"""

import sys
import os
import threading
import traceback

import numpy as np
import soundfile as sf
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox,
    QFrame, QComboBox
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR_DEFAULT = 44100
BLOCK = 4096


# ---------------------------------------------------------------------------
# Audio Engine
# ---------------------------------------------------------------------------

class AudioEngine:
    """
    Single source of truth: `_source` (float32, shape [N,2]).
    Single playback position: `_pos` + `_frac`.
    Callback reads ONLY from `_source`, applies smoothed params, writes outdata.
    """

    def __init__(self, samplerate=SR_DEFAULT, blocksize=BLOCK):
        self.samplerate = samplerate
        self.blocksize = blocksize

        # Source (set once per load)
        self._source: np.ndarray | None = None
        self._n_frames: int = 0

        # Playback state
        self._pos: int = 0
        self._frac: float = 0.0
        self._playing: bool = False
        self._lock = threading.Lock()

        # Target params (set by UI)
        self._t_speed = 1.0
        self._t_pitch = 0.0
        self._t_echo = 0.0
        self._t_volume = 0.8

        # Smoothed params (used in callback)
        self._s_speed = 1.0
        self._s_pitch = 0.0
        self._s_echo = 0.0
        self._s_volume = 0.8
        # Per-sample one-pole coefficient: tau = 3 ms
        self._alpha = 1.0 - np.exp(-1.0 / (0.003 * samplerate))

        # Echo delay line
        self._echo_buf = np.zeros((int(samplerate * 0.5), 2), dtype=np.float32)
        self._echo_idx = 0

        # Fade in/out (10 ms)
        self._fade = 0.0
        self._fade_target = 0.0
        self._fade_step = 1.0 / (0.010 * samplerate)

        # Stream
        self._stream: sd.OutputStream | None = None

    # ---------- loading ----------

    def set_source(self, samples: np.ndarray, samplerate: int):
        with self._lock:
            if samples.ndim == 1:
                samples = np.stack([samples, samples], axis=1)
            samples = np.ascontiguousarray(samples, dtype=np.float32)

            if samplerate != self.samplerate:
                ratio = self.samplerate / float(samplerate)
                new_n = int(samples.shape[0] * ratio)
                idx = np.linspace(0, samples.shape[0] - 1, new_n)
                i0 = np.floor(idx).astype(np.int64)
                i1 = np.minimum(i0 + 1, samples.shape[0] - 1)
                f = (idx - i0).astype(np.float32)[:, None]
                samples = (samples[i0] * (1 - f) + samples[i1] * f).astype(np.float32)

            self._source = samples
            self._n_frames = samples.shape[0]
            self._pos = 0
            self._frac = 0.0
            self._echo_buf[:] = 0.0
            self._echo_idx = 0
            self._fade = 0.0
            self._fade_target = 0.0
            # sync smoothed -> target on load
            self._s_speed = self._t_speed
            self._s_pitch = self._t_pitch
            self._s_echo = self._t_echo
            self._s_volume = self._t_volume

    def clear(self):
        with self._lock:
            self._source = None
            self._n_frames = 0
            self._pos = 0
            self._frac = 0.0
            self._playing = False

    # ---------- transport ----------

    def play(self):
        with self._lock:
            if self._source is None:
                return
            if self._playing:
                return
            self._playing = True
            self._fade_target = 1.0
        self._ensure_stream_started()

    def pause(self):
        with self._lock:
            if not self._playing:
                return
            self._fade_target = 0.0
            self._playing = False

    def stop(self):
        with self._lock:
            self._playing = False
            self._pos = 0
            self._frac = 0.0
            self._fade = 0.0
            self._fade_target = 0.0
        self._close_stream()

    def seek(self, frame_index: int):
        with self._lock:
            if self._source is None:
                return
            self._pos = max(0, min(int(frame_index), self._n_frames - 1))
            self._frac = 0.0

    def position_seconds(self) -> float:
        with self._lock:
            return self._pos / float(self.samplerate)

    def duration_seconds(self) -> float:
        with self._lock:
            if self._source is None:
                return 0.0
            return self._n_frames / float(self.samplerate)

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    # ---------- params ----------

    def set_speed(self, v: float):
        self._t_speed = max(0.25, min(4.0, float(v)))

    def set_pitch(self, s: float):
        self._t_pitch = max(-12.0, min(12.0, float(s)))

    def set_echo(self, v: float):
        self._t_echo = max(0.0, min(1.0, float(v)))

    def set_volume(self, v: float):
        self._t_volume = max(0.0, min(1.0, float(v)))

    def get_speed(self):  return self._t_speed
    def get_pitch(self):  return self._t_pitch
    def get_echo(self):   return self._t_echo
    def get_volume(self): return self._t_volume

    # ---------- stream ----------

    def _ensure_stream_started(self):
        if self._stream is not None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=2,
                dtype='float32',
                callback=self._callback,
                latency='low',
            )
            self._stream.start()
        except Exception as e:
            print("Stream start failed:", e)
            self._stream = None

    def _close_stream(self):
        s = self._stream
        self._stream = None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    # ---------- callback ----------

    def _callback(self, outdata, frames, time_info, status):
        try:
            src = self._source
            if src is None:
                outdata.fill(0.0)
                return

            # --- fast one-pole smoothing (closed-form per block) ---
            a = self._alpha
            decay = (1.0 - a) ** frames
            self._s_speed = self._t_speed + (self._s_speed - self._t_speed) * decay
            self._s_pitch = self._t_pitch + (self._s_pitch - self._t_pitch) * decay
            self._s_echo = self._t_echo + (self._s_echo - self._t_echo) * decay
            self._s_volume = self._t_volume + (self._s_volume - self._t_volume) * decay

            speed = self._s_speed
            vol = self._s_volume
            echo = self._s_echo
            pitch = self._s_pitch

            # --- fade state ---
            fade = self._fade
            fade_t = self._fade_target
            step = self._fade_step

            # --- read source (speed + pitch combined) ---
            n = self._n_frames
            start = self._pos
            pitch_ratio = 2.0 ** (pitch / 12.0)
            advance = speed / pitch_ratio

            idx = start + np.arange(frames, dtype=np.float64) * advance + self._frac

            end_reached = False
            if idx[-1] >= n - 1:
                valid = idx < (n - 1)
                k = int(np.sum(valid))
                if k <= 0:
                    outdata.fill(0.0)
                    with self._lock:
                        self._playing = False
                        self._pos = 0
                        self._frac = 0.0
                    return
                idx_valid = idx[:k]
                block = np.zeros((frames, 2), dtype=np.float32)
                block[:k] = self._interp(src, idx_valid)
                end_reached = True
            else:
                block = self._interp(src, idx)

            # --- echo (read delayed, mix, write dry back) ---
            if echo > 1e-4:
                eb = self._echo_buf
                L = eb.shape[0]
                read_idx = (self._echo_idx + np.arange(frames)) % L
                delayed = eb[read_idx].copy()
                dry = block.copy()
                block = block + echo * delayed
                eb[read_idx] = dry
                self._echo_idx = (self._echo_idx + frames) % L

            # --- volume ---
            block *= vol

            # --- fade ramp (per-sample) ---
            if fade != fade_t:
                if fade < fade_t:
                    end_f = min(fade_t, fade + step * frames)
                else:
                    end_f = max(fade_t, fade - step * frames)
                ramp = np.linspace(fade, end_f, frames, dtype=np.float32)[:, None]
                block *= ramp
                self._fade = end_f
            elif fade < 1.0:
                block *= fade

            # --- soft clip ---
            np.clip(block, -1.0, 1.0, out=block)
            outdata[:] = block

            # --- advance position ---
            if end_reached:
                with self._lock:
                    self._playing = False
                    self._pos = 0
                    self._frac = 0.0
            else:
                last = idx[-1]
                with self._lock:
                    self._pos = int(last)
                    self._frac = float(last - int(last))

        except Exception:
            traceback.print_exc()
            outdata.fill(0.0)

    @staticmethod
    def _interp(src: np.ndarray, idx: np.ndarray) -> np.ndarray:
        n = src.shape[0]
        idx = np.clip(idx, 0.0, n - 1.001)
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        f = (idx - i0).astype(np.float32)[:, None]
        return (src[i0] * (1.0 - f) + src[i1] * f).astype(np.float32)


# ---------------------------------------------------------------------------
# Background Loader / Saver
# ---------------------------------------------------------------------------

class AudioLoader(QObject):
    finished = Signal(object, int, str)
    failed = Signal(str)

    def load(self, path: str):
        def run():
            try:
                data, sr = sf.read(path, dtype='float32', always_2d=False)
                self.finished.emit(data, sr, path)
            except Exception as e:
                self.failed.emit(str(e))
        threading.Thread(target=run, daemon=True).start()


class AudioSaver(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def save(self, path: str, samples: np.ndarray, samplerate: int):
        def run():
            try:
                sf.write(path, samples, samplerate, subtype='PCM_16')
                self.finished.emit(path)
            except Exception as e:
                self.failed.emit(str(e))
        threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

THEMES = {
    "dark": {
        "bg": "#1a1410",
        "panel": "#241b14",
        "text": "#f0e0d0",
        "accent": "#c87f3a",
        "accent2": "#8b5a2b",
        "border": "#3a2a1c",
        "slider": "#c87f3a",
    },
    "light": {
        "bg": "#f7f4f0",
        "panel": "#ffffff",
        "text": "#2a2018",
        "accent": "#c87f3a",
        "accent2": "#a0622a",
        "border": "#e0d6cc",
        "slider": "#c87f3a",
    },
}


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AhmedAudioPlayer — V5.6.2")
        self.resize(920, 620)

        self.engine = AudioEngine()
        self.loader = AudioLoader()
        self.saver = AudioSaver()
        self.loader.finished.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.saver.finished.connect(self._on_saved)
        self.saver.failed.connect(self._on_save_failed)

        self.current_file = None
        self.current_theme = "dark"
        self._seeking = False

        self._build_ui()
        self._apply_theme("dark")

        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ---------- UI ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.header = QLabel("AUDIO PLAYER")
        self.header.setAlignment(Qt.AlignCenter)
        f = QFont("Segoe UI", 22, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 6)
        self.header.setFont(f)
        root.addWidget(self.header)

        self.file_label = QLabel("No file loaded")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setWordWrap(True)
        root.addWidget(self.file_label)

        # Timeline
        tl_row = QHBoxLayout()
        self.time_cur = QLabel("0:00")
        self.time_cur.setFixedWidth(56)
        self.time_cur.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderPressed.connect(self._seek_start)
        self.timeline.sliderReleased.connect(self._seek_end)
        self.time_dur = QLabel("0:00")
        self.time_dur.setFixedWidth(56)
        tl_row.addWidget(self.time_cur)
        tl_row.addWidget(self.timeline, 1)
        tl_row.addWidget(self.time_dur)
        root.addLayout(tl_row)

        # Transport
        tr = QHBoxLayout()
        tr.setSpacing(8)
        self.btn_open = QPushButton("Open Audio")
        self.btn_play = QPushButton("▶ Play")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_stop = QPushButton("⏹ Stop")
        for b in (self.btn_open, self.btn_play, self.btn_pause, self.btn_stop):
            b.setMinimumHeight(38)
            tr.addWidget(b)
        tr.addStretch(1)
        self.btn_save = QPushButton("💾 Save WAV")
        self.btn_reset_audio = QPushButton("↺ Reset Audio")
        self.btn_reset_settings = QPushButton("⚙ Reset Settings")
        for b in (self.btn_save, self.btn_reset_audio, self.btn_reset_settings):
            b.setMinimumHeight(38)
            tr.addWidget(b)
        root.addLayout(tr)

        # Sliders panel
        panel = QFrame()
        panel.setObjectName("panel")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(14, 14, 14, 14)
        pv.setSpacing(10)

        self.sl_volume = self._add_slider(pv, "Volume", 0, 100, 80,
                                          self._on_volume)
        self.sl_speed = self._add_slider(pv, "Speed", 25, 400, 100,
                                         self._on_speed,
                                         fmt=lambda v: f"{v/100:.2f}x")
        self.sl_pitch = self._add_slider(pv, "Pitch", -12, 12, 0,
                                         self._on_pitch,
                                         fmt=lambda v: f"{v:+d} st")
        self.sl_echo = self._add_slider(pv, "Echo", 0, 100, 0,
                                        self._on_echo,
                                        fmt=lambda v: f"{v}%")

        root.addWidget(panel)

        # Theme + status
        th = QHBoxLayout()
        th.addWidget(QLabel("Theme:"))
        self.theme_box = QComboBox()
        self.theme_box.addItems(["Dark Autumn", "Light White"])
        self.theme_box.currentIndexChanged.connect(self._on_theme_change)
        th.addWidget(self.theme_box)
        th.addStretch(1)
        self.status = QLabel("Ready")
        th.addWidget(self.status)
        root.addLayout(th)

        # signals
        self.btn_open.clicked.connect(self._on_open)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_reset_audio.clicked.connect(self._on_reset_audio)
        self.btn_reset_settings.clicked.connect(self._on_reset_settings)

    def _add_slider(self, parent_layout, name, lo, hi, val, cb, fmt=None):
        row = QHBoxLayout()
        lbl = QLabel(name)
        lbl.setFixedWidth(70)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        val_lbl = QLabel(fmt(val) if fmt else str(val))
        val_lbl.setFixedWidth(70)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def on_change(v):
            val_lbl.setText(fmt(v) if fmt else str(v))
            cb(v)

        s.valueChanged.connect(on_change)
        row.addWidget(lbl)
        row.addWidget(s, 1)
        row.addWidget(val_lbl)
        parent_layout.addLayout(row)
        return s

    # ---------- theme ----------

    def _apply_theme(self, name):
        t = THEMES[name]
        self.current_theme = name
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {t['bg']};
                color: {t['text']};
                font-family: 'Segoe UI', 'Tahoma', sans-serif;
                font-size: 13px;
            }}
            QFrame#panel {{
                background: {t['panel']};
                border: 1px solid {t['border']};
                border-radius: 10px;
            }}
            QPushButton {{
                background: {t['panel']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 8px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background: {t['accent2']};
                color: #fff;
                border: 1px solid {t['accent']};
            }}
            QPushButton:pressed {{
                background: {t['accent']};
                color: #fff;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: {t['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {t['slider']};
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {t['accent']};
                border-radius: 3px;
            }}
            QComboBox {{
                background: {t['panel']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 6px;
                padding: 4px 8px;
            }}
        """)

    def _on_theme_change(self, idx):
        self._apply_theme("dark" if idx == 0 else "light")

    # ---------- actions ----------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "",
            "Audio Files (*.wav *.flac *.ogg *.mp3 *.m4a *.aiff *.aif);;All Files (*.*)"
        )
        if not path:
            return
        self.status.setText("Loading...")
        self.btn_open.setEnabled(False)
        self.engine.stop()
        self.loader.load(path)

    def _on_loaded(self, data, sr, path):
        self.engine.set_source(data, sr)
        self.current_file = path
        self.file_label.setText(os.path.basename(path))
        self.time_dur.setText(self._fmt(self.engine.duration_seconds()))
        self.btn_open.setEnabled(True)
        self.status.setText("Loaded")
        self.engine.seek(0)

    def _on_load_failed(self, msg):
        self.btn_open.setEnabled(True)
        self.status.setText("Load failed")
        QMessageBox.critical(self, "Load error", msg)

    def _on_play(self):
        if self.engine._source is None:
            return
        self.engine.play()

    def _on_pause(self):
        self.engine.pause()

    def _on_stop(self):
        self.engine.stop()

    def _on_save(self):
        if self.engine._source is None:
            QMessageBox.information(self, "Save", "No audio loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save WAV", "output.wav", "WAV (*.wav)"
        )
        if not path:
            return
        self.status.setText("Saving...")
        self.saver.save(path, self.engine._source, self.engine.samplerate)

    def _on_saved(self, path):
        self.status.setText(f"Saved: {os.path.basename(path)}")

    def _on_save_failed(self, msg):
        self.status.setText("Save failed")
        QMessageBox.critical(self, "Save error", msg)

    def _on_reset_audio(self):
        self.engine.stop()
        self.engine.seek(0)
        self.status.setText("Audio reset")

    def _on_reset_settings(self):
        self.sl_volume.setValue(80)
        self.sl_speed.setValue(100)
        self.sl_pitch.setValue(0)
        self.sl_echo.setValue(0)
        self.status.setText("Settings reset")

    # ---------- param handlers ----------

    def _on_volume(self, v): self.engine.set_volume(v / 100.0)
    def _on_speed(self, v):  self.engine.set_speed(v / 100.0)
    def _on_pitch(self, v):  self.engine.set_pitch(v)
    def _on_echo(self, v):   self.engine.set_echo(v / 100.0)

    # ---------- timeline ----------

    def _seek_start(self):
        self._seeking = True

    def _seek_end(self):
        self._seeking = False
        dur = self.engine.duration_seconds()
        if dur <= 0:
            return
        frac = self.timeline.value() / 1000.0
        self.engine.seek(int(frac * self.engine._n_frames))

    def _tick(self):
        dur = self.engine.duration_seconds()
        pos = self.engine.position_seconds()
        self.time_cur.setText(self._fmt(pos))
        self.time_dur.setText(self._fmt(dur))
        if not self._seeking and dur > 0:
            self.timeline.blockSignals(True)
            self.timeline.setValue(int((pos / dur) * 1000))
            self.timeline.blockSignals(False)
        self.btn_play.setText("▶ Play" if not self.engine.is_playing() else "▶ Playing")

    @staticmethod
    def _fmt(sec):
        sec = max(0, int(sec))
        return f"{sec // 60}:{sec % 60:02d}"

    # ---------- close ----------

    def closeEvent(self, e):
        self.engine.stop()
        e.accept()


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()