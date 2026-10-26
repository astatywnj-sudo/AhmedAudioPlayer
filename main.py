# -*- coding: utf-8 -*-
import sys, os, threading, traceback, shutil, tempfile, urllib.request, urllib.parse
import random
import json
import math
import numpy as np
import soundfile as sf
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPointF, QSize, QRect
from PySide6.QtGui import (QFont, QKeySequence, QShortcut, QPainter,
                            QColor, QBrush, QPixmap, QIcon, QPen)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QFileDialog, QMessageBox, QFrame, QComboBox, QListWidget,
    QListWidgetItem, QSplitter, QLineEdit, QCheckBox, QScrollArea,
    QGridLayout, QGroupBox, QToolButton, QSizePolicy)

SR_DEFAULT = 44100
BLOCK = 2048
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".mp4",
              ".aiff", ".aif", ".aac", ".opus")


def find_ffmpeg():
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    p = shutil.which("ffmpeg")
    return p if p else None


FFMPEG_PATH = find_ffmpeg()


def auto_detect_performance():
    cores = os.cpu_count() or 1
    if cores <= 2:
        return "low"
    elif cores <= 4:
        return "medium"
    return "high"


def extract_album_art(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".mp3":
            from mutagen.mp3 import MP3
            audio = MP3(path)
            if audio.tags:
                for tag in audio.tags.values():
                    if tag.FrameID == "APIC":
                        return tag.data
        elif ext in (".m4a", ".mp4", ".aac"):
            from mutagen.mp4 import MP4
            audio = MP4(path)
            if audio.tags and "covr" in audio.tags:
                covers = audio.tags["covr"]
                if covers:
                    return bytes(covers[0])
        elif ext == ".flac":
            from mutagen.flac import FLAC
            audio = FLAC(path)
            if audio.pictures:
                return audio.pictures[0].data
        elif ext == ".ogg":
            from mutagen.oggvorbis import OggVorbis
            audio = OggVorbis(path)
            if audio.tags and "metadata_block_picture" in audio.tags:
                import base64
                data = base64.b64decode(audio.tags["metadata_block_picture"][0])
                pos = 4
                mime_len = int.from_bytes(data[pos:pos+4], 'big')
                pos += 4 + mime_len
                desc_len = int.from_bytes(data[pos:pos+4], 'big')
                pos += 4 + desc_len
                pos += 20
                img_len = int.from_bytes(data[pos:pos+4], 'big')
                pos += 4
                return data[pos:pos+img_len]
    except Exception:
        pass
    return None


def load_album_icon(path, size=40):
    art = extract_album_art(path)
    if art is None:
        return None
    try:
        pix = QPixmap()
        pix.loadFromData(art)
        if pix.isNull():
            return None
        pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(size, size)
        canvas.fill(QColor(0, 0, 0, 0))
        p = QPainter(canvas)
        x = (size - pix.width()) // 2
        y = (size - pix.height()) // 2
        p.drawPixmap(x, y, pix)
        p.end()
        return QIcon(canvas)
    except Exception:
        return None


def load_album_pixmap(path, size=80):
    art = extract_album_art(path)
    if art is None:
        return None
    try:
        pix = QPixmap()
        pix.loadFromData(art)
        if pix.isNull():
            return None
        return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception:
        return None


# ============================================================
# Biquad Filter
# ============================================================
class Biquad:
    def __init__(self, sr):
        self.sr = sr
        self.b0 = 1.0; self.b1 = 0.0; self.b2 = 0.0
        self.a1 = 0.0; self.a2 = 0.0
        self.z1 = np.zeros(2, dtype=np.float32)
        self.z2 = np.zeros(2, dtype=np.float32)

    def set_peaking(self, freq, q, gain_db):
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * freq / self.sr
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
        self.b0 = b0 / a0; self.b1 = b1 / a0; self.b2 = b2 / a0
        self.a1 = a1 / a0; self.a2 = a2 / a0

    def set_lowshelf(self, freq, gain_db, S=1.0):
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * freq / self.sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0/A) * (1.0/S - 1.0) + 2.0)
        two_sqrt_A_alpha = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A+1) - (A-1)*cos_w0 + two_sqrt_A_alpha)
        b1 = 2 * A * ((A-1) - (A+1)*cos_w0)
        b2 = A * ((A+1) - (A-1)*cos_w0 - two_sqrt_A_alpha)
        a0 = (A+1) + (A-1)*cos_w0 + two_sqrt_A_alpha
        a1 = -2 * ((A-1) + (A+1)*cos_w0)
        a2 = (A+1) + (A-1)*cos_w0 - two_sqrt_A_alpha
        self.b0 = b0 / a0; self.b1 = b1 / a0; self.b2 = b2 / a0
        self.a1 = a1 / a0; self.a2 = a2 / a0

    def set_highshelf(self, freq, gain_db, S=1.0):
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * freq / self.sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0/A) * (1.0/S - 1.0) + 2.0)
        two_sqrt_A_alpha = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A+1) + (A-1)*cos_w0 + two_sqrt_A_alpha)
        b1 = -2 * A * ((A-1) + (A+1)*cos_w0)
        b2 = A * ((A+1) + (A-1)*cos_w0 - two_sqrt_A_alpha)
        a0 = (A+1) - (A-1)*cos_w0 + two_sqrt_A_alpha
        a1 = 2 * ((A-1) - (A+1)*cos_w0)
        a2 = (A+1) - (A-1)*cos_w0 - two_sqrt_A_alpha
        self.b0 = b0 / a0; self.b1 = b1 / a0; self.b2 = b2 / a0
        self.a1 = a1 / a0; self.a2 = a2 / a0

    def process(self, x):
        b0, b1, b2 = self.b0, self.b1, self.b2
        a1, a2 = self.a1, self.a2
        out = np.empty_like(x)
        z1 = self.z1.copy()
        z2 = self.z2.copy()
        for i in range(x.shape[0]):
            for ch in range(2):
                v = x[i, ch]
                y = b0*v + z1[ch]
                z1[ch] = b1*v - a1*y + z2[ch]
                z2[ch] = b2*v - a2*y
                out[i, ch] = y
        self.z1 = z1
        self.z2 = z2
        return out


class EQ5Band:
    FREQS = [60.0, 230.0, 910.0, 3600.0, 14000.0]
    PRESETS = {
        "Flat": [0, 0, 0, 0, 0],
        "Bass Boost": [8, 5, 0, -1, -2],
        "Vocal": [-3, -1, 2, 4, 2],
        "Rock": [5, 2, -2, 2, 4],
        "Podcast": [-5, -2, 3, 4, 0],
        "Classical": [3, 1, 0, 2, 3],
        "Electronic": [6, 3, 0, 2, 5],
        "Acoustic": [3, 1, 1, 3, 2],
    }

    def __init__(self, sr):
        self.sr = sr
        self.filters = [Biquad(sr) for _ in range(5)]
        self.gains = [0.0] * 5
        self._enabled = False
        self._dirty = True

    def set_gains(self, gains):
        self.gains = [max(-12.0, min(12.0, float(g))) for g in gains]
        self._dirty = True
        self._enabled = any(abs(g) > 0.01 for g in self.gains)

    def process(self, x):
        if not self._enabled:
            return x
        if self._dirty:
            for i, (bq, f, g) in enumerate(zip(self.filters, self.FREQS, self.gains)):
                bq.set_peaking(f, 1.0, g)
            self._dirty = False
        out = x.astype(np.float32).copy()
        for bq in self.filters:
            out = bq.process(out)
        return out

    def is_enabled(self):
        return self._enabled


class Reverb:
    def __init__(self, sr):
        self.sr = sr
        comb_delays = [int(sr * d) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
        self.comb_bufs = [np.zeros((d, 2), dtype=np.float32) for d in comb_delays]
        self.comb_idx = [0] * 4
        self.comb_delays = comb_delays
        ap_delays = [int(sr * d) for d in [0.0050, 0.0017]]
        self.ap_bufs = [np.zeros((d, 2), dtype=np.float32) for d in ap_delays]
        self.ap_idx = [0, 0]
        self.ap_delays = ap_delays
        self.room = 0.5
        self.damp = 0.5
        self.mix = 0.0

    def set_params(self, room, damp, mix):
        self.room = max(0.0, min(1.0, float(room)))
        self.damp = max(0.0, min(1.0, float(damp)))
        self.mix = max(0.0, min(1.0, float(mix)))

    def is_enabled(self):
        return self.mix > 1e-4

    def process(self, x):
        if not self.is_enabled():
            return x
        frames = x.shape[0]
        dry = x.copy()
        feedback = 0.7 + 0.28 * self.room
        damp = 0.2 + 0.6 * self.damp
        comb_out = np.zeros_like(x)
        for i, buf in enumerate(self.comb_bufs):
            d = self.comb_delays[i]
            idx = self.comb_idx[i]
            read_idx = (idx + np.arange(frames)) % d
            delayed = buf[read_idx].copy()
            delayed = delayed * (1.0 - damp)
            write_sig = x + delayed * feedback
            buf[read_idx] = write_sig
            self.comb_idx[i] = (idx + frames) % d
            comb_out += delayed
        comb_out *= 0.25
        ap = comb_out
        for i, buf in enumerate(self.ap_bufs):
            d = self.ap_delays[i]
            idx = self.ap_idx[i]
            read_idx = (idx + np.arange(frames)) % d
            delayed = buf[read_idx].copy()
            out = -ap + delayed
            buf[read_idx] = ap + delayed * 0.5
            self.ap_idx[i] = (idx + frames) % d
            ap = out
        mix = self.mix
        return dry * (1.0 - mix) + ap * mix

    def reset(self):
        for b in self.comb_bufs:
            b[:] = 0.0
        for b in self.ap_bufs:
            b[:] = 0.0
        self.comb_idx = [0] * 4
        self.ap_idx = [0, 0]


# ============================================================
# AudioEngine (بدون Reverse)
# ============================================================
class AudioEngine:
    def __init__(self, samplerate=SR_DEFAULT, blocksize=BLOCK,
                 performance_mode="medium"):
        self.samplerate = samplerate
        self.performance_mode = performance_mode
        if performance_mode == "low":
            self.blocksize = 4096
        elif performance_mode == "high":
            self.blocksize = 1024
        else:
            self.blocksize = blocksize
        self._source = None
        self._n_frames = 0
        self._pos = 0
        self._frac = 0.0
        self._playing = False
        self._stopped = True
        self._lock = threading.Lock()
        self._t_speed = 1.0
        self._t_pitch = 0.0
        self._t_echo = 0.0
        self._t_volume = 0.8
        self._t_normalize = False
        self._t_limiter = True
        self._t_bass = 0.0
        self._t_treble = 0.0
        self._s_speed = 1.0
        self._s_pitch = 0.0
        self._s_echo = 0.0
        self._s_volume = 0.8
        self._s_bass = 0.0
        self._s_treble = 0.0
        self._alpha = 1.0 - np.exp(-1.0 / (0.003 * samplerate))
        self._log1p_neg_alpha = math.log1p(-self._alpha) if self._alpha < 0.99 else -10.0
        self._echo_buf = np.zeros((int(samplerate * 0.3), 2), dtype=np.float32)
        self._echo_idx = 0
        self._fade = 0.0
        self._fade_target = 0.0
        self._fade_step = 1.0 / (0.010 * samplerate)
        self._stream = None
        self._end_of_track_cb = None
        self.eq = EQ5Band(samplerate)
        self.reverb = Reverb(samplerate)
        self.bass_filter = Biquad(samplerate)
        self.treble_filter = Biquad(samplerate)
        self._bass_dirty = True
        self._treble_dirty = True
        self._eq_live_enabled = (performance_mode != "low")

    def set_performance_mode(self, mode):
        self.performance_mode = mode
        if mode == "low":
            self.blocksize = 4096
            self._eq_live_enabled = False
        elif mode == "high":
            self.blocksize = 1024
            self._eq_live_enabled = True
        else:
            self.blocksize = 2048
            self._eq_live_enabled = True

    def set_end_of_track_callback(self, cb):
        self._end_of_track_cb = cb

    def set_source(self, samples, samplerate):
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
            if self._t_normalize:
                peak = np.max(np.abs(samples))
                if peak > 1e-6:
                    samples = samples / peak * 0.95
            self._source = samples
            self._n_frames = samples.shape[0]
            self._pos = 0
            self._frac = 0.0
            self._echo_buf[:] = 0.0
            self._echo_idx = 0
            self._fade = 0.0
            self._fade_target = 0.0
            self._s_speed = self._t_speed
            self._s_pitch = self._t_pitch
            self._s_echo = self._t_echo
            self._s_volume = self._t_volume
            self.reverb.reset()
            self._stopped = True

    def clear(self):
        with self._lock:
            self._source = None
            self._n_frames = 0
            self._pos = 0
            self._frac = 0.0
            self._playing = False
            self._stopped = True

    def play(self):
        with self._lock:
            if self._source is None:
                return
            if self._playing:
                return
            self._playing = True
            self._stopped = False
            self._fade_target = 1.0
        self._ensure_stream_started()

    def pause(self):
        with self._lock:
            if not self._playing:
                return
            self._fade_target = 0.0
            self._playing = False
            self._stopped = True

    def stop(self):
        self._close_stream()
        with self._lock:
            self._playing = False
            self._stopped = True
            self._pos = 0
            self._frac = 0.0
            self._fade = 0.0
            self._fade_target = 0.0

    def seek(self, frame_index):
        with self._lock:
            if self._source is None:
                return
            self._pos = max(0, min(int(frame_index), self._n_frames - 1))
            self._frac = 0.0

    def position_seconds(self):
        with self._lock:
            return self._pos / float(self.samplerate)

    def duration_seconds(self):
        with self._lock:
            if self._source is None:
                return 0.0
            return self._n_frames / float(self.samplerate)

    def is_playing(self):
        with self._lock:
            return self._playing

    def set_speed(self, v): self._t_speed = max(0.25, min(4.0, float(v)))
    def set_pitch(self, s): self._t_pitch = max(-12.0, min(12.0, float(s)))
    def set_echo(self, v): self._t_echo = max(0.0, min(1.0, float(v)))
    def set_volume(self, v): self._t_volume = max(0.0, min(1.0, float(v)))
    def set_bass(self, db):
        self._t_bass = max(-12.0, min(12.0, float(db)))
        self._bass_dirty = True
    def set_treble(self, db):
        self._t_treble = max(-12.0, min(12.0, float(db)))
        self._treble_dirty = True
    def set_normalize(self, on): self._t_normalize = bool(on)
    def set_limiter(self, on): self._t_limiter = bool(on)
    def set_eq_gains(self, gains): self.eq.set_gains(gains)
    def set_reverb(self, room, damp, mix): self.reverb.set_params(room, damp, mix)

    def _ensure_stream_started(self):
        if self._stream is not None:
            return
        try:
            latency = 'high' if self.performance_mode == 'low' else 'low'
            self._stream = sd.OutputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=2,
                dtype='float32',
                callback=self._callback,
                latency=latency,
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

    def _callback(self, outdata, frames, time_info, status):
        try:
            src = self._source
            if src is None or self._stopped:
                outdata.fill(0.0)
                return
            a = self._alpha
            decay = math.exp(frames * self._log1p_neg_alpha) if a < 0.99 else 0.0
            self._s_speed = self._t_speed + (self._s_speed - self._t_speed) * decay
            self._s_pitch = self._t_pitch + (self._s_pitch - self._t_pitch) * decay
            self._s_echo = self._t_echo + (self._s_echo - self._t_echo) * decay
            self._s_volume = self._t_volume + (self._s_volume - self._t_volume) * decay
            speed = self._s_speed
            vol = self._s_volume
            echo = self._s_echo
            pitch = self._s_pitch
            fade = self._fade
            fade_t = self._fade_target
            step = self._fade_step
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
                        if not self._stopped:
                            self._playing = False
                            self._pos = 0
                            self._frac = 0.0
                    if self._end_of_track_cb:
                        cb = self._end_of_track_cb
                        threading.Thread(target=cb, daemon=True).start()
                    return
                idx_valid = idx[:k]
                block = np.zeros((frames, 2), dtype=np.float32)
                block[:k] = self._interp(src, idx_valid)
                end_reached = True
            else:
                block = self._interp(src, idx)

            if echo > 1e-4:
                eb = self._echo_buf
                L = eb.shape[0]
                read_idx = (self._echo_idx + np.arange(frames)) % L
                delayed = eb[read_idx].copy()
                dry = block.copy()
                block = block + echo * delayed
                eb[read_idx] = dry
                self._echo_idx = (self._echo_idx + frames) % L

            if self._eq_live_enabled and self.eq.is_enabled():
                block = self.eq.process(block)

            if self._eq_live_enabled and (abs(self._t_bass) > 0.01 or abs(self._t_treble) > 0.01):
                if self._bass_dirty:
                    self.bass_filter.set_lowshelf(200.0, self._t_bass)
                    self._bass_dirty = False
                if self._treble_dirty:
                    self.treble_filter.set_highshelf(4000.0, self._t_treble)
                    self._treble_dirty = False
                if abs(self._t_bass) > 0.01:
                    block = self.bass_filter.process(block)
                if abs(self._t_treble) > 0.01:
                    block = self.treble_filter.process(block)

            if self._eq_live_enabled and self.reverb.is_enabled():
                block = self.reverb.process(block)

            block *= vol

            if self._t_limiter:
                np.clip(block, -1.0, 1.0, out=block)

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

            np.clip(block, -1.0, 1.0, out=block)
            outdata[:] = block

            if end_reached:
                with self._lock:
                    if not self._stopped:
                        self._playing = False
                        self._pos = 0
                        self._frac = 0.0
                if self._end_of_track_cb:
                    cb = self._end_of_track_cb
                    threading.Thread(target=cb, daemon=True).start()
            else:
                last = idx[-1]
                with self._lock:
                    if not self._stopped:
                        self._pos = int(last)
                        self._frac = float(last - int(last))
        except Exception:
            traceback.print_exc()
            outdata.fill(0.0)

    @staticmethod
    def _interp(src, idx):
        n = src.shape[0]
        idx = np.clip(idx, 0.0, n - 1.001)
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        f = (idx - i0).astype(np.float32)[:, None]
        return (src[i0] * (1.0 - f) + src[i1] * f).astype(np.float32)


class AudioLoader(QObject):
    finished = Signal(object, int, str)
    failed = Signal(str)

    def load(self, path):
        def run():
            try:
                data, sr = sf.read(path, dtype='float32', always_2d=False)
                self.finished.emit(data, sr, path)
                return
            except Exception:
                pass
            try:
                if FFMPEG_PATH is None:
                    self.failed.emit("This format needs ffmpeg, but it's not found.")
                    return
                import subprocess
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                tmp_path = tmp.name
                cmd = [
                    FFMPEG_PATH, "-y", "-i", path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", str(SR_DEFAULT), "-ac", "2", tmp_path,
                ]
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=0x08000000,
                )
                if result.returncode != 0:
                    err = result.stderr.decode('utf-8', errors='ignore')
                    self.failed.emit("ffmpeg failed: " + err[-300:])
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    return
                data, sr = sf.read(tmp_path, dtype='float32', always_2d=False)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                self.finished.emit(data, sr, path)
            except Exception as e:
                self.failed.emit(str(e))
        threading.Thread(target=run, daemon=True).start()


class AudioSaver(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def save_processed(self, path, samples, samplerate, ext="wav",
                       speed=1.0, pitch=0.0, echo=0.0,
                       eq_gains=None, reverb=None, bass=0.0, treble=0.0,
                       normalize=False, limiter=True,
                       bitrate="192k", parallel=True):
        def run():
            try:
                base = samples.astype(np.float32).copy()
                if base.ndim == 1:
                    base = np.stack([base, base], axis=1)
                if normalize:
                    peak = np.max(np.abs(base))
                    if peak > 1e-6:
                        base = base / peak * 0.95
                if eq_gains is not None:
                    eq = EQ5Band(samplerate)
                    eq.set_gains(eq_gains)
                    if eq.is_enabled():
                        base = eq.process(base)
                if abs(bass) > 0.01:
                    bq = Biquad(samplerate)
                    bq.set_lowshelf(200.0, bass)
                    base = bq.process(base)
                if abs(treble) > 0.01:
                    bq = Biquad(samplerate)
                    bq.set_highshelf(4000.0, treble)
                    base = bq.process(base)
                if reverb is not None and reverb.get("mix", 0.0) > 1e-4:
                    rv = Reverb(samplerate)
                    rv.set_params(reverb["room"], reverb["damp"], reverb["mix"])
                    base = rv.process(base)
                out = base
                n = out.shape[0]
                pitch_ratio = 2.0 ** (float(pitch) / 12.0)
                advance = float(speed) / pitch_ratio
                if abs(advance - 1.0) > 1e-6:
                    new_n = max(1, int(n / advance))
                    if parallel and new_n > 200000:
                        from concurrent.futures import ThreadPoolExecutor
                        num_threads = min(4, os.cpu_count() or 2)
                        chunk_size = max(1, new_n // num_threads)
                        out2 = np.zeros((new_n, 2), dtype=np.float32)

                        def process_range(start, end):
                            idx_l = np.arange(start, end, dtype=np.float64) * advance
                            idx_l = np.clip(idx_l, 0.0, n - 1.001)
                            i0 = np.floor(idx_l).astype(np.int64)
                            i1 = np.minimum(i0 + 1, n - 1)
                            f = (idx_l - i0).astype(np.float32)[:, None]
                            return (out[i0] * (1 - f) + out[i1] * f).astype(np.float32)

                        with ThreadPoolExecutor(max_workers=num_threads) as ex:
                            futures = []
                            for t in range(num_threads):
                                s = t * chunk_size
                                e = new_n if t == num_threads - 1 else (t + 1) * chunk_size
                                futures.append((s, e, ex.submit(process_range, s, e)))
                            for s, e, fut in futures:
                                out2[s:e] = fut.result()
                        out = out2
                    else:
                        idx = np.arange(new_n, dtype=np.float64) * advance
                        idx = np.clip(idx, 0.0, n - 1.001)
                        i0 = np.floor(idx).astype(np.int64)
                        i1 = np.minimum(i0 + 1, n - 1)
                        f = (idx - i0).astype(np.float32)[:, None]
                        out = (out[i0] * (1 - f) + out[i1] * f).astype(np.float32)
                if echo and echo > 1e-4:
                    L = int(samplerate * 0.3)
                    buf = np.zeros((L, 2), dtype=np.float32)
                    out2 = out.copy()
                    step = 8192
                    for i in range(0, out.shape[0], step):
                        chunk = out[i:i+step]
                        m = chunk.shape[0]
                        read_idx = (i + np.arange(m)) % L
                        delayed = buf[read_idx].copy()
                        mixed = chunk + float(echo) * delayed
                        buf[read_idx] = chunk
                        out2[i:i+m] = mixed
                    out = out2
                if limiter:
                    out = np.tanh(out * 1.2) * 0.95
                np.clip(out, -1.0, 1.0, out=out)
                if ext == "wav":
                    sf.write(path, out, samplerate, subtype='PCM_16')
                else:
                    if FFMPEG_PATH is None:
                        self.failed.emit("ffmpeg not found.")
                        return
                    from pydub import AudioSegment
                    AudioSegment.converter = FFMPEG_PATH
                    pcm = (out * 32767.0).astype(np.int16)
                    seg = AudioSegment(pcm.tobytes(), frame_rate=samplerate,
                                       sample_width=2, channels=2)
                    if ext == "mp3":
                        seg.export(path, format="mp3", bitrate=bitrate)
                    elif ext == "mp4":
                        seg.export(path, format="mp4", bitrate=bitrate, codec="aac")
                del out
                self.finished.emit(path)
            except Exception as e:
                self.failed.emit(str(e))
        threading.Thread(target=run, daemon=True).start()


# ============================================================
# THEMES
# ============================================================
THEMES = {
    "dark": {
        "bg": "#14100c", "panel": "#1e1812", "panel2": "#2a221a",
        "text": "#f5e8d8", "text_dim": "#a89880",
        "accent": "#e89040", "accent2": "#c87020",
        "border": "#3a2e20", "slider": "#e89040", "sel": "#3a2e20",
        "stars": True, "star_color": (255, 220, 180), "nebula": None, "density": 60,
    },
    "light": {
        "bg": "#faf6f0", "panel": "#ffffff", "panel2": "#f0e8dc",
        "text": "#2a2018", "text_dim": "#7a6a58",
        "accent": "#d07828", "accent2": "#b06018",
        "border": "#e0d4c4", "slider": "#d07828", "sel": "#f0e0c8",
        "stars": False, "star_color": (0, 0, 0), "nebula": None, "density": 0,
    },
    "black": {
        "bg": "#000000", "panel": "#0a0a0a", "panel2": "#121212",
        "text": "#f0f0f0", "text_dim": "#7a7a7a",
        "accent": "#e89040", "accent2": "#c87020",
        "border": "#1a1a1a", "slider": "#e89040", "sel": "#1a1a1a",
        "stars": True, "star_color": (255, 255, 255), "nebula": None, "density": 80,
    },
    "blue": {
        "bg": "#0a1420", "panel": "#0f1e2e", "panel2": "#162a3d",
        "text": "#d8e8f5", "text_dim": "#6a8ba0",
        "accent": "#4a9eff", "accent2": "#2a7ecc",
        "border": "#1e3548", "slider": "#4a9eff", "sel": "#1e3548",
        "stars": True, "star_color": (200, 220, 255), "nebula": (40, 80, 140), "density": 70,
    },
    "galaxy": {
        "bg": "#0d0820", "panel": "#1a0f38", "panel2": "#241650",
        "text": "#e8d8f5", "text_dim": "#8a7aa0",
        "accent": "#b878ff", "accent2": "#9050dd",
        "border": "#2e1e5a", "slider": "#b878ff", "sel": "#2e1e5a",
        "stars": True, "star_color": (220, 180, 255), "nebula": (120, 40, 180), "density": 90,
    },
}


class StarfieldWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.stars = []
        self.shooting = []
        self.star_color = (255, 255, 255)
        self.nebula = None
        self._enabled = False
        self._density = 80
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_anim)
        self._timer.start(60)
        self._regen_needed = True

    def enable(self, on=True, star_color=(255, 255, 255), nebula=None, density=80):
        self._enabled = on
        self.star_color = star_color
        self.nebula = nebula
        self._density = density
        self._regen_needed = True
        self._generate_stars()
        self.raise_()
        self.update()

    def resizeEvent(self, e):
        self._regen_needed = True
        self._generate_stars()
        super().resizeEvent(e)

    def _generate_stars(self):
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        self.stars = []
        for _ in range(self._density):
            self.stars.append({
                'x': random.randint(0, w),
                'y': random.randint(0, h),
                'r': random.choice([1, 1, 1, 2]),
                'phase': random.uniform(0, 6.28),
                'speed': random.uniform(0.05, 0.15),
            })
        self._regen_needed = False

    def _update_anim(self):
        if not self._enabled:
            return
        if self._regen_needed:
            self._generate_stars()
        for s in self.stars:
            s['phase'] += s['speed']
        if random.random() < 0.02:
            self.shooting.append({
                'x': random.randint(0, max(1, self.width())),
                'y': random.randint(0, max(1, self.height() // 2)),
                'vx': random.uniform(6, 10),
                'vy': random.uniform(3, 5),
                'life': 60,
            })
        for sh in self.shooting[:]:
            sh['x'] += sh['vx']
            sh['y'] += sh['vy']
            sh['life'] -= 1
            if sh['life'] <= 0:
                self.shooting.remove(sh)
        self.update()

    def paintEvent(self, e):
        if not self._enabled:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.nebula:
            r, g, b = self.nebula
            for i in range(3):
                alpha = 15 - i * 4
                p.setBrush(QBrush(QColor(r, g, b, alpha)))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(w * 0.3 + i * 40, h * 0.4 + i * 30),
                              w * 0.35, h * 0.3)
        sr, sg, sb = self.star_color
        p.setPen(Qt.NoPen)
        for s in self.stars:
            alpha = int(100 + 155 * (0.5 + 0.5 * math.sin(s['phase'])))
            p.setBrush(QBrush(QColor(sr, sg, sb, alpha)))
            p.drawEllipse(QPointF(s['x'], s['y']), s['r'], s['r'])
        for sh in self.shooting:
            alpha = min(255, sh['life'] * 4)
            p.setPen(QColor(255, 220, 150, alpha))
            p.drawLine(int(sh['x']), int(sh['y']),
                       int(sh['x'] - sh['vx'] * 5), int(sh['y'] - sh['vy'] * 5))
        p.end()


class WaveformWidget(QWidget):
    seek_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self._samples = None
        self._progress = 0.0
        self._peaks = None
        self._accent = QColor(232, 144, 64)
        self._bg = QColor(30, 24, 18)
        self._played = QColor(200, 112, 32)
        self._enabled = True

    def set_enabled_render(self, on):
        self._enabled = on
        self.update()

    def set_samples(self, samples):
        self._samples = samples
        self._compute_peaks()
        self.update()

    def clear(self):
        self._samples = None
        self._peaks = None
        self.update()

    def set_progress(self, frac):
        self._progress = max(0.0, min(1.0, frac))
        self.update()

    def set_colors(self, accent, bg):
        self._accent = QColor(accent)
        self._bg = QColor(bg)
        self._played = QColor(accent).darker(120)
        self.update()

    def _compute_peaks(self):
        if self._samples is None:
            return
        w = max(1, self.width())
        n = self._samples.shape[0]
        if n == 0:
            self._peaks = None
            return
        mono = np.mean(np.abs(self._samples), axis=1) if self._samples.ndim == 2 else np.abs(self._samples)
        step = max(1, n // w)
        peaks = np.zeros(w, dtype=np.float32)
        for i in range(w):
            s = i * step
            e = min(n, s + step)
            if e > s:
                peaks[i] = np.max(mono[s:e])
        m = np.max(peaks) if peaks.size else 1.0
        if m > 1e-6:
            peaks = peaks / m
        self._peaks = peaks

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._compute_peaks()

    def mousePressEvent(self, e):
        if self.width() > 0:
            frac = e.position().x() / self.width()
            self.seek_requested.emit(frac)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self._bg)
        if not self._enabled:
            p.setPen(QPen(QColor(120, 120, 120)))
            p.drawText(0, 0, w, h, Qt.AlignCenter, "📊  Waveform disabled (Performance Mode)")
            return
        if self._peaks is None or self._peaks.size == 0:
            p.setPen(QPen(QColor(120, 120, 120)))
            p.drawText(0, 0, w, h, Qt.AlignCenter, "🎵  Load a file to see waveform")
            return
        peaks = self._peaks
        mid = h // 2
        bar_w = max(1, w // peaks.size)
        progress_x = int(self._progress * w)
        for i, peak in enumerate(peaks):
            x = int(i * w / peaks.size)
            bh = int(peak * (h * 0.45))
            if bh < 1:
                bh = 1
            color = self._played if x < progress_x else self._accent
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRect(x, mid - bh, bar_w, bh * 2)
        p.setPen(QPen(QColor(255, 255, 255, 200), 2))
        p.drawLine(progress_x, 0, progress_x, h)
        p.end()


class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None, expanded=True):
        super().__init__(parent)
        self._title = title
        self._expanded = expanded
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.header = QToolButton()
        self.header.setText(("▼  " if expanded else "▶  ") + title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setObjectName("sectionHeader")
        self.header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setMinimumHeight(34)
        self.header.clicked.connect(self._toggle)
        layout.addWidget(self.header)

        self.content = QWidget()
        self.content.setObjectName("sectionContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 8, 12, 12)
        self.content_layout.setSpacing(8)
        self.content.setVisible(expanded)
        layout.addWidget(self.content)

    def _toggle(self):
        self._expanded = not self._expanded
        self.header.setText(("▼  " if self._expanded else "▶  ") + self._title)
        self.content.setVisible(self._expanded)

    def add_widget(self, w):
        self.content_layout.addWidget(w)

    def add_layout(self, l):
        self.content_layout.addLayout(l)

    def set_expanded(self, on):
        self._expanded = on
        self.header.setChecked(on)
        self.header.setText(("▼  " if on else "▶  ") + self._title)
        self.content.setVisible(on)


# ============================================================
# MainWindow
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 AhmedAudioPlayer — V7.3")
        self.resize(1280, 800)
        self.setMinimumSize(1000, 650)
        self.setAcceptDrops(True)
        self.starfield = None
        self.performance_mode = auto_detect_performance()
        self.engine = AudioEngine(performance_mode=self.performance_mode)
        self.engine.set_end_of_track_callback(self._on_track_finished)
        self.loader = AudioLoader()
        self.saver = AudioSaver()
        self.loader.finished.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.saver.finished.connect(self._on_saved)
        self.saver.failed.connect(self._on_save_failed)
        self.current_file = None
        self.current_theme = "dark"
        self._seeking = False
        self._pending_autoplay = False
        self._all_files = []
        self._recent_files = []
        self._icon_cache = {}
        self._pending_icons = []
        self._loop_enabled = False
        self._loop_a = 0.0
        self._loop_b = 0.0
        self._icon_timer = QTimer(self)
        self._icon_timer.setInterval(80)
        self._icon_timer.timeout.connect(self._process_pending_icons)
        self._build_ui()
        saved_theme = "dark"
        saved_filter = 0
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    saved_theme = data.get("theme", "dark")
                    saved_filter = data.get("filter", 0)
                    self.performance_mode = data.get("performance", self.performance_mode)
                    self._recent_files = data.get("recent", [])
        except Exception:
            pass
        self.engine.set_performance_mode(self.performance_mode)
        perf_map = {"low": 0, "medium": 1, "high": 2}
        self.perf_box.setCurrentIndex(perf_map.get(self.performance_mode, 1))
        theme_map = {"dark": 0, "light": 1, "black": 2, "blue": 3, "galaxy": 4}
        idx = theme_map.get(saved_theme, 0)
        self.theme_box.setCurrentIndex(idx)
        self._apply_theme(saved_theme)
        self.format_filter.setCurrentIndex(saved_filter)
        self._setup_shortcuts()
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        self.starfield = StarfieldWidget(central)
        self.starfield.setGeometry(central.rect())
        self.starfield.show()
        self.starfield.lower()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(56)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(20, 0, 20, 0)
        tb.setSpacing(12)
        self.header = QLabel("🎵  AUDIO PLAYER")
        self.header.setObjectName("header")
        f = QFont("Segoe UI", 16, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 4)
        self.header.setFont(f)
        tb.addWidget(self.header)
        tb.addStretch(1)
        perf_lbl = QLabel("⚡")
        tb.addWidget(perf_lbl)
        self.perf_box = QComboBox()
        self.perf_box.addItems(["🪫  Low", "⚡  Medium", "🚀  High"])
        self.perf_box.setFixedWidth(130)
        self.perf_box.currentIndexChanged.connect(self._on_perf_changed)
        tb.addWidget(self.perf_box)
        self.btn_advanced = QPushButton("🎛️  Advanced")
        self.btn_advanced.setMinimumHeight(36)
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.clicked.connect(self._toggle_advanced)
        tb.addWidget(self.btn_advanced)
        self.theme_box = QComboBox()
        self.theme_box.addItems([
            "🌑 Dark Autumn", "☁️ Light White", "⚫ Pure Black",
            "🌊 Midnight Blue", "🌌 Purple Galaxy",
        ])
        self.theme_box.setFixedWidth(180)
        self.theme_box.currentIndexChanged.connect(self._on_theme_change)
        tb.addWidget(self.theme_box)
        root.addWidget(topbar)

        main = QWidget()
        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        left = QFrame()
        left.setObjectName("panel")
        left.setMinimumWidth(320)
        left.setMaximumWidth(440)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 14, 14, 14)
        lv.setSpacing(10)
        pl_header = QHBoxLayout()
        pl_title = QLabel("📁  Playlist")
        pl_title.setObjectName("sectionTitle")
        pl_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        pl_header.addWidget(pl_title)
        pl_header.addStretch(1)
        self.pl_count = QLabel("0 / 0")
        self.pl_count.setObjectName("badge")
        pl_header.addWidget(self.pl_count)
        lv.addLayout(pl_header)

        search_row = QHBoxLayout()
        search_row.setSpacing(4)
        search_lbl = QLabel("🔍")
        search_lbl.setFixedWidth(24)
        search_row.addWidget(search_lbl)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search... (Ctrl+L)")
        self.search_box.setFixedHeight(32)
        self.search_box.textChanged.connect(self._on_search_changed)
        self.search_box.setObjectName("searchBox")
        search_row.addWidget(self.search_box, 1)
        lv.addLayout(search_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        filter_lbl = QLabel("🎚️")
        filter_lbl.setFixedWidth(24)
        filter_row.addWidget(filter_lbl)
        self.format_filter = QComboBox()
        self.format_filter.addItems([
            "📂 All", "🎵 MP3", "🎬 MP4", "🎼 WAV", "🎸 FLAC",
            "🎧 M4A", "🌊 OGG", "🎤 AAC", "🔊 OPUS", "📼 AIFF", "🕒 Recent"
        ])
        self.format_filter.setFixedHeight(32)
        self.format_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.format_filter, 1)
        self.sort_box = QComboBox()
        self.sort_box.addItems(["⬆ Name", "⬇ Name", "📅 Newest", "📅 Oldest"])
        self.sort_box.setFixedWidth(110)
        self.sort_box.setFixedHeight(32)
        self.sort_box.currentIndexChanged.connect(self._on_sort_changed)
        filter_row.addWidget(self.sort_box)
        lv.addLayout(filter_row)

        self.playlist = QListWidget()
        self.playlist.setIconSize(QSize(40, 40))
        self.playlist.setSpacing(2)
        self.playlist.setUniformItemSizes(True)
        self.playlist.setWordWrap(False)
        self.playlist.setSelectionMode(QListWidget.ExtendedSelection)
        self.playlist.itemDoubleClicked.connect(self._on_playlist_double_click)
        lv.addWidget(self.playlist, 1)

        pl_btns = QHBoxLayout()
        pl_btns.setSpacing(6)
        self.btn_open_folder = QPushButton("📂  Open Folder")
        self.btn_save_pl = QPushButton("💾")
        self.btn_save_pl.setFixedWidth(44)
        self.btn_save_pl.setToolTip("Save playlist as m3u")
        self.btn_load_pl = QPushButton("📥")
        self.btn_load_pl.setFixedWidth(44)
        self.btn_load_pl.setToolTip("Load m3u playlist")
        self.btn_clear_pl = QPushButton("🗑")
        self.btn_clear_pl.setFixedWidth(44)
        pl_btns.addWidget(self.btn_open_folder, 1)
        pl_btns.addWidget(self.btn_save_pl)
        pl_btns.addWidget(self.btn_load_pl)
        pl_btns.addWidget(self.btn_clear_pl)
        lv.addLayout(pl_btns)
        main_layout.addWidget(left)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setObjectName("rightScroll")
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setFrameShape(QFrame.NoFrame)

        right = QWidget()
        right.setObjectName("rightPanel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 8, 0)
        rv.setSpacing(12)

        np_card = QFrame()
        np_card.setObjectName("card")
        npv = QHBoxLayout(np_card)
        npv.setContentsMargins(20, 16, 20, 16)
        npv.setSpacing(16)
        self.album_art = QLabel()
        self.album_art.setFixedSize(80, 80)
        self.album_art.setAlignment(Qt.AlignCenter)
        self.album_art.setObjectName("albumArt")
        self.album_art.setCursor(Qt.PointingHandCursor)
        self.album_art.mousePressEvent = self._on_album_art_click
        npv.addWidget(self.album_art)
        text_col = QVBoxLayout()
        text_col.setSpacing(6)
        self.file_label = QLabel("🎵  No file loaded")
        self.file_label.setObjectName("nowPlaying")
        self.file_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.file_label.setWordWrap(True)
        text_col.addWidget(self.file_label)
        self.file_info = QLabel("Choose a file or folder to start")
        self.file_info.setObjectName("fileInfo")
        text_col.addWidget(self.file_info)
        npv.addLayout(text_col, 1)
        rv.addWidget(np_card)

        self.waveform = WaveformWidget()
        self.waveform.seek_requested.connect(self._on_waveform_seek)
        rv.addWidget(self.waveform)

        tl_row = QHBoxLayout()
        tl_row.setSpacing(10)
        self.time_cur = QLabel("0:00")
        self.time_cur.setObjectName("timeLabel")
        self.time_cur.setFixedWidth(60)
        self.time_cur.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderPressed.connect(self._seek_start)
        self.timeline.sliderReleased.connect(self._seek_end)
        self.time_dur = QLabel("0:00")
        self.time_dur.setObjectName("timeLabel")
        self.time_dur.setFixedWidth(60)
        tl_row.addWidget(self.time_cur)
        tl_row.addWidget(self.timeline, 1)
        tl_row.addWidget(self.time_dur)
        rv.addLayout(tl_row)

        tc = QHBoxLayout()
        tc.setSpacing(10)
        tc.addStretch(1)
        self.btn_prev = QPushButton("⏮")
        self.btn_play = QPushButton("▶")
        self.btn_pause = QPushButton("⏸")
        self.btn_stop = QPushButton("⏹")
        self.btn_next = QPushButton("⏭")
        for b in (self.btn_prev, self.btn_play, self.btn_pause,
                  self.btn_stop, self.btn_next):
            b.setObjectName("transportBtn")
            b.setFixedSize(56, 56)
        self.btn_play.setObjectName("playBtn")
        self.btn_play.setFixedSize(72, 72)
        self.btn_play.setFont(QFont("Segoe UI", 20))
        tc.addWidget(self.btn_prev)
        tc.addWidget(self.btn_play)
        tc.addWidget(self.btn_pause)
        tc.addWidget(self.btn_stop)
        tc.addWidget(self.btn_next)
        self.btn_loop = QPushButton("🔁")
        self.btn_loop.setObjectName("transportBtn")
        self.btn_loop.setFixedSize(56, 56)
        self.btn_loop.setCheckable(True)
        self.btn_loop.clicked.connect(self._on_loop_toggle)
        tc.addWidget(self.btn_loop)
        tc.addStretch(1)
        rv.addLayout(tc)

        fa = QHBoxLayout()
        fa.setSpacing(8)
        self.btn_open = QPushButton("📂  Open Audio")
        self.btn_save_wav = QPushButton("💾  Save WAV")
        self.btn_save_mp3 = QPushButton("🎵  Export MP3")
        self.btn_save_mp4 = QPushButton("🎬  Export MP4")
        self.btn_reset_audio = QPushButton("🔄  Reset Audio")
        self.btn_reset_settings = QPushButton("⚙️  Reset")
        for b in (self.btn_open, self.btn_save_wav, self.btn_save_mp3,
                  self.btn_save_mp4, self.btn_reset_audio, self.btn_reset_settings):
            b.setMinimumHeight(40)
            fa.addWidget(b)
        rv.addLayout(fa)

        # Advanced
        self.adv_card = QFrame()
        self.adv_card.setObjectName("card")
        self.adv_card.setVisible(False)
        av = QVBoxLayout(self.adv_card)
        av.setContentsMargins(14, 14, 14, 14)
        av.setSpacing(8)
        adv_title = QLabel("🎛️  Advanced Audio")
        adv_title.setObjectName("sectionTitle")
        adv_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        av.addWidget(adv_title)

        self.eq_section = CollapsibleSection("🎚️  Equalizer (5 bands)", expanded=True)
        eq_grid = QHBoxLayout()
        eq_grid.setSpacing(6)
        self.eq_sliders = []
        eq_labels = ["60Hz", "230Hz", "910Hz", "3.6k", "14k"]
        for i, lbl in enumerate(eq_labels):
            col = QVBoxLayout()
            col.setSpacing(2)
            s = QSlider(Qt.Vertical)
            s.setRange(-12, 12)
            s.setValue(0)
            s.setFixedHeight(100)
            s.valueChanged.connect(self._on_eq_changed)
            col.addWidget(s, 0, Qt.AlignHCenter)
            l = QLabel(lbl)
            l.setAlignment(Qt.AlignCenter)
            l.setObjectName("sliderLabel")
            col.addWidget(l)
            eq_grid.addLayout(col)
            self.eq_sliders.append(s)
        self.eq_section.add_layout(eq_grid)
        eq_preset_row = QHBoxLayout()
        eq_preset_row.addWidget(QLabel("Preset:"))
        self.eq_preset = QComboBox()
        self.eq_preset.addItems(list(EQ5Band.PRESETS.keys()))
        self.eq_preset.currentIndexChanged.connect(self._on_eq_preset)
        eq_preset_row.addWidget(self.eq_preset, 1)
        self.eq_section.add_layout(eq_preset_row)
        av.addWidget(self.eq_section)

        self.bt_section = CollapsibleSection("🎵  Bass / Treble", expanded=False)
        bt_row = QHBoxLayout()
        bt_row.setSpacing(12)
        b_lbl = QLabel("Bass")
        b_lbl.setFixedWidth(60)
        bt_row.addWidget(b_lbl)
        self.sl_bass = QSlider(Qt.Horizontal)
        self.sl_bass.setRange(-12, 12)
        self.sl_bass.setValue(0)
        self.sl_bass.valueChanged.connect(self._on_bass)
        bt_row.addWidget(self.sl_bass, 1)
        self.bass_lbl = QLabel("+0 dB")
        self.bass_lbl.setFixedWidth(70)
        self.bass_lbl.setAlignment(Qt.AlignRight)
        self.bass_lbl.setObjectName("sliderValue")
        bt_row.addWidget(self.bass_lbl)
        self.bt_section.add_layout(bt_row)

        bt_row2 = QHBoxLayout()
        bt_row2.setSpacing(12)
        t_lbl = QLabel("Treble")
        t_lbl.setFixedWidth(60)
        bt_row2.addWidget(t_lbl)
        self.sl_treble = QSlider(Qt.Horizontal)
        self.sl_treble.setRange(-12, 12)
        self.sl_treble.setValue(0)
        self.sl_treble.valueChanged.connect(self._on_treble)
        bt_row2.addWidget(self.sl_treble, 1)
        self.treble_lbl = QLabel("+0 dB")
        self.treble_lbl.setFixedWidth(70)
        self.treble_lbl.setAlignment(Qt.AlignRight)
        self.treble_lbl.setObjectName("sliderValue")
        bt_row2.addWidget(self.treble_lbl)
        self.bt_section.add_layout(bt_row2)
        av.addWidget(self.bt_section)

        self.rv_section = CollapsibleSection("🔊  Reverb (Schroeder)", expanded=False)
        rv_row1 = QHBoxLayout()
        rv_row1.addWidget(QLabel("Room"))
        self.sl_reverb_room = QSlider(Qt.Horizontal)
        self.sl_reverb_room.setRange(0, 100)
        self.sl_reverb_room.setValue(50)
        self.sl_reverb_room.valueChanged.connect(self._on_reverb_changed)
        rv_row1.addWidget(self.sl_reverb_room, 1)
        self.rv_room_lbl = QLabel("50%")
        self.rv_room_lbl.setFixedWidth(50)
        self.rv_room_lbl.setAlignment(Qt.AlignRight)
        self.rv_room_lbl.setObjectName("sliderValue")
        rv_row1.addWidget(self.rv_room_lbl)
        self.rv_section.add_layout(rv_row1)

        rv_row2 = QHBoxLayout()
        rv_row2.addWidget(QLabel("Damp"))
        self.sl_reverb_damp = QSlider(Qt.Horizontal)
        self.sl_reverb_damp.setRange(0, 100)
        self.sl_reverb_damp.setValue(50)
        self.sl_reverb_damp.valueChanged.connect(self._on_reverb_changed)
        rv_row2.addWidget(self.sl_reverb_damp, 1)
        self.rv_damp_lbl = QLabel("50%")
        self.rv_damp_lbl.setFixedWidth(50)
        self.rv_damp_lbl.setAlignment(Qt.AlignRight)
        self.rv_damp_lbl.setObjectName("sliderValue")
        rv_row2.addWidget(self.rv_damp_lbl)
        self.rv_section.add_layout(rv_row2)

        rv_row3 = QHBoxLayout()
        rv_row3.addWidget(QLabel("Mix"))
        self.sl_reverb_mix = QSlider(Qt.Horizontal)
        self.sl_reverb_mix.setRange(0, 100)
        self.sl_reverb_mix.setValue(0)
        self.sl_reverb_mix.valueChanged.connect(self._on_reverb_changed)
        rv_row3.addWidget(self.sl_reverb_mix, 1)
        self.rv_mix_lbl = QLabel("0%")
        self.rv_mix_lbl.setFixedWidth(50)
        self.rv_mix_lbl.setAlignment(Qt.AlignRight)
        self.rv_mix_lbl.setObjectName("sliderValue")
        rv_row3.addWidget(self.rv_mix_lbl)
        self.rv_section.add_layout(rv_row3)
        av.addWidget(self.rv_section)

        self.out_section = CollapsibleSection("📊  Output (Normalize + Limiter)", expanded=False)
        nl_row = QHBoxLayout()
        self.chk_normalize = QCheckBox("📊  Normalize")
        self.chk_normalize.toggled.connect(self._on_normalize)
        nl_row.addWidget(self.chk_normalize)
        self.chk_limiter = QCheckBox("🚫  Limiter")
        self.chk_limiter.setChecked(True)
        self.chk_limiter.toggled.connect(self._on_limiter)
        nl_row.addWidget(self.chk_limiter)
        nl_row.addStretch(1)
        self.out_section.add_layout(nl_row)
        av.addWidget(self.out_section)

        rv.addWidget(self.adv_card)

        # Effects
        fx_card = QFrame()
        fx_card.setObjectName("card")
        fxv = QVBoxLayout(fx_card)
        fxv.setContentsMargins(20, 14, 20, 14)
        fxv.setSpacing(10)
        fx_title = QLabel("🎛️  Effects")
        fx_title.setObjectName("sectionTitle")
        fx_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        fxv.addWidget(fx_title)
        self.sl_volume = self._add_slider(fxv, "🔊  Volume", 0, 100, 80,
                                          self._on_volume,
                                          fmt=lambda v: str(v) + "%")
        self.sl_speed = self._add_slider(fxv, "⚡  Speed", 25, 400, 100,
                                         self._on_speed,
                                         fmt=lambda v: ("%.2f" % (v/100.0)) + "x")
        self.sl_pitch = self._add_slider(fxv, "🎼  Pitch", -12, 12, 0,
                                         self._on_pitch,
                                         fmt=lambda v: ("+" if v >= 0 else "") + str(v) + " st")
        self.sl_echo = self._add_slider(fxv, "🌊  Echo", 0, 100, 0,
                                        self._on_echo,
                                        fmt=lambda v: str(v) + "%")
        rv.addWidget(fx_card)

        self.status = QLabel("✅  Ready")
        self.status.setObjectName("statusBar")
        rv.addWidget(self.status)

        right_scroll.setWidget(right)
        main_layout.addWidget(right_scroll, 1)
        root.addWidget(main, 1)

        # Signals
        self.btn_open.clicked.connect(self._on_open)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.btn_save_pl.clicked.connect(self._save_playlist_m3u)
        self.btn_load_pl.clicked.connect(self._load_playlist_m3u)
        self.btn_clear_pl.clicked.connect(self._on_clear_playlist)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.btn_save_wav.clicked.connect(self._on_save_wav)
        self.btn_save_mp3.clicked.connect(self._on_save_mp3)
        self.btn_save_mp4.clicked.connect(self._on_save_mp4)
        self.btn_reset_audio.clicked.connect(self._on_reset_audio)
        self.btn_reset_settings.clicked.connect(self._on_reset_settings)

    def _toggle_advanced(self):
        self.adv_card.setVisible(self.btn_advanced.isChecked())

    def _on_perf_changed(self, idx):
        modes = ["low", "medium", "high"]
        if 0 <= idx < len(modes):
            self.performance_mode = modes[idx]
            self.engine.set_performance_mode(self.performance_mode)
            self.waveform.set_enabled_render(self.performance_mode != "low")
            density_map = {"low": 0, "medium": 40, "high": 80}
            t = THEMES.get(self.current_theme, THEMES["dark"])
            self.starfield.enable(
                t.get("stars", False) and self.performance_mode != "low",
                star_color=t.get("star_color", (255, 255, 255)),
                nebula=t.get("nebula", None),
                density=density_map[self.performance_mode],
            )
            self._save_settings()
            self.status.setText(f"⚡ Performance: {self.performance_mode.upper()}")

    def _add_slider(self, parent_layout, name, lo, hi, val, cb, fmt=None):
        row = QHBoxLayout()
        row.setSpacing(12)
        lbl = QLabel(name)
        lbl.setFixedWidth(100)
        lbl.setObjectName("sliderLabel")
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        val_lbl = QLabel(fmt(val) if fmt else str(val))
        val_lbl.setFixedWidth(70)
        val_lbl.setObjectName("sliderValue")
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

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self._on_play_pause)
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_open)
        QShortcut(QKeySequence("Ctrl+F"), self, self._on_open_folder)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save_wav)
        QShortcut(QKeySequence("Ctrl+E"), self, self._on_save_mp3)
        QShortcut(QKeySequence("Ctrl+Shift+E"), self, self._on_save_mp4)
        QShortcut(QKeySequence("Ctrl+L"), self, self._focus_search)
        QShortcut(QKeySequence("Ctrl+A"), self, self._loop_set_a)
        QShortcut(QKeySequence("Ctrl+B"), self, self._loop_set_b)
        QShortcut(QKeySequence("Left"), self, lambda: self._skip(-5))
        QShortcut(QKeySequence("Right"), self, lambda: self._skip(5))

    def _focus_search(self):
        self.search_box.setFocus()
        self.search_box.selectAll()

    def _skip(self, seconds):
        if self.engine._source is None:
            return
        cur = self.engine.position_seconds()
        target = max(0, min(cur + seconds, self.engine.duration_seconds()))
        self.engine.seek(int(target * self.engine.samplerate))

    def _on_track_finished(self):
        QTimer.singleShot(0, self._auto_next)

    def _auto_next(self):
        row = self.playlist.currentRow()
        if row < self.playlist.count() - 1:
            self._load_playlist_row(row + 1, autoplay=True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    fp = os.path.join(p, f)
                    if f.lower().endswith(AUDIO_EXTS):
                        paths.append(fp)
            elif os.path.isfile(p) and p.lower().endswith(AUDIO_EXTS):
                paths.append(p)
        if not paths:
            return
        for p in paths:
            if p not in self._all_files:
                self._all_files.append(p)
        self._rebuild_playlist()
        self.status.setText(f"Added {len(paths)} files")

    def _on_eq_changed(self):
        gains = [s.value() for s in self.eq_sliders]
        self.engine.set_eq_gains(gains)

    def _on_eq_preset(self, idx):
        name = self.eq_preset.currentText()
        if name in EQ5Band.PRESETS:
            gains = EQ5Band.PRESETS[name]
            for s, g in zip(self.eq_sliders, gains):
                s.blockSignals(True)
                s.setValue(g)
                s.blockSignals(False)
            self.engine.set_eq_gains(gains)

    def _on_bass(self, v):
        self.bass_lbl.setText(("+" if v >= 0 else "") + str(v) + " dB")
        self.engine.set_bass(v)

    def _on_treble(self, v):
        self.treble_lbl.setText(("+" if v >= 0 else "") + str(v) + " dB")
        self.engine.set_treble(v)

    def _on_reverb_changed(self):
        room = self.sl_reverb_room.value() / 100.0
        damp = self.sl_reverb_damp.value() / 100.0
        mix = self.sl_reverb_mix.value() / 100.0
        self.rv_room_lbl.setText(str(int(room*100)) + "%")
        self.rv_damp_lbl.setText(str(int(damp*100)) + "%")
        self.rv_mix_lbl.setText(str(int(mix*100)) + "%")
        self.engine.set_reverb(room, damp, mix)

    def _on_normalize(self, on):
        self.engine.set_normalize(on)
        if on and self.engine._source is not None:
            self.engine.set_source(self.engine._source, self.engine.samplerate)

    def _on_limiter(self, on):
        self.engine.set_limiter(on)

    def _loop_set_a(self):
        if self.engine._source is None:
            return
        self._loop_a = self.engine.position_seconds()
        self.status.setText(f"🔁 A = {self._fmt(self._loop_a)}")
        if self._loop_b > self._loop_a:
            self._loop_enabled = True

    def _loop_set_b(self):
        if self.engine._source is None:
            return
        self._loop_b = self.engine.position_seconds()
        self.status.setText(f"🔁 B = {self._fmt(self._loop_b)}")
        if self._loop_b > self._loop_a:
            self._loop_enabled = True

    def _on_loop_toggle(self):
        self._loop_enabled = self.btn_loop.isChecked()
        if self._loop_enabled:
            self.status.setText(f"🔁 Loop ON (A={self._fmt(self._loop_a)}, B={self._fmt(self._loop_b)})")
        else:
            self.status.setText("🔁 Loop OFF")

    def _apply_loop(self):
        if not self._loop_enabled:
            return
        if self._loop_b <= self._loop_a:
            return
        pos = self.engine.position_seconds()
        if pos >= self._loop_b:
            self.engine.seek(int(self._loop_a * self.engine.samplerate))

    def _on_waveform_seek(self, frac):
        if self.engine._source is None:
            return
        target = frac * self.engine.duration_seconds()
        self.engine.seek(int(target * self.engine.samplerate))

    def _save_playlist_m3u(self):
        if not self._all_files:
            QMessageBox.information(self, "Save Playlist", "Playlist is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Playlist", "playlist.m3u",
                                              "M3U (*.m3u);;All Files (*.*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("#EXTM3U\n")
                for fp in self._all_files:
                    f.write("#EXTINF:-1," + os.path.basename(fp) + "\n")
                    f.write(fp + "\n")
            self.status.setText("💾  Saved: " + os.path.basename(path))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_playlist_m3u(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Playlist", "",
                                              "M3U (*.m3u *.m3u8);;All Files (*.*)")
        if not path:
            return
        try:
            files = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if os.path.isfile(line):
                        files.append(line)
                    elif os.path.isfile(line.replace("/", os.sep)):
                        files.append(line.replace("/", os.sep))
            if not files:
                QMessageBox.information(self, "Playlist", "No valid files in playlist.")
                return
            self._all_files = files
            self._rebuild_playlist()
            self.status.setText(f"📥  Loaded {len(files)} tracks")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _add_recent(self, path):
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:20]
        self._save_settings()

    def _save_settings(self):
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
            data = {
                "theme": self.current_theme,
                "filter": self.format_filter.currentIndex(),
                "performance": self.performance_mode,
                "recent": self._recent_files,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _filtered_files(self):
        idx = self.format_filter.currentIndex()
        ext_map = {
            0: None, 1: (".mp3",), 2: (".mp4",), 3: (".wav",), 4: (".flac",),
            5: (".m4a",), 6: (".ogg",), 7: (".aac",), 8: (".opus",),
            9: (".aiff", ".aif"),
        }
        if idx == 10:
            files = list(self._recent_files)
        else:
            exts = ext_map.get(idx, None)
            files = self._all_files
            if exts is not None:
                files = [f for f in files if os.path.splitext(f)[1].lower() in exts]
        query = self.search_box.text().strip().lower()
        if query:
            files = [f for f in files if query in os.path.basename(f).lower()]
        sort_idx = self.sort_box.currentIndex()
        if sort_idx == 0:
            files = sorted(files, key=lambda x: os.path.basename(x).lower())
        elif sort_idx == 1:
            files = sorted(files, key=lambda x: os.path.basename(x).lower(), reverse=True)
        elif sort_idx == 2:
            files = sorted(files, key=lambda x: os.path.getmtime(x) if os.path.isfile(x) else 0, reverse=True)
        elif sort_idx == 3:
            files = sorted(files, key=lambda x: os.path.getmtime(x) if os.path.isfile(x) else 0)
        return files

    def _rebuild_playlist(self):
        files = self._filtered_files()
        self.playlist.clear()
        self._pending_icons = []
        if not files:
            item = QListWidgetItem("   📭  مفيش ملفات مطابقة")
            item.setData(Qt.UserRole, None)
            item.setSizeHint(QSize(0, 52))
            self.playlist.addItem(item)
            self._icon_timer.stop()
            self._update_pl_count()
            return
        for idx, fp in enumerate(files):
            name = os.path.basename(fp)
            if len(name) > 45:
                name = name[:42] + "..."
            item = QListWidgetItem("   " + name)
            item.setData(Qt.UserRole, fp)
            item.setSizeHint(QSize(0, 52))
            if fp in self._icon_cache:
                icon = self._icon_cache[fp]
                if icon is not None:
                    item.setIcon(icon)
            else:
                self._pending_icons.append((idx, fp))
            self.playlist.addItem(item)
        if self._pending_icons:
            self._icon_timer.start()
        self._update_pl_count()

    def _process_pending_icons(self):
        if not self._pending_icons:
            self._icon_timer.stop()
            return
        batch = self._pending_icons[:3]
        self._pending_icons = self._pending_icons[3:]
        for idx, fp in batch:
            if fp in self._icon_cache:
                icon = self._icon_cache[fp]
            else:
                icon = load_album_icon(fp, 40)
                self._icon_cache[fp] = icon
            if icon is not None and idx < self.playlist.count():
                item = self.playlist.item(idx)
                if item and item.data(Qt.UserRole) == fp:
                    item.setIcon(icon)

    def _on_filter_changed(self, idx):
        self._rebuild_playlist()
        self._save_settings()

    def _on_sort_changed(self, idx):
        self._rebuild_playlist()

    def _on_search_changed(self, text):
        self._rebuild_playlist()

    def _on_album_art_click(self, event):
        if self.current_file:
            self._show_album_art_large(self.current_file)

    def _show_album_art_large(self, path):
        art = extract_album_art(path)
        if art is None:
            QMessageBox.information(self, "Album Art", "مفيش صورة غلاف للملف ده.")
            return
        pix = QPixmap()
        pix.loadFromData(art)
        if pix.isNull():
            return
        dlg = QMessageBox(self)
        dlg.setWindowTitle("🖼️  Album Art")
        dlg.setIconPixmap(pix.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        dlg.exec()

    def _load_path_as_new(self, path):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Error", "File not found")
            return
        if path not in self._all_files:
            self._all_files.append(path)
        self._add_recent(path)
        self._rebuild_playlist()
        for i in range(self.playlist.count()):
            if self.playlist.item(i).data(Qt.UserRole) == path:
                self.playlist.setCurrentRow(i)
                self._load_playlist_row(i, autoplay=True)
                return

    def _on_play_pause(self):
        if self.engine.is_playing():
            self._on_pause()
        else:
            self._on_play()

    def _apply_theme(self, name):
        t = THEMES[name]
        self.current_theme = name
        css = (
            "QMainWindow { background: " + t['bg'] + "; color: " + t['text'] + "; "
            "font-family: 'Segoe UI','Tahoma',sans-serif; font-size: 13px; }"
            "QWidget#centralWidget { background: " + t['bg'] + "; }"
            "QFrame#topbar { background: " + t['panel'] + "; border-bottom: 1px solid " + t['border'] + "; }"
            "QFrame#panel { background: " + t['panel'] + "; border: 1px solid " + t['border'] + "; border-radius: 12px; }"
            "QFrame#card { background: " + t['panel2'] + "; border: 1px solid " + t['border'] + "; border-radius: 12px; }"
            "QLabel { background: transparent; }"
            "QLabel#header { color: " + t['accent'] + "; }"
            "QLabel#sectionTitle { color: " + t['text'] + "; }"
            "QLabel#nowPlaying { color: " + t['accent'] + "; }"
            "QLabel#fileInfo { color: " + t['text_dim'] + "; font-size: 12px; }"
            "QLabel#timeLabel { color: " + t['text'] + "; font-size: 14px; font-weight: bold; }"
            "QLabel#sliderLabel { color: " + t['text'] + "; }"
            "QLabel#sliderValue { color: " + t['accent'] + "; font-weight: bold; }"
            "QLabel#statusBar { background: " + t['panel'] + "; color: " + t['text_dim'] + "; "
            "border: 1px solid " + t['border'] + "; border-radius: 8px; padding: 8px 14px; font-size: 12px; }"
            "QLabel#badge { background: " + t['accent'] + "; color: white; "
            "border-radius: 10px; padding: 2px 10px; font-weight: bold; font-size: 11px; }"
            "QLabel#albumArt { background: " + t['panel2'] + "; border: 1px solid " + t['border'] + "; border-radius: 8px; }"
            "QScrollArea#rightScroll { background: transparent; border: none; }"
            "QWidget#rightPanel { background: transparent; }"
            "QLineEdit#searchBox { background: " + t['panel'] + "; color: " + t['text'] + "; "
            "border: 1px solid " + t['border'] + "; border-radius: 8px; padding: 6px 12px; }"
            "QLineEdit#searchBox:focus { border: 1px solid " + t['accent'] + "; }"
            "QPushButton { background: " + t['panel'] + "; color: " + t['text'] + "; "
            "border: 1px solid " + t['border'] + "; border-radius: 8px; padding: 8px 14px; }"
            "QPushButton:hover { background: " + t['accent2'] + "; color: #fff; border: 1px solid " + t['accent'] + "; }"
            "QPushButton:pressed { background: " + t['accent'] + "; color: #fff; }"
            "QPushButton:checked { background: " + t['accent'] + "; color: white; }"
            "QPushButton#transportBtn { background: " + t['panel2'] + "; color: " + t['text'] + "; "
            "border: 1px solid " + t['border'] + "; border-radius: 28px; font-size: 20px; }"
            "QPushButton#transportBtn:hover { background: " + t['accent2'] + "; color: #fff; }"
            "QPushButton#transportBtn:checked { background: " + t['accent'] + "; color: white; }"
            "QPushButton#playBtn { background: " + t['accent'] + "; color: white; "
            "border: none; border-radius: 36px; font-size: 24px; }"
            "QPushButton#playBtn:hover { background: " + t['accent2'] + "; }"
            "QToolButton#sectionHeader { background: " + t['panel'] + "; color: " + t['text'] + "; "
            "border: 1px solid " + t['border'] + "; border-radius: 8px; padding: 6px 12px; "
            "text-align: left; font-weight: bold; font-size: 13px; }"
            "QToolButton#sectionHeader:hover { background: " + t['sel'] + "; }"
            "QWidget#sectionContent { background: transparent; }"
            "QSlider::groove:horizontal { height: 6px; background: " + t['border'] + "; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: " + t['slider'] + "; width: 16px; "
            "margin: -6px 0; border-radius: 8px; }"
            "QSlider::sub-page:horizontal { background: " + t['accent'] + "; border-radius: 3px; }"
            "QSlider::groove:vertical { width: 6px; background: " + t['border'] + "; border-radius: 3px; }"
            "QSlider::handle:vertical { background: " + t['slider'] + "; height: 16px; "
            "margin: 0 -6px; border-radius: 8px; }"
            "QComboBox { background: " + t['panel'] + "; color: " + t['text'] + "; "
            "border: 1px solid " + t['border'] + "; border-radius: 8px; padding: 4px 10px; }"
            "QComboBox:hover { border: 1px solid " + t['accent'] + "; }"
            "QComboBox::drop-down { border: none; width: 20px; }"
            "QComboBox QAbstractItemView { background: " + t['panel'] + "; color: " + t['text'] + "; "
            "selection-background-color: " + t['accent'] + "; }"
            "QCheckBox { color: " + t['text'] + "; spacing: 8px; }"
            "QCheckBox::indicator { width: 18px; height: 18px; }"
            "QCheckBox::indicator:unchecked { background: " + t['panel'] + "; border: 1px solid " + t['border'] + "; border-radius: 4px; }"
            "QCheckBox::indicator:checked { background: " + t['accent'] + "; border: 1px solid " + t['accent2'] + "; border-radius: 4px; }"
            "QListWidget { background: " + t['panel2'] + "; color: " + t['text'] + "; "
            "border: 1px solid " + t['border'] + "; border-radius: 10px; padding: 4px; outline: none; }"
            "QListWidget::item { padding: 6px 8px; border-radius: 8px; margin: 2px 2px; }"
            "QListWidget::item:hover { background: " + t['sel'] + "; }"
            "QListWidget::item:selected { background: " + t['accent'] + "; color: white; }"
            "QListWidget::item:selected:!active { background: " + t['accent'] + "; color: white; }"
            "QScrollBar:vertical { background: transparent; width: 10px; border-radius: 5px; margin: 2px; }"
            "QScrollBar::handle:vertical { background: " + t['border'] + "; border-radius: 5px; min-height: 30px; }"
            "QScrollBar::handle:vertical:hover { background: " + t['accent'] + "; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
        )
        self.setStyleSheet(css)
        if hasattr(self, "waveform"):
            self.waveform.set_colors(t['accent'], t['panel2'])
        if self.starfield is not None:
            density_map = {"low": 0, "medium": 40, "high": 80}
            stars_on = t.get("stars", False) and self.performance_mode != "low"
            self.starfield.enable(
                stars_on,
                star_color=t.get("star_color", (255, 255, 255)),
                nebula=t.get("nebula", None),
                density=density_map.get(self.performance_mode, 60),
            )
            self.starfield.raise_()

    def _on_theme_change(self, idx):
        themes = ["dark", "light", "black", "blue", "galaxy"]
        if 0 <= idx < len(themes):
            self._apply_theme(themes[idx])
            self._save_settings()

    def _update_pl_count(self):
        filtered = self.playlist.count()
        if filtered == 1 and self.playlist.item(0).data(Qt.UserRole) is None:
            filtered = 0
        total = len(self._all_files)
        if total == 0:
            self.pl_count.setText("0")
        else:
            self.pl_count.setText(f"{filtered} / {total}")

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if not folder:
            return
        files = []
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(folder, name))
        if not files:
            QMessageBox.information(self, "Playlist", "No audio files in this folder.")
            return
        self._all_files = files
        self._icon_cache.clear()
        self.format_filter.setCurrentIndex(0)
        self.search_box.clear()
        self._rebuild_playlist()
        self.status.setText("Loaded " + str(len(files)) + " tracks")

    def _on_clear_playlist(self):
        self.playlist.clear()
        self._all_files = []
        self._icon_cache.clear()
        self._pending_icons = []
        self._icon_timer.stop()
        self._update_pl_count()
        self.status.setText("Playlist cleared")

    def _on_playlist_double_click(self, item):
        if item.data(Qt.UserRole) is None:
            return
        row = self.playlist.row(item)
        self._load_playlist_row(row, autoplay=True)

    def _load_playlist_row(self, row, autoplay=False):
        if row < 0 or row >= self.playlist.count():
            return
        item = self.playlist.item(row)
        path = item.data(Qt.UserRole)
        if path is None:
            return
        self.playlist.setCurrentRow(row)
        self._pending_autoplay = autoplay
        self.status.setText("Loading " + os.path.basename(path) + "...")
        self.engine.stop()
        self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.time_cur.setText("0:00")
        self.loader.load(path)

    def _on_prev(self):
        row = self.playlist.currentRow()
        if row > 0:
            self._load_playlist_row(row - 1, autoplay=True)

    def _on_next(self):
        row = self.playlist.currentRow()
        if row < self.playlist.count() - 1:
            self._load_playlist_row(row + 1, autoplay=True)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "",
            "Audio Files (*.wav *.flac *.ogg *.mp3 *.m4a *.mp4 *.aiff *.aif *.aac *.opus);;All Files (*.*)"
        )
        if not path:
            return
        if path not in self._all_files:
            self._all_files.append(path)
        self._add_recent(path)
        self._rebuild_playlist()
        for i in range(self.playlist.count()):
            if self.playlist.item(i).data(Qt.UserRole) == path:
                self.playlist.setCurrentRow(i)
                self._load_playlist_row(i, autoplay=False)
                return
        self.format_filter.setCurrentIndex(0)
        self.search_box.clear()
        for i in range(self.playlist.count()):
            if self.playlist.item(i).data(Qt.UserRole) == path:
                self.playlist.setCurrentRow(i)
                self._load_playlist_row(i, autoplay=False)
                return

    def _on_loaded(self, data, sr, path):
        self.engine.set_source(data, sr)
        self.current_file = path
        self.file_label.setText("🎵  " + os.path.basename(path))
        pix = load_album_pixmap(path, 80)
        if pix is not None:
            self.album_art.setPixmap(pix)
            self.album_art.setText("")
        else:
            self.album_art.setPixmap(QPixmap())
            self.album_art.setText("🎵")
            self.album_art.setFont(QFont("Segoe UI", 32))
        dur = self.engine.duration_seconds()
        self.time_dur.setText(self._fmt(dur))
        self.file_info.setText(
            "⏱ " + self._fmt(dur) +
            "  •  🎚 " + str(self.engine.samplerate) + " Hz" +
            "  •  📼 " + os.path.splitext(path)[1].upper().replace(".", "")
        )
        self.status.setText("✅  Loaded: " + os.path.basename(path))
        self.waveform.set_samples(self.engine._source)
        self.engine.seek(0)
        self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.time_cur.setText(self._fmt(self.engine.position_seconds()))
        if self._pending_autoplay:
            self._pending_autoplay = False
            self.engine.play()

    def _on_load_failed(self, msg):
        self.status.setText("❌  Load failed")
        QMessageBox.critical(self, "Load error", msg)

    def _on_play(self):
        if self.engine._source is None:
            return
        self.engine.play()

    def _on_pause(self):
        self.engine.pause()

    def _on_stop(self):
        self.engine.stop()
        self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.time_cur.setText(self._fmt(self.engine.position_seconds()))
        self.status.setText("⏹  Stopped")

    def _on_save_wav(self):
        if self.engine._source is None:
            QMessageBox.information(self, "Save", "No audio loaded.")
            return
        default_name = "output.wav"
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = base + "_out.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Save WAV", default_name, "WAV (*.wav)")
        if not path:
            return
        self.status.setText("💾  Saving WAV with effects...")
        self.saver.save_processed(
            path, self.engine._source, self.engine.samplerate,
            ext="wav",
            speed=self.engine._t_speed,
            pitch=self.engine._t_pitch,
            echo=self.engine._t_echo,
            eq_gains=[s.value() for s in self.eq_sliders],
            reverb={"room": self.sl_reverb_room.value()/100.0,
                    "damp": self.sl_reverb_damp.value()/100.0,
                    "mix": self.sl_reverb_mix.value()/100.0},
            bass=self.sl_bass.value(),
            treble=self.sl_treble.value(),
            normalize=self.chk_normalize.isChecked(),
            limiter=self.chk_limiter.isChecked(),
            parallel=True,
        )

    def _on_save_mp3(self):
        if self.engine._source is None:
            QMessageBox.information(self, "Export MP3", "No audio loaded.")
            return
        default_name = "output.mp3"
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = base + "_out.mp3"
        path, _ = QFileDialog.getSaveFileName(self, "Export MP3", default_name, "MP3 (*.mp3)")
        if not path:
            return
        self.status.setText("🎵  Exporting MP3 with effects...")
        self.saver.save_processed(
            path, self.engine._source, self.engine.samplerate,
            ext="mp3",
            speed=self.engine._t_speed,
            pitch=self.engine._t_pitch,
            echo=self.engine._t_echo,
            eq_gains=[s.value() for s in self.eq_sliders],
            reverb={"room": self.sl_reverb_room.value()/100.0,
                    "damp": self.sl_reverb_damp.value()/100.0,
                    "mix": self.sl_reverb_mix.value()/100.0},
            bass=self.sl_bass.value(),
            treble=self.sl_treble.value(),
            normalize=self.chk_normalize.isChecked(),
            limiter=self.chk_limiter.isChecked(),
            bitrate="192k",
            parallel=True,
        )

    def _on_save_mp4(self):
        if self.engine._source is None:
            QMessageBox.information(self, "Export MP4", "No audio loaded.")
            return
        default_name = "output.mp4"
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = base + "_out.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Export MP4", default_name, "MP4 (*.mp4)")
        if not path:
            return
        self.status.setText("🎬  Exporting MP4 with effects...")
        self.saver.save_processed(
            path, self.engine._source, self.engine.samplerate,
            ext="mp4",
            speed=self.engine._t_speed,
            pitch=self.engine._t_pitch,
            echo=self.engine._t_echo,
            eq_gains=[s.value() for s in self.eq_sliders],
            reverb={"room": self.sl_reverb_room.value()/100.0,
                    "damp": self.sl_reverb_damp.value()/100.0,
                    "mix": self.sl_reverb_mix.value()/100.0},
            bass=self.sl_bass.value(),
            treble=self.sl_treble.value(),
            normalize=self.chk_normalize.isChecked(),
            limiter=self.chk_limiter.isChecked(),
            bitrate="192k",
            parallel=True,
        )

    def _on_saved(self, path):
        self.status.setText("✅  Saved: " + os.path.basename(path))

    def _on_save_failed(self, msg):
        self.status.setText("❌  Save failed")
        QMessageBox.critical(self, "Save error", msg)

    def _on_reset_audio(self):
        self.engine.stop()
        self.engine.seek(0)
        self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.time_cur.setText(self._fmt(self.engine.position_seconds()))
        self.status.setText("🔄  Audio reset")

    def _on_reset_settings(self):
        self.sl_volume.setValue(80)
        self.sl_speed.setValue(100)
        self.sl_pitch.setValue(0)
        self.sl_echo.setValue(0)
        for s in self.eq_sliders:
            s.setValue(0)
        self.sl_bass.setValue(0)
        self.sl_treble.setValue(0)
        self.sl_reverb_room.setValue(50)
        self.sl_reverb_damp.setValue(50)
        self.sl_reverb_mix.setValue(0)
        self.chk_normalize.setChecked(False)
        self.chk_limiter.setChecked(True)
        self.status.setText("⚙️  Settings reset")

    def _on_volume(self, v):
        self.engine.set_volume(v / 100.0)

    def _on_speed(self, v):
        self.engine.set_speed(v / 100.0)

    def _on_pitch(self, v):
        self.engine.set_pitch(v)

    def _on_echo(self, v):
        self.engine.set_echo(v / 100.0)

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
            frac = pos / dur
            self.timeline.blockSignals(True)
            self.timeline.setValue(int(frac * 1000))
            self.timeline.blockSignals(False)
            self.waveform.set_progress(frac)
        if self.engine.is_playing():
            self.btn_play.setText("⏸")
        else:
            self.btn_play.setText("▶")
        self._apply_loop()

    @staticmethod
    def _fmt(sec):
        sec = max(0, int(sec))
        return str(sec // 60) + ":" + str(sec % 60).zfill(2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.starfield is not None:
            central = self.centralWidget()
            if central is not None:
                self.starfield.setGeometry(central.rect())

    def closeEvent(self, e):
        self.engine.stop()
        e.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.startswith("ahmed-player://") or "url=" in arg:
            import urllib.parse
            try:
                parsed = urllib.parse.urlparse(arg)
                qs = urllib.parse.parse_qs(parsed.query)
                if "url" in qs:
                    url = qs["url"][0]
                    QTimer.singleShot(800, lambda: w._open_from_url(url))
            except Exception as e:
                print("URL parse error:", e)
        elif os.path.isfile(arg):
            QTimer.singleShot(800, lambda: w._load_path_as_new(arg))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()