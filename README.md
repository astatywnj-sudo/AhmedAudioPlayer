# 🎵 AhmedAudioPlayer

مشغل صوتيات احترافي للويندوز — مكتوب بـ Python + PySide6.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PySide6](https://img.shields.io/badge/PySide6-6.11-green)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)

---

## ✨ المميزات

- 🎵 يدعم 9 صيغ: MP3, MP4, WAV, FLAC, OGG, M4A, AAC, OPUS, AIFF
- ⏯ Play / Pause / Stop / Prev / Next
- ⏱ Timeline + Seek
- 🔊 Volume / ⚡ Speed / 🎼 Pitch / 🌊 Echo
- 🎚️ Equalizer (5 bands) + Presets
- 🔊 Reverb (Schroeder)
- 🎵 Bass / Treble
- 📊 Waveform Display
- 📁 Playlist + Filter + Search + Sort
- 💾 Export WAV / MP3 / MP4 (مع التأثيرات)
- 🖼️ Album Art
- 🎨 5 ثيمات (Dark / Light / Black / Blue / Galaxy)
- ⭐ نجوم + شهب (في الثيمات الليلية)
- 🔁 Loop / A-B Repeat
- 🖱️ Drag & Drop
- 💾 حفظ Playlist (m3u)
- ⚡ Performance Modes (Low / Medium / High)
- 🎯 Open with (من قائمة ويندوز)
- 📦 exe مستقل بأيقونة مخصصة
- ✅ يشتغل بدون إنترنت
- 🚫 بدون إعلانات

---

## 📋 المتطلبات

- Python 3.13+
- Windows 10/11
- `ffmpeg.exe` (لقراءة صيغ MP4 والمزيد)

---

## 🚀 التشغيل

1. حمل `ffmpeg.exe` من صفحة الـ **Releases** وحطه جنب `main.py`.
2. نصب المتطلبات:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
