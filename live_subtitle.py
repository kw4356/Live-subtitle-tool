import os
import sys
import re
import time
import queue
import threading
import subprocess
import tempfile
import wave
from collections import deque
import opencc     # ⚡ Import OpenCC (Simplified to Traditional Chinese conversion package)
import webrtcvad  # ⚡ Import Google webrtcvad (Microsecond low-latency VAD)

# ----------------------------------------------------------------------
# 1. Fix OpenMP Conflict (Fixes OMP: Error #15)
# ----------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------------------------------------------------
# 2. Import Main Packages
# ----------------------------------------------------------------------
import numpy as np
import pyaudiowpatch as pyaudio

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QEvent
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QTextEdit, 
                             QDoubleSpinBox, QSpinBox)
from PyQt6.QtGui import QMouseEvent

from llama_cpp import Llama


# ----------------------------------------------------------------------
# 3. Menu & Color Maps
# ----------------------------------------------------------------------
ASR_LANG_MAP = {
    "Auto Detect": None,
    "Mandarin": "zh",
    "Cantonese": "yue",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "German": "de",
    "Portuguese": "pt"
}

TRANS_LANG_MAP = {
    "Original Only": None,  # Do not translate
    "Traditional Chinese": "Traditional Chinese (繁體中文)",
    "Simplified Chinese": "Simplified Chinese (簡體中文)",
    "English": "English",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "French": "French",
    "Spanish": "Spanish",
    "Italian": "Italian",
    "German": "German",
    "Portuguese": "Portuguese"
}

# ⚡ Font color hex code mapping
COLOR_MAP = {
    "Cyan": "#00FFCC",
    "White": "#FFFFFF",
    "Red": "#FF5555",
    "Light Yellow": "#FFFF77",
    "Green": "#55FF55"
}


# ----------------------------------------------------------------------
# 4. Background Worker Thread (webrtcvad + Vulkan Whisper)
# ----------------------------------------------------------------------
class SubtitleWorker(QThread):
    new_subtitle_signal = pyqtSignal(list)

    def __init__(self, audio_queue, whisper_exe_path):
        super().__init__()
        self.audio_queue = audio_queue
        self.whisper_exe_path = whisper_exe_path
        self.whisper_model_path = ""
        self.llm_path = ""
        self.running = False
        
        self.src_lang = None
        self.target_lang = None

        self.converter = opencc.OpenCC('s2hk')
        
        self.buffer_size = 3
        self.text_buffer = deque(maxlen=self.buffer_size)

        self.llm_model = None
        
        # Initialize webrtcvad (Aggressiveness level: 0~3, 2 is optimal balance)
        self.vad = webrtcvad.Vad(2)

    def set_buffer_size(self, size):
        """Dynamically adjust the number of subtitle lines retained in buffer"""
        self.buffer_size = int(size)
        if self.text_buffer.maxlen != self.buffer_size:
            items = list(self.text_buffer)
            self.text_buffer = deque(items[-self.buffer_size:], maxlen=self.buffer_size)

    def set_model_paths(self, whisper_model_path, llm_path):
        if self.llm_path != llm_path:
            self.llm_model = None
        self.whisper_model_path = whisper_model_path
        self.llm_path = llm_path

    def load_models(self):
        print("[System] Preparing models and environment, please wait...")

        if not os.path.exists(self.whisper_exe_path):
            raise FileNotFoundError(f"Whisper executable not found: {self.whisper_exe_path}")
        if not os.path.exists(self.whisper_model_path):
            raise FileNotFoundError(f"Whisper model file (*.bin) not found: {self.whisper_model_path}")

        if self.target_lang is not None:
            if self.llm_model is None:
                if self.llm_path and os.path.exists(self.llm_path):
                    print(f"[Model] Loading LLM translation model ({self.llm_path})...")
                    self.llm_model = Llama(
                        model_path=self.llm_path,
                        n_ctx=512,        
                        n_batch=512,      
                        n_gpu_layers=-1,  
                        n_threads=4,
                        verbose=False
                    )
                else:
                    raise FileNotFoundError(f"LLM model file (*.gguf) not found: {self.llm_path}")

    def update_settings(self, src_lang, target_lang):
        self.src_lang = ASR_LANG_MAP.get(src_lang)
        self.target_lang = TRANS_LANG_MAP.get(target_lang)
        self.text_buffer.clear()

    def translate_text(self, text):
        if not self.target_lang or not self.llm_model or not text.strip():
            return ""
        
        try:
            output = self.llm_model.create_chat_completion(
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            f"You are a real-time subtitle translator. Translate the text into {self.target_lang}. Output ONLY the translated result. Keep it natural and fluent. No explanation, no reasoning."
                        )
                    },
                    {"role": "user", "content": text}
                ],
                max_tokens=512,
                temperature=0.1
            )
            
            raw_result = output["choices"][0]["message"]["content"].strip()
            
            clean_result = raw_result
            if "<think>" in raw_result:
                if "</think>" in raw_result:
                    clean_result = raw_result.split("</think>")[-1].strip()
                else:
                    clean_result = re.sub(r'<think>.*', '', raw_result, flags=re.DOTALL).strip()
            
            if "Traditional Chinese" in self.target_lang:
                clean_result = self.converter.convert(clean_result)

            return clean_result if clean_result else raw_result

        except Exception as e:
            print(f"[Translation Error]: {e}")
            return text

    def run(self):
        self.load_models()
        self.running = True
        
        temp_wav_path = os.path.join(tempfile.gettempdir(), "whisper_temp_chunk.wav")
        CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
        
        while self.running:
            try:
                while self.audio_queue.qsize() > 1:
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break

                audio_data = self.audio_queue.get(timeout=0.5)
                if audio_data is None:
                    continue

                audio_int16 = (audio_data * 32767).astype(np.int16)
                raw_bytes = audio_int16.tobytes()

                frame_bytes = int(16000 * 0.03 * 2)  # 960 bytes (30ms)
                speech_frames = 0
                total_frames = 0

                for i in range(0, len(raw_bytes) - frame_bytes + 1, frame_bytes):
                    frame = raw_bytes[i:i + frame_bytes]
                    total_frames += 1
                    if self.vad.is_speech(frame, 16000):
                        speech_frames += 1

                if total_frames == 0 or (speech_frames / total_frames) < 0.1:
                    continue

                with wave.open(temp_wav_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(raw_bytes)

                cmd = [
                    self.whisper_exe_path,
                    "-m", self.whisper_model_path,
                    "-f", temp_wav_path,
                    "-nt"
                ]
                if self.src_lang:
                    cmd.extend(["-l", self.src_lang])
                else:
                    cmd.extend(["-l", "auto"])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    creationflags=CREATE_NO_WINDOW
                )
                
                text = result.stdout.strip()
                text = re.sub(r'\[.*?\]', '', text).strip()
                
                if text:
                    translated = self.translate_text(text) if self.target_lang else ""
                    self.text_buffer.append((text, translated))
                    self.new_subtitle_signal.emit(list(self.text_buffer))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Error] Audio processing/transcription failed: {e}")

    def stop(self):
        self.running = False
        self.text_buffer.clear()
        self.wait()


# ----------------------------------------------------------------------
# 5. Audio Recorder
# ----------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, audio_queue, sample_rate=16000, chunk_duration=1.7):
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.set_chunk_duration(chunk_duration)
        self.recording = False
        self.p = None
        self.stream = None
        self.buffer = np.zeros(0, dtype=np.float32)

    def set_chunk_duration(self, chunk_duration):
        self.chunk_duration = float(chunk_duration)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

    def _get_loopback_device(self):
        wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        if not default_speakers["isLoopbackDevice"]:
            for loopback in self.p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    return loopback
            return None
        return default_speakers

    def start(self):
        if self.p is None:
            self.p = pyaudio.PyAudio()

        device = self._get_loopback_device()
        if not device:
            print("[Error] System audio loopback device not found!")
            return False

        dev_channel_count = device["maxInputChannels"]
        dev_sample_rate = int(device["defaultSampleRate"])

        def callback(in_data, frame_count, time_info, status):
            if not self.recording:
                return (None, pyaudio.paComplete)
            
            data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if dev_channel_count > 1:
                data = data.reshape(-1, dev_channel_count).mean(axis=1)

            if dev_sample_rate != self.sample_rate:
                step = dev_sample_rate / self.sample_rate
                indices = np.arange(0, len(data), step).astype(int)
                indices = indices[indices < len(data)]
                data = data[indices]

            self.buffer = np.append(self.buffer, data)

            if len(self.buffer) >= self.chunk_size:
                chunk = self.buffer[:self.chunk_size]
                self.buffer = self.buffer[self.chunk_size:]
                self.audio_queue.put(chunk)

            return (None, pyaudio.paContinue)

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=dev_channel_count,
            rate=dev_sample_rate,
            input=True,
            input_device_index=device["index"],
            stream_callback=callback
        )
        self.recording = True
        self.stream.start_stream()
        print(f"[System] Started listening to system audio (Interval: {self.chunk_duration}s)...")
        return True

    def stop(self):
        self.recording = False
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.p is not None:
            try:
                self.p.terminate()
            except Exception:
                pass
            self.p = None


# ----------------------------------------------------------------------
# 6. GUI Interface
# ----------------------------------------------------------------------
class SubtitleWindow(QWidget):
    MARGIN = 8

    EDGE_NONE = 0
    EDGE_LEFT = 1
    EDGE_RIGHT = 2
    EDGE_TOP = 4
    EDGE_BOTTOM = 8

    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.old_pos = None

        self.resizing = False
        self.resize_edge = self.EDGE_NONE
        self.drag_start_pos = QPoint()
        self.start_geometry = QRect()

        self.last_subtitles_list = []  # ⚡ Cache the latest subtitles for instant refresh on font/color changes

        self.whisper_exe_path = os.path.join("whisper-vulkan", "whisper-cli.exe")

        self.init_ui()
        self.install_event_filter_recursively(self)

        self.worker = SubtitleWorker(self.audio_queue, self.whisper_exe_path)
        self.worker.new_subtitle_signal.connect(self.update_subtitles)
        self.recorder = AudioRecorder(self.audio_queue)

    def init_ui(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1000, 220)
        self.setMinimumSize(600, 130)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.panel = QWidget(self)
        self.panel.setStyleSheet("""
            QWidget {
                background-color: rgba(20, 20, 20, 200);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 30);
            }
            QLabel {
        font-size: 11px;
        color: white;
        }
        """)
        panel_layout = QVBoxLayout(self.panel)

        ctrl_layout = QHBoxLayout()
        
        self.cb_src = QComboBox()
        self.cb_src.addItems(list(ASR_LANG_MAP.keys()))
        self.cb_src.setCurrentText("Auto Detect")

        self.cb_target = QComboBox()
        self.cb_target.addItems(list(TRANS_LANG_MAP.keys()))
        self.cb_target.setCurrentText("Traditional Chinese")

        self.cb_src.currentTextChanged.connect(self.on_language_change)
        self.cb_target.currentTextChanged.connect(self.on_language_change)

        self.sp_chunk = QDoubleSpinBox()
        self.sp_chunk.setRange(0.5, 30.0)
        self.sp_chunk.setSingleStep(0.1)
        self.sp_chunk.setValue(1.5)
        self.sp_chunk.setSuffix(" s")
        self.sp_chunk.setFixedWidth(65)

        self.sp_buffer = QSpinBox()
        self.sp_buffer.setRange(1, 30)
        self.sp_buffer.setValue(3)
        self.sp_buffer.setSuffix("")
        self.sp_buffer.setFixedWidth(60)
        self.sp_buffer.valueChanged.connect(self.on_buffer_size_change)

        # ⚡ Added: Font size setting (sp_font_size)
        self.sp_font_size = QSpinBox()
        self.sp_font_size.setRange(12, 48)   # Range: 12px ~ 48px
        self.sp_font_size.setValue(19)       # Default: 19px
        self.sp_font_size.setSuffix(" px")
        self.sp_font_size.setFixedWidth(65)
        self.sp_font_size.valueChanged.connect(self.refresh_subtitle_display)

        # ⚡ Added: Font color setting (cb_color)
        self.cb_color = QComboBox()
        self.cb_color.addItems(list(COLOR_MAP.keys()))
        self.cb_color.setCurrentText("Cyan")  # Default: Cyan
        self.cb_color.currentTextChanged.connect(self.refresh_subtitle_display)

        style_combo = """
            QComboBox, QDoubleSpinBox, QSpinBox {
                background: rgba(255,255,255,30);
                color: white;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 10px;
            }
            QComboBox QAbstractItemView {
                background: #222;
                color: white;
                selection-background-color: #555;
                font-size: 11px;
            }
        """
        self.cb_src.setStyleSheet(style_combo)
        self.cb_target.setStyleSheet(style_combo)
        self.sp_chunk.setStyleSheet(style_combo)
        self.sp_buffer.setStyleSheet(style_combo)
        self.sp_font_size.setStyleSheet(style_combo)
        self.cb_color.setStyleSheet(style_combo)

        self.btn_toggle = QPushButton("Start Subtitles")
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background-color: #2b5c8f;
                color: white;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3a75b5; }
        """)
        self.btn_toggle.clicked.connect(self.toggle_captioning)

        btn_close = QPushButton("✕")
        btn_close.setFixedWidth(24)
        btn_close.setStyleSheet("QPushButton { color: #aaa; background: transparent; font-size: 11px; } QPushButton:hover { color: red; }")
        btn_close.clicked.connect(self.close)

        ctrl_layout.addWidget(QLabel("<font color='white'>Speech:</font>"))
        ctrl_layout.addWidget(self.cb_src)
        ctrl_layout.addWidget(QLabel("<font color='white'>Translate:</font>"))
        ctrl_layout.addWidget(self.cb_target)
        ctrl_layout.addWidget(QLabel("<font color='white'>Interval:</font>"))
        ctrl_layout.addWidget(self.sp_chunk)
        ctrl_layout.addWidget(QLabel("<font color='white'>Lines:</font>"))
        ctrl_layout.addWidget(self.sp_buffer)
        ctrl_layout.addWidget(QLabel("<font color='white'>Font:</font>"))
        ctrl_layout.addWidget(self.sp_font_size)
        ctrl_layout.addWidget(QLabel("<font color='white'>Color:</font>"))
        ctrl_layout.addWidget(self.cb_color)
        ctrl_layout.addWidget(self.btn_toggle)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(btn_close)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFrameStyle(0)
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_display.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: white;
                border: none;
            }
        """)
        self.text_display.setText("<div align='center' style='color: #888;'>Click [Start Subtitles] to start capturing system audio...</div>")

        panel_layout.addLayout(ctrl_layout)
        panel_layout.addWidget(self.text_display, 1)
        main_layout.addWidget(self.panel)
        self.setLayout(main_layout)

    def install_event_filter_recursively(self, widget):
        widget.installEventFilter(self)
        widget.setMouseTracking(True)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)
            child.setMouseTracking(True)

    def _get_edge(self, global_pos: QPoint) -> int:
        local_pos = self.mapFromGlobal(global_pos)
        edge = self.EDGE_NONE
        w, h = self.width(), self.height()

        if local_pos.x() <= self.MARGIN:
            edge |= self.EDGE_LEFT
        elif local_pos.x() >= w - self.MARGIN:
            edge |= self.EDGE_RIGHT

        if local_pos.y() <= self.MARGIN:
            edge |= self.EDGE_TOP
        elif local_pos.y() >= h - self.MARGIN:
            edge |= self.EDGE_BOTTOM

        return edge

    def _update_cursor(self, edge: int):
        if edge in (self.EDGE_LEFT | self.EDGE_TOP, self.EDGE_RIGHT | self.EDGE_BOTTOM):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edge in (self.EDGE_RIGHT | self.EDGE_TOP, self.EDGE_LEFT | self.EDGE_BOTTOM):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif edge in (self.EDGE_LEFT, self.EDGE_RIGHT):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif edge in (self.EDGE_TOP, self.EDGE_BOTTOM):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseMove:
            global_pos = event.globalPosition().toPoint()
            edge = self._get_edge(global_pos)

            if self.resizing:
                delta = global_pos - self.drag_start_pos
                rect = QRect(self.start_geometry)

                if self.resize_edge & self.EDGE_LEFT:
                    new_w = rect.width() - delta.x()
                    if new_w >= self.minimumWidth():
                        rect.setLeft(rect.left() + delta.x())
                elif self.resize_edge & self.EDGE_RIGHT:
                    rect.setWidth(rect.width() + delta.x())

                if self.resize_edge & self.EDGE_TOP:
                    new_h = rect.height() - delta.y()
                    if new_h >= self.minimumHeight():
                        rect.setTop(rect.top() + delta.y())
                elif self.resize_edge & self.EDGE_BOTTOM:
                    rect.setHeight(rect.height() + delta.y())

                self.setGeometry(rect)
                return True

            elif self.old_pos is not None:
                delta = global_pos - self.old_pos
                self.move(self.x() + delta.x(), self.y() + delta.y())
                self.old_pos = global_pos
                return True

            else:
                self._update_cursor(edge)
                if edge != self.EDGE_NONE:
                    return True

        elif event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                global_pos = event.globalPosition().toPoint()
                edge = self._get_edge(global_pos)

                if edge != self.EDGE_NONE:
                    self.resizing = True
                    self.resize_edge = edge
                    self.drag_start_pos = global_pos
                    self.start_geometry = self.geometry()
                    return True
                else:
                    if watched in (self, self.panel, self.text_display):
                        self.old_pos = global_pos

        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.resizing or self.old_pos is not None:
                    self.resizing = False
                    self.resize_edge = self.EDGE_NONE
                    self.old_pos = None
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    return True

        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------
    def auto_find_models(self):
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        bin_files = [f for f in os.listdir(models_dir) if f.endswith('.bin')]
        whisper_path = os.path.join(models_dir, bin_files[0]) if bin_files else None

        gguf_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
        llm_path = os.path.join(models_dir, gguf_files[0]) if gguf_files else None

        return whisper_path, llm_path

    def on_language_change(self):
        self.worker.update_settings(self.cb_src.currentText(), self.cb_target.currentText())

    def on_buffer_size_change(self):
        self.worker.set_buffer_size(self.sp_buffer.value())

    def refresh_subtitle_display(self):
        """⚡ Redraw current subtitles immediately when font size or color changes"""
        if self.last_subtitles_list:
            self.update_subtitles(self.last_subtitles_list)

    def toggle_captioning(self):
        if not self.worker.isRunning():
            whisper_path, llm_path = self.auto_find_models()
            is_translation_needed = (self.cb_target.currentText() != "Original Only")

            if not whisper_path:
                self.text_display.setText("<div align='center' style='color: #ff6b6b;'>Error: *.bin model not found in ./models directory!</div>")
                return

            if is_translation_needed and not llm_path:
                self.text_display.setText("<div align='center' style='color: #ff6b6b;'>Error: *.gguf model not found in ./models directory!</div>")
                return

            self.btn_toggle.setText("Loading...")
            self.btn_toggle.setEnabled(False)
            QApplication.processEvents()

            try:
                self.recorder.set_chunk_duration(self.sp_chunk.value())

                self.worker.set_buffer_size(self.sp_buffer.value())
                self.worker.set_model_paths(whisper_path, llm_path)
                self.worker.update_settings(self.cb_src.currentText(), self.cb_target.currentText())
                self.worker.start()
                self.recorder.start()
                
                self.btn_toggle.setText("Stop Subtitles")
                self.btn_toggle.setStyleSheet("background-color: #a83232; color: white; border-radius: 4px; padding: 4px 12px;")
                self.btn_toggle.setEnabled(True)
                self.text_display.setText("<div align='center' style='color: #888;'>Listening...</div>")
            except Exception as e:
                self.text_display.setText(f"<div align='center' style='color: #ff6b6b;'>Launch failed: {e}</div>")
                self.btn_toggle.setText("Start Subtitles")
                self.btn_toggle.setEnabled(True)
        else:
            self.recorder.stop()
            self.worker.stop()
            self.btn_toggle.setText("Start Subtitles")
            self.btn_toggle.setStyleSheet("background-color: #2b5c8f; color: white; border-radius: 4px; padding: 4px 12px;")
            self.text_display.setText("<div align='center' style='color: #888;'>Stopped</div>")

    def update_subtitles(self, subtitles_list):
        self.last_subtitles_list = subtitles_list
        if not subtitles_list:
            return

        # ⚡ Read current font size and color settings
        font_size = self.sp_font_size.value()
        orig_font_size = max(10, font_size - 4)  # Source text font size is 4px smaller than translation
        main_color = COLOR_MAP.get(self.cb_color.currentText(), "#00FFCC")

        html_blocks = []
        for orig, trans in subtitles_list:
            if trans:
                orig_style = f"font-size: {orig_font_size}px; color: #DDDDDD; margin: 0; padding: 0;"
                trans_style = f"font-size: {font_size}px; color: {main_color}; font-weight: bold; margin: 2px 0 6px 0; padding: 0;"

                html_blocks.append(f"""
                    <p style="{orig_style}">{orig}</p>
                    <p style="{trans_style}">{trans}</p>
                """)
            else:
                orig_style = f"font-size: {font_size}px; color: {main_color}; font-weight: bold; margin: 2px 0 6px 0; padding: 0;"
                html_blocks.append(f"""
                    <p style="{orig_style}">{orig}</p>
                """)

        full_html = f'<div align="center">{"".join(html_blocks)}</div>'
        self.text_display.setHtml(full_html)
        
        self.text_display.verticalScrollBar().setValue(
            self.text_display.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        self.recorder.stop()
        self.worker.stop()
        event.accept()


# ----------------------------------------------------------------------
# 7. Main Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubtitleWindow()
    window.show()
    sys.exit(app.exec())