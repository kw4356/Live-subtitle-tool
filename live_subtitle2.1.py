import os
import sys
import re
import time
import json
import queue
import threading
import subprocess
import traceback
import difflib
from collections import deque
import opencc

# ----------------------------------------------------------------------
# 1. Unlock OpenMP Conflict
# ----------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------------------------------------------------
# 2. Import Main Packages
# ----------------------------------------------------------------------
import numpy as np
import pyaudiowpatch as pyaudio

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QEvent, QTimer
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QTextEdit, 
                             QSpinBox, QMessageBox, QCheckBox)
from PyQt6.QtGui import QMouseEvent

from llama_cpp import Llama


# ----------------------------------------------------------------------
# Global Exception Interceptor
# ----------------------------------------------------------------------
def global_excepthook(exctype, value, tb):
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    print("[System Crash Error]:", err_msg)
    
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Unexpected Error")
    msg.setText("The application encountered a serious error:")
    msg.setDetailedText(err_msg)
    msg.exec()

sys.excepthook = global_excepthook


# ----------------------------------------------------------------------
# 3. Menus and Color Mapping
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
    "Show Original": None,
    "Traditional Chinese": "Traditional Chinese (繁體中文)",
    "English": "English",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "French": "French",
    "Spanish": "Spanish",
    "Italian": "Italian",
    "German": "German",
    "Portuguese": "Portuguese"
}

COLOR_MAP = {
    "Cyan": "#00FFCC",
    "White": "#FFFFFF",
    "Red": "#FF5555",
    "Light Yellow": "#FFFF77",
    "Green": "#55FF55"
}


# ----------------------------------------------------------------------
# 4. CrispASR Native Streaming Worker Thread
# ----------------------------------------------------------------------
class SubtitleWorker(QThread):
    new_subtitle_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, audio_queue, crisp_exe_path):
        super().__init__()
        self.audio_queue = audio_queue
        self.crisp_exe_path = crisp_exe_path
        self.asr_model_path = ""
        self.llm_path = ""
        self.running = False
        
        self.src_lang = None
        self.target_lang = None
        self.max_history = 5  # 預設保留歷史字幕行數

        self.converter = opencc.OpenCC('s2hk')
        
        self.history_buffer = []
        self.current_partial_text = ""

        self.llm_model = None
        self.llm_lock = threading.Lock()

        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None

    def set_model_paths(self, asr_model_path, llm_path):
        if self.llm_path != llm_path:
            self.llm_model = None
        self.asr_model_path = asr_model_path
        self.llm_path = llm_path

    def set_max_history(self, max_history):
        self.max_history = max_history
        if len(self.history_buffer) > self.max_history:
            self.history_buffer = self.history_buffer[-self.max_history:]

    def load_llm_model(self):
        if self.target_lang is not None:
            if self.llm_model is None:
                if self.llm_path and os.path.exists(self.llm_path):
                    print(f"[Model] Loading LLM translation model to GPU: {self.llm_path}...")
                    self.llm_model = Llama(
                        model_path=self.llm_path,
                        n_ctx=512,        
                        n_batch=512,      
                        n_gpu_layers=-1,  
                        n_threads=4,
                        verbose=False
                    )
                else:
                    raise FileNotFoundError(f"LLM model file not found (*.gguf): {self.llm_path}")

    def update_settings(self, src_lang, target_lang):
        self.src_lang = ASR_LANG_MAP.get(src_lang)
        self.target_lang = TRANS_LANG_MAP.get(target_lang)
        self.history_buffer.clear()
        self.current_partial_text = ""

    def translate_text(self, text):
        if not self.target_lang or not self.llm_model or not text.strip():
            return ""
        
        # Translation Lock: If busy, return None to keep previous translation and prevent flickering
        if not self.llm_lock.acquire(blocking=False):
            return None

        try:
            output = self.llm_model.create_chat_completion(
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            f"Translate to {self.target_lang}. only output the translated result without any additional explanation."
                        )
                    },
                    {"role": "user", "content": text}
                ],
                max_tokens=128,
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
        finally:
            self.llm_lock.release()

    def _add_or_update_history(self, text):
        """Stable Prefix Extraction via Difflib"""
        text_clean = text.strip()
        if not text_clean or len(text_clean) < 2:
            return

        if not self.history_buffer:
            trans = self.translate_text(text_clean) if self.target_lang else ""
            self.history_buffer.append([text_clean, trans if trans is not None else ""])
        else:
            matched_index = -1
            merged_result = ""

            # Compare with the latest 3 history records
            for idx in range(len(self.history_buffer) - 1, max(-1, len(self.history_buffer) - 4), -1):
                old_text = self.history_buffer[idx][0]
                
                matcher = difflib.SequenceMatcher(None, old_text, text_clean)
                match = matcher.find_longest_match(0, len(old_text), 0, len(text_clean))
                
                overlap_ratio = match.size / max(1, min(len(old_text), len(text_clean)))
                
                if match.size >= 4 or overlap_ratio > 0.5:
                    matched_index = idx
                    merged_result = old_text[:match.a + match.size] + text_clean[match.b + match.size:]
                    break

            if matched_index != -1:
                old_trans = self.history_buffer[matched_index][1]
                if merged_result != self.history_buffer[matched_index][0]:
                    new_trans = self.translate_text(merged_result) if self.target_lang else ""
                    final_trans = old_trans if new_trans is None else new_trans
                    self.history_buffer[matched_index] = [merged_result, final_trans]
            else:
                new_trans = self.translate_text(text_clean) if self.target_lang else ""
                self.history_buffer.append([text_clean, new_trans if new_trans is not None else ""])

        # 保持歷史紀錄行數不超過 max_history
        if len(self.history_buffer) > self.max_history:
            self.history_buffer = self.history_buffer[-self.max_history:]

    def _read_stdout_loop(self):
        while self.running and self.process and self.process.poll() is None:
            try:
                line_bytes = self.process.stdout.readline()
                if not line_bytes:
                    break
                
                line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                if not line_str:
                    continue

                text = ""
                event_type = "partial"

                try:
                    data = json.loads(line_str)
                    text = data.get("text", "").strip()
                    event_type = data.get("event") or data.get("type", "partial")
                except json.JSONDecodeError:
                    text = line_str

                # ⚡ 1. 當偵測到靜音或完結事件：將當前草稿寫入歷史紀錄（換行），並清空草稿
                if event_type in ("silence", "final"):
                    if self.current_partial_text.strip():
                        self._add_or_update_history(self.current_partial_text)
                        self.current_partial_text = ""
                    self.new_subtitle_signal.emit(self._build_display_list())
                    continue

                text = re.sub(r'\[.*?\]', '', text).strip()
                if not text:
                    continue

                text_traditional = self.converter.convert(text)

                # ⚡ 2. 移除標點切割：直接將最新識別結果更新至當前草稿
                self.current_partial_text = text_traditional
                
                # 發送訊號更新 UI 畫面
                self.new_subtitle_signal.emit(self._build_display_list())

            except Exception as e:
                print(f"[Stream Read Exception]: {e}")
                break

    def _build_display_list(self):
        display_list = [list(item) for item in self.history_buffer]
        if self.current_partial_text.strip():
            if display_list:
                last_text = display_list[-1][0]
                if self.current_partial_text in last_text:
                    return display_list[-self.max_history:] if len(display_list) > self.max_history else display_list
            
            trans_partial = self.translate_text(self.current_partial_text) if self.target_lang else ""
            display_list.append([self.current_partial_text, trans_partial if trans_partial is not None else ""])
        
        if len(display_list) > self.max_history:
            return display_list[-self.max_history:]
        return display_list

    def _read_stderr_loop(self):
        while self.running and self.process and self.process.poll() is None:
            try:
                err_line = self.process.stderr.readline()
                if not err_line:
                    break
                err_str = err_line.decode('utf-8', errors='ignore').strip()
                if err_str:
                    print(f"[CrispASR Engine]: {err_str}")
            except Exception:
                break

    def run(self):
        if not os.path.exists(self.crisp_exe_path):
            self.error_signal.emit(f"CrispASR executable not found: {self.crisp_exe_path}")
            return
        if not os.path.exists(self.asr_model_path):
            self.error_signal.emit(f"CrispASR model file not found: {self.asr_model_path}")
            return

        try:
            self.load_llm_model()
        except Exception as e:
            self.error_signal.emit(f"Failed to load LLM model: {e}")
            return

        crisp_dir = os.path.dirname(os.path.abspath(self.crisp_exe_path))
        CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)

        cmd = [
            os.path.abspath(self.crisp_exe_path),
            "-m", os.path.abspath(self.asr_model_path),
            "--stream",
            "--stream-json",
            "--stream-step", "1800",
            "--stream-length", "10000",
            "--stream-final-on-silence-ms", "800",
            "--split-on-punct"
        ]
        if self.src_lang:
            cmd.extend(["-l", self.src_lang])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=crisp_dir,
                creationflags=CREATE_NO_WINDOW,
                bufsize=0
            )
        except Exception as e:
            self.error_signal.emit(f"Failed to start CrispASR native stream engine: {e}")
            return

        self.running = True

        self.stdout_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()

        while self.running and self.process.poll() is None:
            try:
                audio_data = self.audio_queue.get(timeout=0.2)
                if audio_data is None:
                    continue

                audio_int16 = (audio_data * 32767).astype(np.int16)
                pcm_bytes = audio_int16.tobytes()

                self.process.stdin.write(pcm_bytes)
                self.process.stdin.flush()

            except queue.Empty:
                continue
            except (BrokenPipeError, IOError):
                print("[Stream Warning] CrispASR pipe closed")
                break
            except Exception as e:
                print(f"[Stream Push Error]: {e}")

    def stop(self):
        self.running = False
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
            except Exception:
                pass
            self.process = None

        self.history_buffer.clear()
        self.current_partial_text = ""
        self.wait()


# ----------------------------------------------------------------------
# 5. Low-latency Audio Recorder
# ----------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, audio_queue, sample_rate=16000, chunk_duration=0.2):
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.recording = False
        self.p = None
        self.stream = None
        self.buffer = np.zeros(0, dtype=np.float32)

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
            print("[Error] System audio Loopback device not found!")
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
        print("[System] Started native stream audio capture...")
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

        self.last_subtitles_list = []

        self.crisp_exe_path = os.path.join("crispasr", "crispasr.exe")

        self.init_ui()
        self.install_event_filter_recursively(self)

        self.worker = SubtitleWorker(self.audio_queue, self.crisp_exe_path)
        self.worker.new_subtitle_signal.connect(self.update_subtitles)
        self.worker.error_signal.connect(self.on_worker_error)
        self.recorder = AudioRecorder(self.audio_queue)

    def init_ui(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(900, 150)
        self.setMinimumSize(600, 110)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.panel = QWidget(self)
        self.panel.setStyleSheet("""
            QWidget {
                background-color: rgba(20, 20, 20, 200);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 30);
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
        """)
        panel_layout = QVBoxLayout(self.panel)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(4)

        self.cb_src = QComboBox()
        self.cb_src.addItems(list(ASR_LANG_MAP.keys()))
        self.cb_src.setCurrentText("Auto Detect")

        self.cb_target = QComboBox()
        self.cb_target.addItems(list(TRANS_LANG_MAP.keys()))
        self.cb_target.setCurrentText("Traditional Chinese")

        self.chk_trans_only = QCheckBox("Translation Only")
        self.chk_trans_only.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
        """)
        self.chk_trans_only.toggled.connect(self.refresh_subtitle_display)

        self.cb_src.currentTextChanged.connect(self.on_language_change)
        self.cb_target.currentTextChanged.connect(self.on_language_change)

        self.sp_font_size = QSpinBox()
        self.sp_font_size.setRange(12, 48)
        self.sp_font_size.setValue(18)
        self.sp_font_size.setSuffix("px")
        self.sp_font_size.setFixedWidth(55)
        self.sp_font_size.valueChanged.connect(self.refresh_subtitle_display)

        self.cb_color = QComboBox()
        self.cb_color.addItems(list(COLOR_MAP.keys()))
        self.cb_color.setCurrentText("Cyan")
        self.cb_color.currentTextChanged.connect(self.refresh_subtitle_display)

        # 新增保留歷史字幕行數 SpinBox (預設 10)
        self.sp_lines = QSpinBox()
        self.sp_lines.setRange(1, 100)
        self.sp_lines.setValue(1)
        self.sp_lines.setSuffix("")
        self.sp_lines.setFixedWidth(55)
        self.sp_lines.valueChanged.connect(self.on_lines_change)

        style_combo = """
            QComboBox, QSpinBox {
                background-color: rgba(255, 255, 255, 30);
                color: #ffffff;
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #222222;
                color: #ffffff;
                selection-background-color: #555555;
                font-size: 12px;
            }
        """
        self.cb_src.setStyleSheet(style_combo)
        self.cb_target.setStyleSheet(style_combo)
        self.sp_font_size.setStyleSheet(style_combo)
        self.cb_color.setStyleSheet(style_combo)
        self.sp_lines.setStyleSheet(style_combo)

        self.btn_toggle = QPushButton("Start")
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background-color: #2b5c8f;
                color: #ffffff;
                border-radius: 3px;
                padding: 2px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3a75b5;
            }
        """)
        self.btn_toggle.clicked.connect(self.toggle_captioning)

        btn_close = QPushButton("✕")
        btn_close.setFixedWidth(20)
        btn_close.setStyleSheet("""
            QPushButton {
                color: #aaaaaa;
                background-color: transparent;
                font-size: 12px;
                border: none;
            }
            QPushButton:hover {
                color: #ff0000;
            }
        """)
        btn_close.clicked.connect(self.close)

        ctrl_layout.addWidget(QLabel("Source:"))
        ctrl_layout.addWidget(self.cb_src)
        ctrl_layout.addWidget(QLabel("Target:"))
        ctrl_layout.addWidget(self.cb_target)
        ctrl_layout.addWidget(self.chk_trans_only)
        ctrl_layout.addWidget(QLabel("Size:"))
        ctrl_layout.addWidget(self.sp_font_size)
        ctrl_layout.addWidget(QLabel("Color:"))
        ctrl_layout.addWidget(self.cb_color)
        ctrl_layout.addWidget(self.btn_toggle)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(btn_close)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFrameStyle(0)
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                color: #ffffff;
                border: none;
            }
        """)
        self.text_display.setText("<div align='center' style='color: #888;'>Click [Start] to begin streaming...</div>")

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

        gguf_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
        
        asr_files = [f for f in gguf_files if 'asr' in f.lower()]
        asr_path = os.path.join(models_dir, asr_files[0]) if asr_files else None

        llm_files = [f for f in gguf_files if 'asr' not in f.lower()]
        llm_path = os.path.join(models_dir, llm_files[0]) if llm_files else None

        return asr_path, llm_path

    def on_language_change(self):
        self.worker.update_settings(self.cb_src.currentText(), self.cb_target.currentText())

    def on_lines_change(self):
        val = self.sp_lines.value()
        self.worker.set_max_history(val)
        self.refresh_subtitle_display()

    def refresh_subtitle_display(self):
        if self.last_subtitles_list:
            self.update_subtitles(self.last_subtitles_list)

    def on_worker_error(self, err_msg):
        self.recorder.stop()
        self.worker.stop()
        self.btn_toggle.setText("Start")
        self.btn_toggle.setStyleSheet("background-color: #2b5c8f; color: #ffffff; border-radius: 3px; padding: 2px 8px; font-weight: bold; font-size: 12px;")
        self.text_display.setText(f"<div align='center' style='color: #ff6b6b;'>{err_msg}</div>")

    def toggle_captioning(self):
        if not self.worker.isRunning():
            asr_path, llm_path = self.auto_find_models()
            is_translation_needed = (self.cb_target.currentText() != "Show Original")

            if not asr_path:
                self.text_display.setText("<div align='center' style='color: #ff6b6b;'>Error: ASR model (*asr*.gguf) not found in ./models folder!</div>")
                return

            if is_translation_needed and not llm_path:
                self.text_display.setText("<div align='center' style='color: #ff6b6b;'>Error: LLM translation model (*.gguf) not found in ./models folder!</div>")
                return

            self.btn_toggle.setText("Loading...")
            self.btn_toggle.setEnabled(False)
            QApplication.processEvents()

            try:
                self.worker.set_model_paths(asr_path, llm_path)
                self.worker.update_settings(self.cb_src.currentText(), self.cb_target.currentText())
                self.worker.set_max_history(self.sp_lines.value())
                self.worker.start()
                self.recorder.start()
                
                self.btn_toggle.setText("Stop")
                self.btn_toggle.setStyleSheet("background-color: #a83232; color: #ffffff; border-radius: 3px; padding: 2px 8px; font-weight: bold; font-size: 12px;")
                self.btn_toggle.setEnabled(True)
                self.text_display.setText("<div align='center' style='color: #888;'>Streaming listening...</div>")
            except Exception as e:
                self.text_display.setText(f"<div align='center' style='color: #ff6b6b;'>Startup failed: {e}</div>")
                self.btn_toggle.setText("Start")
                self.btn_toggle.setEnabled(True)
        else:
            self.recorder.stop()
            self.worker.stop()
            self.btn_toggle.setText("Start")
            self.btn_toggle.setStyleSheet("background-color: #2b5c8f; color: #ffffff; border-radius: 3px; padding: 2px 8px; font-weight: bold; font-size: 12px;")
            self.text_display.setText("<div align='center' style='color: #888;'>Stopped</div>")

    def _scroll_to_bottom(self):
        scrollbar = self.text_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_subtitles(self, subtitles_list):
        self.last_subtitles_list = subtitles_list
        if not subtitles_list:
            return

        max_lines = self.sp_lines.value()
        display_list = subtitles_list[-max_lines:] if len(subtitles_list) > max_lines else subtitles_list

        font_size = self.sp_font_size.value()
        orig_font_size = max(10, font_size - 4)
        main_color = COLOR_MAP.get(self.cb_color.currentText(), "#00FFCC")
        trans_only = self.chk_trans_only.isChecked()

        html_blocks = []
        for orig, trans in display_list:
            if trans:
                orig_style = f"font-size: {orig_font_size}px; color: #CCCCCC; margin: 0 0 1px 0; padding: 0;"
                trans_style = f"font-size: {font_size}px; color: {main_color}; font-weight: bold; margin: 0 0 5px 0; padding: 0;"

                if trans_only:
                    html_blocks.append(f'<p style="{trans_style}">{trans}</p>')
                else:
                    html_blocks.append(f'<p style="{orig_style}">{orig}</p><p style="{trans_style}">{trans}</p>')
            else:
                orig_style = f"font-size: {font_size}px; color: {main_color}; font-weight: bold; margin: 0 0 5px 0; padding: 0;"
                html_blocks.append(f'<p style="{orig_style}">{orig}</p>')

        full_html = f'<div align="center">{"".join(html_blocks)}</div>'

        # Anti-flicker: Skip repainting if HTML content hasn't changed
        if getattr(self, '_last_html', None) == full_html:
            return
        self._last_html = full_html

        # Atomic paint freeze to prevent text/scrollbar sync flickering
        self.text_display.setUpdatesEnabled(False)
        try:
            self.text_display.setHtml(full_html)
            scrollbar = self.text_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        finally:
            self.text_display.setUpdatesEnabled(True)
            self.text_display.repaint()

    def closeEvent(self, event):
        self.recorder.stop()
        self.worker.stop()
        event.accept()


# ----------------------------------------------------------------------
# 7. Main Application Entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubtitleWindow()
    window.show()
    sys.exit(app.exec())
