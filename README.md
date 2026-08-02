# Live Subtitle Tool 🎙️💬

A local, real-time bilingual live subtitle application that transcribes and translates system audio output using **OpenAI Whisper** and **LLM models**. Powered by `whisper.cpp` and `llama.cpp` with Vulkan acceleration for cross-GPU hardware support.

> *Note: This project was mostly vibe-coded using Gemini 3.6 Flash.*

---

## ✨ Core Features

- **Floating Overlay UI**: Semi-transparent, resizable, and stretchable subtitle window.
- **Multi-Model Support**: Compatible with various Whisper models (`tiny`, `base`, `small`, `medium`, `large`, `large-v3-turbo`).
- **Simultaneous Transcription & Translation**: Real-time speech recognition and translation into target languages.
- **Vulkan GPU Acceleration**: Supports NVIDIA CUDA, AMD ROCm/Vulkan, and Intel GPUs.
- **Voice Activity Detection (VAD)**: Integrated VAD for filtering silence and audio segments.
- **Chinese Conversion**: Built-in OpenCC integration for Traditional/Simplified Chinese output.

---

## ⚙️ Tech 

System audio output → whisper.cpp & VAD → subtitle → llama cpp → translate subtitle

---

## 💻 System Requirements

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10 / 11 | Windows 10 / 11 |
| **Python** | 3.11 | 3.11 |
| **RAM** | 4 GB | 8 GB+ |
| **VRAM** | 2 GB | 4 GB+ |

---

## 📦 Dependencies & Acknowledgments

- **[llama-cpp-python (Vulkan)](https://github.com/abetlen/llama-cpp-python/releases)** *(Credits to [abetlen](https://github.com/abetlen))*
- **[whisper.cpp (Vulkan)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases)** *(Credits to [lemonade-sdk](https://github.com/lemonade-sdk))*
- `numpy`
- `PyQt6`
- `pyaudiowpatch`
- `opencc` *(for Traditional / Simplified Chinese text conversion)*
- `webrtcvad`

---

## 📁 File Structure

### Python Environment
```text
live_subtitle/
├── live_subtitle.py        # Main entry script
├── requirements.txt       # Python dependency list
├── models/                # Local models directory (download manually)
│   ├── ggml.bin          # Whisper model (GGML format .bin)
│   └── model.gguf         # LLM translation model (.gguf format)
└── whisper-vulkan/        # Whisper Vulkan executable directory
    ├── whisper-cli.exe
    └── *.dll
```

### Executable (.exe) Environment
```text
live_subtitle/
├── live_subtitle.exe       # Main executable
├── models/                # Local models directory
│   ├── ggml.bin          # Whisper model (.bin)
│   └── model.gguf         # LLM model (.gguf)
└── whisper-vulkan/        # Whisper Vulkan folder
    ├── whisper-cli.exe
    └── *.dll
```

---

## 🚀 Get Started

### Option 1: Running with Python

#### 1. Install Dependencies & Binary Wheels
```bash
# 1. Install standard Python packages
pip install -r requirements.txt

# 2. Install pre-built llama-cpp-python Vulkan wheel
# Download the wheel file from https://github.com/abetlen/llama-cpp-python/releases
# Place it under your virtual environment folder and run:
pip install llama_cpp_python-0.3.34-py3-none-win_amd64.whl
```

#### 2. Setup `whisper-vulkan`
1. Download pre-built releases from [whisper.cpp-rocm releases](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases).
2. Create a folder named `whisper-vulkan` under your root folder.
3. Unzip all extracted files (including `whisper-cli.exe` and `.dll` dependencies) into the `whisper-vulkan` directory.

#### 3. Download ASR & LLM Models
1. Create a `models/` directory in your root folder.
2. Download and place your selected models inside:
   - **Whisper Model**: must be in **GGML** format (`.bin`).
   - **LLM Model**: must be in **GGUF** format (`.gguf`).

#### 4. Run Application
```bash
python live_subtitle.py
```

---

### Option 2: Running Standalone Executable (.exe)

1. Create a folder and place `live_subtitle.exe` inside.
2. Download `whisper-vulkan` from [whisper.cpp-rocm releases](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases).
   - Create a folder named `whisper-vulkan` in the same directory.
   - Extract all contents (`whisper-cli.exe`, `.dll` files) into `whisper-vulkan/`.
3. Create a `models/` directory in the same path and add your models:
   - **Whisper model** (`.bin` GGML format)
   - **LLM model** (`.gguf` format)
4. Double-click `live_subtitle.exe` to start.

---

## 🎛️ GUI Configuration Guide

![Screenshot](https://github.com/kw4356/Live-subtitle-tool/blob/main/livesubGUI.PNG)

| Parameter | Description |
| :--- | :--- |
| **Speech** | Select source audio language or choose `Auto Detect`. *(Whisper natively supports 99 languages).* |
| **Translate** | Target translation language for the LLM output (e.g., Traditional Chinese, Simplified Chinese, English, etc.). |
| **Interval** | Defines `chunk_length` (seconds of audio processed per chunk).<br>• **Default**: `1.50 s`<br>• **Recommended**: `1.5s` – `3.0s`<br>• *Note*: Values `< 1.0s` may cause context loss. Higher values increase accuracy but add latency. |
| **Lines** | Number of historical subtitle lines displayed on screen. *(Default: `3`)*. |
| **Font** | Subtitle font size in pixels. *(Default: `19 px`)*. |
| **Color** | Subtitle text color. *(Default: `Cyan`)*. |

---

## 📊 Model Selection Guide

### 1. Whisper Models (Speech Recognition)

| Model Variant | Size | Recommended Use Case & Performance Notes |
| :--- | :--- | :--- |
| **Large-v3-turbo Q8 quant** | ~850 MB | **Top Recommendation** if VRAM permits. Highest accuracy/speed ratio. |
| **Large-v3-turbo Q6 quant** | ~660 MB | Slightly higher fidelity than Q5 quant. |
| **Large-v3-turbo Q5 quant** | ~560 MB | Excellent balance of recognition accuracy and VRAM usage. |
| **Large-v3-turbo Q4 quant** | ~470 MB | Lowest recommended quantization for Large-v3-turbo. |
| **Small Q8 quant** | ~260 MB | Fallback choice for very low VRAM systems. *(Models smaller than Small Q8 are not recommended due to accuracy degradation)*. |

### 2. LLM Models (Translation)

> **Tip**: 1B to 2B parameter non-reasoning/non-thinking instruction models with `Q4_K_M` or `Q3` quantization are ideal for rapid translation tasks.

| Model Name | Size | Notes & Strengths |
| :--- | :--- | :--- |
| `qwen2.5-1.5b-instruct-q4_k_m` | ~950 MB | Strong performance across Asian languages. |
| `gemma-3-1b-it-q4_k_m` | ~750 MB | Fast and compact general instruction model. |
| `HY-MT1.5-1.8B-Q4_K_M` | ~1.1 GB | Specialized machine translation model by Tencent. |
| `nllb-200-distilled-600M-q4_k_m` | ~750 MB | Lightweight multi-lingual translation model by Meta. |
| `nllb-200-distilled-1.3B-q4_k_m` | ~1.2 GB | Higher capacity multi-lingual translation model by Meta. |

💡 **Hardware VRAM Pairing Example:**
- Combining **Large-v3-turbo Q8** (~850 MB VRAM) + **qwen2.5-1.5b-instruct-q4_k_m** (~950 MB VRAM) yields a total VRAM footprint of **~2.5 GB**.
- This setup will run smoothly on GPUs such as the **GTX 1060 (3GB)**.

---

## 📜 License & Copyright

- **Third-Party Libraries**: Individual components (e.g., `whisper.cpp`, `llama.cpp`, and respective dependencies) are governed by their original project licenses.
- **Project License**: Distributed under the **MIT License**.

Copyright © 2026 **kw4356**
