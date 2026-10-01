[中文](https://github.com/kw4356/Live-subtitle-tool/blob/main/README-ZH.md)

# Live Subtitle Tool 🎙️💬

A local, real-time bilingual live subtitle application that transcribes and translates system audio output using **Qwen3-ASR** and **LLM models**. Powered by `CrispASR` and `llama.cpp` with Vulkan acceleration for cross-GPU hardware support.

> *Note: This project was mostly vibe-coded using Gemini 3.6 Flash.*

---

## ✨ Core Features

- **Floating Overlay UI**: Semi-transparent, resizable, and stretchable subtitle window.
- **Simultaneous Transcription & Translation**: Real-time speech recognition and translation into target languages.
- **Vulkan GPU Acceleration**: Supports NVIDIA, AMD and Intel GPUs.
- **Chinese Conversion**: Built-in OpenCC integration for Traditional/Simplified Chinese output.


## ⚙️ Tech 

System audio output → Qwen3-ASR → subtitle → llama cpp → translate subtitle

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
- **[CrispASR (Vulkan)](https://github.com/CrispStrobe/CrispASR/releases)** *(Credits to [CrispStrobe](https://github.com/CrispStrobe/CrispASR))*
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
│   ├── qwen3-asr.gguf          # ASR model
│   └── model.gguf         # LLM translation model

```
### Executable (.exe) Environment

![folder](https://github.com/kw4356/Live-subtitle-tool/blob/main/folder-structure.PNG)

```text
live_subtitle/
├── live_subtitle.exe       # Main executable
├── CrispASR
├── models/                # Local models directory
│   ├── qwen3-asr.gguf          # ASR model
│   └── model.gguf         # LLM model

```

---

## 🚀 Get Started

1. Download and unzip [live_subtitle.rar](https://github.com/kw4356/Live-subtitle-tool/releases/tag/v2.1) .
2. Download ASR and LLM model, put them into `models` folder.
   
   **Recommend**
   ASR:[Qwen3-ASR-1.7B-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) Q4_K or
   [Qwen3-ASR 0.6B-GGUF](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF) q4_k-imatrix if you have limited VRAM
   LLM:[Hy-MT2-1.8B-GGUF](https://huggingface.co/unsloth/Hy-MT2-1.8B-GGUF) UD-Q3_K_XL
   Q3 quants is already good for translation. Quants upper than UD-Q3_K_XL doesn't bring big difference.
   If you have limited VRAM, you can choose Q2 quants but translation will degrade a bit.
4. Double-click `live_subtitle.exe` to start.

---

## 🎛️ GUI Configuration Guide

![Screenshot](https://github.com/kw4356/Live-subtitle-tool/blob/main/livesubGUI.PNG)

| Parameter | Description |
| :--- | :--- |
| **Source** | Select source audio language or choose `Auto Detect`. |
| **Target** | Target translation language for the LLM output (e.g., Traditional Chinese, Simplified Chinese, English, etc.). `Show Original` disable and hide the translated subtitle. |
| **Translation Only** | Hide original subtitle , only show translated subtitle. |
| **Size** | Subtitle font size in pixels. *(Default: `18 px`)*. |
| **Color** | Subtitle text color. *(Default: `Cyan`)*. |

---

## 📊 Model Selection Guide

### 1. ASR Models (Speech Recognition)

| Model Variant | Size | Recommended Use Case & Performance Notes |
| :--- | :--- | :--- |
| **Qwen3-ASR-1.7B-Q4_K** | ~1.5 GB | **Recommendation** Best Quality. |
| **Qwen3-ASR-0.6B-Q4_K** | ~630 MB | Limited GPU/VRAM choice. Degrade in semantic comprehension |

### 2. LLM Models (Translation)

> **Tip**: 1B to 2B parameter non-reasoning/non-thinking instruction models with `Q4_K_M` or `Q3` quantization are ideal for rapid translation tasks.

| Model Name | Size | Notes & Strengths |
| :--- | :--- | :--- |
| `HY-MT1.5-1.8B-UD-Q3_K_XL` | ~990 MB | **Recommendation** Specialized machine translation model by Tencent. |
| `HY-MT1.5-1.8B-Q2` | ~800 MB | Limited GPU/VRAM choice. Degrade in translation quality. |




💡 **Hardware VRAM Pairing Example:**
- Combining **Qwen3-ASR-1.7B-Q4_K** (~1.5 GB) + **HY-MT1.5-1.8B-UD-Q3_K_XL** (~990 MB), total vram usage when running is around 4GB.
---

## 📜 License & Copyright

- **Third-Party Libraries**: Individual components (e.g., `CrispASR`, `llama.cpp`, and respective dependencies) are governed by their original project licenses.
- **Project License**: Distributed under the **[MIT License](https://opensource.org/license/MIT)**.

Copyright © 2026 **kw4356**
