# 即時字幕工具 🎙️💬

一個本機、即時的雙語即時字幕應用程式，利用 **Qwen3-ASR** 與 **LLM 模型** 對系統音訊輸出進行語音轉錄與翻譯。由 `CrispASR` 與具備 Vulkan 加速的 `llama.cpp` 驅動，支援跨 GPU 硬體架構。

> *註：本專案主要使用 Gemini 3.6 Flash 以 vibe-coding 方式開發而成。*

---

## ✨ 核心特色

- **懸浮視窗 UI**：半透明、可調整大小且可拉伸的字幕視窗。
- **同時轉錄與翻譯**：即時語音識別並翻譯為目標語言。
- **Vulkan GPU 加速**：支援 NVIDIA、AMD 與 Intel 顯示卡。
- **繁簡中文轉換**：內建 OpenCC 整合，支援繁體/簡體中文輸出。

---

## ⚙️ 技術架構

系統音訊輸出 → Qwen3-ASR → 字幕 → llama cpp → 翻譯字幕

---

## 💻 系統需求

| 元件 | 最低需求 | 建議配備 |
| :--- | :--- | :--- |
| **作業系統** | Windows 10 / 11 | Windows 10 / 11 |
| **Python** | 3.11 | 3.11 |
| **記憶體 (RAM)** | 4 GB | 8 GB+ |
| **顯示記憶體 (VRAM)** | 2 GB | 4 GB+ |

---

## 📦 相依套件與致謝

- **[llama-cpp-python (Vulkan)](https://github.com/abetlen/llama-cpp-python/releases)** *(感謝 [abetlen](https://github.com/abetlen))*
- **[CrispASR (Vulkan)](https://github.com/CrispStrobe/CrispASR/releases)** *(感謝 [CrispStrobe](https://github.com/CrispStrobe/CrispASR))*
- `PyQt6`
- `pyaudiowpatch`
- `opencc` *(用於繁體/簡體中文文字轉換)*
- `webrtcvad`

---

## 📁 檔案結構

### Python 環境
```text
live_subtitle/
├── live_subtitle.py        # 主程式進入點
├── requirements.txt       # Python 相依套件清單
├── models/                # 本機模型目錄（需手動下載）
│   ├── qwen3-asr.gguf          # ASR 模型 
│   └── model.gguf         # LLM 翻譯模型
```

### 可執行檔 (.exe) 環境

![folder](https://github.com/kw4356/Live-subtitle-tool/blob/main/folder-structure.PNG)

```text
live_subtitle/
├── live_subtitle.exe       # 主執行檔
├── CrispASR
├── models/                # 本機模型目錄
│   ├── qwen3-asr.gguf          # ASR 模型
│   └── model.gguf         # LLM 模型
```

---

## 🚀 快速開始

1. 下載並解壓縮 [live_subtitle.rar](https://github.com/kw4356/Live-subtitle-tool/releases/tag/v2.1)。
2. 下載 ASR 與 LLM 模型，並放入 `models` 資料夾中。
   
   **推薦模型：**
   - ASR：[Qwen3-ASR-1.7B-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) Q4_K；若 VRAM 有限可選擇 [Qwen3-ASR 0.6B-GGUF](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF) q4_k-imatrix。
   - LLM：[Hy-MT2-1.8B-GGUF](https://huggingface.co/unsloth/Hy-MT2-1.8B-GGUF) UD-Q3_K_XL。Q3 量化級別對於翻譯來說品質已經足夠良好，高於 UD-Q3_K_XL 的量化級別不會帶來顯著差異。若 VRAM 受限，可選擇 Q2 量化，但翻譯品質會略有下降。
3. 雙擊 `live_subtitle.exe` 即可啟動。

---

## 🎛️ 圖形介面 (GUI) 設定指南

![Screenshot](https://github.com/kw4356/Live-subtitle-tool/blob/main/livesubGUI.PNG)

| 參數 | 說明 |
| :--- | :--- |
| **Source (來源)** | 選擇來源音訊語言或選擇 `Auto Detect`（自動偵測）。 |
| **Target (目標)** | LLM 輸出的目標翻譯語言（例如：繁體中文、簡體中文、英文等）。選擇 `Show Original` 會停用並隱藏翻譯字幕。 |
| **Translation Only (僅翻譯)** | 隱藏原始字幕，僅顯示翻譯後的字幕。 |
| **Size (大小)** | 字幕字型大小（單位：像素）。*(預設：`18 px`)*。 |
| **Color (顏色)** | 字幕文字顏色。*(預設：`Cyan` / 青色)*。 |

---

## 📊 模型選擇指南

### 1. ASR 模型（語音識別）

| 模型版本 | 大小 | 建議使用場景與效能說明 |
| :--- | :--- | :--- |
| **Qwen3-ASR-1.7B-Q4_K** | ~1.5 GB | **【推薦】** 最佳品質。 |
| **Qwen3-ASR-0.6B-Q4_K** | ~630 MB | 顯卡 / VRAM 有限時的選擇，語意理解能力會有所降低。 |

### 2. LLM 模型（翻譯）

> **提示**：具備 `Q4_K_M` 或 `Q3` 量化的 1B 至 2B 參數非推理/非思考型指令模型，非常適合進行快速翻譯任務。

| 模型名稱 | 大小 | 說明與優勢 |
| :--- | :--- | :--- |
| `HY-MT1.5-1.8B-UD-Q3_K_XL` | ~990 MB | **【推薦】** 騰訊開發的專門機器翻譯模型。 |
| `HY-MT1.5-1.8B-Q2` | ~800 MB | 顯卡 / VRAM 有限時的選擇，翻譯品質會略有下降。 |

💡 **硬體 VRAM 搭配範例：**
- 結合 **Qwen3-ASR-1.7B-Q4_K** (~1.5 GB) + **HY-MT1.5-1.8B-UD-Q3_K_XL** (~990 MB)，執行時的總 VRAM 使用量約為 4 GB。

---

## 📜 授權條款與版權

- **第三方函式庫**：各個別元件（如 `CrispASR`、`llama.cpp` 及相關相依套件）均受其原專案之授權條款約束。
- **專案授權**：本專案採用 **[MIT 授權條款](https://opensource.org/license/MIT)** 釋出。

Copyright © 2026 **kw4356**