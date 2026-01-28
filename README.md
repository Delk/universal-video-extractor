# 🎥 Universal Video Insight Extractor

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)

**Extract clean transcripts and subtitles from YouTube, Xiaohongshu, Bilibili, TikTok, and more.**  
**一键提取 YouTube、小红书、B站、抖音等平台的视频文案与字幕。**

[English](#-features) | [简体中文](#-功能特性)

</div>

---

<a name="english"></a>

## ✨ Features

- **Multi-Platform Support**: Works with **YouTube, Xiaohongshu (RedNote), Bilibili, TikTok**, etc.
- **Smart Extraction**:
  - **YouTube**: Prioritizes official/uploaded subtitles for speed.
  - **Others**: Automatically downloads audio and uses **Whisper AI** for high-accuracy speech-to-text.
- **Xiaohongshu Special**: Built-in logic to resolve short links (`xhslink.com`) and bypass basic anti-bot redirects.
- **Readable Formatting**:
  - Automatically merges fragmented segments into readable paragraphs.
  - **Auto-Translation**: Converts Traditional Chinese to Simplified Chinese automatically.
- **Modern UI**: Clean Web interface powered by Gradio, with CLI support.

## 📦 Installation

### 1. Prerequisites
- [Python 3.10+](https://www.python.org/downloads/)
- [FFmpeg](https://ffmpeg.org/) (Required for audio processing)
  - **Windows**: `winget install ffmpeg`
  - **Mac**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`

### 2. Clone & Install
```bash
git clone https://github.com/yourusername/universal-video-extractor.git
cd universal-video-extractor

# Install dependencies (This may take a while as it installs PyTorch & Whisper)
pip install -r requirements.txt
```

## 🚀 Usage

### Method 1: Web UI (Recommended)
Launch the graphical interface:
```bash
python app.py
```
This will open a local web page (usually `http://127.0.0.1:7860`). Just paste the video URL and click Extract.

### Method 2: Command Line (CLI)
```bash
# Extract from YouTube
python extractor.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Extract from Xiaohongshu with high accuracy model
python extractor.py "http://xhslink.com/o/xxxxxx" --whisper-model large-v3
```

---

<a name="chinese"></a>

## 🌟 功能特性

- **多平台通杀**：支持 **YouTube、小红书、Bilibili、抖音** 等主流视频平台。
- **智能策略**：
  - **YouTube**：优先抓取官方字幕（速度极快）。
  - **其他平台**：自动下载音频并调用 **Whisper AI** 模型进行语音转文字（高准确率）。
- **小红书特化**：内置短链接解析 (`xhslink.com`) 和反重定向策略，轻松搞定小红书视频。
- **排版优化**：
  - 智能分段：将破碎的字幕行合并为通顺的段落。
  - **繁简转换**：自动将繁体中文转换为简体中文。
- **美观界面**：基于 Gradio 打造的现代化 Web 界面，操作简单。

## 📦 安装指南

### 1. 准备工作
请确保电脑已安装 [Python 3.10+](https://www.python.org/downloads/)。
同时，必须安装 **FFmpeg**（用于音频格式转换）：
- **Windows**: 在终端运行 `winget install ffmpeg`
- **Mac**: 运行 `brew install ffmpeg`

### 2. 下载项目
```bash
git clone https://github.com/yourusername/universal-video-extractor.git
cd universal-video-extractor
```

### 3. 安装依赖库
```bash
pip install -r requirements.txt
```
*(注意：首次安装会下载 PyTorch 和 Whisper 模型，可能需要 1-2GB 流量，请保持网络通畅)*

## 🚀 使用说明

### 方式一：Web 图形界面（小白推荐）
双击运行或在终端输入：
```bash
python app.py
```
程序会自动打开浏览器页面。粘贴视频链接，点击“开始提取”即可。生成的 Markdown 文件会自动保存在 `output` 文件夹中。

### 方式二：命令行工具 (CLI)
如果你习惯使用终端：
```bash
# 基础用法
python extractor.py "视频链接"

# 进阶用法 (使用 large-v3 模型提升准确率，适合小红书/抖音)
python extractor.py "视频链接" --whisper-model large-v3
```

## ⚙️ 常见问题
- **Q: 为什么第一次运行很慢？**
  A: 第一次使用 Whisper 时会自动下载模型权重（Base模型约 140MB，Large 模型约 3GB）。之后运行就会很快了。
- **Q: 小红书链接报错？**
  A: 如果遇到验证码拦截，请稍等几分钟再试。工具内置了自动重试逻辑，通常能解决大部分问题。

## 📄 License
MIT License
