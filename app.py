import gradio as gr
import subprocess
import os
import sys
import re

# UI Text Dictionary (Keep as is)
UI_TEXT = {
    "en": {
        "title": "Universal Video Insight Extractor",
        "description": "Extract transcripts from **YouTube, Bilibili, Xiaohongshu, TikTok** and more.\n\n- **YouTube**: Auto-detects subtitles first.\n- **Others**: Uses **Whisper** (AI Speech-to-Text).",
        "url_label": "Video URL",
        "url_placeholder": "Paste link here (e.g., https://www.youtube.com/watch?v=... or http://xhslink.com/...)",
        "adv_settings": "Advanced Settings",
        "model_label": "Whisper Model (for speech recognition)",
        "model_info": "Larger models = higher accuracy but slower. Recommended: 'medium' or 'large-v3' for Chinese.",
        "use_whisper_label": "Use Whisper Fallback",
        "use_whisper_info": "Use AI if no subtitles found",
        "simplify_label": "Force Simplified Chinese",
        "simplify_info": "Auto-enabled for Chinese content",
        "btn_label": "🚀 Extract Transcript",
        "output_file_label": "Download Markdown",
        "preview_label": "Preview",
        "note": "**Note**: First run with a new Whisper model will download weights. Processing time depends on hardware.",
        "lang_label": "Interface Language / 界面语言"
    },
    "zh": {
        "title": "通用视频文案提取器",
        "description": "一键提取 **YouTube, Bilibili, 小红书, 抖音** 等平台视频文案。\n\n- **YouTube**: 优先获取官方字幕。\n- **其他平台**: 自动使用 **Whisper** AI 进行语音转文字。",
        "url_label": "视频链接",
        "url_placeholder": "在此粘贴链接 (例如 https://www.youtube.com/watch?v=... 或 http://xhslink.com/...)",
        "adv_settings": "高级设置",
        "model_label": "Whisper 模型 (语音识别)",
        "model_info": "模型越大越准但越慢。中文推荐使用 'medium' 或 'large-v3'。",
        "use_whisper_label": "启用 Whisper 替补",
        "use_whisper_info": "当无字幕时使用 AI 识别",
        "simplify_label": "强制简体中文",
        "simplify_info": "中文内容自动启用",
        "btn_label": "🚀 开始提取文案",
        "output_file_label": "下载 Markdown 文件",
        "preview_label": "预览",
        "note": "**注意**: 首次使用新模型会自动下载权重文件。处理时间取决于电脑性能。",
        "lang_label": "界面语言 / Interface Language"
    }
}

def process_video(url, whisper_model, use_whisper, simplify_chinese):
    """
    Process video URL by calling the CLI script directly to ensure consistency.
    """
    try:
        if not url.strip():
            return "Error: Please enter a URL.", None

        # Construct command
        cmd = [sys.executable, "extractor.py", url, "--whisper-model", whisper_model]
        
        if not use_whisper:
            cmd.append("--no-whisper")
            
        # Add a flag to output JSON path or just parse stdout
        # But extractor.py default behavior is to print log to stderr and path to stdout? 
        # Actually it prints "Success! Transcript saved to: ..." to stdout
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute command
        # Capture output to find the generated file path
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace' # avoid encoding errors
        )
        
        stdout = process.stdout
        stderr = process.stderr
        
        if process.returncode != 0:
            return f"Error occurred (Exit Code {process.returncode}):\n\n{stderr}\n\n{stdout}", None
            
        # Extract file path from output
        # Look for "Saved to: path/to/file.md"
        match = re.search(r"Saved to:\s+(.+?\.md)", stdout) or re.search(r"Saved to:\s+(.+?\.md)", stderr)
        
        if match:
            filepath = match.group(1).strip()
            # Ensure path is absolute or relative to current dir
            if not os.path.isabs(filepath):
                filepath = os.path.abspath(filepath)
                
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content, filepath
            else:
                return f"Error: Output file not found at {filepath}\n\nLog:\n{stderr}", None
        else:
            return f"Success reported but file path not found in output.\n\nOutput:\n{stdout}\n\nLog:\n{stderr}", None

    except Exception as e:
        import traceback
        return f"System Error:\n{str(e)}\n\n{traceback.format_exc()}", None

# Define Custom CSS (Minimal)
custom_css = """
/* 仅隐藏 Textbox 拉动条 */
textarea { resize: none !important; }
"""

with gr.Blocks(title="Universal Video Extractor", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    # 顶部导航栏布局 (保持简单)
    with gr.Row():
        with gr.Column(scale=5):
            title_md = gr.Markdown(f"# 🎥 {UI_TEXT['zh']['title']}")
        
        with gr.Column(scale=1):
            lang_radio = gr.Radio(
                choices=["en", "zh"], 
                value="zh", 
                show_label=False,
                interactive=True
            )
    
    description_md = gr.Markdown(UI_TEXT["zh"]['description'])
    
    # 主要操作区 (还原到你习惯的上下布局)
    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label=UI_TEXT["zh"]['url_label'],
                placeholder=UI_TEXT["zh"]['url_placeholder'],
                lines=1,
                max_lines=1
            )
            
            with gr.Accordion(UI_TEXT["zh"]['adv_settings'], open=False) as adv_acc:
                model_dropdown = gr.Dropdown(
                    label=UI_TEXT["zh"]['model_label'],
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    value="base",
                    info=UI_TEXT["zh"]['model_info']
                )
                
                with gr.Row():
                    use_whisper_chk = gr.Checkbox(
                        label=UI_TEXT["zh"]['use_whisper_label'], 
                        value=True,
                        info=UI_TEXT["zh"]['use_whisper_info']
                    )
                    simplify_chk = gr.Checkbox(
                        label=UI_TEXT["zh"]['simplify_label'], 
                        value=True, 
                        interactive=False,
                        info=UI_TEXT["zh"]['simplify_info']
                    )

            extract_btn = gr.Button(
                UI_TEXT["zh"]['btn_label'], 
                variant="primary", 
                size="lg"
            )

        with gr.Column(scale=3):
            output_file = gr.File(label=UI_TEXT["zh"]['output_file_label'])
            output_preview = gr.Markdown(label=UI_TEXT["zh"]['preview_label'])
            note_md = gr.Markdown(f"---\n{UI_TEXT['zh']['note']}")

    # Language Change Event
    def update_language(lang):
        t = UI_TEXT[lang]
        return (
            gr.Markdown(value=f"# 🎥 {t['title']}"),
            gr.Markdown(value=t['description']),
            gr.Textbox(label=t['url_label'], placeholder=t['url_placeholder']),
            gr.Accordion(label=t['adv_settings']),
            gr.Dropdown(label=t['model_label'], info=t['model_info']),
            gr.Checkbox(label=t['use_whisper_label'], info=t['use_whisper_info']),
            gr.Checkbox(label=t['simplify_label'], info=t['simplify_info']),
            gr.Button(value=t['btn_label']),
            gr.File(label=t['output_file_label']),
            gr.Markdown(label=t['preview_label']),
            gr.Markdown(value=f"---\n{t['note']}")
        )

    lang_radio.change(
        fn=update_language,
        inputs=[lang_radio],
        outputs=[
            title_md, description_md, url_input, 
            adv_acc, model_dropdown, use_whisper_chk, simplify_chk, extract_btn,
            output_file, output_preview, note_md
        ]
    )

    # Extract Event
    extract_btn.click(
        fn=process_video,
        inputs=[url_input, model_dropdown, use_whisper_chk, simplify_chk],
        outputs=[output_preview, output_file]
    )

if __name__ == "__main__":
    print("Starting Web UI (CLI-Wrapper Mode)...")
    # Launch locally only
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
