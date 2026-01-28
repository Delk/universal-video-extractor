#!/usr/bin/env python3
"""
YouTube Insight Extractor
=========================
Extract transcripts from YouTube videos and format them as readable Markdown.

Author: Aura Reader Team
Usage:
    python extractor.py <youtube_url>
    python extractor.py --help
"""

import re
import sys
import os
import tempfile
import argparse
import requests  # Added for URL resolution
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import yt_dlp
import opencc

# Optional: faster-whisper for speech recognition
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class VideoMetadata:
    """Video metadata container."""
    video_id: str
    title: str
    uploader: str
    duration: int  # seconds
    upload_date: Optional[str] = None
    description: Optional[str] = None
    view_count: Optional[int] = None
    url: str = ""

    @property
    def duration_formatted(self) -> str:
        """Format duration as HH:MM:SS or MM:SS."""
        hours, remainder = divmod(self.duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class TranscriptSegment:
    """Single transcript segment."""
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class ExtractionResult:
    """Complete extraction result."""
    metadata: VideoMetadata
    transcript: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "en"
    is_auto_generated: bool = False


# ============================================================================
# AI Hook Interface (for future Gemini/GPT integration)
# ============================================================================

class AIHook:
    """
    Base class for AI processing hooks.
    
    Extend this class to integrate with Gemini, GPT, or other AI services.
    
    Example:
        class GeminiSummarizer(AIHook):
            def process(self, result: ExtractionResult) -> dict:
                # Call Gemini API here
                summary = gemini.generate(result.transcript)
                return {"summary": summary, "key_points": [...]}
    """
    
    def process(self, result: ExtractionResult) -> dict[str, Any]:
        """
        Process the extraction result with AI.
        
        Args:
            result: The complete extraction result
            
        Returns:
            Dictionary with AI-generated insights
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    @property
    def name(self) -> str:
        """Hook identifier name."""
        return self.__class__.__name__


class AIHookManager:
    """Manage multiple AI processing hooks."""
    
    def __init__(self):
        self._hooks: list[AIHook] = []
        self._callbacks: list[Callable[[ExtractionResult], dict]] = []
    
    def register_hook(self, hook: AIHook) -> None:
        """Register an AI hook."""
        self._hooks.append(hook)
    
    def register_callback(self, callback: Callable[[ExtractionResult], dict]) -> None:
        """Register a simple callback function."""
        self._callbacks.append(callback)
    
    def run_all(self, result: ExtractionResult) -> dict[str, Any]:
        """Run all registered hooks and callbacks."""
        outputs = {}
        
        for hook in self._hooks:
            try:
                outputs[hook.name] = hook.process(result)
            except Exception as e:
                outputs[hook.name] = {"error": str(e)}
        
        for i, callback in enumerate(self._callbacks):
            try:
                outputs[f"callback_{i}"] = callback(result)
            except Exception as e:
                outputs[f"callback_{i}"] = {"error": str(e)}
        
        return outputs


# ============================================================================
# Main Extractor Class
# ============================================================================

class YouTubeExtractor:
    """
    YouTube video transcript extractor.
    
    Extracts transcripts and metadata from YouTube videos,
    formats them into readable Markdown documents.
    
    Usage:
        extractor = YouTubeExtractor()
        result = extractor.extract("https://youtube.com/watch?v=xxx")
        markdown = extractor.to_markdown(result)
    """
    
    # Regex patterns for YouTube URL parsing
    URL_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    
    # Language preferences (priority order)
    LANGUAGE_PREFERENCES = ['en', 'en-US', 'en-GB']
    
    def __init__(self, 
                 output_dir: str = "./output",
                 paragraph_gap: float = 4.0,
                 sentences_per_paragraph: int = 5):
        """
        Initialize the extractor.
        
        Args:
            output_dir: Directory for saving Markdown files
            paragraph_gap: Time gap (seconds) to split paragraphs
            sentences_per_paragraph: Max sentences before forcing paragraph break
        """
        self.output_dir = Path(output_dir)
        self.paragraph_gap = paragraph_gap
        self.sentences_per_paragraph = sentences_per_paragraph
        self.ai_hooks = AIHookManager()
        
        # Initialize OpenCC for Traditional -> Simplified conversion
        self.cc = opencc.OpenCC('t2s')
        
        # yt-dlp options (quiet mode)
        # Revert to default headers as custom ones triggered anti-bot redirects
        self._ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def parse_video_id(self, url: str) -> tuple[Optional[str], str]:
        """
        Extract video ID and platform from URL.
        
        Args:
            url: Video URL
            
        Returns:
            Tuple of (video_id, platform)
            platform can be 'youtube', 'other', or None if invalid
        """
        url = url.strip()
        
        # Check for YouTube patterns
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url, 'youtube'
            
        for pattern in self.URL_PATTERNS:
            match = re.search(pattern, url)
            if match:
                return match.group(1), 'youtube'
        
        # For non-YouTube URLs, we trust yt-dlp to handle them
        # We'll use the URL itself as the ID or let yt-dlp extract it later
        if url.startswith(('http://', 'https://')):
            return None, 'other'
            
        return None, None
    
    def get_metadata(self, url: str) -> VideoMetadata:
        """
        Fetch video metadata using yt-dlp.
        
        Args:
            url: Video URL
            
        Returns:
            VideoMetadata object
        """
        # yt-dlp options for metadata extraction
        opts = self._ydl_opts.copy()
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as e:
                raise ValueError(f"Failed to fetch video metadata: {e}")
        
        return VideoMetadata(
            video_id=info.get('id', 'unknown'),
            title=str(info.get('title') or 'Unknown Title'),
            uploader=str(info.get('uploader') or info.get('uploader_id') or 'Unknown'),
            duration=int(info.get('duration') or 0),
            upload_date=info.get('upload_date'),
            description=info.get('description'),
            view_count=info.get('view_count'),
            url=info.get('webpage_url', url),
        )
    
    def get_transcript(self, video_id: str) -> tuple[list[TranscriptSegment], str, bool]:
        """
        Fetch transcript with language fallback.
        
        Priority:
        1. Manual English subtitles (en, en-US, en-GB)
        2. Auto-generated English subtitles
        3. Any available transcript (first available)
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Tuple of (segments, language_code, is_auto_generated)
            
        Raises:
            ValueError: If no transcript is available
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video")
        except VideoUnavailable:
            raise ValueError("Video is unavailable (private, deleted, or restricted)")
        except Exception as e:
            raise ValueError(f"Failed to fetch transcript list: {e}")
        
        # Try manual English transcripts first
        for lang in self.LANGUAGE_PREFERENCES:
            try:
                transcript = transcript_list.find_transcript([lang])
                if not transcript.is_generated:
                    segments = self._convert_segments(transcript.fetch())
                    return segments, lang, False
            except NoTranscriptFound:
                continue
        
        # Try auto-generated English
        for lang in self.LANGUAGE_PREFERENCES:
            try:
                transcript = transcript_list.find_generated_transcript([lang])
                segments = self._convert_segments(transcript.fetch())
                return segments, lang, True
            except NoTranscriptFound:
                continue
        
        # Fallback: any available transcript
        try:
            available = list(transcript_list)
            if available:
                transcript = available[0]
                segments = self._convert_segments(transcript.fetch())
                return segments, transcript.language_code, transcript.is_generated
        except Exception:
            pass
        
        raise ValueError("No transcript available for this video")
    
    def _convert_segments(self, fetched_transcript) -> list[TranscriptSegment]:
        """Convert raw API segments to TranscriptSegment objects."""
        return [
            TranscriptSegment(
                text=snippet.text,
                start=snippet.start,
                duration=snippet.duration,
            )
            for snippet in fetched_transcript.snippets
        ]
    
    def download_audio(self, url: str, output_dir: Optional[str] = None) -> str:
        """
        Download audio from video using yt-dlp.
        
        Args:
            url: Video URL
            output_dir: Directory to save audio file (default: temp directory)
            
        Returns:
            Path to downloaded audio file
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        # Use a hash of the URL as filename to avoid invalid characters
        import hashlib
        file_hash = hashlib.md5(url.encode()).hexdigest()
        
        # Download audio without conversion (faster-whisper handles multiple formats)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, f"{file_hash}.%(ext)s"),
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                ext = info.get('ext', 'webm')
                # Update file hash with actual video ID if available for clearer debugging
                if 'id' in info:
                    video_id = info['id']
                    # We can't rename easily here without more logic, so stick to hash for now
                    # or rely on yt-dlp's output filename if we didn't force it.
            except Exception as e:
                raise ValueError(f"Failed to download audio: {e}")
        
        output_path = os.path.join(output_dir, f"{file_hash}.{ext}")
        
        if not os.path.exists(output_path):
            # Try common extensions
            for try_ext in ['webm', 'm4a', 'mp4', 'opus', 'mp3']:
                try_path = os.path.join(output_dir, f"{file_hash}.{try_ext}")
                if os.path.exists(try_path):
                    output_path = try_path
                    break
            else:
                raise ValueError(f"Audio file not found after download")
        
        return output_path
    
    def transcribe_audio(self, audio_path: str, 
                         model_size: str = "base",
                         language: Optional[str] = None) -> tuple[list[TranscriptSegment], str]:
        """
        Transcribe audio using faster-whisper.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            language: Language code (e.g., 'en', 'zh'). None for auto-detect.
            
        Returns:
            Tuple of (segments, detected_language)
        """
        if not WHISPER_AVAILABLE:
            raise ValueError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        print(f"[*] Loading Whisper model ({model_size})...", file=sys.stderr)
        # Use CPU to avoid CUDA dependency issues
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        print(f"[*] Transcribing audio...", file=sys.stderr)
        segments_gen, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,  # Filter out silence
        )
        
        detected_lang = info.language
        print(f"[*] Detected language: {detected_lang}", file=sys.stderr)
        
        segments = []
        for seg in segments_gen:
            segments.append(TranscriptSegment(
                text=seg.text.strip(),
                start=seg.start,
                duration=seg.end - seg.start,
            ))
        
        return segments, detected_lang
    
    def merge_segments_to_paragraphs(self, segments: list[TranscriptSegment]) -> str:
        """
        Merge transcript segments into readable paragraphs.
        
        Uses time gaps, sentence count, and character length to determine breaks.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Formatted text with paragraphs
        """
        if not segments:
            return ""
        
        paragraphs = []
        current_paragraph = []
        sentence_count = 0
        char_count = 0
        last_end = 0
        
        # Punctuation marks that indicate end of sentence
        # English: . ! ?
        # Chinese: 。 ！ ？
        SENTENCE_ENDINGS = r'[.!?。！？]'
        
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            
            # Clean up text (remove music notes, fix spacing)
            text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause]
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                continue
            
            # Check for paragraph break conditions
            time_gap = segment.start - last_end if last_end else 0
            
            # Calculate current segment length
            seg_len = len(text)
            
            # Count sentence endings in this segment
            # Using regex to support both English and Chinese punctuation
            new_sentences = len(re.findall(SENTENCE_ENDINGS, text))
            
            # Heuristic for breaking paragraphs:
            # 1. Significant time gap (> paragraph_gap)
            # 2. Enough sentences (> sentences_per_paragraph)
            # 3. Text getting too long (> 500 chars) AND we just finished a sentence
            
            should_break = False
            
            if time_gap > self.paragraph_gap:
                should_break = True
            elif sentence_count >= self.sentences_per_paragraph:
                should_break = True
            elif char_count > 600 and new_sentences > 0:
                # Force break if too long and likely at sentence end
                should_break = True
            elif char_count > 1000:
                # Hard limit just in case
                should_break = True
                
            if should_break and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
                sentence_count = 0
                char_count = 0
            
            current_paragraph.append(text)
            last_end = segment.end
            sentence_count += max(1, new_sentences) # Assume at least 1 sentence if no punctuation
            char_count += seg_len
        
        # Add remaining content
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)
    
    def _convert_to_simplified(self, text: str) -> str:
        """Convert Traditional Chinese to Simplified Chinese."""
        return self.cc.convert(text)

    def _resolve_url(self, url: str) -> str:
        """
        Resolve short URLs (e.g. xhslink.com, youtu.be) to their full versions.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Domains known to need resolution
        short_domains = ['xhslink.com', 'v.douyin.com', 'b23.tv']
        
        if any(d in url for d in short_domains):
            try:
                print(f"[*] Resolving short URL: {url} ...", file=sys.stderr)
                resp = requests.get(url, headers=headers, allow_redirects=False, timeout=10)
                
                if resp.status_code in (301, 302) and 'Location' in resp.headers:
                    resolved = resp.headers['Location']
                    
                    # Security Check: Did we get a valid content link?
                    # If redirected to captcha, login, or root, ABORT resolution and use original URL.
                    bad_patterns = ['captcha', 'login', 'explore', 'xiaohongshu.com/$', 'xiaohongshu.com$']
                    if any(p in resolved for p in bad_patterns) or 'discovery/item' not in resolved:
                        print(f"[*] Warning: Anti-bot detected (redirected to {resolved}). Reverting to original URL.", file=sys.stderr)
                        return url
                        
                    print(f"[*] Resolved to: {resolved}", file=sys.stderr)
                    return resolved
                
            except Exception as e:
                print(f"[*] Warning: URL resolution failed: {e}", file=sys.stderr)
        
        return url

    def extract(self, url: str, use_whisper_fallback: bool = True,
                whisper_model: str = "base") -> ExtractionResult:
        """
        Main extraction method.
        
        Args:
            url: Video URL (YouTube, Bilibili, TikTok, etc.)
            use_whisper_fallback: If True, use Whisper when no subtitles available
            whisper_model: Whisper model size (tiny, base, small, medium, large-v3)
            
        Returns:
            ExtractionResult with metadata and transcript
            
        Raises:
            ValueError: If URL is invalid or extraction fails
        """
        # Resolve URL first
        url = self._resolve_url(url)
        
        video_id, platform = self.parse_video_id(url)
        
        if not platform:
            # If parse_video_id failed but we resolved the URL, maybe it works now?
            # Or trust yt-dlp to handle it
            platform = 'other'
        
        # Fetch metadata (works for all platforms via yt-dlp)
        try:
            metadata = self.get_metadata(url)
        except Exception as e:
             # Soft fail: If metadata fetch fails (e.g. anti-bot), try to proceed to audio download anyway
             print(f"Warning: Failed to fetch metadata: {e}. Trying to download audio directly...", file=sys.stderr)
             metadata = VideoMetadata(
                 video_id="unknown",
                 title="Unknown Video (Metadata Fetch Failed)",
                 uploader="Unknown",
                 duration=0,
                 url=url
             )

        segments = []
        language = "unknown"
        is_auto = False
        
        # 1. Try Native Subtitles (YouTube only)
        if platform == 'youtube' and video_id:
            try:
                segments, language, is_auto = self.get_transcript(video_id)
            except ValueError:
                # Fallback to Whisper below
                pass
        
        # 2. Use Whisper if no segments found (or non-YouTube platform)
        if not segments:
            if not use_whisper_fallback:
                if platform == 'youtube':
                    raise ValueError("No subtitles found and Whisper fallback disabled")
                else:
                    raise ValueError("Non-YouTube platform requires Whisper (enable fallback)")
            
            if not WHISPER_AVAILABLE:
                 raise ValueError("Install faster-whisper to support this video/platform: pip install faster-whisper")
            
            print(f"[*] Extracting audio for transcription (Platform: {platform})...", file=sys.stderr)
            audio_path = None
            try:
                audio_path = self.download_audio(url)
                segments, language = self.transcribe_audio(audio_path, model_size=whisper_model)
                is_auto = True
            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass

        # Merge into readable text
        transcript = self.merge_segments_to_paragraphs(segments)
        
        # Convert to Simplified Chinese if detected language is Chinese
        if language and language.startswith('zh'):
            metadata.title = self._convert_to_simplified(metadata.title)
            if metadata.description:
                metadata.description = self._convert_to_simplified(metadata.description)
            transcript = self._convert_to_simplified(transcript)
            for seg in segments:
                seg.text = self._convert_to_simplified(seg.text)
        
        return ExtractionResult(
            metadata=metadata,
            transcript=transcript,
            segments=segments,
            language=language,
            is_auto_generated=is_auto,
        )
    
    def to_markdown(self, result: ExtractionResult, 
                    include_ai_section: bool = True) -> str:
        """
        Format extraction result as Markdown.
        
        Args:
            result: Extraction result
            include_ai_section: Whether to include placeholder for AI insights
            
        Returns:
            Formatted Markdown string
        """
        meta = result.metadata
        
        # Format upload date if available
        upload_date_str = ""
        if meta.upload_date:
            try:
                date = datetime.strptime(meta.upload_date, "%Y%m%d")
                upload_date_str = date.strftime("%Y-%m-%d")
            except ValueError:
                upload_date_str = meta.upload_date
        
        # Build Markdown
        lines = [
            f"# {meta.title}",
            "",
            "## Metadata",
            "",
            f"- **Video URL**: [{meta.url}]({meta.url})",
            f"- **Author**: {meta.uploader}",
            f"- **Duration**: {meta.duration_formatted}",
        ]
        
        if upload_date_str:
            lines.append(f"- **Upload Date**: {upload_date_str}")
        
        if meta.view_count:
            lines.append(f"- **Views**: {meta.view_count:,}")
        
        subtitle_type = "Auto-generated" if result.is_auto_generated else "Manual"
        lines.append(f"- **Transcript Language**: {result.language} ({subtitle_type})")
        
        lines.extend([
            "",
            "---",
            "",
            "## Transcript",
            "",
            result.transcript,
        ])
        
        # AI insights placeholder
        if include_ai_section:
            lines.extend([
                "",
                "---",
                "",
                "## AI Insights",
                "",
                "<!-- AI_HOOK_PLACEHOLDER -->",
                "<!-- This section will be populated by AI analysis -->",
                "",
                "*No AI analysis available. Connect an AI hook to generate insights.*",
            ])
        
        lines.append("")  # Trailing newline
        
        return '\n'.join(lines)
    
    def save_markdown(self, result: ExtractionResult, 
                      filename: Optional[str] = None) -> Path:
        """
        Save extraction result as Markdown file.
        
        Args:
            result: Extraction result
            filename: Custom filename (default: video_id.md)
            
        Returns:
            Path to saved file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            # Use ONLY video ID for filename to avoid encoding issues on Windows
            # The title is inside the markdown file anyway
            safe_id = re.sub(r'[<>:"/\\|?*]', '', result.metadata.video_id)
            filename = f"{safe_id}.md"
        
        filepath = self.output_dir / filename
        content = self.to_markdown(result)
        
        filepath.write_text(content, encoding='utf-8')
        return filepath
    
    # ========================================================================
    # AI Hook Integration
    # ========================================================================
    
    def register_ai_hook(self, hook: AIHook) -> None:
        """
        Register an AI processing hook.
        
        Example:
            class MySummarizer(AIHook):
                def process(self, result):
                    return {"summary": "..."}
            
            extractor.register_ai_hook(MySummarizer())
        """
        self.ai_hooks.register_hook(hook)
    
    def register_ai_callback(self, callback: Callable[[ExtractionResult], dict]) -> None:
        """
        Register a simple callback for AI processing.
        
        Example:
            def analyze(result):
                # Call your AI API here
                return {"analysis": "..."}
            
            extractor.register_ai_callback(analyze)
        """
        self.ai_hooks.register_callback(callback)
    
    def run_ai_analysis(self, result: ExtractionResult) -> dict[str, Any]:
        """
        Run all registered AI hooks on the result.
        
        Returns:
            Dictionary with all AI hook outputs
        """
        return self.ai_hooks.run_all(result)


# ============================================================================
# CLI Interface
# ============================================================================

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='youtube-extractor',
        description='Extract YouTube video transcripts to Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extractor.py https://youtube.com/watch?v=dQw4w9WgXcQ
  python extractor.py youtu.be/dQw4w9WgXcQ -o ./transcripts
  python extractor.py dQw4w9WgXcQ --no-save
        """
    )
    
    parser.add_argument(
        'url',
        help='YouTube video URL or video ID'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='./output',
        help='Output directory for Markdown files (default: ./output)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Print to stdout instead of saving to file'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output metadata as JSON (for API integration)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    parser.add_argument(
        '--no-whisper',
        action='store_true',
        help='Disable Whisper fallback for videos without subtitles'
    )
    
    parser.add_argument(
        '--whisper-model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
        help='Whisper model size (default: base). Larger = more accurate but slower'
    )
    
    return parser


def main():
    """CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    extractor = YouTubeExtractor(output_dir=args.output)
    
    def log(msg: str):
        if not args.quiet:
            print(f"[*] {msg}", file=sys.stderr)
    
    try:
        log(f"Processing: {args.url}")
        
        # Check URL validity (weak check, let extract handle details)
        vid, platform = extractor.parse_video_id(args.url)
        if not platform:
            print(f"Error: Invalid or unsupported URL - {args.url}", file=sys.stderr)
            sys.exit(1)
        
        log(f"Platform: {platform}" + (f" (ID: {vid})" if vid else ""))
        
        # Extract transcript and metadata
        result = extractor.extract(
            args.url,
            use_whisper_fallback=not args.no_whisper,
            whisper_model=args.whisper_model
        )
        
        log(f"Title: {result.metadata.title}")
        log(f"Transcript language: {result.language} "
            f"({'auto-generated' if result.is_auto_generated else 'manual'})")
        log(f"Segments: {len(result.segments)}")
        
        # Output handling
        if args.json:
            import json
            output = {
                "video_id": result.metadata.video_id,
                "title": result.metadata.title,
                "uploader": result.metadata.uploader,
                "duration": result.metadata.duration,
                "url": result.metadata.url,
                "language": result.language,
                "is_auto_generated": result.is_auto_generated,
                "transcript": result.transcript,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        
        elif args.no_save:
            print(extractor.to_markdown(result))
        
        else:
            filepath = extractor.save_markdown(result)
            log(f"Saved to: {filepath}")
            print(f"Success! Transcript saved to: {filepath}")
    
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
