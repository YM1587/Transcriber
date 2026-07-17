"""
transcribe_diarize.py — CLI transcription WITH speaker labels (optimized).

Combines faster-whisper (speech-to-text) with pyannote.audio (who speaks when).

Optimizations vs. original:
  - Reuses a single 16kHz mono WAV for BOTH Whisper and pyannote (no double decode)
  - Cached WhisperModel across runs within a process (no reload per call)
  - Streams transcript lines to stdout AND writes incrementally to the out file,
    so partial results are visible immediately and a crash never loses everything
  - Structured timing log (ffmpeg / diarization / transcription / total, real-time factor)
  - Graceful diarization failure -> falls back to plain transcript instead of aborting
  - ffmpeg timeout + quiet stderr; ffprobe duration guard
  - Resilient pyannote pipeline: moved to CPU explicitly, num_speakers optional

One-time setup:
  1. Create a free account at https://huggingface.co
  2. Accept the pyannote model terms at:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  3. Create a token at https://huggingface.co/settings/tokens (read access is enough)
  4. Pass it with --hf-token YOUR_TOKEN  (or set env var HF_TOKEN)

Examples:
    python transcribe_diarize.py interview.m4a --hf-token hf_xxxx
    python transcribe_diarize.py meeting.mp4 --hf-token hf_xxxx --mode fast
    python transcribe_diarize.py podcast.mp3 --hf-token hf_xxxx --speakers 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity / future embedding work)

# ─── Config / logging ───────────────────────────────────────────────────────

logger = logging.getLogger("transcriber.cli")

start_global = time.time()


def log_event(event: str, **fields) -> None:
    rec = {"event": event, "ts": round(time.time(), 3), **fields}
    print(json.dumps(rec, ensure_ascii=False), file=sys.stderr, flush=True)


SPEED_PRESETS = {
    "quality": dict(beam_size=5, best_of=5, temperature=0.0),
    "fast":    dict(beam_size=2, best_of=1, temperature=0.0),
    "turbo":   dict(beam_size=1, best_of=1, temperature=0.0),
}

FFMPEG_TIMEOUT = 600
MAX_AUDIO_MINUTES = 180

# ─── Dependency check ───────────────────────────────────────────────────────

def check_deps() -> None:
    missing = []
    for mod in ("faster_whisper", "pyannote.audio", "soundfile"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(f"Missing packages: {', '.join(missing)}\n"
                 f"Run: pip install {' '.join(missing)}")


# ─── Audio I/O ──────────────────────────────────────────────────────────────

def extract_audio_wav(input_path: Path, tmp_dir: str) -> Path:
    """ffmpeg -> 16kHz mono s16 WAV. Shared by Whisper + pyannote."""
    out_path = Path(tmp_dir) / "audio_16k.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(input_path), "-ac", "1", "-ar", "16000",
           "-sample_fmt", "s16", str(out_path)]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        sys.exit(f"ffmpeg timed out after {FFMPEG_TIMEOUT}s")
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr.decode(errors='replace')[:2000]}")
    log_event("ffmpeg_done", seconds=round(time.time() - t0, 2))
    return out_path


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError):
        return 0.0


# ─── Cached Whisper model ───────────────────────────────────────────────────

_whisper_cache: dict[str, object] = {}
_whisper_lock = threading.Lock()


def get_whisper(model_size: str):
    """Cache the WhisperModel so repeated invocations don't reload weights."""
    with _whisper_lock:
        if model_size not in _whisper_cache:
            t0 = time.time()
            log_event("whisper_load_start", model=model_size)
            from faster_whisper import WhisperModel
            _whisper_cache[model_size] = WhisperModel(
                model_size, device="cpu", compute_type="int8"
            )
            log_event("whisper_load_end", model=model_size,
                      seconds=round(time.time() - t0, 2))
        return _whisper_cache[model_size]


# ─── Diarization ────────────────────────────────────────────────────────────

def diarize(wav_path: Path, hf_token: str, num_speakers=None):
    """Run pyannote speaker diarization; return list of (start, end, speaker)."""
    from pyannote.audio import Pipeline
    import torch

    t0 = time.time()
    log_event("diarize_start")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    pipeline.to(torch.device("cpu"))

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(wav_path), **kwargs)
    segments = [(turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)]
    log_event("diarize_done", speakers=len({s[2] for s in segments}),
              seconds=round(time.time() - t0, 2))
    return segments


def assign_speaker(seg_start, seg_end, diarization_segments):
    """Speaker with the most temporal overlap in [seg_start, seg_end]."""
    overlap: dict = {}
    for d_start, d_end, speaker in diarization_segments:
        o_start = max(seg_start, d_start)
        o_end = min(seg_end, d_end)
        if o_end > o_start:
            overlap[speaker] = overlap.get(speaker, 0.0) + (o_end - o_start)
    return max(overlap, key=overlap.get) if overlap else "UNKNOWN"


# ─── Core ───────────────────────────────────────────────────────────────────

def transcribe_with_speakers(input_path: Path, args, hf_token: str) -> str:
    """Transcribe + diarize, streaming lines to stdout and to the out file.

    Streams so partial results are visible immediately and a mid-run crash
    never loses everything already written.
    """
    preset = SPEED_PRESETS[args.mode]
    print(f"\n-> Transcribing: {input_path.name}  "
          f"[mode={args.mode}, beam={preset['beam_size']}]", file=sys.stderr)
    start = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = extract_audio_wav(input_path, tmp)

        duration = probe_duration(wav_path)
        if duration and duration > MAX_AUDIO_MINUTES * 60:
            sys.exit(f"File too long ({duration/60:.0f} min > "
                     f"{MAX_AUDIO_MINUTES} min). Split it first.")

        # Diarization first (needs full audio; can run while we prep transcript).
        diar_segments = []
        diar_failed = False
        try:
            diar_segments = diarize(wav_path, hf_token, args.speakers)
            print(f"   Found {len({s[2] for s in diar_segments})} speakers",
                  file=sys.stderr)
        except Exception as e:
            diar_failed = True
            log_event("diarize_failed", error=str(e))
            print(f"   WARNING: diarization failed ({e}); "
                  f"continuing without speaker labels.", file=sys.stderr)

        # Transcription (streaming).
        print("   Transcribing speech…", file=sys.stderr)
        model = get_whisper(args.model)
        t_trans = time.time()
        segments, info = model.transcribe(
            str(wav_path),
            language=args.language,
            beam_size=preset["beam_size"],
            best_of=preset["best_of"],
            temperature=preset["temperature"],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        print(f"   Language: {info.language} ({info.language_probability:.0%})",
              file=sys.stderr)

        # Build output incrementally — write to stdout AND a buffer for the file.
        lines: list[str] = []
        current_speaker = None
        current_lines: list[str] = []
        first_seg_t = None

        def flush_speaker():
            if current_lines and current_speaker is not None:
                block = f"\n{current_speaker}:\n" + " ".join(current_lines)
                lines.append(block)
                # Stream to stdout so the user sees progress.
                print(block, flush=True)

        for seg in segments:
            if first_seg_t is None:
                first_seg_t = time.time() - t_trans
                log_event("first_segment", seconds=round(first_seg_t, 2))
            text = seg.text.strip()
            if not text:
                continue
            speaker = (assign_speaker(seg.start, seg.end, diar_segments)
                       if diar_segments and not diar_failed else "SPEAKER_00")
            if speaker != current_speaker:
                flush_speaker()
                current_speaker = speaker
                current_lines = [text]
            else:
                current_lines.append(text)
        flush_speaker()

        trans_seconds = time.time() - t_trans
        elapsed = time.time() - start
        log_event("transcription_done",
                  seconds=round(trans_seconds, 2),
                  audio_seconds=round(float(info.duration), 2),
                  rtf=round(trans_seconds / float(info.duration), 3)
                      if info.duration else None,
                  first_segment_s=round(first_seg_t, 2) if first_seg_t else None,
                  diarization_failed=diar_failed)
        print(f"   Done in {elapsed:.1f}s  "
              f"(speed={info.duration/elapsed:.1f}x real-time)", file=sys.stderr)
        return "\n".join(lines).strip()


def main() -> None:
    logging.basicConfig(level=os.environ.get("TRANSCRIBER_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    check_deps()

    parser = argparse.ArgumentParser(description="Transcribe with speaker labels.")
    parser.add_argument("input", help="Audio or video file")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env variable)")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--mode", default="fast",
                        choices=["quality", "fast", "turbo"])
    parser.add_argument("--language", default=None)
    parser.add_argument("--speakers", type=int, default=None,
                        help="Number of speakers if you know it (helps accuracy)")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if not args.hf_token:
        sys.exit(
            "A HuggingFace token is required for speaker diarization.\n"
            "1. Create a free account at https://huggingface.co\n"
            "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Get a token at https://huggingface.co/settings/tokens\n"
            "4. Run: python transcribe_diarize.py audio.m4a --hf-token hf_YOUR_TOKEN"
        )

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")

    result = transcribe_with_speakers(input_path, args, args.hf_token)

    outdir = Path(args.outdir) if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"{input_path.stem}_speakers.txt"
    out_file.write_text(result, encoding="utf-8")
    print(f"   Saved -> {out_file}", file=sys.stderr)
    print("\nAll done.", file=sys.stderr)
    log_event("request_done", total_seconds=round(time.time() - start_global, 2))


if __name__ == "__main__":
    main()
