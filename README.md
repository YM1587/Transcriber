# Free Transcriber (optimized)

100% offline speech-to-text with optional speaker identification.
- Web UI: `app.py` (Gradio, port 7860) — uses faster-whisper + offline MFCC diarization (no token needed).
- CLI:    `transcribe_diarize.py` — uses faster-whisper + pyannote diarization (needs a free HuggingFace token).

## Quick start (local)

```bash
# 1. install ffmpeg (one time) — macOS:  brew install ffmpeg
#                                Ubuntu: sudo apt install ffmpeg
#                                Win:    choco install ffmpeg
pip install -r requirements.txt

# 2. run the web UI
python app.py
# open http://localhost:7860
```

First run downloads the Whisper model (~240 MB for `small`). After that it's fully offline.

## Quick start (Docker)

```bash
docker build -t transcriber .
# mount a volume so the model download is reused next time:
docker run -p 7860:7860 -v whisper-models:/root/.cache/huggingface transcriber
# open http://localhost:7860
```

## CLI with pyannote diarization (optional, needs HF token)

```bash
# 1. create a free account at https://huggingface.co
# 2. accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
# 3. get a token at https://huggingface.co/settings/tokens
pip install pyannote.audio torch
python transcribe_diarize.py interview.m4a --hf-token hf_YOUR_TOKEN --mode fast
```

## Runtime knobs (env vars)

| Variable | Default | What it does |
|----------|---------|--------------|
| `TRANSCRIBER_MAX_CONCURRENT` | 1 | Max simultaneous transcriptions (raise on big multi-core boxes) |
| `TRANSCRIBER_MAX_MODELS` | 2 | Whisper models kept in RAM (LRU eviction) |
| `TRANSCRIBER_MAX_AUDIO_MINUTES` | 180 | Refuses files longer than this (memory guard) |
| `TRANSCRIBER_FFMPEG_TIMEOUT` | 600 | ffmpeg seconds before giving up |
| `TRANSCRIBER_RESULT_CACHE` | 8 | Re-runs of the same file+params return instantly |
| `TRANSCRIBER_LOG_LEVEL` | INFO | Set to DEBUG for verbose logs |

## What's new in this optimized version

- **Live partial results** — text appears as it's transcribed, no more waiting for the whole file.
- **Faster** — single audio decode (was decoding twice when diarization was on).
- **Lower memory** — reads WAV directly via soundfile (no resample waste).
- **Stable under load** — thread-safe model cache + concurrency limiter.
- **Never loses work** — diarization failure falls back to plain transcript.
- **Observable** — structured JSON logs (time-to-first-token, real-time factor, diarization time).
- **Safer** — ffmpeg timeout, max-duration guard, Docker healthcheck, Cancel button.
- **Instant re-runs** — result cache keyed on file hash + params.

All transcription-accuracy defaults are unchanged — same quality, better speed/reliability.
