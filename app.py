"""
app.py — Free Transcriber Web UI (optimized)
Run:  python app.py  then open  http://localhost:7860

Optimized vs. the original in:
  - Streaming partial results (segments yielded live, no list() blocking)
  - Single WAV extraction shared by Whisper + diarization (no double decode)
  - Thread-safe LRU models cache + per-model load lock
  - Concurrency limiter so CPU-bound runs don't thrash each other
  - In-process result cache keyed by file hash + params
  - Memory-efficient WAV read (soundfile, no librosa resample)
  - Diarization failure falls back to plain transcript (never lose work)
  - Structured JSON logging with time-to-first-segment / xRT / diarization time
  - ffmpeg timeout + quiet stderr
  - Cancel button + advanced VAD knobs in the UI

Speaker diarization uses librosa MFCCs + sklearn clustering — fully offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import subprocess
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import gradio as gr
from faster_whisper import WhisperModel

# ─── Configuration ───────────────────────────────────────────────────────────

class Config:
    SR = 16000
    # How many transcriptions may run at once. faster-whisper on CPU saturates
    # all cores, so >1 usually thrashes. Bump only on big-iron multi-core boxes.
    MAX_CONCURRENT = int(os.environ.get("TRANSCRIBER_MAX_CONCURRENT", "1"))
    # Keep at most N Whisper models resident (LRU eviction by size).
    MAX_MODELS_IN_CACHE = int(os.environ.get("TRANSCRIBER_MAX_MODELS", "2"))
    # Hard guard against runaway memory: refuse files longer than this (minutes).
    MAX_AUDIO_MINUTES = int(os.environ.get("TRANSCRIBER_MAX_AUDIO_MINUTES", "180"))
    FFMPEG_TIMEOUT = int(os.environ.get("TRANSCRIBER_FFMPEG_TIMEOUT", "600"))
    # Result cache (re-runs of the same file/params return instantly).
    RESULT_CACHE_ENTRIES = int(os.environ.get("TRANSCRIBER_RESULT_CACHE", "8"))
    # Throttle UI updates during streaming so we don't flood the browser.
    STREAM_YIELD_INTERVAL_S = 0.25


SPEED_PRESETS = {
    "quality  (most accurate, slowest)": dict(beam_size=5, best_of=5),
    "fast     (recommended, ~2x quicker)": dict(beam_size=2, best_of=1),
    "turbo    (quickest, minor accuracy drop)": dict(beam_size=1, best_of=1),
}

# ─── Logging / observability ────────────────────────────────────────────────

logger = logging.getLogger("transcriber")
logging.basicConfig(
    level=os.environ.get("TRANSCRIBER_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def log_event(event: str, **fields) -> None:
    """Emit one structured JSON line per request milestone."""
    rec = {"event": event, "ts": round(time.time(), 3), **fields}
    logger.info(json.dumps(rec, ensure_ascii=False))


# ─── Model cache (thread-safe, LRU) ─────────────────────────────────────────

class ModelCache:
    """LRU cache of WhisperModel instances with per-size load locking.

    The previous code used a bare dict with no locking — under concurrent
    Gradio requests two callers could simultaneously load the same model size,
    wasting memory and CPU. This fixes that and bounds residency.
    """

    def __init__(self, max_entries: int):
        self._lock = threading.Lock()
        self._models: "OrderedDict[str, WhisperModel]" = OrderedDict()
        self._load_locks: dict[str, threading.Lock] = {}
        self._max = max_entries

    def get(self, size: str) -> WhisperModel:
        # Fast path: already resident.
        with self._lock:
            if size in self._models:
                self._models.move_to_end(size)
                return self._models[size]
            load_lock = self._load_locks.setdefault(size, threading.Lock())

        # Load outside the global lock so a different size can be served
        # concurrently; but serialize loads of the *same* size.
        with load_lock:
            with self._lock:
                if size in self._models:
                    self._models.move_to_end(size)
                    return self._models[size]
            t0 = time.time()
            log_event("model_load_start", size=size)
            model = WhisperModel(size, device="cuda", compute_type="float16")
            log_event("model_load_end", size=size, seconds=round(time.time() - t0, 2))
            with self._lock:
                self._models[size] = model
                self._models.move_to_end(size)
                while len(self._models) > self._max:
                    evicted, _ = self._models.popitem(last=False)
                    log_event("model_evicted", size=evicted)
            return model


_model_cache = ModelCache(Config.MAX_MODELS_IN_CACHE)


def get_model(size: str) -> WhisperModel:
    return _model_cache.get(size)


# ─── Result cache ───────────────────────────────────────────────────────────

class ResultCache:
    def __init__(self, max_entries: int):
        self._lock = threading.Lock()
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._max = max_entries

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key, value) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)


_result_cache = ResultCache(Config.RESULT_CACHE_ENTRIES)


def file_hash(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


# ─── Concurrency limiter ────────────────────────────────────────────────────

# faster-whisper on CPU is CPU-bound and saturates all cores. Allowing N>1
# concurrent runs causes context-switch/cache thrash and slows *both* users.
# Serialize heavy work; cheap requests (cache hits) skip the semaphore.
_inference_sem = threading.Semaphore(Config.MAX_CONCURRENT)


# ─── Audio I/O ──────────────────────────────────────────────────────────────

def probe_duration(path: str) -> float:
    """Best-effort duration probe via ffprobe (used for progress + guards)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def extract_wav(input_path: str, out_path: str) -> None:
    """Decode anything ffmpeg understands -> 16kHz mono s16 WAV.

    Always called once and shared by both Whisper and diarization, which
    removes the original bug where Whisper read the raw upload while
    diarization read a separately-decoded WAV (double decode + possible
    timestamp drift between the two paths).
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-ac", "1", "-ar", str(Config.SR),
        "-sample_fmt", "s16",
        out_path,
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=Config.FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after {Config.FFMPEG_TIMEOUT}s")
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg error: {result.stderr.decode(errors='replace')[:2000]}"
        )
    log_event("ffmpeg_done", seconds=round(time.time() - t0, 2))


def read_wav_mono(wav_path: str) -> Tuple[np.ndarray, int]:
    """Read the 16kHz mono WAV as float32 WITHOUT librosa resampling.

    librosa.load() decodes + resamples + normalizes; since ffmpeg already
    produced 16kHz mono s16, that work is pure waste and also holds the whole
    signal in RAM twice. soundfile reads directly into a float32 view.
    """
    import soundfile as sf
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1, dtype="float32")
    return np.ascontiguousarray(y), int(sr)


# ─── Formatting helpers ─────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def render_plain(segs) -> str:
    return " ".join(s.text.strip() for s in segs if s.text.strip())


def render_timestamped(segs) -> str:
    return "\n".join(f"[{s.start:.1f}s] {s.text.strip()}"
                     for s in segs if s.text.strip())


def render_srt(segs) -> str:
    lines: List[str] = []
    for i, s in enumerate(segs, 1):
        t = s.text.strip()
        if not t:
            continue
        lines += [str(i),
                  f"{format_timestamp(s.start)} --> {format_timestamp(s.end)}",
                  t, ""]
    return "\n".join(lines)


def render_with_speakers(segs, diar_segs) -> str:
    """Group consecutive segments by speaker, like the original UI output."""
    lines: List[str] = []
    current_speaker = None
    buffer: List[str] = []
    for seg in segs:
        text = seg.text.strip()
        if not text:
            continue
        speaker = assign_speaker(seg.start, seg.end, diar_segs)
        if speaker != current_speaker:
            if buffer:
                lines.append(f"\n{current_speaker}:\n{' '.join(buffer)}")
            current_speaker = speaker
            buffer = [text]
        else:
            buffer.append(text)
    if buffer:
        lines.append(f"\n{current_speaker}:\n{' '.join(buffer)}")
    return "\n".join(lines).strip()


# ─── Diarization ────────────────────────────────────────────────────────────

def diarize_mfcc(wav_path: str, whisper_segments, num_speakers: Optional[int] = None):
    """Speaker diarization using MFCC embeddings + agglomerative clustering.

    Improvements:
      - Reads WAV via soundfile (no librosa resample/normalize waste)
      - Skips tiny segments (<0.3s)
      - Excludes MFCC 0 to reduce energy/loudness bias
      - Dynamically finds optimal cluster count via Silhouette Coefficient
      - Gracefully falls back to 1 speaker if Silhouette Score is low (<0.15)
    """
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    SR = Config.SR
    N_MFCC = 20
    MIN_DURATION = 0.3

    y, _ = read_wav_mono(wav_path)

    embeddings: List[np.ndarray] = []
    valid_segs = []
    for seg in whisper_segments:
        duration = seg.end - seg.start
        if duration < MIN_DURATION:
            continue
        start_i = int(seg.start * SR)
        end_i = int(seg.end * SR)
        chunk = y[start_i:end_i]
        if len(chunk) < int(MIN_DURATION * SR):
            continue
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=N_MFCC)
        # Exclude MFCC 0 (index 0) because it correlates with volume/loudness
        # rather than speaker identity characteristics.
        mfcc_feat = mfcc[1:]
        emb = np.concatenate([np.mean(mfcc_feat, axis=1), np.std(mfcc_feat, axis=1)])
        embeddings.append(emb)
        valid_segs.append(seg)

    if not valid_segs:
        return []
    if len(valid_segs) == 1:
        return [(valid_segs[0].start, valid_segs[0].end, "SPEAKER_00")]

    X = StandardScaler().fit_transform(np.asarray(embeddings))

    n = int(num_speakers) if num_speakers and int(num_speakers) > 1 else None

    if n is None:
        # Determine the number of speakers automatically using the Silhouette Coefficient
        max_k = min(8, len(X) - 1)
        if max_k < 2:
            n_clusters = 1
        else:
            best_score = -1
            best_k = 1
            for k in range(2, max_k + 1):
                clustering = AgglomerativeClustering(
                    n_clusters=k,
                    metric="euclidean",
                    linkage="ward",
                )
                cluster_labels = clustering.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # If the best clustering has very low silhouette score, assume a single speaker
            if best_score < 0.15:
                n_clusters = 1
            else:
                n_clusters = best_k
    else:
        n_clusters = n

    if n_clusters <= 1:
        labels = np.zeros(len(X), dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="ward",
        )
        labels = clustering.fit_predict(X)

    return [(seg.start, seg.end, f"SPEAKER_{label:02d}")
            for seg, label in zip(valid_segs, labels)]


def assign_speaker(seg_start: float, seg_end: float, diar_segs) -> str:
    """Speaker with the most temporal overlap in [seg_start, seg_end]."""
    overlap: dict = {}
    for d_start, d_end, spk in diar_segs:
        o = min(seg_end, d_end) - max(seg_start, d_start)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0.0) + o
    return max(overlap, key=overlap.get) if overlap else "SPEAKER_00"


# ─── Core transcription (streaming generator) ───────────────────────────────

def _build_cache_key(fhash, model_size, speed_label, lang, fmt,
                     diar, n_spk, vad_sil, vad_pad):
    return (fhash, model_size, speed_label, lang, fmt, diar, n_spk, vad_sil, vad_pad)


def run_transcription(
    audio_file, model_size, speed_label, language,
    output_format, enable_diarization, num_speakers,
    vad_silence_ms, vad_pad_ms,
    progress=gr.Progress(),
):
    """Streaming generator: yields partial transcript as segments arrive,
    then the final formatted transcript + download path at the end.
    """
    if audio_file is None:
        yield "Please upload an audio or video file.", None
        return

    lang = (language or "").strip().lower() or None
    preset = SPEED_PRESETS[speed_label]
    n_spk_req = int(num_speakers) if num_speakers and int(num_speakers) > 0 else None
    req_t0 = time.time()
    log_event("request_start", model=model_size, mode=speed_label,
              language=lang, format=output_format, diar=enable_diarization,
              n_speakers=n_spk_req)

    # ── Cache hit short-circuit (skips the CPU semaphore) ──
    fhash = file_hash(audio_file)
    key = _build_cache_key(fhash, model_size, speed_label, lang, output_format,
                           enable_diarization, n_spk_req, vad_silence_ms, vad_pad_ms)
    cached = _result_cache.get(key) if fhash else None
    if cached is not None:
        log_event("cache_hit", key=str(key))
        progress(1.0, desc="Cached result")
        yield cached[0], cached[1]
        return

    # ── Serialize heavy CPU work ──
    with _inference_sem:
        try:
            for partial in _transcribe_stream(
                audio_file, model_size, preset, lang, output_format,
                enable_diarization, n_spk_req, vad_silence_ms, vad_pad_ms,
                progress, req_t0, key,
            ):
                yield partial
        except Exception as e:
            import traceback
            log_event("request_error", error=str(e))
            yield f"Error: {e}\n\n{traceback.format_exc()}", None


def _transcribe_stream(
    audio_file, model_size, preset, lang, output_format,
    enable_diarization, n_spk_req, vad_silence_ms, vad_pad_ms,
    progress, req_t0, cache_key,
):
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")

        # 1) Single shared WAV extraction (fixes the double-decode bug).
        progress(0.06, desc="Extracting audio…")
        try:
            extract_wav(audio_file, wav_path)
        except Exception as e:
            yield f"Audio extraction failed: {e}", None
            return

        duration = probe_duration(wav_path)
        if duration and duration > Config.MAX_AUDIO_MINUTES * 60:
            yield (f"File too long ({duration/60:.0f} min > "
                   f"{Config.MAX_AUDIO_MINUTES} min limit). Split it first."), None
            return

        # 2) Load model (cached) and start transcription.
        progress(0.10, desc="Loading Whisper model…")
        model = get_model(model_size)

        progress(0.15, desc="Transcribing…")
        t_trans_start = time.time()
        segments, info = model.transcribe(
            wav_path,
            language=lang,
            beam_size=preset["beam_size"],
            best_of=preset["best_of"],
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=int(vad_silence_ms),
                speech_pad_ms=int(vad_pad_ms),
            ),
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        log_event("language_detected", language=info.language,
                  probability=round(float(info.language_probability), 3),
                  duration=round(float(info.duration), 2))

        # 3) STREAM segments — the single biggest latency/UX win.
        collected: List = []
        preview_lines: List[str] = []
        t_first = None
        last_yield = 0.0

        for seg in segments:
            if t_first is None:
                t_first = time.time() - t_trans_start
                log_event("first_segment", seconds=round(t_first, 2),
                          mode=model_size)
            text = seg.text.strip()
            collected.append(seg)
            if output_format == "Timestamped text":
                preview_lines.append(f"[{seg.start:.1f}s] {text}")
            else:
                preview_lines.append(text)

            now = time.time()
            throttle = now - last_yield >= Config.STREAM_YIELD_INTERVAL_S
            if throttle or len(collected) == 1:
                progress(
                    min(0.92, 0.15 + 0.77 * (seg.end / duration if duration else 0)),
                    desc=(f"Transcribing… {seg.end/60:.1f}/{duration/60:.1f} min"
                          if duration else "Transcribing…"),
                )
                yield "\n".join(preview_lines), None
                last_yield = now

        trans_seconds = time.time() - t_trans_start
        log_event("transcription_done",
                  seconds=round(trans_seconds, 2),
                  audio_seconds=round(float(info.duration), 2),
                  rtf=round(trans_seconds / float(info.duration), 3)
                      if info.duration else None,
                  segments=len(collected),
                  first_segment_s=round(t_first, 2) if t_first else None)

        if not collected:
            yield "No speech detected.", None
            return

        # 4) Optional diarization (failure -> fall back to plain, never lose work).
        diar_segs = []
        if enable_diarization:
            progress(0.94, desc="Detecting speakers…")
            try:
                diar_segs = diarize_mfcc(wav_path, collected, n_spk_req)
                log_event("diarization_done",
                          speakers=len({s[2] for s in diar_segs}),
                          seconds=round(time.time() - t_trans_start - trans_seconds, 2))
            except Exception as e:
                log_event("diarization_failed", error=str(e))

        # 5) Final render in the chosen format.
        if enable_diarization and diar_segs:
            transcript_text = render_with_speakers(collected, diar_segs)
        elif output_format == "SRT subtitles":
            transcript_text = render_srt(collected)
        elif output_format == "Timestamped text":
            transcript_text = render_timestamped(collected)
        else:
            transcript_text = render_plain(collected)

        ext = "srt" if output_format == "SRT subtitles" else "txt"
        stable_path = os.path.join(
            tempfile.gettempdir(), f"transcript_{int(time.time())}.{ext}"
        )
        with open(stable_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        n_spk = len({s[2] for s in diar_segs}) if diar_segs else 0
        summary = (
            f"Language: {info.language.upper()} "
            f"({info.language_probability:.0%}) | "
            f"Audio: {info.duration/60:.1f} min"
            + (f" | {n_spk} speakers" if n_spk else "")
            + f" | {trans_seconds:.1f}s ({info.duration/trans_seconds:.1f}x)"
        )
        progress(1.0, desc="Done!")
        result = (f"{summary}\n\n{transcript_text}", stable_path)

        # Cache for instant re-runs.
        if cache_key is not None:
            _result_cache.put(cache_key, result)

        log_event("request_done",
                  total_seconds=round(time.time() - req_t0, 2))
        yield result


# ── UI ───────────────────────────────────────────────────────────────────────

css = """
.gradio-container { max-width: 860px !important; margin: 0 auto; }
#title  { text-align: center; padding: 24px 0 4px; }
#sub    { text-align: center; color: #64748b; margin-bottom: 20px; font-size:.95rem; }
#run-btn { background: #1e293b !important; border: none !important; }
#run-btn:hover { background: #334155 !important; }
"""

theme = gr.themes.Base(
    primary_hue="slate", neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(theme=theme, css=css) as demo:
    gr.HTML('<h1 id="title">Free Transcriber</h1>')
    gr.HTML('<p id="sub">Powered by OpenAI Whisper · runs 100% on your machine · '
            'no subscriptions · live partial results</p>')

    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.File(
                label="Audio or Video File",
                file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg",
                            ".mp4", ".mkv", ".mov", ".avi", ".webm"],
            )
        with gr.Column(scale=2):
            model_size = gr.Dropdown(
                ["tiny", "base", "small", "medium", "large-v3"],
                value="small", label="Model size",
                info="larger = more accurate, slower",
            )
            speed_choice = gr.Dropdown(
                list(SPEED_PRESETS.keys()),
                value=list(SPEED_PRESETS.keys())[1],
                label="Speed mode",
            )
            language = gr.Textbox(
                label="Language (optional)",
                placeholder="e.g. en, sw, fr — blank = auto-detect",
                max_lines=1,
            )
            output_format = gr.Radio(
                ["Plain text", "Timestamped text", "SRT subtitles"],
                value="Plain text", label="Output format",
            )

    with gr.Accordion("Speaker identification — who said what", open=False):
        gr.Markdown(
            "Labels each speaker as SPEAKER_00, SPEAKER_01, etc.\n\n"
            "**No token or internet needed** — uses MFCC voice fingerprinting + "
            "clustering, runs fully offline."
        )
        enable_diarization = gr.Checkbox(
            label="Enable speaker identification", value=False
        )
        num_speakers = gr.Number(
            label="Number of speakers (0 = auto-detect)",
            value=0, minimum=0, maximum=20, precision=0,
        )

    with gr.Accordion("Advanced (VAD / performance)", open=False):
        vad_silence_ms = gr.Number(
            label="Min silence to split (ms)", value=300,
            minimum=50, maximum=2000, precision=0,
            info="Lower = more, shorter segments. Higher = fewer, longer ones.",
        )
        vad_pad_ms = gr.Number(
            label="Speech padding (ms)", value=200,
            minimum=0, maximum=500, precision=0,
            info="Audio kept around each speech segment to avoid clipping.",
        )

    with gr.Row():
        run_btn = gr.Button("Transcribe", variant="primary", elem_id="run-btn")
        cancel_btn = gr.Button("Cancel", variant="stop")

    output_text = gr.Textbox(label="Transcript", lines=18, interactive=False)
    download_file = gr.File(label="Download transcript")

    inputs = [audio_input, model_size, speed_choice, language,
              output_format, enable_diarization, num_speakers,
              vad_silence_ms, vad_pad_ms]

    run_event = run_btn.click(
        fn=run_transcription,
        inputs=inputs,
        outputs=[output_text, download_file],
    )
    cancel_btn.click(fn=None, cancels=[run_event])

    gr.Markdown(
        "---\nFirst run downloads the Whisper model (~240 MB for `small`). "
        "Every run after is fully offline. Partial results stream live while "
        "transcribing."
    )


if __name__ == "__main__":
    log_event("server_start",
              max_concurrent=Config.MAX_CONCURRENT,
              max_models=Config.MAX_MODELS_IN_CACHE)
    demo.launch(
        share=False, server_name="0.0.0.0",
        server_port=7860, theme=theme, css=css,
        # Prevent queue from silently dropping long jobs; allow one active +
        # a small queue. Tune to your hardware.
        max_threads=Config.MAX_CONCURRENT + 1,
    )
