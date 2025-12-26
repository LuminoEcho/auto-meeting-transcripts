import os, sys, time, subprocess, pathlib, warnings, hashlib, json, traceback
from datetime import datetime

# ----- Third-party -----
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ===== Load config =====
try:
    CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
except Exception as e:
    print("Failed to read config.yaml:", e)
    sys.exit(1)

WATCH_FOLDER    = pathlib.Path(CFG["watch_folder"]).expanduser()
OUTPUT_ROOT     = pathlib.Path(CFG.get("output_root", str(WATCH_FOLDER / "output")))
DATE_FOLDERS    = bool(CFG.get("date_folders", True))

# File types the watcher should react to (configurable)
def _normalize_ext_list(seq):
    out = set()
    for ext in (seq or []):
        if not isinstance(ext, str):
            continue
        e = ext.lower().strip()
        if not e:
            continue
        if not e.startswith('.'):
            e = '.' + e
        out.add(e)
    return out

_DEFAULT_WATCH_EXTS = [
    # video
    ".mkv", ".mp4", ".mov", ".webm",
    # audio
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
]
WATCH_EXTENSIONS = _normalize_ext_list(CFG.get("watch_extensions", _DEFAULT_WATCH_EXTS))

FFMPEG          = CFG.get("ffmpeg_path", "ffmpeg")
AUDIO_FORMAT    = CFG.get("audio_format", "wav")

WHISPER_MODEL   = CFG.get("whisper_model", "base")
LANGUAGE        = CFG.get("language", None)

GEMINI_MODEL    = CFG.get("gemini_model", "gemini-2.5-pro")
SUMMARY_PROMPT  = CFG["summary_prompt"]
GEMINI_API_KEY  = (os.environ.get("GEMINI_API_KEY") or CFG.get("gemini_api_key") or "").strip()
GEMINI_MAX_RETRIES = int(CFG.get("gemini_max_retries", 3))
GEMINI_RETRY_DELAY = int(CFG.get("gemini_retry_delay_sec", 60))
GEMINI_PROMPT_ON_FAIL = bool(CFG.get("gemini_prompt_on_fail", True))
GEMINI_FALLBACK_MODELS = [
    str(x).strip() for x in CFG.get("gemini_fallback_models", [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]) if str(x).strip()
]

ENABLE_DIAR     = bool(CFG.get("enable_diarization", True))
PYANNOTE_PIPE   = CFG.get("pyannote_pipeline", "pyannote/speaker-diarization-3.1")
HF_TOKEN        = (os.environ.get(CFG.get("hf_token_env", "HF_TOKEN")) or CFG.get("hf_token_value") or "").strip()

# Dedupe / idempotency
SKIP_IF_PROCESSED = bool(CFG.get("skip_if_processed", True))
DEDUP_CACHE_DIR   = pathlib.Path(CFG.get("dedupe_cache_dir", str(OUTPUT_ROOT / ".cache")))
DEDUP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

if CFG.get("suppress_pooling_warning", True):
    warnings.filterwarnings(
        "ignore",
        message=r"std\(\): degrees of freedom is <= 0",
        module=r"pyannote\.audio\.models\.blocks\.pooling",
    )

# ===== Gemini client =====
try:
    from google import genai
except ImportError:
    print("Missing google-genai. Install with: pip install google-genai")
    sys.exit(1)

if not GEMINI_API_KEY:
    print("ERROR: Gemini API key not set. Set GEMINI_API_KEY env var or config.gemini_api_key.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

# ===== Utility =====
def wait_for_complete_write(path: pathlib.Path, sleep_s=1.0, checks=5):
    """Wait until file size stops growing (OBS just finished writing)."""
    size = -1
    stable = 0
    while stable < checks:
        try:
            new_size = path.stat().st_size
        except FileNotFoundError:
            time.sleep(sleep_s)
            continue
        if new_size == size:
            stable += 1
        else:
            stable = 0
            size = new_size
        time.sleep(sleep_s)

def ensure_output_folder() -> pathlib.Path:
    if DATE_FOLDERS:
        today = datetime.now()
        out = OUTPUT_ROOT / f"{today:%Y}" / f"{today:%Y-%m-%d}"
    else:
        out = OUTPUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    return out

def _fast_file_hash(path: pathlib.Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    """
    Quick fingerprint: sha256(size || first 16MB || mtime_ns).
    Fast and stable across renames/locations.
    """
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as f:
        chunk = f.read(max_bytes)
        h.update(chunk)
    h.update(str(path.stat().st_mtime_ns).encode())
    return h.hexdigest()

def _fingerprint_marker_path(fingerprint: str) -> pathlib.Path:
    return DEDUP_CACHE_DIR / f"{fingerprint}.done"

def was_already_processed(video_path: pathlib.Path) -> bool:
    if not SKIP_IF_PROCESSED:
        return False
    try:
        fp = _fast_file_hash(video_path)
        return _fingerprint_marker_path(fp).exists()
    except Exception:
        # On any fingerprinting issue, don't skip
        return False

def mark_processed(video_path: pathlib.Path, outputs: dict):
    if not SKIP_IF_PROCESSED:
        return
    try:
        fp = _fast_file_hash(video_path)
        marker = _fingerprint_marker_path(fp)
        marker.write_text(json.dumps({
            "video": str(video_path),
            "size": video_path.stat().st_size,
            "mtime_ns": video_path.stat().st_mtime_ns,
            "outputs": outputs,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print("Warning: could not write dedupe marker:", e)

def run_ffmpeg_extract_wav(input_video: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Create 16 kHz, mono, 16-bit PCM WAV (good for both Whisper & pyannote)."""
    out_wav = out_dir / (input_video.stem + f".{AUDIO_FORMAT}")
    cmd = [FFMPEG, "-y", "-i", str(input_video), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out_wav)]
    print("FFmpeg:", " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print("FFmpeg error:\n", cp.stderr or cp.stdout)
        raise RuntimeError("FFmpeg failed.")
    return out_wav

# ===== Diarization =====
_diar_pipeline = None
_diar_device_msg_printed = False

def diarize(wav_path: pathlib.Path, out_dir: pathlib.Path):
    """Run pyannote diarization; returns (annotation, rttm_path) or (None, None)."""
    global _diar_pipeline, _diar_device_msg_printed
    if not ENABLE_DIAR:
        return None, None

    if not HF_TOKEN:
        raise RuntimeError("HF token missing: set HF_TOKEN env var or config.hf_token_value.")

    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as e:
        raise RuntimeError("pyannote.audio (and torch) not installed. pip install pyannote.audio") from e

    if _diar_pipeline is None:
        print("Loading diarization pipeline:", PYANNOTE_PIPE)
        _diar_pipeline = Pipeline.from_pretrained(PYANNOTE_PIPE, use_auth_token=HF_TOKEN)
        # Push to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _diar_pipeline.to(device)
        if not _diar_device_msg_printed:
            print("Diarization device:", device)
            _diar_device_msg_printed = True

    print("Running diarization...")
    diar = _diar_pipeline(str(wav_path))

    # Save RTTM for reference/debugging
    rttm_path = out_dir / (wav_path.stem + ".rttm")
    with open(rttm_path, "w", encoding="utf-8") as f:
        diar.write_rttm(f)
    return diar, rttm_path

def assign_speakers_to_segments(diar, segments):
    """
    Map Whisper segments to speakers:
    - take the midpoint of each segment
    - find diarization turn containing that midpoint
    """
    if diar is None:
        return [("SPK?", seg.get("text", "").strip()) for seg in segments]

    diar_turns = []
    # diar.itertracks(yield_label=True) => (segment, track, label)
    for track, _, label in diar.itertracks(yield_label=True):
        diar_turns.append((float(track.start), float(track.end), str(label)))
    diar_turns.sort()

    tagged = []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        mid = 0.5 * (s + e)
        spk = "SPK?"
        for ds, de, lab in diar_turns:
            if ds <= mid <= de:
                spk = lab
                break
        tagged.append((spk, seg.get("text", "").strip()))
    return tagged

# ===== Whisper (GPU if available) =====
def transcribe_with_whisper(audio_path: pathlib.Path):
    try:
        import whisper, torch
    except ImportError as e:
        raise RuntimeError("openai-whisper not installed. pip install openai-whisper") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"Transcribing with Whisper ({WHISPER_MODEL}) on {device}...")
    # ask for segment timestamps so we can fuse with diarization
    result = model.transcribe(str(audio_path), language=LANGUAGE, verbose=False)
    return result  # dict with ["text"] and ["segments"]

# ===== Gemini summary =====
def _log_gemini_error(err: BaseException, attempt: int, model: str, context: str = ""):
    try:
        ts = datetime.now().isoformat(timespec="seconds")
        lines = [
            f"[{ts}] Gemini error (attempt {attempt})",
            f"  model: {model}",
            f"  type: {type(err).__name__}",
            f"  message: {str(err)}",
        ]
        if context:
            lines.append(f"  context: {context}")
        tb = traceback.format_exc()
        if tb:
            lines.append("  traceback:")
            lines.append(tb)
        log_path = DEDUP_CACHE_DIR / "gemini_errors.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print("Warning: failed to write gemini error log:", e)

def summarize_with_gemini(transcript_text: str) -> str:
    # Fill the NOW_WIB placeholder with current time in WIB (UTC+07:00)
    from datetime import timezone, timedelta
    wib_tz = timezone(timedelta(hours=7), name="WIB")
    now_wib = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M WIB")
    prompt_filled = (SUMMARY_PROMPT or "").replace("{{NOW_WIB}}", now_wib)

    contents = [{"role": "user", "parts": [{"text": prompt_filled + "\n\nTranscript:\n" + transcript_text}]}]

    model = GEMINI_MODEL
    while True:
        for attempt in range(1, max(GEMINI_MAX_RETRIES, 0) + 1):
            try:
                resp = client.models.generate_content(model=model, contents=contents)
                return (getattr(resp, "text", "") or "").strip()
            except Exception as e:
                _log_gemini_error(e, attempt, model, context="generate_content")
                print(f"Gemini error on attempt {attempt}/{GEMINI_MAX_RETRIES}: {e}")
                if attempt < GEMINI_MAX_RETRIES:
                    print(f"Retrying in {GEMINI_RETRY_DELAY} seconds...")
                    time.sleep(GEMINI_RETRY_DELAY)
                else:
                    break

        if not GEMINI_PROMPT_ON_FAIL:
            print("Gemini summarization failed after max retries; continuing with empty summary.")
            return ""

        print("Gemini summarization failed after max retries.")
        print("Options: [r]etry, [s]witch model, [c]ancel")
        while True:
            choice = input("Enter r/s/c: ").strip().lower()
            if choice.startswith("r"):
                # retry another block with the same model
                break
            elif choice.startswith("s"):
                if GEMINI_FALLBACK_MODELS:
                    print("Available Gemini models:")
                    for i, m in enumerate(GEMINI_FALLBACK_MODELS, start=1):
                        print(f"  {i}) {m}")
                prompt = "Enter number to select, a custom model name, or leave blank to keep current: "
                sel = input(prompt).strip()
                chosen = None
                if sel.isdigit():
                    idx = int(sel)
                    if 1 <= idx <= len(GEMINI_FALLBACK_MODELS):
                        chosen = GEMINI_FALLBACK_MODELS[idx - 1]
                elif sel:
                    chosen = sel
                if chosen:
                    model = chosen
                    print(f"Switched model to '{model}'. Retrying...")
                    break
                else:
                    print("Model unchanged.")
            elif choice.startswith("c"):
                print("User chose to cancel summarization.")
                return ""
            else:
                print("Please enter 'r', 's', or 'c'.")

# ===== Main processing =====
def process_video(video_path: pathlib.Path):
    print(f"\n=== Processing: {video_path.name} ===")
    if was_already_processed(video_path):
        print("Already processed. Skipping:", video_path.name)
        return
    wait_for_complete_write(video_path)
    out_dir = ensure_output_folder()

    # 1) Extract/normalize audio
    wav = run_ffmpeg_extract_wav(video_path, out_dir)

    # 2) Diarize (who spoke when)
    diar, rttm_path = (None, None)
    if ENABLE_DIAR:
        diar, rttm_path = diarize(wav, out_dir)

    # 3) Transcribe (what was said)
    asr = transcribe_with_whisper(wav)
    segments = asr.get("segments", [])
    transcript_plain = (asr.get("text") or "").strip()

    # 4) Fuse speakers into the transcript lines
    tagged_segments = assign_speakers_to_segments(diar, segments)
    lines = [f"[{spk}] {txt}" for spk, txt in tagged_segments if txt]
    transcript_speakers = "\n".join(lines).strip() or transcript_plain

    # 5) Summarize with Gemini (use speaker-tagged version for clarity)
    summary_md = summarize_with_gemini(transcript_speakers)

    # 6) Save outputs
    stem = video_path.stem
    out_plain    = out_dir / f"{stem}.transcript.txt"
    out_speakers = out_dir / f"{stem}.transcript_speakers.txt"
    out_summary  = out_dir / f"{stem}.summary.md"

    out_plain.write_text(transcript_plain, encoding="utf-8")
    out_speakers.write_text(transcript_speakers, encoding="utf-8")
    out_summary.write_text(summary_md, encoding="utf-8")

    if rttm_path:
        print(f"RTTM saved: {rttm_path}")
    print("Saved to:", out_dir)

    # Write the dedupe marker pointing to outputs
    mark_processed(video_path, {
        "out_dir": str(out_dir),
        "transcript_plain": str(out_plain),
        "transcript_speakers": str(out_speakers),
        "summary_md": str(out_summary),
        "rttm": str(rttm_path) if rttm_path else None,
    })

# ===== Watcher =====
class NewVideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        p = pathlib.Path(event.src_path)
        if p.suffix.lower() in WATCH_EXTENSIONS:
            try:
                process_video(p)
            except Exception as e:
                print("ERROR while processing:", e)

def main():
    if not WATCH_FOLDER.exists():
        print(f"Watch folder not found: {WATCH_FOLDER}")
        sys.exit(1)
    print(f"Watching: {WATCH_FOLDER}")
    print("Extensions:", ", ".join(sorted(WATCH_EXTENSIONS)))
    observer = Observer()
    observer.schedule(NewVideoHandler(), str(WATCH_FOLDER), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
