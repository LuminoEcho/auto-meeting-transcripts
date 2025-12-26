# Auto Meeting Transcripts

Pipeline that watches a folder for new audio/video files, converts them to audio, transcribes, diarizes (optional), and summarizes meetings. Works best on Windows via the included runner script.

## Prerequisites
- Python 3.9+
- FFmpeg on `PATH`
- Recommended: Git Credential Manager on Windows
- API keys in env or `config.yaml`: `GEMINI_API_KEY` (required for summaries), `HF_TOKEN` (for diarization with pyannote)

## Quick Start (Windows)
```powershell
./run_pipeline.bat           # creates .venv, installs deps, starts watcher
./run_pipeline.bat --skip-install  # reuse existing env
./run_pipeline.bat --no-prompt     # non-interactive Gemini error handling
```
Drop supported media into `input/` (or the folder set by `watch_folder` in `config.yaml`). The watcher writes outputs under `output/YYYY/YYYY-MM-DD/`.

## Configuration
Edit `config.yaml` to adjust:
- `watch_folder`: directory to monitor (default `input/`)
- `watch_extensions`: allowed file types (video/audio)
- `whisper_model`: transcription model size (e.g., `tiny`, `base`)
- Gemini retry behavior: `gemini_max_retries`, `gemini_retry_delay_sec`, `gemini_prompt_on_fail`, `gemini_fallback_models`
- Diarization: enable pyannote, provide `HF_TOKEN`, choose model

## Outputs
For each processed file (date-stamped directory):
- `*.wav`: extracted audio
- `*.transcript.txt`: raw transcript
- `*.transcript_speakers.txt`: transcript with speaker tags (if diarization)
- `*.summary.md`: meeting summary
- `*.rttm`: optional diarization metadata

Cache and error logs live in `output/.cache/` (e.g., `gemini_errors.log`).

## Notes
- Avoid committing media in `input/` and results in `output/`.
- To run manually without the batch file: `python meetings_pipeline.py` (uses `config.yaml`).
- For constrained machines, set `whisper_model` to a smaller size or CPU-only Torch wheel via `TORCH_INDEX_URL`.
