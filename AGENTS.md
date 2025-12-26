# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `meetings_pipeline.py` (watcher + pipeline), `config.yaml` (runtime settings), `plan.txt` (notes).
- Inputs: drop supported media in `input/` (configured by `config.yaml: watch_folder`).
  - File types are configurable via `config.yaml: watch_extensions` (default includes common video: `.mkv/.mp4/.mov/.webm` and audio: `.wav/.mp3/.m4a/.aac/.flac/.ogg/.opus/.wma`).
- Outputs: `output/YYYY/YYYY-MM-DD/` contains `*.wav`, `*.transcript.txt`, `*.transcript_speakers.txt`, `*.summary.md`, and optional `*.rttm`. A fingerprint cache lives in `output/.cache/`.

## Build, Test, and Development Commands
- Python: 3.9+ recommended. Requires FFmpeg on PATH.
- Install deps (example): `pip install pyyaml watchdog google-genai openai-whisper pyannote.audio torch`
  - Tip: Install the correct `torch` build for your OS/GPU from pytorch.org.
- Environment: set `GEMINI_API_KEY` and (if diarization enabled) `HF_TOKEN`, or fill them in `config.yaml`.
- Run locally: `python meetings_pipeline.py` (uses `config.yaml`, watches `watch_folder`). Drop a file into `input/` to process.

## Gemini Error Handling
- Retries: configure `gemini_max_retries` (default 3) and `gemini_retry_delay_sec` (default 60).
- Interactive prompt: when retries are exhausted and `gemini_prompt_on_fail` is `true`, choose to retry, switch model, or cancel.
- Model switching: preset options come from `gemini_fallback_models` (defaults include `gemini-2.5-flash` and `gemini-2.5-flash-lite`), or type a custom model name.
- Logging: detailed failures append to `output/.cache/gemini_errors.log` with timestamp, model, error type/message, and traceback.
 - Blocking prompt: the interactive prompt runs in the watcher processing path and blocks new file handling until you answer. To avoid blocking, set `gemini_prompt_on_fail: false` to skip prompts and continue with an empty summary on failure.

### Windows Runner
- Preferred on Windows: `run_pipeline.bat` in the repo root.
  - First run (create `.venv`, install deps, start watcher): `./run_pipeline.bat`
  - Reuse env, skip install: `./run_pipeline.bat --skip-install`
  - Non-interactive (no prompts): `./run_pipeline.bat --no-prompt`
  - Forward extra args to Python: `./run_pipeline.bat --skip-install --your-arg value`
  - Torch source: set `TORCH_INDEX_URL` to pick wheel index (e.g. CPU-only)
    - Example: `set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu && ./run_pipeline.bat`

## Coding Style & Naming Conventions
- Language: Python. Use 4‑space indentation, `snake_case` for functions/variables, and type hints where practical.
- Keep side‑effects isolated; prefer small, testable functions. Avoid blocking the watchdog thread.
- New utilities: colocate in `meetings_pipeline.py` or extract to a small module when it improves clarity.

## Testing Guidelines
- No test suite yet. If adding tests, use `pytest` with files under `tests/` named `test_*.py`.
- Mock external tools/services (FFmpeg, Whisper, pyannote, Gemini). Validate that functions produce expected paths/text for small fixtures.

## Commit & Pull Request Guidelines
- Commits: present tense, scoped changes. Conventional Commit style is welcome: `feat: add diarization cache`, `fix: guard ffmpeg errors`.
- PRs: include a clear description, linked issue (if any), and sample before/after outputs (e.g., a few lines of transcript/summary). Note config or env changes.

## Security & Configuration Tips
- Do not commit API keys, tokens, or large media. Use environment variables and keep `output/` and `input/` out of commits.
- Verify `ffmpeg` availability and GPU drivers if using CUDA. Choose lighter models for constrained machines (`whisper_model: tiny/base`).
