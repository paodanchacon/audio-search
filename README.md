# 🎧 Audio Detection & Search Sandbox

This directory contains the audio classification pipeline, interactive exploration interfaces, and semantic query utilities.

---

## 📁 Organized Directory Structure

The files are structured as follows:

```
audio-search/
├── audio/                      # Audio files (.wav, .mp3, etc.)
├── video/                      # Video files (.mp4, etc.)
├── results/                    # Exported classification metadata (.json)
│
├── docs/                       # Component Guides
│   ├── classify_segments_guide.md    # Guide for the CLI & segment classification module
│   └── classifier_explorer_guide.md  # Guide for the Marimo explorer notebook
│
├── notebooks/                  # Interactive Marimo Notebooks
│   ├── audio_search.py               # Transcription & semantic search (Whisper + Ollama RAG)
│   ├── audio_segment_classifier.py    # Batch segment classifier with metadata JSON exporter
│   ├── classifier_explorer.py        # Reactive audio waveform & spectrogram explorer/classifier
│   └── test_dataset.py               # Clip-classification explorer for standard datasets
│
├── src/                        # Modular Source Code
│   ├── classify_segments.py          # Core audio segment classification and video extraction functions
│   └── label_audio.py                # Standalone segment labeling validation script
│
├── audio_mcp_server.py         # MCP Server exposing transcripts to LLM clients (Claude Desktop)
├── transcript.json             # Transcribed text with timestamps (shared artifact)
├── pyproject.toml              # Project dependencies and environment config
└── README.md                   # This overview guide
```

---

## 🚀 How to Run the Notebooks

Launch any interactive **Marimo** notebook from the `audio-search` directory:

```bash
# 1. Reactive Waveform & Spectrogram Explorer
uv run marimo edit notebooks/classifier_explorer.py

# 2. Batch Segment Classifier & Metadata Exporter
uv run marimo edit notebooks/audio_segment_classifier.py

# 3. Speech-to-Text & Ollama Search
uv run marimo edit notebooks/audio_search.py

# 4. Standard Datasets (ESC-50, UrbanSound8K) Explorer
uv run marimo edit notebooks/test_dataset.py
```

---

## 🛠️ CLI & Programmatic Usage

The core classification logic is located in `src/classify_segments.py`.

### Run via Command Line
Run segment-by-segment classification directly from the terminal (works with both audio and video files):
```bash
# Analyze a video file and save results to JSON
uv run python src/classify_segments.py \
  --audio video/2026-03-14_12-28-47.mp4 \
  --output results/video_results.json
```

### Import Programmatically
Integrate the pipeline into your own scripts:
```python
from src.classify_segments import classify_audio_segments

results = classify_audio_segments(
    audio_path="audio/clean.wav",
    model_name="bioamla/ast-esc50",
    overlap_seconds=1.5
)
```

See [docs/classify_segments_guide.md](file:///Users/chocodani/dev/audio_detection/audio_detection/audio-search/docs/classify_segments_guide.md) for more details.
