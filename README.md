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

## 🛠️ CLI & Programmatic Usage (`classify_segments.py`)

The core classification logic is located in [src/classify_segments.py](file:///Users/chocodani/dev/audio_detection/audio_detection/audio-search/src/classify_segments.py).

### 📋 Prerequisites

1. **Python & `uv`**: Ensure you have Python 3.12+ and the `uv` tool suite installed.
2. **`ffmpeg`**: Required if you plan to analyze video files (for automatic audio extraction).
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

---

### 🚀 Running the CLI

Run segment-by-segment classification directly from the terminal inside the `audio-search` directory. It works natively with both audio and video files.

#### 1. Basic Classification (Defaults to AST Model)
Prints classification results directly to the console:
```bash
uv run python src/classify_segments.py --audio audio/clean.wav
```

#### 2. Process Video & Export Results to JSON
Extracts audio from video, classifies it, and writes the structured results:
```bash
uv run python src/classify_segments.py \
  --audio video/2026-03-14_12-28-47.mp4 \
  --output results/video_results.json
```

#### 3. Custom Model & Overlap Settings
Modify the model or overlap duration (in seconds):
```bash
uv run python src/classify_segments.py \
  --audio audio/clean.wav \
  --model MIT/ast-finetuned-audioset-10-10-0.4593 \
  --overlap 2.5
```

#### 4. Zero-Shot CLAP Classification
Use CLAP with custom search labels:
```bash
uv run python src/classify_segments.py \
  --audio audio/clean.wav \
  --model laion/clap-htsat-unfused \
  --labels "dog barking" "office noise" "human speech" "applause"
```

#### 5. CLI Help
View all available CLI arguments and options:
```bash
uv run python src/classify_segments.py --help
```

---

### 📖 Function Details & API Reference

#### 1. `extract_audio_from_video(video_path, output_audio_path=None)`
Extracts the audio track from a video file and converts it to a 16kHz mono WAV file using `ffmpeg`.
* **Parameters**:
  - `video_path` (`str` or `Path`): Path to the input video file (e.g. `.mp4`, `.mkv`, `.mov`).
  - `output_audio_path` (`str` or `Path`, optional): Target path for the output `.wav` file. If `None`, it creates a temporary file in your system's temp folder.
* **Returns**:
  - `Path`: The absolute path to the generated WAV file.
* **Under the Hood**:
  - Spawns an `ffmpeg` subprocess using the arguments `-y -i <input> -ar 16000 -ac 1 <output>`. This forces the audio to a single channel (`-ac 1`) and resamples it to 16,000 Hz (`-ar 16000`), which is the native rate required by AST models.

#### 2. `classify_audio_segments(audio_path, model_name="bioamla/ast-esc50", overlap_seconds=1.0, candidate_labels=None, output_json_path=None)`
Slices an audio or video file into overlapping 5-second segments and classifies each segment using a pre-trained Hugging Face model.
* **Parameters**:
  - `audio_path` (`str` or `Path`): Path to the audio or video file. If a video is supplied, `extract_audio_from_video` is automatically run on it first.
  - `model_name` (`str`): Hugging Face model identifier (defaults to `"bioamla/ast-esc50"`).
  - `overlap_seconds` (`float`): Overlap duration in seconds between adjacent segments (e.g. `1.0` seconds of overlap means windows start every `4.0` seconds).
  - `candidate_labels` (`list of str`, optional): Text labels to search for when using zero-shot models (like CLAP).
  - `output_json_path` (`str` or `Path`, optional): Target JSON file path. If specified, the results dictionary will be exported as pretty-printed JSON.
* **Returns**:
  - `dict`: A dictionary containing:
    - `"input_file"`: Name of the processed file.
    - `"input_duration_seconds"`: Total duration of the audio.
    - `"model_name"`: Model used for inference.
    - `"overlap_seconds"`: Overlap used between windows.
    - `"segments"`: A list of dicts, each with `"segment_index"`, `"start_time"`, `"end_time"`, and `"predictions"` (a list of the top 2 class predictions with confidence scores).
* **Under the Hood**:
  1. Detects video formats by checking extension suffixes. If a match is found, it extracts audio into a temporary WAV file and schedules it for cleanup in a `finally` block.
  2. Dynamically selects the execution device (`cuda` if CUDA GPU is available, `mps` for Apple Silicon GPU acceleration, or fallback to `cpu`).
  3. Slices the audio waveform in-memory. Each slice is written into a memory buffer (`io.BytesIO`) as a WAV format block. This memory-first approach ensures the Hugging Face pipeline can decode and resample the audio on-the-fly without creating physical temporary segment files on disk.

---

### 🐍 Programmatic Python Import

You can import both `classify_audio_segments` and `extract_audio_from_video` directly from `src/classify_segments.py` into your own pipelines.

#### 1. Extracting Audio from Video (`extract_audio_from_video`)
This function extracts the audio channel from any supported video format, downmixes it to mono, and resamples it to 16kHz for classification models.

```python
from pathlib import Path
from src.classify_segments import extract_audio_from_video

# Example A: Extract to a specific file path
output_file = extract_audio_from_video(
    video_path="video/2026-03-14_12-28-47.mp4",
    output_audio_path="audio/custom_extracted.wav"
)
print(f"Audio saved to: {output_file}")

# Example B: Extract to a temporary file (ideal for ephemeral preprocessing)
temp_file_path = extract_audio_from_video(
    video_path="video/2026-03-14_12-28-47.mp4"
)
print(f"Temporary audio saved to: {temp_file_path}")
# Note: You should delete the temporary file after you're done processing it!
```

#### 2. Slicing & Classifying Audio (`classify_audio_segments`)
This function divides the audio (or extracted video audio) into 5.0-second segments, offsets them by a sliding window determined by `overlap_seconds`, and classifies each segment.

```python
from src.classify_segments import classify_audio_segments

# Run segment-by-segment classification on a video/audio file
results = classify_audio_segments(
    audio_path="audio/clean.wav",
    model_name="bioamla/ast-esc50",
    overlap_seconds=1.5,
    output_json_path="results/output_results.json"
)

# Access results in code
print(f"File: {results['input_file']}")
print(f"Duration: {results['input_duration_seconds']}s")
for seg in results["segments"]:
    print(f"[{seg['start_time']}s - {seg['end_time']}s] Top Predict: {seg['predictions'][0]['class']}")
```

For more details, see the [classify_segments_guide.md](file:///Users/chocodani/dev/audio_detection/audio_detection/audio-search/docs/classify_segments_guide.md).

---

### 🤖 Tested Classification Models

We have successfully tested and integrated the following Hugging Face models in this workspace:

| Model ID | Architecture / Dataset | Type | Description |
|---|---|---|---|
| **`bioamla/ast-esc50`** *(Default)* | Audio Spectrogram Transformer (AST) / ESC-50 | Standard | Finetuned on 50 classes of environmental sound recordings (animals, natural sounds, human non-speech, domestic noises). |
| **`MIT/ast-finetuned-audioset-10-10-0.4593`** | Audio Spectrogram Transformer (AST) / AudioSet | Standard | Finetuned on AudioSet's 527 classes. Best for wide-range acoustic event detection. |
| **`laion/clap-htsat-unfused`** | Contrastive Language-Audio Pretraining (CLAP) | Zero-Shot | Maps text queries to audio embeddings. Allows custom, arbitrary classification queries by specifying custom labels (via CLI `--labels` or Python `candidate_labels`). |

---

## 🔌 MCP Server Setup (`audio_mcp_server.py`)

The workspace includes a Model Context Protocol (MCP) server in [audio_mcp_server.py](file:///Users/chocodani/dev/audio_detection/audio_detection/audio-search/audio_mcp_server.py) that exposes audio transcription, segment classification, and metadata search tools directly to LLM clients (like **Claude Desktop**).

### 🛠️ Developer Mode (Testing Tools)

To run the MCP server in interactive developer mode and inspect available tools:
```bash
uv run mcp dev audio_mcp_server.py
```

---

### 💻 Integrating with Claude Desktop

To configure Claude Desktop to run your local audio search tools:

1. Locate and open your Claude Desktop configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
2. Add the `audio-search` server configuration inside the `mcpServers` object (ensure the paths are absolute and match your setup):

```json
{
  "mcpServers": {
    "audio-search": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/chocodani/dev/audio_detection/audio_detection/audio-search",
        "run",
        "audio_mcp_server.py"
      ]
    }
  }
}
```

3. **Restart/Relaunch** Claude Desktop completely.
4. If configured correctly, a **hammer icon** 🛠️ will appear in Claude's text input box, indicating that the following local tools are available:
   - `list_recordings`: List the names of audio recordings available to search.
   - `get_transcript`: Get the full timestamped transcript of a recording.
   - `search_transcript`: Find lines containing specific words inside a transcript.
   - `extract_audio`: Extract the audio track from a local video file.
   - `classify_audio`: Run segment-by-segment classification on a local audio/video file.
