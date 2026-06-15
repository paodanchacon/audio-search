"""
Audio Search MCP server.

Exposes a DVR/CCTV audio transcript to an LLM client (e.g. Claude Desktop)
as a small set of tools. The LLM does the reasoning; this server just
provides the data (the transcript and matching lines with timestamps).

Everything runs locally — no audio or transcript leaves the machine.
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from src.classify_segments import classify_audio_segments, extract_audio_from_video


# The MCP server. The name is how it shows up in the client.
mcp = FastMCP("audio-search")

# Where this file lives, so we can find the transcript next to it.
HERE = Path(__file__).parent

# Map a friendly recording name -> its transcript file.
# Add more entries here as you transcribe more recordings.
RECORDINGS = {
    "teller": HERE / "transcript.json",
}


def _load(recording: str):
    """Load a recording's transcript segments, or None if it doesn't exist."""
    path = RECORDINGS.get(recording)
    if not path or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _format(segments) -> str:
    """Turn transcript segments into timestamped lines of text."""
    return "\n".join(
        f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments
    )


@mcp.tool()
def list_recordings() -> list[str]:
    """List the names of audio recordings available to search."""
    return list(RECORDINGS.keys())


@mcp.tool()
def get_transcript(recording: str) -> str:
    """Return the full timestamped transcript for one recording.

    Args:
        recording: the recording name (see list_recordings).
    """
    segments = _load(recording)
    if segments is None:
        return f"No recording named '{recording}'. Available: {list(RECORDINGS)}"
    return _format(segments)


@mcp.tool()
def search_transcript(recording: str, query: str) -> str:
    """Find lines in a recording that contain the given words.

    Returns the matching lines with their timestamps, so the answer can
    be traced back to the exact moment in the audio.

    Args:
        recording: the recording name (see list_recordings).
        query: words to look for, e.g. "wire transfer".
    """
    segments = _load(recording)
    if segments is None:
        return f"No recording named '{recording}'. Available: {list(RECORDINGS)}"
    q = query.lower()
    hits = [s for s in segments if q in s["text"].lower()]
    if not hits:
        return f"No lines matching '{query}' in '{recording}'."
    return _format(hits)


@mcp.tool()
def extract_audio(video_path: str, output_audio_path: str = None) -> str:
    """Extracts the audio track from a video file and converts it to a 16kHz mono WAV file.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Optional destination path for the WAV file. If not provided, a temporary file is created.
    """
    try:
        path = extract_audio_from_video(video_path, output_audio_path)
        return f"Successfully extracted audio to: {path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def classify_audio(
    audio_path: str,
    model_name: str = "bioamla/ast-esc50",
    overlap_seconds: float = 1.0,
    candidate_labels: list[str] = None,
    output_json_path: str = None
) -> str:
    """Slices an audio or video file into overlapping 5-second segments and classifies each segment using a pre-trained model.
    If a video file is provided, its audio track is automatically extracted first.

    Args:
        audio_path: Path to the input audio or video file.
        model_name: Hugging Face model identifier (e.g. "bioamla/ast-esc50").
        overlap_seconds: Number of seconds of overlap between adjacent segments.
        candidate_labels: Optional list of candidate labels for zero-shot classification (CLAP).
        output_json_path: Optional path to write the output JSON results.
    """
    try:
        results = classify_audio_segments(
            audio_path=audio_path,
            model_name=model_name,
            overlap_seconds=overlap_seconds,
            candidate_labels=candidate_labels,
            output_json_path=output_json_path
        )
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    # Runs over stdio — the transport Claude Desktop uses.
    mcp.run()

