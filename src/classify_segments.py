import io
import json
import os
import subprocess
import tempfile
import torch
import soundfile as sf
from pathlib import Path
from transformers import pipeline

def extract_audio_from_video(video_path, output_audio_path=None):
    """
    Extracts the audio track from a video file and converts it to a 16kHz mono WAV file using ffmpeg.

    Args:
        video_path (str or Path): Path to the input video file.
        output_audio_path (str or Path, optional): Destination path for the WAV file. 
            If None, a WAV file is created in a temporary directory.

    Returns:
        Path: Path to the extracted WAV file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_audio_path is None:
        # Create a temporary file that won't be deleted automatically on close
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        output_audio_path = Path(temp_file.name)
    else:
        output_audio_path = Path(output_audio_path)

    # Ensure parent directory exists
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_audio_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg executable not found. Please ensure ffmpeg is installed and added to your PATH.")

    return output_audio_path

def classify_audio_segments(
    audio_path,
    model_name="bioamla/ast-esc50",
    overlap_seconds=1.0,
    candidate_labels=None,
    output_json_path=None
):
    """
    Slices an audio or video file into overlapping 5-second segments and classifies each segment using a pre-trained model.
    If a video file is provided, its audio track is automatically extracted first.

    Args:
        audio_path (str or Path): Path to the input audio or video file.
        model_name (str): Hugging Face model identifier.
        overlap_seconds (float): Number of seconds of overlap between adjacent segments.
        candidate_labels (list of str, optional): Candidate labels for zero-shot classification (CLAP).
        output_json_path (str or Path, optional): If provided, writes the results dictionary to this JSON file.

    Returns:
        dict: A dictionary containing details of the input file, model, overlap, and segment predictions.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Input file not found: {audio_path}")

    # Detect if the input is a video format and needs audio extraction
    video_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".ogg"}
    temp_wav_path = None
    target_audio_path = audio_path

    if audio_path.suffix.lower() in video_extensions:
        print(f"⏳ Video file detected. Extracting audio from {audio_path.name}...")
        temp_wav_path = extract_audio_from_video(audio_path)
        target_audio_path = temp_wav_path

    try:
        # 1. Determine device & Load classification pipeline
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        task = "zero-shot-audio-classification" if "clap" in model_name.lower() else "audio-classification"
        classifier = pipeline(task, model=model_name, device=device)

        # 2. Load audio file
        data, samplerate = sf.read(str(target_audio_path))
        if len(data.shape) > 1:
            data = data.mean(axis=1)  # Convert stereo to mono
        duration = len(data) / samplerate

        # 3. Calculate segment boundaries (defaulting to 5.0 seconds segments)
        segment_duration = 5.0
        step_duration = segment_duration - overlap_seconds

        segment_ranges = []
        t_start = 0.0
        while t_start + segment_duration <= duration:
            segment_ranges.append((t_start, t_start + segment_duration))
            t_start += step_duration

        if not segment_ranges:
            segment_ranges.append((0.0, duration))
        elif t_start < duration and duration - t_start >= 1.0:  # Segment must be at least 1 second
            segment_ranges.append((t_start, duration))

        # 4. Process and classify each segment
        results = {
            "input_file": audio_path.name,
            "input_duration_seconds": round(duration, 3),
            "model_name": model_name,
            "overlap_seconds": overlap_seconds,
            "segments": []
        }

        is_clap = "clap" in model_name.lower()
        
        for idx, (start_time, end_time) in enumerate(segment_ranges):
            start_sample = int(start_time * samplerate)
            end_sample = int(end_time * samplerate)
            segment_data = data[start_sample:end_sample]

            # Skip extremely short segments
            if len(segment_data) < 100:
                continue

            # Convert to WAV bytes in-memory for automatic resampling/compatibility
            buf = io.BytesIO()
            sf.write(buf, segment_data, samplerate, format="WAV")
            wav_bytes = buf.getvalue()

            # Run classification
            if is_clap:
                labels = candidate_labels if candidate_labels else ["sound"]
                res = classifier(wav_bytes, candidate_labels=labels)
            else:
                res = classifier(wav_bytes)

            # Extract top 2 predictions
            predictions_top2 = []
            for pred in res[:2]:
                predictions_top2.append({
                    "class": pred["label"],
                    "confidence": float(pred["score"])
                })

            # Pad with None if fewer than 2 predictions
            while len(predictions_top2) < 2:
                predictions_top2.append({
                    "class": "None",
                    "confidence": 0.0
                })

            results["segments"].append({
                "segment_index": idx,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "predictions": predictions_top2
            })

        # 5. Save output if requested
        if output_json_path:
            output_json_path = Path(output_json_path)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=2)

        return results

    finally:
        # Clean up temporary WAV file if it was created
        if temp_wav_path and temp_wav_path.exists():
            try:
                os.remove(temp_wav_path)
            except Exception as e:
                print(f"⚠️ Failed to remove temporary file {temp_wav_path}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify audio segments and export results to JSON.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio or video file (e.g. .wav, .mp3, .mp4, .mkv)")
    parser.add_argument("--model", type=str, default="bioamla/ast-esc50", help="HuggingFace audio model")
    parser.add_argument("--overlap", type=float, default=1.0, help="Overlap in seconds between segments")
    parser.add_argument("--output", type=str, help="Path to write output JSON results")
    parser.add_argument("--labels", type=str, nargs="+", help="Candidate labels for zero-shot CLAP model (space-separated list)")
    
    args = parser.parse_args()
    
    try:
        results_data = classify_audio_segments(
            audio_path=args.audio,
            model_name=args.model,
            overlap_seconds=args.overlap,
            candidate_labels=args.labels,
            output_json_path=args.output
        )
        print(f"✅ Successfully processed {args.audio}!")
        if args.output:
            print(f"📁 Saved results to: {args.output}")
        else:
            print(json.dumps(results_data, indent=2))
    except Exception as e:
        print(f"⚠️ Error: {e}")
