import torch
import soundfile as sf
from transformers import pipeline
from pathlib import Path

def main():
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    print("Loading bioamla/ast-esc50 model...")
    classifier = pipeline(
        "audio-classification",
        model="bioamla/ast-esc50",
        device=device,
    )

    # Read the 16kHz mono audio file
    # Check audio folder
    audio_path = Path("audio/environmental_sounds_16k.wav")
    if not audio_path.exists():
        audio_path = Path("../audio/environmental_sounds_16k.wav")
    if not audio_path.exists():
        audio_path = Path("audio-search/audio/environmental_sounds_16k.wav")
    if not audio_path.exists():
        audio_path = Path("../audio-search/audio/environmental_sounds_16k.wav")
    if not audio_path.exists():
        audio_path = Path("environmental_sounds_16k.wav")
    if not audio_path.exists():
        audio_path = Path("../environmental_sounds_16k.wav")

    print(f"Reading {audio_path.resolve()}...")
    data, samplerate = sf.read(str(audio_path))
    
    assert samplerate == 16000, f"Expected samplerate to be 16000, but got {samplerate}"
    
    duration = len(data) / samplerate
    print(f"Audio duration: {duration:.2f} seconds")

    # The ESC-50 dataset clips are typically 5 seconds long.
    # We will slice the audio into 5-second segments and classify each segment.
    segment_duration = 5.0
    samples_per_segment = int(segment_duration * samplerate)

    print("\n--- Predictions by 5-second segments ---")
    print(f"{'Time Range':<15} | {'Predicted Label':<25} | {'Confidence':<10}")
    print("-" * 58)

    for start_sample in range(0, len(data), samples_per_segment):
        end_sample = start_sample + samples_per_segment
        segment = data[start_sample:end_sample]
        
        # Skip segments shorter than 1 second
        if len(segment) < samplerate * 1.0:
            continue
            
        start_time = start_sample / samplerate
        end_time = min(end_sample / samplerate, duration)

        # Run prediction
        result = classifier({"array": segment, "sampling_rate": samplerate})
        
        top_prediction = result[0]
        label = top_prediction["label"]
        score = top_prediction["score"]

        time_range = f"{start_time:.1f}s - {end_time:.1f}s"
        print(f"{time_range:<15} | {label:<25} | {score:.2%}")

if __name__ == "__main__":
    main()
