import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import soundfile as sf
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from transformers import pipeline
    from pathlib import Path
    import tempfile
    import io
    import json
    import os

    mo.md(
        """
        # 🔊 Audio Segment Classifier & Metadata Extractor
        This interactive reactive notebook lets you:
        1. Extract audio from video inputs (e.g. `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`, etc.) using `ffmpeg`.
        2. Load any local audio file (including extracted ones).
        3. Configure overlapping segments and select classification models.
        4. Run segment-by-segment classification to capture bounds, segment numbers, and **top 2 predictions** (class and confidence).
        5. Export the complete metadata dictionary to a JSON file.
        """
    )
    return Path, io, mo, np, pipeline, pl, plt, sf, tempfile, torch, json, os


@app.cell
def _(mo):
    refresh_trigger, set_refresh = mo.state(0)
    return refresh_trigger, set_refresh


@app.cell
def _(Path, mo):
    video_dir = Path("video")
    if not video_dir.exists():
        video_dir = Path("audio-search/video")
    if not video_dir.exists():
        video_dir = Path("../video")
        
    video_files = []
    if video_dir.exists():
        # Look for various video formats
        local_videos = (
            list(video_dir.glob("*.mp4")) +
            list(video_dir.glob("*.mkv")) +
            list(video_dir.glob("*.mov")) +
            list(video_dir.glob("*.avi")) +
            list(video_dir.glob("*.webm")) +
            list(video_dir.glob("*.flv"))
        )
        video_files = sorted(list(set([p.name for p in local_videos])))

    video_dropdown = mo.ui.dropdown(
        options={f: f for f in video_files},
        label="Select video in workspace 🎥"
    ) if video_files else None

    custom_video_input = mo.ui.text(
        value="",
        placeholder="Or type path to video file...",
        label="Custom video path 🎥",
        full_width=True
    )

    convert_button = mo.ui.run_button(label="Extract Audio & Convert to WAV 🔄", kind="warn")

    converter_ui = mo.vstack([
        mo.md("### 🎥 Video Audio Extractor"),
        mo.md("If you have a video file, select it or enter its path below to extract its audio track to a 16kHz mono WAV file:"),
        video_dropdown if video_dropdown else mo.md("_No video files detected in workspace. You can type a custom path below._"),
        custom_video_input,
        convert_button
    ])

    mo.accordion({"🎥 Convert Video to WAV": converter_ui})
    return convert_button, custom_video_input, video_dropdown


@app.cell
def _(
    Path,
    convert_button,
    custom_video_input,
    mo,
    set_refresh,
    video_dropdown,
):
    import subprocess

    video_file = ""
    if video_dropdown and video_dropdown.value:
        video_file = video_dropdown.value
    elif custom_video_input.value:
        video_file = custom_video_input.value

    # Stop cell execution if the button is not clicked
    mo.stop(not convert_button.value or not video_file)

    # Resolve video path
    video_path = Path(video_file)
    _paths_to_try = [
        video_path,
        Path("video") / video_file,
        Path("audio-search/video") / video_file,
        Path("../video") / video_file,
        Path("..") / video_file
    ]
    _resolved_path = None
    for _p in _paths_to_try:
        if _p.exists() and _p.is_file():
            _resolved_path = _p
            break

    if _resolved_path is None:
        mo.output.replace(mo.md(f"⚠️ **Error**: Video file `{video_file}` not found."))
        mo.stop(True)

    out_wav_name = f"{_resolved_path.slice_data if hasattr(_resolved_path, 'slice_data') else _resolved_path.stem}_extracted.wav"
    
    # Save the output WAV directly in the audio/ folder
    out_wav_dir = Path("audio")
    if not out_wav_dir.exists():
        out_wav_dir = Path("audio-search/audio")
    if not out_wav_dir.exists():
        out_wav_dir = Path("../audio")
    if not out_wav_dir.exists():
        out_wav_dir = Path(".")
        
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_wav_path = out_wav_dir / out_wav_name

    mo.output.replace(mo.md("⏳ *Extracting audio with ffmpeg...*"))

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(_resolved_path),
            "-ar", "16000",
            "-ac", "1",
            str(out_wav_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Trigger directory refresh
        set_refresh(lambda x: x + 1)

        mo.output.replace(mo.md(f"✅ **Success**: Extracted audio to **`{out_wav_path}`**! It will now appear in the **Audio File** dropdown below."))
    except Exception as e:
        mo.output.replace(mo.md(f"⚠️ **Conversion Error**: {str(e)}"))
    return


@app.cell
def _(Path, mo, refresh_trigger):
    # Reference refresh_trigger to automatically re-scan when a file is converted or directory updates
    _ = refresh_trigger

    # Scan the audio folder
    audio_dirs = [Path("audio"), Path("audio-search/audio"), Path("../audio"), Path(".")]
    audio_files = []
    
    for d in audio_dirs:
        if d.exists():
            local_wavs = list(d.glob("*.wav")) + list(d.glob("*.mp3")) + list(d.glob("*.aiff")) + list(d.glob("*.m4a")) + list(d.glob("*.flac"))
            audio_files.extend([p.name for p in local_wavs])
            
    audio_files = sorted(list(set(audio_files)))

    if not audio_files:
        audio_files = ["environmental_sounds_16k.wav"]

    # Try to find a default file
    default_val = audio_files[0]
    for pref in ["environmental_sounds_16k.wav", "clean.wav"]:
        if pref in audio_files:
            default_val = pref
            break

    audio_picker = mo.ui.dropdown(
        options={f: f for f in audio_files},
        value=default_val,
        label="Audio File 📁"
    )

    model_picker = mo.ui.dropdown(
        options=[
            "bioamla/ast-esc50",
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            "laion/clap-htsat-unfused"
        ],
        value="bioamla/ast-esc50",
        label="Classifier Model 🤖"
    )

    overlap_slider = mo.ui.slider(
        start=0.0,
        stop=4.5,
        step=0.5,
        value=1.0,
        label="Overlap (seconds) 🔄",
        show_value=True
    )
    return audio_picker, model_picker, overlap_slider


@app.cell
def _(audio_picker, mo, model_picker, overlap_slider):
    candidate_labels_input = mo.ui.text_area(
        value="people screaming in panic\nsounds of a physical fight or struggle\ngunshot or gunfire\nnormal conversation\noffice ambient noise\nbouncing ball\nbarking dog\nraining day",
        label="Candidate Labels (one per line, for CLAP only)",
        placeholder="Enter custom labels..."
    )

    # Display settings
    layout = mo.vstack([
        mo.md("### ⚙️ Classification Parameters"),
        mo.hstack([audio_picker, model_picker, overlap_slider], justify="start", gap=2),
        candidate_labels_input if model_picker.value == "laion/clap-htsat-unfused" else mo.md("")
    ])
    layout
    return (candidate_labels_input,)


@app.cell
def _(Path, audio_picker, mo, model_picker, pipeline, sf, torch, np):
    # Determine execution device (MPS for Mac GPU acceleration, CPU otherwise)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    @mo.cache
    def load_classifier(model_name):
        task = "zero-shot-audio-classification" if "clap" in model_name.lower() else "audio-classification"
        return pipeline(task, model=model_name, device=device)

    classifier_model = load_classifier(model_picker.value)

    # Locate and read the file
    file_name = audio_picker.value
    _file_path = Path(file_name)
    _paths_to_try = [
        _file_path,
        Path("audio") / file_name,
        Path("audio-search/audio") / file_name,
        Path("../audio") / file_name,
        Path(".") / file_name
    ]
    resolved_path = None
    for _p in _paths_to_try:
        if _p.exists() and _p.is_file():
            resolved_path = _p
            break

    if resolved_path is None:
        # Provide fallback/dummy audio data if file doesn't exist
        samplerate = 16000
        duration = 5.0
        data = np.zeros(int(samplerate * duration))
        resolved_path = Path(file_name)
    else:
        data, samplerate = sf.read(str(resolved_path))
        # Convert stereo to mono if needed
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        duration = len(data) / samplerate

    return classifier_model, data, duration, resolved_path, samplerate


@app.cell
def _(duration, resolved_path, mo, samplerate):
    mo.md(f"""
    ### 📝 Audio File Info
    * **Path**: `{resolved_path.resolve()}`
    * **Sample Rate**: `{samplerate} Hz`
    * **Channels**: `Mono`
    * **Duration**: `{duration:.2f} seconds` (`{int(duration // 60)}m {int(duration % 60)}s`)
    """)
    return


@app.cell
def _(duration, mo, overlap_slider):
    _segment_duration = 5.0
    _overlap_duration = overlap_slider.value
    _step_duration = _segment_duration - _overlap_duration

    # Generate list of overlapping segments
    segment_ranges = []
    t_start = 0.0
    while t_start + _segment_duration <= duration:
        segment_ranges.append((t_start, t_start + _segment_duration))
        t_start += _step_duration
        
    # If the file is shorter than 5 seconds, or if there's remaining tail
    if not segment_ranges:
        segment_ranges.append((0.0, duration))
    elif t_start < duration and duration - t_start >= 1.0: # Segment must be at least 1s
        segment_ranges.append((t_start, duration))

    segment_options = {f"Segment {i}: {r[0]:.1f}s - {r[1]:.1f}s": (i, r) for i, r in enumerate(segment_ranges)}

    segment_dropdown = mo.ui.dropdown(
        options=segment_options,
        value=list(segment_options.keys())[0] if segment_options else None,
        label="Select Overlapping Segment 🎚️"
    )

    mo.vstack([
        mo.md("### 🔍 Select a single segment to inspect & hear:"),
        segment_dropdown
    ])
    return segment_dropdown, segment_ranges


@app.cell
def _(data, mo, samplerate, segment_dropdown, sf, tempfile):
    # Slicing the selected segment
    if segment_dropdown.value is not None:
        _, (start_time, end_time) = segment_dropdown.value
    else:
        start_time, end_time = 0.0, min(5.0, len(data)/samplerate)
        
    _start_sample = int(start_time * samplerate)
    _end_sample = int(end_time * samplerate)
    slice_data = data[_start_sample:_end_sample]

    # Play selected audio segment
    audio_player = mo.md("")
    if len(slice_data) > 0:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_wav.name, slice_data, samplerate)
        audio_player = mo.audio(src=temp_wav.name)

    mo.hstack([mo.md("**▶ Listen to selected segment:**"), audio_player], align="center", gap=2)
    return slice_data, start_time, end_time


@app.cell
def _(
    candidate_labels_input,
    classifier_model,
    io,
    mo,
    model_picker,
    np,
    plt,
    samplerate,
    slice_data,
    start_time,
    end_time
):
    # Create the figures for Waveform and Spectrogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))

    # Waveform Plot
    t = np.arange(len(slice_data)) / samplerate
    axes[0].plot(t, slice_data, linewidth=0.5, color="#1abc9c")
    axes[0].set_title(f"Waveform ({start_time:.1f}s - {end_time:.1f}s)", fontsize=10)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Spectrogram Plot
    if len(slice_data) > 256:
        axes[1].specgram(slice_data, Fs=samplerate, NFFT=min(1024, len(slice_data)), noverlap=min(512, len(slice_data)//2), cmap="magma")
    axes[1].set_title("Spectrogram", fontsize=10)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")

    fig.tight_layout()

    # Convert slice to WAV bytes in memory
    _buf = io.BytesIO()
    sf.write(_buf, slice_data, samplerate, format="WAV")
    _wav_bytes = _buf.getvalue()

    # Classification
    if "clap" in model_picker.value.lower():
        _labels = [line.strip() for line in candidate_labels_input.value.split("\n") if line.strip()]
        if not _labels:
            _labels = ["sound"]
        result = classifier_model(_wav_bytes, candidate_labels=_labels)
    else:
        result = classifier_model(_wav_bytes)

    prediction_rows = "\n".join(
        f"| {i+1} | `{pred['label']}` | {pred['score']:.2%} |"
        for i, pred in enumerate(result[:5])
    )

    mo.hstack([
        mo.as_html(fig),
        mo.md(f"""
        ### 🎯 Slice Predictions
 
        | # | Class | Confidence |
        |---|---|---|
        {prediction_rows}
        """)
    ], gap=2)
    return


@app.cell
def _(mo):
    run_full_btn = mo.ui.run_button(label="🚀 Run Full Timeline Classification", kind="success")
    run_full_btn
    return (run_full_btn,)


@app.cell
def _(
    candidate_labels_input,
    classifier_model,
    data,
    duration,
    io,
    mo,
    model_picker,
    resolved_path,
    segment_ranges,
    samplerate,
    run_full_btn,
    sf,
    pl,
):
    # Require button click
    mo.stop(not run_full_btn.value, mo.md("*Click the button above to run classification for the entire audio file.*"))

    timeline_preds = []
    
    status_bar = mo.status.progress_bar(title="Classifying audio segments...", total=len(segment_ranges))

    is_clap = "clap" in model_picker.value.lower()
    _labels = []
    if is_clap:
        _labels = [line.strip() for line in candidate_labels_input.value.split("\n") if line.strip()]
        if not _labels:
            _labels = ["sound"]

    # Store results in a dictionary format requested by user
    # Calculate overlap duration represented in the timeline bounds
    actual_overlap = 0.0
    if len(segment_ranges) > 1:
        # segment_duration - step_duration
        seg_dur = segment_ranges[0][1] - segment_ranges[0][0]
        step_dur = segment_ranges[1][0] - segment_ranges[0][0]
        actual_overlap = float(round(seg_dur - step_dur, 3))

    metadata_dict = {
        "input_file": str(resolved_path.name),
        "input_duration_seconds": round(duration, 3),
        "model_name": model_picker.value,
        "overlap_seconds": actual_overlap,
        "segments": []
    }

    with status_bar as bar:
        for idx, (_start_time, _end_time) in enumerate(segment_ranges):
            bar.update(subtitle=f"Segment {idx}: {_start_time:.1f}s - {_end_time:.1f}s")
            
            _start_sample = int(_start_time * samplerate)
            _end_sample = int(_end_time * samplerate)
            segment_data = data[_start_sample:_end_sample]
            
            if len(segment_data) < 100:
                continue

            # Convert segment to WAV bytes
            _buf = io.BytesIO()
            sf.write(_buf, segment_data, samplerate, format="WAV")
            _wav_bytes = _buf.getvalue()

            if is_clap:
                res = classifier_model(_wav_bytes, candidate_labels=_labels)
            else:
                res = classifier_model(_wav_bytes)

            # Extract top 2 predictions
            predictions_top2 = []
            for pred in res[:2]:
                predictions_top2.append({
                    "class": pred["label"],
                    "confidence": float(pred["score"])
                })
                
            # If model returned fewer than 2 predictions, pad it
            while len(predictions_top2) < 2:
                predictions_top2.append({
                    "class": "None",
                    "confidence": 0.0
                })

            segment_info = {
                "segment_index": idx,
                "start_time": round(_start_time, 2),
                "end_time": round(_end_time, 2),
                "predictions": predictions_top2
            }
            metadata_dict["segments"].append(segment_info)
            
            # For table display
            timeline_preds.append({
                "Seg #": idx,
                "Time Range": f"{_start_time:.1f}s - {_end_time:.1f}s",
                "1st Prediction": f"{predictions_top2[0]['class']} ({predictions_top2[0]['confidence']:.2%})",
                "2nd Prediction": f"{predictions_top2[1]['class']} ({predictions_top2[1]['confidence']:.2%})"
            })

    # Display results
    df = pl.DataFrame(timeline_preds)
    
    mo.output.replace(
        mo.vstack([
            mo.md("### 📊 Classification Summary Table"),
            mo.ui.table(df, label="Timeline Classification Table", page_size=15)
        ])
    )
    
    return df, metadata_dict


@app.cell
def _(metadata_dict, mo, resolved_path, Path):
    mo.stop(metadata_dict is None, mo.md(""))

    import json as json_lib
    
    # Format the dictionary as pretty JSON string
    json_str = json_lib.dumps(metadata_dict, indent=2)

    # File save components
    save_dir = Path("results")
    if not save_dir.exists():
        save_dir = Path("audio-search/results")
    if not save_dir.exists():
        save_dir = Path("../results")
    if not save_dir.exists():
        save_dir = Path(".")
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    default_save_path = save_dir / f"{resolved_path.stem}_classification_results.json"
    
    save_path_input = mo.ui.text(
        value=str(default_save_path),
        label="Save Path 📁",
        full_width=True
    )
    
    save_btn = mo.ui.run_button(label="💾 Save Dictionary to JSON File", kind="success")
    
    # Display elements
    mo.vstack([
        mo.md("### 💾 Export Classification Metadata"),
        save_path_input,
        save_btn
    ])
    
    return default_save_path, json_str, save_btn, save_path_input


@app.cell
def _(json_str, mo, save_btn, save_path_input, Path):
    mo.stop(not save_btn.value, mo.md(""))
    
    _file_path = Path(save_path_input.value)
    try:
        # Create parent dirs if necessary
        _file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_file_path, "w") as f:
            f.write(json_str)
        mo.output.replace(mo.md(f"✅ **Success**: Dictionary saved to [**`{_file_path.name}`**](file://{_file_path.resolve()})"))
    except Exception as e:
        mo.output.replace(mo.md(f"⚠️ **Error saving file**: {str(e)}"))
    return


@app.cell
def _(metadata_dict, mo):
    mo.stop(metadata_dict is None, mo.md(""))
    
    mo.vstack([
        mo.md("### 🌳 Result Dictionary (Interactive Tree)"),
        mo.md("Expand the tree below to inspect the extracted segment bounds and predictions directly in the notebook:"),
        mo.tree(metadata_dict)
    ])
    return


if __name__ == "__main__":
    app.run()
