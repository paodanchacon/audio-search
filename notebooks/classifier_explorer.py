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

    mo.md(
        """
        # 🔊 YouTube Audio Classifier Explorer
        This interactive reactive notebook lets you load any local audio file, visualize its waveform & spectrogram, listen to specific segments, and classify the sounds using the pre-trained Audio Spectrogram Transformer (`bioamla/ast-esc50`).
        """
    )
    return Path, io, mo, np, pipeline, pl, plt, sf, tempfile, torch


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
        
    mp4_files = []
    if video_dir.exists():
        local_mp4s = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mkv")) + list(video_dir.glob("*.mov"))
        mp4_files = sorted(list(set([p.name for p in local_mp4s])))

    video_dropdown = mo.ui.dropdown(
        options={f: f for f in mp4_files},
        label="Select video in workspace 🎥"
    ) if mp4_files else None

    custom_video_input = mo.ui.text(
        value="",
        placeholder="Or type path to video file...",
        label="Custom video path 🎥",
        full_width=True
    )

    convert_button = mo.ui.run_button(label="Extract Audio & Convert to WAV 🔄", kind="warn")

    converter_ui = mo.vstack([
        mo.md("### 🎥 Video Audio Extractor"),
        mo.md("If you have an MP4 (or other video) file, select it or enter its path below to extract its audio track to a 16kHz mono WAV file:"),
        video_dropdown if video_dropdown else mo.md("_No video files (.mp4, .mov, .mkv) detected in directory. You can type a custom path below._"),
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
    if not video_path.exists():
        video_path = Path("video") / video_file
    if not video_path.exists():
        video_path = Path("audio-search/video") / video_file
    if not video_path.exists():
        video_path = Path("../video") / video_file
    if not video_path.exists():
        video_path = Path("..") / video_file

    if not video_path.exists():
        mo.output.replace(mo.md(f"⚠️ **Error**: Video file `{video_file}` not found."))
        mo.stop(True)

    out_wav_name = f"{video_path.stem}_extracted.wav"
    
    # Save the output WAV directly in the audio/ folder
    out_wav_dir = Path("audio")
    if not out_wav_dir.exists():
        out_wav_dir = Path("audio-search/audio")
    if not out_wav_dir.exists():
        out_wav_dir = Path("../audio")
    if not out_wav_dir.exists():
        out_wav_dir = Path(".")
        
    out_wav_path = out_wav_dir / out_wav_name

    mo.output.replace(mo.md("⏳ *Extracting audio with ffmpeg...*"))

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
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
    # Reference refresh_trigger to automatically re-scan when a file is converted
    _ = refresh_trigger

    # Scan the audio folder
    audio_dir = Path("audio")
    if not audio_dir.exists():
        audio_dir = Path("audio-search/audio")
    if not audio_dir.exists():
        audio_dir = Path("../audio")

    audio_files = []
    if audio_dir.exists():
        local_wavs = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.aiff"))
        audio_files = sorted(list(set([p.name for p in local_wavs])))

    if not audio_files:
        audio_files = ["A_continuous,_natura_#1-1781463844197.wav"]

    default_val = "A_continuous,_natura_#1-1781463844197.wav" if "A_continuous,_natura_#1-1781463844197.wav" in audio_files else audio_files[0]

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
        stop=4.0,
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

    # Conditionally display the custom candidate labels area
    layout = mo.vstack([
        mo.hstack([audio_picker, model_picker, overlap_slider], justify="start", gap=2),
        candidate_labels_input if model_picker.value == "laion/clap-htsat-unfused" else mo.md("")
    ])
    layout
    return (candidate_labels_input,)


@app.cell
def _(Path, audio_picker, mo, model_picker, pipeline, sf, torch):
    # Determine execution device (MPS for Mac GPU acceleration, CPU otherwise)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    @mo.cache
    def load_classifier(model_name):
        task = "zero-shot-audio-classification" if "clap" in model_name.lower() else "audio-classification"
        return pipeline(task, model=model_name, device=device)

    classifier_model = load_classifier(model_picker.value)

    # Locate and read the file from the audio folder
    file_path = Path("audio") / audio_picker.value
    if not file_path.exists():
        file_path = Path("audio-search/audio") / audio_picker.value
    if not file_path.exists():
        file_path = Path("../audio") / audio_picker.value
    if not file_path.exists():
        file_path = Path(audio_picker.value)

    data, samplerate = sf.read(str(file_path))

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    duration = len(data) / samplerate
    return classifier_model, data, duration, file_path, samplerate


@app.cell
def _(duration, file_path, mo, samplerate):
    mo.md(f"""
    ### 📝 File Info
    * **Path**: `{file_path.resolve()}`
    * **Sample Rate**: `{samplerate} Hz`
    * **Channels**: `Mono` (mixed down if stereo)
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

    segment_options = {f"{r[0]:.1f}s - {r[1]:.1f}s": r for r in segment_ranges}

    segment_dropdown = mo.ui.dropdown(
        options=segment_options,
        value=list(segment_options.keys())[0] if segment_options else None,
        label="Select Overlapping Segment 🎚️"
    )

    mo.vstack([
        mo.md("### 🔍 Select a computed overlapping segment to inspect:"),
        segment_dropdown
    ])
    return (segment_dropdown,)


@app.cell
def _(data, mo, samplerate, segment_dropdown, sf, tempfile):
    # Slicing the selected segment
    _bounds = segment_dropdown.value if segment_dropdown.value is not None else (0.0, 5.0)
    _start_time, _end_time = _bounds
    start_sample = int(_start_time * samplerate)
    end_sample = int(_end_time * samplerate)
    slice_data = data[start_sample:end_sample]

    # Write a temporary wav file so that marimo audio player can play it
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, slice_data, samplerate)

    audio_player = mo.audio(src=temp_wav.name)

    mo.hstack([mo.md("**▶ Listen to selected segment:**"), audio_player], align="center", gap=2)
    return (slice_data,)


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
    segment_dropdown,
    sf,
    slice_data,
):
    # Extract selected segment bounds
    _bounds = segment_dropdown.value if segment_dropdown.value is not None else (0.0, 5.0)
    _start_time, _end_time = _bounds

    # Create the figures for Waveform and Spectrogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))

    # Waveform Plot
    t = np.arange(len(slice_data)) / samplerate
    axes[0].plot(t, slice_data, linewidth=0.5, color="#1abc9c")
    axes[0].set_title(f"Waveform ({_start_time:.1f}s - {_end_time:.1f}s)", fontsize=10)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, 5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Spectrogram Plot
    axes[1].specgram(slice_data, Fs=samplerate, NFFT=1024, noverlap=512, cmap="magma")
    axes[1].set_title("Spectrogram", fontsize=10)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")

    fig.tight_layout()

    # Convert slice to WAV bytes in memory to automatically handle resampling and format support
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
    run_full_btn = mo.ui.run_button(label="🚀 Run Full Timeline Classification", kind="primary")
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
    overlap_slider,
    pl,
    run_full_btn,
    samplerate,
    sf,
):
    mo.stop(not run_full_btn.value, mo.md("*Click the button above to run classification for the entire audio file.*"))

    timeline_preds = []
    _segment_duration = 5.0
    _overlap_duration = overlap_slider.value
    _step_duration = _segment_duration - _overlap_duration

    samples_per_segment = int(_segment_duration * samplerate)
    samples_per_step = int(_step_duration * samplerate)

    # Calculate total steps for progress bar
    total_segments = max(1, (len(data) - samples_per_segment) // samples_per_step + 1)

    status_bar = mo.status.progress_bar(title="Classifying audio timeline...", max=total_segments)

    # Parse labels if CLAP is selected
    is_clap = "clap" in model_picker.value.lower()
    _labels = []
    if is_clap:
        _labels = [line.strip() for line in candidate_labels_input.value.split("\n") if line.strip()]
        if not _labels:
            _labels = ["sound"]

    with status_bar:
        for i, start_s in enumerate(range(0, len(data) - samples_per_segment + 1, samples_per_step)):
            status_bar.update(value=i)
            end_s = start_s + samples_per_segment
            segment = data[start_s:end_s]

            _start_time = start_s / samplerate
            _end_time = min(end_s / samplerate, duration)

            # Convert segment to WAV bytes
            _buf = io.BytesIO()
            sf.write(_buf, segment, samplerate, format="WAV")
            _wav_bytes = _buf.getvalue()

            if is_clap:
                res = classifier_model(_wav_bytes, candidate_labels=_labels)
            else:
                res = classifier_model(_wav_bytes)

            top_pred = res[0]

            timeline_preds.append({
                "Time Range": f"{_start_time:.1f}s - {_end_time:.1f}s",
                "Predicted Class": top_pred["label"],
                "Confidence": f"{top_pred['score']:.2%}"
            })

    df = pl.DataFrame(timeline_preds)

    mo.vstack([
        mo.md("### 📊 Complete Timeline Predictions"),
        mo.ui.table(df, label="Timeline Classification Table", page_size=15)
    ])
    return


if __name__ == "__main__":
    app.run()
