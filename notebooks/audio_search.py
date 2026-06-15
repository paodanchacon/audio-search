import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import subprocess
    from pathlib import Path
    
    # Locate audio directory dynamically
    audio_dir = Path("audio")
    if not audio_dir.exists():
        audio_dir = Path("../audio")
    if not audio_dir.exists():
        audio_dir = Path("audio-search/audio")
        
    input_file = audio_dir / "teller.aiff"
    output_file = audio_dir / "clean.wav"
    
    subprocess.run([
        "ffmpeg",
        "-y",                       # overwrite the output if it already exists
        "-i", str(input_file),      # input file
        "-ar", "16000",             # set audio rate to 16,000 Hz
        "-ac", "1",                 # set to 1 channel (mono)
        str(output_file)            # output file
    ])
    print(f"Done — created {output_file}")
    return


@app.cell
def _():
    import whisper
    from pathlib import Path

    model = whisper.load_model("small")        # first run downloads ~460 MB once
    
    audio_path = Path("audio/clean.wav")
    if not audio_path.exists():
        audio_path = Path("../audio/clean.wav")
    if not audio_path.exists():
        audio_path = Path("audio-search/audio/clean.wav")
        
    result = model.transcribe(str(audio_path))

    print(result["text"])
    print("\n--- with timestamps ---\n")
    for seg in result["segments"]:
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}")
    return (result,)


@app.cell
def _(result):
    import json
    from pathlib import Path

    out_path = Path("transcript.json")
    # If this is run from notebooks/, save it in the parent directory so the MCP server can access it
    if Path("../audio_mcp_server.py").exists():
        out_path = Path("../transcript.json")
    elif Path("audio-search/audio_mcp_server.py").exists():
        out_path = Path("audio-search/transcript.json")

    with open(out_path, "w") as f:
        json.dump(result["segments"], f, indent=2)

    print(f"Saved {out_path}")
    return (json,)


@app.cell
def _(json):
    def _():
        import ollama
        from pathlib import Path
        
        transcript_path = Path("transcript.json")
        if not transcript_path.exists():
            transcript_path = Path("../transcript.json")
        if not transcript_path.exists():
            transcript_path = Path("audio-search/transcript.json")

        # Load the transcript we saved in Phase 3
        with open(transcript_path) as f:
            segments = json.load(f)

        # Turn it into a timestamped text block the model can read
        transcript_text = "\n".join(
            f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments
        )

        # Ask anything in plain language here:
        question = "Did the customer withdraw cash?"

        prompt = f"""You are searching a bank audio transcript.
        Answer using ONLY the transcript below.
        Always include the timestamp (in seconds) where the answer is found.
        If the answer is not in the transcript, say so.

        Transcript:
        {transcript_text}

        Question: {question}
        """

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        return print(response["message"]["content"])

    _()
    return


@app.cell
def _():
    import marimo as mo
    search = mo.ui.text(
        placeholder="Ask about the recording…",
        label="Ask the audio:",
        full_width=True,
    ).form()
    search
    return mo, search


@app.cell
def _(mo, search):
    def _answer(q):
        import json, ollama
        from pathlib import Path
        
        transcript_path = Path("transcript.json")
        if not transcript_path.exists():
            transcript_path = Path("../transcript.json")
        if not transcript_path.exists():
            transcript_path = Path("audio-search/transcript.json")
            
        with open(transcript_path) as f:
            segs = json.load(f)
            
        transcript = "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segs)
        prompt = (
            "You are searching a bank audio transcript. "
            "Answer using ONLY the transcript below. "
            "Always include the timestamp in seconds where the answer is found. "
            "If the answer is not in the transcript, say so.\n\n"
            f"Transcript:\n{transcript}\n\nQuestion: {q}"
        )
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]

    mo.stop(not search.value, mo.md("*Type a question above and press the submit button.*"))
    mo.md(f"**Answer:**\n\n{_answer(search.value)}")
    return


if __name__ == "__main__":
    app.run()
