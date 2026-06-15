import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 🎧 Audio Clip-Classification — Dataset Explorer

        Walk through the **surveillance-datasets** pipeline:
        **load → inspect → decode → PyTorch / HuggingFace → evaluate**

        Clip-classification assigns **one label per audio clip**
        (e.g. *siren*, *gunshot*, *dog bark*).
        """
    )
    return (mo,)


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options={"ESC-50 (50 classes, ~600 MB)": "esc50",
                 "UrbanSound8K (10 classes, ~6 GB)": "urbansound8k",
                 "SESA (4 classes, ~26 MB)": "sesa"},
        value="ESC-50 (50 classes, ~600 MB)",
        label="Dataset",
    )
    split_input = mo.ui.text(value="fold3", label="Split")

    mo.hstack([dataset_picker, split_input], justify="start", gap=1)
    return dataset_picker, split_input


@app.cell
def _(dataset_picker, mo, split_input):
    import surveillance_datasets as svd

    ds = svd.load(dataset_picker.value, split=split_input.value)

    mo.md(
        f"""
        ## 1 · Dataset loaded

        | Property | Value |
        |----------|-------|
        | **Name** | `{ds.name}` |
        | **Split** | `{ds.split}` |
        | **Samples** | {len(ds)} |
        | **Classes** | {len(ds.classes)} |
        | **Tasks** | {', '.join(ds.tasks)} |

        Classes: {', '.join(f'`{c}`' for c in ds.classes[:15])}{'…' if len(ds.classes) > 15 else ''}
        """
    )
    return (ds,)


@app.cell
def _(ds, mo):
    sample_slider = mo.ui.slider(
        start=0, stop=len(ds) - 1, value=0, label="Sample index", show_value=True
    )
    sample_slider
    return (sample_slider,)


@app.cell
def _(ds, mo, sample_slider):
    s = ds[sample_slider.value]

    mo.md(
        f"""
        ## 2 · Sample inspection

        | Field | Value |
        |-------|-------|
        | **ID** | `{s.id}` |
        | **Modality** | `{s.modality}` |
        | **Media path** | `{s.media.path}` |
        | **Clip labels** | {s.clip_labels} |
        """
    )
    return (s,)


@app.cell
def _(mo, s):
    mo.md(f"""
    **▶ Listen to sample `{s.id}`:**
    """)
    return


@app.cell
def _(mo, s):
    mo.audio(src=s.media.path)
    return


@app.cell
def _(mo, s):
    import matplotlib.pyplot as plt
    import numpy as np

    wav, sr = s.audio(sample_rate=16000, mono=True)
    wav = wav.squeeze()  # (1, N) → (N,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # Waveform
    t = np.arange(len(wav)) / sr
    axes[0].plot(t, wav, linewidth=0.4, color="#4f8cff")
    axes[0].set_title(f"Waveform — {s.id}", fontsize=10)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, t[-1])

    # Spectrogram
    axes[1].specgram(wav, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
    axes[1].set_title("Spectrogram", fontsize=10)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")

    fig.tight_layout()

    mo.md(
        f"""
        ## 3 · Audio decoded

        **Shape:** `{wav.shape}` &nbsp;|&nbsp; **Sample rate:** `{sr}` Hz
        &nbsp;|&nbsp; **Duration:** `{len(wav)/sr:.2f}` s
        """
    )
    return (fig,)


@app.cell
def _(fig, mo):
    mo.as_html(fig)
    return


@app.cell
def _(ds, mo):
    tds = ds.to_torch(task="clip_classification", sample_rate=16000)
    item = tds[50]

    mo.md(
        f"""
        ## 4 · PyTorch export

        ```
        item keys : {list(item.keys())}
        audio shape: {item['audio'].shape}
        target     : {item['target']}  (class index)
        ```
        """
    )
    return (tds,)


@app.cell
def _(ds):
    ds.classes[15]  # → e.g. "engine"
    return


@app.cell
def _(mo, tds):
    from torch.utils.data import DataLoader
    from surveillance_datasets.integrations.torch import make_collate

    loader = DataLoader(
        tds, batch_size=8, collate_fn=make_collate("clip_classification")
    )
    batch = next(iter(loader))

    mo.md(
        f"""
        ### DataLoader batch

        | Tensor | Shape |
        |--------|-------|
        | `audio.data` | `{tuple(batch['audio']['data'].shape)}` |
        | `audio.lengths` | `{tuple(batch['audio']['lengths'].shape)}` |
        | `target` | `{tuple(batch['target'].shape)}` |

        Audio is **padded** to the longest clip in the batch.
        """
    )
    return


@app.cell
def _(ds, mo):
    hf = ds.to_hf_dataset()

    mo.md(
        f"""
        ## 5 · HuggingFace `datasets` export

        ```
        {hf}
        ```

        **Label feature:** `{hf.features['label']}`

        First row sampling rate: `{hf[12]['audio']['sampling_rate']}` Hz
        """
    )
    return (hf,)


@app.cell
def _(hf):
    audio = hf[0]["audio"]
    audio["array"], audio["sampling_rate"]
    return


@app.cell
def _(hf):
    row = hf[0]
    label_name = hf.features["label"].int2str(row["label"])

    # row["label"] is the integer index, label_name is the class string
    row["label"], label_name
    return


@app.cell
def _(ds, mo):
    gt = ds.ground_truth()
    results = ds.evaluate(gt)

    # Build per-class table rows
    per_class = results.get("per_class", {})
    pc_rows = "\n".join(
        f"| `{cls}` | {vals} |"
        for cls, vals in list(per_class.items())[:10]
    )

    mo.md(
        f"""
        ## 6 · Evaluate (sanity check — perfect predictions)

        | Metric | Value |
        |--------|-------|
        | **Task** | `{results['task']}` |
        | **N** | {results['n']} |
        | **Accuracy** | {results['accuracy']:.4f} |
        | **Macro F1** | {results['macro_f1']:.4f} |

        > ✅ Ground-truth predictions should score **1.0** — confirming the
        > pipeline is wired correctly.

        {f"### Per-class breakdown (first 10)" if pc_rows else ""}
        {"| Class | Metrics |" if pc_rows else ""}
        {"|----|---|" if pc_rows else ""}
        {pc_rows}
        """
    )
    return


@app.cell
def _(mo):
    from transformers import pipeline

    mo.md("## 7 · Real model predictions\n\n⏳ Running pre-trained model on all samples...")
    return (pipeline,)


@app.cell
def _(ds, mo, pipeline):
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    classifier = pipeline(
        "audio-classification",
        model="bioamla/ast-esc50",
        device=device,
    )

    preds = {}
    for sample in ds:
        _wav, _sr = sample.audio(sample_rate=16000, mono=True)
        result = classifier({"array": _wav.squeeze(), "sampling_rate": _sr})
        preds[sample.id] = result[0]["label"]

    model_results = ds.evaluate(preds)

    # Build per-class table
    pc = model_results.get("per_class", {})
    _pc_rows = "\n".join(
        f"| `{cls}` | {vals} |"
        for cls, vals in list(pc.items())[:15]
    )

    mo.md(
        f"""
        ### Model results (`bioamla/ast-esc50`)

        | Metric | Value |
        |--------|-------|
        | **Task** | `{model_results['task']}` |
        | **N** | {model_results['n']} |
        | **Accuracy** | {model_results['accuracy']:.4f} |
        | **Macro F1** | {model_results['macro_f1']:.4f} |

        {f"### Per-class breakdown (first 15)" if _pc_rows else ""}
        {"| Class | Metrics |" if _pc_rows else ""}
        {"|----|---|" if _pc_rows else ""}
        {_pc_rows}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🚀 Next steps

    - **Swap dataset**: change the dropdown above to `sesa` (26 MB, fast)
      or `urbansound8k`
    - **Train a model**: wrap `tds` in a training loop or use
      `SurveillanceDataModule` with PyTorch Lightning
    - **Sound-event detection**: try onset/offset tasks with a different
      notebook
    """)
    return


if __name__ == "__main__":
    app.run()
