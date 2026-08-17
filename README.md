# TIMIT × self-supervised speech models

One TIMIT utterance worked through end to end on five self-supervised speech encoders, with
the sample-to-frame alignment derived from scratch and verified empirically rather than
assumed.

The utterance is `TRAIN/DR1/FCJF0/SA1` — *"She had your dark suit in greasy wash water all
year."*

## Models

| # | Model | HF checkpoint |
|---|-------|---------------|
| A | wav2vec 2.0 base | `facebook/wav2vec2-base` |
| B | WavLM base | `microsoft/wavlm-base` |
| C | WavLM base+ | `microsoft/wavlm-base-plus` |
| D | HuBERT base | `facebook/hubert-base-ls960` |
| E | data2vec audio base | `facebook/data2vec-audio-base` |

## Structure

`Init_Play.ipynb` is organised as **Section 0 (shared helpers) → model (A–E) → task (1/2/3) →
step**. Each model section is self-contained once Section 0 has run, and the three tasks are
independent of one another.

1. **Read one utterance** — parse the NIST SPHERE header and PCM, read `.TXT`/`.WRD`/`.PHN`,
   play the audio, plot the waveform with phone and word tiers overlaid, crop and play
   individual phones.
2. **Sample-to-model-frame alignment** — print the conv output and all 13 hidden-state shapes,
   map `.PHN`'s `[start_sample, end_sample)` onto encoder frames two different ways, check
   whether a frame's receptive-field centre really lands inside the target phone, and
   visualise the hidden-state rows for a phone span.
3. **Probe the frozen encoder's layers** — freeze and `eval()`, forward with
   `output_hidden_states=True`, then per-layer statistics, layer-to-layer cosine and linear
   CKA, adjacent-frame boundary contrast, self-similarity matrices, and per-layer phone
   separability.

## Getting the data

**TIMIT is not included in this repository.** It is distributed by the LDC as
[LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1) and is licensed, not freely redistributable.
You need your own copy.

Point the notebook at it by editing `UTT_STEM` in Section 0.1:

```python
UTT_STEM = "/path/to/TIMIT/TRAIN/DR1/FCJF0/SA1"   # no extension
```

TIMIT's `.WAV` files are **NIST SPHERE**, not RIFF/WAVE. `libsndfile` (and therefore
`soundfile`) reads them natively; the notebook also includes a manual header parser as a
fallback, which is verified to produce bit-identical output.

## Requirements

Developed and run on Python 3.12 with:

```
torch          2.13.0
transformers   5.15.0
soundfile      0.14.0
numpy          2.1.3
pandas         2.3.0
matplotlib     3.11.0
scikit-learn   1.6.1
jinja2         3.1.6        # for the styled pandas tables
ipywidgets                  # optional: enables the phone-picker dropdown
```

The five checkpoints download automatically from the Hugging Face Hub on first run
(~6 GB total). The notebook runs on CPU in about 30 seconds once they are cached.

Notebook outputs are stripped from version control — run it to regenerate them.

## Some things that came out of it

All five models share an identical convolutional feature extractor
(`kernel = [10,3,3,3,3,2,2]`, `stride = [5,2,2,2,2,2,2]`), giving a stride of 320 samples
(20 ms hop) and a receptive field of 400 samples (25 ms window). Frame `t` is therefore
computed from samples `[320t, 320t+400)`.

- **The receptive field is exact — but only for the convolutions.** Replacing the conv stack's
  normalisation with `Identity` and perturbing a single sample confirms the
  `[320t, 320t+400)` span exactly, for every model. With normalisation left in, however,
  wav2vec2 / WavLM / HuBERT leak globally: their GroupNorm takes statistics along the *time*
  axis, and the inside-vs-outside sensitivity ratio is only 0.5–1.5×. data2vec uses per-frame
  LayerNorm and stays strictly local. Past the first self-attention block the receptive field
  is the whole utterance regardless, so frame-to-phone alignment is a *positional*
  correspondence, not a bound on where information came from.
- **Short phones vanish from centre-based labels.** In this utterance `d` (8.8 ms) and `dx`
  (15.4 ms) receive no frame at all: frame centres sit on a rigid 320-sample grid, and a phone
  shorter than that can fall entirely between two of them. `d` does — yet frame 39's receptive
  field *fully contains* it. The audio is in the model's input; only the label is lost.
- **Centre-mapping partitions the frames exactly** (sum = 145 = T), which is what makes it
  usable for per-frame labels; overlap-mapping deliberately double-counts (186).
- **Every layer of every model retains phone-boundary information.** Adjacent-frame cosine
  similarity is higher within a phone than across a boundary at all 13 layers of all 5 models.
  The gap is largest at `hidden_states[0]` and shrinks with depth as attention smooths across
  time — which also independently confirms the alignment is correct.

The per-layer phone-separability curves (silhouette, linear probe) are included as a **method
demonstration and are deliberately labelled as unreliable**: one utterance gives 138 frames
over 6 classes, which is far too little to reproduce published layer-wise trends. Scaling that
up needs hundreds of utterances and speaker-disjoint splits.
