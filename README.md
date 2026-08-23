# TIMIT × self-supervised speech models

One TIMIT utterance worked through end to end on five self-supervised speech encoders, with
the sample-to-frame alignment derived from scratch and verified empirically rather than
assumed.

The utterance is `TRAIN/DR1/FCJF0/SA1` — *"She had your dark suit in greasy wash water all
year."*

## Notebooks

| Notebook | What it does |
|---|---|
| `Init_Play.ipynb` | One TIMIT utterance through five self-supervised encoders (below) |
| `Whisper_Play.ipynb` | Whisper `base` ASR + WER on a 10-utterance TIMIT sample, with audio playback |
| `Offset_Play.ipynb` | Does *where* an utterance sits in Whisper's 30 s window change the transcript? |

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

---

# Where an utterance sits in Whisper's 30 s window

`Offset_Play.ipynb`. Whisper always encodes a fixed 30 s input, so a 3 s utterance fills ~10% of
the window and the rest is padding. This asks whether **position** inside the window changes the
transcript, holding the audio bit-identical and moving only where it sits.

1000 TIMIT `TEST` utterances (< 5 s, no `SA*`, 168 speakers, 482 distinct sentences) x 6 offsets
(0, 5, 10, 15, 20, 25 s) x 2 arms (timestamp tokens on / off) = 12 000 decodes, ~22 min on MPS.

Controls: fp32 throughout, greedy, `language="en"`, `task="transcribe"`, no prompt or prefix
(the `whisper.decode()` equivalent of `condition_on_previous_text=False` with an empty
`initial_prompt`), zero padding only, and Whisper's own `EnglishTextNormalizer` applied to both
hypothesis and reference. WER is pooled corpus-level, not a mean of per-file rates.

`whisper.decode()` is used rather than `model.transcribe()`: `transcribe()` wraps the decoder in
a seek loop plus `no_speech_threshold` / `logprob_threshold` / `compression_ratio_threshold`, any
of which can blank or re-roll a segment in an offset-dependent way - confounding the exact effect
being measured.

## Results

| offset | WER (timestamps ON) | WER (timestamps OFF) |
|--------:|-----:|-----:|
| 0 s  | 0.0705 | 0.0810 |
| 5 s  | 0.0970 | 0.0732 |
| 10 s | 0.0962 | 0.0779 |
| 15 s | 0.1129 | 0.0707 |
| 20 s | 0.1547 | 0.0716 |
| 25 s | **0.2981** | 0.0780 |

- **With timestamps off, position is nearly irrelevant** - WER stays in 0.071-0.081 with no
  trend. The encoder handles a late-placed utterance fine.
- **With timestamps on, WER quadruples** from 0.070 to 0.298 across the window.
- **But the median per-file WER is 0.0000 at every offset in both arms.** The corpus rise is
  entirely tail-driven: at offset 25 s the top 1% of files carry 44% of all errors, and 612/1000
  files are still transcribed perfectly. Position does not gradually degrade transcription - it
  raises the probability of *catastrophic looping hallucination*. Files with runaway output
  (>3x the reference length) go 0 -> 19 as the offset grows, and insertions go 114 -> 1451.
- Example at 25 s: `DR6/MESD0/SI1632` ("Fuss, fuss, old man.") returns "I'm going to be a little
  bit more careful." repeated until the token limit. The same file at offset 0 is exact.

## Two things the controls caught

- **The mel really is shift-invariant, but only away from the array edge.** Offsets 5-25 s give
  *bit-identical* mel frames for all 1000 files (max deviation 0.000e+00), which is what makes
  the comparison valid. Offset 0 does not: `torch.stft(center=True)` reflect-pads 200 samples, so
  at offset 0 the padding mirrors the utterance's own opening instead of zeros, perturbing mel
  frames 0-1. Offset 0 is still the canonical `pad_or_trim` condition and belongs in the plot -
  it just is not a pure translation of the others.
- **`max_initial_timestamp=1.0` is a confound in the timestamps-on arm.** Whisper forbids the
  first timestamp token from exceeding 1.0 s, so an utterance at 25 s cannot open with
  `<|25.00|>` - it is forced to `<|0.00|>` while the closing timestamp still tracks the true end.
  Part of the ON curve is therefore a hard-coded decoding rule, not the model's acoustic response
  to position. The stock value is kept (this is Whisper as shipped) and recorded as a control.

Outputs (`offset_manifest.json`, `offset_results.csv`, the PNGs) are gitignored - the manifest
and CSV embed TIMIT reference transcripts, which are licensed.
