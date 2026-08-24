# Positional dependence of Whisper's predictions

**Objective: determine whether *where* a short utterance sits inside Whisper's fixed 30 s input
window changes what the model transcribes — and if so, why, and whether it survives scale.**

Whisper always encodes exactly 30 s. A 3 s utterance therefore occupies ~10% of the window and
the remaining ~90% is padding. Nothing about the acoustics changes when that utterance is moved
from the start of the window to the end; the audio is bit-identical and only its position differs.
If the transcript changes, that is a property of the model, not of the speech.

It does change — substantially, and only under some decoding configurations.

## Headline result

Whisper `base`, 1000 TIMIT `TEST` utterances, corpus WER at six offsets:

| offset | timestamps ON | timestamps OFF |
|--------:|-----:|-----:|
| 0 s  | 0.0705 | 0.0810 |
| 5 s  | 0.0970 | 0.0732 |
| 10 s | 0.0962 | 0.0779 |
| 15 s | 0.1129 | 0.0707 |
| 20 s | 0.1547 | 0.0716 |
| 25 s | **0.2981** | 0.0780 |

- **With timestamps disabled, position is nearly irrelevant** — WER stays inside 0.071–0.081 with
  no trend.
- **With timestamps enabled, WER quadruples** across the window.
- **But the median per-file WER is 0.0000 at every offset in both arms.** The corpus effect is
  entirely tail-driven: at a 25 s offset the top 1% of files carry 44% of all errors and 612/1000
  files are still transcribed perfectly. Position does not gradually degrade transcription — it
  raises the probability of *catastrophic looping hallucination*. Runaway outputs (>3× the
  reference length) go 0 → 19 as the offset grows; insertions go 114 → 1451.
- Example: `DR6/MESD0/SI1632`, reference *"Fuss, fuss, old man."* At a 25 s offset the model
  returns *"I'm going to be a little bit more careful."* repeated until the token limit. The same
  file at offset 0 is exact.

Because the effect is tail-driven, **per-file averages hide it** and small corpora understate it.
Quote the corpus WER, and keep the corpus large enough to contain the tail.

## Notebooks

| Notebook | Question it answers |
|---|---|
| `Aug_23.ipynb` | Does position change the transcript? Isolates the effect with `whisper.decode()` — one encoder pass, one greedy decode, nothing else. |
| `Transcribe_Offset.ipynb` | What does the full `transcribe()` pipeline add on top? Same clips, greedy and stock-fallback arms. |
| `Colab_ModelScaling.ipynb` | Does the effect shrink with model size? `tiny`/`base`/`small`/`medium` on a Colab GPU. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgrfhyL/audio_model_initial_testing/blob/main/Colab_ModelScaling.ipynb) |

Earlier, discontinued work lives in [`archive/`](archive/) — a five-encoder self-supervised
probing study and a 10-file Whisper WER warm-up. Neither is being continued.

## What makes the comparison valid

The experiment is only meaningful if moving the audio changes *nothing but* its position. Three
controls establish that, and each is asserted in the notebooks rather than assumed:

- **The mel is bit-identical under shift.** Offsets are integer multiples of `HOP_LENGTH = 160`,
  so shifting the waveform shifts the mel frames exactly. Verified across all 1000 files at
  offsets 5–25 s: maximum deviation `0.000e+00`.
- **The global scale does not move.** `log_mel_spectrogram` clamps against `log_spec.max() - 8.0`
  computed over the *entire* 30 s frame, so a position-dependent maximum would silently rescale
  everything. Drift measured at `0.000e+00`.
- **Offset 0 is the one exception, by construction.** `torch.stft(center=True)` reflect-pads 200
  samples; at offset 0 that mirrors the utterance's own opening instead of zeros, perturbing mel
  frames 0–1. Offset 0 is still the canonical `pad_or_trim` condition and belongs in the results —
  it just is not a pure translation of the others.

Padding is zeros only, written into an exact `np.zeros(480000, float32)` buffer — never
`pad_or_trim`, which right-pads only and cannot express an offset. Decoding is greedy
(`temperature=0.0`, no beam, no best-of), language forced to `en`, task `transcribe`, no prompt or
prefix. Both hypothesis and reference pass through Whisper's own `EnglishTextNormalizer`, and WER
is pooled corpus-level, not a mean of per-file rates.

## Two mechanisms the follow-ups isolated

**`transcribe()` differs from `decode()` only by re-seeking.** With timestamps off the two are
bit-identical at every offset — without timestamp tokens the seek loop always advances a full
window and terminates in one pass. With timestamps on they diverge on exactly the files where the
loop re-seeks (19/19 at 15 s, 67/67 at 20 s, 120/121 at 25 s): when the closing timestamp falls
short of the window end, the tail is decoded again and a duplicate fragment is appended.

**The temperature fallback is what rescues the late-offset collapse.** Whisper's stock
`temperature=(0.0, 0.2, … 1.0)` re-decodes a window whose compression ratio exceeds 2.4. That cuts
the 25 s offset from 0.2981 to 0.1626 while firing on only 109/1000 files. A scalar
`temperature=0.0` makes the ladder unreachable, so `compression_ratio_threshold` and
`logprob_threshold` flag the runaway output and return it anyway. The position effect survives the
rescue regardless — stock fallback still runs 0.069 → 0.163 across the window.

Practical consequence: **on short clips with timestamps enabled, keep the stock temperature
fallback.** Switching to strict greedy silently disarms the one mechanism that catches positional
hallucination.

## Corpus

1000 TIMIT `TEST` utterances: under 5 s (so a 25 s offset still fits the window), at most 4.975 s
(so ≥400 zero samples of tail remain and the STFT's edge reflection stays over silence), excluding
the `SA*` shibboleth sentences that every speaker records. Drawn speaker-balanced with
`random.Random(0)` across all 168 speakers — 482 distinct sentences, 7918 reference words.

`corpus_digests.json` ships that selection as **paths and SHA-256 digests only, with no transcript
text**, ordered by the original draw so an N-clip subset is the first N entries. It lets any
machine rebuild and *verify* the identical corpus from its own TIMIT copy. The digest is taken
over the decoded float32 samples, so it pins the array that actually reaches the mel.

Locally, `corpus_frozen/` holds a self-contained copy plus provenance; `load_frozen()` (section 9
of `Aug_23.ipynb`) is the entry point for follow-up work and needs no TIMIT install.

### Subset size matters more than it looks

Because the effect lives in the tail, a small subset does not merely add noise — it systematically
understates the finding:

| N | offset-25 WER (ts on) | max deviation from the full curve |
|---:|---:|---:|
| 300 | 0.2064 | 0.092 |
| 500 | 0.2471 | 0.051 |
| 700 | 0.2975 | 0.005 |
| 1000 | 0.2981 | 0 |

At 300 clips only 3 runaway files survive instead of 19, and the timestamps-off curve gains a
spurious spike at offset 0 from a single hallucination — which would make the "position is
irrelevant without timestamps" conclusion look false.

## Getting the data

**TIMIT is not included in this repository.** It is distributed by the LDC as
[LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1) and is licensed, not freely redistributable. You
need your own copy. Point `TIMIT_TEST` in `Aug_23.ipynb` at it.

TIMIT's `.WAV` files are **NIST SPHERE**, not RIFF/WAVE — `libsndfile` (and therefore `soundfile`)
reads them natively, returning 16 kHz mono float32. `scipy.io.wavfile` cannot. The layout is
`SPLIT/DR{1..8}/SPEAKER/UTT.WAV`, so a flat `os.listdir` finds nothing; reference transcripts are
in a sibling `.TXT` whose first two integers are sample offsets and must be stripped.

Everything that embeds TIMIT audio or transcripts is gitignored: `corpus_frozen/`,
`offset_manifest.json`, `offset_results.csv`, `transcribe_results.csv`.

## Requirements

Python 3.12, with `openai-whisper` (20250625), `torch` 2.13, `soundfile`, `jiwer`, `numpy`,
`matplotlib`. Whisper checkpoints download on first use and cache to `~/.cache/whisper`.

The local experiments ran on Apple M2 via MPS in fp32 — verified deterministic, batch-size
invariant, and text-identical to CPU. The mel is always computed on **CPU in fp32**, which is a
control rather than a performance choice: it guarantees the bit-identical-shift property no matter
what the encoder runs on. `Colab_ModelScaling.ipynb` uses CUDA and fp16, and reads `n_mels` from
`model.dims` rather than hardcoding 80, since `large-v3` and `turbo` use 128.
