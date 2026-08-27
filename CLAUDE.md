# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Objective: the positional dependence of Whisper's predictions** — whether *where* a short
utterance sits inside the fixed 30 s input window changes what the model transcribes.

A research scratchpad, one experiment per notebook. There is **no package, no test suite, no
build step, and no shared Python module** — each notebook is deliberately self-contained and
duplicates its own helpers. Do not refactor shared logic into a `.py` module unless asked; the
redundancy is the design.

| Notebook | Experiment |
|---|---|
| `Aug_23.ipynb` | The core experiment: does position change the transcript? 1000 files × 6 offsets × 2 timestamp arms, via `whisper.decode()` |
| `Transcribe_Offset.ipynb` | Same clips through `transcribe()`, in a greedy and a stock-fallback arm |
| `Colab_ModelScaling.ipynb` | Does the effect shrink with model size? tiny/base/small/medium on a Colab GPU (CUDA, fp16) |
| `Colab_DeltaSweep.ipynb` | Per-utterance difference-in-differences on the held-out 300 — which utterances carry the timestamp-specific positional penalty |

`archive/` holds **discontinued** work — `Init_Play.ipynb` (five SSL encoders) and
`Whisper_Play.ipynb` (10-file Whisper WER warm-up). Neither is being continued; do not extend them
or treat them as the current line of work. Both use absolute TIMIT paths so they still run in
place. `Whisper_Play.ipynb`'s stored outputs embed TIMIT audio as base64 — it is already in git
history, so removing it needs a history rewrite, not just a commit.

Note the active notebooks read **bare relative paths** (`corpus_frozen/`, `offset_results.csv`,
`offset_manifest.json`) and that data sits gitignored at the repo root. Moving those notebooks
into a subdirectory breaks them unless the paths are rewritten.

## Running notebooks

**`nbconvert` is not installed; `nbclient`/`ipykernel` are.** So:

```bash
jupyter execute --inplace Aug_23.ipynb        # runs with a real kernel, saves outputs inline
```

`jupyter nbconvert` will fail — reach for `jupyter execute` instead. To run cells headlessly
without a kernel (no inline figures, notebook outputs stay empty), extract and `exec` the code
cells in order in one namespace; this is equivalent computationally but does not update the
`.ipynb`.

There is nothing to lint or build. The closest thing to a test suite is the verification cell at
the end of `Aug_23.ipynb` (section 8), which asserts manifest integrity, offset legality, mel
invariance, zero-padding, decode determinism, grid completeness, and figure existence. Run that
section to check the pipeline end to end.

## TIMIT: licensing and format

TIMIT is **LDC93S1 — licensed, not redistributable**. Never commit audio, `.TXT`/`.PHN`/`.WRD`
transcripts, or any derived artifact that embeds reference transcripts. `.gitignore` already
covers `*.WAV`, `corpus_frozen/`, `offset_manifest.json`, and `offset_results.csv` — the last two
because they carry reference text. Section 9 of `Aug_23.ipynb` hard-asserts via
`git check-ignore` before writing licensed audio, and that guard should be preserved in anything
similar.

TIMIT `.WAV` files are **NIST SPHERE, not RIFF/WAVE**. `soundfile` (libsndfile) reads them
natively and returns 16 kHz mono float32 — `sf.read(path, dtype="float32")`. A flat
`os.listdir()` finds no `.WAV` at all; the layout is `SPLIT/DR{1..8}/SPEAKER/UTT.WAV` and needs a
recursive glob. Reference transcripts live in a sibling `.TXT` whose first two integers are
start/end sample offsets and must be stripped: `f.read().strip().split(None, 2)[2]`.

The TIMIT root is **hardcoded per notebook** and must be edited when the install moves:

- `Aug_23.ipynb` → `TIMIT_TEST`
- `Colab_ModelScaling.ipynb` → `DRIVE_ROOT` (auto-locates `TEST/` beneath it)
- `archive/Whisper_Play.ipynb` → `TIMIT_TRAIN`
- `archive/Init_Play.ipynb` → `UTT_STEM` (a single utterance stem, no extension)

`SA1`/`SA2` are the two shibboleth sentences recorded by *every* speaker; exclude them from any
corpus-level metric or they dominate the score.

## The offset experiment (`Aug_23.ipynb`)

Pipeline, in order: corpus manifest → window placement → mel-invariance gate → decode grid →
corpus WER → figures → verification → corpus freeze.

**Caching and resumability are load-bearing.** `offset_results.csv` is keyed by
`(path, offset_s, timestamps)` and appended incrementally; the decode cell skips any cell already
present. A full grid is ~22 min on MPS, a re-run against the cache ~40 s. Deleting the CSV means
re-decoding everything. `offset_manifest.json` is likewise treated as frozen once written — the
selection is never re-derived.

**Experimental invariants that must not be broken:**

- Offsets must be integer multiples of `HOP_LENGTH = 160`, or shifting the audio is not a clean
  shift of the mel frames and the whole comparison collapses.
- Utterances must be < 5 s so that `offset 25 s + duration` fits the 30 s window, and
  ≤ 4.975 s so ≥ 400 zero samples of tail remain (the STFT reflect-pads 200 samples at the array
  edge).
- Audio is placed via `np.zeros(480000, float32)` with a slice assignment — **never**
  `pad_or_trim`, which only right-pads and cannot express an offset.
- Mel is always computed on **CPU in fp32**; `fp16=False` everywhere.
- Offsets 5–25 s give *bit-identical* mel frames. **Offset 0 does not**, and this is expected:
  `torch.stft(center=True)` reflect-pads, so at offset 0 the padding mirrors the utterance's own
  opening rather than zeros, perturbing mel frames 0–1. Offset 0 is still the canonical
  `pad_or_trim` condition and belongs in results — it just is not a pure translation.

**Use `whisper.decode()`, not `model.transcribe()`.** `transcribe()` wraps the decoder in a seek
loop plus temperature fallback (`compression_ratio_threshold`, `logprob_threshold`) and a
`no_speech_threshold` skip, each of which fires in an *offset-dependent* way and confounds the
measurement. `decode()` is one encoder pass + one greedy decode, and accepts a batched mel
`(B, 80, 3000)`. `condition_on_previous_text=False` and an empty `initial_prompt` map onto
`prompt=None` / `prefix=None`, since `decode()` carries no state between calls.

`max_initial_timestamp=1.0` (stock) forbids the first timestamp token from exceeding 1.0 s, so an
utterance at 25 s cannot open with `<|25.00|>`. This measurably changes transcribed text and is a
known confound in the timestamps-on arm; it is kept at stock and recorded as a control rather
than silently altered.

WER is **corpus-level** (pooled S+D+I over pooled reference length via `jiwer.process_words` on
lists), not a mean of per-file rates. Both hypothesis and reference go through Whisper's own
`EnglishTextNormalizer`. Note that median per-file WER is 0.0 at every offset — the corpus effect
is entirely tail-driven by looping hallucinations, so quoting per-file averages hides the finding.

## transcribe() vs decode()

`Transcribe_Offset.ipynb` re-runs the offset experiment through `model.transcribe()`. Two traps
it documents, both verified:

- **`initial_prompt=""` is not "no prompt".** `transcribe()` computes
  `tokenizer.encode(" " + initial_prompt.strip())`, which for `""` returns `[220]` — a literal
  space token wrapped in `<|startofprev|>`. Use `initial_prompt=None` to match `decode()`'s
  `prompt=None`.
- **A scalar `temperature=0.0` disarms the fallback ladder.** `decode_with_fallback` iterates a
  temperature tuple; with a scalar there is no second temperature, so
  `compression_ratio_threshold=2.4` and `logprob_threshold=-1.0` flag a bad window and return it
  anyway. Only `no_speech_threshold` stays live. Passing the stock tuple re-enables the ladder but
  makes decoding stochastic — that arm seeds per-cell from a hash of
  `(path, offset, timestamps, arm)` so it stays reproducible and resume-safe.

Empirically: with timestamps **off**, `transcribe()` greedy is bit-identical to `decode()` at
every offset (the seek loop never re-seeks without timestamp tokens). With timestamps **on**, the
two diverge on exactly the files where the loop re-seeks. Results cache is
`transcribe_results.csv`, keyed by `(path, offset_s, timestamps, fallback)`.

## Model scaling on Colab

`Colab_ModelScaling.ipynb` runs the offset experiment across model sizes. Three things differ from
the local notebooks and matter:

- **`n_mels` comes from `model.dims.n_mels`, never hardcoded 80.** `large-v3` and `turbo` use 128;
  hardcoding mis-shapes the encoder input on those checkpoints.
- **fp16 on CUDA**, so it will not reproduce the local fp32/MPS numbers exactly — `base` is in the
  sweep as the anchor that keeps the sizes internally comparable. The mel stays fp32 on CPU
  (`decode()` calls `mel.half()` itself), preserving the bit-identical-shift property.
- **`corpus_digests.json`** (committed, root) carries paths + `sha256_audio` only, no transcript
  text, ordered by the original draw. Colab rebuilds and verifies the corpus from a private TIMIT
  copy rather than receiving licensed data.

Subset size is not free: the effect is tail-driven, so 300 clips understate offset-25 WER by ~31%
relative (0.2064 vs 0.2981). 700 tracks the full curve within 0.005.

**Already run** (700 clips, T4, 41 min): the positional penalty (WER at 25 s / WER at 0 s,
timestamps on) shrinks monotonically with scale but does not vanish — tiny 14.4×, base 4.3×,
small 3.6×, medium 2.2×. With timestamps off every model is flat. `tiny` exceeds 100% WER at a
25 s offset (1.4752) because it emits several times the reference length in looping
hallucination. Results live on Drive at `MyDrive/NAACL/scaling_results.csv`, not in the repo.

## Per-utterance delta sweep

`Colab_DeltaSweep.ipynb` computes
`delta_m = [WER(25s,on)-WER(5s,on)] - [WER(25s,off)-WER(5s,off)]` per utterance, over draw slice
`[700:1000]` — disjoint from the scaling sweep's `[0:700]`, still all 168 speakers. Baseline is
5 s, not 0 s, because offset 0's mel is not a pure shift (reflect padding).

`delta_m` is **zero-inflated and heavy-tailed** (local base preview: 237/300 exactly zero, top 5
carrying 69% of the total). Never summarize it with a mean alone — report prevalence (sign split)
and severity separately.

### Licensing discipline for Colab notebooks

Saving a Colab notebook back to GitHub commits its outputs, so **a printed transcript is a
committed transcript**. The rules these notebooks follow:

- never print reference or hypothesis text — counts, digests and numeric scores only
- split results: a Drive-only file carrying hypotheses (`delta_results_full.csv`, gitignored) and
  a git-safe file of paths and numbers (`delta_per_utterance.csv`)
- assert the split before writing: no forbidden column names, no free-text values
- pin references with a SHA-256 over `path \t normalized_reference` rather than storing text

Note `.gitignore` does **not** support inline comments — `foo.csv  # note` is a literal pattern
that matches nothing. Put comments on their own line.

### Persistence

All outputs go to `MyDrive/NAACL/`, never to the runtime's local disk, so the runtime can be
disconnected once section 7 has run:

- `delta_per_utterance.csv` — one row per (model, utterance) with `delta_m`, the four component
  WERs, both inner differences, and `speaker` / `region` / `sec` / `n_ref_words`. Keeping
  `n_ref_words` is what lets per-file WERs be re-pooled into a corpus-level number later; a file
  of WER ratios alone could not.
- `delta_results_full.csv` — raw hypotheses, appended batch by batch with `flush()`, so a
  disconnect mid-sweep only costs the batch in flight and the run resumes.
- `delta_provenance.json` — versions, digests, decoding options, summary with CIs.

Section 10 is a **standalone reload cell**: no GPU, no TIMIT, no earlier cells. Section 7 reads
`delta_per_utterance.csv` back and compares it to the in-memory rows, which proves the write
actually reached Drive rather than sitting in a FUSE buffer.

### Provenance recorded

`delta_provenance.json` captures: package versions; SHA-256 of `whisper`'s `audio.py`,
`decoding.py`, `model.py` (version strings are coarse, hashing the code that runs is not); each
checkpoint's on-disk digest verified against the hash embedded in whisper's download URL
(`url.split("/")[-2]`); GPU name/capability and precision; verbatim `DecodingOptions`; and digests
over both the audio arrays and the normalized references.

## Frozen corpus

`corpus_frozen/` is a self-contained copy of the 1000 selected clips plus `corpus_manifest.json`
(every manifest field, plus `sha256_file` over SPHERE bytes and `sha256_audio` over decoded
float32 samples, plus provenance: library versions, the full `CONTROLS` dict, and a digest of
`offset_results.csv`).

Follow-up experiments should start from `load_frozen()` (section 9 of `Aug_23.ipynb`) rather than
the TIMIT install — it needs no TIMIT copy and no path fixups, and re-verifies every clip against
its `sha256_audio` on load. To claim comparability with existing numbers, assert that
`results_csv_sha256` in the frozen manifest still matches the `offset_results.csv` being compared
against. Paths in the frozen manifest are the same `path` keys used in `offset_results.csv`, so
rows join with no translation.

## Environment

Python 3.12, torch 2.13 with **MPS available**. On this machine MPS decoding was verified
run-to-run deterministic, batch-size invariant, and text-identical to CPU — so MPS at batch 16 is
the default for decode grids, with mel kept on CPU. `whisper` is `openai-whisper` (20250625);
`jiwer` has no `__version__` attribute.
