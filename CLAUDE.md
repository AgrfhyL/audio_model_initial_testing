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
| `Colab_DeltaSweep.ipynb` | Per-utterance difference-in-differences over all 1000 clips — which utterances carry the timestamp-specific positional penalty |
| `Colab_PositionalEmbedding.ipynb` | P0–P4: displaces the encoder positional embedding directly to test whether it *causes* the positional penalty |
| `Colab_ModelScaling copy.ipynb` | The same sweep re-run at the project-standard configuration: 1000 clips, five models incl. `large-v3`, batch 8. Writes `scaling_results_1000.csv`; supersedes the original for cross-experiment comparisons |
| `Figures.ipynb` | Redraws every figure as **vector PDF** from the stored per-condition CSVs. No GPU, no TIMIT, no decoding |
| `Colab_ExperimentC.ipynb` | Localizes the penalty: teacher-forced ΔNLL, timestamp-margin pressure, and generated behaviour across the model ladder |
| `Colab_Hallucinations.ipynb` | Classifies experiment A's stored hypotheses to count *hallucination prevalence* per model size. No GPU, no TIMIT, no decoding |
| `Colab_EncoderSimilarity.ipynb` | Experiment H: reads the **encoder directly** — cosine/CKA between the utterance's frames at 5 s and 25 s, against a position floor. No decoding, no tokenizer, no references |
| `Colab_HallucinationTaxonomy.ipynb` | Supersedes the counting notebook: a seven-way taxonomy over degeneracy, length pathology and **grounding**. No decoding, but needs TIMIT `.TXT` for the reference text |
| `Colab_LLMVerdict.ipynb` | **The project's hallucination detector** (Atwany et al. 2025, fine-grained prompt): `gpt-5.6-luna` labels the 20 000 hypotheses, and looping counts as hallucination. Submit / collect / analyse phases; raw responses saved to Drive, so analysis never calls the API. **The only notebook that sends reference text off the machine** |

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

Two variants exist. The original ran 700 clips over four models at batch 16 and its numbers are
what the README quotes. `Colab_ModelScaling copy.ipynb` re-runs the same design at the
configuration every other notebook uses — 1000 clips, five models including `large-v3`, batch 8 —
so O, A, B and C all draw on one corpus under one decoding setup. It writes to
`scaling_results_1000.csv` rather than resuming `scaling_results.csv`, because the cached rows came
from 700-clip batches of 16 and reusing them would leave the grid half-decoded under each
configuration.

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

## Editing the positional embedding

`AudioEncoder.positional_embedding` is a `register_buffer` of fixed sinusoids, shape
`(n_audio_ctx, n_state)` = `(1500, n_state)`, added after the conv stack. 1500 frames / 30 s =
**50 frames per second**, so N seconds = `N * 50` rows.

**Displace it with a cyclic roll, never by rebuilding sinusoids at shifted positions.**
Extrapolation gives leading-silence frames negative positions that the model never saw in
training; measured effect is corpus WER 0.0586 -> 6.4163 with 18/60 clips collapsing into
repetition loops. A roll keeps every row a genuine training-time position.

```python
pe.copy_(torch.roll(saved, shifts=int(-seconds * 50), dims=0))   # not sinusoids(arange - k)
```

Use `pe.copy_()` rather than reassignment so dtype/device survive under fp16, and always restore
in a `finally` — `load_model` caches, so a leaked displacement contaminates every later decode.
`rolled_pe` asserts restoration on exit.

Note a +N second displacement **wraps**: audio at 25 s displaced +20 s would be 45 s, outside the
window, so it lands at 15 s. That makes P3 an "incorrectly but in-range relabelled" control rather
than a literal +20 s.

## Experiment C: localizing the penalty (`Colab_ExperimentC.ipynb`)

Rolling the positional embedding shows it is *sufficient* to produce the penalty, but not *where*
the penalty lives — a roll changes the encoder output, which is what the decoder cross-attends to.
Experiment C separates an **encoder account** (the representation at 25 s is degraded) from a
**decoder account** (the representation is fine, but carries a position signal the decoder
mishandles once timestamp tokens are live).

Conditions are C0 (5 s), C1 (25 s), C2 (25 s, PE rolled −20 s), each with timestamps on and off —
6 cells per (model, clip), 30 000 in total across the five-model ladder.

**The discriminator is the shape across scale, not any single model.** The WER penalty is already
known to shrink 14.4× → 2.2×. Encoder account predicts ΔNLL shrinks with it; decoder account
predicts ΔNLL stays flat while the runaway rate falls. So it must run on the same ladder and the
same 1000 clips, or there is nothing to regress against.

### Teacher forcing

`Whisper.logits(tokens, audio_features)` is `decoder(tokens, audio_features)` with `kv_cache=None`,
which is one non-cached forward under the causal mask — exactly teacher forcing. Logits are cast
to fp32 by `TextDecoder.forward`, so the log-softmax is fp32 even under fp16 weights.

`ΔNLL = NLL(C1) − NLL(C0)` is **paired per utterance**, over text tokens only. NLL *levels* are not
comparable across model sizes; the within-model difference cancels baseline competence, which is
what makes it the right cross-scale axis.

- **Sequences are right-padded with EOT.** Inert under a causal mask — verified to 5e-7 in fp32.
- **The encoder runs once per (batch, condition).** `DecodingTask._get_audio_features` skips
  encoding when handed a tensor already shaped `(B, n_audio_ctx, n_audio_state)`, so one
  `audio_features` feeds the forced pass, the free `decode()` and both arms. That also means
  `rolled_pe` need only wrap the encoder call, so a displacement cannot leak into decoding.
- **C2's forced timestamps describe the *perceived* position (5 s), not 25 s.** The roll tells the
  model the audio sits at 5 s; forcing `<|25.00|>` would score a path inconsistent with the model's
  own positional evidence.
- **Scoring reads raw logits, no `LogitFilter`.** Forcing `<|25.00|>` while `max_initial_timestamp`
  masks it to −inf would make the target unreachable and its NLL infinite.

### The margin is Whisper's own rule, not an invented metric

`margin = logsumexp(timestamp logprobs) − max(text logprobs)` is verbatim the decision variable at
the end of `ApplyTimestampRules` in `decoding.py`. When it exceeds 0, every text token is masked to
−inf and the decoder **must** emit a timestamp. That filter is installed only when
`without_timestamps=False`, which is precisely how the two arms diverge while sharing an encoder.

Two counting traps, both hit and fixed:

- **Count crossings over text steps only.** Over all steps the legitimate closing-timestamp step
  always crosses, so the count is trivially ≥ 1 and useless.
- **First-timestamp diagnostics must be conditional on emitting a timestamp** — renormalized within
  the timestamp block. Unconditionally the mass on any single timestamp is ~1e-5, because unmasked
  the model would rather emit text. Conditioned properly, `ts_forbidden_mass` is exactly what
  `max_initial_timestamp=1.0` deletes.

### Precision

**Never call `model.half()`.** Load fp32 and cast only the mel — which is what `decode()` does
internally, so this reproduces the earlier Colab sweeps exactly. PyTorch promotes, the encoder
returns fp16, and the decoder runs in fp16. Two reasons this matters:

- an fp32 → fp16 → fp32 round trip permanently damages the weights, silently contaminating the
  fp32 leg of any precision comparison;
- on **MPS**, `model.half()` makes the encoder emit NaN, while an fp32 model fed a half mel is
  fine. CPU fp16 is unavailable altogether (`layer_norm` rejects mixed dtype), so fp16 cannot be
  tested locally at all — the notebook measures it on the actual GPU instead.

Measured on 64 clips: fp16 vs fp32 ΔNLL bias 4e-5, per-clip RMS 7e-4, r = 0.9999 — the effect is
~600× the fp16 noise on an n=1000 mean. Separately, fp16 kernel selection varies with **batch
shape** (~1e-3), which cancels in the paired difference only because every condition for a clip is
scored at the same batch index and size. That is why `BATCH` is pinned rather than discovered.

### Outputs

`MyDrive/NAACL/` again: `expc_results_full.csv` (carries hypotheses, gitignored),
`expc_per_utterance.csv` (git-safe; asserted numeric apart from `model`/`cond`/`timestamps` and
`path`/`speaker`/`region`), `expc_provenance.json`. Raw S/D/I counts and `n_ref_words` are kept per
row so corpus WER stays re-poolable from the git-safe file alone; section 14 is a standalone reload
that needs no GPU, no TIMIT and no earlier cells.

## Counting hallucinations (`Colab_Hallucinations.ipynb`)

WER cannot say whether a bad hypothesis is a hallucination: `WER = 1.4` is either forty words of
loop against a ten-word reference, or ten wrong words plus four spurious ones. This notebook applies
a text test to experiment A's stored hypotheses instead — **runaway** (`n_hyp >= 2 x n_ref`) or
**loop** (an n-gram, `n <= 5`, repeated >= 3x consecutively), reported separately and swept over both
thresholds, since the claim is the ordering across model size, not the level.

It **decodes nothing**. Inputs are `delta_results_full.csv` (hypotheses, Drive-only) joined to
`delta_per_utterance.csv` on `(model, path)`. Reading both is deliberate: taking `n_ref_words` from
the numbers-only file is what removes the need for a TIMIT copy, and it carries the four WERs across
so the text verdict can be cross-tabulated against the `WER > 1` proxy. It is the only notebook here
that consumes hypothesis text and emits no text, so `halluc_per_utterance.csv` needs no Drive-only
companion; it keeps `len_ratio` and `max_run` so the definition can be revisited without the text.

Three things that are easy to get wrong:

- **Normalize before tokenizing.** `EnglishTextNormalizer` strips punctuation, so
  `Thank you. Thank you. Thank you.` becomes a clean 3-repetition of a bigram rather than three
  distinct token sequences. It also keeps this notebook consistent with the WERs it compares against.
- **`max_repeat_run` steps its outer loop by 1 and its repeat count by `n`.** Advancing by one *and*
  counting overlapping matches scores `a b a b a b` as 4 repetitions rather than 3, because `[b a]`
  at position 1 also matches at position 3; advancing only by `n` misses a loop that starts off-phase.
  The function is unit-tested on exactly these cases.
- **Assert no `<|` in `text`.** `Tokenizer.decode` drops ids at or above `<|endoftext|>` so timestamp
  markers should never appear, but if they did they would inflate the length ratio *and* hand the
  repeat detector a spurious periodic structure.

Calibration against the proxy, measured locally on `base` from `offset_results.csv`: at 25 s with
timestamps on, the text test flags 23/1000 and `WER > 1` flags 27, overlapping on 22. So the
numbers-only file recovers ~96% of hallucinations and over-flags by ~5 — good enough for a scale
trend, not for a headline count.

## The taxonomy (`Colab_HallucinationTaxonomy.ipynb`)

The counting notebook's two tests are both tests of **form** — is it too long, does it repeat.
Neither asks whether the text is *about the audio*, so four failures pass untouched: canned
boilerplate (`thank you for watching` is `len_ratio` 0.44 with nothing repeated), partial
hallucination (9 correct + 6 invented is 1.67, under the 2.0 cut), drifting loops (exact n-gram
equality never matches), and empty output (`n_hyp = 0` scores as clean). This notebook replaces the
boolean with seven categories over three axes — degeneracy, length pathology, and **grounding**.

`hallucination = degenerate ∪ runaway ∪ untethered`. `errorful` exists to hold *ordinary* bad
transcription, which is what stops the hallucination count from being "high WER" under another name;
`empty` and `truncated` are separate because they are real positional failures in the other
direction. Categories are assigned in a **fixed priority order**, so a looping hypothesis that is
also over-length counts once, as `degenerate`.

Three measurements are new:

- **`compression_ratio`** is imported from `whisper.utils`, not reimplemented — it is the exact
  statistic `transcribe()` thresholds at `compression_ratio_threshold = 2.4`. zlib has a fixed
  header cost, so short strings compress *worse* than they should: a normal 11-word TIMIT
  hypothesis scores ~0.96, well under 2.4. It only becomes informative above roughly 60 words, which
  makes it a complement to `max_run` on long output, not a replacement on short output.
- **Fuzzy repeat matching** (`tol=1`) catches loops that drift as they run, and is gated to
  `n >= 3` — at `n = 1` a one-token tolerance makes every word match every other. `MAX_N` also rose
  5 → 10; a 7-gram loop repeated 3× previously scored `max_run = 1`. Exact and fuzzy runs are both
  stored.
- **`precision = hits / n_hyp`** comes from the same `jiwer` alignment that produces the WER —
  deliberately not a bag-of-words overlap against a hand-written stopword list, which would be an
  invented artifact with an indefensible threshold. `bag_overlap` is kept alongside as an
  order-insensitive robustness check only. Direction separates the categories: `untethered` is low
  precision *and* low recall; `runaway` is low precision with high recall, because the correct
  transcript is usually still in there under the invented text.

**It needs TIMIT**, unlike the counting notebook, because grounding cannot be computed without the
reference text. Only the `.TXT` sidecars though — no audio, no `soundfile`, no corpus verification
pass. Integrity comes instead from rebuilding the `reference_digest` that `Colab_DeltaSweep.ipynb`
pinned in its provenance, which is both faster than re-verifying 1000 audio arrays and stricter: it
proves these are the exact strings that produced the WERs being compared against.

**Caveat (found 2026-09-13): that check has never actually run.** The notebook looks for a key named
`reference_digest`, but the sweep writes `corpus.reference_digest_sha256`, so it prints a "cannot
verify the pin" notice, carries on, and records `reference_digest_verified: false`. Not yet fixed
there; `Colab_LLMVerdict.ipynb` reads the right key.

`old_halluc` reproduces the previous definition verbatim so the gain is measured rather than
asserted. Outputs are `halluc_taxonomy.csv` (numbers only, git-safe, keeps every raw measurement so
the taxonomy is re-cuttable without touching text again) and `halluc_label_sample.csv` — a
~200-row stratified sample **carrying reference and hypothesis text**, Drive-only and gitignored,
which is what turns the detector from a definition into a validated instrument. It stratifies by
`(condition, category)` pooled over models: the detector never sees the model, so per-model strata
would multiply the labelling cost by five without validating anything extra.

## The LLM arm (`Colab_LLMVerdict.ipynb`)

**The project's hallucination detector** — the author uses this notebook exclusively for detection;
the taxonomy notebook is at most an optional comparison. It runs Atwany et al.'s (ACL Findings 2025)
LLM classifier over the delta sweep's 20 000 hypotheses. **The judge is `gpt-5.6-luna`** (July 2026,
nano-tier), given their **Figure 6 fine-grained prompt** verbatim.

### What counts as hallucination

`HALLUCINATION_LABELS = {"Hallucination Error", "Oscillation Error"}` in §1 — fabrication **and
looping**. The two sources are easy to swap, so precisely: **Atwany et al. do not count loops** (they
file repetition under *Oscillation Error*, a non-hallucination class, and their HER counts
Hallucination Error only); **Jasiński et al. do** (HALAS labels "hallucinations containing looping,
i.e. repeated phrases"). The project follows Jasiński et al.

- The definition is applied to **stored labels at analysis time** and is deliberately not in `CONFIG`
  (it changes no verdict), so redefining never needs a new run. `verdict_per_utterance.csv` keeps the
  full five-way label beside the `halluc` flag.
- §8 prints each counted label's share per cell; its Hallucination Error column *is* Atwany et al.'s
  HER, so the paper-comparable and project numbers come from one run.
- **The coarse prompt (Figure 5) cannot express this definition** — it merges oscillation into
  Non-Hallucination Error at classification time, irrecoverably. §3 asserts
  `HALLUCINATION_LABELS ⊆ LABELS` before anything is built or sent; §7 re-checks every run it loads.
- The Figure 6 text was once **truncated** in this notebook (no examples, no format blocks, no
  sub-bullets) while described as verbatim. Both prompts are now checked against the PDF text by a
  whitespace-normalized diff; the only residual differences are one space each (`-Phonetic`,
  `Task:Classify`), consistent with PDF extraction. Re-run that diff after any edit to either prompt.

§2 verifies the rebuilt references against **`corpus.reference_digest_sha256`** in
`delta_provenance.json` — that exact key, which is what `Colab_DeltaSweep.ipynb` writes. An earlier
version scanned for a key named `reference_digest`, never found it, and failed on Colab with "reference
digest mismatch (or absent)" even when the references were correct; absent and mismatched now fail
with separate messages, and a mismatch prints both digest prefixes (digests, not text).

`halluc_taxonomy.csv` is **optional**. §7 derives speaker and region from the TIMIT path
(`DR4/FADG0/SX289.WAV`); agreement, the per-category table and the taxonomy series in the figure
appear only when the file is on Drive.

### Pay once, analyse forever: three phases

| phase | run | needs | API |
|---|---|---|---|
| submit | §1 → §5 | key, TIMIT `.TXT`, delta sweep files | §5, **the only billed call** |
| collect | §1 → §6 | key | polling + downloads, not billed |
| analyse | §1 → §7–§12 | Drive files only | **never** |

**§1 imports nothing that needs a key.** `connect()` in the helpers cell installs `openai` and reads
the key lazily, and only §5/§6 call it — so the analysis phase runs in a fresh runtime (or locally,
via `$NAACL_DATA` / `data/`) with no key, no `openai`, no TIMIT and no network. Do not move auth back
into §1 or make an analysis cell depend on `full`/`REF`/`HYP`/`client`; that is what the split exists
to prevent.

**The raw responses are the record.** §6 writes the Batch object, `output_*.jsonl` and
`errors_*.jsonl` to Drive byte-for-byte *before* anything parses them, reads each back, and records
its SHA-256 in `run.json`; `parse_run()` re-checks those digests on every read. Everything from §7 on
re-derives from those files. This is not optional hygiene: **OpenAI deletes a batch's output file
30 days after completion**, and GPT-5 models have no greedy decoding, so a re-run would neither be
free nor reproduce the verdicts.

### Run folders

`MyDrive/NAACL/llm_runs/<judge>.<grain>.effort-<level>/` (`.greedy` for non-reasoning judges) — one
folder per configuration that can change a verdict, so turning the knob starts a new run beside the
old one instead of overwriting it. Contents: `run.json` (config incl. full prompt text, input
digests, licensing decision, every batch with its rows and file digests), `index.csv`
(`custom_id` → model/path/offset/arm, which is what frees collection and analysis from TIMIT),
`requests_*.jsonl` (exact uploaded bytes — **carry reference text**), `batch_*.json`, `output_*`,
`errors_*`, and the derived `verdict_per_utterance.csv` + `provenance.json` written by §10.
`llm_runs/` is gitignored because of the request files.

§5 **refuses to add batches to a folder whose recorded `CONFIG` or input digests differ** —
`CONFIG` covers judge, grain/labels, effort, temperature, prompt digest, user template, schema and
endpoint. `max_tokens` is recorded per batch instead, so a truncation retry at a larger budget is
allowed.

`ANALYSE` in §7 lists run folders to analyse together; HER, agreement pairs, the per-category table
and the figure all extend to every listed run, so two effort levels compare with no API call. It
defaults to the delta-sweep run (`RUN_DELTA`) whatever `SOURCE` is set to.

### Never paying twice

- **§5 submits only rows without an `ok` verdict that are not in flight.** After a partial failure it
  sends just those rows as a new batch. §7 stops with the count and this remedy rather than analysing
  an incomplete grid.
- **Unrecorded batches are adopted, not resubmitted.** Each batch carries `metadata={"run", "k",
  "rows"}` (rows = 32-hex digest of its row list). If the runtime dies between `batches.create` and
  the Drive write of `run.json` — including a write stuck in the FUSE buffer — the next §5 finds the
  tagged batch via `batches.list` and records it. Its rows are read back from its own
  `requests_<k>.jsonl` on Drive (written before the upload) and checked against the tag, so adoption
  does not depend on this session's token packing choosing the same rows. Metadata limits: 16 pairs,
  keys ≤ 64 chars, values ≤ 512.
- Row status is one of `ok`, `truncated` (empty content — reasoning spent the budget), `refusal`
  (structured outputs put this in `message.refusal`, not `content`), `invalid`, `error`, `missing`.
  When a row appears in several batches, `ok` beats a failure and later beats earlier; `billed` usage
  sums every response, retried ones included.

### The knob, and deviations from the paper

**`REASONING_EFFORT` in §1** — `none`, `low`, `medium` (API default), `high`, `xhigh`, `max`.
`none` ships because it matches the paper (their §A.4.1 avoids chain-of-thought "that could
introduce errors") and because reasoning tokens bill at the **output** rate on a twelve-token
answer. **`MAX_TOKENS` is derived from it** (64 at `none`, 8192 otherwise): `max_completion_tokens`
caps reasoning *and* the answer, so a label-sized ceiling starves any other level; unused headroom is
never billed. §7 *warns* (does not assert — the run is already paid for) when `effort="none"` still
shows ≥ 5 reasoning tokens per row.

Deviations, recorded in each `provenance.json`: strict `json_schema` `response_format` replaces the
prompt's "produce only the classification"; no greedy decoding (GPT-5 family rejects
`temperature`); and `reasoning_effort` itself. `JUDGE = "gpt-4o-mini"` gives a separate greedy run.

**Request shape is family-dependent** and mixing the shapes is a **400**: reasoning models reject
`temperature` and require `max_completion_tokens`; older chat models want `max_tokens`. `body_for`
dispatches on a `gpt-5`/`gpt-6`/`o1`/`o3`/`o4` prefix.

### Reading the results

§8 (rate + label composition) is the headline. §9 does work only with a second source — another run
in `ANALYSE`, or the taxonomy file — and then reports raw agreement, Cohen's kappa and positive-class
Jaccard (raw agreement alone is dominated by rows both call clean). Against the taxonomy, Atwany et
al.'s human–heuristic agreement of **0.00** is the reference point. With loops counted on both sides,
the taxonomy's `degenerate` rows should mostly agree; disagreement there means the judge read a loop
as something else (typically Language Error).

§11 writes `llm_halluc_rate.pdf/.png` in the `Figures.ipynb` house style and verifies the PDF by
inflating its streams. The style lives in one **figure style** cell under §1, shared by §11 and §13,
with a `finish(fig, stem)` that saves, verifies and downloads. To redraw without the API: §1, its
helper and style cells, then §7 → §8 → §11 (or §13's two cells).

### Hallucination rate against offset (§13)

The delta sweep decoded only 5 s and 25 s, so a 5–25 s curve needs the scaling sweep's hypotheses
(`scaling_results_1000.csv`, from `Colab_ModelScaling copy.ipynb`: same 1000 clips, five checkpoints,
greedy fp16 batch 8). `SOURCE = "scaling"` in §1 makes §2–§6 classify those, into a separate run
folder `<RUN_DELTA>.offsets`. **Every point of the curve describes the scaling sweep's hypotheses.**

- **Reuse is per row and by judge input.** A scaling-sweep 5 s / 25 s row whose *normalized*
  hypothesis equals the delta sweep's would send the judge the same bytes, so it takes the delta run's
  verdict and is not sent; rows that differ are classified. KEYS therefore depends only on the input
  files, never on the delta run's state, which keeps `index.csv` stable across sessions. `reused.csv`
  records the reused keys with a digest of each shared judge input. §2 prints the identical-row counts
  per condition — that table is the free check of whether the two sweeps decoded alike.
- §5 refuses, **before writing or sending anything**, to start an offset run whose delta run is
  missing or was classified under a different `CONFIG`. §13 re-checks config equality before
  stitching, and prints the model snapshot that served each run — the two runs happen at different
  times, and a snapshot change would sit exactly on the 5/10 s and 20/25 s seams.
- `REUSE_DELTA_VERDICTS = False` classifies all five offsets in one run (`.offsets-all`) instead: no
  seam, and the 5 s / 25 s rows are paid for again.
- §13's first cell writes `verdict_by_offset.csv` into the offset run folder (git-safe: identifiers,
  label, `halluc`, `from_run`, `response_model`) and prints the rate table; the second draws
  `llm_halluc_by_offset.pdf/.png` from that file alone. Both need only §1 and its helper/style cells.
- The analysis sections §7–§12 still work on the 5 s / 25 s grid only; an offset run never appears in
  `ANALYSE`, and §12's standalone reload ignores it (it has no `verdict_per_utterance.csv`).

### Licensing

**The one notebook that sends reference text to a third party.** The full-grid run transmits all
1000 TIMIT references to the OpenAI API — a deliberate choice, recorded under `licensing` in
`run.json` and carried into `provenance.json`. §5 is gated behind `CONFIRM = True`. §6's
`DELETE_REMOTE = True` removes the run's input/output files from OpenAI once the Drive copies verify.
Nothing prints text: output and errors are counted, and batch rejections print codes plus, for
`token_limit_exceeded`, the limit number parsed out of the message — never the message itself.

### Cost

$0.20/1M input, $1.20/1M output, halved by the Batch API. The Figure 6 instruction block is
**705 tokens** (`o200k_base`; Figure 5 is 654); the user turn measured over the 12 000 locally
available `base` rows averages **28 tokens** (p99 43); chat framing and schema add ~30 (estimated).
§4 measures the user turn over every row rather than one example.

| reasoning tokens / row | batch-discounted total (fine prompt) |
|---|---|
| 0 (`effort="none"`) | **$1.67** |
| 50 | $2.27 |
| 200 | $4.07 |
| 500 | $7.67 |
| 1000 | $13.67 |

It is a sweep, not a prediction — there is no measured reasoning figure for this prompt above `none`.
§7 prints observed reasoning tokens and the realised cost. Input is bounded: every hypothesis at
Whisper's 224-token ceiling would add roughly $0.20. The prompt is below OpenAI's **1024-token**
automatic-caching bar and deliberately not padded past it. Their App. A.4.1 rate card quotes the
*cached* input rate. **A subscription does not cover API usage** (Claude.ai, ChatGPT, or Claude Code
with `ANTHROPIC_API_KEY` set all bill API calls separately).

### Batch API specifics that are easy to get wrong

- It takes an uploaded JSONL file (`purpose="batch"`), one call per line with `custom_id`, `method`,
  `url`, `body`; ceilings 50 000 requests / 200 MB. Measured: 10 000 rows = 42 MB.
- **The binding limit is the batch *queue*, not the batch.** OpenAI caps input tokens *enqueued across
  all in-flight batches* per model per organization (limits page: "Batch queue limit"). Exceeding it
  fails the batch at validation with `token_limit_exceeded` — no output, no error file, not billed —
  and the batch still reaches a terminal state, so "every batch collected" does not mean "every row
  classified". The first live submission hit exactly this: two 10 000-row batches (~7.6 M tokens each)
  sent back to back, both rejected, and the old §6 reported success. §5 now packs by §4's per-row token
  counts to `QUEUE_FILL` (0.8) of the limit minus what is in flight, and with `WAIT_FOR_QUEUE` keeps
  submitting as batches finish. The limit is the lower of `QUEUE_LIMIT` and the number OpenAI stated
  in an earlier rejection of this run (parsed from the stored `batch_<k>.json`), else
  `QUEUE_FALLBACK`. A batch still rejected after packing stops §5 rather than retrying into it.
- **Failed requests are not in the output file** — they go to `error_file_id`.
- **Results are not in submission order** — key on `custom_id` (`r<i>`, `i` indexes `index.csv`).
- **Expired/cancelled batches still return completed results** (and bill them); download them and let
  §5 resubmit the rest.

The notebook was first built against a **mocked** OpenAI client with synthetic text (crash-and-adopt,
partial failure and retry, config drift, coarse grain refused, analysis with and without the taxonomy
file while network is blocked and `openai` is unimportable, no text in printed output or git-safe
files). That mock had no queue limit, which is why the first live run failed on one. The queue
handling was then tested against a mock that enforces an enqueued-token limit and counts 8% more
tokens than tiktoken: replaying the failed live run with the original cells reproduced its output
line for line, and the fixed cells recovered it, stopped after a single bounce when the limit was set
too high, adopted a batch orphaned mid-submit without sending any row twice, and completed by hand
with `WAIT_FOR_QUEUE = False`. **The rejection message format (`Limit: N enqueued tokens`) is assumed,
not observed** — if it differs, no number is parsed and §5 falls back to `QUEUE_FALLBACK`.

## Publication figures

Every experiment notebook writes PNG at 160 dpi, sized for a notebook cell — the wrong artefact for
a paper. `Figures.ipynb` redraws them as **vector PDF** at ACL column widths (3.15 in single,
6.30 in full) from the numbers-only files each sweep leaves on Drive, so nothing is decoded again:

| figure (`figures/<stem>.pdf`) | source file |
|---|---|
| `wer_vs_offset` | `scaling_per_condition.csv` |
| `delta_by_model` | `delta_per_utterance.csv` (+ `delta_provenance.json` if present) |
| `encoder_cka` | `encsim_per_utterance.csv` |
| `cka_by_depth` | `encsim_per_layer.csv` |
| `localization` | `expc_per_utterance.csv` |
| `positional_embedding` | `pe_per_condition.csv` |

It also emits the two `booktabs` tables.

**Sections 1 and 2 are the only shared state; every figure section is independent.** §1 is data and
mechanics, §2 is every visual knob. A section is `rows = need("x.csv")` … `finish(fig, "stem")`, so
a new figure is added without touching or re-running any other section. `need()` returns `None` and
prints a notice for a sweep that has not run — it never raises, so the notebook stays runnable
before the grid is complete. `load()` caches; call `reload()` after pulling fresh CSVs off Drive.
`cached_ci()` memoizes the clustered BCa bootstrap, which is ~10 s of pure Python per model and
would otherwise be re-paid on every style tweak.

`finish()` writes the PDF and PNG **and verifies the PDF in the same call**, so one section run on
its own still proves its own output. Section 10 globs `figures/` rather than tracking writes in a
global, for the same reason. The verification inflates the Flate streams first: matplotlib
compresses its object streams, so a byte-level grep for `/Type3` or `/FontFile2` passes on every
file and checks nothing.

Locally the notebook reads `$NAACL_DATA`, else `data/` if it exists, else the working directory —
`data/` is gitignored, so downloaded sweep outputs cannot be committed by accident.

### House style (stated 2026-09-10; applies to figures added later)

1. **No checkpoint parameter counts.** Checkpoints are named (`tiny` … `large-v3`), never sized —
   no `39M` under a tick, no parameter-count axis.
2. **No titles**, on the figure or the axes. Where two panels must be told apart they carry an
   inset `(a)` / `(b)` label via `panel()`; the caption carries everything else.
3. **Vector PDF**, verified in `finish()`.
4. **Axis labels of at most three words** — `offset (s)`, `corpus WER`, `relative depth`, `CKA`.
5. Boxed axes (all four spines), a light grid, frameless legends, STIX serif to sit with the ACL
   template's Times, and one colour *and* dash pattern per model so the ladder survives greyscale
   printing. `series()` applies the pair.

`pdf.fonttype = 42` is not optional — matplotlib's default Type 3 embedding is rejected by arXiv's
checker and by some venues. Set the figure to its final print width here rather than rescaling with
`\includegraphics`, which is what makes axis labels unreadable.

Note the scaling sweep originally had **no** numbers-only output at all — its WER existed only in
printed cell output — so `Colab_ModelScaling copy.ipynb` now writes `scaling_per_condition.csv`
with pooled S/D/I per (model, offset, arm), asserted to reproduce `corpus_wer` from the counts.

The delta figure re-derives the speaker-clustered BCa interval and asserts it against
`delta_provenance.json`. That check cannot be written as an equality: the sweep bootstrapped
**unrounded** floats, while `delta_per_utterance.csv` stores `delta_m` at `:.6f`, so recomputing
from the CSV lands a few times `1e-9` away. The tolerance is one unit in the CSV's last stored
place (`1e-6`), which still fires on the mistake the check exists to catch — reading `mean_ci_lo`,
the utterance-level interval, sits ~3e-3 away. An earlier `1e-9` tolerance failed the whole cell on
a run where the numbers were in fact correct.

`encoder_cka` deliberately omits the silence **position floor**. It is the control for the raw
cosine, which is dominated by "this is a different window position"; CKA is not, and drawing the
floor on a CKA axis invites it to be read as a baseline for a metric it is not a baseline for.

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
