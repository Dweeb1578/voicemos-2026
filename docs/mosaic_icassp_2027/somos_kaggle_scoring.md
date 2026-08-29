# SOMOS v2 frozen-bank Kaggle scoring

Status: build-ready, post-release exploratory scoring plan. The prospective
protocol remains frozen at SHA-256
`81daeb5dbfcac387ea9bad14dffe0603715999524028a2902757fc6aa1c241d9` and is
not modified by this orchestration.

## What is generated

`scripts/somos_kaggle_orchestrate.py` creates 40 private Kaggle kernel folders:
one of the ten frozen public runners and one deterministic SHA-256 ID shard per
kernel. Four shards per runner keep every kernel below the 12-hour ceiling by
stopping at 660 minutes. DNSMOS and P.808 are CPU kernels. The other eight
runners request a T4 GPU.

Every generated kernel embeds the local scorer and frozen protocol bytes,
records their SHA-256 values in `orchestration.lock.json`, clones external
repositories at the frozen revisions, downloads Hugging Face artifacts at their
frozen revisions, and hashes declared artifacts before it accepts a new score
row. Each final shard has a cache, initialization record, environment snapshot,
score CSV, and provenance JSON.

No generated code calls the Kaggle CLI or launches a kernel.

## Input firewall

The scorer accepts only a private, audio-only Kaggle dataset in this layout:

```text
somos_audio_manifest.csv
audio/
  TRAINSET/ or train/
  VALIDSET/ or valid/
  TESTSET/ or test/
```

`sample_id` is the exact WAV filename, including `.wav`. It is the sole score
join key and matches the `utt_id` values in the frozen label manifest exactly.
The scorer rejects mounted `*_mos_list.txt` files and manifests with target-like
columns. The audio-only manifest itself is produced by scanning only the WAV
filenames after the separate retrieval pipeline has materialized the three
official splits. Do not include the clean directory or any MOS list when
creating the scorer input dataset.

DNSMOS and P.808 fetch their public ONNX files during scoring from the declared
source URLs, then hard-fail unless their SHA-256 values match the two frozen
artifact hashes. The source URL is a retrieval location, not a version pin. The
accepted artifact digest is the immutable pin and is retained in each shard's
provenance.

## Build and score later

After the audio-only and DNSMOS model datasets have been created manually, build
the local kernel directories. This command does not need Kaggle credentials and
does not execute inference:

```powershell
py -3.14 -m scripts.somos_kaggle_orchestrate --username <account> --audio-kernel <account>/somos-v2-audio-only-ingestion --build
```

At the time of writing, Kaggle CLI authentication is absent, so do not push or
launch these directories yet. The `--audio-kernel` source must be Luna's private
audio-only ingestion-kernel output, not a dataset containing release labels.
Once authentication and a GPU budget are available, upload the generated
directories unchanged, preserving their `kernel-metadata.json` and
`orchestration.lock.json` files.

Do not start full scoring immediately. VoiceMOS SCOREQ was close to 11 hours on
a much smaller pass. The current one-path-at-a-time adapter may therefore
exceed Kaggle's roughly 30-hour GPU quota even though each individual shard is
bounded below 12 hours. First build a non-final, fixed-size rate smoke with one
shard per runner:

```powershell
py -3.14 -m scripts.somos_kaggle_orchestrate --username <account> --audio-kernel <account>/somos-v2-audio-only-ingestion --shard-count 1 --smoke-items 25 --build --out notebooks/somos-kaggle-smoke
```

The smoke mode hashes initialized artifacts and writes only a resumable cache
plus `*.smoke.provenance.json`, never a final score CSV. Record rows per second
and projected wall time for every runner. Launch the four-shard full pass only
after the projections fit the available quota, or record the primary bank as
computationally infeasible.

For a failed or pre-empted kernel, use its private prediction-only Kaggle output
as the optional `--resume-kernel` source while rebuilding the retry directory.
The notebook accepts at most one matching cache named
`<runner>-partXX-of-04.cache.csv`; it resumes missing rows and rejects ambiguous
caches. Caches are flushed and synced every 50 rows, plus at normal completion.
A crash can at most repeat the uncheckpointed tail.

## Merge after all shards complete

Download all four score shards and their adjacent provenance files per runner.
Then merge into the canonical `scores/<runner>.csv` schema without mounting or
opening targets:

```powershell
py -3.14 -m scripts.somos_merge_shards --runner dnsmos --audio-root <audio-only-dataset>/audio --audio-manifest <audio-only-dataset>/somos_audio_manifest.csv --shard-root <downloaded-shards> --out-dir scores
```

The merge validates all four shard indices, protocol hashes, score-file hashes,
exact `sample_id` coverage, metadata agreement, finite values, and
non-constant outputs within each split. The separate frozen MOS analysis may
join only these canonical score files to the frozen target manifest after the
full 27-output score matrix is complete.

## Known untested runtime risks

- No real SOMOS audio, target file, or GPU job has been opened or run by this
  orchestration work. All inference adapters require a Kaggle smoke shard.
- Kaggle package resolution may make `torchaudio==2.11.0` incompatible with its
  preinstalled Torch, or an upstream repository may have dependency conflicts.
  Each kernel pins the requested version and captures `pip freeze`, but a failed
  dependency must be corrected and retained as a failed run before scoring is
  accepted.
- NISQA, UTMOSv2, Audiobox Aesthetics, and Uni-VERSA use upstream loading APIs
  that are not exercised in the offline tests. Their file names, revision IDs,
  and output schemas must be verified on a one-shard run before the remaining
  shards are started.
- Repository commit pinning and Hugging Face revision pinning protect source
  selection, not hidden upstream training-data overlap with SOMOS. That caveat
  remains part of the eventual paper's claim boundary.
