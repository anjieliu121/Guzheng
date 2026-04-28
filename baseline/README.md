# From-scratch baseline

A hierarchical patch+character GPT trained **from scratch** on guzheng ABC, with
the same architecture as NotaGen (~231M params at `medium`). Comparing this to
fine-tuned NotaGen isolates the contribution of NotaGen's pre-training on ~1M
Western scores — if this baseline performs significantly worse, pre-training is
what's doing the work, not the architecture.

## Files

| File | Purpose |
|------|---------|
| [`train.py`](train.py) | Train the baseline from scratch on `../data/dataset.jsonl`. |
| [`generate.py`](generate.py) | Sample ABC + render to MIDI from a trained checkpoint. |

## Setup

```bash
pip install -r ../requirements.txt
brew install abcmidi             # provides `abc2midi` for ABC -> MIDI rendering
```

## Configs

| Preset | Params | d_model | Patch layers | Char layers | d_ff |
|--------|-------:|--------:|-------------:|------------:|-----:|
| `medium` (default) | ~231M | 1024 | 16 | 3 | 4096 |
| `small` | ~50M | 512 | 8 | 2 | 2048 |
| `tiny` | ~10M | 256 | 4 | 2 | 1024 |

`medium` matches NotaGen-medium so the comparison is apples-to-apples.

## Train

```bash
python train.py                  # medium, 200 epochs
python train.py --config small   # smaller config
python train.py --smoke          # 2-epoch sanity check
```

Checkpoints save every epoch to `checkpoints/baseline_<config>_latest.pth`,
with the best at `_best.pth`. Re-running resumes from `_latest`.

The dataloader applies the same NotaGen-style triangular ±3-semitone key
augmentation and oversamples repertoire 5× via the per-row `weight` field
in `dataset.jsonl`.

## Generate

```bash
python generate.py --n 10                # 10 samples from best checkpoint
python generate.py --n 5 --temperature 1.0
```

Outputs `.abc` and `.mid` under `generated/<timestamp>/`.

## Hosted artifacts

The medium checkpoint (~900 MB after stripping optimizer state) is hosted on
Hugging Face Hub:
**[huggingface.co/anjieliu/baseline-guzheng](https://huggingface.co/anjieliu/baseline-guzheng)**.

```bash
huggingface-cli download anjieliu/baseline-guzheng --local-dir checkpoints/
```
