# Fine-tuned NotaGen for Guzheng

Fine-tunes [NotaGen](https://github.com/ElectricAlexis/NotaGen) — a hierarchical
patch-character GPT pre-trained on ~1M Western scores in interleaved ABC notation
— on guzheng repertoire and the [Guzheng Tech99](https://ccmusic-database.github.io)
dataset.

## Files

| File | Purpose |
|------|---------|
| [`train.py`](train.py) | Fine-tune small/medium/large NotaGen on `../data/dataset.jsonl`. Supports MPS/CUDA/CPU. |
| [`generate.py`](generate.py) | Sample ABC + render to MIDI from a fine-tuned checkpoint. |
| [`NotaGen/`](NotaGen) | Vendored upstream NotaGen code (LICENSE preserved). Required by `train.py` and `generate.py` for the model class and tokenizer. |

## Setup

```bash
pip install -r ../requirements.txt
pip install -e NotaGen           # NotaGen + abctoolkit (its peer dependency)
brew install abcmidi             # provides `abc2midi` for ABC -> MIDI rendering
```

Then download the NotaGen pretrained checkpoint into `checkpoints/`:

| Size | File | Source |
|------|------|--------|
| small | `notagen_small_pretrain.pth` | [HuggingFace](https://huggingface.co/ElectricAlexis/NotaGen) |
| medium | `notagen_medium_pretrain.pth` | same |
| large | `notagen_large_pretrain.pth` | same |

## Train

```bash
python train.py --size small --epochs 30
```

Checkpoints are saved every epoch to `checkpoints/notagen_guzheng_<size>_latest.pth`,
with the best one (by eval loss) at `notagen_guzheng_<size>_best.pth`. Re-running
the same command resumes from `_latest`.

`--smoke` runs 2 epochs on the first 20 train pieces to sanity-check the pipeline.

The dataloader applies NotaGen's triangular ±3-semitone key augmentation and
oversamples repertoire pieces 5× relative to Tech99 via `WeightedRandomSampler`
(weights are read from the per-row `weight` field in `dataset.jsonl`).

## Generate

```bash
python generate.py --size small --num 10 --temperature 0.9
```

Outputs `.abc` and `.mid` files under `generated/<timestamp>/`. The `--ckpt`
flag overrides the default best-checkpoint path if you want to compare epochs.

## Hosted artifacts

The thesis's headline fine-tuned checkpoint is **NotaGen-medium**
(`notagen_guzheng_medium_best.pth`, ~900 MB after stripping optimizer state),
trained with 5× repertoire oversampling (see `train.py` for details). Too large
for Git, hosted on Hugging Face Hub:
**[huggingface.co/anjieliu/notagen-guzheng](https://huggingface.co/anjieliu/notagen-guzheng)**.

```bash
huggingface-cli download anjieliu/notagen-guzheng --local-dir checkpoints/
```

Smaller `small` configurations are supported by `train.py` for local
experimentation but are not published — train your own from the upstream
pretrained weights if you need them.
