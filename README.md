# Guzheng music generation

Code, data, and models from a thesis on generating traditional Chinese guzheng
music with hierarchical patch-character language models. Compares a fine-tune of
[NotaGen-medium](https://huggingface.co/ElectricAlexis/NotaGen) (pre-trained on
~1M Western scores) against an identical-architecture model trained from scratch
on the same guzheng corpus, isolating the contribution of pre-training.

| Model | Eval loss | URL |
|-------|----------:|-----|
| NotaGen-medium (fine-tuned) | **0.47** | [anjieliu/notagen-guzheng](https://huggingface.co/anjieliu/notagen-guzheng) |
| Baseline (from-scratch) | 1.04 | [anjieliu/baseline-guzheng](https://huggingface.co/anjieliu/baseline-guzheng) |

The 2.2× eval-loss gap between the two — same architecture, same data, same
training recipe — is attributable to NotaGen's pre-training on Western scores.

## Listen

10 paired samples per model (same seeds, deterministic): [`samples/`](samples).

## Layout

```
guzheng/
├── data/
│   └── dataset.jsonl           # Bundled training data: 125 pieces × 15 keys
├── raw_data/
│   ├── MIDI/                   # 26 hand-curated guzheng repertoire pieces
│   └── guzheng_tech99/         # 99 pieces from the Guzheng-Tech99 dataset
├── notagen/                    # Fine-tuned NotaGen (the headline model)
│   ├── train.py    generate.py
│   ├── README.md
│   └── NotaGen/                # Vendored upstream code (MIT-licensed)
├── baseline/                   # From-scratch baseline (same architecture, no pre-training)
│   ├── train.py    generate.py
│   └── README.md
├── samples/                    # 20 representative MIDI generations
├── evaluation/
│   └── musicality_metrics.py   # OA, compression ratio, structureness, groove consistency
├── metadata/
│   ├── MIDI.json
│   └── guzheng_scales.json
├── LICENSE                     # MIT (code) + CC-BY-SA 4.0 (data and weights)
├── requirements.txt
└── README.md (this file)
```

## Setup

```bash
git clone https://github.com/anjieliu/guzheng
cd guzheng

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# System dependency for ABC -> MIDI rendering during generation
brew install abcmidi          # macOS
# apt-get install abcmidi     # Ubuntu
```

## Reproduce

### Fine-tune NotaGen

```bash
# Download NotaGen-medium pretrained weights from upstream
cd notagen/checkpoints/
huggingface-cli download ElectricAlexis/NotaGen notagen_medium_pretrain.pth --local-dir .
cd ../..

# Train (~30 epochs, several hours on Apple MPS)
python notagen/train.py --size medium --epochs 30
```

### Train baseline from scratch

```bash
python baseline/train.py --config medium --epochs 200
```

### Generate from published checkpoints

Skip training entirely and use the published weights:

```bash
huggingface-cli download anjieliu/notagen-guzheng --local-dir notagen/checkpoints/
python notagen/generate.py --size medium --num 10

huggingface-cli download anjieliu/baseline-guzheng --local-dir baseline/checkpoints/
python baseline/generate.py --config medium --n 10
```

Outputs land in `notagen/generated/<timestamp>/` and `baseline/generated/<timestamp>/`
as `.abc` + `.mid` files.

### Evaluate

```bash
python evaluation/musicality_metrics.py \
    "training_data=data/" \
    "notagen=notagen/generated/<timestamp>" \
    "baseline=baseline/generated/<timestamp>"
```

Computes compression ratio, structureness indicator, 2nd-order pitch transition
entropy, and groove consistency across each batch and prints a comparison table.

## Dataset

`data/dataset.jsonl` contains 125 pieces, each pre-augmented to all 15 keys via
[abctoolkit](https://github.com/sander-wood/abctoolkit) transposition. One JSON
line per piece with all 15 keys inlined as ABC text:

```json
{
  "name": "cai_yun_zhui_yue",
  "original_key": "C",
  "source": "repertoire",
  "split": "train",
  "weight": 5.0,
  "abc": {"A": "L:1/32\nM:4/4\n...", "B": "...", ...}
}
```

`source: repertoire` (26 hand-curated) is oversampled 5× during training relative
to `source: tech99` (99 scraped) — see `weight` field. The training script
streams from this file directly; no separate per-key directory tree needed.

## License and citation

Code is MIT licensed. Data and trained weights are CC-BY-SA 4.0.
See [LICENSE](LICENSE) for the full breakdown including upstream attributions.

If you use this work, please cite the [NotaGen paper](https://arxiv.org/abs/2502.18008)
and the thesis (citation TBD).
