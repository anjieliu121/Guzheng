# Sample outputs

Representative MIDI generations from each model. 10 samples per model, drawn
from the same fixed seeds (1000–1009) so the two sets are paired for direct
A/B comparison: `notagen/sample_01.mid` and `baseline/sample_01.mid` started
from identical conditions.

## Layout

| Path | Source |
|------|--------|
| [`notagen/`](notagen) | Fine-tuned NotaGen-medium (`notagen_guzheng_medium_best.pth`) — the headline thesis model |
| [`baseline/`](baseline) | From-scratch baseline-medium — same architecture as NotaGen, no pre-training |

## How they were produced

1. **Generate** — `notagen/generate.py` and `baseline/generate.py` were called
   with `--num 50`, `temperature 1.0` for NotaGen and `temperature 1.2` for
   the baseline, prompted with the same metadata (`L:1/32`, `M:4/4`, `K:C`, single
   guzheng voice). Output: ABC text rendered to MIDI via `abc2midi`.

2. **Post-process** (not part of the public repo):
   - **D-center** — transposes the C-prompted output to D pentatonic (the
     canonical guzheng tuning) and quantizes to a 16th-note grid.
   - **Keyswitches** — emits MIDI keyswitch events for sample-library articulation
     (vibrato, glissando, etc.) so the file plays correctly through Garritan
     World Instruments or similar.

   Both steps were applied via internal tooling and are described in the thesis;
   the public repo only ships the unprocessed `generate.py` outputs path.

3. **Curate** — first 10 of 50 samples (deterministic seeds), no rejection.

## Reproducing without post-processing

The raw output of `python notagen/generate.py --num 10` (or `baseline/generate.py`)
is in C major with no keyswitches. It will play back as a clean melodic line on
any standard MIDI synth, but won't trigger guzheng-specific articulations. The
samples here let you hear the polished thesis result without needing the
post-processing toolchain.
