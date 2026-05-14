## TO-DO
### D
* bai_jia_chun.mid (闽南)
### G
* chang_an_ba_jing.mid 
* deng_huo_jiao_hui.mid
* die_lian_hua.mid
* chen_xing_yuan_luo_yuan.mid
* chen_xing_yuan_he_fan.mid
### F
* chun_jian_liu_chuan.mid
### C
### A
* chun_dao_xiang_jiang.mid
* cao_yuan_ying_xiong_xiao_jie_mei.mid

## Validate MIDI note quality (too short, too long, overlap, off-grid)
```bash
# check one file
python scripts/check_midi_note_quality.py raw_data/MIDI/chun_miao.mid
# apply
python scripts/check_midi_note_quality.py raw_data/MIDI/chun_miao.mid --apply
# whole directory
python scripts/check_midi_note_quality.py MIDI
```

## Add MIDI metadata to `index.json`
```bash
python scripts/add_midi_metadata.py "MIDI/bu_bu_gao.mid" \
  --title-en "Every Step is Higher" \
  --title-zh "步步高" \
  --source-sheet-url "https://m.guzheng.cn/qupu/2186.html" \
  --source "unknown, modified by the author" \
  --notes "changed 3-note mock vibrato to just the note" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/gao_shan_liu_shui.mid" \
  --title-en "igh Mountains and Flowing Water" \
  --title-zh "高山流水" \
  --source-sheet-url "https://m.guzheng.cn/qupu/53.html" \
  --source "created by the author based on (Cheng, 2022)" \
  --notes "no glissandos" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/han_tian_lei.mid" \
  --title-en "Thunder in Drought" \
  --title-zh "旱天雷" \
  --source-sheet-url "https://m.guzheng.cn/qupu/493.html" \
  --source "created by the author" \
  --notes "varying tempo" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/qian_sheng_fo.mid" \
  --title-en "Thousand Buddha Chants" \
  --title-zh "千声佛" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1268.html" \
  --source "created by the author" \
  --notes "each note starts and ends exactly on-grid" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/yu_zhou_chang_wan.mid" \
  --title-en "Fisherman's Song at Dusk" \
  --title-zh "渔舟唱晚" \
  --source-sheet-url "https://m.guzheng.cn/qupu/203.html" \
  --source "created by the author based on (Cheng, 2022)" \
  --notes "ocr doesn't work" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/shang_lou.mid" \
  --title-en "Going Upstairs" \
  --title-zh "上楼" \
  --source-sheet-url "https://m.guzheng.cn/qupu/921.html" \
  --source "created by the author based on (Cheng, 2022)" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/nan_zheng_gong.mid" \
  --title-en "Nan Zheng Palace" \
  --title-zh "南正宫" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1981.html" \
  --source "created by the author based on (Cheng, 2022)" \
  --notes "different but better glissando" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/zai_bei_jing_de_jin_shan_shang.mid" \
  --title-en "Over the Golden Hill of Beijing" \
  --title-zh "在北京的金山上" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1091.html" \
  --source "(Cheng, 2022), bpm dropped from 100 to 88" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/ya_shan_ai.mid" \
  --title-en "Mourning in Ya Shan" \
  --title-zh "崖山哀" \
  --title-alt "靠山" \
  --title-alt "哭山" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1526.html" \
  --source "created by the author based on (Cheng, 2022)" \
  --notes "some bent notes were recorded while some decorative (subjectively determined) bent notes were disregarded" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/cai_yun_zhui_yue.mid" \
  --title-en "Colorful Clouds Chasing the Moon" \
  --title-zh "彩云追月" \
  --source-sheet-url "https://m.guzheng.cn/qupu/5411.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "modified directly in Sibelius" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/zhan_tai_feng.mid" \
  --title-en "Battling the Typhoon" \
  --title-zh "战台风" \
  --source-sheet-url "https://m.guzheng.cn/qupu/964.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "modified directly in Sibelius, retouched glissando in MIDI" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/chun_dao_la_sa.mid" \
  --title-en "Spring Has Arrived in Lhasa" \
  --title-zh "春到拉萨" \
  --source-sheet-url "https://m.guzheng.cn/qupu/966.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "modified directly in Sibelius, retouched glissando and corrected notes in MIDI" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/dan_dian_tou_luan_cha_hua.mid" \
  --title-en "To Arrange Flowers in Disorder" \
  --title-zh "单点头乱插花" \
  --title-alt "丹凤点头" \
  --title-alt "乱插花" \
  --title-alt "单点头" \
  --source-sheet-url "https://m.guzheng.cn/qupu/25.html" \
  --source "created by the author" \
  --notes "basic sheet in Sibelius, corrected notes in MIDI" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/chun_miao.mid" \
  --title-en "Spring Sprouts" \
  --title-zh "春苗" \
  --source-sheet-url "https://m.guzheng.cn/qupu/520.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "modified in Sibelius, retouched glissando in MIDI" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/chu_shui_lian.mid" \
  --title-en "Lotus Flowers Emerging from Water" \
  --title-zh "出水莲" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1003.html" \
  --source "created by the author" \
  --notes "basic sheet in Sibelius, corrected notes in MIDI" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/ba_yue_gui_hua_bian_di_kai.mid" \
  --title-en "Osmanthus Flowers Blooming Everywhere in August" \
  --title-zh "八月桂花遍地开" \
  --source-sheet-url "https://m.guzheng.cn/qupu/519.html" \
  --source "created by the author" \
  --notes "basic sheet in Sibelius, adjusted overlapping notes in MIDI" \
  --overwrite

python scripts/add_midi_metadata.py "MIDI/da_yan.mid" \
  --title-en "Shooting the Wild Goose" \
  --title-zh "打雁" \
  --source-sheet-url "https://m.guzheng.cn/qupu/30.html" \
  --source "created by the author" \
  --notes "basic sheet in Sibelius, retouched glissando in MIDI, could add f/mf/mp/p in the sheet later" \
  --overwrite 

python scripts/add_midi_metadata.py "MIDI/bai_jia_chun_han_gao.mid" \
  --title-en "The Joyful Spring (Hakka ver.)" \
  --title-zh "百家春（汉皋）" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1885.html" \
  --source "created by the author" \
  --notes "basic sheet in Sibelius" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/chun_jiang_hua_yue_ye.mid" \
  --title-en "The Moon over the River on a Spring Night" \
  --title-zh "春江花月夜" \
  --source-sheet-url "http://www.eryixian.com/qpbz/qpdq/1217.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius, Bb scale to A scale, change url later" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/zhao_jun_yuan.mid" \
  --title-en "Lament of Lady Zhaojun" \
  --title-zh "昭君怨" \
  --source-sheet-url "https://m.guzheng.cn/qupu/542.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius, 7b to 7" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/he_nan_ba_ban.mid" \
  --title-en "Henan Eight Beats" \
  --title-zh "河南八板" \
  --title-alt "天下大同" \
  --source-sheet-url "https://m.guzheng.cn/qupu/63.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius, 4# to 5, notes corrected in MIDI, (55) to 5" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/fen_die_cai_hua.mid" \
  --title-en "Pink Butterflies Gathering Pollen" \
  --title-zh "粉蝶采花" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1513.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/yu_mei_ren.mid" \
  --title-en "Much Sorrow" \
  --title-zh "虞美人" \
  --title-alt "几多愁" \
  --source-sheet-url "https://m.guzheng.cn/qupu/1513.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/fang_zhi_mang.mid" \
  --title-en "Busy Weaving" \
  --title-zh "纺织忙" \
  --source-sheet-url "https://m.guzheng.cn/qupu/36.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius, add left hand later" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/dong_ting_xin_ge.mid" \
  --title-en "New Song of Dongting Lake" \
  --title-zh "洞庭新歌" \
  --source-sheet-url "https://m.guzheng.cn/qupu/34.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius, removed decorative tremolos, added section 4 in MIDI" \
  --overwrite

python scripts/add_midi_metadata.py "raw_data/MIDI/liu_yang_he.mid" \
  --title-en "Liuyang River" \
  --title-zh "浏阳河" \
  --source-sheet-url "https://m.guzheng.cn/qupu/525.html" \
  --source "(Cheng, 2022), modified by the author" \
  --notes "basic sheet in Sibelius, corrected arpeggios in MIDI" \
  --overwrite
```

## Update `guzheng_scales.json`
```bash
python scripts/scan_midi_non_pentatonic.py raw_data/MIDI/chu_shui_lian.mid --scale D --apply
```

## Transpose MIDI into 5 scales
```bash
# one file
python scripts/transpose_midi_to_pentatonic_scales.py raw_data/MIDI/foo.mid
# whole directory
python scripts/transpose_midi_to_pentatonic_scales.py raw_data/MIDI
```

## Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Jianpu transcription script
Create a JSON file with metadata (id, title_zh, key_signature, source_sheet_images, notes) and measures (measures is a list of bars. for each bar, record the bar number, tempo_bpm, time_signature, and events. for each event, record the start, length, and a list of notes. for each note, record the jianpu notation, scientific pitch notation, guzheng string number, and note type).
Read the sheet images and record the numbers. Pay attention to the dot(s) above or below the number, which signify the octaves. The dot(s) may be below the line(s). So if there are two lines below a 6 and a dot below these two lines, then this note is "6,". 
Do not use OCR. Only read from the images.
The time signature may change across bars, so pay attention to the different time signature and record the correct time signature per bar.
The line(s) under the number indicates the note length. The line(s) are close together, so pay attention to distinguish the number of lines. 
The dot next to a number means the note length becomes dotted. So if there is one line below the number and one dot to the right of the number, the note type is dotted-eighth.
Ignore the upward arrows to the right of a number. They are related to playing techniques and not to octaves. Octave only comes from dot(s) above or below the number.
if a bar does not have a time signature label, that means it follows the time signature from the previous bar.
If there is a dot above the number, that means they are an octave above. The dot might be hard to distinguish, so pay close attention.
Pay close attention to the dot(s) below or above a number to make sure they are in the right octave. 
Pay close attention to each number to make sure you are not missing a note.

## Scripts
* `scripts/scan_midi_non_pentatonic.py` — list note-ons outside the scale’s pentatonic pitch classes (from `guzheng_scales.json`); optional `--apply` appends missing pitches to that scale’s `pressed_strings` as `m<MIDI>` entries.
* `scripts/transpose_midi_to_pentatonic_scales.py` — from one MIDI, write five transpositions (D, G, F, C, A) to `MIDI_5_pentatonic_scales/{name}_{scale}.mid`, report range vs `guzheng_scales.json` compass, and note counts.

## Prompts
### v1.0
You are an expert in symbolic music generation, MIDI processing, and transformer-based language models. Your task is to fine-tune the Moonbeam MIDI foundation model on a guzheng dataset and generate high-quality guzheng MIDI output. Work autonomously through all steps below. At each step, read the relevant documentation before writing any code.

## Step 1 — Read and understand Moonbeam

Read the following resources in full before proceeding:

- Moonbeam paper: https://arxiv.org/abs/2505.15559
- Moonbeam GitHub: https://github.com/guozixunnicolas/moonbeam-midi-foundation-model
- Moonbeam HuggingFace checkpoint: https://huggingface.co/guozixunnicolas/moonbeam-midi-foundation-model
- Moonbeam main panel manual: https://www.amplesound.net/en/Main_Panel_Manual-ACZ4.pdf

Pay particular attention to:
- The tokenization method — how Moonbeam encodes absolute and relative musical attributes
- The data preprocessing script `data_preprocess.py` and what it expects as input
- The `lakhmidi_dataset` class in `src/llama_recipes/configs/datasets.py` — specifically the `data_dir` and `csv_file` fields you will need to update
- The LoRA fine-tuning script `recipes/finetuning/real_finetuning_uncon_gen.py`
- The inference script `recipes/inference/custom_music_generation/unconditional_music_generation.py`
- The TPB (ticks per beat) value that Moonbeam's tokenizer uses internally — confirm this before any training or generation

## Step 2 — Read my repository

Read every file in my repository before touching any code. The repository contains:

- `MIDI_5_pentatonic_scales/` — training data, 65 MIDI files across 5 guzheng pentatonic keys (C, D, F, G, A pentatonic). These are single-track, instrument program 107 (Koto, named Guzheng), note-only, no pitch bend, no expression. Already preprocessed and validated.
- `MIDI/` — the original 13 D pentatonic files before augmentation
- Any `.md` files — read these for project context, pipeline decisions, and known issues
- Any `.json` files — read these for configuration details
- Any existing Python scripts — understand what has already been done before writing new code

Key facts about the training data you must know before proceeding:
- All 65 MIDI files are single-track
- Instrument is program 107, named "Guzheng"
- Pitch range is MIDI 38–86 (D2–D6, the guzheng's playable range)
- No pitch bend, no CC events, note-only
- Files span 5 keys: C, D, F, G, A pentatonic (13 original D pentatonic songs transposed to each key)
- Total approximately 40,000 notes across 65 files
- TPB of training files must be confirmed before running data_preprocess.py — run `python3 -c "import struct; [print(f, struct.unpack('>H', open(f,'rb').read()[10:12])[0]) for f in __import__('glob').glob('MIDI_5_pentatonic_scales/*.mid')]"` and confirm all files share the same TPB value. Record this value — it must match the generated output TPB exactly.

## Step 3 — Fine-tune Moonbeam

Follow these steps exactly in order:

**3.1 Install dependencies**
```bash
conda create --name moonbeam python=3.12
pip install .
pip install src/llama_recipes/transformers_minimal/.
```

**3.2 Download pretrained checkpoint**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='guozixunnicolas/moonbeam-midi-foundation-model',
    local_dir='checkpoints/pretrained'
)
```

**3.3 Preprocess training data**
```bash
python data_preprocess.py \
  --dataset_name guzheng_pentatonic \
  --dataset_folder MIDI_5_pentatonic_scales/ \
  --output_folder data/moonbeam_preprocessed/ \
  --model_config src/llama_recipes/configs/model_config.json \
  --train_test_split_file None \
  --train_ratio 0.9 \
  --ts_threshold None
```

After running, verify:
- Training and test CSV files were created
- The number of preprocessed files matches the number of input MIDI files
- No files were silently dropped

**3.4 Update dataset config**
Edit `src/llama_recipes/configs/datasets.py`. In the `lakhmidi_dataset` class, set:
- `data_dir` to the absolute path of `data/moonbeam_preprocessed/`
- `csv_file` to the absolute path of the training CSV file

Then reinstall: `pip install src/llama_recipes/transformers_minimal/.`

**3.5 Fine-tune with LoRA**
```bash
torchrun --nnodes 1 --nproc_per_node 1 \
  recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 3e-4 \
  --val_batch_size 2 \
  --run_validation True \
  --validation_interval 10 \
  --save_metrics True \
  --dist_checkpoint_root_folder checkpoints/finetuned/guzheng \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path checkpoints/pretrained \
  --pure_bf16 True \
  --enable_ddp True \
  --use_peft True \
  --peft_method lora \
  --quantization False \
  --model_name guzheng_pentatonic \
  --dataset lakhmidi_dataset \
  --output_dir checkpoints/finetuned/guzheng \
  --batch_size_training 2 \
  --context_length 2048 \
  --num_epochs 300 \
  --use_wandb False \
  --gamma 0.99
```

Monitor training loss every 10 epochs. Stop early if validation loss stops decreasing or begins rising. Record the epoch at which the best validation loss was achieved and use that checkpoint for generation.

## Step 4 — Generate MIDI

Generate from both the original pretrained Moonbeam and the fine-tuned model so results can be compared.

**4.1 Prepare prompts**
Extract prompts from the test set. Choose prompts that begin at phrase boundaries — do not start mid-phrase. Prefer prompts from pieces that are clearly pentatonic and melodically active. Create a CSV file listing the test set preprocessed files.

**4.2 Generate from fine-tuned model**
```bash
torchrun --nproc_per_node 1 \
  recipes/inference/custom_music_generation/unconditional_music_generation.py \
  --csv_file data/test_prompts.csv \
  --top_p 0.85 \
  --temperature 0.7 \
  --model_config_path src/llama_recipes/configs/model_config.json \
  --ckpt_dir checkpoints/pretrained \
  --finetuned_PEFT_weight_path checkpoints/finetuned/guzheng \
  --tokenizer_path tokenizer.model \
  --max_seq_len 1024 \
  --max_gen_len 1024 \
  --max_batch_size 6 \
  --num_test_data 20 \
  --prompt_len 50
```

Temperature 0.7 and top_p 0.85 are critical — do not raise temperature above 0.8 as this produces non-pentatonic pitches and malformed duration tokens.

**4.3 Generate from original pretrained model (baseline)**
Run the same inference command but without `--finetuned_PEFT_weight_path`. This gives you the baseline pretrained Moonbeam output for comparison.

**4.4 Post-generation repair — this step is mandatory**

The generated MIDI files will have a TPB header mismatch. The tokenizer encodes timing in ticks but writes an incorrect TPB value to the MIDI header. Fix every generated file before any evaluation:

```python
import struct, os

def get_tpb(path):
    with open(path,'rb') as f: return struct.unpack('>H',f.read()[10:12])[0]

def fix_tpb(in_path, out_path, target_tpb):
    with open(in_path,'rb') as f: data = bytearray(f.read())
    old = struct.unpack('>H',data[10:12])[0]
    data[10:12] = struct.pack('>H', target_tpb)
    with open(out_path,'wb') as f: f.write(data)
    print(f'TPB fixed: {old} -> {target_tpb} | {in_path}')

# confirm training TPB first — all training files must share one value
training_tpbs = set(get_tpb(os.path.join('MIDI_5_pentatonic_scales',f))
                    for f in os.listdir('MIDI_5_pentatonic_scales')
                    if f.endswith('.mid'))
assert len(training_tpbs)==1, f'TPB mismatch in training data: {training_tpbs}'
TRAINING_TPB = training_tpbs.pop()
print(f'Training TPB: {TRAINING_TPB}')

# fix all generated files
for fname in os.listdir('outputs/generated/'):
    if not fname.endswith('.mid'): continue
    fix_tpb(
        os.path.join('outputs/generated/', fname),
        os.path.join('outputs/fixed/', fname),
        TRAINING_TPB
    )
```

After fixing, validate every file:
- Exactly 1 track
- Program 107, name "Guzheng"
- All pitches in range 38–86
- No overlapping notes
- Note density ≥ 0.5 notes per second
- Mean note duration between 0.1 and 3.0 seconds

Any file failing validation should be flagged and regenerated rather than patched.

## Step 5 — Evaluate

Run the following objective metrics on all generated files, comparing fine-tuned Moonbeam vs pretrained Moonbeam vs training data distribution:

**Pitch class distribution** — what percentage of notes fall on each of the 12 pitch classes. A good guzheng output should show strong concentration on 5 pitch classes corresponding to the pentatonic scale. Non-pentatonic pitch classes should be near zero.

**Note density** — mean notes per second. Compare to training data. Fine-tuned model should be closer to training data density than pretrained model.

**Interval histogram** — distribution of melodic intervals (semitones between consecutive notes). Guzheng melody favors stepwise motion and small leaps within the pentatonic scale.

**Mean note duration** — compare to training data. Excessively long durations indicate the model is still producing malformed outputs despite TPB fix.

**Phrase length distribution** — group notes into phrases separated by gaps >0.3 seconds. Compare phrase length distributions between models and training data.

Save all metrics as JSON and all plots as PNG. Do not delete any intermediate outputs.

## Step 6 — Iterate toward better output

After evaluating the first generation, adjust parameters and regenerate if output quality is poor. Follow this decision tree:

**If pitch class distribution shows many non-pentatonic notes** — lower temperature to 0.6 and regenerate. If still present, check training data for non-pentatonic contamination.

**If note density is too low (sparse output, long silences)** — lower temperature further to 0.6. Check that the TPB fix was applied correctly. Verify mean note duration is reasonable.

**If output sounds repetitive** — increase top_p slightly to 0.90. Try different prompt seeds from the test set.

**If output sounds incoherent** — the model may be undertrained. Check validation loss curve. If loss had not plateaued at 300 epochs, run for 100 more epochs and regenerate.

**If fine-tuned model sounds worse than pretrained baseline** — this indicates overfitting. Reduce num_epochs to 150 and retrain. Alternatively reduce LoRA rank in the peft config.

For each iteration, document what changed, what improved, and what did not. Keep all generated files from all iterations — do not overwrite previous outputs.

## Additional context

**About the guzheng** — the guzheng is a traditional Chinese plucked zither with 21 strings, tuned to a pentatonic scale. It is not a chromatic instrument — each string is tuned to one specific pitch in the pentatonic scale, and half-steps are produced by pressing the string behind the bridge. This means generated output should strongly favor the 5 pitches of whichever pentatonic key is being generated. Any chromatic note is an artifact, not a feature.

**About the research goal** — this is an undergraduate thesis investigating whether a pretrained MIDI foundation model (Moonbeam) can be adapted to generate idiomatic guzheng music with minimal fine-tuning data (~40,000 notes, 65 files). The comparison between pretrained and fine-tuned output is the core result. The evaluation should clearly show what fine-tuning adds and where it still falls short.

**About the Markov baseline** — also implement a simple n-gram Markov chain baseline trained on the same 65 files. This serves as the lower bound comparison. Use n=3, tokens = (pitch, quantized_duration). Generate 20 samples at the same target length as the Moonbeam outputs.

**Known issues to watch for:**
- TPB mismatch in generated files — always fix before evaluation
- Non-pentatonic notes from high temperature — keep temperature at 0.7
- Overlapping notes — resolve by truncating earlier note to next note start minus 10ms
- Notes outside guzheng range — transpose by octave to bring in range, do not hard clip
- Flat dynamics in output — apply velocity humanization and CC7 arc after generation if rendering to audio

**Output folder structure to maintain:**
```
outputs/
  generated_finetuned/     # raw output from fine-tuned model
  generated_pretrained/    # raw output from pretrained model
  fixed_finetuned/         # after TPB fix and repair
  fixed_pretrained/        # after TPB fix and repair
  markov/                  # Markov baseline output
  evaluation/              # all metrics JSON and PNG files
  audio/                   # Ample Sound renders (if applicable)
```

### v2.0
You are an expert in symbolic music generation, MIDI processing, and transformer-based sequence models. Your task is to fine-tune two models — Moonbeam and MIDI-RWKV — on a guzheng dataset, generate output from all four model variants (original and fine-tuned for each), evaluate the results, and iterate until the output sounds like a coherent guzheng music piece. Work autonomously through all steps. Read all documentation before writing any code.

Step 1 — Read and understand both models
Read the following resources in full before proceeding. Do not skip any of them.
Moonbeam:

Paper: https://arxiv.org/abs/2505.15559
GitHub: https://github.com/guozixunnicolas/moonbeam-midi-foundation-model
HuggingFace checkpoint: https://huggingface.co/guozixunnicolas/moonbeam-midi-foundation-model
Pay attention to: tokenization method (factored pitch class + octave + duration), data_preprocess.py expected input format, lakhmidi_dataset class in src/llama_recipes/configs/datasets.py, LoRA fine-tuning script, inference script, and the known limitation that pitch class and octave are generated as independent tokens by the GRU sub-decoder — this causes wild octave jumps in generated output that must be addressed at inference time with an octave constraint.

MIDI-RWKV:

Paper: https://arxiv.org/abs/2506.13001
GitHub: https://github.com/christianazinn/MIDI-RWKV
HuggingFace: search for MIDI-RWKV weights on HuggingFace
Pay attention to: the low-sample fine-tuning scheme (state tuning), MIDITok tokenization used, how the pretrained checkpoint is loaded and fine-tuned, inference/generation script, and context length limitations.

MIDITok (used by MIDI-RWKV):

GitHub: https://github.com/Natooz/MidiTok
Documentation: https://miditok.readthedocs.io
Pay attention to: REMI tokenization scheme, unified pitch token representation (0–127 single token per note, not factored), how to configure pitch range, velocity bins, duration resolution, and how to convert MIDI files to token sequences and back.

After reading all resources, write a brief summary of the key architectural differences between the two models before proceeding, specifically addressing how each model represents pitch and why this matters for melodic coherence.

Step 2 — Read the project repository
Read every non-ignored file in the Guzheng/ folder before writing any code. Respect .gitignore — do not read or process any file listed there.
Files you will find include:

MIDI files in MIDI_transposed/ — training data across 5 guzheng pentatonic keys (C, D, F, G, A pentatonic), transposed from original D pentatonic recordings. All files are single-track, instrument program 107 (Koto, named Guzheng), note-only, no pitch bend, no CC expression data.
JSON configuration files — read these for any existing pipeline settings, model configs, or evaluation parameters.
Markdown files — read these for project context, decisions made, known issues, and pipeline documentation accumulated over the course of this project.

Before proceeding, confirm:

How many MIDI files are in MIDI_transposed/
The TPB (ticks per beat) value shared across all training files — run: python3 -c "import struct, glob; [print(f, struct.unpack('>H', open(f,'rb').read()[10:12])[0]) for f in glob.glob('MIDI_transposed/*.mid')]" and verify all files share the same TPB. Record this value — it is critical for post-generation repair.
The pitch range present in the files — confirm all notes fall within MIDI 38–86 (D2–D6, the guzheng's playable range)
The note duration distribution — check whether notes under 10ms are present (Moonbeam's minimum representable duration is 10ms per its log-scale tokenizer; notes below this should be removed before Moonbeam preprocessing)
Whether any non-pentatonic pitch classes are present — the five valid pentatonic key sets are: D={2,4,6,9,11}, C={0,2,4,7,9}, F={5,7,9,0,2}, G={7,9,11,2,4}, A={9,11,1,4,6}


Step 3 — Preprocess training data
Both models require different preprocessing pipelines from the same source MIDI files. Run both pipelines on the files in MIDI_transposed/.
3.1 Shared validation before any preprocessing
Run this validation on every file in MIDI_transposed/ and fix any issues found before proceeding:
pythonimport pretty_midi, struct, os

GUZHENG_MIN = 38   # D2
GUZHENG_MAX = 86   # D6
MIN_DUR_SEC = 0.010  # 10ms — Moonbeam's minimum representable duration

def validate_training_file(path):
    # check TPB
    with open(path, 'rb') as f: data = f.read()
    tpb = struct.unpack('>H', data[10:12])[0]

    midi  = pretty_midi.PrettyMIDI(path)
    assert len(midi.instruments) == 1, f'Expected 1 track: {path}'
    assert midi.instruments[0].program == 107, f'Wrong program: {path}'
    assert midi.instruments[0].name == 'Guzheng', f'Wrong name: {path}'

    notes = midi.instruments[0].notes
    assert len(notes) > 0, f'No notes: {path}'

    pitches = [n.pitch for n in notes]
    assert min(pitches) >= GUZHENG_MIN, f'Pitch below range: {path}'
    assert max(pitches) <= GUZHENG_MAX, f'Pitch above range: {path}'

    under_10ms = [n for n in notes if n.end - n.start < MIN_DUR_SEC]

    return {
        'tpb': tpb,
        'notes': len(notes),
        'under_10ms': len(under_10ms),
        'pitch_classes': sorted(set(n.pitch % 12 for n in notes))
    }
If any file has notes under 10ms, create a cleaned copy with those notes removed before passing to Moonbeam. MIDI-RWKV can use the original files since MIDITok handles very short notes differently.
3.2 Moonbeam preprocessing
Moonbeam's tokenizer encodes pitch as two independent tokens: pitch class (0–11) and octave (2–7). This is a known architectural limitation causing octave coherence issues in generated output — you will address this at inference time, not here.
bashpython data_preprocess.py \
  --dataset_name guzheng_transposed \
  --dataset_folder MIDI_transposed/ \
  --output_folder data/moonbeam_preprocessed/ \
  --model_config src/llama_recipes/configs/model_config.json \
  --train_test_split_file None \
  --train_ratio 0.9 \
  --ts_threshold None
After running, verify:

Training and test CSV files were created
Number of preprocessed files matches number of input files — if files were silently dropped, investigate why
Check the TPB written into the preprocessed data matches the training file TPB recorded in Step 2

Update src/llama_recipes/configs/datasets.py lakhmidi_dataset class with the correct data_dir and csv_file paths, then reinstall: pip install src/llama_recipes/transformers_minimal/.
3.3 MIDI-RWKV preprocessing with MIDITok
MIDI-RWKV uses MIDITok REMI tokenization with unified pitch tokens (single token per note covering full MIDI range). This directly addresses Moonbeam's octave incoherence problem.
pythonfrom miditok import REMI, TokenizerConfig
from pathlib import Path

config = TokenizerConfig(
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_pitch_intervals=False,    # unified pitch, not relative intervals
    use_programs=False,           # single instrument
    beat_res={(0, 4): 8, (4, 12): 4},
    pitch_range=range(38, 87),    # guzheng range only — reduces vocab size
)

tokenizer = REMI(config)

midi_files = list(Path('MIDI_transposed/').glob('*.mid'))
print(f'Tokenizing {len(midi_files)} files')

# convert all files to token sequences
# follow MIDI-RWKV's expected input format exactly as documented in its repo
# check whether MIDI-RWKV expects a pre-built tokenizer vocabulary or builds it from data
Read MIDI-RWKV's GitHub README and data preparation scripts carefully before finalizing the tokenization configuration. The tokenizer configuration above is a starting point — match it exactly to what MIDI-RWKV's training script expects. If MIDI-RWKV uses a pretrained tokenizer vocabulary, load that instead of building from scratch.

Step 4 — Fine-tune both models
4.1 Fine-tune Moonbeam with LoRA
bashtorchrun --nnodes 1 --nproc_per_node 1 \
  recipes/finetuning/real_finetuning_uncon_gen.py \
  --lr 3e-4 \
  --val_batch_size 2 \
  --run_validation True \
  --validation_interval 10 \
  --save_metrics True \
  --dist_checkpoint_root_folder checkpoints/moonbeam_finetuned \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path checkpoints/pretrained \
  --pure_bf16 True \
  --enable_ddp True \
  --use_peft True \
  --peft_method lora \
  --quantization False \
  --model_name guzheng_transposed \
  --dataset lakhmidi_dataset \
  --output_dir checkpoints/moonbeam_finetuned \
  --batch_size_training 2 \
  --context_length 2048 \
  --num_epochs 300 \
  --use_wandb False \
  --gamma 0.99
Monitor validation loss every 10 epochs. Stop early if validation loss stops decreasing or rises. Record the best checkpoint epoch.
4.2 Fine-tune MIDI-RWKV
Follow the fine-tuning procedure documented in MIDI-RWKV's GitHub exactly. Use the state tuning approach described in the paper for the low-sample regime — this is MIDI-RWKV's specific contribution for datasets like yours with a limited number of files.
Key parameters to set or verify:

Use the pretrained MIDI-RWKV checkpoint as starting point
Apply the state tuning method for low-sample fine-tuning as described in the paper
Match context length to the average token sequence length of your training files
Save checkpoints every N steps and monitor validation loss


Step 5 — Generate MIDI from all four model variants
Generate output from: (1) original pretrained Moonbeam, (2) fine-tuned Moonbeam, (3) original pretrained MIDI-RWKV, (4) fine-tuned MIDI-RWKV.
Generate at least 10 samples from each model variant. Use prompts from your test set files for seeded generation where supported.
5.1 Moonbeam generation — with inference-time octave constraint
This constraint is mandatory. Without it, Moonbeam generates wild octave jumps (observed at 28% large leaps vs 16% in real guzheng) because pitch class and octave are generated independently.
python# After sampling pitch_class token, constrain octave token to ±1 octave
# of the previous note's octave before sampling
# Modify the inference loop in the generation script:

def constrained_octave_sample(logits, prev_pitch, pitch_class, valid_range=(38, 86)):
    prev_octave = prev_pitch // 12
    # allow ±1 octave from previous note
    valid_octaves = [prev_octave - 1, prev_octave, prev_octave + 1]
    # further constrain to guzheng range
    valid_octaves = [
        o for o in valid_octaves
        if valid_range[0] <= (pitch_class + o * 12) <= valid_range[1]
    ]
    # mask all invalid octave tokens before sampling
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    for o in valid_octaves:
        token_idx = octave_to_token_idx(o)  # map octave to token index
        mask[token_idx] = False
    logits[mask] = float('-inf')
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)
Generation parameters:

temperature: 0.7 (do not raise above 0.8 — higher values produce non-pentatonic pitches and malformed duration tokens)
top_p: 0.85
max_gen_len: 1024 tokens
prompt_len: 50 tokens from test set files starting at phrase boundaries

5.2 MIDI-RWKV generation
Follow the generation script in MIDI-RWKV's GitHub. If text prompts are supported, use prompts such as:

"traditional Chinese guzheng music, pentatonic scale, D major pentatonic, melodic, expressive"
"Chinese zither, 古筝, pentatonic melody, traditional folk music"
"guzheng solo, pentatonic, slow melodic phrase, traditional Chinese"

5.3 Post-generation repair — mandatory for Moonbeam output
Moonbeam generated files have a TPB header mismatch. The tokenizer encodes timing correctly but writes the wrong TPB value to the MIDI header. Fix every Moonbeam-generated file before evaluation:
pythonimport struct, os

def get_tpb(path):
    with open(path, 'rb') as f:
        return struct.unpack('>H', f.read()[10:12])[0]

def fix_tpb_header(in_path, out_path, target_tpb):
    """
    Fix MIDI TPB by rewriting header bytes 10-11 only.
    Do NOT rescale event times — tick values are already correct.
    The mismatch is a header labeling error, not a timing error.
    Multiplying event times by a scale factor makes the file worse.
    """
    with open(in_path, 'rb') as f:
        data = bytearray(f.read())
    old_tpb = struct.unpack('>H', data[10:12])[0]
    data[10:12] = struct.pack('>H', target_tpb)
    with open(out_path, 'wb') as f:
        f.write(data)
    print(f'TPB fixed: {old_tpb} -> {target_tpb}')

# get training TPB confirmed in Step 2
training_tpbs = set(get_tpb(f) for f in glob.glob('MIDI_transposed/*.mid'))
assert len(training_tpbs) == 1, f'TPB mismatch in training data: {training_tpbs}'
TRAINING_TPB = training_tpbs.pop()

# fix all Moonbeam generated files
for fname in os.listdir('outputs/moonbeam_generated/'):
    if fname.endswith('.mid'):
        fix_tpb_header(
            f'outputs/moonbeam_generated/{fname}',
            f'outputs/moonbeam_fixed/{fname}',
            TRAINING_TPB
        )
Then run full repair on all Moonbeam outputs:
pythonimport pretty_midi, numpy as np

def repair_generated(in_path, out_path):
    midi = pretty_midi.PrettyMIDI(in_path)
    assert len(midi.instruments) == 1
    inst = midi.instruments[0]
    inst.program = 107
    inst.name = 'Guzheng'

    # clip pitches to guzheng range — transpose by octave, not hard clip
    for note in inst.notes:
        while note.pitch < 38: note.pitch += 12
        while note.pitch > 86: note.pitch -= 12

    # remove notes shorter than 10ms
    inst.notes = [n for n in inst.notes if n.end - n.start >= 0.010]

    # sort by onset
    inst.notes.sort(key=lambda n: n.start)

    # resolve overlaps — truncate note if it bleeds into next note onset
    for i in range(len(inst.notes) - 1):
        max_end = inst.notes[i+1].start - 0.010
        if inst.notes[i].end > max_end:
            inst.notes[i].end = max(inst.notes[i].start + 0.010, max_end)

    midi.write(out_path)
```

---

## Step 6 — Evaluate all four model outputs

Run the following metrics on all outputs and on the training data for comparison. Save all results as JSON and all plots as PNG in `outputs/evaluation/`.

### 6.1 Objective metrics

For each of the four model variants, compute:

**Pitch class distribution** — percentage of notes on each of the 12 pitch classes. Good guzheng output should show strong concentration on 5 pitch classes corresponding to the pentatonic key, with near-zero non-pentatonic pitch classes. Report the percentage of non-pentatonic notes explicitly.

**Large leap rate** — percentage of consecutive note pairs where the interval exceeds 12 semitones (one octave). Training data baseline is approximately 16%. Moonbeam without octave constraint produces approximately 28%. This is the single most important metric for melodic coherence.

**Note density** — mean notes per second. Compare to training data distribution.

**Mean note duration** — compare to training data. Excessively long durations (mean > 3s) indicate TPB fix was not applied or was applied incorrectly.

**Interval histogram** — distribution of melodic intervals between consecutive notes. Guzheng melody favors stepwise motion and small leaps within the pentatonic scale.

**Pitch class entropy** — Shannon entropy of the pitch class distribution. Higher entropy means more even spread across pitch classes. Training data has moderate entropy across 5 pitch classes. Collapsed distributions (e.g., B at 27%, D at 21% as observed in earlier Moonbeam output) have low entropy.

### 6.2 Informal listening evaluation

Listen to at least 3 outputs from each model variant. For each, note:
- Does it sound like a continuous melody or random disconnected notes?
- Does it stay in a recognizable pentatonic key?
- Are there audible octave jumps that break melodic flow?
- Does it have any phrase structure — a sense of beginning, development, and resolution?
- Does it sound at all like guzheng music?

Record observations as structured notes, not just subjective impressions.

---

## Step 7 — Iterate toward better output

After the first evaluation, apply the following fixes in priority order based on what the evaluation reveals.

**If large leap rate exceeds 20% in Moonbeam output** — the inference-time octave constraint was not applied correctly or not applied at all. Debug and reapply. This is the single highest-impact fix available.

**If pitch class distribution is collapsed** (two pitch classes above 25% each) — lower temperature to 0.6 and regenerate. Check training data for pentatonic purity issues.

**If non-pentatonic notes exceed 5%** — lower temperature further. Consider applying a post-generation scale quantize step that snaps non-pentatonic notes to the nearest valid pentatonic pitch in the estimated key.

**If note density is too low** (fewer than 1 note per second on average) — verify TPB fix was applied. Check mean note duration — if above 3 seconds, the TPB fix failed.

**If MIDI-RWKV output lacks phrase structure** — the model may need more fine-tuning steps. Check validation loss — if not converged, run more epochs.

**If fine-tuned models sound worse than pretrained baselines** — this indicates overfitting. For Moonbeam, reduce num_epochs to 150 and retrain. For MIDI-RWKV, reduce the number of state tuning steps.

For each iteration, document what changed, what metric improved, and what did not. Keep all generated files from all iterations in separate timestamped folders — never overwrite previous outputs.

---

## Step 8 — Output folder structure

Maintain this structure throughout:
```
outputs/
  moonbeam_raw/           # raw Moonbeam output before TPB fix
  moonbeam_fixed/         # after TPB fix and repair
  moonbeam_finetuned_raw/ # fine-tuned Moonbeam raw output
  moonbeam_finetuned_fixed/ # fine-tuned Moonbeam fixed
  midi_rwkv_original/     # original pretrained MIDI-RWKV output
  midi_rwkv_finetuned/    # fine-tuned MIDI-RWKV output
  evaluation/             # all metrics JSON and PNG files
  iterations/             # timestamped subfolders for each iteration

Critical context for all steps
About the guzheng: The guzheng is a traditional Chinese plucked zither with 21 strings tuned to a pentatonic scale. It is not chromatic — each string is tuned to one pitch in the pentatonic scale, and semitone alterations are produced by pressing strings behind the bridge. Generated output must strongly favor the 5 pitches of the pentatonic key. Any chromatic note is a generation artifact, not a stylistic choice.
About the training data: All files in MIDI_transposed/ are the same 13 original D pentatonic guzheng recordings transposed to C, D, F, G, and A pentatonic keys. The transpositions are musically valid guzheng keys. Note-only, no pitch bend, no expression data. Single track, program 107, name "Guzheng".
About the known Moonbeam problems: From prior experimentation on this dataset, Moonbeam fine-tuned on these files produces: (1) wild octave jumps at 28% vs 16% in real guzheng due to independent pitch class and octave generation — fix with inference-time octave constraint; (2) pitch class collapse toward B and D — fix with lower temperature; (3) no phrase structure — this is an architectural limitation to document rather than fix; (4) TPB header mismatch in every generated file — fix with header byte rewrite as described above.
About MIDI-RWKV's advantage: MIDI-RWKV uses unified pitch tokens via MIDITok REMI — a single token represents pitch 0–127 without factoring into octave and pitch class. This directly prevents the octave jump problem at the architectural level. The comparison between Moonbeam and MIDI-RWKV is therefore a controlled experiment isolating tokenization scheme as the variable, which is the core research contribution of this thesis.
About evaluation: The goal is not just metrics but perceptual quality. A piece that passes all objective metrics but sounds like random notes is a failure. A piece with slightly elevated leap rate but clear pentatonic melody and phrase structure is a success. Objective metrics guide iteration — listening is the final judge.
Do not proceed to the next step until the current step is complete and verified. Report what you find at each step before taking action. If something is unclear or underdocumented, read the source code rather than guessing.