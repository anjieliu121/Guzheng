"""Generate guzheng ABC samples from a fine-tuned NotaGen checkpoint, MPS-friendly.

NotaGen's stock inference.py is hardcoded for CUDA + the multi-voice rest_unreduce
post-processor (which is for orchestral scores). Our data is single-voice so we
skip that step entirely.

Run:
    python test_and_trial_7/03_generate.py --size small --num 5
    python test_and_trial_7/03_generate.py --size small --num 5 --temperature 0.9
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
NOTAGEN = ROOT / "NotaGen"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "generated"
OUT_DIR.mkdir(exist_ok=True)

# ---- args ----
ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=["small", "medium", "large"], default="small")
ap.add_argument("--num", type=int, default=5, help="number of samples")
ap.add_argument("--top_k", type=int, default=9)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--max_minutes", type=float, default=10.0,
                help="abort one sample if it takes longer than this")
ap.add_argument("--ckpt", type=str, default=None,
                help="override checkpoint path (default: notagen_guzheng_<size>_best.pth)")
args = ap.parse_args()

SIZE_CONFIGS = {
    "small":  dict(PATCH_NUM_LAYERS=12, CHAR_NUM_LAYERS=3, HIDDEN_SIZE=768),
    "medium": dict(PATCH_NUM_LAYERS=16, CHAR_NUM_LAYERS=3, HIDDEN_SIZE=1024),
    "large":  dict(PATCH_NUM_LAYERS=20, CHAR_NUM_LAYERS=6, HIDDEN_SIZE=1280),
}
sc = SIZE_CONFIGS[args.size]

# ---- inject NotaGen config BEFORE importing utils ----
sys.path.insert(0, str(NOTAGEN / "finetune"))
import config
config.PATCH_STREAM = True
config.PATCH_SIZE = 16
config.PATCH_LENGTH = 1024
config.CHAR_NUM_LAYERS = sc["CHAR_NUM_LAYERS"]
config.PATCH_NUM_LAYERS = sc["PATCH_NUM_LAYERS"]
config.HIDDEN_SIZE = sc["HIDDEN_SIZE"]

from utils import Patchilizer, NotaGenLMHeadModel  # noqa: E402
from transformers import GPT2Config  # noqa: E402

# ---- device ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

# ---- model ----
patch_cfg = GPT2Config(num_hidden_layers=config.PATCH_NUM_LAYERS,
                       max_length=config.PATCH_LENGTH,
                       max_position_embeddings=config.PATCH_LENGTH,
                       n_embd=config.HIDDEN_SIZE,
                       num_attention_heads=config.HIDDEN_SIZE // 64,
                       vocab_size=1)
char_cfg = GPT2Config(num_hidden_layers=config.CHAR_NUM_LAYERS,
                      max_length=config.PATCH_SIZE + 1,
                      max_position_embeddings=config.PATCH_SIZE + 1,
                      hidden_size=config.HIDDEN_SIZE,
                      num_attention_heads=config.HIDDEN_SIZE // 64,
                      vocab_size=128)

model = NotaGenLMHeadModel(encoder_config=patch_cfg, decoder_config=char_cfg)

ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / f"notagen_guzheng_{args.size}_best.pth"
print(f"[ckpt] loading {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model = model.to(device).eval()
print(f"[ckpt] loaded epoch {ckpt['epoch']} eval_loss={ckpt.get('eval_loss', '?')}")

patchilizer = Patchilizer()

# ---- prompt ----
# Single-voice 4/4 in C, 1/32 unit. The model has only seen this exact metadata
# in training, so we lock it in and let it generate the body.
PROMPT_LINES = [
    "L:1/32\n",
    "M:4/4\n",
    "K:C\n",
    "V:1 treble nm=\"Guzheng\"\n",
]


def generate_one(seed: int, deadline_seconds: float):
    torch.manual_seed(seed)
    bos_patch = [patchilizer.bos_token_id] * (config.PATCH_SIZE - 1) + [patchilizer.eos_token_id]
    prompt_patches = patchilizer.patchilize_metadata(PROMPT_LINES)
    byte_list = list("".join(PROMPT_LINES))
    prompt_patches = [
        [ord(c) for c in p] + [patchilizer.special_token_id] * (config.PATCH_SIZE - len(p))
        for p in prompt_patches
    ]
    prompt_patches.insert(0, bos_patch)
    input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

    t0 = time.time()
    tunebody_flag = False
    while True:
        if time.time() - t0 > deadline_seconds:
            return None
        predicted = model.generate(input_patches.unsqueeze(0),
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    temperature=args.temperature)
        # When the model first emits a tunebody marker, NotaGen rewinds to anchor [r:0/
        if not tunebody_flag and patchilizer.decode([predicted]).startswith("[r:"):
            tunebody_flag = True
            r0 = torch.tensor([ord(c) for c in "[r:0/"]).unsqueeze(0).to(device)
            tmp = torch.cat([input_patches, r0], dim=-1)
            predicted = model.generate(tmp.unsqueeze(0),
                                        top_k=args.top_k,
                                        top_p=args.top_p,
                                        temperature=args.temperature)
            predicted = [ord(c) for c in "[r:0/"] + predicted

        if (predicted[0] == patchilizer.bos_token_id
                and predicted[1] == patchilizer.eos_token_id):
            break

        nxt = patchilizer.decode([predicted])
        for ch in nxt:
            byte_list.append(ch)

        # Pad anything past EOS within the patch
        end_seen = False
        for j in range(len(predicted)):
            if end_seen:
                predicted[j] = patchilizer.special_token_id
            if predicted[j] == patchilizer.eos_token_id:
                end_seen = True

        predicted = torch.tensor([predicted], device=device)
        input_patches = torch.cat([input_patches, predicted], dim=1)

        if input_patches.shape[1] >= config.PATCH_LENGTH * config.PATCH_SIZE:
            # Hit context limit; stop cleanly.
            break
        if len(byte_list) > 100000:
            break

    return "".join(byte_list)


def stream_to_standard(stream_text: str) -> str:
    """Convert NotaGen 'stream' output (with [r:i/N] line tags and [V:1] inline)
    back to a standard ABC body so abc2midi can parse it."""
    import re
    lines = stream_text.split("\n")
    out_meta = []
    out_body = []
    in_body = False
    for ln in lines:
        if ln.startswith("[r:") or "[V:" in ln:
            in_body = True
        if not in_body:
            if ln.strip():
                out_meta.append(ln)
            continue
        # Strip [r:i/N] prefix and [V:1] tags
        ln = re.sub(r"^\[r:[^\]]*\]", "", ln)
        ln = ln.replace("[V:1]", "")
        ln = ln.strip()
        if ln:
            out_body.append(ln)
    # Force a valid X/T header in case the model didn't emit them
    has_x = any(l.startswith("X:") for l in out_meta)
    has_t = any(l.startswith("T:") for l in out_meta)
    header = []
    if not has_x:
        header.append("X:1")
    if not has_t:
        header.append("T:Generated Guzheng")
    return "\n".join(header + out_meta + ["V:1"] + out_body) + "\n"


def abc_to_midi(abc_path: Path, midi_path: Path):
    res = subprocess.run(
        ["abc2midi", str(abc_path), "-o", str(midi_path)],
        capture_output=True, text=True,
    )
    return res.returncode == 0, res.stdout + res.stderr


# ---- main ----
run_dir = OUT_DIR / time.strftime("%Y%m%d-%H%M%S")
run_dir.mkdir()
print(f"[out] {run_dir}")

ok = 0
for i in range(args.num):
    print(f"\n--- sample {i+1}/{args.num} ---")
    t0 = time.time()
    text = generate_one(seed=1000 + i, deadline_seconds=args.max_minutes * 60)
    if text is None:
        print(f"  TIMEOUT after {args.max_minutes} min")
        continue
    elapsed = time.time() - t0
    raw_path = run_dir / f"sample_{i+1:02d}_raw.abc"
    raw_path.write_text(text)
    std_text = stream_to_standard(text)
    abc_path = run_dir / f"sample_{i+1:02d}.abc"
    abc_path.write_text(std_text)
    midi_path = run_dir / f"sample_{i+1:02d}.mid"
    midi_ok, midi_log = abc_to_midi(abc_path, midi_path)
    bars = std_text.count("|")
    status = "OK" if midi_ok else "ABC ok, MIDI failed"
    print(f"  {elapsed:.1f}s  {len(text)} bytes  ~{bars} bars  -> {status}")
    if not midi_ok:
        print("  abc2midi:", midi_log.strip().split("\n")[-1])
    if midi_ok:
        ok += 1

print(f"\n[done] {ok}/{args.num} valid MIDI in {run_dir}")
