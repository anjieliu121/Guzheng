"""Generate guzheng ABC samples from the character-level GPT baseline.

Run:
    python test_and_trial_8/generate.py --num 50
    python test_and_trial_8/generate.py --num 50 --temperature 1.1
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "generated"
OUT_DIR.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--num", type=int, default=10)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_k", type=int, default=50)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--max_chars", type=int, default=8000,
                help="max characters to generate")
ap.add_argument("--ckpt", type=str, default=None)
args = ap.parse_args()

# Import model class — must patch sys.argv first to avoid argparse conflict
_saved_argv = sys.argv
sys.argv = sys.argv[:1]
sys.path.insert(0, str(ROOT))
from train import CharGPT
sys.argv = _saved_argv

# ---- device ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

# ---- load model ----
ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / "chargpt_weighted_best.pth"
print(f"[ckpt] loading {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = CharGPT(
    vocab_size=128,
    d_model=256,
    n_heads=4,
    n_layers=6,
    d_ff=256,
    max_seq_len=4096,
    dropout=0.0,  # no dropout at inference
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"[ckpt] epoch {ckpt['epoch']}, eval_loss={ckpt.get('eval_loss', '?'):.4f}")

# ---- prompt: same metadata as NotaGen ----
PROMPT = (
    "L:1/32\n"
    "M:4/4\n"
    "K:C\n"
    "V:1 treble nm=\"Guzheng\"\n"
    "[V:1]"
)


def generate_one(seed):
    torch.manual_seed(seed)
    prompt_ids = [1] + [ord(c) for c in PROMPT]  # BOS + prompt
    tokens = model.generate(
        prompt_ids,
        max_new_tokens=args.max_chars,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    # Decode tokens to text (skip BOS=1, stop at EOS=2)
    chars = []
    for t in tokens[1:]:  # skip BOS
        if t == 2:  # EOS
            break
        if 32 <= t < 127 or t in (10, 13, 9):  # printable + newline/tab
            chars.append(chr(t))
    return "".join(chars)


def clean_abc(raw_text):
    """Clean up generated ABC into valid format for abc2midi."""
    lines = raw_text.split("\n")
    meta_lines = []
    body_lines = []
    in_body = False

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("[V:") or "[V:" in ln:
            in_body = True
        if not in_body:
            meta_lines.append(ln)
        else:
            # Strip [V:1] tags for abc2midi
            clean = ln.replace("[V:1]", "").strip()
            if clean:
                body_lines.append(clean)

    # Ensure required headers
    has_x = any(l.startswith("X:") for l in meta_lines)
    has_t = any(l.startswith("T:") for l in meta_lines)
    header = []
    if not has_x:
        header.append("X:1")
    if not has_t:
        header.append("T:CharGPT Guzheng")

    return "\n".join(header + meta_lines + ["V:1"] + body_lines) + "\n"


def abc_to_midi(abc_path, midi_path):
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
    raw = generate_one(seed=2000 + i)
    elapsed = time.time() - t0

    # Save raw
    raw_path = run_dir / f"sample_{i+1:02d}_raw.abc"
    raw_path.write_text(raw)

    # Clean and save
    clean = clean_abc(raw)
    abc_path = run_dir / f"sample_{i+1:02d}.abc"
    abc_path.write_text(clean)

    # Convert to MIDI
    midi_path = run_dir / f"sample_{i+1:02d}.mid"
    midi_ok, midi_log = abc_to_midi(abc_path, midi_path)

    bars = clean.count("|")
    status = "OK" if midi_ok else "MIDI failed"
    print(f"  {elapsed:.1f}s  {len(raw)} chars  ~{bars} bars  -> {status}")
    if not midi_ok:
        err_lines = [l for l in midi_log.strip().split("\n") if l.strip()]
        if err_lines:
            print(f"  abc2midi: {err_lines[-1]}")
    if midi_ok:
        ok += 1

print(f"\n[done] {ok}/{args.num} valid MIDI in {run_dir}")
