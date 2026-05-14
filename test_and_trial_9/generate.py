"""Generate samples from a trained PatchGPT model.

Usage:
    python test_and_trial_9/generate.py --n 50
    python test_and_trial_9/generate.py --n 10 --config small
    python test_and_trial_9/generate.py --n 5 --ckpt path/to/checkpoint.pth
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# ── Import model class from train.py ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# Patch sys.argv to avoid argparse conflict with train.py
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]

from train import PatchGPT, PATCH_SIZE, BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE, CONFIGS

sys.argv = _saved_argv

REPO = ROOT.parent
CKPT_DIR = ROOT / "checkpoints"

# ── Args ──────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50, help="number of samples")
ap.add_argument("--config", choices=["medium", "small", "tiny"], default="medium")
ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: auto)")
ap.add_argument("--max_patches", type=int, default=128, help="max patches to generate")
ap.add_argument("--window", type=int, default=64, help="sliding window for patch decoder")
ap.add_argument("--temperature", type=float, default=1.2)
ap.add_argument("--top_k", type=int, default=9)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--outdir", type=str, default=None)
args = ap.parse_args()

# ── Device ────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

# ── Load model ────────────────────────────────────────────────────────────
if args.ckpt:
    ckpt_path = Path(args.ckpt)
else:
    ckpt_path = CKPT_DIR / f"patchgpt_{args.config}_best.pth"

print(f"[ckpt] {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Read config from checkpoint or from args
cfg_name = ckpt.get("config", args.config)
CFG = CONFIGS[cfg_name]
max_patches = ckpt.get("max_patches", args.max_patches)

model = PatchGPT(
    d_model=ckpt.get("d_model", CFG["d_model"]),
    n_heads=ckpt.get("n_heads", CFG["n_heads"]),
    patch_layers=ckpt.get("patch_layers", CFG["patch_layers"]),
    char_layers=ckpt.get("char_layers", CFG["char_layers"]),
    d_ff=ckpt.get("d_ff", CFG["d_ff"]),
    max_patches=max_patches,
    dropout=0.0,  # no dropout at inference
).to(device)

model.load_state_dict(ckpt["model"])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"[model] PatchGPT-{cfg_name} params: {n_params/1e6:.2f}M  "
      f"epoch={ckpt.get('epoch','?')}  eval_loss={ckpt.get('eval_loss','?')}")

# ── Output directory ──────────────────────────────────────────────────────
if args.outdir:
    outdir = Path(args.outdir)
else:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "generated" / stamp
outdir.mkdir(parents=True, exist_ok=True)
print(f"[output] {outdir}")

# ── Prompt patches (same metadata as NotaGen/CharGPT) ─────────────────────
PROMPT_TEXT = "L:1/32\nM:4/4\nK:C\nV:1 treble nm=\"Guzheng\"\n[V:1]"

def text_to_patches(text):
    ids = [ord(c) for c in text if ord(c) < VOCAB_SIZE]
    patches = []
    patches.append([BOS_ID] * 15 + [EOS_ID])
    for i in range(0, len(ids), PATCH_SIZE):
        chunk = ids[i:i + PATCH_SIZE]
        if len(chunk) < PATCH_SIZE:
            chunk = chunk + [PAD_ID] * (PATCH_SIZE - len(chunk))
        patches.append(chunk)
    return patches


def patches_to_text(patches):
    """Decode patches back to ABC text, skipping BOS/EOS/PAD."""
    text = []
    for patch in patches:
        for c in patch:
            if c > 2:  # skip PAD(0), BOS(1), EOS(2)
                text.append(chr(c))
    return "".join(text)


def clean_abc(raw):
    """Clean generated ABC text for abc2midi.

    PatchGPT tends to re-emit [V:1] headers and produce fragmented lines.
    We merge all tunebody content into continuous bars.
    """
    lines = raw.split("\n")
    # Remove stream markers
    lines = [re.sub(r'\[r:[^\]]*\]', '', l).rstrip() for l in lines]
    lines = [l for l in lines if l]

    # Separate header lines from tunebody
    header = []
    body_parts = []
    in_body = False
    for line in lines:
        if not in_body:
            if line.startswith("[V:"):
                in_body = True
                # Keep first [V:1] as header
                header.append(line.split("]")[0] + "]")
                # Rest of this line is body content
                rest = "]".join(line.split("]")[1:]).strip()
                if rest:
                    body_parts.append(rest)
            else:
                header.append(line)
        else:
            # Remove duplicate [V:1] or [V:1...] headers in body
            cleaned = re.sub(r'\[V:1[^\]]*\]', '', line).strip()
            if cleaned:
                body_parts.append(cleaned)

    # Join body into one line, then split at bar lines for readability
    body = " ".join(body_parts)
    # Remove double spaces
    body = re.sub(r'\s+', ' ', body).strip()

    # Ensure we have required headers
    out_lines = []
    has_x = False
    for h in header:
        if h.startswith("X:"): has_x = True
        out_lines.append(h)
    if not has_x:
        out_lines.insert(0, "X:1")

    # Split body into lines at | for readability
    if body:
        bars = body.split("|")
        line = ""
        for bar in bars:
            bar = bar.strip()
            if not bar:
                continue
            if line:
                line += " | " + bar
            else:
                line = bar
            if len(line) > 60:
                out_lines.append(line + " |")
                line = ""
        if line:
            out_lines.append(line + " |]")

    return "\n".join(out_lines) + "\n"


prompt_patches = text_to_patches(PROMPT_TEXT)
print(f"[prompt] {len(prompt_patches)} patches from metadata")

# ── Generate ──────────────────────────────────────────────────────────────
valid = 0
total = 0

for idx in range(1, args.n * 3 + 1):  # try up to 3x to get n valid
    if valid >= args.n:
        break
    total += 1

    t0 = time.time()
    with torch.no_grad():
        gen_patches = model.generate(
            prompt_patches,
            max_patches=args.max_patches,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            window=args.window,
        )

    abc_raw = patches_to_text(gen_patches)
    abc_clean = clean_abc(abc_raw)
    elapsed = time.time() - t0

    # Write ABC
    abc_path = outdir / f"sample_{valid+1:02d}.abc"
    abc_path.write_text(abc_clean, encoding="utf-8")

    # Convert to MIDI
    mid_path = outdir / f"sample_{valid+1:02d}.mid"
    try:
        result = subprocess.run(
            ["abc2midi", str(abc_path), "-o", str(mid_path)],
            capture_output=True, text=True, timeout=30)
        if mid_path.exists() and mid_path.stat().st_size > 100:
            valid += 1
            n_patches = len(gen_patches) - len(prompt_patches)
            print(f"  [{valid:2d}/{args.n}] sample_{valid:02d} "
                  f"patches={n_patches} chars={len(abc_raw)} {elapsed:.1f}s")
        else:
            abc_path.unlink(missing_ok=True)
            mid_path.unlink(missing_ok=True)
            print(f"  [skip] attempt {total}: abc2midi produced empty MIDI")
    except Exception as e:
        abc_path.unlink(missing_ok=True)
        mid_path.unlink(missing_ok=True)
        print(f"  [skip] attempt {total}: {e}")

print(f"\n[done] {valid}/{args.n} valid MIDI in {outdir}")
