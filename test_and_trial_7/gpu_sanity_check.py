"""GPU (MPS) sanity check for NotaGen-small fine-tuning on Apple M3 Pro.

Goal: before committing to a real fine-tune, verify that this machine can
actually run a forward + backward pass at the target sequence length and
parameter count, and measure how long one accumulation step takes.

We do NOT load the real NotaGen weights here — we build a decoder-only
transformer with a comparable parameter budget (~110M) and run it under the
same conditions we plan to use for fine-tuning:

    device       = mps
    dtype        = fp32   (bf16 is unstable on MPS as of torch 2.5)
    batch size   = 1
    seq length   = 1024
    grad accum   = 8
    grad ckpt    = on
    optimizer    = AdamW

If this script completes without OOM and the per-step time is reasonable
(<~30s), the real NotaGen-small fine-tune is feasible locally.
"""
import gc
import platform
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------- target config (matches planned NotaGen-small fine-tune) ----------
P_LENGTH      = 1024
BATCH         = 1
ACCUM         = 8
VOCAB         = 256          # char-level ABC ~ byte vocab
D_MODEL       = 768
N_HEADS       = 12
N_LAYERS      = 12           # ~110M params with d=768, layers=12
FFN_MULT      = 4
LR            = 1e-5
WARMUP_STEPS  = 2            # tiny — we just want to time a real step
GRAD_CKPT     = True


# ---------- model ----------
class Block(nn.Module):
    def __init__(self, d, h, ffn_mult):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ffn_mult),
            nn.GELU(),
            nn.Linear(d * ffn_mult, d),
        )

    def forward(self, x, attn_mask):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class TinyDecoder(nn.Module):
    def __init__(self, vocab, d, h, layers, ffn_mult, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, ffn_mult) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool),
                          diagonal=1)
        for blk in self.blocks:
            if GRAD_CKPT and self.training:
                x = checkpoint(blk, x, mask, use_reentrant=False)
            else:
                x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)


# ---------- helpers ----------
def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def report_env():
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"  platform     : {platform.platform()}")
    print(f"  python       : {platform.python_version()}")
    print(f"  torch        : {torch.__version__}")
    print(f"  mps available: {torch.backends.mps.is_available()}")
    print(f"  mps built    : {torch.backends.mps.is_built()}")
    print(f"  cuda         : {torch.cuda.is_available()}")
    print()


def report_config(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("MODEL / TRAINING CONFIG")
    print("=" * 60)
    print(f"  d_model      : {D_MODEL}")
    print(f"  n_heads      : {N_HEADS}")
    print(f"  n_layers     : {N_LAYERS}")
    print(f"  vocab        : {VOCAB}")
    print(f"  seq length   : {P_LENGTH}")
    print(f"  batch        : {BATCH}")
    print(f"  grad accum   : {ACCUM}")
    print(f"  grad ckpt    : {GRAD_CKPT}")
    print(f"  dtype        : fp32")
    print(f"  params total : {n_params/1e6:.1f} M")
    print(f"  params train : {n_train/1e6:.1f} M")
    # rough memory budget: params + grads + adam(m,v) = 4x params at fp32
    budget = n_params * 4 * 4
    print(f"  optimizer mem (rough): {human_bytes(budget)}")
    print()


def mps_mem():
    if hasattr(torch.mps, "current_allocated_memory"):
        cur = torch.mps.current_allocated_memory()
        drv = torch.mps.driver_allocated_memory()
        return cur, drv
    return 0, 0


# ---------- run ----------
def main():
    report_env()
    if not torch.backends.mps.is_available():
        print("MPS not available — aborting.")
        return

    device = torch.device("mps")
    torch.manual_seed(0)

    print("Building model on MPS ...")
    model = TinyDecoder(VOCAB, D_MODEL, N_HEADS, N_LAYERS, FFN_MULT, P_LENGTH).to(device)
    model.train()
    report_config(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    cur, drv = mps_mem()
    print(f"After model+opt build : allocated={human_bytes(cur)}  driver={human_bytes(drv)}")
    print()

    # one warmup forward to JIT-compile MPS kernels
    print("Warmup pass (compiles MPS kernels — may be slow) ...")
    x = torch.randint(0, VOCAB, (BATCH, P_LENGTH), device=device)
    y = torch.randint(0, VOCAB, (BATCH, P_LENGTH), device=device)
    t0 = time.time()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    print(f"  warmup wall time   : {time.time()-t0:.2f}s   loss={loss.item():.3f}")
    cur, drv = mps_mem()
    print(f"  after warmup       : allocated={human_bytes(cur)}  driver={human_bytes(drv)}")
    print()

    # timed accumulation step
    print(f"Timing one full accumulation step ({ACCUM} micro-batches) ...")
    gc.collect()
    torch.mps.synchronize()
    t0 = time.time()
    micro_times = []
    for i in range(ACCUM):
        ti = time.time()
        x = torch.randint(0, VOCAB, (BATCH, P_LENGTH), device=device)
        y = torch.randint(0, VOCAB, (BATCH, P_LENGTH), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1)) / ACCUM
        loss.backward()
        torch.mps.synchronize()
        micro_times.append(time.time() - ti)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    step_time = time.time() - t0
    print(f"  per micro-batch    : avg {sum(micro_times)/len(micro_times):.2f}s "
          f"min {min(micro_times):.2f}s max {max(micro_times):.2f}s")
    print(f"  full step          : {step_time:.2f}s")
    cur, drv = mps_mem()
    print(f"  peak allocated     : {human_bytes(cur)}")
    print(f"  peak driver        : {human_bytes(drv)}")
    print()

    # extrapolation
    n_train_files = 590 * 3  # with 3x augmentation
    steps_per_epoch = n_train_files // (BATCH * ACCUM)
    print("=" * 60)
    print("EXTRAPOLATION (assuming similar workload to NotaGen-small)")
    print("=" * 60)
    print(f"  files (aug 3x)     : {n_train_files}")
    print(f"  steps / epoch      : {steps_per_epoch}")
    print(f"  time / epoch       : {steps_per_epoch * step_time / 60:.1f} min")
    print(f"  time / 30 epochs   : {steps_per_epoch * step_time * 30 / 3600:.1f} hours")
    print(f"  time / 50 epochs   : {steps_per_epoch * step_time * 50 / 3600:.1f} hours")
    print()
    print("VERDICT")
    if step_time < 30 and drv < 14 * 1024**3:
        print("  PASS — fine-tune is feasible on this machine.")
    elif step_time < 60 and drv < 16 * 1024**3:
        print("  MARGINAL — feasible but slow; consider reducing P_LENGTH to 512.")
    else:
        print("  FAIL — too slow or too close to memory limit; reduce model/seqlen.")


if __name__ == "__main__":
    main()
