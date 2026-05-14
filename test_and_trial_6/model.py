"""Decoder-only transformer for guzheng MIDI generation."""

from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TokenizerConfig


def allowed_token_ids_for_step(
    cur_len: int,
    prefix_len: int,
    cfg: TokenizerConfig,
    pitch_subset: Optional[Set[int]],
) -> List[int]:
    """After BOS+KEY, cycle: time_shift, pitch, duration, velocity (+EOS on last)."""
    pos = (cur_len - prefix_len) % 4
    if pos == 0:
        return list(range(cfg.time_shift_offset, cfg.pitch_offset))
    if pos == 1:
        ids = list(range(cfg.pitch_offset, cfg.duration_offset))
        if pitch_subset is not None:
            ids = sorted(set(ids) & pitch_subset)
        return ids
    if pos == 2:
        return list(range(cfg.duration_offset, cfg.velocity_offset))
    ids = list(
        range(cfg.velocity_offset, cfg.velocity_offset + cfg.num_velocity_bins)
    )
    ids.append(cfg.eos_token)
    return ids


class GuzhengTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 2048,
        dropout: float = 0.15,
        pad_token: int = 0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len
        self.label_smoothing = label_smoothing

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.drop(self.token_emb(x) + self.pos_emb(positions))

        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        h = self.transformer(h, mask=mask)
        h = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_token,
                label_smoothing=self.label_smoothing,
            )
        return logits, loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _layer_forward_cached(self, layer, h, cache_k, cache_v):
        """Manual forward through one TransformerEncoderLayer (norm_first=True)
        with optional KV cache.

        h: (B, T_new, d_model) — new tokens to process
        cache_k, cache_v: prior K/V of shape (B, n_heads, T_prev, head_dim) or None
        Returns: (out, new_k, new_v)
        """
        attn = layer.self_attn
        n_heads = attn.num_heads
        B, T_new, d_model = h.shape
        head_dim = d_model // n_heads

        # --- Self-attention sub-block (norm_first) ---
        x = layer.norm1(h)
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T_new, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T_new, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T_new, n_heads, head_dim).transpose(1, 2)

        if cache_k is not None:
            k_full = torch.cat([cache_k, k], dim=2)
            v_full = torch.cat([cache_v, v], dim=2)
        else:
            k_full = k
            v_full = v

        # Causal mask only matters when processing >1 new token (the prompt).
        # For single-token incremental steps the new query is at the last
        # position, so it can attend to all cached keys without a mask.
        is_causal = T_new > 1
        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=is_causal
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_new, d_model)
        attn_out = attn.out_proj(attn_out)
        h = h + attn_out

        # --- Feed-forward sub-block (norm_first) ---
        y = layer.norm2(h)
        y = layer.linear2(F.gelu(layer.linear1(y)))
        h = h + y

        return h, k_full, v_full

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        eos_token=2,
        tok_cfg: Optional[TokenizerConfig] = None,
        allowed_pitch_token_ids: Optional[Set[int]] = None,
        prefix_len: int = 2,
    ):
        self.eval()
        device = next(self.parameters()).device

        if isinstance(prompt, list):
            tokens = torch.tensor([prompt], dtype=torch.long, device=device)
        else:
            tokens = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
            tokens = tokens.to(device)

        # --- Initial pass over the prompt: fill KV caches ---
        T = tokens.shape[1]
        positions = torch.arange(T, device=device)
        h = self.token_emb(tokens) + self.pos_emb(positions)
        caches = []
        for layer in self.transformer.layers:
            h, k, v = self._layer_forward_cached(layer, h, None, None)
            caches.append((k, v))
        h = self.ln_f(h)
        last_logits = self.head(h[:, -1])  # (1, vocab)

        # --- Incremental sampling loop ---
        for _ in range(max_new_tokens):
            row = last_logits[0] / max(temperature, 1e-8)

            cur_len = tokens.shape[1]
            if tok_cfg is not None and cur_len >= prefix_len:
                allow = allowed_token_ids_for_step(
                    cur_len, prefix_len, tok_cfg, allowed_pitch_token_ids
                )
            else:
                allow = list(range(row.numel()))

            idx_t = torch.tensor(allow, device=device, dtype=torch.long)
            sub = row[idx_t]
            sub = torch.nan_to_num(sub, nan=-1e9, posinf=1e9, neginf=-1e9)

            if top_k > 0 and top_k < sub.numel():
                tk = min(top_k, int(sub.numel()))
                topv, _ = torch.topk(sub, tk)
                thr = topv[-1]
                sub = torch.where(sub < thr, torch.full_like(sub, float("-inf")), sub)

            if top_p < 1.0:
                sorted_logits, sorted_local = torch.sort(sub, descending=True)
                cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                sorted_logits = sorted_logits.clone()
                sorted_logits[remove] = float("-inf")
                sub = sub.clone()
                sub.scatter_(0, sorted_local, sorted_logits)

            probs = F.softmax(sub, dim=-1)
            if probs.sum() < 1e-8 or torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.numel()
            pick = torch.multinomial(probs, num_samples=1).item()
            nxt_id = int(allow[pick])
            nxt = torch.tensor([[nxt_id]], dtype=torch.long, device=device)

            tokens = torch.cat([tokens, nxt], dim=1)

            if nxt_id == eos_token:
                break

            # Stop if we'd run out of position embeddings.
            cur_pos = tokens.shape[1] - 1
            if cur_pos >= self.max_seq_len:
                break

            # Incremental forward for the new token, updating caches.
            pos_t = torch.tensor([cur_pos], device=device)
            h = self.token_emb(nxt) + self.pos_emb(pos_t)
            new_caches = []
            for layer, (ck, cv) in zip(self.transformer.layers, caches):
                h, k, v = self._layer_forward_cached(layer, h, ck, cv)
                new_caches.append((k, v))
            caches = new_caches
            h = self.ln_f(h)
            last_logits = self.head(h[:, -1])

        return tokens[0].tolist()
