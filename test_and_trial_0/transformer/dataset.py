import csv
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from config import TokenizerConfig, TrainConfig, repo_root
from tokenizer import MidiTokenizer


class GuzhengDataset(Dataset):
    """Chunks note streams so every sample starts with BOS + KEY(scale).

    Full sequences are [BOS, KEY, ...note tokens..., EOS]; sliding windows only
    apply to the note stream, then prefix [BOS, KEY] is reattached.
    """

    def __init__(
        self,
        sequences: List[List[int]],
        context_length: int,
        stride: int,
        tok_cfg: TokenizerConfig,
    ):
        self.seq_len = context_length + 1
        self.pad_token = tok_cfg.pad_token
        self.bos_token = tok_cfg.bos_token
        self.eos_token = tok_cfg.eos_token
        self.cfg = tok_cfg
        self.prefix_len = 2
        self.body_len = self.seq_len - self.prefix_len
        self.chunks: List[List[int]] = []

        for seq in sequences:
            if (
                len(seq) < self.prefix_len + 1
                or seq[0] != self.bos_token
                or not tok_cfg.is_key_token_id(seq[1])
            ):
                raise ValueError("Expected each sequence to begin with [BOS, KEY, ...]")

            key_tok = seq[1]
            if seq[-1] == self.eos_token:
                core = seq[self.prefix_len : -1]
            else:
                core = seq[self.prefix_len :]

            if len(core) <= self.body_len:
                body = core + [self.pad_token] * (self.body_len - len(core))
                self.chunks.append([self.bos_token, key_tok] + body[: self.body_len])
            else:
                for start in range(0, len(core) - self.body_len + 1, stride):
                    self.chunks.append(
                        [self.bos_token, key_tok]
                        + core[start : start + self.body_len]
                    )
                tail_start = len(core) - self.body_len
                if tail_start % stride != 0:
                    self.chunks.append(
                        [self.bos_token, key_tok] + core[tail_start:]
                    )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        if len(chunk) < self.seq_len:
            chunk = chunk + [self.pad_token] * (self.seq_len - len(chunk))
        t = torch.tensor(chunk, dtype=torch.long)
        return t[:-1], t[1:]


def _piece_base_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    for suffix in ("_A", "_C", "_D", "_F", "_G"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _npy_basename_to_mid(file_base_name: str) -> str:
    """MIDI_transposed_shang_lou_D.npy -> shang_lou_D.mid"""
    base = os.path.basename(file_base_name)
    prefix = "MIDI_transposed_"
    if base.startswith(prefix) and base.endswith(".npy"):
        return base[len(prefix) : -4] + ".mid"
    if base.endswith(".npy"):
        return base[:-4] + ".mid"
    return base


def _load_official_split(split_csv_path: str) -> Tuple[Set[str], Set[str]]:
    train_mids: Set[str] = set()
    val_mids: Set[str] = set()
    with open(split_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = _npy_basename_to_mid(row["file_base_name"])
            split = row["split"].strip().lower()
            if split == "train":
                train_mids.add(mid)
            elif split == "test":
                val_mids.add(mid)
    return train_mids, val_mids


def _resolve_split_csv(train_cfg: TrainConfig) -> Optional[str]:
    if train_cfg.split_csv:
        return train_cfg.split_csv if os.path.isfile(train_cfg.split_csv) else None
    if not train_cfg.use_official_split:
        return None
    candidate = os.path.join(
        repo_root(), "data", "moonbeam_preprocessed", "train_test_split.csv"
    )
    return candidate if os.path.isfile(candidate) else None


def create_datasets(
    train_cfg: TrainConfig, tok_cfg: TokenizerConfig
) -> Tuple["GuzhengDataset", "GuzhengDataset", MidiTokenizer]:
    tokenizer = MidiTokenizer(tok_cfg)
    midi_dir = os.path.abspath(train_cfg.midi_dir)

    files = sorted(f for f in os.listdir(midi_dir) if f.endswith(".mid"))

    split_path = _resolve_split_csv(train_cfg)
    val_mids: Set[str]

    if split_path:
        train_mids_official, val_mids = _load_official_split(split_path)
        print(f"Using official split: {split_path}")
        listed = train_mids_official | val_mids
        extra = [f for f in files if f not in listed]
        if extra:
            print(
                f"  {len(extra)} MIDI file(s) not in split CSV — assigning to train: "
                f"{extra[:5]}{'...' if len(extra) > 5 else ''}"
            )
    else:
        groups: Dict[str, List[str]] = {}
        for f in files:
            groups.setdefault(_piece_base_name(f), []).append(f)

        bases = sorted(groups.keys())
        random.seed(train_cfg.seed)
        random.shuffle(bases)

        n_val = max(1, int(len(bases) * train_cfg.val_split))
        val_bases = set(bases[:n_val])
        val_mids = set()
        for f in files:
            if _piece_base_name(f) in val_bases:
                val_mids.add(f)
        print(f"Pieces: {len(bases)} unique, {len(files)} files")
        print(f"Train/val by piece: val pieces {sorted(val_bases)}")

    train_seqs: List[List[int]] = []
    val_seqs: List[List[int]] = []
    for f in files:
        tokens = tokenizer.encode_midi(os.path.join(midi_dir, f))
        if split_path:
            (val_seqs if f in val_mids else train_seqs).append(tokens)
        else:
            (val_seqs if f in val_mids else train_seqs).append(tokens)

    print(
        f"Train: {len(train_seqs)} seqs, {sum(len(s) for s in train_seqs)} tokens | "
        f"Val: {len(val_seqs)} seqs, {sum(len(s) for s in val_seqs)} tokens"
    )

    train_ds = GuzhengDataset(train_seqs, train_cfg.context_length, train_cfg.stride, tok_cfg)
    val_ds = GuzhengDataset(val_seqs, train_cfg.context_length, train_cfg.stride, tok_cfg)
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer
