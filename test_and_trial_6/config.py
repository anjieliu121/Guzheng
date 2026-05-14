"""Configuration for the Trial 6 guzheng transformer experiment.

Differences from Trial 1:
- tick_resolution 10 -> 20  (fixes Trial 1's note-density-too-high problem)
- max_time_shift 200 -> 100 (window halved because resolution doubled)
- max_duration 400 -> 200   (window halved because resolution doubled)
- model d_model 256 -> 320, d_ff 512 -> 1280  (modest scale-up to ~7M params)
- batch_size 16, dropout 0.15, weight_decay 0.05, label_smoothing 0.1 (same as t1)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


def trial_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def repo_root() -> str:
    return os.path.dirname(trial_root())


@dataclass
class TokenizerConfig:
    """MIDI-to-token vocabulary layout.

    Sequence: BOS  KEY(scale)  (TIME_SHIFT PITCH DURATION VELOCITY)*  EOS
    """

    tick_resolution: int = 20            # was 10 in Trial 1
    max_time_shift: int = 100            # was 200 in Trial 1
    max_duration: int = 200              # was 400 in Trial 1
    num_velocity_bins: int = 32
    num_pitches: int = 128
    key_scale_letters: Tuple[str, ...] = ("A", "C", "D", "F", "G")

    @property
    def pad_token(self):
        return 0

    @property
    def bos_token(self):
        return 1

    @property
    def eos_token(self):
        return 2

    @property
    def first_key_token(self) -> int:
        return 3

    @property
    def num_key_tokens(self) -> int:
        return len(self.key_scale_letters)

    @property
    def last_key_token(self) -> int:
        return self.first_key_token + self.num_key_tokens - 1

    def is_key_token_id(self, tid: int) -> bool:
        return self.first_key_token <= tid <= self.last_key_token

    def key_token_id(self, scale: str) -> int:
        scale = scale.upper()
        if scale not in self.key_scale_letters:
            scale = "D"
        return self.first_key_token + self.key_scale_letters.index(scale)

    def scale_from_key_token(self, tid: int) -> Optional[str]:
        if self.is_key_token_id(tid):
            return self.key_scale_letters[tid - self.first_key_token]
        return None

    @property
    def time_shift_offset(self):
        return self.first_key_token + self.num_key_tokens

    @property
    def pitch_offset(self):
        return self.time_shift_offset + self.max_time_shift + 1

    @property
    def duration_offset(self):
        return self.pitch_offset + self.num_pitches

    @property
    def velocity_offset(self):
        return self.duration_offset + self.max_duration

    @property
    def vocab_size(self):
        return self.velocity_offset + self.num_velocity_bins


@dataclass
class ModelConfig:
    # Trial 1: 256 / 4 / 6 / 512  -> 3.9M params
    # Trial 6: 320 / 5 / 6 / 1280 -> ~7M params
    d_model: int = 320
    n_heads: int = 5
    n_layers: int = 6
    d_ff: int = 1280
    max_seq_len: int = 2048
    dropout: float = 0.15


@dataclass
class TrainConfig:
    midi_dir: str = field(
        default_factory=lambda: os.path.join(trial_root(), "data", "train")
    )
    val_dir: str = field(
        default_factory=lambda: os.path.join(trial_root(), "data", "val")
    )
    output_dir: str = field(
        default_factory=lambda: trial_root()
    )
    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 200
    warmup_steps: int = 500
    context_length: int = 512
    stride: int = 256
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    early_stopping_patience: int = 20
    save_every: int = 20
    log_every: int = 10
    seed: int = 42
