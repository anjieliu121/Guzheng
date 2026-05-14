import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


def transformer_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def repo_root() -> str:
    return os.path.dirname(transformer_root())


@dataclass
class TokenizerConfig:
    """MIDI-to-token vocabulary layout.

    Sequence structure: BOS  KEY(scale)  (TIME_SHIFT PITCH DURATION VELOCITY)*  EOS

    Token ID ranges (contiguous):
        0            PAD
        1            BOS
        2            EOS
        3 .. 3+K-1   KEY (one token per entry in key_scale_letters, default A C D F G)
        T .. T+S     TIME_SHIFT
        P .. P+127   PITCH (MIDI 0-127)
        D .. D+M-1   DURATION
        V .. V+B-1   VELOCITY
    """

    tick_resolution: int = 10
    max_time_shift: int = 200
    max_duration: int = 400
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
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 2048
    dropout: float = 0.1


@dataclass
class TrainConfig:
    midi_dir: str = field(
        default_factory=lambda: os.path.join(repo_root(), "MIDI_transposed")
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(transformer_root(), "output")
    )
    # If set, use this CSV (file_base_name,split,...) for train/test; if None and
    # use_official_split is True, uses data/moonbeam_preprocessed/train_test_split.csv when present.
    split_csv: Optional[str] = None
    use_official_split: bool = True
    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 200
    warmup_steps: int = 200
    context_length: int = 512
    stride: int = 256
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_split: float = 0.15
    save_every: int = 20
    log_every: int = 10
    seed: int = 42
