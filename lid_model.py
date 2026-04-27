"""
lid_model.py  — Task 1.1
========================
Multi-Head Frame-Level Language Identification (LID) for Code-Switched Speech.

Architecture
────────────
  Input : Mel-spectrogram  [B, T, n_mels]
  ↓
  CNN Encoder         – 3-layer Conv1d stack; extracts local spectro-temporal
                         patterns from overlapping mel frames.
  ↓
  Positional Encoding – sinusoidal, added to CNN output [B, T', d_model]
  ↓
  Transformer Encoder – L stacked self-attention + FFN layers
                         (shared, language-agnostic representation)
  ↓
  Multi-Head Outputs  – three parallel linear classifiers:
      Head-A : 4-class frame label  (English | Hindi | Mixed | Silence)
      Head-B : binary English vs rest  → for English F1 computation
      Head-C : binary Hindi   vs rest  → for Hindi   F1 computation

Labels (Head-A)
───────────────
  0 – English
  1 – Hindi
  2 – Mixed / Code-switch
  3 – Silence / Non-speech

Usage
─────
    from lid_model import MultiHeadLID, LIDConfig

    cfg   = LIDConfig()
    model = MultiHeadLID(cfg)
    mel   = torch.randn(2, 300, 80)          # [B, T, n_mels]
    out   = model(mel)
    # out.logits_4cls  [B, T', 4]
    # out.logits_en    [B, T', 2]
    # out.logits_hi    [B, T', 2]
    # out.labels_4cls  [B, T']   (argmax of 4-class head)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LIDConfig:
    # Acoustic features
    n_mels: int = 80
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    f_min: float = 0.0
    f_max: float = 8_000.0

    # CNN encoder
    cnn_channels: List[int] = field(default_factory=lambda: [256, 256, 512])
    cnn_kernels:  List[int] = field(default_factory=lambda: [5, 3, 3])
    cnn_strides:  List[int] = field(default_factory=lambda: [1, 1, 1])
    cnn_dropout:  float = 0.1

    # Transformer encoder
    d_model:    int = 512
    n_heads:    int = 8
    n_layers:   int = 4
    ffn_dim:    int = 2048
    attn_drop:  float = 0.1
    ffn_drop:   float = 0.1

    # Classification
    n_classes:  int = 4        # English | Hindi | Mixed | Silence

    # Training
    label_smoothing: float = 0.05


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class LIDOutput:
    logits_4cls: torch.Tensor        # [B, T', 4]
    logits_en:   torch.Tensor        # [B, T', 2]
    logits_hi:   torch.Tensor        # [B, T', 2]

    @property
    def labels_4cls(self) -> torch.Tensor:
        return self.logits_4cls.argmax(dim=-1)   # [B, T']

    @property
    def english_probs(self) -> torch.Tensor:
        return F.softmax(self.logits_en, dim=-1)[..., 1]   # [B, T']

    @property
    def hindi_probs(self) -> torch.Tensor:
        return F.softmax(self.logits_hi, dim=-1)[..., 1]   # [B, T']


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, T, d_model]"""
        return self.dropout(x + self.pe[:, : x.shape[1]])


# ---------------------------------------------------------------------------
# CNN Encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    Stack of 1-D depthwise-separable convolutions along the time axis.
    Input  : [B, T, n_mels]
    Output : [B, T', d_model]   (T' ≤ T depending on strides)
    """

    def __init__(self, cfg: LIDConfig) -> None:
        super().__init__()
        in_ch = cfg.n_mels
        layers: List[nn.Module] = []

        for out_ch, k, s in zip(
            cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides
        ):
            layers += [
                # depthwise
                nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s,
                          padding=k // 2, groups=in_ch, bias=False),
                # pointwise
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(cfg.cnn_dropout),
            ]
            in_ch = out_ch

        self.layers = nn.Sequential(*layers)
        self.proj = nn.Linear(in_ch, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, T, n_mels]  → [B, T', d_model]"""
        x = x.transpose(1, 2)          # [B, n_mels, T]
        x = self.layers(x)             # [B, C_last, T']
        x = x.transpose(1, 2)         # [B, T', C_last]
        return self.proj(x)            # [B, T', d_model]


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 attn_drop: float, ffn_drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(ffn_drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-Norm (stabilises training)
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=src_key_padding_mask,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Multi-Head Language Identification Model
# ---------------------------------------------------------------------------

class MultiHeadLID(nn.Module):
    """
    Multi-head frame-level Language Identification model.

    Three parallel classification heads share a single CNN + Transformer
    encoder:
        Head-A (4-class)  : English / Hindi / Mixed / Silence
        Head-B (binary)   : English vs rest
        Head-C (binary)   : Hindi vs rest

    Parameters
    ----------
    cfg : LIDConfig  – all hyperparameters.
    """

    LABEL_NAMES = ["English", "Hindi", "Mixed", "Silence"]

    def __init__(self, cfg: LIDConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Shared encoder ────────────────────────────────────────────────
        self.cnn = CNNEncoder(cfg)
        self.pe  = SinusoidalPE(cfg.d_model)
        self.transformer = nn.ModuleList([
            TransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.ffn_dim,
                cfg.attn_drop, cfg.ffn_drop,
            )
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

        # ── Classification heads ─────────────────────────────────────────
        def _head(n_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.d_model // 2, n_out),
            )

        self.head_4cls = _head(cfg.n_classes)   # Head-A: 4-class
        self.head_en   = _head(2)               # Head-B: English binary
        self.head_hi   = _head(2)               # Head-C: Hindi binary

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(
        self,
        mel: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shared encoder forward pass.

        Parameters
        ----------
        mel          : [B, T, n_mels]
        padding_mask : [B, T'] bool  True = pad position  (optional)

        Returns
        -------
        z : [B, T', d_model]  contextualised frame representations
        """
        x = self.cnn(mel)           # [B, T', d_model]
        x = self.pe(x)              # + positional encoding

        for block in self.transformer:
            x = block(x, src_key_padding_mask=padding_mask)

        return self.norm(x)         # [B, T', d_model]

    def forward(
        self,
        mel: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> LIDOutput:
        """
        Parameters
        ----------
        mel          : [B, T, n_mels]  mel-spectrogram (not log-mel yet;
                       apply log inside if needed — see MelFeatureExtractor).
        padding_mask : [B, T'] bool  True = padding frame.

        Returns
        -------
        LIDOutput  with logits for all three heads.
        """
        z = self.encode(mel, padding_mask)     # [B, T', d_model]

        return LIDOutput(
            logits_4cls = self.head_4cls(z),   # [B, T', 4]
            logits_en   = self.head_en(z),     # [B, T', 2]
            logits_hi   = self.head_hi(z),     # [B, T', 2]
        )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class MultiHeadLIDLoss(nn.Module):
    """
    Combined cross-entropy loss for all three heads.

    label_4cls : [B, T']  with values in {0, 1, 2, 3}
    label_en   : [B, T']  1 = English frame,  0 = otherwise
    label_hi   : [B, T']  1 = Hindi   frame,  0 = otherwise

    The 4-class head carries the primary weight; the two binary heads
    act as auxiliary losses that enforce specialisation.

    Parameters
    ----------
    w_4cls   : Weight for the 4-class head.
    w_en, w_hi : Weights for binary auxiliary heads.
    smoothing  : Label smoothing (reduces overconfidence).
    """

    def __init__(
        self,
        w_4cls: float = 1.0,
        w_en:   float = 0.3,
        w_hi:   float = 0.3,
        smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.w_4cls = w_4cls
        self.w_en   = w_en
        self.w_hi   = w_hi
        self.ce_4cls = nn.CrossEntropyLoss(
            label_smoothing=smoothing, ignore_index=-1
        )
        self.ce_bin  = nn.CrossEntropyLoss(
            label_smoothing=smoothing, ignore_index=-1
        )

    def forward(
        self,
        out: LIDOutput,
        label_4cls: torch.Tensor,
        label_en:   torch.Tensor,
        label_hi:   torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, {'loss_4cls': ..., 'loss_en': ..., 'loss_hi': ...})
        """
        B, T, _ = out.logits_4cls.shape

        # Flatten over batch × time
        l4  = out.logits_4cls.reshape(-1, 4)
        len_ = out.logits_en.reshape(-1, 2)
        lhi = out.logits_hi.reshape(-1, 2)
        y4  = label_4cls.reshape(-1)
        yen = label_en.reshape(-1)
        yhi = label_hi.reshape(-1)

        loss_4 = self.ce_4cls(l4,  y4)
        loss_e = self.ce_bin(len_, yen)
        loss_h = self.ce_bin(lhi,  yhi)

        total = self.w_4cls * loss_4 + self.w_en * loss_e + self.w_hi * loss_h
        return total, {
            "loss_4cls": loss_4.item(),
            "loss_en":   loss_e.item(),
            "loss_hi":   loss_h.item(),
        }


# ---------------------------------------------------------------------------
# Mel Feature Extractor (torchaudio-based, used inside the pipeline)
# ---------------------------------------------------------------------------

class MelFeatureExtractor(nn.Module):
    """
    Computes log-mel spectrogram from raw waveform.
    Wraps torchaudio.transforms.MelSpectrogram.

    Parameters
    ----------
    cfg : LIDConfig

    Returns [B, T, n_mels] log-mel spectrogram frames.
    """

    def __init__(self, cfg: LIDConfig) -> None:
        super().__init__()
        try:
            import torchaudio.transforms as Ta
        except ImportError:
            raise ImportError("torchaudio is required for MelFeatureExtractor")

        self.mel_transform = Ta.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
        )
        self.amplitude_to_db = Ta.AmplitudeToDB(stype="power", top_db=80.0)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform : [B, 1, T] or [B, T]  at cfg.sample_rate

        Returns  : [B, T', n_mels]  log-mel feature tensor
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)          # [B, T]
        mel = self.mel_transform(waveform)          # [B, n_mels, T']
        log_mel = self.amplitude_to_db(mel)         # [B, n_mels, T']
        # Normalise to roughly zero-mean unit-variance
        log_mel = (log_mel + 40.0) / 40.0
        return log_mel.transpose(1, 2)              # [B, T', n_mels]


# ---------------------------------------------------------------------------
# Inference helper – frame-to-segment decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_language_segments(
    model: MultiHeadLID,
    feature_extractor: MelFeatureExtractor,
    waveform: torch.Tensor,          # [1, T]
    sample_rate: int = 16_000,
    min_segment_frames: int = 5,     # merge tiny segments
) -> List[dict]:
    """
    Run LID on a mono waveform and return a list of language segments.

    Returns
    -------
    List of dicts:
        {
          "start_s"  : float,    # segment start in seconds
          "end_s"    : float,    # segment end   in seconds
          "label"    : str,      # 'English' | 'Hindi' | 'Mixed' | 'Silence'
          "label_id" : int,
          "en_prob"  : float,    # mean English probability over segment
          "hi_prob"  : float,    # mean Hindi   probability over segment
        }
    """
    model.eval()
    device = next(model.parameters()).device
    wav = waveform.to(device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)           # [1, T]

    mel = feature_extractor(wav.unsqueeze(0))  # [1, T', n_mels]
    out = model(mel)

    labels   = out.labels_4cls[0].cpu().tolist()    # [T']
    en_probs = out.english_probs[0].cpu().tolist()
    hi_probs = out.hindi_probs[0].cpu().tolist()

    # Compute seconds per frame
    hop_s = model.cfg.hop_length / sample_rate

    # Merge consecutive same-label frames into segments
    segments = []
    i = 0
    while i < len(labels):
        j = i
        cur_label = labels[i]
        while j < len(labels) and labels[j] == cur_label:
            j += 1
        if j - i >= min_segment_frames:
            segments.append({
                "start_s":  i * hop_s,
                "end_s":    j * hop_s,
                "label":    MultiHeadLID.LABEL_NAMES[cur_label],
                "label_id": cur_label,
                "en_prob":  float(sum(en_probs[i:j]) / (j - i)),
                "hi_prob":  float(sum(hi_probs[i:j]) / (j - i)),
            })
        i = j

    return segments


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total params: {total:,}  |  Trainable: {trainable:,}"


# ---------------------------------------------------------------------------
# Quick architecture smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = LIDConfig()
    model = MultiHeadLID(cfg)
    print(count_parameters(model))

    # Dummy input: batch=2, 300 mel frames, 80 mel bins
    mel = torch.randn(2, 300, cfg.n_mels)
    out = model(mel)

    print(f"logits_4cls : {tuple(out.logits_4cls.shape)}")
    print(f"logits_en   : {tuple(out.logits_en.shape)}")
    print(f"logits_hi   : {tuple(out.logits_hi.shape)}")
    print(f"labels_4cls : {tuple(out.labels_4cls.shape)}")
    print(f"english_prob sample : {out.english_probs[0, :5].tolist()}")
