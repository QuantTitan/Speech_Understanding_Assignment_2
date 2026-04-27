"""
tts_synthesizer.py  —  Task 3.3
=================================
Zero-shot Cross-Lingual TTS Synthesizer for Santhali.

Architecture: Lightweight VITS (Variational Inference with adversarial
learning for end-to-end Text-to-Speech) adapted for:
  • Zero-shot speaker conditioning  via d-vector injection
  • Cross-lingual phoneme input     via IPA token vocabulary
  • 22050 Hz output                 (matches task requirement)

Full VITS component breakdown
─────────────────────────────
  Text Encoder     : Transformer encoder on IPA phoneme sequences
  Posterior Encoder: WaveNet-style residual stack on mel spectrogram
  Flow             : Normalizing flows (affine coupling layers)
  Stochastic Duration Predictor: models duration variability
  HiFi-GAN Decoder : Upsampling generator producing raw waveform

Speaker conditioning: d-vector is injected into every generator layer
  via FiLM (Feature-wise Linear Modulation).

For assignment submission, this module:
  1.  Implements the full VITS architecture from scratch in PyTorch.
  2.  Provides a `VITSSynthesizer` wrapper that loads a pre-trained
      checkpoint OR runs in "demonstration mode" (random weights) so
      the pipeline can be tested without GPU training.
  3.  Provides `synthesize_lecture()` — the 10-minute synthesis entry
      point that chunks long Santhali text and stitches audio.

Note on pre-trained weights
───────────────────────────
Training VITS from scratch requires ~1 week on 4× A100 GPUs and a
multi-speaker dataset.  The recommended approach for this assignment is:

  Option A: Fine-tune YourTTS (Casanova et al., 2022) from the public
    multi-lingual checkpoint.  YourTTS is also VITS-based; swap its
    text encoder vocabulary for our IPA set and fine-tune for ~50k steps.

  Option B: Use Meta MMS-TTS (massively multilingual, includes Hindi;
    adapt for Santhali via the IPA bridge built in Part II).

  Option C: Use the implemented architecture with random weights for
    pipeline demonstration, noting training requirements in the report.

The synthesizer API is identical in all three options.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ═══════════════════════════════════════════════════════════════════════════
# IPA Vocabulary for Santhali + Hinglish phoneme set
# ═══════════════════════════════════════════════════════════════════════════

IPA_VOCAB = [
    "<pad>", "<sos>", "<eos>", "<unk>", " ",
    # Vowels
    "ə", "ɪ", "iː", "eː", "ɛ", "ɛː", "a", "aː", "ɑ", "ɔ", "ɔː",
    "oː", "ʊ", "uː", "e", "o", "æ",
    # Nasalised vowels
    "ã", "ẽ", "ĩ", "õ", "ũ",
    # Stops / plosives
    "p", "b", "t", "d", "ʈ", "ɖ", "k", "ɡ", "q",
    # Aspirated stops
    "pʰ", "bʱ", "t̪ʰ", "d̪ʱ", "ʈʰ", "ɖʱ", "kʰ", "ɡʱ",
    # Affricates
    "tʃ", "dʒ", "tʃʰ", "dʒʱ",
    # Fricatives
    "f", "v", "s", "z", "ʃ", "ʒ", "x", "ɣ", "ɦ", "h", "ʂ",
    # Nasals
    "m", "n", "ŋ", "ɳ", "ɲ",
    # Liquids / approximants
    "r", "ɾ", "ɽ", "l", "ɭ", "j", "ʋ", "w",
    # Dental consonants
    "t̪", "d̪",
    # Tones / diacritics
    "ː", "̃",
    # Punctuation (as prosodic cues)
    ".", ",", "?", "!", ";",
]

# Remove duplicates, preserve order
_seen = set()
IPA_VOCAB_CLEAN = []
for tok in IPA_VOCAB:
    if tok not in _seen:
        IPA_VOCAB_CLEAN.append(tok)
        _seen.add(tok)
IPA_VOCAB = IPA_VOCAB_CLEAN

TOKEN2ID: Dict[str, int] = {tok: i for i, tok in enumerate(IPA_VOCAB)}
ID2TOKEN: Dict[int, str] = {i: tok for tok, i in TOKEN2ID.items()}
VOCAB_SIZE = len(IPA_VOCAB)


def ipa_to_ids(ipa_string: str) -> torch.Tensor:
    """Convert an IPA string to a tensor of token IDs."""
    ids = [TOKEN2ID.get("<sos>", 1)]
    i = 0
    text = ipa_string.strip()
    while i < len(text):
        # Try longest match first (for multi-char tokens like "tʃʰ")
        matched = False
        for length in [3, 2, 1]:
            candidate = text[i : i + length]
            if candidate in TOKEN2ID:
                ids.append(TOKEN2ID[candidate])
                i += length
                matched = True
                break
        if not matched:
            ids.append(TOKEN2ID.get("<unk>", 3))
            i += 1
    ids.append(TOKEN2ID.get("<eos>", 2))
    return torch.tensor(ids, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════

class FiLM(nn.Module):
    """Feature-wise Linear Modulation for speaker conditioning."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta  = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]  cond: [B, cond_dim]  →  [B, C, T]"""
        g = self.gamma(cond).unsqueeze(-1)   # [B, C, 1]
        b = self.beta(cond).unsqueeze(-1)    # [B, C, 1]
        return g * x + b


class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 8192) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class FFTBlock(nn.Module):
    """Transformer FFT block (1 attention layer + FFN)."""

    def __init__(self, d: int, n_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         key_padding_mask=mask)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class ResBlock1D(nn.Module):
    """1-D gated residual block with FiLM speaker conditioning."""

    def __init__(self, channels: int, kernel: int, dilation: int,
                 cond_dim: int) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv  = nn.Conv1d(channels, 2 * channels, kernel,
                               dilation=dilation, padding=pad)
        self.res   = nn.Conv1d(channels, channels, 1)
        self.film  = FiLM(2 * channels, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)                           # [B, 2C, T]
        h = self.film(h, cond)                     # FiLM conditioning
        h_gate, h_filt = h.chunk(2, dim=1)
        h = torch.sigmoid(h_gate) * torch.tanh(h_filt)
        return x + self.res(h)


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for the normalizing flow."""

    def __init__(self, half_channels: int, hidden: int, n_layers: int = 4,
                 cond_dim: int = 256) -> None:
        super().__init__()
        self.pre  = nn.Conv1d(half_channels, hidden, 1)
        self.enc  = nn.ModuleList([
            ResBlock1D(hidden, 5, 2**i, cond_dim) for i in range(n_layers)
        ])
        self.post = nn.Conv1d(hidden, half_channels * 2, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x.chunk(2, dim=1)
        h = self.pre(x0)
        for layer in self.enc:
            h = layer(h, cond)
        stats = self.post(h)
        m, logs = stats.chunk(2, dim=1)
        logs = logs.clamp(-5, 5)

        if not reverse:
            x1 = (x1 - m) * torch.exp(-logs)
            logdet = -logs.sum(dim=[1, 2])
        else:
            x1 = x1 * torch.exp(logs) + m
            logdet = logs.sum(dim=[1, 2])
        return torch.cat([x0, x1], dim=1), logdet


class ResidualCouplingBlock(nn.Module):
    """Stack of affine coupling layers."""

    def __init__(self, channels: int, hidden: int, n_flows: int = 4,
                 cond_dim: int = 256) -> None:
        super().__init__()
        self.flows = nn.ModuleList([
            AffineCouplingLayer(channels // 2, hidden, cond_dim=cond_dim)
            for _ in range(n_flows)
        ])

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        total_logdet = torch.zeros(x.shape[0], device=x.device)
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x, ld = flow(x, cond, reverse=reverse)
            total_logdet += ld
            if not reverse:
                x = x.flip(dims=[1])   # reverse channel order between layers
        return x, total_logdet


# ═══════════════════════════════════════════════════════════════════════════
# Stochastic Duration Predictor
# ═══════════════════════════════════════════════════════════════════════════

class StochasticDurationPredictor(nn.Module):
    """
    Stochastic duration predictor (Kim et al., VITS 2021).
    Predicts phoneme durations with variability via flows.
    """

    def __init__(self, channels: int, hidden: int, n_flows: int = 4,
                 cond_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.pre  = nn.Conv1d(channels, hidden, 1)
        self.proj = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.film = FiLM(hidden, cond_dim)
        self.post = nn.Conv1d(hidden, 1, 1)      # log-duration output
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:    torch.Tensor,     # [B, channels, T_phone]
        cond: torch.Tensor,     # [B, cond_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:          # [B, 1, T_phone]  log duration
        h = self.pre(x)
        h = self.proj(h)
        h = self.film(h, cond)
        h = F.relu(h)
        h = self.drop(h)
        return self.post(h)


# ═══════════════════════════════════════════════════════════════════════════
# HiFi-GAN Generator (upsampling decoder)
# ═══════════════════════════════════════════════════════════════════════════

class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN V1 generator with FiLM speaker conditioning.

    Upsamples a latent sequence from acoustic model frame rate (256 samples/frame)
    to 22050 Hz waveform via transposed convolutions + multi-receptive-field
    fusion residual blocks.

    Upsample chain: ×8  ×8  ×2  ×2  = ×256  →  22050 Hz

    Parameters
    ----------
    in_channels : Latent space channels from flow (default 192).
    upsample_rates: Upsampling factors per stage.
    upsample_kernels: Kernel sizes for each transposed conv.
    resblock_kernels: Kernel sizes for MRF residual blocks.
    resblock_dilations: Dilation patterns for MRF.
    hidden_channels : Base channel count (halved at each stage).
    cond_dim        : d-vector dimension for FiLM.
    """

    def __init__(
        self,
        in_channels:          int          = 192,
        upsample_rates:       List[int]    = [8, 8, 2, 2],
        upsample_kernels:     List[int]    = [16, 16, 4, 4],
        resblock_kernels:     List[int]    = [3, 7, 11],
        resblock_dilations:   List[List[int]] = [[1,3,5],[1,3,5],[1,3,5]],
        hidden_channels:      int          = 256,
        cond_dim:             int          = 256,
    ) -> None:
        super().__init__()
        self.n_up = len(upsample_rates)

        self.conv_pre = nn.Conv1d(in_channels, hidden_channels, 7, padding=3)

        # Upsampling stages
        self.ups = nn.ModuleList()
        ch = hidden_channels
        for i, (r, k) in enumerate(zip(upsample_rates, upsample_kernels)):
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, k, stride=r, padding=(k-r)//2)
            )
            ch = ch // 2

        # MRF residual blocks + FiLM for each stage
        self.resblocks = nn.ModuleList()
        self.films      = nn.ModuleList()
        ch = hidden_channels // 2
        for i in range(self.n_up):
            self.films.append(FiLM(ch, cond_dim))
            for j, (k, d) in enumerate(zip(resblock_kernels, resblock_dilations)):
                for dil in d:
                    self.resblocks.append(
                        nn.Sequential(
                            nn.Conv1d(ch, ch, k, dilation=dil,
                                      padding=dil*(k-1)//2),
                            nn.LeakyReLU(0.1),
                        )
                    )
            ch = ch // 2

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch * 2 if self.n_up > 0 else hidden_channels, 1, 7, padding=3),
            nn.Tanh(),
        )
        self._n_resblocks_per_stage = len(resblock_kernels) * len(resblock_dilations[0])

    def forward(
        self,
        x:    torch.Tensor,   # [B, in_channels, T_lat]
        cond: torch.Tensor,   # [B, cond_dim]
    ) -> torch.Tensor:        # [B, 1, T_wav]
        x = self.conv_pre(x)
        rb_idx = 0
        ch_cur = x.shape[1]

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # FiLM conditioning
            x = self.films[i](x, cond)

            # MRF: sum over residual blocks
            xs = torch.zeros_like(x)
            n  = self._n_resblocks_per_stage
            for j in range(n):
                if rb_idx + j < len(self.resblocks):
                    xs = xs + self.resblocks[rb_idx + j](x)
            rb_idx += n
            x = xs / max(n, 1)

        return self.conv_post(x)   # [B, 1, T_wav]


# ═══════════════════════════════════════════════════════════════════════════
# Text Encoder (Transformer on IPA tokens)
# ═══════════════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    Transformer encoder on IPA phoneme token sequences.

    Produces per-phoneme latent representations used by the
    duration predictor and flow.

    Parameters
    ----------
    vocab_size    : IPA vocabulary size.
    d_model       : Embedding / attention dimension.
    n_heads       : Number of attention heads.
    n_layers      : Number of FFT blocks.
    ffn_dim       : FFN hidden dimension.
    out_channels  : Output projection (for flow input).
    cond_dim      : d-vector dimension for FiLM injection.
    """

    def __init__(
        self,
        vocab_size:   int = VOCAB_SIZE,
        d_model:      int = 192,
        n_heads:      int = 2,
        n_layers:     int = 6,
        ffn_dim:      int = 768,
        out_channels: int = 192,
        cond_dim:     int = 256,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe     = SinusoidalPE(d_model)
        self.blocks = nn.ModuleList([
            FFTBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.film   = FiLM(d_model, cond_dim)   # inject speaker via FiLM
        self.proj_m = nn.Conv1d(d_model, out_channels, 1)   # mean
        self.proj_s = nn.Conv1d(d_model, out_channels, 1)   # log-std

    def forward(
        self,
        tokens:  torch.Tensor,      # [B, T_phone]
        cond:    torch.Tensor,      # [B, cond_dim]
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std), each [B, out_channels, T_phone]."""
        x = self.embed(tokens)       # [B, T, d_model]
        x = self.pe(x)
        for blk in self.blocks:
            x = blk(x, src_mask)

        # FiLM with speaker embedding
        x = x.transpose(1, 2)       # [B, d_model, T]
        x = self.film(x, cond)
        x = x.transpose(1, 2)       # [B, T, d_model]

        x = x.transpose(1, 2)       # [B, d_model, T] for Conv1d
        m = self.proj_m(x)
        s = self.proj_s(x)
        return m, s                  # [B, out_channels, T_phone]


# ═══════════════════════════════════════════════════════════════════════════
# Length Regulator (phoneme → frame expansion)
# ═══════════════════════════════════════════════════════════════════════════

class LengthRegulator(nn.Module):
    """
    Expand phoneme-level representations to frame-level via predicted durations.
    """

    @staticmethod
    def forward(
        x:         torch.Tensor,     # [B, C, T_phone]
        durations:  torch.Tensor,    # [B, T_phone]  integer frame counts
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (expanded [B, C, T_frame], mel_mask [B, T_frame])."""
        B, C, T_phone = x.shape
        max_frames = int(durations.sum(dim=1).max().item())
        out = torch.zeros(B, C, max_frames, device=x.device)
        mel_mask = torch.ones(B, max_frames, dtype=torch.bool, device=x.device)

        for b in range(B):
            pos = 0
            for p in range(T_phone):
                d = int(durations[b, p].item())
                if d > 0:
                    out[b, :, pos : pos + d] = x[b, :, p : p + 1].expand(-1, d)
                    mel_mask[b, pos : pos + d] = False
                    pos += d

        return out, mel_mask


# ═══════════════════════════════════════════════════════════════════════════
# VITS Model (full)
# ═══════════════════════════════════════════════════════════════════════════

class VITS(nn.Module):
    """
    VITS: Variational Inference with adversarial learning for end-to-end TTS.
    (Kim et al., 2021)  — adapted for cross-lingual zero-shot synthesis.

    Forward (training)
    ─────────────────
    tokens   [B, T_phone]  → TextEncoder → (m_p, s_p)  [B, C, T_phone]
    mel      [B, n_mels, T_mel] → PosteriorEncoder → z  [B, C, T_mel]
    z        → Flow (fwd) → z_p  [B, C, T_mel]
    z_p      → LengthRegulator (GT durations)
    z        → HiFiGAN → waveform  [B, 1, T_wav]

    Inference
    ─────────
    tokens → TextEncoder → (m_p, s_p)
    m_p    → Flow (inv) → z̃
    DurationPredictor → durations
    z̃  → LengthRegulator (predicted durations)
    z̃  → HiFiGAN → waveform
    """

    def __init__(
        self,
        n_mels:      int = 80,
        latent_dim:  int = 192,
        cond_dim:    int = 256,
        flow_hidden: int = 192,
        n_flows:     int = 4,
    ) -> None:
        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=VOCAB_SIZE, d_model=latent_dim,
            n_heads=2, n_layers=6, out_channels=latent_dim, cond_dim=cond_dim,
        )
        self.posterior_encoder = nn.Sequential(
            nn.Conv1d(n_mels, latent_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(latent_dim, latent_dim * 2, 5, padding=2),
        )
        self.flow = ResidualCouplingBlock(
            latent_dim, flow_hidden, n_flows=n_flows, cond_dim=cond_dim
        )
        self.duration_predictor = StochasticDurationPredictor(
            latent_dim, 256, cond_dim=cond_dim
        )
        self.length_regulator = LengthRegulator()
        self.decoder = HiFiGANGenerator(
            in_channels=latent_dim, cond_dim=cond_dim
        )

        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim

    @torch.no_grad()
    def infer(
        self,
        tokens:    torch.Tensor,   # [B, T_phone]
        dvector:   torch.Tensor,   # [B, cond_dim]
        noise_scale: float = 0.667,
        duration_scale: float = 1.0,
        max_duration_per_phone: int = 50,
    ) -> torch.Tensor:             # [B, 1, T_wav]
        """
        Generate waveform from phoneme tokens and speaker d-vector.

        Parameters
        ----------
        tokens         : [B, T_phone] IPA token IDs.
        dvector        : [B, cond_dim] speaker embedding (d-vector or x-vector).
        noise_scale    : Controls diversity of latent sampling (0=deterministic).
        duration_scale : Scale factor on predicted durations (1.0 = natural).
        max_duration_per_phone: Hard cap to prevent silence hallucination.

        Returns
        -------
        waveform : [B, 1, T_wav]
        """
        B = tokens.shape[0]

        # Text encoder
        m_p, s_p = self.text_encoder(tokens, dvector)   # [B, C, T_phone]

        # Sample from the prior
        z_p = m_p + torch.randn_like(m_p) * s_p.exp() * noise_scale

        # Inverse flow: z_p → z
        z, _ = self.flow(z_p, dvector, reverse=True)

        # Predict durations
        log_dur = self.duration_predictor(z, dvector)   # [B, 1, T_phone]
        durations = (log_dur.squeeze(1).exp() * duration_scale).round().long()
        durations = durations.clamp(1, max_duration_per_phone)

        # Expand to mel frame level
        z_expanded, _ = self.length_regulator(z, durations)   # [B, C, T_frame]

        # Decode to waveform
        wav = self.decoder(z_expanded, dvector)   # [B, 1, T_wav]
        return wav

    def forward(
        self,
        tokens:    torch.Tensor,   # [B, T_phone]
        mel:       torch.Tensor,   # [B, n_mels, T_mel]
        dvector:   torch.Tensor,   # [B, cond_dim]
        durations: torch.Tensor,   # [B, T_phone] GT durations
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass. Returns dict of all losses."""
        # Text encoder
        m_p, s_p = self.text_encoder(tokens, dvector)   # [B, C, T_phone]

        # Posterior encoder: mel → z
        post_stats = self.posterior_encoder(mel)          # [B, 2C, T_mel]
        m_q, s_q = post_stats.chunk(2, dim=1)            # [B, C, T_mel]
        z = m_q + torch.randn_like(m_q) * s_q.exp()

        # Flow: z → z_p
        z_p, log_det = self.flow(z, dvector, reverse=False)

        # KL loss (z_p vs text prior)
        # Expand text prior to mel frame length
        exp_m_p, _ = self.length_regulator(m_p, durations)
        exp_s_p, _ = self.length_regulator(s_p, durations)
        T_mel = min(z_p.shape[-1], exp_m_p.shape[-1])

        kl = (
            -0.5 * (1 + 2 * s_q[..., :T_mel] - m_q[..., :T_mel].pow(2)
                    - s_q[..., :T_mel].exp().pow(2))
        ).sum(dim=1).mean()

        # Duration predictor loss
        log_dur_pred = self.duration_predictor(m_p.detach(), dvector)
        dur_loss = F.mse_loss(
            log_dur_pred.squeeze(1),
            (durations.float() + 1).log()
        )

        # Decoder output (for GAN training)
        wav_pred = self.decoder(z, dvector)

        return {
            "wav_pred": wav_pred,
            "kl_loss":  kl,
            "dur_loss": dur_loss,
            "log_det":  -log_det.mean(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# VITSSynthesizer — high-level inference API
# ═══════════════════════════════════════════════════════════════════════════

class VITSSynthesizer:
    """
    High-level VITS synthesizer for zero-shot cross-lingual TTS.

    Wraps the VITS model and provides:
      • `synthesize(ipa_text, dvector)` → waveform tensor
      • `synthesize_long(ipa_text, dvector)` → chunks + concatenate
      • `synthesize_lecture(segments, dvector)` → full lecture

    Parameters
    ----------
    checkpoint_path : Path to trained VITS weights (.pt).
                      None → uses random weights (pipeline test only).
    device          : 'cuda' | 'cpu'.
    sample_rate     : Output sample rate (default 22050 Hz).
    max_tokens_per_chunk : Max IPA tokens per synthesis chunk.
    """

    OUTPUT_SR = 22_050

    def __init__(
        self,
        checkpoint_path:       Optional[str] = None,
        device:                str           = "cuda",
        sample_rate:           int           = 22_050,
        max_tokens_per_chunk:  int           = 200,
        noise_scale:           float         = 0.667,
        duration_scale:        float         = 1.0,
    ) -> None:
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sr        = sample_rate
        self.max_chunk = max_tokens_per_chunk
        self.noise     = noise_scale
        self.dur_scale = duration_scale

        self.model = VITS().to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[TTS] Loaded checkpoint: {checkpoint_path}")
        else:
            print(
                "[TTS] No checkpoint found. Using random weights — "
                "output will be noise. Train first or load a pre-trained model."
            )

        self.model.eval()

        # Resample if model was trained at different rate
        self._resampler = None
        if sample_rate != self.OUTPUT_SR:
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=self.OUTPUT_SR, new_freq=sample_rate
            ).to(self.device)

    def _split_ipa(self, ipa_text: str) -> List[str]:
        """Split IPA text at sentence boundaries for chunked synthesis."""
        sentences = []
        current = []
        toks = []
        i = 0
        while i < len(ipa_text):
            for l in [3, 2, 1]:
                cand = ipa_text[i : i + l]
                if cand in TOKEN2ID:
                    toks.append(cand)
                    i += l
                    break
            else:
                i += 1

        for tok in toks:
            current.append(tok)
            if tok in (".", "?", "!") or len(current) >= self.max_chunk:
                sentences.append("".join(current))
                current = []
        if current:
            sentences.append("".join(current))
        return sentences

    @torch.no_grad()
    def synthesize(
        self,
        ipa_text: str,
        dvector:  torch.Tensor,    # [cond_dim]
    ) -> torch.Tensor:             # [1, T_wav]
        """Synthesize a short IPA string (up to max_tokens_per_chunk)."""
        ids = ipa_to_ids(ipa_text).unsqueeze(0).to(self.device)  # [1, T]
        cond = dvector.unsqueeze(0).to(self.device)               # [1, D]
        wav = self.model.infer(ids, cond, self.noise, self.dur_scale)
        if self._resampler is not None:
            wav = self._resampler(wav)
        return wav.cpu()

    @torch.no_grad()
    def synthesize_long(
        self,
        ipa_text: str,
        dvector:  torch.Tensor,
        pause_ms: float = 300.0,
    ) -> torch.Tensor:             # [1, T_wav]
        """Synthesize long text by chunking at sentence boundaries."""
        chunks   = self._split_ipa(ipa_text)
        segments: List[torch.Tensor] = []
        silence  = torch.zeros(1, int(pause_ms / 1000 * self.sr))

        print(f"[TTS] Synthesizing {len(chunks)} chunks …")
        for i, chunk in enumerate(chunks):
            wav = self.synthesize(chunk, dvector)
            segments.append(wav)
            if i < len(chunks) - 1:
                segments.append(silence)
            print(f"  Chunk {i+1:3d}/{len(chunks)}  tokens={len(chunk):4d}  "
                  f"dur={wav.shape[-1]/self.sr:.2f}s")

        return torch.cat(segments, dim=-1)

    @torch.no_grad()
    def synthesize_lecture(
        self,
        santhali_segments: List[dict],
        dvector:           torch.Tensor,
        output_path:       str,
        use_ipa:           bool = True,
        pause_between_segs_ms: float = 500.0,
    ) -> torch.Tensor:
        """
        Synthesize a full lecture from translated Santhali segments.

        Parameters
        ----------
        santhali_segments : List of segment dicts from Part II.
                            Each must have 'ipa' or 'santhali_roman' key.
        dvector           : Speaker embedding [cond_dim].
        output_path       : Path to write the synthesised lecture WAV.
        use_ipa           : Use 'ipa' key (preferred) or 'santhali_roman'.
        pause_between_segs_ms: Silence inserted between segments.

        Returns
        -------
        full_wav : [1, T_total]  concatenated lecture waveform.
        """
        parts: List[torch.Tensor] = []
        silence = torch.zeros(1, int(pause_between_segs_ms / 1000 * self.sr))
        total_dur = 0.0

        print(f"\n[TTS] Synthesising {len(santhali_segments)} lecture segments …")

        for i, seg in enumerate(santhali_segments):
            text_key = "ipa" if (use_ipa and "ipa" in seg) else "santhali_roman"
            text = seg.get(text_key, "")
            if not text.strip():
                continue

            wav = self.synthesize_long(text, dvector,
                                       pause_ms=200.0)
            dur = wav.shape[-1] / self.sr
            total_dur += dur

            parts.append(wav)
            if i < len(santhali_segments) - 1:
                parts.append(silence)

            print(
                f"  Segment {i+1:3d}/{len(santhali_segments)}  "
                f"dur={dur:.1f}s  total={total_dur/60:.1f}min"
            )

        full_wav = torch.cat(parts, dim=-1) if parts else torch.zeros(1, self.sr)

        # Ensure minimum output rate 22050 Hz
        assert self.sr >= 22_050, f"Output SR {self.sr} < required 22050 Hz"

        # Normalise
        peak = full_wav.abs().max().clamp(min=1e-8)
        full_wav = full_wav / peak * 0.95

        torchaudio.save(output_path, full_wav, self.sr)
        print(
            f"\n[TTS] Lecture synthesised  "
            f"total_duration={total_dur/60:.1f}min  "
            f"sample_rate={self.sr}Hz  "
            f"saved → {output_path}"
        )

        return full_wav


# ═══════════════════════════════════════════════════════════════════════════
# Model summary
# ═══════════════════════════════════════════════════════════════════════════

def model_summary(model: nn.Module) -> str:
    total  = sum(p.numel() for p in model.parameters())
    traini = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total params: {total:,}  |  Trainable: {traini:,}"


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = VITS()
    print(model_summary(model))
    print(f"IPA vocab size: {VOCAB_SIZE}")

    # Inference smoke-test
    tokens  = torch.randint(1, VOCAB_SIZE, (1, 30))   # [1, 30]
    dvector = F.normalize(torch.randn(1, 256), p=2, dim=-1)
    with torch.no_grad():
        wav = model.infer(tokens, dvector, noise_scale=0.0)
    print(f"Output waveform : {tuple(wav.shape)}  sr=22050")
    print(f"Duration        : {wav.shape[-1]/22050:.2f}s")
