"""
speaker_embedding.py  —  Task 3.1
===================================
Speaker Embedding Extraction from a 60-second reference recording.

Produces both:
  • d-vector  — GE2E-trained LSTM encoder (Wan et al., 2018)
  • x-vector  — TDNN + statistics pooling (Snyder et al., 2018)

Both are extracted from mel-spectrogram frames of the speaker audio.
The final embedding is an L2-normalised vector suitable for conditioning
a VITS/YourTTS synthesiser.

Architecture choices
─────────────────────
d-vector  (default, recommended for TTS)
    3-layer LSTM  →  linear projection  →  L2 norm
    Trained with Generalized End-to-End (GE2E) loss on speaker batches.
    Output: 256-d unit hypersphere embedding.

x-vector
    5-layer TDNN (time-delay neural network) with dilated context
    →  attentive statistics pooling (mean + std across time)
    →  segment-level embedding  →  L2 norm
    Output: 512-d unit hypersphere embedding.

Usage
─────
    from speaker_embedding import SpeakerEncoderGE2E, extract_speaker_embedding

    # Quick single-file embedding
    emb = extract_speaker_embedding("my_voice_60s.wav")   # → [256] tensor

    # Full encoder object (for batches, fine-tuning, saving)
    encoder = SpeakerEncoderGE2E()
    emb = encoder.embed_utterance("my_voice_60s.wav")
    encoder.save("speaker_encoder.pt")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Ta


# ═══════════════════════════════════════════════════════════════════════════
# Mel feature extractor (shared by both d-vector and x-vector)
# ═══════════════════════════════════════════════════════════════════════════

class SpeakerMelExtractor(nn.Module):
    """
    Log mel-spectrogram extractor tuned for speaker characterisation.

    Parameters match the SV2TTS / VITS speaker-encoder conventions:
        sample_rate  = 16 000 Hz
        n_fft        = 400   (25 ms windows)
        hop_length   = 160   (10 ms hop)
        n_mels       = 40
    """

    SR        = 16_000
    N_FFT     = 400
    HOP       = 160
    WIN       = 400
    N_MELS    = 40
    F_MIN     = 20.0
    F_MAX     = 7_600.0
    TOP_DB    = 80.0
    PARTIAL_S = 1.6        # seconds of partial utterance windows
    MIN_PAD_S = 0.5        # min audio length before padding

    def __init__(self) -> None:
        super().__init__()
        self.mel = Ta.MelSpectrogram(
            sample_rate=self.SR, n_fft=self.N_FFT,
            hop_length=self.HOP, win_length=self.WIN,
            n_mels=self.N_MELS, f_min=self.F_MIN, f_max=self.F_MAX,
        )
        self.amp_to_db = Ta.AmplitudeToDB(stype="power", top_db=self.TOP_DB)

    @torch.no_grad()
    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [T] → log-mel [T_frames, n_mels]"""
        mel = self.mel(wav)           # [n_mels, T_frames]
        log_mel = self.amp_to_db(mel)
        log_mel = (log_mel + 40.0) / 40.0   # roughly [-1, 1]
        return log_mel.T              # [T_frames, n_mels]

    def load_wav(self, path: str, target_sr: int = 16_000) -> torch.Tensor:
        """Load any WAV → mono 16 kHz float32 [T]."""
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)                            # mono
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        # Soft clipping
        wav = wav.clamp(-1.0, 1.0)
        return wav

    def split_to_partials(
        self, mel: torch.Tensor, overlap: float = 0.5
    ) -> torch.Tensor:
        """
        Slice mel spectrogram into overlapping partial windows.

        Returns [N_partials, partial_frames, n_mels]
        """
        partial_frames = int(self.PARTIAL_S * self.SR / self.HOP)
        step = max(1, int(partial_frames * (1 - overlap)))
        T = mel.shape[0]

        if T < partial_frames:
            # Pad short utterances
            pad = partial_frames - T
            mel = F.pad(mel, (0, 0, 0, pad))
            return mel.unsqueeze(0)

        slices = []
        for start in range(0, T - partial_frames + 1, step):
            slices.append(mel[start : start + partial_frames])

        return torch.stack(slices)      # [N, partial_frames, n_mels]


# ═══════════════════════════════════════════════════════════════════════════
# d-vector encoder  (GE2E-style LSTM)
# ═══════════════════════════════════════════════════════════════════════════

class SpeakerEncoderGE2E(nn.Module):
    """
    GE2E d-vector speaker encoder (Wan et al., 2018).

    Architecture
    ─────────────
    Input  : Log-mel frames  [B, T, n_mels]
    3-layer LSTM  hidden_dim=768  →  last-step hidden
    Linear projection  768 → proj_dim (default 256)
    L2 normalisation  →  unit d-vector  [B, 256]

    The embedding of a long utterance is obtained by:
      1. Splitting into 1.6 s partial windows
      2. Encoding each partial
      3. L2-normalising and mean-pooling

    Parameters
    ----------
    n_mels    : Input mel bins (must match SpeakerMelExtractor.N_MELS).
    hidden_dim: LSTM hidden state size.
    proj_dim  : Output embedding dimensionality.
    n_layers  : Number of LSTM layers.
    """

    def __init__(
        self,
        n_mels:     int = 40,
        hidden_dim: int = 768,
        proj_dim:   int = 256,
        n_layers:   int = 3,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        self.n_mels     = n_mels
        self.hidden_dim = hidden_dim
        self.proj_dim   = proj_dim

        self.lstm = nn.LSTM(
            input_size=n_mels, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, proj_dim)
        self.feat       = SpeakerMelExtractor()

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel : [B, T, n_mels]  →  d-vector [B, proj_dim]  (L2 normalised)
        """
        _, (h_n, _) = self.lstm(mel)        # h_n: [n_layers, B, H]
        emb = self.projection(h_n[-1])      # [B, proj_dim]
        return F.normalize(emb, p=2, dim=-1)

    @torch.no_grad()
    def embed_utterance(
        self,
        wav_path: str,
        min_coverage: float = 0.75,
    ) -> torch.Tensor:
        """
        Compute d-vector for a full utterance (handles long recordings by
        partial-window averaging).

        Parameters
        ----------
        wav_path     : Path to speaker reference WAV (ideally ~60 s).
        min_coverage : Fraction of partials that must be non-silent.

        Returns
        -------
        d_vector : [proj_dim]  (L2-normalised)
        """
        self.eval()
        wav = self.feat.load_wav(wav_path)
        mel = self.feat.wav_to_mel(wav)                     # [T, n_mels]
        partials = self.feat.split_to_partials(mel)         # [N, T', n_mels]

        # VAD: skip nearly-silent partials
        energies = partials.pow(2).mean(dim=(1, 2))
        threshold = energies.quantile(1.0 - min_coverage)
        mask = energies >= threshold
        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)

        active = partials[mask]                             # [M, T', n_mels]
        device = next(self.parameters()).device
        active = active.to(device)

        partial_embs = self.forward(active)                 # [M, proj_dim]
        d_vec = F.normalize(partial_embs.mean(0), p=2, dim=-1)  # [proj_dim]
        return d_vec.cpu()

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "n_mels":     self.n_mels,
                "hidden_dim": self.hidden_dim,
                "proj_dim":   self.proj_dim,
            },
        }, path)
        print(f"[SpeakerEncoder] Saved d-vector encoder → {path}")

    @classmethod
    def load(cls, path: str) -> "SpeakerEncoderGE2E":
        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[SpeakerEncoder] Loaded from {path}  proj_dim={cfg['proj_dim']}")
        return model


# ═══════════════════════════════════════════════════════════════════════════
# x-vector encoder  (TDNN + attentive statistics pooling)
# ═══════════════════════════════════════════════════════════════════════════

class TDNNLayer(nn.Module):
    """
    Single Time-Delay Neural Network layer.

    Implements a 1-D convolution over a dilated temporal context window.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        context_size: int = 5,
        dilation:     int = 1,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=context_size,
            dilation=dilation,
            padding=(context_size - 1) * dilation // 2,
        )
        self.bn      = nn.BatchNorm1d(out_channels)
        self.act     = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, T]  →  [B, C', T]"""
        return self.dropout(self.act(self.bn(self.conv(x))))


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling (Okabe et al., 2018).

    Computes attention-weighted mean and std over the time axis,
    producing a frame-independent segment-level representation.
    """

    def __init__(self, channels: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, T]  →  [B, 2C]  (weighted mean ‖ weighted std)"""
        w = self.attn(x)                                   # [B, C, T]
        mean = (w * x).sum(dim=-1)                        # [B, C]
        std  = (w * (x - mean.unsqueeze(-1)).pow(2)).sum(dim=-1).clamp(min=1e-8).sqrt()
        return torch.cat([mean, std], dim=-1)              # [B, 2C]


class XVectorEncoder(nn.Module):
    """
    x-vector encoder (Snyder et al., 2018) with attentive pooling extension.

    Architecture
    ─────────────
    Input  : Log-mel  [B, T, n_mels]
    5 TDNN layers  (dilations 1,2,3,1,1 ; channels 512)
    Attentive Statistics Pooling  →  [B, 1024]
    2 FC layers  →  x-vector  [B, 512]
    L2 normalisation

    Parameters
    ----------
    n_mels     : Input feature dimension.
    tdnn_dim   : Number of TDNN channels.
    emb_dim    : Output x-vector dimensionality.
    """

    def __init__(
        self,
        n_mels:   int = 40,
        tdnn_dim: int = 512,
        emb_dim:  int = 512,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()

        self.tdnn = nn.Sequential(
            TDNNLayer(n_mels,    tdnn_dim, context_size=5, dilation=1, dropout=dropout),
            TDNNLayer(tdnn_dim,  tdnn_dim, context_size=3, dilation=2, dropout=dropout),
            TDNNLayer(tdnn_dim,  tdnn_dim, context_size=3, dilation=3, dropout=dropout),
            TDNNLayer(tdnn_dim,  tdnn_dim, context_size=1, dilation=1, dropout=dropout),
            TDNNLayer(tdnn_dim,  tdnn_dim, context_size=1, dilation=1, dropout=dropout),
        )

        self.pooling = AttentiveStatisticsPooling(tdnn_dim)

        self.fc = nn.Sequential(
            nn.Linear(2 * tdnn_dim, tdnn_dim),
            nn.BatchNorm1d(tdnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tdnn_dim, emb_dim),
        )

        self.feat    = SpeakerMelExtractor()
        self.emb_dim = emb_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel : [B, T, n_mels]  →  x-vector [B, emb_dim]  (L2 normalised)
        """
        x = mel.transpose(1, 2)               # [B, n_mels, T]
        x = self.tdnn(x)                      # [B, tdnn_dim, T]
        x = self.pooling(x)                   # [B, 2*tdnn_dim]
        x = self.fc(x)                        # [B, emb_dim]
        return F.normalize(x, p=2, dim=-1)

    @torch.no_grad()
    def embed_utterance(self, wav_path: str) -> torch.Tensor:
        """Full-utterance x-vector extraction."""
        self.eval()
        wav = self.feat.load_wav(wav_path)
        mel = self.feat.wav_to_mel(wav).unsqueeze(0)   # [1, T, n_mels]
        return self.forward(mel).squeeze(0).cpu()      # [emb_dim]


# ═══════════════════════════════════════════════════════════════════════════
# GE2E training loss
# ═══════════════════════════════════════════════════════════════════════════

class GE2ELoss(nn.Module):
    """
    Generalised End-to-End (GE2E) speaker verification loss (Wan et al., 2018).

    Operates on a batch of shape [N_speakers, M_utterances, emb_dim].
    Maximises similarity of same-speaker embeddings vs different-speaker ones.

    Parameters
    ----------
    init_w : Initial value of the scaling parameter w.
    init_b : Initial value of the bias parameter b.
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings : [N, M, D]  (N speakers, M utterances each, D-dim)
        Returns scalar GE2E loss.
        """
        N, M, D = embeddings.shape

        # Centroid per speaker (excluding the probe utterance — leave-one-out)
        # Shape: [N, D]
        centroids_full = embeddings.mean(dim=1)            # [N, D]

        total_loss = torch.tensor(0.0, device=embeddings.device)

        for j in range(N):
            for i in range(M):
                # Leave-one-out centroid for speaker j
                loo_centroid = (
                    centroids_full[j] * M - embeddings[j, i]
                ) / (M - 1)

                # Cosine similarity of embedding (j,i) with all speaker centroids
                probe = embeddings[j, i]               # [D]
                sims = []
                for k in range(N):
                    if k == j:
                        c = loo_centroid
                    else:
                        c = centroids_full[k]
                    sims.append(F.cosine_similarity(probe.unsqueeze(0),
                                                    c.unsqueeze(0)).squeeze())
                sim_vec = torch.stack(sims)            # [N]
                scaled  = self.w.clamp(min=0.01) * sim_vec + self.b

                # Softmax loss: probe should match speaker j
                loss = -scaled[j] + torch.logsumexp(scaled, dim=0)
                total_loss = total_loss + loss

        return total_loss / (N * M)


# ═══════════════════════════════════════════════════════════════════════════
# Speaker encoder trainer
# ═══════════════════════════════════════════════════════════════════════════

class SpeakerEncoderTrainer:
    """
    Fine-tuner for SpeakerEncoderGE2E on a small personal dataset.

    Dataset layout expected::

        data/speaker/
        ├── speaker_A/   *.wav
        ├── speaker_B/   *.wav
        └── your_voice/  *.wav   ← your own 60-s recording (split into chunks)

    Fine-tuning for even 10–20 epochs helps personalise the embedding space
    if starting from a randomly initialised encoder.

    Usage
    ─────
        trainer = SpeakerEncoderTrainer(model, data_dir="data/speaker")
        trainer.train(epochs=20)
        model.save("speaker_encoder.pt")
    """

    def __init__(
        self,
        model:      SpeakerEncoderGE2E,
        data_dir:   str,
        n_speakers: int = 4,    # speakers per batch
        n_utts:     int = 5,    # utterances per speaker per batch
        lr:         float = 1e-4,
        device:     str = "cuda",
    ) -> None:
        import os
        self.model  = model
        self.feat   = SpeakerMelExtractor()
        self.loss_fn = GE2ELoss()
        self.N      = n_speakers
        self.M      = n_utts
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        model.to(self.device)
        self.loss_fn.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.loss_fn.parameters()), lr=lr
        )

        # Discover files grouped by speaker directory
        self.speaker_files: Dict[str, List[str]] = {}
        for spk_dir in sorted(Path(data_dir).iterdir()):
            if not spk_dir.is_dir():
                continue
            wavs = list(spk_dir.glob("*.wav"))
            if len(wavs) >= n_utts:
                self.speaker_files[spk_dir.name] = [str(w) for w in wavs]

        print(f"[Trainer] Found {len(self.speaker_files)} speakers  "
              f"(need ≥ {n_speakers})")

    def _sample_batch(self) -> torch.Tensor:
        """Sample [N, M, n_mels_frames] batch for GE2E."""
        import random
        speakers = random.sample(list(self.speaker_files.keys()), self.N)
        partial_frames = int(self.feat.PARTIAL_S * self.feat.SR / self.feat.HOP)
        batch_mels = []
        for spk in speakers:
            files = random.sample(self.speaker_files[spk], self.M)
            spk_mels = []
            for f in files:
                wav = self.feat.load_wav(f)
                mel = self.feat.wav_to_mel(wav)
                if mel.shape[0] < partial_frames:
                    mel = F.pad(mel, (0, 0, 0, partial_frames - mel.shape[0]))
                else:
                    start = torch.randint(0, mel.shape[0] - partial_frames + 1, (1,)).item()
                    mel = mel[start : start + partial_frames]
                spk_mels.append(mel)
            batch_mels.append(torch.stack(spk_mels))
        return torch.stack(batch_mels).to(self.device)  # [N, M, T, n_mels]

    def train(self, epochs: int = 20, steps_per_epoch: int = 50) -> None:
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                batch = self._sample_batch()          # [N, M, T, n_mels]
                N, M, T, C = batch.shape
                flat_mel = batch.reshape(N * M, T, C)
                embs = self.model(flat_mel)            # [N*M, D]
                embs_3d = embs.reshape(N, M, -1)      # [N, M, D]

                loss = self.loss_fn(embs_3d)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"  Epoch {epoch:3d}/{epochs}  GE2E loss = {epoch_loss/steps_per_epoch:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: extract & save embeddings for the 60-second reference
# ═══════════════════════════════════════════════════════════════════════════

def extract_speaker_embedding(
    wav_path:    str,
    encoder_path: Optional[str] = None,
    embedding_type: str = "dvector",          # "dvector" | "xvector"
    save_path:   Optional[str] = None,
    device:      str = "cpu",
) -> torch.Tensor:
    """
    Extract a speaker embedding from a WAV file.

    Parameters
    ----------
    wav_path       : Path to ~60 s reference recording.
    encoder_path   : Pre-trained encoder checkpoint; if None, uses random init.
    embedding_type : 'dvector' (LSTM GE2E) or 'xvector' (TDNN).
    save_path      : If given, save the embedding tensor as a .pt file.
    device         : Compute device.

    Returns
    -------
    embedding : [D] float32 tensor (L2 normalised, D=256 for d-vec, 512 x-vec)
    """
    if embedding_type == "dvector":
        if encoder_path and Path(encoder_path).exists():
            enc = SpeakerEncoderGE2E.load(encoder_path)
        else:
            enc = SpeakerEncoderGE2E()
            if encoder_path:
                print(f"[Warning] Encoder checkpoint not found: {encoder_path}. "
                      "Using random init — fine-tune for best quality.")
        enc = enc.to(device)
        emb = enc.embed_utterance(wav_path)
    else:
        enc = XVectorEncoder()
        enc = enc.to(device)
        emb = enc.embed_utterance(wav_path)

    print(f"[SpeakerEmb] Extracted {embedding_type}  shape={tuple(emb.shape)}  "
          f"norm={emb.norm().item():.4f}")

    if save_path:
        torch.save(emb, save_path)
        print(f"[SpeakerEmb] Saved → {save_path}")

    return emb


def verify_60s_recording(wav_path: str) -> Dict:
    """
    Verify that a recording meets the 60-second requirement and print diagnostics.

    Returns dict with {'duration_s', 'sr', 'rms', 'is_valid'}.
    """
    wav, sr = torchaudio.load(wav_path)
    wav_mono = wav.mean(0)
    duration = wav_mono.shape[0] / sr
    rms = wav_mono.pow(2).mean().sqrt().item()

    is_valid = duration >= 55.0    # allow 5 s tolerance

    print(f"\n── Speaker Recording Verification ──")
    print(f"  File     : {wav_path}")
    print(f"  Duration : {duration:.1f}s  {'✓' if is_valid else '✗ (need ≥ 60s)'}")
    print(f"  SR       : {sr} Hz  {'✓' if sr >= 16000 else '(recommend ≥ 16kHz)'}")
    print(f"  RMS      : {rms:.4f}  {'✓' if rms > 0.01 else '✗ (signal too quiet?)'}")

    return {"duration_s": duration, "sr": sr, "rms": rms, "is_valid": is_valid}


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Speaker Embedding Extractor (Task 3.1)")
    p.add_argument("wav",            help="Reference WAV (~60 s)")
    p.add_argument("--encoder",      default=None, help="Pre-trained encoder .pt")
    p.add_argument("--type",         default="dvector", choices=["dvector","xvector"])
    p.add_argument("--save",         default="speaker_embedding.pt")
    p.add_argument("--device",       default="cpu")
    args = p.parse_args()

    verify_60s_recording(args.wav)
    emb = extract_speaker_embedding(
        args.wav,
        encoder_path=args.encoder,
        embedding_type=args.type,
        save_path=args.save,
        device=args.device,
    )
    print(f"\nEmbedding stats:")
    print(f"  dim   = {emb.shape[0]}")
    print(f"  norm  = {emb.norm():.6f}  (should be 1.0 if L2-normalised)")
    print(f"  min   = {emb.min():.4f}  max = {emb.max():.4f}")
    print(f"  saved → {args.save}")
