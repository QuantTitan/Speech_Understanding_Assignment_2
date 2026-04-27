"""
speaker_encoder.py  —  Task 3.1
=================================
Speaker embedding (d-vector) extraction from 60-second voice recordings.

Architecture
────────────
  Input  : Log-mel spectrogram  [B, T, 40]
  ↓
  3-layer LSTM   (hidden=768, proj=256)  ← generalization-aware
  ↓
  L2-normalized mean-pooled embedding    → d-vector  [B, 256]

Training objective: Generalized End-to-End (GE2E) loss
  (Wan et al., 2018, https://arxiv.org/abs/1710.10467)
  Pulls together embeddings of the same speaker, pushes apart
  embeddings of different speakers via softmax contrastive loss.

The trained encoder is used at inference to produce a 256-dim
speaker embedding from any ≥ 5 s utterance of the target voice.

Usage
─────
    enc = SpeakerEncoder.from_pretrained("checkpoints/speaker_enc.pt")
    dvec = enc.embed_utterance("my_voice_60s.wav")   # [256]

    # Or batch
    dvecs = enc.embed_batch(["speaker1.wav", "speaker2.wav"])  # [2, 256]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Ta


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

class SpeakerMelExtractor(nn.Module):
    """
    Log-mel spectrogram for speaker verification.

    Follows the d-vector / GE2E paper specs:
      • 25 ms window  (400 samples @ 16 kHz)
      • 10 ms hop     (160 samples)
      • 40 mel bins
      • log(mel + 1e-6)

    Returns [T, 40] per utterance, or [B, T, 40] batch.
    """

    SR        = 16_000
    N_FFT     = 400
    HOP       = 160
    N_MELS    = 40
    F_MIN     = 20.0
    F_MAX     = 7_600.0
    CLIP_VAL  = 1e-6

    def __init__(self) -> None:
        super().__init__()
        self.mel = Ta.MelSpectrogram(
            sample_rate=self.SR,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            n_mels=self.N_MELS,
            f_min=self.F_MIN,
            f_max=self.F_MAX,
            power=2.0,
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [T] or [B, T]  → [T', 40] or [B, T', 40]"""
        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav.unsqueeze(0)                   # [1, T]
        mel = self.mel(wav)                          # [B, 40, T']
        log_mel = torch.log(mel.clamp(min=self.CLIP_VAL))
        log_mel = log_mel.transpose(1, 2)            # [B, T', 40]
        if squeeze:
            log_mel = log_mel.squeeze(0)             # [T', 40]
        return log_mel


# ═══════════════════════════════════════════════════════════════════════════
# LSTM Speaker Encoder (d-vector)
# ═══════════════════════════════════════════════════════════════════════════

class SpeakerEncoder(nn.Module):
    """
    3-layer LSTM speaker encoder producing 256-dim L2-normalised d-vectors.

    Architecture matches the GE2E paper (Wan et al., 2018):
      • 3 LSTM layers, hidden=768, each followed by a linear projection
      • Final output: mean-pooled hidden states  → linear → L2 norm

    Parameters
    ----------
    n_mels      : Number of mel bins (default 40).
    hidden_dim  : LSTM hidden size (default 768).
    embed_dim   : Output d-vector dimensionality (default 256).
    n_layers    : Number of LSTM layers (default 3).
    """

    def __init__(
        self,
        n_mels:    int = 40,
        hidden_dim: int = 768,
        embed_dim:  int = 256,
        n_layers:   int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.n_layers   = n_layers

        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        # Final projection to embedding space
        self.proj = nn.Linear(hidden_dim, embed_dim)

        # Learnable scaling for GE2E loss (w, b)
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))

        self._feat = SpeakerMelExtractor()
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        mel: torch.Tensor,                   # [B, T, n_mels]
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                        # [B, embed_dim]
        """
        Encode a batch of mel sequences to d-vectors.

        Parameters
        ----------
        mel     : [B, T, n_mels]  log-mel features (padded).
        lengths : [B] int  actual sequence lengths (for mean pooling).

        Returns
        -------
        dvec : [B, embed_dim]  L2-normalised speaker embeddings.
        """
        B, T, _ = mel.shape

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                mel, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.lstm(mel)        # [B, T, hidden_dim]

        # Mean pooling over time (masked if lengths given)
        if lengths is not None:
            mask = (
                torch.arange(out.shape[1], device=out.device)
                .unsqueeze(0) < lengths.unsqueeze(1)
            ).float()                                 # [B, T]
            pooled = (out * mask.unsqueeze(-1)).sum(1) / lengths.float().unsqueeze(1)
        else:
            pooled = out.mean(dim=1)                  # [B, hidden_dim]

        emb = self.proj(pooled)                       # [B, embed_dim]
        return F.normalize(emb, p=2, dim=-1)          # L2 norm


    # ── Inference helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def embed_waveform(
        self,
        wav: torch.Tensor,         # [T] mono at 16 kHz
        segment_len_s: float = 1.6,
        overlap_s:     float = 0.4,
    ) -> torch.Tensor:             # [embed_dim]
        """
        Embed a long utterance by averaging embeddings over sliding windows.

        Parameters
        ----------
        wav          : [T] mono waveform at 16 kHz.
        segment_len_s: Window length in seconds (default 1.6 s = 256 frames).
        overlap_s    : Window overlap in seconds.

        Returns
        -------
        d-vector : [embed_dim]  averaged, L2-normalised.
        """
        self.eval()
        feat = self._feat.to(wav.device)
        sr   = SpeakerMelExtractor.SR
        hop  = SpeakerMelExtractor.HOP

        seg_samples = int(segment_len_s * sr)
        hop_samples = int((segment_len_s - overlap_s) * sr)

        # Segment the waveform
        segments: List[torch.Tensor] = []
        i = 0
        while i + seg_samples <= wav.shape[0]:
            segments.append(wav[i : i + seg_samples])
            i += hop_samples
        if not segments:
            segments = [wav]   # utterance shorter than window

        # Batch-extract mel and encode
        mels = [feat(seg) for seg in segments]          # list of [T', 40]
        max_T = max(m.shape[0] for m in mels)
        mel_pad = torch.zeros(len(mels), max_T, mels[0].shape[-1], device=wav.device)
        lengths = torch.zeros(len(mels), dtype=torch.long, device=wav.device)
        for k, m in enumerate(mels):
            mel_pad[k, : m.shape[0]] = m
            lengths[k] = m.shape[0]

        dvecs = self.forward(mel_pad, lengths)          # [N_segs, D]
        mean_dvec = dvecs.mean(dim=0)
        return F.normalize(mean_dvec, p=2, dim=-1)      # [D]

    @torch.no_grad()
    def embed_file(self, audio_path: str) -> torch.Tensor:
        """Load a WAV file and return its d-vector embedding."""
        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0)                            # mono
        if sr != SpeakerMelExtractor.SR:
            wav = torchaudio.functional.resample(wav, sr, SpeakerMelExtractor.SR)
        device = next(self.parameters()).device
        return self.embed_waveform(wav.to(device))

    @torch.no_grad()
    def embed_files(self, paths: List[str]) -> torch.Tensor:
        """Embed multiple files. Returns [N, embed_dim]."""
        return torch.stack([self.embed_file(p) for p in paths])

    def similarity_matrix(self, dvecs: torch.Tensor) -> torch.Tensor:
        """
        Compute GE2E cosine similarity matrix (w · cos + b).
        dvecs: [N, D]  → [N, N] scaled similarity.
        """
        cos = dvecs @ dvecs.T                             # [N, N]
        return self.w.abs() * cos + self.b

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(),
                    "config": {"n_mels": 40, "hidden_dim": self.hidden_dim,
                               "embed_dim": self.embed_dim, "n_layers": self.n_layers}},
                   path)
        print(f"[SpeakerEncoder] Saved → {path}")

    @classmethod
    def from_pretrained(cls, path: str) -> "SpeakerEncoder":
        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt.get("config", {})
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[SpeakerEncoder] Loaded ← {path}  embed_dim={model.embed_dim}")
        return model


# ═══════════════════════════════════════════════════════════════════════════
# GE2E Loss
# ═══════════════════════════════════════════════════════════════════════════

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End speaker verification loss (Wan et al., 2018).

    Expects embeddings arranged as [N_speakers, M_utterances, D].
    For each utterance, the centroid of the same speaker (excluding self)
    should be closest; centroids of other speakers should be far.

    Two variants:
      softmax  : log-softmax cross-entropy (better for large batches)
      contrast : pairwise contrastive (original paper default)
    """

    def __init__(self, variant: str = "softmax") -> None:
        super().__init__()
        assert variant in ("softmax", "contrast")
        self.variant = variant

    def forward(
        self,
        dvecs:   torch.Tensor,   # [N, M, D]  L2-normed embeddings
        w:       torch.Tensor,   # scalar, learnable scale
        b:       torch.Tensor,   # scalar, learnable bias
    ) -> torch.Tensor:
        N, M, D = dvecs.shape

        # Centroids: [N, D]  (exclude self for each centroid)
        centroids_incl = dvecs.mean(dim=1)             # [N, D] (inclusive)

        loss = torch.tensor(0.0, device=dvecs.device, requires_grad=True)

        for j in range(N):
            for i in range(M):
                # Centroid for speaker j excluding utterance i
                c_j = (centroids_incl[j] * M - dvecs[j, i]) / (M - 1)
                c_j = F.normalize(c_j, p=2, dim=-1)

                # Similarity of dvec[j,i] with all centroids
                sims = []
                for k in range(N):
                    c_k = centroids_incl[k] if k != j else c_j
                    c_k = F.normalize(c_k, p=2, dim=-1)
                    sim = w.abs() * F.cosine_similarity(
                        dvecs[j, i].unsqueeze(0), c_k.unsqueeze(0)
                    ) + b
                    sims.append(sim)
                sims = torch.stack(sims)               # [N]

                if self.variant == "softmax":
                    l = -sims[j] + torch.logsumexp(sims, dim=0)
                else:
                    pos  = torch.sigmoid(-sims[j])
                    neg  = torch.stack([
                        torch.sigmoid(sims[k]) for k in range(N) if k != j
                    ]).sum()
                    l = pos + neg
                loss = loss + l

        return loss / (N * M)


# ═══════════════════════════════════════════════════════════════════════════
# Training Scaffold
# ═══════════════════════════════════════════════════════════════════════════

class SpeakerEncoderTrainer:
    """
    Training loop for the GE2E speaker encoder.

    Dataset format
    ──────────────
    Directory tree: data/speakers/{speaker_id}/*.wav
    Each forward pass samples N_spk speakers × M_utt utterances.

    Parameters
    ----------
    encoder     : SpeakerEncoder to train.
    data_root   : Root path containing per-speaker subdirectories.
    N_spk       : Speakers per batch (default 64).
    M_utt       : Utterances per speaker per batch (default 10).
    lr          : Learning rate.
    """

    def __init__(
        self,
        encoder:   SpeakerEncoder,
        data_root: str,
        N_spk:     int   = 64,
        M_utt:     int   = 10,
        lr:        float = 1e-4,
        device:    str   = "cuda",
    ) -> None:
        self.encoder   = encoder.to(device)
        self.device    = device
        self.N_spk     = N_spk
        self.M_utt     = M_utt
        self.criterion = GE2ELoss(variant="softmax")
        self.feat      = SpeakerMelExtractor().to(device)

        # Build speaker → file list
        import random, os
        self.speakers: dict = {}
        root = Path(data_root)
        for spk_dir in root.iterdir():
            if spk_dir.is_dir():
                wavs = list(spk_dir.glob("*.wav"))
                if len(wavs) >= M_utt:
                    self.speakers[spk_dir.name] = wavs

        self.speaker_list = list(self.speakers.keys())
        print(f"[Trainer] {len(self.speaker_list)} speakers in {data_root}")

        # Separate LR for w, b vs main params (paper recommendation)
        self.optimizer = torch.optim.Adam([
            {"params": [p for n, p in encoder.named_parameters()
                        if n not in ("w", "b")], "lr": lr},
            {"params": [encoder.w, encoder.b], "lr": lr * 0.01},
        ])

    def _load_segment(self, path: Path) -> torch.Tensor:
        """Load a 1.6-second random segment from a WAV file."""
        import random
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0)
        if sr != SpeakerMelExtractor.SR:
            wav = torchaudio.functional.resample(wav, sr, SpeakerMelExtractor.SR)
        seg_len = int(1.6 * SpeakerMelExtractor.SR)
        if wav.shape[0] > seg_len:
            start = random.randint(0, wav.shape[0] - seg_len)
            wav = wav[start : start + seg_len]
        return wav.to(self.device)

    def train_step(self) -> float:
        """Run one GE2E batch update. Returns scalar loss."""
        import random
        self.encoder.train()

        # Sample N_spk speakers
        spk_sample = random.sample(self.speaker_list, min(self.N_spk, len(self.speaker_list)))

        dvec_batch = []
        for spk in spk_sample:
            files = random.sample(self.speakers[spk], self.M_utt)
            wavs  = [self._load_segment(f) for f in files]

            # Pad to equal length
            max_T = max(w.shape[0] for w in wavs)
            wav_pad = torch.zeros(self.M_utt, max_T, device=self.device)
            for k, w in enumerate(wavs):
                wav_pad[k, : w.shape[0]] = w

            mels = self.feat(wav_pad)                     # [M, T', 40]
            dvecs = self.encoder(mels)                    # [M, D]
            dvec_batch.append(dvecs)

        dvec_batch = torch.stack(dvec_batch)              # [N, M, D]

        loss = self.criterion(dvec_batch, self.encoder.w, self.encoder.b)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=3.0)
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        n_steps: int = 1_500_000,
        save_every: int = 10_000,
        save_dir: str = "checkpoints/speaker_enc",
    ) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Trainer] Training for {n_steps} steps …")
        for step in range(1, n_steps + 1):
            loss = self.train_step()
            if step % 1000 == 0:
                print(f"  Step {step:7d}  GE2E loss = {loss:.4f}")
            if step % save_every == 0:
                self.encoder.save(f"{save_dir}/speaker_enc_{step}.pt")
        self.encoder.save(f"{save_dir}/speaker_enc_final.pt")


# ═══════════════════════════════════════════════════════════════════════════
# 60-second Voice Registration Pipeline (Task 3.1 core)
# ═══════════════════════════════════════════════════════════════════════════

class VoiceRegistrar:
    """
    Registers a speaker's voice from a 60-second recording.

    Steps
    ─────
    1.  Load 60-second mono WAV at 16 kHz.
    2.  Remove silence frames (VAD).
    3.  Segment into overlapping 1.6 s windows.
    4.  Encode each window → d-vector.
    5.  Average + L2-normalise → final speaker embedding.
    6.  Save embedding to disk.

    Parameters
    ----------
    encoder_path : Path to a trained SpeakerEncoder checkpoint.
                   Pass None to use a randomly initialised encoder
                   (embedding will be semantically meaningless but
                   structurally correct for pipeline testing).
    device       : 'cuda' | 'cpu'.
    """

    MIN_DURATION_S  = 5.0    # minimum accepted recording duration
    TARGET_DURATION_S = 60.0

    def __init__(
        self,
        encoder_path: Optional[str] = None,
        device:       str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if encoder_path and Path(encoder_path).exists():
            self.encoder = SpeakerEncoder.from_pretrained(encoder_path).to(self.device)
        else:
            print("[VoiceRegistrar] WARNING: No checkpoint found. Using random encoder.")
            self.encoder = SpeakerEncoder().to(self.device)
        self.encoder.eval()

    def _vad_trim(self, wav: torch.Tensor, sr: int, top_db: float = 30.0) -> torch.Tensor:
        """Remove leading/trailing silence."""
        # Energy-based VAD
        frame = 512
        hop   = 128
        energy = wav.unfold(0, min(frame, wav.shape[0]), hop).pow(2).mean(-1)
        threshold = energy.max() / (10 ** (top_db / 10.0))
        mask = energy > threshold
        if mask.any():
            first = mask.nonzero()[0].item() * hop
            last  = (mask.nonzero()[-1].item() + 1) * hop
            wav   = wav[first:last]
        return wav

    def register(self, audio_path: str, save_path: Optional[str] = None) -> torch.Tensor:
        """
        Extract and return the speaker d-vector from a recording.

        Parameters
        ----------
        audio_path : Path to the 60-second voice recording WAV.
        save_path  : If given, save the embedding tensor (.pt file).

        Returns
        -------
        embedding : [embed_dim] float32 tensor.
        """
        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0)                               # mono
        if sr != SpeakerMelExtractor.SR:
            wav = torchaudio.functional.resample(wav, sr, SpeakerMelExtractor.SR)

        duration_s = wav.shape[0] / SpeakerMelExtractor.SR
        print(f"[VoiceRegistrar] Recording: {duration_s:.1f}s  path={audio_path}")

        if duration_s < self.MIN_DURATION_S:
            raise ValueError(
                f"Recording too short ({duration_s:.1f}s < {self.MIN_DURATION_S}s). "
                "Please record at least 5 seconds."
            )
        if duration_s < self.TARGET_DURATION_S:
            print(f"  WARNING: Recording is {duration_s:.1f}s < target {self.TARGET_DURATION_S}s.")

        wav = self._vad_trim(wav, SpeakerMelExtractor.SR)
        wav = wav.to(self.device)

        with torch.no_grad():
            embedding = self.encoder.embed_waveform(wav)

        print(f"  d-vector shape : {tuple(embedding.shape)}")
        print(f"  d-vector norm  : {embedding.norm().item():.4f}  (should be ≈1.0)")
        print(f"  d-vector mean  : {embedding.mean().item():.4f}")
        print(f"  d-vector std   : {embedding.std().item():.4f}")

        if save_path:
            torch.save(embedding.cpu(), save_path)
            print(f"  Saved → {save_path}")

        return embedding


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Speaker Encoder (Task 3.1)")
    p.add_argument("audio",       nargs="?", help="60-second voice WAV")
    p.add_argument("--encoder",   default=None, help="Checkpoint path")
    p.add_argument("--save",      default="my_dvector.pt")
    p.add_argument("--device",    default="cuda")
    args = p.parse_args()

    if args.audio:
        registrar = VoiceRegistrar(encoder_path=args.encoder, device=args.device)
        emb = registrar.register(args.audio, save_path=args.save)
        print(f"\nRegistration complete. d-vector saved to {args.save}")
    else:
        # Architecture smoke-test
        model = SpeakerEncoder()
        total = sum(p.numel() for p in model.parameters())
        print(f"SpeakerEncoder  params={total:,}")
        dummy = torch.randn(4, 160, 40)   # [B=4, T=160, n_mels=40]
        out   = model(dummy)
        print(f"Output shape  : {tuple(out.shape)}")
        print(f"Output norms  : {out.norm(dim=-1).tolist()}")

        # GE2E loss test
        criterion = GE2ELoss()
        dvecs = F.normalize(torch.randn(8, 10, 256), p=2, dim=-1)  # [N=8, M=10, D=256]
        loss  = criterion(dvecs, model.w, model.b)
        print(f"GE2E loss     : {loss.item():.4f}")
