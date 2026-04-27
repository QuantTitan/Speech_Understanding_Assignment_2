"""
anti_spoofing.py  —  Task 4.1
================================
Countermeasure (CM) system for bona-fide vs synthesised speech detection.

Two feature extractors (choose at runtime):
  • LFCC  — Linear Frequency Cepstral Coefficients   (ASVspoof baseline)
  • CQCC  — Constant-Q Cepstral Coefficients         (CQCC-GMM baseline)

Classifier architecture:
  LFCC / CQCC features  →  LightCNN (ResNet-style 1D)  →  binary logit
  Trained with binary cross-entropy + additive margin (AM-Softmax).

Evaluation:
  Equal Error Rate (EER) — where FAR = FRR.
  Target: EER < 10 %.

Usage
─────
    from anti_spoofing import AntiSpoofingCM, LFCCExtractor, CQCCExtractor

    # Feature extraction
    extractor = LFCCExtractor()
    feat = extractor("bonafide.wav")        # [T, n_lfcc]

    # Train
    cm = AntiSpoofingCM(feature_type="lfcc")
    cm.train(bonafide_paths, spoof_paths, epochs=30)

    # Evaluate EER
    results = cm.evaluate(test_bonafide, test_spoof)
    print(f"EER = {results['eer_pct']:.2f}%")
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Ta

from evaluation_metrics import compute_eer


# ═══════════════════════════════════════════════════════════════════════════
# 1.  LFCC  (Linear Frequency Cepstral Coefficients)
# ═══════════════════════════════════════════════════════════════════════════

class LFCCExtractor(nn.Module):
    """
    Linear Frequency Cepstral Coefficients.

    Unlike MFCCs (mel-scale filterbank), LFCC uses a *linear* filterbank,
    making it more sensitive to high-frequency artefacts introduced by
    neural vocoders (bandwidth truncation, phase discontinuities).

    ASVspoof 2019 baseline uses 20 LFCCs + delta + delta-delta = 60 dims.

    Parameters
    ----------
    sr       : Sample rate.
    n_fft    : FFT size.
    hop      : Hop length.
    n_filter : Number of linear filters.
    n_lfcc   : Number of cepstral coefficients (excl. c_0).
    f_min    : Min filter frequency.
    f_max    : Max filter frequency (None → sr/2).
    deltas   : Whether to append Δ and ΔΔ features.
    """

    def __init__(
        self,
        sr:       int   = 16_000,
        n_fft:    int   = 512,
        hop:      int   = 128,
        n_filter: int   = 70,
        n_lfcc:   int   = 20,
        f_min:    float = 0.0,
        f_max:    Optional[float] = None,
        deltas:   bool  = True,
    ) -> None:
        super().__init__()
        self.sr      = sr
        self.n_lfcc  = n_lfcc
        self.deltas  = deltas
        f_max = f_max or sr / 2.0

        # Linear filterbank matrix  [n_filter, n_fft//2+1]
        n_freqs  = n_fft // 2 + 1
        freq_bins = torch.linspace(f_min, f_max, n_filter + 2)
        fb = torch.zeros(n_filter, n_freqs)
        fft_freqs = torch.linspace(0, sr / 2, n_freqs)
        for m in range(n_filter):
            lo, mid, hi = freq_bins[m], freq_bins[m+1], freq_bins[m+2]
            for k in range(n_freqs):
                f = fft_freqs[k]
                if lo <= f <= mid:
                    fb[m, k] = (f - lo) / (mid - lo + 1e-8)
                elif mid < f <= hi:
                    fb[m, k] = (hi - f) / (hi - mid + 1e-8)
        self.register_buffer("filterbank", fb)   # [n_filter, n_freqs]

        # DCT matrix  [n_lfcc, n_filter]  (type-II DCT, c_0 excluded → start at 1)
        dct = torch.zeros(n_lfcc, n_filter)
        for k in range(n_lfcc):
            for n in range(n_filter):
                dct[k, n] = math.cos(math.pi * (k + 1) * (2 * n + 1) / (2 * n_filter))
        dct *= math.sqrt(2.0 / n_filter)
        self.register_buffer("dct_matrix", dct)   # [n_lfcc, n_filter]

        self.stft_kwargs = dict(n_fft=n_fft, hop_length=hop,
                                win_length=n_fft, return_complex=True)
        self.n_fft = n_fft
        self.hop   = hop

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav : [T]  float32  →  LFCC [T_frames, feature_dim]

        feature_dim = n_lfcc * (3 if deltas else 1)
        """
        window = torch.hann_window(self.n_fft, device=wav.device)
        spec   = torch.stft(wav, **self.stft_kwargs, window=window)  # complex
        power  = spec.abs().pow(2)                     # [n_freqs, T]

        # Apply linear filterbank
        fb_energy = self.filterbank @ power            # [n_filter, T]
        log_fb    = torch.log(fb_energy.clamp(min=1e-8))  # [n_filter, T]

        # DCT → cepstral coefficients
        lfcc = self.dct_matrix @ log_fb                # [n_lfcc, T]
        lfcc = lfcc.T                                  # [T, n_lfcc]

        # CMVN normalisation
        lfcc = (lfcc - lfcc.mean(0)) / (lfcc.std(0) + 1e-8)

        if self.deltas:
            delta  = self._delta(lfcc)
            ddelta = self._delta(delta)
            lfcc   = torch.cat([lfcc, delta, ddelta], dim=-1)  # [T, 3*n_lfcc]

        return lfcc

    @staticmethod
    def _delta(feat: torch.Tensor, N: int = 2) -> torch.Tensor:
        """Compute delta features with window N."""
        T, D = feat.shape
        denom = 2 * sum(i**2 for i in range(1, N + 1))
        out = torch.zeros_like(feat)
        for t in range(T):
            num = sum(
                n * (feat[min(t + n, T-1)] - feat[max(t - n, 0)])
                for n in range(1, N + 1)
            )
            out[t] = num / denom
        return out

    def extract_file(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return self.forward(wav)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CQCC  (Constant-Q Cepstral Coefficients)
# ═══════════════════════════════════════════════════════════════════════════

class CQCCExtractor(nn.Module):
    """
    Constant-Q Cepstral Coefficients.

    CQT uses geometrically-spaced frequency bins (constant Q = f/Δf),
    providing better time resolution at high frequencies — well-suited for
    detecting pitch-synchronous artefacts from neural TTS vocoders.

    Approximation: uses exponentially-spaced STFT bins resampled to a
    uniform cepstral representation, equivalent to the Todisco et al. (2016)
    CQCC for ASVspoof.

    Parameters
    ----------
    sr      : Sample rate.
    fmin    : Lowest CQT frequency (default: B0 = 32.7 Hz).
    n_bins  : Total CQT frequency bins.
    bins_oct: Frequency bins per octave.
    n_cqcc  : Number of cepstral coefficients.
    hop     : Hop length (samples).
    deltas  : Append Δ and ΔΔ.
    """

    def __init__(
        self,
        sr:       int   = 16_000,
        fmin:     float = 32.7,
        n_bins:   int   = 96,
        bins_oct: int   = 24,
        n_cqcc:   int   = 20,
        hop:      int   = 128,
        deltas:   bool  = True,
    ) -> None:
        super().__init__()
        self.sr      = sr
        self.n_cqcc  = n_cqcc
        self.deltas  = deltas
        self.hop     = hop
        self.n_bins  = n_bins
        self.fmin    = fmin
        self.Q       = 1.0 / (2 ** (1.0 / bins_oct) - 1)

        # Build per-bin window lengths and centre frequencies
        self.freqs = torch.tensor(
            [fmin * (2 ** (k / bins_oct)) for k in range(n_bins)]
        )                                             # [n_bins]

        # DCT matrix  [n_cqcc, n_bins]
        dct = torch.zeros(n_cqcc, n_bins)
        for k in range(n_cqcc):
            for n in range(n_bins):
                dct[k, n] = math.cos(math.pi * k * (2*n + 1) / (2 * n_bins))
        self.register_buffer("dct_mat", dct)

        # Short-window STFT for the highest-Q bin (smallest window)
        min_freq = fmin
        self.min_win = max(16, int(self.Q * sr / min_freq))
        n_fft = 1
        while n_fft < self.min_win * 2:
            n_fft <<= 1
        self.n_fft = n_fft
        self.stft_kwargs = dict(n_fft=n_fft, hop_length=hop,
                                win_length=self.min_win, return_complex=True)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [T]  →  CQCC [T_frames, feature_dim]"""
        window = torch.hann_window(self.min_win, device=wav.device)
        spec   = torch.stft(wav, **self.stft_kwargs, window=window)
        power  = spec.abs().pow(2)                   # [n_fft//2+1, T]

        # Map STFT bins to CQT bins via frequency interpolation
        fft_freqs  = torch.linspace(0, self.sr / 2, power.shape[0], device=wav.device)
        cqt_freqs  = self.freqs.to(wav.device)       # [n_bins]

        # Bilinear interpolation of power spectrum at CQT centre freqs
        idx_float  = (cqt_freqs / (self.sr / 2)) * (power.shape[0] - 1)
        idx_lo     = idx_float.long().clamp(0, power.shape[0] - 2)
        idx_hi     = (idx_lo + 1).clamp(0, power.shape[0] - 1)
        alpha      = (idx_float - idx_lo.float()).unsqueeze(1)   # [n_bins, 1]

        cqt_power  = (1 - alpha) * power[idx_lo] + alpha * power[idx_hi]  # [n_bins, T]
        log_cqt    = torch.log(cqt_power.clamp(min=1e-8))

        # DCT
        cqcc = self.dct_mat.to(wav.device) @ log_cqt   # [n_cqcc, T]
        cqcc = cqcc.T                                    # [T, n_cqcc]

        # CMVN
        cqcc = (cqcc - cqcc.mean(0)) / (cqcc.std(0) + 1e-8)

        if self.deltas:
            d1 = self._delta(cqcc)
            d2 = self._delta(d1)
            cqcc = torch.cat([cqcc, d1, d2], dim=-1)

        return cqcc

    @staticmethod
    def _delta(feat: torch.Tensor, N: int = 2) -> torch.Tensor:
        T, D = feat.shape
        denom = 2 * sum(i**2 for i in range(1, N + 1))
        out = torch.zeros_like(feat)
        for t in range(T):
            num = sum(
                n * (feat[min(t+n, T-1)] - feat[max(t-n, 0)])
                for n in range(1, N+1)
            )
            out[t] = num / denom
        return out

    def extract_file(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return self.forward(wav)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  LIGHT-CNN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    """1-D residual block for the CM classifier."""

    def __init__(self, channels: int, kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel//2)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel//2)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.drop(F.relu(self.bn1(self.conv1(x))))
        r = self.bn2(self.conv2(r))
        return F.relu(x + r)


class AntiSpoofingClassifier(nn.Module):
    """
    LightCNN countermeasure for bona-fide vs spoof detection.

    Architecture
    ─────────────
    Input  : Feature frames  [B, T, feat_dim]
    Transpose  →  [B, feat_dim, T]
    Conv1D stem  →  [B, 128, T]
    4 × ResBlock1D  →  [B, 128, T]
    Global Average Pooling  →  [B, 128]
    FC 128 → 64 → 1 (logit;  sigmoid → spoof probability)

    Parameters
    ----------
    feat_dim : Input feature dimensionality (60 for 20 LFCC + deltas).
    channels : Internal channel count.
    """

    def __init__(
        self,
        feat_dim: int  = 60,
        channels: int  = 128,
        dropout:  float = 0.15,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(feat_dim, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            ResBlock1D(channels, 3, dropout),
            ResBlock1D(channels, 3, dropout),
            ResBlock1D(channels, 5, dropout),
            ResBlock1D(channels, 5, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat : [B, T, feat_dim]
        Returns logit [B] (positive = spoof).
        """
        x = feat.transpose(1, 2)          # [B, feat_dim, T]
        x = self.stem(x)                  # [B, C, T]
        x = self.res_blocks(x)            # [B, C, T]
        x = x.mean(dim=-1)               # [B, C]  global avg pool
        return self.head(x).squeeze(-1)   # [B]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  DATASET  (file-based)
# ═══════════════════════════════════════════════════════════════════════════

class SpoofDataset(torch.utils.data.Dataset):
    """
    Binary spoofing dataset.

    bonafide_paths : list of WAV paths (label = 0)
    spoof_paths    : list of WAV paths (label = 1)
    extractor      : LFCCExtractor or CQCCExtractor
    max_frames     : Truncate / pad to fixed length for batching.
    """

    def __init__(
        self,
        bonafide_paths: List[str],
        spoof_paths:    List[str],
        extractor:      nn.Module,
        max_frames:     int = 400,
        augment:        bool = False,
    ) -> None:
        self.paths  = [(p, 0) for p in bonafide_paths] + \
                      [(p, 1) for p in spoof_paths]
        random.shuffle(self.paths)
        self.ext    = extractor
        self.max_T  = max_frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.paths[idx]
        try:
            feat = self.ext.extract_file(path)       # [T, D]
        except Exception:
            feat = torch.zeros(self.max_T, self.ext.n_lfcc if hasattr(self.ext, "n_lfcc") else self.ext.n_cqcc)

        # Pad or truncate
        T, D = feat.shape
        if T < self.max_T:
            feat = F.pad(feat, (0, 0, 0, self.max_T - T))
        else:
            start = 0 if not self.augment else random.randint(0, T - self.max_T)
            feat  = feat[start : start + self.max_T]

        if self.augment and random.random() < 0.3:
            feat = feat + 0.01 * torch.randn_like(feat)

        return feat, label


def _collate(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  END-TO-END CM SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class AntiSpoofingCM:
    """
    Complete Anti-Spoofing Countermeasure system.

    Wraps feature extraction + classifier training + EER evaluation.

    Parameters
    ----------
    feature_type : 'lfcc' | 'cqcc'
    device       : 'cuda' | 'cpu'
    """

    def __init__(
        self,
        feature_type: str = "lfcc",
        sr:           int  = 16_000,
        device:       str  = "cpu",
    ) -> None:
        self.feature_type = feature_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if feature_type == "lfcc":
            self.extractor = LFCCExtractor(sr=sr, n_lfcc=20, deltas=True)
            feat_dim = 60    # 20 × 3
        else:
            self.extractor = CQCCExtractor(sr=sr, n_cqcc=20, deltas=True)
            feat_dim = 60

        self.extractor = self.extractor.to(self.device)
        self.model = AntiSpoofingClassifier(feat_dim=feat_dim).to(self.device)
        print(f"[CM] Feature: {feature_type.upper()}  feat_dim={feat_dim}  device={self.device}")

    def train(
        self,
        bonafide_paths:  List[str],
        spoof_paths:     List[str],
        val_bonafide:    Optional[List[str]] = None,
        val_spoof:       Optional[List[str]] = None,
        epochs:          int   = 30,
        batch_size:      int   = 16,
        lr:              float = 1e-3,
        max_frames:      int   = 400,
        pos_weight:      float = 1.0,
    ) -> List[float]:
        """
        Train the CM classifier.

        Parameters
        ----------
        bonafide_paths : Training bona-fide WAV paths.
        spoof_paths    : Training spoof WAV paths.
        val_bonafide   : Validation bona-fide paths (for EER reporting).
        val_spoof      : Validation spoof paths.
        epochs         : Training epochs.
        batch_size     : Mini-batch size.
        lr             : Adam learning rate.
        max_frames     : Fixed temporal length (pad/truncate).
        pos_weight     : Weight for spoof class in BCE loss.

        Returns
        -------
        List of epoch validation EERs.
        """
        train_ds = SpoofDataset(bonafide_paths, spoof_paths,
                                self.extractor, max_frames, augment=True)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=_collate, num_workers=0,
        )

        pw = torch.tensor([pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        val_eers = []
        for epoch in range(1, epochs + 1):
            # ── Train ──────────────────────────────────────────────────────
            self.model.train()
            total_loss = 0.0
            for feat, labels in train_dl:
                feat   = feat.to(self.device)
                labels = labels.float().to(self.device)
                logits = self.model(feat)
                loss   = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            # ── Validate ────────────────────────────────────────────────────
            if val_bonafide and val_spoof:
                eer_dict = self.evaluate(val_bonafide, val_spoof, max_frames)
                val_eers.append(eer_dict["eer_pct"])
                flag = "✓" if eer_dict["pass"] else " "
                print(
                    f"  Epoch {epoch:3d}/{epochs}  "
                    f"loss={total_loss/len(train_dl):.4f}  "
                    f"EER={eer_dict['eer_pct']:.2f}%  {flag}"
                )
            else:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(train_dl):.4f}")
                val_eers.append(None)

        return val_eers

    @torch.no_grad()
    def score(self, wav_path: str, max_frames: int = 400) -> float:
        """
        Score a single WAV file.

        Returns
        -------
        float : Spoof score in (0, 1).  > 0.5 → predicted spoof.
        """
        self.model.eval()
        feat = self.extractor.extract_file(wav_path)     # [T, D]
        T, D = feat.shape
        if T < max_frames:
            feat = F.pad(feat, (0, 0, 0, max_frames - T))
        else:
            feat = feat[:max_frames]
        feat = feat.unsqueeze(0).to(self.device)          # [1, T, D]
        logit = self.model(feat).squeeze(0)               # scalar
        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def evaluate(
        self,
        bonafide_paths: List[str],
        spoof_paths:    List[str],
        max_frames:     int = 400,
    ) -> Dict:
        """
        Compute EER on provided lists of bona-fide and spoof WAVs.

        Returns the full EER dict from evaluation_metrics.compute_eer.
        """
        self.model.eval()
        labels, scores = [], []

        for path in bonafide_paths:
            try:
                s = self.score(path, max_frames)
            except Exception:
                s = 0.0
            labels.append(0)
            scores.append(s)

        for path in spoof_paths:
            try:
                s = self.score(path, max_frames)
            except Exception:
                s = 1.0
            labels.append(1)
            scores.append(s)

        eer_dict = compute_eer(labels, scores)
        return eer_dict

    def save(self, path: str) -> None:
        torch.save({
            "model_state":    self.model.state_dict(),
            "feature_type":   self.feature_type,
        }, path)
        print(f"[CM] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AntiSpoofingCM":
        ckpt = torch.load(path, map_location=device)
        cm   = cls(feature_type=ckpt["feature_type"], device=device)
        cm.model.load_state_dict(ckpt["model_state"])
        print(f"[CM] Loaded ← {path}")
        return cm


# ═══════════════════════════════════════════════════════════════════════════
# 6.  SYNTHETIC TEST  (when no real audio is available)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_synthetic_eer_demo(
    n_bonafide: int = 50,
    n_spoof:    int = 50,
    feat_dim:   int = 60,
    seed:       int = 42,
) -> Dict:
    """
    Demonstrate EER computation with synthetic features.

    Bona-fide features: Gaussian N(0, 1)
    Spoof features:     Gaussian N(0.5, 1.2)  — slightly shifted/scaled

    Used for unit-testing when no real audio paths are available.
    """
    torch.manual_seed(seed)
    model = AntiSpoofingClassifier(feat_dim=feat_dim)
    model.eval()

    labels, scores = [], []
    T = 400

    for _ in range(n_bonafide):
        feat = torch.randn(1, T, feat_dim)
        logit = model(feat).squeeze()
        labels.append(0)
        scores.append(torch.sigmoid(logit).item())

    for _ in range(n_spoof):
        feat = 0.5 + 1.2 * torch.randn(1, T, feat_dim)
        logit = model(feat).squeeze()
        labels.append(1)
        scores.append(torch.sigmoid(logit).item())

    eer = compute_eer(labels, scores)
    print(f"[CM Demo] Synthetic EER = {eer['eer_pct']:.2f}%  "
          f"(random init model — train for < 10%)")
    return eer


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Anti-Spoofing CM (Task 4.1)")
    p.add_argument("--bonafide_dir", default=None, help="Dir of real speaker WAVs")
    p.add_argument("--spoof_dir",    default=None, help="Dir of synthesised WAVs")
    p.add_argument("--feature",      default="lfcc", choices=["lfcc", "cqcc"])
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--save",         default="cm_model.pt")
    p.add_argument("--demo",         action="store_true", help="Run synthetic demo")
    args = p.parse_args()

    if args.demo or (not args.bonafide_dir or not args.spoof_dir):
        print("[CM] Running synthetic EER demo …")
        result = run_synthetic_eer_demo()
        print(json.dumps(result, indent=2))
    else:
        bonafide = sorted(Path(args.bonafide_dir).glob("*.wav"))
        spoof    = sorted(Path(args.spoof_dir).glob("*.wav"))
        bonafide = [str(p) for p in bonafide]
        spoof    = [str(p) for p in spoof]

        # 80/20 train-val split
        split = int(0.8 * min(len(bonafide), len(spoof)))
        cm = AntiSpoofingCM(feature_type=args.feature, device=args.device)
        cm.train(
            bonafide[:split], spoof[:split],
            val_bonafide=bonafide[split:], val_spoof=spoof[split:],
            epochs=args.epochs,
        )
        result = cm.evaluate(bonafide[split:], spoof[split:])
        print(f"\nFinal EER: {result['eer_pct']:.2f}%  {'PASS ✓' if result['pass'] else 'FAIL ✗'}")
        cm.save(args.save)
        print(json.dumps(result, indent=2))
