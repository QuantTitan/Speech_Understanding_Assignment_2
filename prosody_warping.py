"""
prosody_warping.py  —  Task 3.2
=================================
Prosody extraction and Dynamic Time Warping (DTW) transfer.

Goal: Extract F0 (fundamental frequency) and energy contours from the
professor's lecture and warp them onto the synthesised Santhali speech,
preserving "teaching style" (pitch dynamics, emphasis patterns, rhythm).

Pipeline
────────
  1. F0 Extraction      – RAPT-style SHR + autocorrelation (pure PyTorch)
  2. Energy Extraction  – RMS energy per frame
  3. Voiced/Unvoiced    – Energy + Zero-Crossing Rate classifier
  4. DTW Alignment      – Sakoe-Chiba band constrained DTW
  5. Prosody Warp       – Apply warped F0/energy via PSOLA-inspired
                          time-domain pitch shifting + amplitude scaling

All computations are pure PyTorch (no scipy, librosa, or praat-parselmouth).

Usage
─────
    warper = ProsodyWarper(sample_rate=22050)

    # Extract reference prosody from professor's lecture
    ref_prosody = warper.extract_prosody("lecture_segment.wav")

    # Warp onto synthesised Santhali audio
    warped_wav = warper.warp(synth_wav, ref_prosody)

    torchaudio.save("output_warped.wav", warped_wav.unsqueeze(0), 22050)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ═══════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProsodyFeatures:
    """Frame-level prosodic features for one utterance."""
    f0:          torch.Tensor    # [T]  Hz,  0.0 = unvoiced frame
    energy:      torch.Tensor    # [T]  RMS energy (linear scale)
    voiced_mask: torch.Tensor    # [T]  bool
    frame_len:   int             # samples per frame
    hop_len:     int             # samples per hop
    sample_rate: int

    @property
    def n_frames(self) -> int:
        return self.f0.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.hop_len / self.sample_rate

    def to(self, device) -> "ProsodyFeatures":
        return ProsodyFeatures(
            f0=self.f0.to(device), energy=self.energy.to(device),
            voiced_mask=self.voiced_mask.to(device),
            frame_len=self.frame_len, hop_len=self.hop_len,
            sample_rate=self.sample_rate,
        )


# ═══════════════════════════════════════════════════════════════════════════
# F0 Extractor (autocorrelation-based, PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

class AutocorrF0Extractor(nn.Module):
    """
    Autocorrelation-based fundamental frequency extractor.

    Method (Boersma 1993, simplified):
      For each voiced frame:
      1. Apply Gaussian window
      2. Compute normalised autocorrelation via FFT
      3. Find the dominant peak in the lag range [T_min, T_max]
         corresponding to [f0_min, f0_max] Hz
      4. Parabolic interpolation for sub-sample precision

    Voicing decision uses a combination of:
      • Autocorrelation peak strength (SHR proxy)
      • Short-time energy threshold
      • Zero-crossing rate threshold

    Parameters
    ----------
    sample_rate : Audio sample rate.
    frame_len   : Analysis window length in samples.
    hop_len     : Hop length in samples.
    f0_min      : Minimum F0 in Hz (default 60 Hz).
    f0_max      : Maximum F0 in Hz (default 500 Hz).
    voiced_thr  : Normalised autocorrelation threshold for voicing.
    """

    def __init__(
        self,
        sample_rate: int   = 22_050,
        frame_len:   int   = 1024,
        hop_len:     int   = 256,
        f0_min:      float = 60.0,
        f0_max:      float = 500.0,
        voiced_thr:  float = 0.45,
    ) -> None:
        super().__init__()
        self.sr        = sample_rate
        self.frame_len = frame_len
        self.hop_len   = hop_len
        self.f0_min    = f0_min
        self.f0_max    = f0_max
        self.voiced_thr = voiced_thr

        # Precompute lag range in samples
        self.lag_min = max(1, int(sample_rate / f0_max))
        self.lag_max = min(frame_len - 1, int(sample_rate / f0_min))

        # Gaussian window
        t = torch.arange(frame_len).float()
        sigma = frame_len / 4.0
        window = torch.exp(-0.5 * ((t - frame_len / 2) / sigma) ** 2)
        self.register_buffer("window", window)

    def _frame_signal(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [T]  →  frames: [N_frames, frame_len]"""
        n_frames = max(1, (wav.shape[0] - self.frame_len) // self.hop_len + 1)
        frames = torch.zeros(n_frames, self.frame_len, device=wav.device)
        for i in range(n_frames):
            start = i * self.hop_len
            end   = start + self.frame_len
            seg   = wav[start : min(end, wav.shape[0])]
            frames[i, : seg.shape[0]] = seg
        return frames

    def _normalised_autocorr(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised autocorrelation for each frame.

        frames: [N, frame_len]
        Returns: [N, frame_len]   r[τ] / r[0]
        """
        w = frames * self.window.unsqueeze(0)          # apply Gaussian window
        n_fft = 2 * self.frame_len                     # zero-pad for circular free
        W = torch.fft.rfft(w, n=n_fft)                # [N, n_fft//2+1]
        power = W * W.conj()                           # |W(f)|²
        acf   = torch.fft.irfft(power, n=n_fft)       # [N, n_fft]
        acf   = acf[:, : self.frame_len]               # [N, frame_len]

        # Normalise by zero-lag (energy)
        r0 = acf[:, 0].clamp(min=1e-8)
        return acf / r0.unsqueeze(1)                   # [N, frame_len]

    def _pick_peak(self, acf_row: torch.Tensor) -> Tuple[float, float]:
        """
        Find the dominant peak in acf[lag_min : lag_max+1].
        Returns (f0_hz, peak_strength).
        Uses parabolic interpolation for sub-sample precision.
        """
        region = acf_row[self.lag_min : self.lag_max + 1]  # [lag_range]
        if region.shape[0] < 3:
            return 0.0, 0.0

        peak_idx = region.argmax().item()
        strength = region[peak_idx].item()

        # Parabolic interpolation
        if 0 < peak_idx < region.shape[0] - 1:
            y0 = region[peak_idx - 1].item()
            y1 = region[peak_idx].item()
            y2 = region[peak_idx + 1].item()
            denom = 2 * (2 * y1 - y0 - y2)
            if abs(denom) > 1e-8:
                delta = (y0 - y2) / denom
                peak_idx += delta

        lag   = self.lag_min + peak_idx
        f0_hz = self.sr / lag if lag > 0 else 0.0
        return f0_hz, strength

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract F0, energy, and voicing mask.

        Parameters
        ----------
        wav : [T]  mono waveform at self.sr.

        Returns
        -------
        f0          : [N_frames]  F0 in Hz (0 = unvoiced).
        energy      : [N_frames]  RMS energy.
        voiced_mask : [N_frames]  bool.
        """
        frames = self._frame_signal(wav)               # [N, frame_len]
        acf    = self._normalised_autocorr(frames)     # [N, frame_len]

        # RMS energy
        energy = frames.pow(2).mean(dim=-1).sqrt()     # [N]

        # Zero-crossing rate
        zcr = ((frames[:, 1:] * frames[:, :-1]) < 0).float().mean(dim=-1)  # [N]

        # F0 + strength
        n_frames = frames.shape[0]
        f0_arr   = torch.zeros(n_frames, device=wav.device)
        strength_arr = torch.zeros(n_frames, device=wav.device)

        for i in range(n_frames):
            f0_hz, strength = self._pick_peak(acf[i])
            f0_arr[i]       = f0_hz
            strength_arr[i] = strength

        # Voicing decision: strong autocorr peak + sufficient energy + low ZCR
        energy_thr  = energy.max().clamp(min=1e-8) * 0.05
        voiced_mask = (
            (strength_arr >= self.voiced_thr) &
            (energy > energy_thr) &
            (zcr < 0.30)
        )

        # Zero out F0 for unvoiced frames
        f0_arr = f0_arr * voiced_mask.float()

        return f0_arr, energy, voiced_mask


# ═══════════════════════════════════════════════════════════════════════════
# DTW (Sakoe-Chiba band constrained)
# ═══════════════════════════════════════════════════════════════════════════

class DTW(nn.Module):
    """
    Dynamic Time Warping with Sakoe-Chiba band constraint.

    Used to align reference (professor) and target (synthesised) prosody
    sequences before applying the transfer.

    Parameters
    ----------
    band_radius : Sakoe-Chiba band half-width (frames).
                  None = unconstrained (classic DTW).

    Returns
    -------
    cost        : scalar DTW distance.
    path        : [(ref_idx, tgt_idx), ...]  alignment path.
    """

    def __init__(self, band_radius: Optional[int] = None) -> None:
        super().__init__()
        self.band_radius = band_radius

    @torch.no_grad()
    def forward(
        self,
        ref: torch.Tensor,     # [T_ref, D] or [T_ref]
        tgt: torch.Tensor,     # [T_tgt, D] or [T_tgt]
    ) -> Tuple[torch.Tensor, list]:
        """
        Compute DTW cost and alignment path.

        Supports:
          • 1D sequences (F0, energy) — scalar distance |r - t|
          • 2D sequences (mel vectors) — Euclidean distance
        """
        if ref.dim() == 1:
            ref = ref.unsqueeze(1)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(1)

        T, D = ref.shape
        S    = tgt.shape[0]

        INF = torch.tensor(float("inf"), device=ref.device)

        # Distance matrix [T, S]
        dist = torch.cdist(ref.float(), tgt.float(), p=2)   # [T, S]

        # Accumulated cost matrix
        acc  = torch.full((T, S), float("inf"), device=ref.device)
        acc[0, 0] = dist[0, 0]

        for i in range(T):
            for j in range(S):
                if i == 0 and j == 0:
                    continue
                # Sakoe-Chiba band
                if self.band_radius is not None:
                    if abs(i - j) > self.band_radius:
                        continue

                candidates = []
                if i > 0:              candidates.append(acc[i-1, j])
                if j > 0:              candidates.append(acc[i, j-1])
                if i > 0 and j > 0:    candidates.append(acc[i-1, j-1])

                if candidates:
                    acc[i, j] = dist[i, j] + min(candidates)

        cost = acc[T-1, S-1]

        # Backtrack path
        path = self._backtrack(acc, T, S)
        return cost, path

    @staticmethod
    def _backtrack(acc: torch.Tensor, T: int, S: int) -> list:
        path = [(T-1, S-1)]
        i, j = T-1, S-1
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                tri = torch.stack([acc[i-1, j], acc[i, j-1], acc[i-1, j-1]])
                best = tri.argmin().item()
                if best == 0:   i -= 1
                elif best == 1: j -= 1
                else:           i -= 1; j -= 1
            path.append((i, j))
        return list(reversed(path))


# ═══════════════════════════════════════════════════════════════════════════
# PSOLA-inspired Pitch Shifting (pure PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

class PSOLAPitchShifter(nn.Module):
    """
    TD-PSOLA (Time-Domain Pitch-Synchronous Overlap-Add) pitch shifter.

    For each voiced frame, adjusts the pitch period to match the target F0
    by resampling the local waveform segment and overlap-adding.

    Parameters
    ----------
    sample_rate : Audio sample rate.
    frame_len   : Analysis frame length (should match F0 extractor).
    hop_len     : Hop length.
    """

    def __init__(
        self,
        sample_rate: int = 22_050,
        frame_len:   int = 1024,
        hop_len:     int = 256,
    ) -> None:
        super().__init__()
        self.sr        = sample_rate
        self.frame_len = frame_len
        self.hop_len   = hop_len

    @torch.no_grad()
    def shift_frame(
        self,
        frame:   torch.Tensor,   # [frame_len]
        src_f0:  float,          # source F0 in Hz
        tgt_f0:  float,          # target F0 in Hz
    ) -> torch.Tensor:           # [frame_len]
        """Pitch-shift a single frame from src_f0 to tgt_f0."""
        if src_f0 < 1.0 or tgt_f0 < 1.0:
            return frame                          # unvoiced: no shift

        ratio = src_f0 / tgt_f0                  # >1 = lower pitch, <1 = higher
        ratio = max(0.25, min(4.0, ratio))        # clamp to ±2 octaves

        L = self.frame_len
        # Resample the frame to simulate pitch change
        new_len = max(1, round(L * ratio))
        # Use 1D interpolation to resample
        frame_resamp = F.interpolate(
            frame.view(1, 1, -1).float(),
            size=new_len,
            mode="linear",
            align_corners=False,
        ).view(-1)

        # Trim or pad back to original length
        if frame_resamp.shape[0] >= L:
            return frame_resamp[:L]
        else:
            out = torch.zeros(L, device=frame.device)
            out[: frame_resamp.shape[0]] = frame_resamp
            return out

    @torch.no_grad()
    def apply(
        self,
        wav:           torch.Tensor,   # [T]
        src_f0:        torch.Tensor,   # [N_frames] Hz
        tgt_f0:        torch.Tensor,   # [N_frames] Hz (from DTW-warped ref)
        voiced_mask:   torch.Tensor,   # [N_frames] bool
    ) -> torch.Tensor:                 # [T]
        """
        Apply frame-wise pitch shifting via overlap-add synthesis.

        Parameters
        ----------
        wav         : Input waveform.
        src_f0      : F0 of the input (synthesised) waveform, frame-level.
        tgt_f0      : Target F0 (from professor, DTW-aligned), frame-level.
        voiced_mask : Combined voiced mask.

        Returns
        -------
        Pitch-shifted waveform [T].
        """
        n_frames = src_f0.shape[0]
        output   = torch.zeros_like(wav)
        weights  = torch.zeros_like(wav)

        hann = torch.hann_window(self.frame_len, device=wav.device)

        for i in range(n_frames):
            start = i * self.hop_len
            end   = start + self.frame_len
            if end > wav.shape[0]:
                break

            frame = wav[start:end]

            if voiced_mask[i]:
                frame = self.shift_frame(
                    frame,
                    src_f0[i].item(),
                    tgt_f0[i].item(),
                )

            # Overlap-add with Hann window
            output[start:end] += frame * hann
            weights[start:end] += hann

        # Normalise by overlap-add weights
        mask = weights > 1e-8
        output[mask] /= weights[mask]
        return output


# ═══════════════════════════════════════════════════════════════════════════
# Energy Scaler
# ═══════════════════════════════════════════════════════════════════════════

class EnergyScaler(nn.Module):
    """
    Frame-wise amplitude scaling to match reference energy contour.

    Applies:  out[frame] = in[frame] × sqrt(E_ref[frame] / E_src[frame])

    Parameters
    ----------
    smooth_frames : Number of frames over which to smooth the gain
                    (median-filter to avoid abrupt jumps).
    max_gain      : Hard clamp on gain (prevents over-amplification).
    """

    def __init__(self, smooth_frames: int = 5, max_gain: float = 4.0) -> None:
        super().__init__()
        self.smooth = smooth_frames
        self.max_gain = max_gain

    @torch.no_grad()
    def apply(
        self,
        wav:        torch.Tensor,   # [T]
        src_energy: torch.Tensor,   # [N_frames]
        tgt_energy: torch.Tensor,   # [N_frames]  from DTW-warped ref
        hop_len:    int,
    ) -> torch.Tensor:              # [T]
        """Scale waveform frame-by-frame to match target energy."""
        # Gain: sqrt(E_tgt / E_src), clamped
        gain = torch.sqrt(
            (tgt_energy + 1e-8) / (src_energy + 1e-8)
        ).clamp(1.0 / self.max_gain, self.max_gain)

        # Median-filter smoothing to avoid abrupt gain changes
        if self.smooth > 1:
            k = self.smooth
            gain_padded = F.pad(gain.unsqueeze(0).unsqueeze(0),
                                (k//2, k//2), mode="reflect")
            # Unfold → [1, 1, N, k] → median per frame
            unfolded = gain_padded.unfold(-1, k, 1).squeeze(0).squeeze(0)
            gain = unfolded.median(dim=-1).values

        # Sample-level gain interpolation (upsample gain to waveform length)
        gain_samples = F.interpolate(
            gain.unsqueeze(0).unsqueeze(0).float(),
            size=wav.shape[0],
            mode="linear",
            align_corners=False,
        ).squeeze()

        return wav * gain_samples


# ═══════════════════════════════════════════════════════════════════════════
# Full Prosody Warper
# ═══════════════════════════════════════════════════════════════════════════

class ProsodyWarper:
    """
    End-to-end prosody extraction and DTW-based transfer.

    Usage
    ─────
        warper = ProsodyWarper(sample_rate=22050)

        # Step 1: extract reference prosody from professor's lecture
        ref = warper.extract_prosody("lecture.wav")

        # Step 2: warp the synthesised Santhali audio
        warped = warper.warp(synth_wav, ref_prosody=ref)

    Parameters
    ----------
    sample_rate  : Target sample rate (should match TTS output).
    frame_len    : Analysis frame length in samples.
    hop_len      : Analysis hop length in samples.
    f0_min/max   : F0 search range in Hz.
    dtw_radius   : Sakoe-Chiba band half-width in frames.
    """

    def __init__(
        self,
        sample_rate:  int   = 22_050,
        frame_len:    int   = 1024,
        hop_len:      int   = 256,
        f0_min:       float = 60.0,
        f0_max:       float = 500.0,
        dtw_radius:   Optional[int] = 100,
    ) -> None:
        self.sr        = sample_rate
        self.frame_len = frame_len
        self.hop_len   = hop_len

        self.f0_extractor = AutocorrF0Extractor(
            sample_rate=sample_rate, frame_len=frame_len,
            hop_len=hop_len, f0_min=f0_min, f0_max=f0_max,
        )
        self.dtw            = DTW(band_radius=dtw_radius)
        self.pitch_shifter  = PSOLAPitchShifter(sample_rate, frame_len, hop_len)
        self.energy_scaler  = EnergyScaler()

    def _load(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def extract_prosody(
        self,
        audio: str | torch.Tensor,
        label: str = "audio",
    ) -> ProsodyFeatures:
        """
        Extract F0 and energy from an audio file or waveform tensor.

        Parameters
        ----------
        audio : path string or [T] waveform tensor.
        label : description string for logging.

        Returns
        -------
        ProsodyFeatures
        """
        if isinstance(audio, str):
            wav = self._load(audio)
        else:
            wav = audio

        with torch.no_grad():
            f0, energy, voiced = self.f0_extractor(wav)

        voiced_f0 = f0[voiced]
        if voiced_f0.numel() > 0:
            print(
                f"[Prosody] {label}  frames={f0.shape[0]}  "
                f"voiced={voiced.sum().item()}  "
                f"F0: mean={voiced_f0.mean():.1f}Hz  "
                f"range=[{voiced_f0.min():.1f}, {voiced_f0.max():.1f}]Hz"
            )

        return ProsodyFeatures(
            f0=f0, energy=energy, voiced_mask=voiced,
            frame_len=self.frame_len, hop_len=self.hop_len,
            sample_rate=self.sr,
        )

    def _warp_sequence_via_path(
        self,
        src_seq:  torch.Tensor,    # [T_src]
        path:     list,            # [(ref_idx, tgt_idx)] from DTW
        tgt_len:  int,             # length of target sequence
    ) -> torch.Tensor:             # [T_tgt]
        """
        Resample src_seq onto tgt_len frames using DTW alignment path.
        For each target frame, the path gives the corresponding source frame.
        """
        warped = torch.zeros(tgt_len, device=src_seq.device)
        # Build tgt→src mapping from path (ref=src, tgt=tgt in our convention)
        tgt2src = {}
        for r_idx, t_idx in path:
            if t_idx not in tgt2src:
                tgt2src[t_idx] = []
            tgt2src[t_idx].append(r_idx)

        for t in range(tgt_len):
            src_indices = tgt2src.get(t, [min(t, src_seq.shape[0]-1)])
            vals = src_seq[[min(i, src_seq.shape[0]-1) for i in src_indices]]
            warped[t] = vals.mean()
        return warped

    @torch.no_grad()
    def warp(
        self,
        synth_wav:   torch.Tensor,      # [T]  synthesised LRL audio
        ref_prosody: ProsodyFeatures,   # reference (professor) prosody
        transfer_f0:     bool = True,
        transfer_energy: bool = True,
    ) -> torch.Tensor:                  # [T]  prosody-warped audio
        """
        Apply DTW-aligned prosody transfer to synthesised waveform.

        Steps
        ─────
        1. Extract prosody from synth_wav (src).
        2. DTW-align src and ref contours (separately for F0 and energy).
        3. Map ref F0/energy onto src time axis.
        4. Apply PSOLA pitch shifting + frame-wise amplitude scaling.

        Parameters
        ----------
        synth_wav    : Synthesised waveform [T] at self.sr.
        ref_prosody  : ProsodyFeatures from the professor's lecture.
        transfer_f0     : Enable F0 transfer (default True).
        transfer_energy : Enable energy transfer (default True).

        Returns
        -------
        Warped waveform [T].
        """
        # ── Extract synth prosody ─────────────────────────────────────────
        src_f0, src_energy, src_voiced = self.f0_extractor(synth_wav)
        T_src = src_f0.shape[0]
        T_ref = ref_prosody.f0.shape[0]

        print(
            f"[ProsodyWarp] src_frames={T_src}  ref_frames={T_ref}  "
            f"src_voiced={src_voiced.sum().item()}  "
            f"ref_voiced={ref_prosody.voiced_mask.sum().item()}"
        )

        warped = synth_wav.clone()

        # ── F0 Transfer ───────────────────────────────────────────────────
        if transfer_f0:
            # DTW on voiced-only F0 (use zeros for unvoiced, still align)
            f0_ref = ref_prosody.f0.to(synth_wav.device)

            # Normalise F0 contours to zero-mean log scale before DTW
            def _log_f0(f0: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
                out = torch.zeros_like(f0)
                if voiced.any():
                    vf = f0[voiced]
                    mu = vf.log().mean()
                    out[voiced] = f0[voiced].log() - mu
                return out

            src_log_f0 = _log_f0(src_f0, src_voiced)
            ref_log_f0 = _log_f0(f0_ref, ref_prosody.voiced_mask.to(synth_wav.device))

            # DTW alignment (ref as reference, src as target)
            _, path = self.dtw(ref_log_f0, src_log_f0)

            # Warp ref F0 onto src time axis
            warped_f0 = self._warp_sequence_via_path(f0_ref, path, T_src)

            # Combined voiced mask
            combined_voiced = src_voiced & (warped_f0 > 0)

            # Apply PSOLA pitch shifting
            warped = self.pitch_shifter.apply(
                warped, src_f0, warped_f0, combined_voiced
            )
            print(
                f"[ProsodyWarp] F0 transfer applied  "
                f"voiced_frames={combined_voiced.sum().item()}"
            )

        # ── Energy Transfer ───────────────────────────────────────────────
        if transfer_energy:
            ref_energy = ref_prosody.energy.to(synth_wav.device)

            # DTW alignment on energy (in dB)
            src_db = 20 * torch.log10(src_energy.clamp(min=1e-8))
            ref_db = 20 * torch.log10(ref_energy.clamp(min=1e-8))
            _, path_e = self.dtw(ref_db, src_db)

            warped_energy = self._warp_sequence_via_path(ref_energy, path_e, T_src)

            warped = self.energy_scaler.apply(
                warped, src_energy, warped_energy, self.hop_len
            )
            print("[ProsodyWarp] Energy transfer applied")

        return warped.clamp(-1.0, 1.0)

    def warp_file(
        self,
        synth_path:   str,
        ref_path:     str,
        output_path:  str,
        transfer_f0:     bool = True,
        transfer_energy: bool = True,
    ) -> None:
        """File-level convenience: load, warp, save."""
        synth_wav = self._load(synth_path)
        ref_pros  = self.extract_prosody(ref_path, label="professor")

        warped = self.warp(synth_wav, ref_pros, transfer_f0, transfer_energy)
        torchaudio.save(output_path, warped.unsqueeze(0), self.sr)
        print(f"[ProsodyWarp] Warped audio saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Prosody visualisation report (no matplotlib dependency — text-based)
# ─────────────────────────────────────────────────────────────────────────────

def prosody_text_report(
    ref: ProsodyFeatures,
    synth: ProsodyFeatures,
    label: str = "Prosody Report",
) -> str:
    """Print a text-based summary of prosody features."""
    def _stats(f0: torch.Tensor, voiced: torch.Tensor) -> str:
        if not voiced.any():
            return "no voiced frames"
        vf = f0[voiced]
        return (
            f"mean={vf.mean():.1f}Hz  "
            f"std={vf.std():.1f}Hz  "
            f"min={vf.min():.1f}Hz  "
            f"max={vf.max():.1f}Hz  "
            f"voiced={voiced.float().mean():.1%}"
        )

    lines = [
        f"── {label} ──",
        f"  Reference  : {_stats(ref.f0,   ref.voiced_mask)}",
        f"  Synthesised: {_stats(synth.f0, synth.voiced_mask)}",
        f"  Ref  energy: mean={ref.energy.mean():.4f}  std={ref.energy.std():.4f}",
        f"  Synth energy:mean={synth.energy.mean():.4f}  std={synth.energy.std():.4f}",
    ]
    report = "\n".join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Prosody Warping (Task 3.2)")
    p.add_argument("--synth",   required=True, help="Synthesised TTS WAV")
    p.add_argument("--ref",     required=True, help="Reference lecture WAV (professor)")
    p.add_argument("--output",  required=True, help="Output warped WAV")
    p.add_argument("--sr",      type=int,   default=22050)
    p.add_argument("--no_f0",   action="store_true")
    p.add_argument("--no_energy", action="store_true")
    args = p.parse_args()

    warper = ProsodyWarper(sample_rate=args.sr)
    warper.warp_file(
        args.synth, args.ref, args.output,
        transfer_f0=not args.no_f0,
        transfer_energy=not args.no_energy,
    )
