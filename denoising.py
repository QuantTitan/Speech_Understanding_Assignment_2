"""
denoising.py  — Task 1.3
========================
Preprocessing pipeline for classroom audio denoising and normalisation.

Method: **Spectral Subtraction** (Boll 1979) with the following extensions
  • Over-subtraction factor α   – amplify noise estimate to avoid residual noise
  • Spectral floor       β      – prevent musical noise artefacts
  • VAD-based noise update      – estimate noise PSD from silence frames only
  • Power normalisation         – standardise RMS energy for downstream ASR

Dependencies: torchaudio (PyTorch audio), torch, numpy (for minor helpers)
No external denoising libraries are required.

Usage
-----
    from denoising import SpectralSubtractionDenoiser, denoise_file

    # Quick file → file
    denoise_file("lecture_raw.wav", "lecture_clean.wav")

    # Programmatic (for pipeline integration)
    denoiser = SpectralSubtractionDenoiser(sr=16000)
    waveform, sr = torchaudio.load("lecture_raw.wav")
    clean      = denoiser(waveform)          # [1, T] float32 tensor
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


# ---------------------------------------------------------------------------
# Voice Activity Detector (energy-based, frame-level)
# ---------------------------------------------------------------------------

class EnergyVAD(torch.nn.Module):
    """
    Simple energy-based VAD.

    Labels a frame as *speech* if its log-energy exceeds a threshold
    derived from the lowest-energy percentile of the utterance.

    Parameters
    ----------
    frame_len   : STFT window length in samples.
    hop_len     : STFT hop length in samples.
    energy_floor: Minimum log-energy to avoid log(0).
    percentile  : Noise floor estimated from this bottom percentile.
    snr_threshold: Frame is speech if log-energy > noise_floor + threshold (dB).
    """

    def __init__(
        self,
        frame_len: int = 512,
        hop_len: int = 128,
        energy_floor: float = 1e-8,
        percentile: float = 10.0,
        snr_threshold: float = 6.0,
    ) -> None:
        super().__init__()
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.energy_floor = energy_floor
        self.percentile = percentile
        self.snr_threshold = snr_threshold

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : [C, T] or [T]  mono (or first channel is used).

        Returns
        -------
        vad_mask : BoolTensor [F]  True = speech frame, where F = num STFT frames.
        """
        x = waveform[0] if waveform.dim() == 2 else waveform   # → [T]

        # Frame-level energy via STFT magnitude
        stft = torch.stft(
            x,
            n_fft=self.frame_len,
            hop_length=self.hop_len,
            win_length=self.frame_len,
            window=torch.hann_window(self.frame_len, device=x.device),
            return_complex=True,
        )                                              # [F_bins, T_frames]
        energy = stft.abs().pow(2).mean(dim=0)        # [T_frames]
        log_energy = torch.log(energy.clamp(min=self.energy_floor))

        # Noise floor from low-energy percentile
        noise_floor = torch.quantile(log_energy, self.percentile / 100.0)
        speech_mask = log_energy > (noise_floor + self.snr_threshold)
        return speech_mask


# ---------------------------------------------------------------------------
# Core Spectral Subtraction Denoiser
# ---------------------------------------------------------------------------

class SpectralSubtractionDenoiser(torch.nn.Module):
    """
    Spectral Subtraction denoiser (Boll 1979 + Ephraim-Malah floor).

    Algorithm
    ---------
    1.  Compute STFT:  X(k, t)  =  STFT{ x(t) }
    2.  Estimate noise PSD   D̂(k)  from VAD-silent frames (exponential average).
    3.  Enhanced magnitude:
            |Ŝ(k,t)|² = max( |X(k,t)|² − α·D̂(k) ,  β·|X(k,t)|² )
    4.  Reconstruct: Ŝ(k,t) = |Ŝ(k,t)| · exp(j·∠X(k,t))
    5.  ISTFT → ŝ(t)

    Parameters
    ----------
    sample_rate     : Target sample rate (default 16 000 Hz).
    n_fft           : FFT size  (default 512  → 32 ms @ 16 kHz).
    hop_length      : Hop size  (default 128  →  8 ms @ 16 kHz).
    over_subtraction: α  ∈ [1, 3]  – higher = more aggressive denoising.
    spectral_floor  : β  ∈ (0, 1]  – prevents over-subtraction artefacts.
    noise_frames    : Number of initial frames assumed silence for cold start.
    smoothing_coef  : Exponential smoothing α for noise PSD update (0–1).
    target_rms      : RMS level for output normalisation (None = skip).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        hop_length: int = 128,
        over_subtraction: float = 2.0,
        spectral_floor: float = 0.002,
        noise_frames: int = 20,
        smoothing_coef: float = 0.98,
        target_rms: Optional[float] = 0.05,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = over_subtraction        # over-subtraction
        self.beta = spectral_floor           # spectral floor
        self.noise_frames = noise_frames
        self.smoothing = smoothing_coef
        self.target_rms = target_rms

        self.vad = EnergyVAD(frame_len=n_fft, hop_len=hop_length)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.sample_rate:
            return waveform
        resamp = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
        return resamp(waveform)

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        """Average channels → mono [1, T]."""
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        return waveform.mean(dim=0, keepdim=True)

    def _estimate_noise_psd(
        self,
        magnitude: torch.Tensor,          # [F_bins, T_frames]
        vad_mask: torch.Tensor,           # [T_frames] bool
    ) -> torch.Tensor:                    # [F_bins]
        """
        Estimate noise PSD using VAD-identified silence frames.
        Falls back to the first `noise_frames` frames if VAD yields too few.
        """
        silent_frames = (~vad_mask).nonzero(as_tuple=True)[0]

        if silent_frames.numel() >= self.noise_frames:
            sel = magnitude[:, silent_frames]              # [F, N_silent]
        else:
            # Cold-start: assume beginning is noise
            n = min(self.noise_frames, magnitude.shape[1])
            sel = magnitude[:, :n]

        noise_psd = sel.pow(2).mean(dim=1)                # [F_bins]
        return noise_psd

    def _spectral_subtraction(
        self,
        magnitude: torch.Tensor,          # [F_bins, T_frames]
        noise_psd: torch.Tensor,          # [F_bins]
    ) -> torch.Tensor:                    # [F_bins, T_frames]
        """Apply over-subtraction with spectral floor."""
        noise_psd = noise_psd.unsqueeze(1)                 # [F, 1]
        power = magnitude.pow(2)                           # [F, T]

        # Subtracted power
        subtracted = power - self.alpha * noise_psd        # [F, T]

        # Spectral floor: max(subtracted, β·|X|²)
        floored = torch.maximum(subtracted, self.beta * power)

        # Enhanced magnitude
        enhanced_mag = floored.clamp(min=0.0).sqrt()      # [F, T]
        return enhanced_mag

    @staticmethod
    def _normalise_rms(waveform: torch.Tensor, target_rms: float) -> torch.Tensor:
        rms = waveform.pow(2).mean().sqrt().clamp(min=1e-8)
        return waveform * (target_rms / rms)

    # ── Forward pass ────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Denoise a waveform tensor.

        Parameters
        ----------
        waveform    : [C, T] or [T] float32 tensor (any sample rate if
                      `sample_rate` is provided, else assumed = self.sample_rate).
        sample_rate : Original sample rate.  If None, uses self.sample_rate.

        Returns
        -------
        clean : [1, T] float32 denoised mono waveform at self.sample_rate.
        """
        sr = sample_rate or self.sample_rate
        wav = self._to_mono(waveform)     # [1, T]
        wav = self._resample(wav, sr)     # [1, T] @ target sr
        x = wav[0]                        # [T]

        window = torch.hann_window(self.n_fft, device=x.device)

        # ── STFT ──────────────────────────────────────────────────────────
        stft_complex = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )                                             # [F, T]

        magnitude = stft_complex.abs()               # [F, T]
        phase = stft_complex.angle()                 # [F, T]

        # ── Voice Activity Detection ───────────────────────────────────────
        vad_mask = self.vad(wav)                      # [T]  True = speech

        # ── Noise PSD Estimation ──────────────────────────────────────────
        noise_psd = self._estimate_noise_psd(magnitude, vad_mask)

        # ── Dynamic noise tracking (exponential smoothing on silence) ──────
        tracked_noise = noise_psd.clone()
        for t in range(magnitude.shape[1]):
            if t < vad_mask.numel() and not vad_mask[t]:
                frame_power = magnitude[:, t].pow(2)
                tracked_noise = (
                    self.smoothing * tracked_noise + (1 - self.smoothing) * frame_power
                )

        # ── Spectral Subtraction ──────────────────────────────────────────
        enhanced_mag = self._spectral_subtraction(magnitude, tracked_noise)

        # ── Reconstruct with original phase ──────────────────────────────
        enhanced_complex = torch.polar(enhanced_mag, phase)

        # ── ISTFT ────────────────────────────────────────────────────────
        enhanced = torch.istft(
            enhanced_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            length=x.shape[0],
        )                                            # [T]

        enhanced = enhanced.unsqueeze(0)             # [1, T]

        # ── RMS Normalisation ─────────────────────────────────────────────
        if self.target_rms is not None:
            enhanced = self._normalise_rms(enhanced, self.target_rms)

        return enhanced.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Reverb suppression via spectral Wiener post-filter
# ---------------------------------------------------------------------------

class WienerPostFilter(torch.nn.Module):
    """
    Single-channel Wiener post-filter for mild dereverberation.

    Applies  H(k,t) = SNR(k,t) / (1 + SNR(k,t))  per frequency bin,
    where SNR is estimated from a running maximum of the power spectrum
    (late reverberation assumption: reverb PSD ~ fraction of direct PSD).

    Parameters
    ----------
    n_fft      : FFT window length.
    hop_length : STFT hop.
    reverb_est : Fraction of running max power attributed to reverberation.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        reverb_est: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.reverb_est = reverb_est

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """[1, T] → [1, T]  Wiener filtered."""
        x = waveform[0]
        window = torch.hann_window(self.n_fft, device=x.device)

        stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, return_complex=True,
        )
        mag = stft.abs()                          # [F, T]
        phase = stft.angle()

        # Running max power as proxy for direct speech PSD
        running_max = torch.zeros_like(mag[:, 0])
        out_mag = torch.zeros_like(mag)
        decay = 0.999

        for t in range(mag.shape[1]):
            running_max = torch.maximum(decay * running_max, mag[:, t])
            reverb_psd = self.reverb_est * running_max.pow(2)
            signal_psd = mag[:, t].pow(2)
            snr = signal_psd / (reverb_psd.clamp(min=1e-8))
            gain = snr / (1.0 + snr)
            out_mag[:, t] = gain * mag[:, t]

        enhanced = torch.polar(out_mag, phase)
        y = torch.istft(
            enhanced, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, length=x.shape[0],
        )
        return y.unsqueeze(0)


# ---------------------------------------------------------------------------
# Full preprocessing chain
# ---------------------------------------------------------------------------

class AudioPreprocessor(torch.nn.Module):
    """
    End-to-end audio preprocessing chain:
        1. Mono conversion + resampling  (16 kHz)
        2. Spectral Subtraction denoising
        3. Wiener post-filter (dereverberation)
        4. RMS normalisation

    Parameters
    ----------
    sample_rate      : Target sample rate.
    over_subtraction : Spectral subtraction α (higher = more aggressive).
    spectral_floor   : Spectral subtraction β.
    denoise_only     : If True, skip the Wiener dereverberation step.
    target_rms       : Output RMS level (None = no normalisation).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        over_subtraction: float = 2.0,
        spectral_floor: float = 0.002,
        denoise_only: bool = False,
        target_rms: float = 0.05,
    ) -> None:
        super().__init__()
        self.denoiser = SpectralSubtractionDenoiser(
            sample_rate=sample_rate,
            over_subtraction=over_subtraction,
            spectral_floor=spectral_floor,
            target_rms=None,   # normalise at the end
        )
        self.dereverb = None if denoise_only else WienerPostFilter()
        self.target_rms = target_rms

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16_000,
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns
        -------
        (clean_waveform [1, T], target_sample_rate)
        """
        clean = self.denoiser(waveform, sample_rate=sample_rate)   # [1, T]

        if self.dereverb is not None:
            clean = self.dereverb(clean)

        # Final RMS normalisation
        if self.target_rms is not None:
            rms = clean.pow(2).mean().sqrt().clamp(min=1e-8)
            clean = clean * (self.target_rms / rms)

        clean = clean.clamp(-1.0, 1.0)
        return clean, self.denoiser.sample_rate


# ---------------------------------------------------------------------------
# File-level convenience function
# ---------------------------------------------------------------------------

def denoise_file(
    input_path: str,
    output_path: str,
    over_subtraction: float = 2.0,
    spectral_floor: float = 0.002,
    target_rms: float = 0.05,
    denoise_only: bool = False,
) -> None:
    """
    Load a WAV file, denoise it, and write the cleaned file.

    Parameters
    ----------
    input_path      : Path to the raw classroom audio WAV.
    output_path     : Destination path for the cleaned audio.
    over_subtraction: α – aggressiveness of spectral subtraction.
    spectral_floor  : β – prevents musical noise floor.
    target_rms      : RMS energy of the output (set None to skip).
    denoise_only    : Skip Wiener dereverberation if True.
    """
    waveform, sr = torchaudio.load(input_path)
    print(f"[Denoiser] Loaded  {input_path}  shape={tuple(waveform.shape)}  sr={sr}")

    preprocessor = AudioPreprocessor(
        sample_rate=16_000,
        over_subtraction=over_subtraction,
        spectral_floor=spectral_floor,
        denoise_only=denoise_only,
        target_rms=target_rms,
    )

    clean, out_sr = preprocessor(waveform, sample_rate=sr)

    # Diagnostics
    input_rms  = waveform.pow(2).mean().sqrt().item()
    output_rms = clean.pow(2).mean().sqrt().item()
    print(
        f"[Denoiser] RMS  in={input_rms:.4f}  out={output_rms:.4f}  "
        f"duration={clean.shape[-1]/out_sr:.1f}s"
    )

    torchaudio.save(output_path, clean, out_sr)
    print(f"[Denoiser] Saved  → {output_path}")


# ---------------------------------------------------------------------------
# Smoke test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Spectral Subtraction Denoiser")
    parser.add_argument("input",  help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--alpha", type=float, default=2.0, help="Over-subtraction α")
    parser.add_argument("--beta",  type=float, default=0.002, help="Spectral floor β")
    parser.add_argument("--rms",   type=float, default=0.05,  help="Target RMS")
    parser.add_argument("--denoise-only", action="store_true")
    args = parser.parse_args()

    denoise_file(
        args.input, args.output,
        over_subtraction=args.alpha,
        spectral_floor=args.beta,
        target_rms=args.rms,
        denoise_only=args.denoise_only,
    )
