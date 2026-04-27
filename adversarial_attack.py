"""
adversarial_attack.py  —  Task 4.2
=====================================
Adversarial Noise Injection for LID Robustness Analysis.

Goal
─────
Find the minimum FGSM perturbation ε that causes the LID system (Task 1.1)
to misclassify a **Hindi** 5-second segment as **English**, while keeping the
audio inaudible to humans:
    SNR  > 40 dB  (perturbation below perceptual threshold)
    ε_min : smallest such ε where LID flips Hindi → English.

Methods implemented
────────────────────
  1. FGSM  (Fast Gradient Sign Method, Goodfellow et al. 2014)
       δ = ε · sign(∇_x L(f(x), y_target))
       where y_target = "Hindi" and L = cross-entropy on the LID output.

  2. PGD   (Projected Gradient Descent, Madry et al. 2018)
       Multi-step FGSM with ℓ∞ projection — stronger attack, better
       reveals model vulnerabilities.

  3. SNR-constrained ε sweep
       Sweep ε ∈ [1e-5, 5e-2] logarithmically; report flip rate and SNR
       at each level; flag which are inaudible (SNR > 40 dB).

Usage
─────
    from adversarial_attack import FGSMAttacker, run_epsilon_sweep

    attacker = FGSMAttacker(lid_model, mel_extractor, device="cuda")
    result   = attacker.attack_segment("hindi_segment.wav", epsilon=0.01)
    print(result["flipped"], result["snr_db"])

    sweep = run_epsilon_sweep("hindi_segment.wav", lid_model, mel_extractor)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from evaluation_metrics import compute_snr, adversarial_epsilon_report


# ═══════════════════════════════════════════════════════════════════════════
# Label constants (must match LID model's LABEL_NAMES)
# ═══════════════════════════════════════════════════════════════════════════

LABEL_ENGLISH = 0
LABEL_HINDI   = 1
LABEL_MIXED   = 2
LABEL_SILENCE = 3


# ═══════════════════════════════════════════════════════════════════════════
# 1.  AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def load_segment(
    wav_path: str,
    start_s:  float = 0.0,
    duration_s: float = 5.0,
    target_sr: int = 16_000,
) -> Tuple[torch.Tensor, int]:
    """
    Load a fixed-duration segment from a WAV file.

    Returns (waveform [1, T], sample_rate).
    """
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(0, keepdim=True)      # mono [1, T]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    start = int(start_s * target_sr)
    end   = start + int(duration_s * target_sr)
    if end > wav.shape[-1]:
        pad = end - wav.shape[-1]
        wav = F.pad(wav, (0, pad))
    return wav[:, start:end], target_sr


def add_perturbation_clipped(
    original: torch.Tensor,
    delta: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Apply ℓ∞ perturbation and clip to valid audio range."""
    perturbed = original + delta.clamp(-epsilon, epsilon)
    return perturbed.clamp(-1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FGSM ATTACKER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttackResult:
    epsilon:        float
    snr_db:         float
    original_label: int
    attacked_label: int
    flipped:        bool             # True if Hindi → English
    flip_rate:      float            # fraction of frames flipped
    original_probs: torch.Tensor     # [n_classes]
    attacked_probs: torch.Tensor     # [n_classes]
    delta_l2:       float            # ‖δ‖₂  perturbation magnitude
    original_wav:   Optional[torch.Tensor] = None
    perturbed_wav:  Optional[torch.Tensor] = None


class FGSMAttacker:
    """
    FGSM-based adversarial attacker for the Multi-Head LID model.

    The attack is performed in the **waveform domain** by backpropagating
    through the mel-feature extractor and the LID model jointly, using
    a differentiable STFT.

    Targeted attack: we push the model towards predicting English (label 0)
    when the true label is Hindi (label 1).

    Parameters
    ----------
    lid_model       : Trained MultiHeadLID instance.
    feat_extractor  : MelFeatureExtractor (must be differentiable).
    device          : Compute device.
    target_label    : Adversarial target (default: English = 0).
    n_fft           : STFT window for mel computation (must match extractor).
    hop_length      : STFT hop.
    """

    def __init__(
        self,
        lid_model,
        feat_extractor,
        device:       str = "cpu",
        target_label: int = LABEL_ENGLISH,
        n_fft:        int = 512,
        hop_length:   int = 128,
    ) -> None:
        self.model        = lid_model
        self.feat_ext     = feat_extractor
        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_label = target_label
        self.n_fft        = n_fft
        self.hop_length   = hop_length

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _wav_to_mel_diff(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Differentiable waveform → log-mel spectrogram.

        Uses torchaudio's MelSpectrogram which is autograd-compatible.
        wav : [1, T]  →  mel [1, T', n_mels]
        """
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            n_mels=80,
            f_min=0.0,
            f_max=8_000.0,
            power=2.0,
        ).to(self.device)

        wav_mono = wav.squeeze(0)                    # [T]
        mel      = mel_fn(wav_mono)                  # [n_mels, T']
        log_mel  = (torchaudio.transforms.AmplitudeToDB(top_db=80)(mel) + 40) / 40
        return log_mel.T.unsqueeze(0)                # [1, T', n_mels]

    def _get_label_and_prob(
        self, mel: torch.Tensor
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Run LID on mel features.

        Returns (dominant_label_int, mean_target_prob, mean_class_probs [4]).
        """
        with torch.no_grad():
            out = self.model(mel)
        probs = F.softmax(out.logits_4cls[0], dim=-1)  # [T', 4]
        mean_probs  = probs.mean(0)                     # [4]
        dom_label   = mean_probs.argmax().item()
        target_prob = mean_probs[self.target_label].item()
        return dom_label, target_prob, mean_probs.detach().cpu()

    def fgsm_attack(
        self,
        wav: torch.Tensor,      # [1, T]  on device
        epsilon: float,
        n_steps: int = 1,       # >1 → PGD-style multi-step
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute FGSM (or PGD) perturbation.

        Parameters
        ----------
        wav      : Input waveform [1, T] with grad enabled.
        epsilon  : ℓ∞ perturbation budget.
        n_steps  : 1 = FGSM; >1 = PGD.
        step_size: PGD step size (default ε/n_steps * 2).

        Returns
        -------
        perturbed_wav : [1, T]  (clipped to [-1, 1]).
        """
        alpha = step_size or (2 * epsilon / max(n_steps, 1))
        wav_adv = wav.clone().detach().requires_grad_(False)

        for _ in range(n_steps):
            wav_in = wav_adv.clone().requires_grad_(True)
            mel = self._wav_to_mel_diff(wav_in)          # [1, T', 80]

            # Forward through LID
            out = self.model(mel)                        # LIDOutput
            logits = out.logits_4cls                     # [1, T', 4]

            # Targeted loss: minimise probability of TRUE label (Hindi),
            # maximise probability of TARGET label (English).
            # Loss = CE(logit, target_label) — we gradient-DESCEND on this.
            T_frames = logits.shape[1]
            target_t = torch.full(
                (1, T_frames), self.target_label,
                dtype=torch.long, device=self.device,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, 4),
                target_t.reshape(-1),
            )

            loss.backward()
            grad_sign = wav_in.grad.sign()
            # Gradient descent: subtract (since we want to INCREASE confidence
            # of target_label = minimise CE of target_label)
            wav_adv = (wav_adv - alpha * grad_sign).detach()

            # ℓ∞ projection onto ε-ball around original
            delta = (wav_adv - wav).clamp(-epsilon, epsilon)
            wav_adv = (wav + delta).clamp(-1.0, 1.0).detach()

        return wav_adv

    def attack_segment(
        self,
        wav_path:   str,
        epsilon:    float,
        start_s:    float = 0.0,
        duration_s: float = 5.0,
        n_steps:    int   = 1,
        save_path:  Optional[str] = None,
    ) -> AttackResult:
        """
        Attack a 5-second Hindi segment and report results.

        Parameters
        ----------
        wav_path   : Path to audio containing a Hindi segment.
        epsilon    : ℓ∞ perturbation budget.
        start_s    : Start of the 5-second segment in the file.
        duration_s : Segment duration (default 5.0 s).
        n_steps    : FGSM steps (1 = FGSM, >1 = PGD).
        save_path  : If given, save perturbed audio.

        Returns
        -------
        AttackResult
        """
        self.model.eval()

        # ── Load original segment ──────────────────────────────────────────
        orig_wav, sr = load_segment(wav_path, start_s, duration_s)
        orig_wav = orig_wav.to(self.device)

        # ── Original LID prediction ────────────────────────────────────────
        mel_orig = self._wav_to_mel_diff(orig_wav.detach())
        orig_label, _, orig_probs = self._get_label_and_prob(mel_orig.detach())

        # ── FGSM / PGD attack ──────────────────────────────────────────────
        adv_wav = self.fgsm_attack(orig_wav, epsilon, n_steps=n_steps)

        # ── Attacked LID prediction ────────────────────────────────────────
        mel_adv = self._wav_to_mel_diff(adv_wav.detach())
        adv_label, _, adv_probs = self._get_label_and_prob(mel_adv.detach())

        # ── Metrics ────────────────────────────────────────────────────────
        snr_db  = compute_snr(orig_wav.cpu(), adv_wav.cpu())
        delta   = (adv_wav - orig_wav).cpu()
        delta_l2 = delta.norm().item()

        # Frame-level flip rate
        with torch.no_grad():
            out_orig = self.model(mel_orig.detach())
            out_adv  = self.model(mel_adv.detach())
        orig_frame_labels = out_orig.logits_4cls[0].argmax(-1)  # [T']
        adv_frame_labels  = out_adv.logits_4cls[0].argmax(-1)

        # Flip: was Hindi, now English
        hindi_orig = (orig_frame_labels == LABEL_HINDI)
        now_english = (adv_frame_labels == LABEL_ENGLISH)
        flip_rate = (hindi_orig & now_english).float().mean().item() \
                    if hindi_orig.any() else 0.0

        flipped = (orig_label == LABEL_HINDI and adv_label == LABEL_ENGLISH)

        result = AttackResult(
            epsilon        = epsilon,
            snr_db         = snr_db,
            original_label = orig_label,
            attacked_label = adv_label,
            flipped        = flipped,
            flip_rate      = flip_rate,
            original_probs = orig_probs,
            attacked_probs = adv_probs,
            delta_l2       = delta_l2,
            original_wav   = orig_wav.cpu(),
            perturbed_wav  = adv_wav.cpu(),
        )

        # ── Save perturbed audio ───────────────────────────────────────────
        if save_path:
            torchaudio.save(save_path, adv_wav.cpu(), sr)
            print(f"  Saved perturbed audio → {save_path}")

        return result


# ═══════════════════════════════════════════════════════════════════════════
# 3.  EPSILON SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def run_epsilon_sweep(
    wav_path:    str,
    lid_model,
    feat_extractor,
    start_s:     float = 0.0,
    duration_s:  float = 5.0,
    eps_min:     float = 1e-5,
    eps_max:     float = 5e-2,
    n_eps:       int   = 20,
    n_pgd_steps: int   = 1,
    device:      str   = "cpu",
    output_dir:  Optional[str] = None,
) -> Dict:
    """
    Sweep ε logarithmically and report SNR + LID flip rate at each level.

    Parameters
    ----------
    wav_path      : Hindi audio segment to attack.
    lid_model     : Trained MultiHeadLID model.
    feat_extractor: MelFeatureExtractor.
    start_s       : Segment start in seconds.
    duration_s    : Segment duration (5 s default).
    eps_min/max   : Sweep range.
    n_eps         : Number of epsilon values to test.
    n_pgd_steps   : Steps per attack (1 = FGSM, 10 = PGD).
    device        : Compute device.
    output_dir    : If given, save perturbed WAVs per epsilon.

    Returns
    -------
    dict from adversarial_epsilon_report()
    """
    attacker = FGSMAttacker(lid_model, feat_extractor, device=device)
    epsilons = torch.logspace(
        math.log10(eps_min), math.log10(eps_max), n_eps
    ).tolist()

    results = []
    print(f"\n── Adversarial ε Sweep  [{eps_min:.1e} → {eps_max:.1e}]  n={n_eps} ──")
    print(f"  {'ε':>10s}  {'SNR (dB)':>10s}  {'Flip?':>6s}  {'Flip%':>6s}  "
          f"{'InAudible?':>11s}")
    print(f"  {'-'*55}")

    for eps in epsilons:
        save = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save = f"{output_dir}/adv_eps{eps:.2e}.wav"

        res = attacker.attack_segment(
            wav_path, eps, start_s=start_s, duration_s=duration_s,
            n_steps=n_pgd_steps, save_path=save,
        )
        inaudible = res.snr_db > 40.0
        row = {
            "epsilon":    eps,
            "snr_db":     res.snr_db,
            "flipped":    res.flipped,
            "flip_rate":  res.flip_rate,
            "inaudible":  inaudible,
            "delta_l2":   res.delta_l2,
            "orig_label": res.original_label,
            "adv_label":  res.attacked_label,
        }
        results.append(row)

        flip_char = "YES" if res.flipped else " no"
        inaud_str = "✓ inaudible" if inaudible else "× audible  "
        print(
            f"  {eps:10.2e}  {res.snr_db:10.1f}  {flip_char:>6s}  "
            f"{res.flip_rate:6.1%}  {inaud_str}"
        )

    report = adversarial_epsilon_report(results)
    _print_sweep_summary(report)
    return report


def _print_sweep_summary(report: Dict) -> None:
    print("\n── Sweep Summary ──")
    min_eps  = report.get("min_flip_epsilon")
    snr_flip = report.get("snr_at_min_flip_db")

    if min_eps is not None:
        print(f"  Min ε to flip LID       : {min_eps:.4e}")
        print(f"  SNR at min flip ε       : {snr_flip:.1f} dB")
    else:
        print("  LID was NOT flipped at any tested ε")

    inaud_flip = report.get("first_inaudible_flip")
    if inaud_flip:
        print(f"  First inaudible flip ε  : {inaud_flip['epsilon']:.4e}  "
              f"SNR={inaud_flip['snr_db']:.1f} dB  flip={inaud_flip['flip_rate']:.1%}")
        print(f"  SNR constraint (>40dB)  : ✓ PASS")
    else:
        print(f"  SNR constraint (>40dB)  : ✗ No inaudible flip found")
        print(f"    (Try reducing eps_min or increasing n_pgd_steps)")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PGD ATTACKER  (stronger, multi-step variant)
# ═══════════════════════════════════════════════════════════════════════════

class PGDAttacker(FGSMAttacker):
    """
    Projected Gradient Descent attacker (Madry et al., 2018).

    Inherits FGSMAttacker; overrides default n_steps to 20 and provides
    random restart initialisation for a stronger attack.

    Parameters
    ----------
    n_restarts : Number of random restart initialisations.
    n_steps    : PGD iteration count per restart.
    """

    def __init__(self, *args, n_restarts: int = 3, n_steps: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_restarts = n_restarts
        self.n_steps_pgd = n_steps

    def attack_segment(
        self,
        wav_path:   str,
        epsilon:    float,
        start_s:    float = 0.0,
        duration_s: float = 5.0,
        n_steps:    Optional[int] = None,
        save_path:  Optional[str] = None,
    ) -> AttackResult:
        """PGD attack with multiple random restarts; returns best (highest flip rate)."""
        n_steps = n_steps or self.n_steps_pgd
        best: Optional[AttackResult] = None

        for restart in range(self.n_restarts):
            # Random start inside ε-ball
            orig_wav, sr = load_segment(wav_path, start_s, duration_s)
            orig_wav = orig_wav.to(self.device)

            init_delta = (2 * torch.rand_like(orig_wav) - 1) * epsilon
            wav_init   = (orig_wav + init_delta).clamp(-1.0, 1.0)

            # Write temp WAV for FGSMAttacker base method
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            torchaudio.save(tmp_path, wav_init.cpu(), sr)

            result = super().attack_segment(
                tmp_path, epsilon, start_s=0.0,
                duration_s=duration_s, n_steps=n_steps,
            )
            Path(tmp_path).unlink(missing_ok=True)

            if best is None or result.flip_rate > best.flip_rate:
                best = result

        if save_path and best is not None and best.perturbed_wav is not None:
            torchaudio.save(save_path, best.perturbed_wav, sr)
        return best


# ═══════════════════════════════════════════════════════════════════════════
# 5.  SYNTHETIC DEMO  (no real audio needed)
# ═══════════════════════════════════════════════════════════════════════════

def synthetic_attack_demo(
    lid_model,
    feat_extractor,
    device: str = "cpu",
    duration_s: float = 5.0,
) -> Dict:
    """
    Run FGSM sweep on a synthetic random waveform (Gaussian noise).
    Used for unit-testing when no real audio is available.
    """
    import tempfile
    sr = 16_000
    wav = torch.randn(1, int(sr * duration_s)) * 0.1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    torchaudio.save(tmp, wav, sr)

    result = run_epsilon_sweep(
        tmp, lid_model, feat_extractor,
        eps_min=1e-4, eps_max=1e-1, n_eps=8, device=device,
    )
    Path(tmp).unlink(missing_ok=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json, sys

    p = argparse.ArgumentParser(description="FGSM Adversarial Attack on LID (Task 4.2)")
    p.add_argument("wav",             help="Hindi WAV segment to attack")
    p.add_argument("--lid_ckpt",      default=None,   help="LID checkpoint .pt")
    p.add_argument("--start_s",       type=float, default=0.0)
    p.add_argument("--duration",      type=float, default=5.0)
    p.add_argument("--eps_min",       type=float, default=1e-5)
    p.add_argument("--eps_max",       type=float, default=5e-2)
    p.add_argument("--n_eps",         type=int,   default=20)
    p.add_argument("--n_steps",       type=int,   default=1,
                   help="1=FGSM, >1=PGD")
    p.add_argument("--device",        default="cpu")
    p.add_argument("--output_dir",    default=None)
    args = p.parse_args()

    # Lazy import to avoid circular deps at module level
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from lid_model import LIDConfig, MultiHeadLID, MelFeatureExtractor

    cfg   = LIDConfig()
    model = MultiHeadLID(cfg)
    feat  = MelFeatureExtractor(cfg)

    if args.lid_ckpt and Path(args.lid_ckpt).exists():
        ckpt = torch.load(args.lid_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Attack] Loaded LID checkpoint: {args.lid_ckpt}")
    else:
        print("[Attack] WARNING: Using randomly initialised LID model.")

    model.eval()
    report = run_epsilon_sweep(
        args.wav, model, feat,
        start_s=args.start_s, duration_s=args.duration,
        eps_min=args.eps_min, eps_max=args.eps_max, n_eps=args.n_eps,
        n_pgd_steps=args.n_steps, device=args.device,
        output_dir=args.output_dir,
    )

    # Print clean JSON (omit wav tensors)
    clean = {k: v for k, v in report.items() if k != "all_results"}
    print("\n" + json.dumps(clean, indent=2))
