"""
pipeline_part3.py  —  Part III: Zero-Shot Cross-Lingual Voice Cloning
======================================================================
Full pipeline:

  Stage 6  (Task 3.1) → Extract d-vector speaker embedding from 60-second
                         voice recording (GE2E LSTM encoder).
  Stage 7  (Task 3.2) → Extract F0 + energy from professor's lecture;
                         apply DTW prosody warping onto synthesised audio.
  Stage 8  (Task 3.3) → VITS-based zero-shot synthesis of the Santhali
                         lecture at ≥ 22050 Hz.

Reads Part II outputs (santhali_transcript, ipa_transcript) and produces:
  my_dvector.pt               speaker embedding
  lecture_santhali_raw.wav    raw synthesised lecture (22050 Hz)
  lecture_santhali_final.wav  prosody-warped final output
  part3_report.json           metrics + config

Usage
─────
    python pipeline_part3.py \
        --voice_recording  my_voice_60s.wav \
        --ref_lecture      lecture_raw.wav \
        --santhali_json    results/ipa_transcript.json \
        --output           results/ \
        --speaker_encoder  checkpoints/speaker_enc.pt \
        --vits_checkpoint  checkpoints/vits.pt \
        --device           cuda

    # Demo mode (random weights — pipeline validation only)
    python pipeline_part3.py \
        --voice_recording  my_voice_60s.wav \
        --ref_lecture      lecture_raw.wav \
        --santhali_json    results/ipa_transcript.json \
        --output           results/ \
        --device           cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

from speaker_encoder import VoiceRegistrar, SpeakerEncoder
from prosody_warping import ProsodyWarper, prosody_text_report
from tts_synthesizer import VITSSynthesizer


# ---------------------------------------------------------------------------
# Stage 6: Speaker Embedding (Task 3.1)
# ---------------------------------------------------------------------------

def run_speaker_embedding(
    voice_path:     str,
    output_dir:     Path,
    encoder_path:   Optional[str],
    device:         str,
) -> torch.Tensor:
    """
    Extract the d-vector from a 60-second voice recording.

    Returns
    -------
    dvector : [256]  L2-normalised speaker embedding.
    """
    print("\n" + "─" * 60)
    print("STAGE 6  │  Task 3.1 — Speaker Embedding Extraction")
    print("─" * 60)
    print(f"  Recording : {voice_path}")
    if encoder_path:
        print(f"  Encoder   : {encoder_path}")
    else:
        print("  Encoder   : random init (demo mode)")

    t0 = time.time()
    registrar = VoiceRegistrar(encoder_path=encoder_path, device=device)
    dvec_path = str(output_dir / "my_dvector.pt")
    dvector   = registrar.register(voice_path, save_path=dvec_path)
    elapsed   = time.time() - t0

    print(f"\n  d-vector extracted in {elapsed:.2f}s")
    print(f"  Shape  : {tuple(dvector.shape)}")
    print(f"  Norm   : {dvector.norm().item():.4f}  (should be 1.0)")
    print(f"  Saved  → {dvec_path}")

    return dvector


# ---------------------------------------------------------------------------
# Stage 7: Prosody Extraction + DTW Warp (Task 3.2)
# ---------------------------------------------------------------------------

def run_prosody_warp(
    synth_path:  str,
    ref_path:    str,
    output_dir:  Path,
    sample_rate: int = 22_050,
) -> str:
    """
    Extract professor's prosody and warp the synthesised audio.

    Returns path to the warped WAV.
    """
    print("\n" + "─" * 60)
    print("STAGE 7  │  Task 3.2 — Prosody Warping (DTW F0 + Energy)")
    print("─" * 60)
    print(f"  Synth   : {synth_path}")
    print(f"  Ref     : {ref_path}")

    warper = ProsodyWarper(sample_rate=sample_rate)

    t0 = time.time()

    # Extract reference prosody
    ref_prosody = warper.extract_prosody(ref_path, label="Professor (reference)")

    # Extract synth prosody for comparison
    synth_wav, sr = torchaudio.load(synth_path)
    synth_wav = synth_wav.mean(dim=0)
    if sr != sample_rate:
        synth_wav = torchaudio.functional.resample(synth_wav, sr, sample_rate)
    synth_prosody = warper.extract_prosody(synth_wav, label="Synthesised")

    # Print comparison
    prosody_text_report(ref_prosody, synth_prosody)

    # Apply DTW warp
    warped = warper.warp(synth_wav, ref_prosody, transfer_f0=True, transfer_energy=True)

    output_path = str(output_dir / "lecture_santhali_final.wav")
    torchaudio.save(output_path, warped.unsqueeze(0), sample_rate)
    elapsed = time.time() - t0

    print(f"\n  DTW prosody warp complete in {elapsed:.1f}s")
    print(f"  Output  → {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Stage 8: VITS Synthesis (Task 3.3)
# ---------------------------------------------------------------------------

def run_synthesis(
    santhali_segments: List[dict],
    dvector:           torch.Tensor,
    output_dir:        Path,
    vits_checkpoint:   Optional[str],
    device:            str,
    sample_rate:       int = 22_050,
    noise_scale:       float = 0.667,
    duration_scale:    float = 1.0,
) -> str:
    """
    Synthesise the full Santhali lecture with zero-shot voice cloning.

    Returns path to the raw synthesised WAV.
    """
    print("\n" + "─" * 60)
    print("STAGE 8  │  Task 3.3 — VITS Zero-Shot TTS (≥22050 Hz)")
    print("─" * 60)
    print(f"  Model   : {vits_checkpoint or 'random weights (demo)'}")
    print(f"  SR      : {sample_rate} Hz")
    print(f"  Segments: {len(santhali_segments)}")
    print(f"  Noise   : {noise_scale}  Duration scale: {duration_scale}")
    assert sample_rate >= 22_050, "Output sample rate must be ≥ 22050 Hz (task requirement)"

    synthesizer = VITSSynthesizer(
        checkpoint_path=vits_checkpoint,
        device=device,
        sample_rate=sample_rate,
        noise_scale=noise_scale,
        duration_scale=duration_scale,
    )

    raw_path = str(output_dir / "lecture_santhali_raw.wav")

    t0 = time.time()
    full_wav = synthesizer.synthesize_lecture(
        santhali_segments=santhali_segments,
        dvector=dvector,
        output_path=raw_path,
        use_ipa=True,
    )
    elapsed = time.time() - t0

    duration_s   = full_wav.shape[-1] / sample_rate
    duration_min = duration_s / 60

    print(f"\n  Synthesis complete in {elapsed:.1f}s")
    print(f"  Total audio duration: {duration_min:.1f} min  ({duration_s:.0f}s)")
    print(f"  Sample rate        : {sample_rate} Hz  ({'✓' if sample_rate >= 22050 else '✗'} ≥22050)")
    print(f"  Raw lecture saved  → {raw_path}")

    return raw_path


# ---------------------------------------------------------------------------
# Part III report
# ---------------------------------------------------------------------------

def write_part3_report(
    output_dir:     Path,
    args:           argparse.Namespace,
    dvector:        torch.Tensor,
    raw_path:       str,
    warped_path:    str,
    timings:        Dict[str, float],
) -> None:
    """Write JSON summary of Part III."""
    # Load final audio for stats
    try:
        wav, sr = torchaudio.load(warped_path)
        dur_s   = wav.shape[-1] / sr
    except Exception:
        dur_s = 0.0
        sr    = args.sample_rate

    report = {
        "config": vars(args),
        "timings_s": timings,
        "speaker_embedding": {
            "shape":       list(dvector.shape),
            "norm":        round(dvector.norm().item(), 4),
            "method":      "GE2E d-vector (LSTM encoder)",
        },
        "synthesis": {
            "model":           "VITS (Variational Inference TTS)",
            "sample_rate":     sr,
            "duration_min":    round(dur_s / 60, 2),
            "output_raw":      raw_path,
            "output_warped":   warped_path,
            "sr_requirement":  f"{'PASS' if sr >= 22050 else 'FAIL'} (≥22050 Hz)",
        },
        "prosody_transfer": {
            "method":     "DTW (Sakoe-Chiba band constrained)",
            "features":   ["F0 (autocorrelation RAPT-style)", "RMS Energy"],
            "pitch_shift":"PSOLA (time-domain overlap-add)",
        },
    }

    path = str(output_dir / "part3_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Part III report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    print("\n" + "=" * 60)
    print("  Part III: Zero-Shot Cross-Lingual Voice Cloning")
    print("=" * 60)
    print(f"  Voice recording : {args.voice_recording}")
    print(f"  Ref lecture     : {args.ref_lecture}")
    print(f"  Santhali JSON   : {args.santhali_json}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Device          : {args.device}")

    # ── Load Santhali segments ───────────────────────────────────────────
    with open(args.santhali_json, encoding="utf-8") as f:
        santhali_segments = json.load(f)
    print(f"\n  Loaded {len(santhali_segments)} Santhali segments.")

    # ── Stage 6: Speaker embedding ───────────────────────────────────────
    t0 = time.time()
    dvector = run_speaker_embedding(
        voice_path=args.voice_recording,
        output_dir=output_dir,
        encoder_path=args.speaker_encoder,
        device=args.device,
    )
    timings["speaker_embedding"] = round(time.time() - t0, 2)

    # ── Stage 8: Synthesis ───────────────────────────────────────────────
    # (Synthesis before prosody warp — warp is applied post-synthesis)
    t0 = time.time()
    raw_wav_path = run_synthesis(
        santhali_segments=santhali_segments,
        dvector=dvector,
        output_dir=output_dir,
        vits_checkpoint=args.vits_checkpoint,
        device=args.device,
        sample_rate=args.sample_rate,
        noise_scale=args.noise_scale,
        duration_scale=args.duration_scale,
    )
    timings["synthesis"] = round(time.time() - t0, 2)

    # ── Stage 7: Prosody warp ────────────────────────────────────────────
    t0 = time.time()
    warped_path = run_prosody_warp(
        synth_path=raw_wav_path,
        ref_path=args.ref_lecture,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
    )
    timings["prosody_warp"] = round(time.time() - t0, 2)

    # ── Report ───────────────────────────────────────────────────────────
    write_part3_report(
        output_dir, args, dvector, raw_wav_path, warped_path, timings
    )

    total = sum(timings.values())
    print("\n" + "=" * 60)
    print("  Part III Complete")
    print("=" * 60)
    print(f"  Speaker embedding : {timings['speaker_embedding']:.1f}s")
    print(f"  TTS synthesis     : {timings['synthesis']:.1f}s")
    print(f"  Prosody warping   : {timings['prosody_warp']:.1f}s")
    print(f"  Total             : {total:.1f}s")
    print(f"\n  Outputs in: {output_dir}")
    print("    my_dvector.pt")
    print("    lecture_santhali_raw.wav    (22050 Hz, pre-warp)")
    print("    lecture_santhali_final.wav  (22050 Hz, DTW prosody)")
    print("    part3_report.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part III: Zero-Shot Cross-Lingual Voice Cloning"
    )
    # Required
    p.add_argument("--voice_recording", required=True,
                   help="60-second personal voice WAV recording")
    p.add_argument("--ref_lecture",     required=True,
                   help="Professor's original lecture WAV (for prosody reference)")
    p.add_argument("--santhali_json",   required=True,
                   help="Part II ipa_transcript.json (with 'ipa' keys)")

    # Checkpoints (optional — demo mode without them)
    p.add_argument("--speaker_encoder", default=None,
                   help="Trained GE2E speaker encoder checkpoint (.pt)")
    p.add_argument("--vits_checkpoint", default=None,
                   help="Trained VITS model checkpoint (.pt)")

    # I/O
    p.add_argument("--output",          default="results/")

    # Synthesis config
    p.add_argument("--sample_rate",    type=int,   default=22_050,
                   help="Output sample rate (≥22050 required)")
    p.add_argument("--noise_scale",    type=float, default=0.667,
                   help="VITS noise scale (0=deterministic, 1=diverse)")
    p.add_argument("--duration_scale", type=float, default=1.0,
                   help="Duration scale factor (>1 = slower speech)")
    p.add_argument("--device",          default="cuda")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
