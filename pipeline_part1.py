"""
pipeline_part1.py  —  Part I: Robust Code-Switched Transcription
================================================================
End-to-end STT pipeline for Hinglish lecture audio.

Stage 1 (Task 1.3)  → Denoising + Normalisation (Spectral Subtraction)
Stage 2 (Task 1.1)  → Frame-level Language Identification (Multi-Head LID)
Stage 3 (Task 1.2)  → Constrained Whisper Transcription
                        (N-gram logit bias + LID-aware language constraint)

Quick Start
───────────
    # Full pipeline (all three tasks)
    python pipeline_part1.py \
        --input  lecture_raw.wav \
        --output results/ \
        --lid_checkpoint checkpoints/lid/lid_best.pt \
        --whisper_model openai/whisper-large-v3 \
        --lm ngram_lm.pkl \
        --device cuda

    # Skip LID (run without a trained checkpoint)
    python pipeline_part1.py \
        --input lecture_raw.wav \
        --output results/ \
        --skip_lid \
        --device cuda

Output artefacts in --output/
──────────────────────────────
    lecture_clean.wav       cleaned audio  (Task 1.3)
    lid_segments.json       per-frame LID predictions  (Task 1.1)
    transcript.txt          full annotated transcript  (Task 1.2)
    pipeline_report.json    metrics + config summary
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

from denoising import AudioPreprocessor
from lid_model import LIDConfig, MelFeatureExtractor, MultiHeadLID, predict_language_segments
from ngram_lm import build_syllabus_lm, NGramLM
from constrained_decoding import ConstrainedWhisperTranscriber


# ---------------------------------------------------------------------------
# Stage 1: Denoising (Task 1.3)
# ---------------------------------------------------------------------------

def run_denoising(
    input_path: str,
    output_dir: Path,
    over_subtraction: float = 2.0,
    spectral_floor: float = 0.002,
    target_rms: float = 0.05,
    denoise_only: bool = False,
) -> str:
    """
    Apply spectral subtraction + Wiener dereverberation.

    Returns
    -------
    Path to the cleaned WAV file.
    """
    print("\n" + "─" * 60)
    print("STAGE 1  │  Task 1.3 — Audio Denoising & Normalisation")
    print("─" * 60)

    preprocessor = AudioPreprocessor(
        sample_rate=16_000,
        over_subtraction=over_subtraction,
        spectral_floor=spectral_floor,
        denoise_only=denoise_only,
        target_rms=target_rms,
    )

    waveform, sr = torchaudio.load(input_path)
    print(f"  Input  : {input_path}  shape={tuple(waveform.shape)}  sr={sr}")

    t0 = time.time()
    clean, out_sr = preprocessor(waveform, sample_rate=sr)
    elapsed = time.time() - t0

    # Diagnostics
    in_rms  = waveform.pow(2).mean().sqrt().item()
    out_rms = clean.pow(2).mean().sqrt().item()
    print(f"  RMS     in={in_rms:.4f}  out={out_rms:.4f}")
    print(f"  Duration: {clean.shape[-1]/out_sr:.1f}s  (processed in {elapsed:.2f}s)")

    output_path = str(output_dir / "lecture_clean.wav")
    torchaudio.save(output_path, clean, out_sr)
    print(f"  Saved  → {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Stage 2: Language Identification (Task 1.1)
# ---------------------------------------------------------------------------

def run_lid(
    clean_audio_path: str,
    output_dir: Path,
    checkpoint_path: Optional[str],
    device: torch.device,
    min_segment_frames: int = 5,
) -> List[dict]:
    """
    Run the multi-head LID model on the denoised audio.

    Returns
    -------
    List of language segment dicts (see `predict_language_segments`).
    """
    print("\n" + "─" * 60)
    print("STAGE 2  │  Task 1.1 — Frame-Level Language Identification")
    print("─" * 60)

    cfg = LIDConfig()
    model = MultiHeadLID(cfg).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        metrics = ckpt.get("metrics", {})
        print(f"  Loaded checkpoint: {checkpoint_path}  (epoch {epoch})")
        if metrics:
            print(f"  Saved F1 metrics: {metrics}")
    else:
        print(
            "  WARNING: No LID checkpoint provided or found. "
            "Using randomly initialised weights — LID predictions will be unreliable.\n"
            "  Train first:  python lid_train.py --data_dir data/lid --save_dir checkpoints/lid"
        )

    model.eval()
    feat_extractor = MelFeatureExtractor(cfg).to(device)

    waveform, sr = torchaudio.load(clean_audio_path)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0, keepdim=True)   # mono

    t0 = time.time()
    segments = predict_language_segments(
        model, feat_extractor,
        waveform.squeeze(0),
        sample_rate=sr,
        min_segment_frames=min_segment_frames,
    )
    elapsed = time.time() - t0

    print(f"  Found {len(segments)} language segments  ({elapsed:.2f}s)")
    for seg in segments[:10]:   # print first 10
        print(
            f"    [{seg['start_s']:6.1f}s – {seg['end_s']:6.1f}s]  "
            f"{seg['label']:10s}  "
            f"en_p={seg['en_prob']:.2f}  hi_p={seg['hi_prob']:.2f}"
        )
    if len(segments) > 10:
        print(f"    … ({len(segments) - 10} more segments)")

    # Language distribution statistics
    lang_dur: Dict[str, float] = {}
    for seg in segments:
        dur = seg["end_s"] - seg["start_s"]
        lang_dur[seg["label"]] = lang_dur.get(seg["label"], 0.0) + dur
    total_dur = sum(lang_dur.values()) + 1e-8
    print("\n  Language distribution:")
    for lang, dur in sorted(lang_dur.items(), key=lambda x: -x[1]):
        print(f"    {lang:12s}: {dur:.1f}s  ({100*dur/total_dur:.1f}%)")

    # Save JSON
    lid_json_path = str(output_dir / "lid_segments.json")
    with open(lid_json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"\n  Saved  → {lid_json_path}")

    return segments


# ---------------------------------------------------------------------------
# Stage 3: Constrained Transcription (Task 1.2)
# ---------------------------------------------------------------------------

def run_transcription(
    clean_audio_path: str,
    lid_segments: List[dict],
    output_dir: Path,
    whisper_model: str,
    lm_path: Optional[str],
    device: str,
    beam_size: int,
    lm_scale: float,
    tech_boost: float,
    lang_penalty: float,
) -> Dict:
    """
    Run constrained Whisper transcription with N-gram logit bias and
    LID-aware language constraint.

    Returns the full result dict from ConstrainedWhisperTranscriber.
    """
    print("\n" + "─" * 60)
    print("STAGE 3  │  Task 1.2 — Constrained Whisper Transcription")
    print("─" * 60)
    print(f"  Model       : {whisper_model}")
    print(f"  Beam size   : {beam_size}")
    print(f"  LM scale    : {lm_scale}")
    print(f"  Tech boost  : {tech_boost}")
    print(f"  Lang penalty: {lang_penalty}")

    t0 = time.time()
    transcriber = ConstrainedWhisperTranscriber(
        model_name=whisper_model,
        lm_path=lm_path,
        device=device,
        beam_size=beam_size,
        lm_scale=lm_scale,
        tech_boost=tech_boost,
        lang_penalty=lang_penalty,
    )

    result = transcriber.transcribe(clean_audio_path, lid_segments=lid_segments)
    elapsed = time.time() - t0

    # Write transcript
    transcript_path = str(output_dir / "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("HINGLISH LECTURE TRANSCRIPT — Constrained Whisper + NGram LM\n")
        f.write("=" * 70 + "\n\n")
        f.write("── Full Text ──\n\n")
        f.write(result["text"])
        f.write("\n\n")
        f.write("── Segments (with language tags) ──\n\n")
        for seg in result["segments"]:
            lang_tag = {
                "english": "[EN]",
                "hindi":   "[HI]",
                "mixed":   "[CS]",   # code-switched
            }.get(seg["language"], "[??]")
            f.write(
                f"{lang_tag}  [{seg['start_s']:6.1f}s – {seg['end_s']:6.1f}s]\n"
                f"{seg['text']}\n\n"
            )

    print(f"\n  Transcription complete in {elapsed:.1f}s")
    print(f"  Saved  → {transcript_path}")

    return result


# ---------------------------------------------------------------------------
# Pipeline report
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    lid_segments: List[dict],
    transcript_result: Dict,
    timings: Dict[str, float],
) -> None:
    """Write a JSON summary of the pipeline run."""
    report = {
        "config": vars(args),
        "timings_s": timings,
        "lid": {
            "n_segments": len(lid_segments),
            "language_distribution": {},
        },
        "transcript": {
            "total_chars": len(transcript_result.get("text", "")),
            "n_chunks":    len(transcript_result.get("segments", [])),
        },
    }
    for seg in lid_segments:
        lang = seg["label"]
        dur  = seg["end_s"] - seg["start_s"]
        report["lid"]["language_distribution"][lang] = (
            report["lid"]["language_distribution"].get(lang, 0.0) + dur
        )

    path = str(output_dir / "pipeline_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Pipeline report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    timings: Dict[str, float] = {}

    print("\n" + "=" * 60)
    print("  Part I: Robust Code-Switched STT Pipeline")
    print("=" * 60)
    print(f"  Input  : {args.input}")
    print(f"  Output : {output_dir}")
    print(f"  Device : {device}")

    # ── Stage 1: Denoising ──────────────────────────────────────────────
    t0 = time.time()
    clean_path = run_denoising(
        input_path=args.input,
        output_dir=output_dir,
        over_subtraction=args.alpha,
        spectral_floor=args.beta,
        target_rms=args.target_rms,
        denoise_only=args.denoise_only,
    )
    timings["denoising"] = round(time.time() - t0, 2)

    # ── Stage 2: LID ────────────────────────────────────────────────────
    t0 = time.time()
    if args.skip_lid:
        print("\n[Pipeline] Skipping LID (--skip_lid flag set).")
        lid_segments: List[dict] = []
    else:
        lid_segments = run_lid(
            clean_audio_path=clean_path,
            output_dir=output_dir,
            checkpoint_path=args.lid_checkpoint,
            device=device,
            min_segment_frames=args.min_segment_frames,
        )
    timings["lid"] = round(time.time() - t0, 2)

    # ── Stage 3: Transcription ──────────────────────────────────────────
    t0 = time.time()
    transcript_result = run_transcription(
        clean_audio_path=clean_path,
        lid_segments=lid_segments,
        output_dir=output_dir,
        whisper_model=args.whisper_model,
        lm_path=args.lm,
        device=args.device,
        beam_size=args.beams,
        lm_scale=args.lm_scale,
        tech_boost=args.tech_boost,
        lang_penalty=args.lang_penalty,
    )
    timings["transcription"] = round(time.time() - t0, 2)

    # ── Summary ─────────────────────────────────────────────────────────
    write_report(output_dir, args, lid_segments, transcript_result, timings)

    total = sum(timings.values())
    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    print(f"  Denoising    : {timings['denoising']:.1f}s")
    print(f"  LID          : {timings['lid']:.1f}s")
    print(f"  Transcription: {timings['transcription']:.1f}s")
    print(f"  Total        : {total:.1f}s")
    print(f"\n  Output files in: {output_dir}")
    print("    lecture_clean.wav")
    print("    lid_segments.json")
    print("    transcript.txt")
    print("    pipeline_report.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part I STT Pipeline: Denoise → LID → Constrained Transcription"
    )

    # I/O
    p.add_argument("--input",  required=True,  help="Raw lecture WAV file")
    p.add_argument("--output", default="results/", help="Output directory")

    # Task 1.3 – Denoising
    p.add_argument("--alpha",      type=float, default=2.0,
                   help="Spectral subtraction over-subtraction α  [1.0–3.0]")
    p.add_argument("--beta",       type=float, default=0.002,
                   help="Spectral subtraction floor β")
    p.add_argument("--target_rms", type=float, default=0.05,
                   help="Target RMS for output normalisation")
    p.add_argument("--denoise_only", action="store_true",
                   help="Skip Wiener dereverberation post-filter")

    # Task 1.1 – LID
    p.add_argument("--lid_checkpoint", default=None,
                   help="Path to trained LID checkpoint (.pt)")
    p.add_argument("--skip_lid", action="store_true",
                   help="Skip LID stage (run in monolingual Whisper mode)")
    p.add_argument("--min_segment_frames", type=int, default=5,
                   help="Min frames for a valid LID segment")

    # Task 1.2 – Constrained Decoding
    p.add_argument("--whisper_model", default="openai/whisper-large-v3",
                   help="HuggingFace Whisper model ID")
    p.add_argument("--lm",           default="ngram_lm.pkl",
                   help="Path to pickled NGramLM (built if missing)")
    p.add_argument("--beams",        type=int,   default=5)
    p.add_argument("--lm_scale",     type=float, default=3.0,
                   help="N-gram log-prob scale  (α in logit bias)")
    p.add_argument("--tech_boost",   type=float, default=2.5,
                   help="Extra logit bonus for technical terms")
    p.add_argument("--lang_penalty", type=float, default=-5.0,
                   help="Negative penalty for cross-language tokens")

    # Compute
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
