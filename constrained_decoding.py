"""
constrained_decoding.py  — Task 1.2
=====================================
Whisper-based ASR with two interleaved constrained-decoding mechanisms:

  1. **N-gram Logit Bias**       — At every beam-search step, log-probs from
     the syllabus-trained NGramLM are added (scaled) to Whisper's raw logits,
     boosting technical terms.

  2. **Language-Aware Logit Mask** — LID frame predictions are used to
     *suppress* tokens belonging to the wrong language at each decoder step,
     implementing a soft constrained beam search that respects code-switching
     boundaries.

Implementation notes
────────────────────
• Uses the `transformers` Whisper model via its `generate()` API.
• Constrained decoding is injected through a custom `LogitsProcessorList`:
    - `NGramLogitBiasProcessor`       (boosts LM-favoured tokens)
    - `LanguageConstraintProcessor`   (suppresses cross-language tokens)
• A custom `BeamScorer` override is *not* needed — the processors run inside
  the standard HuggingFace beam search loop.

Dependencies
────────────
    pip install transformers>=4.40 sentencepiece

Usage
─────
    from constrained_decoding import ConstrainedWhisperTranscriber

    transcriber = ConstrainedWhisperTranscriber(
        model_name="openai/whisper-large-v3",
        lm_path="ngram_lm.pkl",              # pre-built NGramLM
        device="cuda",
    )
    result = transcriber.transcribe("lecture_clean.wav", lid_segments=segs)
    print(result["text"])
    print(result["segments"])                # list with timestamps + lang
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio

# HuggingFace Transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
)

from ngram_lm import NGramLM, build_syllabus_lm


# ---------------------------------------------------------------------------
# Helper: convert LID segment list to per-Whisper-chunk language label
# ---------------------------------------------------------------------------

def _dominant_language(segments: List[dict], start_s: float, end_s: float) -> str:
    """
    Return the dominant language label ('english'|'hindi'|'mixed') for the
    time span [start_s, end_s] based on LID segment predictions.
    """
    duration: Dict[str, float] = {"english": 0.0, "hindi": 0.0,
                                   "mixed": 0.0, "silence": 0.0}
    for seg in segments:
        overlap = min(seg["end_s"], end_s) - max(seg["start_s"], start_s)
        if overlap > 0:
            lang = seg["label"].lower()
            duration[lang] = duration.get(lang, 0.0) + overlap

    # Silence doesn't count toward dominant language
    duration.pop("silence", None)
    if not duration or max(duration.values()) == 0.0:
        return "mixed"
    return max(duration, key=lambda k: duration[k])


# ---------------------------------------------------------------------------
# Logits Processor 1: N-gram Language Model Logit Bias
# ---------------------------------------------------------------------------

class NGramLogitBiasProcessor(LogitsProcessor):
    """
    Adds N-gram LM log-probability bias to Whisper's raw logits at each
    decoding step.

    The bias is computed only for tokens whose *surface text* appears in the
    NGramLM vocabulary (i.e. the syllabus), so the vocabulary intersection is
    cached at construction time.

    Parameters
    ----------
    lm           : Trained NGramLM.
    tokenizer    : Whisper tokenizer (provides vocabulary).
    scale        : Multiplier for LM log-probs before adding to logits.
    tech_boost   : Extra log-boost for registered technical terms.
    max_bias     : Hard cap on the maximum bias value (prevents domination).
    """

    def __init__(
        self,
        lm: NGramLM,
        tokenizer,
        scale: float = 3.0,
        tech_boost: float = 2.5,
        max_bias: float = 6.0,
    ) -> None:
        self.lm = lm
        self.tokenizer = tokenizer
        self.scale = scale
        self.tech_boost = tech_boost
        self.max_bias = max_bias

        # Pre-build: {surface_text → token_id} for words in LM vocab
        self._vocab_map: Dict[str, int] = {}
        full_vocab = tokenizer.get_vocab()                 # {str: int}
        lm_words = lm.vocab | set(lm._technical_terms)

        for surface, tok_id in full_vocab.items():
            # Whisper uses " word" (leading space) for mid-sentence tokens
            word = surface.strip().lower()
            word_clean = re.sub(r"[^a-z0-9\-']", "", word)
            if word_clean in lm_words:
                self._vocab_map[word_clean] = tok_id

        print(
            f"[NGramBias] {len(self._vocab_map)} LM vocabulary tokens matched "
            f"to Whisper tokenizer."
        )

        # Running decoded-word context (shared across batch by resetting per-chunk)
        self._decoded_words: List[str] = []

    def reset_context(self, seed_words: Optional[List[str]] = None) -> None:
        """Call before each new audio chunk."""
        self._decoded_words = seed_words or []

    def _update_context_from_ids(self, input_ids: torch.Tensor) -> None:
        """Decode the most recent token from the beam and update word context."""
        if input_ids.shape[1] < 1:
            return
        last_tok = input_ids[0, -1].item()
        surface = self.tokenizer.decode([last_tok], skip_special_tokens=True)
        words = surface.strip().lower().split()
        self._decoded_words.extend(words)
        # Keep only last n-1 words as context
        self._decoded_words = self._decoded_words[-(self.lm.n - 1):]

    def __call__(
        self,
        input_ids: torch.Tensor,     # [batch, seq_len]
        scores: torch.Tensor,        # [batch, vocab_size]
    ) -> torch.Tensor:
        self._update_context_from_ids(input_ids)
        ctx = tuple(self._decoded_words[-(self.lm.n - 1):])

        bias = torch.zeros_like(scores[0])   # [vocab_size]

        for word, tok_id in self._vocab_map.items():
            lp = self.lm.log_prob(ctx, word)
            b  = self.scale * lp
            if word in self.lm._technical_terms:
                b += self.tech_boost
            bias[tok_id] = min(b, self.max_bias)

        return scores + bias.unsqueeze(0)    # broadcast over batch


# ---------------------------------------------------------------------------
# Logits Processor 2: Language Constraint (soft masking)
# ---------------------------------------------------------------------------

class LanguageConstraintProcessor(LogitsProcessor):
    """
    Suppresses tokens belonging to the *wrong* language at each decode step,
    implementing a soft constraint that follows LID segment boundaries.

    Mechanism
    ---------
    For each audio chunk, we know the dominant language from the LID system.
    • If dominant == 'english', apply a negative bias to Hindi-script tokens.
    • If dominant == 'hindi',   apply a negative bias to purely-ASCII tokens
      that are not technical terms.
    • If dominant == 'mixed',   no suppression.

    The suppression is a soft penalty (not hard masking) to preserve beam
    search diversity and allow borrowed words.

    Parameters
    ----------
    tokenizer     : Whisper tokenizer.
    penalty       : Negative bias applied to off-language tokens (e.g. -5.0).
    tech_tokens   : Set of token IDs that should never be suppressed (technical).
    """

    # Devanagari Unicode range: U+0900 – U+097F
    _DEVANAGARI = re.compile(r"[\u0900-\u097F]")

    def __init__(
        self,
        tokenizer,
        penalty: float = -5.0,
        tech_tokens: Optional[set] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.tech_tokens = tech_tokens or set()

        # Pre-compute sets of English (ASCII-only) and Hindi (Devanagari) tokens
        vocab = tokenizer.get_vocab()
        self._english_token_ids: set = set()
        self._hindi_token_ids:   set = set()

        for surface, tok_id in vocab.items():
            clean = surface.strip()
            if not clean or clean.startswith("<"):
                continue
            if self._DEVANAGARI.search(clean):
                self._hindi_token_ids.add(tok_id)
            elif clean.isascii() and re.search(r"[a-zA-Z]", clean):
                self._english_token_ids.add(tok_id)

        # Current dominant language (set per chunk)
        self._dominant: str = "mixed"

    def set_language(self, dominant: str) -> None:
        """Call before transcribing each chunk with the dominant LID label."""
        self._dominant = dominant.lower()

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if self._dominant == "mixed":
            return scores

        modified = scores.clone()

        if self._dominant == "english":
            # Suppress Hindi/Devanagari tokens
            suppress_ids = self._hindi_token_ids - self.tech_tokens
        elif self._dominant == "hindi":
            # Suppress ASCII tokens (except technical terms + numbers)
            suppress_ids = self._english_token_ids - self.tech_tokens
        else:
            return scores

        for tok_id in suppress_ids:
            if tok_id < modified.shape[-1]:
                modified[:, tok_id] += self.penalty

        return modified


# ---------------------------------------------------------------------------
# Constrained Whisper Transcriber
# ---------------------------------------------------------------------------

class ConstrainedWhisperTranscriber:
    """
    End-to-end constrained Whisper transcriber for code-switched lectures.

    Integrates:
        • Pre-trained Whisper (large-v3 or medium)
        • NGram logit bias   (Task 1.2a)
        • LID-aware language constraint  (Task 1.2b)

    Parameters
    ----------
    model_name  : HuggingFace model ID (e.g. "openai/whisper-large-v3").
    lm_path     : Path to a pickled NGramLM, or None to build from corpus.
    device      : 'cuda' | 'cpu'.
    beam_size   : Number of beams for beam search.
    lm_scale    : N-gram log-prob scale (α in logit = α·log P_LM + log P_whisper).
    tech_boost  : Additional bonus for registered technical terms.
    lang_penalty: Negative penalty for cross-language tokens.
    chunk_s     : Whisper audio chunk length in seconds (max 30).
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        lm_path: Optional[str] = None,
        device: str = "cuda",
        beam_size: int = 5,
        lm_scale: float = 3.0,
        tech_boost: float = 2.5,
        lang_penalty: float = -5.0,
        chunk_s: float = 30.0,
    ) -> None:
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.beam_size = beam_size
        self.chunk_s   = chunk_s

        print(f"[Whisper] Loading model: {model_name}  →  {self.device}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model     = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

        self.tokenizer = self.processor.tokenizer
        self.sr        = self.processor.feature_extractor.sampling_rate  # 16 000

        # ── NGram LM ──────────────────────────────────────────────────────
        if lm_path and Path(lm_path).exists():
            self.lm = NGramLM.load(lm_path)
        else:
            print("[Whisper] Building syllabus LM from default corpus …")
            self.lm = build_syllabus_lm(n=3)

        # Pre-compute technical term token IDs for the constraint processor
        tech_token_ids = set()
        vocab = self.tokenizer.get_vocab()
        for term in self.lm._technical_terms:
            for surface, tok_id in vocab.items():
                if term in surface.strip().lower():
                    tech_token_ids.add(tok_id)

        # ── Logits Processors ─────────────────────────────────────────────
        self.ngram_processor = NGramLogitBiasProcessor(
            self.lm, self.tokenizer, scale=lm_scale, tech_boost=tech_boost
        )
        self.lang_processor = LanguageConstraintProcessor(
            self.tokenizer, penalty=lang_penalty, tech_tokens=tech_token_ids
        )

    # ── Audio loading ────────────────────────────────────────────────────

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load WAV → mono 16 kHz float32 [T]."""
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)                                # mono [T]
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    # ── Chunk-level transcription ────────────────────────────────────────

    def _transcribe_chunk(
        self,
        audio_chunk: torch.Tensor,         # [T]  at 16 kHz
        dominant_lang: str,
        init_prompt: Optional[str] = None,
    ) -> str:
        """Transcribe a single audio chunk (≤ 30 s) with constrained decoding."""

        # Feature extraction
        inputs = self.processor(
            audio_chunk.numpy(),
            sampling_rate=self.sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            self.device,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        # Reset N-gram context for this chunk
        seed = init_prompt.lower().split()[-3:] if init_prompt else []
        self.ngram_processor.reset_context(seed)
        self.lang_processor.set_language(dominant_lang)

        logits_processor = LogitsProcessorList([
            self.ngram_processor,
            self.lang_processor,
        ])

        # Forced decoder IDs: language token + task token
        # For code-switched Hindi+English we use multilingual mode
        forced_ids = self.processor.get_decoder_prompt_ids(
            language="hi" if dominant_lang == "hindi" else "en",
            task="transcribe",
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                num_beams=self.beam_size,
                logits_processor=logits_processor,
                forced_decoder_ids=forced_ids,
                max_new_tokens=448,
                repetition_penalty=1.2,
            )

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return text

    # ── Full-file transcription ──────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str,
        lid_segments: Optional[List[dict]] = None,
    ) -> Dict:
        """
        Transcribe an audio file with constrained decoding.

        Parameters
        ----------
        audio_path   : Path to (denoised) WAV file.
        lid_segments : List of LID segment dicts from `predict_language_segments`.
                       If None, assumes mixed language throughout.

        Returns
        -------
        {
          "text"     : str,            # full concatenated transcript
          "segments" : List[dict],     # per-chunk info with timestamps & lang
          "lm_used"  : bool,
        }
        """
        audio = self._load_audio(audio_path)           # [T]
        duration_s = audio.shape[0] / self.sr
        print(f"[Transcriber] Audio: {duration_s:.1f}s  →  {audio_path}")

        chunk_samples = int(self.chunk_s * self.sr)
        n_chunks = math.ceil(audio.shape[0] / chunk_samples)

        result_segments = []
        full_text_parts = []
        prev_text = ""

        for i in range(n_chunks):
            start_sample = i * chunk_samples
            end_sample   = min((i + 1) * chunk_samples, audio.shape[0])
            start_s = start_sample / self.sr
            end_s   = end_sample   / self.sr

            chunk = audio[start_sample:end_sample]

            # Determine dominant language from LID
            if lid_segments:
                dominant = _dominant_language(lid_segments, start_s, end_s)
            else:
                dominant = "mixed"

            print(
                f"  Chunk {i+1:02d}/{n_chunks}  "
                f"[{start_s:.1f}s – {end_s:.1f}s]  lang={dominant}"
            )

            text = self._transcribe_chunk(
                chunk,
                dominant_lang=dominant,
                init_prompt=prev_text[-200:] if prev_text else None,
            )
            prev_text = text

            result_segments.append({
                "start_s":   start_s,
                "end_s":     end_s,
                "text":      text,
                "language":  dominant,
            })
            full_text_parts.append(text)

        full_text = " ".join(full_text_parts)
        print(f"\n[Transcriber] Transcript ({len(full_text)} chars):\n{full_text[:500]}…")

        return {
            "text":     full_text,
            "segments": result_segments,
            "lm_used":  True,
        }


# ---------------------------------------------------------------------------
# Convenience function for pipeline integration
# ---------------------------------------------------------------------------

def transcribe_lecture(
    audio_path: str,
    lid_segments: Optional[List[dict]] = None,
    model_name: str = "openai/whisper-large-v3",
    lm_path: Optional[str] = "ngram_lm.pkl",
    device: str = "cuda",
    beam_size: int = 5,
    lm_scale: float = 3.0,
    tech_boost: float = 2.5,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Single-function interface for the full constrained transcription pipeline.
    Writes transcript to `output_path` if provided.
    """
    transcriber = ConstrainedWhisperTranscriber(
        model_name=model_name,
        lm_path=lm_path,
        device=device,
        beam_size=beam_size,
        lm_scale=lm_scale,
        tech_boost=tech_boost,
    )
    result = transcriber.transcribe(audio_path, lid_segments=lid_segments)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("=== Full Transcript ===\n\n")
            f.write(result["text"])
            f.write("\n\n=== Segments ===\n\n")
            for seg in result["segments"]:
                f.write(
                    f"[{seg['start_s']:.1f}s – {seg['end_s']:.1f}s]"
                    f"  [{seg['language'].upper()}]\n"
                    f"{seg['text']}\n\n"
                )
        print(f"[Transcriber] Transcript saved → {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Constrained Whisper Transcription (Task 1.2)")
    p.add_argument("audio",              help="Path to (denoised) WAV file")
    p.add_argument("--model",            default="openai/whisper-large-v3")
    p.add_argument("--lm",               default="ngram_lm.pkl")
    p.add_argument("--lid_json",         default=None, help="JSON file of LID segments")
    p.add_argument("--output",           default="transcript.txt")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--beams",    type=int, default=5)
    p.add_argument("--lm_scale", type=float, default=3.0)
    args = p.parse_args()

    lid_segs = None
    if args.lid_json:
        with open(args.lid_json) as f:
            lid_segs = json.load(f)

    result = transcribe_lecture(
        args.audio,
        lid_segments=lid_segs,
        model_name=args.model,
        lm_path=args.lm,
        device=args.device,
        beam_size=args.beams,
        lm_scale=args.lm_scale,
        output_path=args.output,
    )
    print("\nDone.")
