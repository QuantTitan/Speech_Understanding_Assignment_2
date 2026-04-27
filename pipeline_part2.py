"""
pipeline_part2.py  —  Part II: Phonetic Mapping & Translation
=============================================================
Orchestrates the full Part II pipeline:

  Stage 4  (Task 2.1) → Hinglish G2P → Unified IPA string
  Stage 5  (Task 2.2) → Santhali semantic translation
                         + parallel corpus export

Reads Part I outputs (transcript.txt / lid_segments.json) and produces:
  ipa_transcript.json       per-segment IPA strings + word alignment
  santhali_transcript.txt   Santhali translation (Ol Chiki + Roman)
  santhali_corpus.json      full 500-word parallel corpus
  santhali_corpus.tsv       TSV format for MT toolkit compatibility
  part2_report.json         metrics, coverage, timing

Usage
─────
    python pipeline_part2.py \
        --transcript  results/transcript.txt \
        --lid_json    results/lid_segments.json \
        --output      results/ \
        --target_lang santhali

    # Or run standalone on raw text (no Part I needed)
    python pipeline_part2.py \
        --raw_text "yeh ek deep learning model hai jo speech recognize karta hai" \
        --output   results/
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from g2p_hinglish import HinglishG2P, HinglishIPAResult
from translator_santhali import (
    SanthaliTranslator,
    export_corpus_json,
    export_corpus_tsv,
    corpus_statistics,
    PARALLEL_CORPUS,
)


# ---------------------------------------------------------------------------
# Transcript reader (reads Part I output format)
# ---------------------------------------------------------------------------

def _parse_transcript_txt(path: str) -> List[dict]:
    """
    Parse the transcript.txt produced by pipeline_part1.py.
    Returns list of segment dicts with 'text', 'start_s', 'end_s', 'language'.
    """
    segments = []
    content = Path(path).read_text(encoding="utf-8")

    # Match segment blocks: [EN/HI/CS]  [ 0.0s – 30.0s]\ntext
    pattern = re.compile(
        r"\[(EN|HI|CS)\]\s+\[(\d+\.\d+)s\s*–\s*(\d+\.\d+)s\]\s*\n(.*?)(?=\n\[|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(content):
        lang_tag, start_s, end_s, text = m.groups()
        lang_map = {"EN": "english", "HI": "hindi", "CS": "mixed"}
        segments.append({
            "language": lang_map.get(lang_tag, "mixed"),
            "start_s":  float(start_s),
            "end_s":    float(end_s),
            "text":     text.strip(),
        })

    if not segments:
        # Fallback: treat entire file content as a single segment
        full_text = re.sub(r"=.*?=\n\n", "", content, flags=re.DOTALL).strip()
        segments = [{"language": "mixed", "start_s": 0.0, "end_s": 0.0, "text": full_text}]

    return segments


# ---------------------------------------------------------------------------
# Stage 4: G2P (Task 2.1)
# ---------------------------------------------------------------------------

def run_g2p(
    segments: List[dict],
    lid_segments: Optional[List[dict]],
    output_dir: Path,
) -> List[dict]:
    """
    Convert transcript segments to unified IPA.

    Returns segments with added 'ipa' and 'word_alignments' keys.
    """
    print("\n" + "─" * 60)
    print("STAGE 4  │  Task 2.1 — Hinglish G2P → Unified IPA")
    print("─" * 60)

    g2p = HinglishG2P()
    t0 = time.time()
    ipa_segments = g2p.convert_transcript(segments, lid_segments=lid_segments)
    elapsed = time.time() - t0

    # Statistics
    total_words = sum(len(s.get("word_alignments", [])) for s in ipa_segments)
    lang_counts: Dict[str, int] = {}
    for seg in ipa_segments:
        for w in seg.get("word_alignments", []):
            lang = w["lang"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"  Processed {len(segments)} segments  ({total_words} words)  in {elapsed:.2f}s")
    print(f"  Language breakdown:")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(total_words, 1)
        print(f"    {lang:16s}: {cnt:4d} words  ({pct:.1f}%)")

    # Sample output
    if ipa_segments:
        seg = ipa_segments[0]
        print(f"\n  Sample (segment 0):")
        print(f"    TEXT : {seg['text'][:100]}")
        print(f"    IPA  : {seg['ipa'][:100]}")
        if seg.get("word_alignments"):
            print(f"    Words: ", end="")
            for w in seg["word_alignments"][:5]:
                print(f"  {w['word']} → {w['ipa']} [{w['lang'][:2]}]", end="")
            print()

    # Write IPA transcript JSON
    ipa_path = str(output_dir / "ipa_transcript.json")
    with open(ipa_path, "w", encoding="utf-8") as f:
        json.dump(ipa_segments, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved  → {ipa_path}")

    return ipa_segments


# ---------------------------------------------------------------------------
# Stage 5: Santhali Translation (Task 2.2)
# ---------------------------------------------------------------------------

def run_translation(
    ipa_segments: List[dict],
    output_dir: Path,
) -> List[dict]:
    """
    Translate each segment's text to Santhali.

    Returns segments augmented with 'santhali', 'santhali_roman',
    'translation_coverage', and 'word_map' keys.
    """
    print("\n" + "─" * 60)
    print("STAGE 5  │  Task 2.2 — Santhali Semantic Translation")
    print("─" * 60)

    corpus_statistics()

    translator = SanthaliTranslator(
        min_fuzzy_sim=0.75,
        min_semantic_sim=0.60,
        use_semantic=True,
    )

    t0 = time.time()
    translated = translator.translate_segments(ipa_segments)
    elapsed = time.time() - t0

    # Aggregate coverage
    total_words  = 0
    exact_words  = 0
    fuzzy_words  = 0
    sem_words    = 0
    borrow_words = 0

    for seg in translated:
        for wmap in seg.get("word_map", []):
            total_words += 1
            m = wmap.get("method", "borrowed")
            if m == "exact":       exact_words  += 1
            elif m == "fuzzy":     fuzzy_words  += 1
            elif m == "semantic":  sem_words    += 1
            else:                  borrow_words += 1

    overall_cov = (exact_words + fuzzy_words + sem_words) / max(total_words, 1)

    print(f"  Translated {len(translated)} segments  ({total_words} words)  in {elapsed:.2f}s")
    print(f"\n  Translation method breakdown:")
    print(f"    Exact dictionary : {exact_words:4d} ({100*exact_words/max(total_words,1):.1f}%)")
    print(f"    Fuzzy match      : {fuzzy_words:4d} ({100*fuzzy_words/max(total_words,1):.1f}%)")
    print(f"    Semantic embed   : {sem_words:4d}  ({100*sem_words/max(total_words,1):.1f}%)")
    print(f"    Borrowed/OOV     : {borrow_words:4d} ({100*borrow_words/max(total_words,1):.1f}%)")
    print(f"\n  Overall translation coverage: {overall_cov:.1%}")

    # Sample
    if translated:
        seg = translated[0]
        print(f"\n  Sample (segment 0):")
        print(f"    TEXT : {seg.get('text','')[:100]}")
        print(f"    SA   : {seg.get('santhali','')[:100]}")
        print(f"    RO   : {seg.get('santhali_roman','')[:100]}")

    # Write Santhali transcript
    sa_path = str(output_dir / "santhali_transcript.txt")
    with open(sa_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SANTHALI TRANSLATION — Lecture Transcript\n")
        f.write("Target Language: Santhali (ᱥᱟᱱᱛᱟᱲᱤ / Ol Chiki script)\n")
        f.write("=" * 70 + "\n\n")

        for seg in translated:
            lang_tag = {"english": "[EN]", "hindi": "[HI]", "mixed": "[CS]"}.get(
                seg.get("language", "mixed"), "[??]"
            )
            start = seg.get("start_s", 0.0)
            end   = seg.get("end_s", 0.0)
            cov   = seg.get("translation_coverage", 0.0)

            f.write(f"{lang_tag}  [{start:.1f}s – {end:.1f}s]  (coverage={cov:.0%})\n")
            f.write(f"  SOURCE  : {seg.get('text', '')}\n")
            f.write(f"  IPA     : {seg.get('ipa', '')}\n")
            f.write(f"  SANTHALI: {seg.get('santhali', '')}\n")
            f.write(f"  ROMAN   : {seg.get('santhali_roman', '')}\n")

            # Word-by-word alignment table
            f.write(f"\n  Word Alignment:\n")
            f.write(f"  {'SOURCE':20s}  {'SANTHALI':25s}  {'METHOD':10s}\n")
            f.write(f"  {'-'*60}\n")
            for wmap in seg.get("word_map", [])[:15]:
                f.write(
                    f"  {wmap['source']:20s}  "
                    f"{wmap['santhali']:25s}  "
                    f"{wmap['method']:10s}\n"
                )
            f.write("\n")

    print(f"\n  Santhali transcript → {sa_path}")

    # Export parallel corpus
    corpus_json = str(output_dir / "santhali_corpus.json")
    corpus_tsv  = str(output_dir / "santhali_corpus.tsv")
    export_corpus_json(corpus_json)
    export_corpus_tsv(corpus_tsv)

    return translated


# ---------------------------------------------------------------------------
# Part II report
# ---------------------------------------------------------------------------

def write_part2_report(
    output_dir: Path,
    args: argparse.Namespace,
    ipa_segments: List[dict],
    translated_segments: List[dict],
    timings: Dict[str, float],
) -> None:
    """Summarise Part II in JSON."""
    total_cov = sum(s.get("translation_coverage", 0) for s in translated_segments)
    avg_cov   = total_cov / max(len(translated_segments), 1)

    report = {
        "config":    vars(args),
        "timings_s": timings,
        "g2p": {
            "n_segments": len(ipa_segments),
            "total_words": sum(len(s.get("word_alignments", [])) for s in ipa_segments),
        },
        "translation": {
            "target_language": "Santhali",
            "corpus_size":     len(PARALLEL_CORPUS),
            "n_segments":      len(translated_segments),
            "avg_coverage":    round(avg_cov, 3),
        },
    }
    path = str(output_dir / "part2_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Part II report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    print("\n" + "=" * 60)
    print("  Part II: Phonetic Mapping & Translation")
    print("=" * 60)

    # ── Load transcript ──────────────────────────────────────────────────
    if args.raw_text:
        print(f"\n[Pipeline] Using raw text input.")
        segments = [{"text": args.raw_text, "language": "mixed",
                     "start_s": 0.0, "end_s": 0.0}]
    elif args.transcript:
        print(f"\n[Pipeline] Loading transcript: {args.transcript}")
        segments = _parse_transcript_txt(args.transcript)
        print(f"  Loaded {len(segments)} segments.")
    else:
        raise ValueError("Provide either --transcript or --raw_text")

    # ── Load LID segments ────────────────────────────────────────────────
    lid_segments = None
    if args.lid_json and Path(args.lid_json).exists():
        with open(args.lid_json, encoding="utf-8") as f:
            lid_segments = json.load(f)
        print(f"  LID segments: {len(lid_segments)}")

    # ── Stage 4: G2P ────────────────────────────────────────────────────
    t0 = time.time()
    ipa_segments = run_g2p(segments, lid_segments, output_dir)
    timings["g2p"] = round(time.time() - t0, 2)

    # ── Stage 5: Translation ────────────────────────────────────────────
    t0 = time.time()
    translated = run_translation(ipa_segments, output_dir)
    timings["translation"] = round(time.time() - t0, 2)

    # ── Report ───────────────────────────────────────────────────────────
    write_part2_report(output_dir, args, ipa_segments, translated, timings)

    total = sum(timings.values())
    print("\n" + "=" * 60)
    print("  Part II Complete")
    print("=" * 60)
    print(f"  G2P          : {timings['g2p']:.1f}s")
    print(f"  Translation  : {timings['translation']:.1f}s")
    print(f"  Total        : {total:.1f}s")
    print(f"\n  Output files in: {output_dir}")
    print("    ipa_transcript.json")
    print("    santhali_transcript.txt")
    print("    santhali_corpus.json")
    print("    santhali_corpus.tsv")
    print("    part2_report.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part II Pipeline: G2P → Santhali Translation"
    )
    p.add_argument("--transcript",   default=None,
                   help="Part I transcript.txt path")
    p.add_argument("--lid_json",     default=None,
                   help="Part I lid_segments.json path")
    p.add_argument("--raw_text",     default=None,
                   help="Raw Hinglish text (bypasses Part I)")
    p.add_argument("--output",       default="results/",
                   help="Output directory")
    p.add_argument("--target_lang",  default="santhali",
                   help="Target LRL (currently: santhali)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
