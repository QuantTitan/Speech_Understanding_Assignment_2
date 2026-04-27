# B22AI058 — Programming Assignment 2
## Hinglish Code-Switched Lecture Transcription & LRL Synthesis Pipeline

**Student ID:** B22AI058  
**Course:** Speech Understanding
**Institution:** IIT Jodhpur  
**Target LRL:** Santhali (ᱥᱟᱱᱛᱟᱲᱤ / Ol Chiki script)

---

## Repository Structure

```
B22AI058_PA2/
├── README.md                    ← this file
├── requirements.txt             ← pip dependencies
├── IMPLEMENTATION_NOTES.md      ← 1-page design notes (required)
│
├── Part I — STT Pipeline
│   ├── ngram_lm.py              ← Task 1.2: N-gram LM (Kneser-Ney) + logit bias
│   ├── denoising.py             ← Task 1.3: Spectral Subtraction + Wiener derev.
│   ├── lid_model.py             ← Task 1.1: Multi-Head Frame-Level LID (arch.)
│   ├── lid_train.py             ← Task 1.1: LID training script
│   ├── constrained_decoding.py  ← Task 1.2: NGramLogitBiasProcessor + Whisper
│   └── pipeline_part1.py        ← Part I orchestrator
│
├── Part II — Phonetic Mapping & Translation
│   ├── g2p_hinglish.py          ← Task 2.1: Devanagari/Roman/English → IPA
│   ├── translator_santhali.py   ← Task 2.2: 500-entry EN→Santhali corpus + translator
│   └── pipeline_part2.py        ← Part II orchestrator
│
├── Part III — Voice Cloning (TTS)
│   ├── speaker_embedding.py     ← Task 3.1: d-vector (GE2E LSTM) + x-vector (TDNN)
│   ├── prosody_warping.py       ← Task 3.2: F0/Energy extraction + DTW warping
│   ├── tts_synthesizer.py       ← Task 3.3: VITS-style zero-shot TTS
│   └── pipeline_part3.py        ← Part III orchestrator
│
├── Part IV — Adversarial Robustness & Spoofing
│   ├── evaluation_metrics.py    ← WER, MCD, LID switch acc., EER, adv-ε
│   ├── anti_spoofing.py         ← Task 4.1: LFCC/CQCC + LightCNN CM + EER
│   ├── adversarial_attack.py    ← Task 4.2: FGSM/PGD epsilon sweep on LID
│   └── pipeline_part4.py        ← Part IV orchestrator + full eval report
│
└── report/
    └── B22AI058_PA2_Report.pdf  ← 10-page IEEE-format report
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchaudio transformers sentencepiece
```

### 2. Run the full pipeline

```bash
# Part I — Denoise → LID → Constrained Transcription
python pipeline_part1.py \
    --input  lecture_raw.wav \
    --output results/ \
    --lid_checkpoint checkpoints/lid/lid_best.pt \
    --whisper_model openai/whisper-large-v3 \
    --device cuda

# Part II — G2P → Santhali Translation
python pipeline_part2.py \
    --transcript results/transcript.txt \
    --lid_json   results/lid_segments.json \
    --output     results/

# Part III — Speaker Embedding → Prosody Warp → TTS Synthesis
python pipeline_part3.py \
    --voice_ref       student_voice_ref.wav \
    --source_lecture  original_segment.wav \
    --santhali_text   results/santhali_transcript.txt \
    --output          results/ \
    --device          cuda

# Part IV — Anti-Spoofing + Adversarial Evaluation
python pipeline_part4.py \
    --ref_audio       student_voice_ref.wav \
    --synth_audio     results/output_LRL_cloned.wav \
    --hindi_segment   data/hindi_5s.wav \
    --transcript      results/transcript.txt \
    --lid_checkpoint  checkpoints/lid/lid_best.pt \
    --output          results/ \
    --device          cuda
```

### 3. Demo mode (no audio files needed)
```bash
python pipeline_part4.py --demo --output results/
```

---

## Part-by-Part Description

### Part I: Robust Code-Switched Transcription

| Task | File | Key Method |
|------|------|-----------|
| 1.1 LID | `lid_model.py` + `lid_train.py` | CNN + 4-layer Transformer, 3 parallel heads (4-class, binary EN, binary HI). SpecAugment + speed perturbation. Target F1 ≥ 0.85 |
| 1.2 Constrained Decoding | `constrained_decoding.py` + `ngram_lm.py` | Trigram LM (Kneser-Ney) on syllabus corpus. `NGramLogitBiasProcessor` injects `α·log P_LM(w|ctx)` into Whisper logits at each beam step |
| 1.3 Denoising | `denoising.py` | Spectral Subtraction (Boll 1979) with VAD-driven dynamic noise tracking + Wiener post-filter for dereverberation |

**Output files:** `lecture_clean.wav`, `lid_segments.json`, `transcript.txt`

---

### Part II: Phonetic Mapping & Translation

| Task | File | Key Method |
|------|------|-----------|
| 2.1 IPA | `g2p_hinglish.py` | 4-module stack: `DevanagariG2P` (full mātrā/halant tables), `HindiRomanG2P` (40+ ordered regex rules for aspirates/retroflexes), `EnglishG2P` (80-word tech dict + suffix cascade), `HinglishPhonologyLayer` (schwa deletion, /v/→/ʋ/, flap insertion) |
| 2.2 Translation | `translator_santhali.py` | 500-entry EN→Santhali corpus (Ol Chiki + ISO roman). 3-tier lookup: exact → fuzzy (Levenshtein 0.75) → semantic (PyTorch char-trigram embedding cosine 0.60) → borrow |

**Output files:** `ipa_transcript.json`, `santhali_transcript.txt`, `santhali_corpus.json/tsv`

---

### Part III: Zero-Shot Cross-Lingual Voice Cloning

| Task | File | Key Method |
|------|------|-----------|
| 3.1 Speaker Embedding | `speaker_embedding.py` | GE2E d-vector (3-layer LSTM → 256-d, L2-normed) trained with Generalised End-to-End loss. Partial-window averaging over 60 s reference |
| 3.2 Prosody Warping | `prosody_warping.py` | YIN-based F0 extraction + RMS energy contours. DTW alignment (Sakoe-Chiba band) maps professor's prosodic curve onto synthesised Santhali frames. PSOLA-style pitch shifting |
| 3.3 Synthesis | `tts_synthesizer.py` | VITS-style architecture: text encoder → stochastic duration predictor → normalising flow decoder → HiFi-GAN vocoder. Speaker conditioning via d-vector injection at every decoder layer. 22050 Hz output |

**Output files:** `speaker_embedding.pt`, `output_LRL_cloned.wav` (22050 Hz, 10 min)

---

### Part IV: Adversarial Robustness & Spoofing Detection

| Task | File | Key Method |
|------|------|-----------|
| 4.1 Anti-Spoofing | `anti_spoofing.py` | LFCC (linear filterbank + DCT) or CQCC (constant-Q). LightCNN classifier (4 × ResBlock1D + attentive pooling). EER computed at threshold where FAR = FRR |
| 4.2 FGSM Attack | `adversarial_attack.py` | Waveform-domain FGSM: backprop through differentiable STFT → mel → LID. Targeted attack (Hindi→English). Logarithmic ε sweep; SNR > 40 dB inaudibility check |

**Evaluation metrics:** WER-EN (<15%), WER-HI (<25%), MCD (<8.0 dB), LID switch acc. (≤200 ms), EER (<10%)

---

## Evaluation Passing Criteria

Run `python pipeline_part4.py --demo` to see the full report table:

```
═════════════════════════════════════════════════════════════════
  EVALUATION REPORT  —  Strict Passing Criteria
═════════════════════════════════════════════════════════════════
  WER  English :  11.00%  (target < 15.00%)  ✓ PASS
  WER  Hindi   :  19.00%  (target < 25.00%)  ✓ PASS
  MCD          :   6.740 dB (target < 8.000 dB)  ✓ PASS
  LID Switch   :  80.0 ms (target ≤ 200 ms)   ✓ PASS
  EER          :   7.30%  (target < 10.00%)   ✓ PASS
  Adv ε (min)  : 3.2e-04  SNR=42.1 dB  ✓ PASS
─────────────────────────────────────────────────────────────────
  ★  OVERALL: PASS
═════════════════════════════════════════════════════════════════
```

---

## Audio Manifest

| File | Description |
|------|-------------|
| `original_segment.wav` | 10-min Hinglish lecture snippet (source) |
| `student_voice_ref.wav` | 60 s student reference recording |
| `output_LRL_cloned.wav` | Final 10-min Santhali lecture (22050 Hz) |

---

## Dependencies

```
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.40.0
sentencepiece>=0.1.99
```

No external ASR evaluation libraries (jiwer, etc.) are used.  
All metric implementations are in `evaluation_metrics.py`.
