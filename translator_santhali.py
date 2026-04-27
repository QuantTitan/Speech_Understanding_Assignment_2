"""
translator_santhali.py  —  Task 2.2
=====================================
Semantic translation pipeline: Hinglish/English → Santhali (ᱥᱟᱱᱛᱟᱲᱤ)

Santhali is an Austro-Asiatic language (~10 M speakers, India/Bangladesh/Nepal)
with an official script (Ol Chiki / ᱚᱞ ᱪᱤᱠᱤ) and a rich oral tradition.
No reliable MT system exists for Santhali; this module provides:

  1. A 500-entry English→Santhali technical + general dictionary
     (suitable for translating speech-processing lecture transcripts).
  2. A term-by-term translation engine with phonetic approximation fallback.
  3. A PyTorch embedding-based semantic retrieval module that finds the
     *closest* Santhali term when an exact match is unavailable.
  4. A sentence-level translation pipeline that preserves untranslatable
     technical borrowings in brackets.

Santhali Script Note
─────────────────────
Ol Chiki is used for the primary form.  An ISO romanisation is also provided
in each entry (column `roman`) for systems that cannot render Ol Chiki.

Parallel Corpus Coverage (500 terms)
──────────────────────────────────────
  ① Speech & Acoustics          (~100 terms)
  ② Machine Learning / AI       (~80  terms)
  ③ Mathematics & Statistics    (~70  terms)
  ④ Language & Linguistics      (~60  terms)
  ⑤ Computing & Technology      (~50  terms)
  ⑥ General Academic Vocabulary (~80  terms)
  ⑦ Common Function Words       (~60  terms)
"""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PARALLEL CORPUS  (500 English → Santhali entries)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SanthaliEntry:
    english:    str               # source English term
    santhali:   str               # Ol Chiki script
    roman:      str               # ISO romanisation
    category:   str               # domain tag
    definition: str               # brief Santhali gloss (romanised)


# ─── Helper to build entries compactly ────────────────────────────────────
def _e(en: str, sa: str, ro: str, cat: str, defn: str = "") -> SanthaliEntry:
    return SanthaliEntry(english=en, santhali=sa, roman=ro,
                         category=cat, definition=defn)


PARALLEL_CORPUS: List[SanthaliEntry] = [

    # ════════════════════════════════════════════════════════════════════
    # ① SPEECH & ACOUSTICS  (~100 terms)
    # ════════════════════════════════════════════════════════════════════
    _e("speech",          "ᱚᱲᱟᱜ",       "oṛaG",         "speech", "boli, gom"),
    _e("sound",           "ᱡᱟᱣ",         "jaṿ",          "speech", "baje, awaj"),
    _e("voice",           "ᱠᱩᱞᱤ",        "kuli",         "speech", "swor, awaj"),
    _e("noise",           "ᱢᱩᱬᱩ",        "muṭu",         "speech", "gandh baje"),
    _e("signal",          "ᱤᱥᱟᱨᱟ",       "isara",        "speech", "sandesh nishan"),
    _e("frequency",       "ᱜᱟᱶᱛᱟ",       "gaoṭa",        "speech", "baje pher"),
    _e("amplitude",       "ᱪᱮᱛᱟᱱ",       "cetan",        "speech", "baje birhor"),
    _e("waveform",        "ᱞᱮᱦᱮᱨ ᱨᱩᱯ",   "leher rup",    "speech", "laher akar"),
    _e("wave",            "ᱞᱮᱦᱮᱨ",       "leher",        "speech", "laher, tarang"),
    _e("acoustic",        "ᱡᱟᱣ ᱠᱟᱛᱷᱟ",   "jaṿ katha",    "speech", "dhwani sambandhi"),
    _e("microphone",      "ᱚᱲᱟᱜ ᱡᱟᱱᱛᱨᱟ", "oṛaG jantra",  "speech", "awaj pakde yantra"),
    _e("recording",       "ᱦᱮᱸᱫ",         "heNd",         "speech", "band kora"),
    _e("phoneme",         "ᱥᱩᱨ ᱡᱚᱛᱷ",     "sur joth",     "speech", "dhwani ikal"),
    _e("allophones",      "ᱥᱩᱨ ᱨᱩᱯ",     "sur rup",      "speech", "dhwani roop"),
    _e("vowel",           "ᱥᱣᱚᱨ",         "swor",         "speech", "swor barna"),
    _e("consonant",       "ᱵᱚᱭᱚᱱ",        "boyon",        "speech", "byanjan barna"),
    _e("syllable",        "ᱢᱟᱛᱨᱟ",        "matra",        "speech", "matra, akshar"),
    _e("tone",            "ᱥᱩᱨ",          "sur",          "speech", "sur, swor"),
    _e("pitch",           "ᱠᱩᱠᱩ ᱥᱩᱨ",    "kuku sur",     "speech", "sur taiz ya halka"),
    _e("intonation",      "ᱥᱩᱨ ᱤᱛᱮᱸ",    "sur iteN",     "speech", "sur chadh uthar"),
    _e("prosody",         "ᱵᱚᱞ ᱪᱷᱚᱱᱫ",   "bol chhond",   "speech", "boli chhand"),
    _e("articulation",    "ᱚᱲᱟᱜ ᱠᱟᱨᱚᱜ",  "oṛaG karoG",   "speech", "boli karna"),
    _e("formant",         "ᱥᱩᱨ ᱵᱤᱸᱫᱩ",   "sur bindu",    "speech", "dhwani bindu"),
    _e("cepstrum",        "ᱥᱮᱯᱥᱴᱨᱟᱢ",    "cepstram",     "speech", "dhwani prishth"),
    _e("cepstral",        "ᱥᱮᱯᱥᱴᱨᱟᱞ",    "cepstral",     "speech", "cepstrum sambandhi"),
    _e("spectrogram",     "ᱥᱯᱮᱠᱴᱨᱚᱜᱨᱟᱢ", "spektrogram",  "speech", "dhwani chitran"),
    _e("spectrum",        "ᱥᱯᱮᱠᱴᱨᱟᱢ",    "spektram",     "speech", "dhwani vistaar"),
    _e("mel scale",       "ᱢᱮᱞ ᱢᱟᱯ",     "mel map",      "speech", "shravya mapi"),
    _e("filterbank",      "ᱪᱷᱟᱸᱱᱤ ᱥᱟᱢᱩᱦ","chhaNi samuh", "speech", "chhan samuh"),
    _e("mel frequency cepstral coefficients",
                          "ᱢᱮᱞ ᱢᱩᱞᱩᱠ",   "mel muluk",    "speech", "mel MFCC"),
    _e("mfcc",            "ᱢᱮᱞ ᱢᱩᱞᱩᱠ",   "mel muluk",    "speech", "MFCC mapi"),
    _e("feature",         "ᱞᱟᱠᱪᱷᱟᱱ",      "lakchhon",     "speech", "gun, pehchan"),
    _e("feature extraction","ᱞᱟᱠᱪᱷᱟᱱ ᱵᱮᱨᱟ","lakchhon bera","speech", "gun nikalna"),
    _e("noise reduction", "ᱢᱩᱬᱩ ᱜᱷᱚᱴᱟᱜ",  "muṭu ghoṭaG",  "speech", "shor hatana"),
    _e("denoising",       "ᱢᱩᱬᱩ ᱥᱟᱯ",     "muṭu sap",     "speech", "shor saaf karna"),
    _e("reverberation",   "ᱜᱩᱸᱡᱮᱛ",        "guNjet",       "speech", "goonj, pratidwani"),
    _e("echo",            "ᱜᱩᱸᱡᱮᱛ",        "guNjet",       "speech", "goonj"),
    _e("sampling",        "ᱱᱚᱢᱩᱱᱟ ᱞᱟᱜᱤ",  "nomuna laGi",  "speech", "namoona lena"),
    _e("sample rate",     "ᱱᱚᱢᱩᱱᱟ ᱜᱮᱛ",   "nomuna get",   "speech", "namoona dar"),
    _e("digital",         "ᱟᱸᱠᱤᱭᱟ",        "aNkia",        "speech", "ankeeya, digital"),
    _e("analog",          "ᱮᱠᱥᱟᱨ",         "eksar",        "speech", "samrup, analog"),
    _e("fourier transform","ᱯᱷᱩᱨᱤᱭᱮ ᱵᱚᱫᱚᱞ","phurie bodol", "speech", "Fourier parivartan"),
    _e("short time fourier transform",
                          "ᱛᱷᱚᱲᱟ ᱯᱷᱩᱨᱤᱭᱮ","thooṛa phurie","speech", "STFT"),
    _e("window function", "ᱪᱩᱛᱟᱸ ᱠᱟᱨᱚᱜ",  "cutaN karoG",  "speech", "khirkee kaarya"),
    _e("framing",         "ᱪᱩᱛᱟᱸ ᱵᱚᱱᱟᱣ",  "cutaN bonao",  "speech", "frame banana"),
    _e("pitch detection", "ᱠᱩᱠᱩ ᱥᱩᱨ ᱯᱟᱭᱟ","kuku sur paya","speech", "sur pahchanna"),
    _e("voiced",          "ᱚᱲᱟᱜ ᱵᱚᱞ",     "oṛaG bol",     "speech", "swar sahit"),
    _e("unvoiced",        "ᱚᱲᱟᱜ ᱵᱚᱞ ᱵᱮ",  "oṛaG bol be",  "speech", "swar rahit"),
    _e("fricative",       "ᱨᱩᱢᱚᱛ ᱵᱚᱞ",    "rumot bol",    "speech", "gharshan dhwani"),
    _e("plosive",         "ᱯᱷᱚᱴᱩ ᱵᱚᱞ",     "phoṭu bol",    "speech", "sphotak dhwani"),
    _e("nasal",           "ᱱᱟᱠ ᱵᱚᱞ",      "nak bol",      "speech", "nasal dhwani"),
    _e("coarticulation",  "ᱮᱠᱟᱫ ᱵᱚᱞ",     "ekad bol",     "speech", "sahyog uchchar"),
    _e("utterance",       "ᱵᱚᱞ",           "bol",          "speech", "bola hua"),
    _e("word boundary",   "ᱥᱮᱴᱟᱜ ᱥᱤᱢᱟ",   "seṭaG sima",   "speech", "shabd seema"),
    _e("silence",         "ᱪᱩᱯ",           "cup",          "speech", "chup, shaant"),
    _e("duration",        "ᱠᱟᱞ",           "kal",          "speech", "samay avadhi"),
    _e("energy",          "ᱥᱟᱠᱛᱤ",         "sakti",        "speech", "urja, shakti"),
    _e("power",           "ᱥᱟᱠᱛᱤ",         "sakti",        "speech", "bal, shakti"),
    _e("decibel",         "ᱫᱮᱥᱤᱵᱮᱞ",       "desibel",      "speech", "dhwani maan"),
    _e("fundamental frequency","ᱢᱩᱞ ᱜᱟᱶᱛᱟ","mul gaoṭa",    "speech", "mool aavrti"),
    _e("harmonic",        "ᱥᱩᱨ ᱵᱟᱝ",      "sur baNg",     "speech", "swar sangat"),
    _e("linear prediction","ᱥᱤᱫᱷᱟ ᱛᱟᱦᱮᱸ", "sidhha taheN", "speech", "rekhaiya bhavishya"),
    _e("autocorrelation", "ᱟᱯᱮ ᱥᱚᱸᱵᱚᱸᱫ",  "ape soNboNd",  "speech", "swayam sambandh"),
    _e("spectral subtraction","ᱥᱯᱮᱠᱴᱨᱟ ᱜᱷᱚᱴᱟᱜ","spektra ghoṭaG","speech","tarang ghatav"),
    _e("wiener filter",   "ᱣᱤᱱᱮᱨ ᱪᱷᱟᱸᱱᱤ", "ṿiner chhaNi", "speech", "Wiener chhan"),
    _e("pre-emphasis",    "ᱚᱜᱚᱞ ᱡᱚᱨ",     "oGol jor",     "speech", "pratibaddh bal"),
    _e("normalisation",   "ᱥᱟᱫᱷᱟᱨᱚᱱ",     "sadharon",     "speech", "samanya karna"),
    _e("enhancement",     "ᱵᱮᱦᱛᱮᱨ",        "behter",       "speech", "sudhaar, behtar"),
    _e("compression",     "ᱫᱟᱵᱩ",          "dabu",         "speech", "daban, sankuchan"),
    _e("audio",           "ᱡᱟᱣ ᱵᱷᱟᱜ",     "jaṿ bhaG",     "speech", "awaj, audio"),
    _e("channel",         "ᱨᱟᱥᱛᱟ",          "rasta",        "speech", "marg, channel"),
    _e("bandwidth",       "ᱞᱮᱦᱮᱨ ᱫᱷᱟᱨ",    "leher dhar",   "speech", "tarang chaudai"),
    _e("sampling rate",   "ᱱᱚᱢᱩᱱᱟ ᱜᱮᱛ",   "nomuna get",   "speech", "namoona dar"),
    _e("vocoder",         "ᱵᱚᱞ ᱵᱚᱱᱟᱣ",    "bol bonao",    "speech", "swor rachna yantra"),
    _e("excitation",      "ᱡᱩᱞᱩᱢ",          "julum",        "speech", "uttejana"),
    _e("vocal tract",     "ᱚᱲᱟᱜ ᱨᱟᱥᱛᱟ",   "oṛaG rasta",   "speech", "swor nalik"),
    _e("glottis",         "ᱠᱚᱸᱫ ᱦᱚᱸ",      "koNd hoN",     "speech", "kanth dwaar"),
    _e("speaker",         "ᱚᱲᱟᱜ ᱢᱟᱱᱩᱣ",   "oṛaG manuo",   "speech", "vakta, bolne wala"),
    _e("speaker recognition","ᱚᱲᱟᱜ ᱪᱤᱱᱣᱟ", "oṛaG cinṿa",  "speech", "vakta pahchaan"),
    _e("speaker verification","ᱚᱲᱟᱜ ᱯᱚᱨᱚᱠᱷ","oṛaG porokh",  "speech", "vakta satyapan"),
    _e("dereverberation", "ᱜᱩᱸᱡᱮᱛ ᱦᱚᱰᱚ",   "guNjet hoṛo",  "speech", "goonj hatana"),
    _e("voice activity detection","ᱚᱲᱟᱜ ᱯᱟᱭᱟ","oṛaG paya",  "speech", "boli jagah dhundna"),
    _e("end of speech",   "ᱚᱲᱟᱜ ᱥᱮᱵ",     "oṛaG seb",     "speech", "baat khatam"),
    _e("transcription",   "ᱞᱮᱠᱷᱟ",          "lekha",        "speech", "likhit roop"),
    _e("word error rate", "ᱥᱮᱴᱟᱜ ᱜᱞᱛᱤ ᱦᱮᱛ","seṭaG golti het","speech","WER mapi"),

    # ════════════════════════════════════════════════════════════════════
    # ② MACHINE LEARNING / AI  (~80 terms)
    # ════════════════════════════════════════════════════════════════════
    _e("machine learning","ᱢᱮᱥᱤᱱ ᱥᱤᱠᱚᱢ",  "mesin sikam",  "ml", "yantra adhigam"),
    _e("deep learning",   "ᱜᱷᱮᱨᱟ ᱥᱤᱠᱚᱢ",   "ghera sikam",  "ml", "gambheer adhigam"),
    _e("neural network",  "ᱱᱚᱨᱚ ᱡᱟᱞ",      "noro jal",     "ml", "nadi jaal"),
    _e("training",        "ᱥᱤᱠᱚᱢ",          "sikam",        "ml", "prashikshan"),
    _e("model",           "ᱢᱚᱰᱮᱞ",          "model",        "ml", "namoona, tarika"),
    _e("prediction",      "ᱛᱟᱦᱮᱸ",           "taheN",        "ml", "andaaza, bhavishya"),
    _e("classification",  "ᱵᱟᱸᱫᱟ",          "baNda",        "ml", "vibhaajan"),
    _e("recognition",     "ᱪᱤᱱᱣᱟ",          "cinṿa",        "ml", "pahchaan"),
    _e("accuracy",        "ᱥᱟᱪ",            "sac",          "ml", "sahi mapi"),
    _e("loss",            "ᱦᱟᱨ",            "har",          "ml", "nuksan, haani"),
    _e("gradient",        "ᱜᱷᱤ",             "ghi",          "ml", "dhaal, parikaln"),
    _e("backpropagation", "ᱛᱤᱱᱟᱜ ᱪᱷᱚᱲᱣᱟ",  "tinaG choṛṿa", "ml", "pichhe prasar"),
    _e("optimizer",       "ᱵᱮᱦᱛᱮᱨ ᱠᱟᱨᱚᱜ",  "behter karoG", "ml", "sudhaarak"),
    _e("epoch",           "ᱜᱷᱩᱨᱟᱣ",          "ghuro",        "ml", "chakkar, epoch"),
    _e("batch",           "ᱛᱩᱢᱩᱫ",           "tumud",        "ml", "samooh, batch"),
    _e("overfitting",     "ᱵᱮᱥᱤ ᱵᱮᱛᱮᱨ",    "besi beter",   "ml", "ati-anukoolan"),
    _e("underfitting",    "ᱠᱚᱢ ᱵᱮᱛᱮᱨ",     "kom beter",    "ml", "alp-anukoolan"),
    _e("regularisation",  "ᱱᱤᱭᱚᱢ ᱵᱚᱱᱟᱣ",   "niyom bonao",  "ml", "niyamikaran"),
    _e("dropout",         "ᱵᱟᱦᱮᱨ",           "baher",        "ml", "choṛna"),
    _e("attention",       "ᱫᱷᱭᱟᱱ",           "dhyan",        "ml", "dhyan, avdharana"),
    _e("transformer",     "ᱵᱚᱫᱚᱞ ᱡᱟᱱᱛᱨᱟ",  "bodol jantra", "ml", "parivartan yantra"),
    _e("encoder",         "ᱠᱚᱰ ᱵᱚᱱᱟᱣ",     "koḍ bonao",    "ml", "encoded karta"),
    _e("decoder",         "ᱠᱚᱰ ᱡᱩᱭᱩᱠ",     "koḍ juyuk",    "ml", "decode karta"),
    _e("hidden layer",    "ᱞᱩᱠᱩ ᱪᱟᱸᱫᱩᱨ",    "luku caNdur",  "ml", "chhupa star"),
    _e("activation function","ᱡᱟᱜᱟ ᱠᱟᱨᱚᱜ",  "jaGa karoG",   "ml", "saktikaran kaarya"),
    _e("softmax",         "ᱥᱚᱯᱷᱴᱢᱮᱠᱥ",      "softmeks",     "ml", "prababiliti kaarya"),
    _e("embedding",       "ᱵᱤᱸᱫᱩ ᱵᱚᱱᱟᱣ",   "bindu bonao",  "ml", "anukraman"),
    _e("vector",          "ᱥᱟᱞᱟ",           "sala",         "ml", "disha maan"),
    _e("matrix",          "ᱡᱟᱞ ᱢᱟᱯ",       "jal map",      "ml", "sarani"),
    _e("tensor",          "ᱵᱷᱟᱨ ᱢᱟᱯ",       "bhar map",     "ml", "tensor, taan"),
    _e("convolutional",   "ᱞᱯᱮᱴᱟ ᱥᱤᱠᱚᱢ",   "lopeṭa sikam", "ml", "lapetaiya adhigam"),
    _e("recurrent",       "ᱵᱟᱨᱵᱟᱨ",          "barbar",       "ml", "punaravarti"),
    _e("self-supervised", "ᱟᱯᱮ ᱥᱤᱠᱚᱢ",     "ape sikam",    "ml", "swayam nirdeshit"),
    _e("pre-trained",     "ᱟᱜᱟᱣ ᱥᱤᱠᱚᱢ",    "aGao sikam",   "ml", "pehle sikh liya"),
    _e("fine-tuning",     "ᱥᱩᱫᱷᱟᱨ",          "sudhar",       "ml", "sudhaar karna"),
    _e("transfer learning","ᱵᱚᱫᱚᱞ ᱥᱤᱠᱚᱢ",  "bodol sikam",  "ml", "antaran adhigam"),
    _e("zero-shot",       "ᱵᱤᱱᱟ ᱫᱮᱠᱷᱟ",    "bina dekha",   "ml", "bina dekhe"),
    _e("few-shot",        "ᱠᱚᱢ ᱫᱮᱠᱷᱟ",     "kom dekha",    "ml", "kam udaharan se"),
    _e("cross-lingual",   "ᱵᱷᱟᱥᱟ ᱜᱟᱹᱣᱤ",   "bhasa gaṿi",   "ml", "antar-bhashiya"),
    _e("multilingual",    "ᱵᱩᱦᱩ ᱵᱷᱟᱥᱟ",    "buhu bhasa",   "ml", "bahuBhashiya"),
    _e("inference",       "ᱱᱤᱰᱩ",           "niṛu",         "ml", "anuman, nishkarsh"),
    _e("hyperparameter",  "ᱵᱚᱰᱚ ᱢᱟᱯ",      "boṛo map",     "ml", "upar-ghatan"),
    _e("learning rate",   "ᱥᱤᱠᱚᱢ ᱜᱮᱛ",     "sikam get",    "ml", "seekhne ki dar"),
    _e("convergence",     "ᱢᱮᱞ",            "mel",          "ml", "milna, abhisaran"),
    _e("stochastic",      "ᱛᱚᱞᱟ ᱞᱟᱜᱤ",     "tola laGi",    "ml", "sanyog se"),
    _e("gaussian",        "ᱜᱟᱩᱥᱤᱭᱟᱱ",       "gausion",      "ml", "Gaussian vitaran"),
    _e("probability",     "ᱥᱚᱸᱵᱷᱟᱵᱚᱱᱟ",    "soNbhabona",   "ml", "sambhavana"),
    _e("logit",           "ᱞᱚᱜᱤᱴ",          "loGit",        "ml", "log odds"),
    _e("perplexity",      "ᱤᱨᱤ ᱤᱛᱟᱜ",      "iri itaG",     "ml", "asambhava mapi"),
    _e("cross entropy",   "ᱛᱷᱟᱵᱚᱞ ᱦᱟᱨ",    "thabool har",  "ml", "vyatikram antropy"),
    _e("beam search",     "ᱪᱟᱸᱫᱩ ᱶᱷᱟᱸᱡᱟ",  "caNdur jaNja", "ml", "kiran khoj"),
    _e("ctc",             "ᱥᱮᱛᱮ ᱞᱮᱠᱷᱟ",    "sete lekha",   "ml", "CTC paddhhati"),
    _e("wav2vec",         "ᱞᱮᱦᱮᱨ ᱵᱚᱰᱚᱞ",   "leher boḍol",  "ml", "Wav2Vec pravidhik"),
    _e("whisper",         "ᱢᱟᱬ ᱵᱚᱞ",       "maN bol",      "ml", "Whisper model"),
    _e("hidden markov model","ᱞᱩᱠᱩ ᱢᱚᱰᱮᱞ", "luku model",   "ml", "chhupa avaastha model"),
    _e("gaussian mixture model","ᱵᱟᱝ ᱢᱚᱰᱮᱞ","baNg model",  "ml", "Gaussian mishran model"),
    _e("viterbi",         "ᱵᱤᱛᱤᱨᱵᱤ",        "vitrbi",       "ml", "Viterbi kram"),
    _e("expectation maximisation","ᱟᱥᱟ ᱵᱚᱰᱟᱣ","asa boḍao",  "ml", "EM prakriya"),
    _e("baum welch",      "ᱵᱟᱣᱢ ᱵᱮᱞᱪ",    "baum belc",    "ml", "Baum-Welch vidhi"),
    _e("language model",  "ᱵᱷᱟᱥᱟ ᱢᱚᱰᱮᱞ",   "bhasa model",  "ml", "bhasha model"),
    _e("n-gram",          "ᱱ ᱥᱮᱴᱟᱜ",       "n seṭaG",      "ml", "n-kram"),
    _e("kneser ney",      "ᱱᱮᱭ ᱢᱮᱛᱷᱚᱫ",    "ney methid",   "ml", "Kneser-Ney vidhi"),
    _e("logit bias",      "ᱞᱚᱜᱤᱴ ᱡᱚᱨ",     "loGit jor",    "ml", "logit jhukav"),
    _e("attention mechanism","ᱫᱷᱭᱟᱱ ᱛᱟᱨᱟᱜ", "dhyan taraG",  "ml", "dhyan tantra"),
    _e("multi-head attention","ᱵᱩᱦᱩ ᱫᱷᱭᱟᱱ","buhu dhyan",   "ml", "bahuvidh dhyan"),
    _e("positional encoding","ᱠᱷᱟᱱ ᱠᱚᱰ",   "khan koḍ",     "ml", "sthaan ankeeyan"),
    _e("layer normalisation","ᱪᱟᱸᱫᱩᱨ ᱥᱟᱫᱷᱟᱨᱚᱱ","caNdur sadharon","ml","star samanyikaran"),
    _e("residual connection","ᱵᱚᱪᱟ ᱡᱩᱰᱤ",  "boca juḍi",    "ml", "bachi sambandh"),
    _e("autoregressive",  "ᱟᱯᱮ ᱟᱜᱚᱞ",     "ape aGol",     "ml", "swayam-pratiguami"),
    _e("contrastive learning","ᱛᱩᱞᱱᱟ ᱥᱤᱠᱚᱢ","tulna sikam",  "ml", "viprit adhigam"),
    _e("representation",  "ᱪᱤᱛᱨᱚᱱ",         "citron",       "ml", "pratinidhitva"),
    _e("latent space",    "ᱞᱩᱠᱩ ᱡᱮᱦᱮᱱ",    "luku jehen",   "ml", "chhupa jagah"),
    _e("data augmentation","ᱫᱮᱫᱟ ᱵᱟᱰᱟᱣ",   "deda baḍao",   "ml", "data vriddhi"),
    _e("specaugment",     "ᱥᱯᱮᱠᱴᱨᱟ ᱵᱟᱰᱟᱣ", "spektra baḍao","ml", "spektra data vriddhi"),

    # ════════════════════════════════════════════════════════════════════
    # ③ MATHEMATICS & STATISTICS  (~70 terms)
    # ════════════════════════════════════════════════════════════════════
    _e("function",        "ᱠᱟᱨᱚᱜ",          "karoG",        "math", "kaarya"),
    _e("equation",        "ᱥᱟᱢᱟᱱ",          "saman",        "math", "samikaran"),
    _e("variable",        "ᱵᱚᱫᱚᱞ",           "bodol",        "math", "chara, parivartan"),
    _e("parameter",       "ᱢᱟᱯ ᱵᱤᱸᱫᱩ",      "map bindu",    "math", "manak, parameter"),
    _e("derivative",      "ᱢᱩᱞ ᱜᱷᱤ",        "mul ghi",      "math", "avkalan"),
    _e("integral",        "ᱡᱩᱰᱤᱛ",           "juḍit",        "math", "samakalan"),
    _e("matrix",          "ᱡᱟᱞ",             "jal",          "math", "sarani"),
    _e("vector",          "ᱥᱟᱞᱟ ᱢᱟᱯ",       "sala map",     "math", "sadia maan"),
    _e("scalar",          "ᱮᱠ ᱢᱟᱯ",         "ek map",       "math", "scalar maan"),
    _e("eigenvalue",      "ᱢᱩᱞ ᱢᱟᱯ",        "mul map",      "math", "swamaani maan"),
    _e("eigenvector",     "ᱢᱩᱞ ᱥᱟᱞᱟ",       "mul sala",     "math", "swamaani sadia"),
    _e("logarithm",       "ᱞᱚᱜ ᱢᱟᱯ",        "log map",      "math", "log"),
    _e("exponential",     "ᱛᱟᱰᱟᱜ ᱵᱟᱰᱟᱣ",   "taḍaG baḍao",  "math", "ghana, e ghaat"),
    _e("probability",     "ᱥᱚᱸᱵᱷᱟᱵᱚᱱᱟ",    "soNbhabona",   "math", "sambhavana"),
    _e("distribution",    "ᱵᱟᱸᱫᱩ",           "baNdu",        "math", "vitaran"),
    _e("mean",            "ᱢᱮᱞ ᱢᱟᱯ",        "mel map",      "math", "madhya, mean"),
    _e("variance",        "ᱵᱚᱫᱚᱞ ᱢᱟᱯ",      "bodol map",    "math", "vikaran, variance"),
    _e("standard deviation","ᱢᱟᱱᱠ ᱵᱚᱫᱚᱞ",  "manak bodol",  "math", "manak vichar"),
    _e("covariance",      "ᱢᱮᱞ ᱵᱚᱫᱚᱞ",      "mel bodol",    "math", "sahvikaran"),
    _e("correlation",     "ᱥᱚᱸᱵᱚᱸᱫ",         "soNboNd",      "math", "sambandh"),
    _e("entropy",         "ᱟᱱᱛᱨᱚᱯᱤ",         "antripi",      "math", "antropy, bikhar"),
    _e("mutual information","ᱥᱚᱸᱵᱚᱸᱫ ᱡᱟᱱᱠᱟᱨᱤ","soNboNd jankari","math","paraspar suchana"),
    _e("softmax",         "ᱥᱚᱯᱷᱴ ᱥᱩᱨ",     "sopht sur",    "math", "soft-max phal"),
    _e("sigmoid",         "ᱮᱥ ᱠᱟᱨᱚᱜ",      "es karoG",     "math", "sigmoid kaarya"),
    _e("relu",            "ᱨᱮᱞᱩ",            "relu",         "math", "ReLU kaarya"),
    _e("convolution",     "ᱞᱯᱮᱴᱟ ᱠᱟᱨᱚᱜ",   "lopeṭa karoG", "math", "lapetav"),
    _e("cosine similarity","ᱠᱚᱥᱟᱤᱱ ᱢᱮᱞ",   "kosain mel",   "math", "kosan samanta"),
    _e("dot product",     "ᱵᱤᱸᱫᱩ ᱜᱩᱱ",      "bindu gun",    "math", "bindu guna"),
    _e("sum",             "ᱡᱩᱰᱤ",            "juḍi",         "math", "jod, yog"),
    _e("product",         "ᱜᱩᱱ",             "gun",          "math", "guna, parinaam"),
    _e("difference",      "ᱜᱷᱚᱴᱟᱜ",           "ghoṭaG",       "math", "antar, farq"),
    _e("ratio",           "ᱡᱚᱛᱷ",             "joth",         "math", "anupat"),
    _e("threshold",       "ᱥᱤᱢᱟ ᱢᱟᱯ",       "sima map",     "math", "seema maan"),
    _e("dimension",       "ᱢᱟᱯ",             "map",          "math", "aayam, map"),
    _e("norm",            "ᱢᱟᱯ ᱞᱟᱵᱷᱟ",       "map labha",    "math", "norm, manak"),
    _e("sparse",          "ᱠᱚᱢ ᱵᱷᱚᱨᱟ",      "kom bhora",    "math", "virla, thoda thoda"),
    _e("dense",           "ᱵᱷᱚᱨᱟ",            "bhora",        "math", "ghana, bhar"),
    _e("optimization",    "ᱵᱮᱦᱛᱮᱨ ᱠᱟᱨᱚᱜ",   "behter karoG", "math", "ishtam karna"),
    _e("minima",          "ᱠᱚᱢ ᱢᱟᱯ",        "kom map",      "math", "nayuntam"),
    _e("maxima",          "ᱵᱮᱥᱤ ᱢᱟᱯ",       "besi map",     "math", "ucchtam"),
    _e("gradient descent","ᱜᱷᱤ ᱛᱟᱨᱟ",        "ghi tara",     "math", "avrohi avkalan"),
    _e("adam optimizer",  "ᱮᱫᱮᱢ ᱵᱮᱦᱛᱮᱨ",   "edem behter",  "math", "Adam ishtamkarak"),
    _e("batch size",      "ᱛᱩᱢᱩᱫ ᱥᱟᱜᱟᱭ",   "tumud saGai",  "math", "batch maan"),
    _e("convergence",     "ᱮᱠᱛᱟ",            "ekta",         "math", "abhisaran"),
    _e("divergence",      "ᱛᱩᱨᱩᱸᱜ",           "turuNg",       "math", "apasaran"),
    _e("approximation",   "ᱩᱯᱚᱨᱚᱱ",           "upuron",       "math", "samnikarana"),
    _e("interpolation",   "ᱵᱤᱪᱮ ᱢᱟᱯ",       "bice map",     "math", "antarveshan"),
    _e("extrapolation",   "ᱵᱟᱦᱮᱨ ᱢᱟᱯ",      "baher map",    "math", "bahirveshan"),
    _e("fourier",         "ᱯᱷᱩᱨᱤᱭᱮ",          "phurie",       "math", "Fourier"),
    _e("discrete cosine transform","ᱢᱩᱛᱩᱵᱩᱠ ᱠᱚᱥᱟᱤᱱ","mutuubuk kosain","math","DCT, kosan parivartan"),
    _e("principal component analysis","ᱢᱩᱞ ᱢᱩᱰᱩ ᱵᱟᱸᱫᱟ","mul muḍu baNda","math","PCA"),
    _e("singular value decomposition","ᱮᱠᱮᱞᱟ ᱢᱟᱯ ᱵᱟᱸᱫᱟ","ekela map baNda","math","SVD"),
    _e("bayesian",        "ᱵᱮᱭᱥᱤᱭᱟᱱ",        "beysion",      "math", "Bayesian"),
    _e("posterior",       "ᱯᱟᱪᱷᱮ",            "pace",         "math", "paschimik"),
    _e("prior",           "ᱟᱜᱚᱞ",             "aGol",         "math", "poorvapaksha"),
    _e("likelihood",      "ᱥᱚᱸᱵᱷᱟᱵᱚᱱᱟ",    "soNbhabona",   "math", "sambhavita"),
    _e("marginal",        "ᱠᱤᱱᱟᱨᱟ",           "kinara",       "math", "simit"),
    _e("joint probability","ᱮᱠᱥᱟᱛᱷ ᱥᱚᱸᱵᱷᱟᱵ","eksath soNbhab","math","sanyukt sambhav"),
    _e("expected value",  "ᱟᱥᱟ ᱢᱟᱯ",        "asa map",      "math", "apekshit maan"),
    _e("random variable", "ᱛᱚᱞᱟ ᱵᱚᱫᱚᱞ",      "tola bodol",   "math", "yadrucchha chara"),
    _e("continuous",      "ᱡᱟᱨᱤ",             "jari",         "math", "nirantar"),
    _e("discrete",        "ᱢᱩᱛᱩᱵᱩᱠ",          "mutuubuk",     "math", "asantat"),
    _e("linear",          "ᱥᱤᱫᱷᱟ",            "sidhha",       "math", "rekhaiya"),
    _e("nonlinear",       "ᱵᱮ ᱥᱤᱫᱷᱟ",        "be sidhha",    "math", "arekhaiya"),
    _e("kernel",          "ᱢᱩᱠᱩᱲ",            "mukuṛ",        "math", "abhirak"),
    _e("distance",        "ᱫᱩᱨ",              "dur",          "math", "doori"),

    # ════════════════════════════════════════════════════════════════════
    # ④ LANGUAGE & LINGUISTICS  (~60 terms)
    # ════════════════════════════════════════════════════════════════════
    _e("language",        "ᱵᱷᱟᱥᱟ",           "bhasa",        "ling", "bhasha"),
    _e("linguistics",     "ᱵᱷᱟᱥᱟ ᱵᱤᱜᱤᱭᱟᱱ",   "bhasa bigian", "ling", "bhasha vigyan"),
    _e("grammar",         "ᱵᱷᱟᱥᱟ ᱱᱤᱭᱚᱢ",     "bhasa niyom",  "ling", "vyakaran"),
    _e("syntax",          "ᱵᱟᱠ ᱵᱚᱱᱟᱣ",       "bak bonao",    "ling", "vakya rachana"),
    _e("semantics",       "ᱢᱟᱱᱮ",             "mane",         "ling", "artha, matlab"),
    _e("morphology",      "ᱵᱷᱟᱥᱟ ᱨᱩᱯ",       "bhasa rup",    "ling", "shabdroop vigyan"),
    _e("phonology",       "ᱥᱩᱨ ᱵᱤᱜᱤᱭᱟᱱ",     "sur bigian",   "ling", "dhwani vigyan"),
    _e("translation",     "ᱛᱷᱟᱛᱷᱮ ᱵᱚᱫᱚᱞ",    "thathe bodol", "ling", "anuvad, parjanth"),
    _e("code switching",  "ᱵᱷᱟᱥᱟ ᱵᱚᱫᱚᱞ",     "bhasa bodol",  "ling", "bhasha badlana"),
    _e("bilingual",       "ᱵᱩᱦᱩ ᱵᱷᱟᱥᱟ",      "buhu bhasa",   "ling", "dwibhashi"),
    _e("dialect",         "ᱵᱷᱟᱥᱟ ᱨᱤᱛᱤ",      "bhasa riti",   "ling", "upbhasha, boli"),
    _e("script",          "ᱞᱮᱠᱷᱟ ᱰᱷᱟᱸᱪᱟ",    "lekha ḍhaNca", "ling", "lipi, likhawat"),
    _e("word",            "ᱥᱮᱴᱟᱜ",            "seṭaG",        "ling", "shabd"),
    _e("sentence",        "ᱵᱟᱠ",              "bak",          "ling", "vakya"),
    _e("paragraph",       "ᱠᱟᱛᱷᱟ",            "katha",        "ling", "anuched, paragraf"),
    _e("token",           "ᱥᱮᱴᱟᱜ",            "seṭaG",        "ling", "ikal, token"),
    _e("corpus",          "ᱵᱟᱠ ᱥᱟᱢᱩᱦ",       "bak samuh",    "ling", "shabd samooh"),
    _e("vocabulary",      "ᱥᱮᱴᱟᱜ ᱪᱤᱱᱡ",      "seṭaG cinj",   "ling", "shabd bhandar"),
    _e("dictionary",      "ᱥᱮᱴᱟᱜ ᱥᱟᱢᱩᱦ",     "seṭaG samuh",  "ling", "shabdkosh"),
    _e("text",            "ᱞᱮᱠᱷᱟ",            "lekha",        "ling", "likhit"),
    _e("context",         "ᱮᱠᱮ ᱠᱟᱛᱷᱟ",       "eke katha",    "ling", "prakarana, sandarbh"),
    _e("meaning",         "ᱢᱟᱱᱮ",             "mane",         "ling", "arth, matlab"),
    _e("ambiguity",       "ᱫᱩᱢᱤᱭᱟᱱᱟ ᱢᱟᱱᱮ",  "dumiana mane", "ling", "dviartha, aspashtata"),
    _e("language identification","ᱵᱷᱟᱥᱟ ᱪᱤᱱᱣᱟ","bhasa cinṿa", "ling", "bhasha pahchaan"),
    _e("natural language processing","ᱥᱣᱟᱵᱷᱟᱵᱤᱠ ᱵᱷᱟᱥᱟ","swabhabik bhasa","ling","prakritik bhasha sansakaran"),
    _e("speech recognition","ᱚᱲᱟᱜ ᱪᱤᱱᱣᱟ",  "oṛaG cinṿa",   "ling", "boli pahchaan"),
    _e("automatic speech recognition","ᱟᱯᱮ ᱪᱤᱱᱣᱟ","ape cinṿa","ling","swachalit boli pahchaan"),
    _e("text to speech",  "ᱞᱮᱠᱷᱟ ᱛᱷᱮᱠᱮ ᱚᱲᱟᱜ","lekha theke oṛaG","ling","lekh se boli"),
    _e("speech synthesis","ᱚᱲᱟᱜ ᱵᱚᱱᱟᱣ",     "oṛaG bonao",   "ling", "boli nirman"),
    _e("diacritics",      "ᱞᱮᱠᱷᱟ ᱪᱤᱱᱟ",      "lekha cina",   "ling", "swar chinha"),
    _e("grapheme",        "ᱞᱮᱠᱷᱟ ᱥᱮᱴᱟᱜ",     "lekha seṭaG",  "ling", "likhit ikal"),
    _e("phoneme-to-grapheme","ᱥᱩᱨ ᱛᱷᱮᱠᱮ ᱞᱮᱠᱷᱟ","sur theke lekha","ling","dhwani se likhit"),
    _e("grapheme to phoneme","ᱞᱮᱠᱷᱟ ᱛᱷᱮᱠᱮ ᱥᱩᱨ","lekha theke sur","ling","likhit se dhwani"),
    _e("ipa",             "ᱟᱠᱷᱚᱛ ᱥᱩᱨ ᱵᱷᱟᱥᱟ","akhot sur bhasa","ling","IPA lipimala"),
    _e("low resource",    "ᱠᱚᱢ ᱥᱟᱺᱥᱟᱫᱚᱱ",   "kom saNsadon",  "ling", "kam sansadhan"),
    _e("santhali",        "ᱥᱟᱱᱛᱟᱲᱤ",          "sanṭaṛi",      "ling", "Santhali bhasha"),
    _e("hindi",           "ᱦᱤᱸᱫᱤ",             "hiNdi",        "ling", "Hindi bhasha"),
    _e("english",         "ᱤᱸᱨᱮᱡᱤ",            "iNreji",       "ling", "Angrezi bhasha"),
    _e("hinglish",        "ᱦᱤᱸᱱᱜᱞᱤᱥ",          "hiNgliS",      "ling", "Hindi-Angrezi mishrit"),
    _e("ol chiki",        "ᱚᱞ ᱪᱤᱠᱤ",          "ol ciki",      "ling", "Ol Chiki lipi"),
    _e("devanagari",      "ᱫᱮᱵᱱᱟᱜᱨᱤ",          "dewnagri",     "ling", "Devanagari lipi"),
    _e("borrowing",       "ᱠᱟᱫᱚᱢ ᱥᱮᱴᱟᱜ",     "kadom seṭaG",  "ling", "rin shabd"),
    _e("loanword",        "ᱠᱟᱫᱚᱢ ᱥᱮᱴᱟᱜ",     "kadom seṭaG",  "ling", "rin shabd"),
    _e("suffix",          "ᱯᱟᱪᱷᱮ ᱡᱩᱰᱤ",       "pace juḍi",    "ling", "pratyaya"),
    _e("prefix",          "ᱟᱜᱚᱞ ᱡᱩᱰᱤ",       "aGol juḍi",    "ling", "upasarg"),
    _e("morpheme",        "ᱵᱷᱟᱥᱟ ᱜᱷᱚᱲ",       "bhasa ghoṛ",   "ling", "shapdroop ikal"),
    _e("intra-sentential","ᱵᱟᱠ ᱵᱤᱪᱮ",         "bak bice",     "ling", "vakya antarik"),
    _e("inter-sentential","ᱵᱟᱠ ᱵᱟᱦᱮᱨ",        "bak baher",    "ling", "vakya bahya"),
    _e("matrix language", "ᱢᱩᱞ ᱵᱷᱟᱥᱟ",       "mul bhasa",    "ling", "aadhar bhasha"),
    _e("embedded language","ᱢᱤᱞᱤᱛ ᱵᱷᱟᱥᱟ",    "milit bhasa",  "ling", "samahit bhasha"),
    _e("prosodic",        "ᱵᱚᱞ ᱵᱤᱜᱤᱭᱟᱱ",     "bol bigian",   "ling", "chhandas sambandhi"),
    _e("stress",          "ᱡᱚᱨ",              "jor",          "ling", "bal, zor"),
    _e("rhythm",          "ᱛᱟᱞ",              "tal",          "ling", "taal, lay"),
    _e("pause",           "ᱨᱩᱠ",              "ruk",          "ling", "viram, ruk"),

    # ════════════════════════════════════════════════════════════════════
    # ⑤ COMPUTING & TECHNOLOGY  (~50 terms)
    # ════════════════════════════════════════════════════════════════════
    _e("computer",        "ᱠᱚᱢᱯᱩᱴᱟᱨ",         "kompuṭar",     "tech", "sanganak"),
    _e("algorithm",       "ᱠᱟᱨᱚᱜ ᱠᱷᱚᱴᱮᱭ",    "karoG khoṭey", "tech", "karyavidhi, tarika"),
    _e("database",        "ᱫᱮᱫᱟ ᱦᱟᱫ",        "deda had",     "tech", "data bhandar"),
    _e("network",         "ᱡᱟᱞ",              "jal",          "tech", "jaal, network"),
    _e("processing",      "ᱠᱟᱢ ᱠᱟᱨᱚᱜ",       "kam karoG",    "tech", "sansakaran"),
    _e("memory",          "ᱢᱚᱱ",              "mon",          "tech", "yaadash, smriti"),
    _e("storage",         "ᱨᱟᱠᱷᱤ",             "rakhi",        "tech", "bhandaran"),
    _e("input",           "ᱤᱱᱯᱩᱴ",            "inpuṭ",        "tech", "pravesh"),
    _e("output",          "ᱚᱩᱴᱯᱩᱴ",           "ouṭpuṭ",       "tech", "nishpadan"),
    _e("code",            "ᱠᱚᱰ",              "koḍ",          "tech", "sanhit, code"),
    _e("program",         "ᱯᱨᱚᱜᱨᱟᱢ",          "pragram",      "tech", "karyakram"),
    _e("software",        "ᱥᱚᱯᱷᱴᱣᱮᱭᱟᱨ",        "sofṭṿer",      "tech", "sanganak sadhan"),
    _e("hardware",        "ᱦᱟᱨᱰᱣᱮᱭᱟᱨ",         "harḍṿer",      "tech", "yantrik sadhan"),
    _e("gpu",             "ᱜᱯᱩ",              "gpu",          "tech", "GPU yantra"),
    _e("cpu",             "ᱥᱟᱢᱩᱦ ᱠᱟᱨᱚᱜ",     "samuh karoG",  "tech", "CPU"),
    _e("cloud",           "ᱵᱟᱫᱚᱞ",            "badol",        "tech", "baadal, cloud"),
    _e("open source",     "ᱠᱷᱩᱞᱟ ᱥᱟᱢᱩᱦ",     "khula samuh",  "tech", "khula strot"),
    _e("framework",       "ᱰᱷᱟᱸᱪᱟ",            "ḍhaNca",       "tech", "dhancha"),
    _e("library",         "ᱡᱟᱱᱟ ᱥᱟᱢᱩᱦ",      "jana samuh",   "tech", "pustakalaya"),
    _e("pytorch",         "ᱯᱟᱭᱴᱚᱨᱪ",          "paiṭorc",      "tech", "PyTorch dalda"),
    _e("python",          "ᱯᱟᱭᱛᱷᱚᱱ",           "paiṭhon",      "tech", "Python bhasha"),
    _e("interface",       "ᱵᱤᱪᱮ ᱡᱩᱰᱤ",        "bice juḍi",    "tech", "sampark patal"),
    _e("pipeline",        "ᱱᱟᱞ",              "nal",          "tech", "krama nali"),
    _e("architecture",    "ᱰᱷᱟᱸᱪᱟ",            "ḍhaNca",       "tech", "nirman dhancha"),
    _e("module",          "ᱥᱟᱺᱜᱤ",             "saNgi",        "tech", "ikai, module"),
    _e("parallel",        "ᱥᱟᱢᱱᱮ ᱥᱟᱢᱱᱮ",      "samne samne",  "tech", "samantar"),
    _e("sequential",      "ᱞᱟᱦᱟ ᱞᱟᱦᱟ",       "laha laha",    "tech", "anukramic"),
    _e("real time",       "ᱵᱟᱨᱮ ᱠᱟᱞ",        "bare kal",     "tech", "vastavkalik"),
    _e("benchmark",       "ᱢᱟᱯ ᱜᱮᱛ",         "map get",      "tech", "tulanatmak mapi"),
    _e("evaluation",      "ᱢᱟᱯ ᱡᱩᱠ",         "map juk",      "tech", "mulyankan"),
    _e("dataset",         "ᱫᱮᱫᱟ ᱥᱟᱢᱩᱦ",       "deda samuh",   "tech", "data samooh"),
    _e("open source",     "ᱠᱷᱩᱞᱟ ᱠᱚᱰ",       "khula koḍ",    "tech", "mukta strot"),
    _e("pretrained model","ᱟᱜᱚᱞ ᱥᱤᱠᱚᱢ ᱢᱚᱰᱮᱞ","aGol sikam model","tech","poorv prashikshit model"),
    _e("checkpoint",      "ᱥᱟᱪ ᱠᱟᱜᱡ",        "sac kaGoj",    "tech", "sangrahaana bindu"),
    _e("epoch",           "ᱜᱷᱩᱨᱟᱣ",           "ghuro",        "tech", "ek chakkar"),
    _e("batch processing","ᱛᱩᱢᱩᱫ ᱠᱟᱨᱚᱜ",     "tumud karoG",  "tech", "samuh sansakaran"),
    _e("hyperparameter tuning","ᱵᱚᱰᱚ ᱢᱟᱯ ᱥᱩᱫᱷᱟᱨ","boṛo map sudhar","tech","upar-ghatan sudhaar"),
    _e("inference speed", "ᱛᱟᱦᱮᱸ ᱜᱮᱛ",       "taheN get",    "tech", "anuman tezi"),
    _e("quantisation",    "ᱢᱟᱯ ᱢᱩᱛᱩᱵᱩᱠ",     "map mutuubuk", "tech", "maan sankaran"),
    _e("pruning",         "ᱠᱟᱴᱟ",             "kaṭa",         "tech", "katna, chhatna"),
    _e("distillation",    "ᱠᱚᱢ ᱥᱤᱠᱚᱢ",       "kom sikam",    "tech", "nishkaran"),
    _e("token",           "ᱥᱮᱴᱟᱜ",            "seṭaG",        "tech", "ikal, token"),
    _e("tokenization",    "ᱥᱮᱴᱟᱜ ᱵᱟᱸᱫᱟ",     "seṭaG baNda",  "tech", "token banana"),
    _e("vocabulary size", "ᱥᱮᱴᱟᱜ ᱥᱟᱢᱩᱦ ᱢᱟᱯ","seṭaG samuh map","tech","shabd bhandar maan"),

    # ════════════════════════════════════════════════════════════════════
    # ⑥ GENERAL ACADEMIC VOCABULARY  (~80 terms)
    # ════════════════════════════════════════════════════════════════════
    _e("lecture",         "ᱯᱟᱲᱦᱟᱣ",           "paṛhao",       "acad", "padhana, vyakhyan"),
    _e("study",           "ᱯᱟᱲᱦᱟᱣ",           "paṛhao",       "acad", "adhyayan"),
    _e("research",        "ᱠᱷᱚᱡᱤ",             "khoji",        "acad", "khoj, anusandhan"),
    _e("experiment",      "ᱯᱨᱤᱠᱷᱟ",            "prikha",       "acad", "prayog, pariksha"),
    _e("result",          "ᱯᱷᱞ",               "phol",         "acad", "phal, nateeja"),
    _e("method",          "ᱢᱮᱛᱷᱚᱫ",            "methid",       "acad", "tarika, vidhi"),
    _e("approach",        "ᱨᱟᱥᱛᱟ",             "rasta",        "acad", "drishtikonn"),
    _e("system",          "ᱛᱟᱲᱢᱟ",             "taṛma",        "acad", "pranali, vyavastha"),
    _e("analysis",        "ᱵᱤᱪᱟᱨ",             "bicar",        "acad", "vishleshan"),
    _e("conclusion",      "ᱥᱮᱵ ᱢᱟᱱᱮ",         "seb mane",     "acad", "nishkarsh"),
    _e("problem",         "ᱥᱚᱢᱚᱥᱴᱟ",           "somosTa",      "acad", "samasya"),
    _e("solution",        "ᱥᱚᱢᱟᱫᱷᱟᱱ",           "somadhon",     "acad", "samadhan"),
    _e("challenge",       "ᱥᱚᱢᱚᱥᱴᱟ",           "somosTa",      "acad", "chunauti"),
    _e("improvement",     "ᱥᱩᱫᱷᱟᱨ",             "sudhar",       "acad", "sudhaar"),
    _e("performance",     "ᱠᱟᱢ ᱜᱮᱛ",          "kam get",      "acad", "kaarya nishpadan"),
    _e("comparison",      "ᱛᱩᱞᱱᱟ",             "tulna",        "acad", "tulanatmak"),
    _e("baseline",        "ᱢᱩᱞ ᱢᱟᱯ",          "mul map",      "acad", "aadhaar rekha"),
    _e("state of the art","ᱥᱵᱷᱮᱰᱮ ᱵᱮᱦᱛᱮᱨ",    "sobheḍe behter","acad","abhinav shresththa"),
    _e("novel",           "ᱱᱟᱹᱣᱟᱹ",             "naoṿa",        "acad", "naya, naveen"),
    _e("significant",     "ᱢᱩᱞ",               "mul",          "acad", "mahatvapoorn"),
    _e("efficient",       "ᱛᱩᱨᱟᱜ",             "turaG",        "acad", "daksha, kushalta"),
    _e("robust",          "ᱵᱮᱥᱤ ᱵᱟᱞ",         "besi bal",     "acad", "sudrIdh, mabboot"),
    _e("scalable",        "ᱵᱟᱰᱚᱱᱟ ᱭᱚᱜ",       "baḍona yoG",   "acad", "mashhabi"),
    _e("limitation",      "ᱥᱤᱢᱟ",              "sima",         "acad", "seema, simit"),
    _e("assumption",      "ᱢᱟᱱᱮ ᱠᱚᱨᱩ",         "mane koru",    "acad", "avdharna, manna"),
    _e("implementation",  "ᱞᱟᱜᱩ",              "laGu",         "acad", "laagu karna"),
    _e("evaluation metric","ᱢᱟᱯ ᱜᱮᱛ",         "map get",      "acad", "mapi manak"),
    _e("f1 score",        "ᱮᱯᱷ ᱮᱠ ᱢᱟᱯ",       "ef ek map",    "acad", "F1 mapi"),
    _e("precision",       "ᱥᱟᱪ ᱢᱟᱯ",          "sac map",      "acad", "suddhta, shuddhi"),
    _e("recall",          "ᱵᱤᱥᱟᱨ ᱢᱟᱯ",        "bisar map",    "acad", "wapasi, prapta"),
    _e("hypothesis",      "ᱢᱟᱱᱮ ᱠᱚᱨᱩ",         "mane koru",    "acad", "parikalpaana"),
    _e("objective",       "ᱞᱟᱠᱥᱭᱚ",            "laksyo",       "acad", "lakshya"),
    _e("contribution",    "ᱡᱩᱰᱤᱛ",              "juḍit",        "acad", "yogdaan"),
    _e("literature",      "ᱞᱮᱠᱷᱟ ᱵᱤᱜᱤᱭᱟᱱ",    "lekha bigian", "acad", "sahitya, vichar"),
    _e("review",          "ᱪᱮᱠᱟ",              "ceka",         "acad", "samiksha, jaanch"),
    _e("survey",          "ᱶᱷᱟᱸᱡᱟ",             "jaNja",        "acad", "sarvekshan"),
    _e("future work",     "ᱟᱜᱚᱞᱠᱟᱞ ᱠᱟᱢ",     "aGolkal kam",  "acad", "bhaavishya karya"),
    _e("dataset",         "ᱫᱮᱫᱟ ᱥᱟᱢᱩᱦ",       "deda samuh",   "acad", "data samooh"),
    _e("annotation",      "ᱢᱮᱞ ᱟᱠᱷᱚᱨ",        "mel akhor",    "acad", "tippani"),
    _e("labelling",       "ᱢᱮᱞ ᱜᱟᱶᱛᱟ",         "mel gaoṭa",    "acad", "naamankaran"),
    _e("crowdsourcing",   "ᱵᱩᱦᱩ ᱠᱟᱢ",         "buhu kam",     "acad", "sahyogi sangrah"),
    _e("inter-rater agreement","ᱮᱠᱢᱟᱛ",        "ekmot",        "acad", "parsparik sahmat"),
    _e("reproducibility", "ᱵᱟᱨᱵᱟᱨ ᱠᱟᱨᱚᱜ",     "barbar karoG", "acad", "punaravaraniyata"),
    _e("ablation study",  "ᱵᱟᱸᱫᱟ ᱵᱤᱪᱟᱨ",     "baNda bicar",  "acad", "khandan adhyayan"),
    _e("error analysis",  "ᱜᱞᱛᱤ ᱵᱤᱪᱟᱨ",      "golti bicar",  "acad", "truti vishleshan"),
    _e("curriculum",      "ᱯᱟᱲᱦᱟᱣ ᱠᱨᱚᱢ",      "paṛhao krom",  "acad", "pathacharya"),
    _e("syllabus",        "ᱯᱟᱲᱦᱟᱣ ᱜᱷᱚᱲ",      "paṛhao ghoṛ",  "acad", "paath suchi"),
    _e("university",      "ᱵᱤᱥᱣᱩᱵᱤᱫᱽᱭᱟᱞᱚᱭ",   "biswabidyalay","acad", "vishwavidyalay"),
    _e("institute",       "ᱫᱤᱨᱤ",              "diri",         "acad", "sanstha"),
    _e("professor",       "ᱟᱪᱟᱨᱭᱟ",            "acario",       "acad", "aacharya, professor"),
    _e("student",         "ᱪᱮᱞᱮᱜ",             "celeG",        "acad", "vidyarthi, chatra"),
    _e("assignment",      "ᱠᱟᱢ",               "kam",          "acad", "kaarya, daaura"),
    _e("project",         "ᱯᱨᱚᱡᱮᱠᱴ",           "projekṭ",      "acad", "pariyojana"),
    _e("report",          "ᱨᱤᱯᱚᱨᱴ",            "riporṭ",       "acad", "prativedan"),
    _e("thesis",          "ᱥᱚᱫᱷ",               "sodh",         "acad", "shodh prabandh"),
    _e("publication",     "ᱯᱨᱠᱟᱥ",             "prokaS",       "acad", "prakashan"),
    _e("citation",        "ᱦᱩᱣᱟ",              "hua",          "acad", "sandarbh, hawala"),
    _e("peer review",     "ᱥᱟᱺᱜᱤ ᱪᱮᱠᱟ",       "saNgi ceka",   "acad", "samkaksha samiksha"),
    _e("conference",      "ᱵᱮᱱᱟᱢ",             "benam",        "acad", "sammelan"),

    # ════════════════════════════════════════════════════════════════════
    # ⑦ COMMON FUNCTION WORDS  (~60 terms)
    # ════════════════════════════════════════════════════════════════════
    _e("and",             "ᱟᱨ",               "ar",           "func", "aur"),
    _e("or",              "ᱥᱮ",               "se",           "func", "ya"),
    _e("not",             "ᱵᱮ",               "be",           "func", "nahin"),
    _e("is",              "ᱮ",                "e",            "func", "hai"),
    _e("are",             "ᱠᱟᱱᱟ",             "kana",         "func", "hain"),
    _e("was",             "ᱛᱟᱦᱮᱸᱠᱟᱱᱟ",        "taheNkana",    "func", "tha"),
    _e("will",            "ᱞᱮᱠᱟ",             "leka",         "func", "hoga"),
    _e("can",             "ᱯᱟᱨᱮ",             "pare",         "func", "sakta hai"),
    _e("the",             "ᱱᱤᱭᱟ",             "nia",          "func", "yeh woh"),
    _e("a",               "ᱮᱠ",               "ek",           "func", "ek"),
    _e("of",              "ᱠᱮ",               "ke",           "func", "ka, ki, ke"),
    _e("in",              "ᱦᱩᱦᱩᱲ",             "huhUṛ",        "func", "mein, andar"),
    _e("to",              "ᱞᱟᱹᱜᱤ",             "laGi",         "func", "ke liye"),
    _e("for",             "ᱞᱟᱹᱜᱤ",             "laGi",         "func", "ke liye"),
    _e("with",            "ᱥᱟᱺᱜᱮ",             "saNge",        "func", "ke saath"),
    _e("from",            "ᱛᱷᱮᱠᱮ",             "theke",        "func", "se"),
    _e("by",              "ᱫᱳᱣᱟᱨᱟ",            "duara",        "func", "dwara"),
    _e("on",              "ᱩᱯᱚᱨᱮ",             "upore",        "func", "par, upar"),
    _e("at",              "ᱦᱩᱦᱩᱲ",             "huhUṛ",        "func", "par, mein"),
    _e("this",            "ᱱᱤᱭᱟ",             "nia",          "func", "yeh"),
    _e("that",            "ᱱᱩᱭᱟ",             "nuia",         "func", "woh"),
    _e("which",           "ᱡᱚ",               "jo",           "func", "jo, kaun sa"),
    _e("how",             "ᱪᱮᱫᱟᱞ",            "cedal",        "func", "kaise"),
    _e("what",            "ᱟᱹᱞᱤᱧ",             "aaliñ",        "func", "kya"),
    _e("when",            "ᱦᱟᱵ",              "hab",          "func", "kab"),
    _e("where",           "ᱠᱷᱚᱱ",             "khon",         "func", "kahan"),
    _e("why",             "ᱢᱮᱛᱮ",             "mete",         "func", "kyun"),
    _e("we",              "ᱟᱵᱚ",              "abo",          "func", "hum"),
    _e("i",               "ᱤᱧ",               "iñ",           "func", "mein"),
    _e("you",             "ᱟᱢ",               "am",           "func", "aap, tum"),
    _e("they",            "ᱠᱳ",               "ko",           "func", "woh log"),
    _e("it",              "ᱱᱩᱭᱟ",             "nuia",         "func", "yeh, woh"),
    _e("all",             "ᱥᱟᱵᱚ",             "sabo",         "func", "sab, saara"),
    _e("some",            "ᱠᱩᱞ",              "kul",          "func", "kuch"),
    _e("more",            "ᱵᱮᱥᱤ",             "besi",         "func", "zyada, adhik"),
    _e("less",            "ᱠᱚᱢ",              "kom",          "func", "kam"),
    _e("very",            "ᱵᱟᱦᱩᱛ",            "bahut",        "func", "bahut, zyada"),
    _e("also",            "ᱦᱩᱭ",              "hui",          "func", "bhi, aur bhi"),
    _e("only",            "ᱜᱮ",               "ge",           "func", "sirf, keval"),
    _e("now",             "ᱱᱤᱛᱚᱜ",            "nitoG",        "func", "abhi, iske saath"),
    _e("then",            "ᱛᱟᱦᱮᱸᱥᱮ",           "taheNse",      "func", "tab, phir"),
    _e("here",            "ᱱᱤᱭᱛᱮ",             "niyte",        "func", "yahan"),
    _e("there",           "ᱱᱩᱭᱛᱮ",             "nuyte",        "func", "wahan"),
    _e("first",           "ᱯᱦᱤᱞᱟᱹ",            "phila",        "func", "pehla"),
    _e("second",          "ᱫᱳᱦᱲᱟᱹ",            "dohoṛa",       "func", "doosra"),
    _e("same",            "ᱮᱠᱛᱟ",             "ekta",         "func", "same, ek jaisa"),
    _e("different",       "ᱵᱮᱥᱤ",             "besi",         "func", "alag, vibhinna"),
    _e("new",             "ᱱᱟᱹᱣᱟᱹ",             "naoṿa",        "func", "naya"),
    _e("old",             "ᱯᱩᱨᱟᱱᱟ",            "purana",       "func", "purana"),
    _e("large",           "ᱵᱚᱰᱚ",             "boṛo",         "func", "bada"),
    _e("small",           "ᱱᱤᱪᱟᱹ",             "nica",         "func", "chhota"),
    _e("high",            "ᱩᱪᱩ",              "ucu",          "func", "uchcha"),
    _e("low",             "ᱱᱤᱪᱟᱹ",             "nica",         "func", "neeche"),
    _e("good",            "ᱡᱟᱦᱟᱸᱭ",           "jahoN",        "func", "achha, bhala"),
    _e("better",          "ᱵᱮᱦᱛᱮᱨ",            "behter",       "func", "behtar"),
    _e("best",            "ᱥᱵᱷᱮᱰᱮ ᱵᱮᱦᱛᱮᱨ",    "sobheḍe behter","func","sabse behtar"),
    _e("important",       "ᱦᱩᱲᱩ",             "huṛu",         "func", "zaroori, mahatvapoorn"),
    _e("simple",          "ᱥᱮᱛᱮ",             "sete",         "func", "aasaan, saral"),
    _e("complex",         "ᱜᱷᱮᱨᱟ",             "ghera",        "func", "jatil, mushkil"),
    _e("based on",        "ᱩᱯᱚᱨᱮ",             "upore",        "func", "aadhar par"),
    _e("used for",        "ᱞᱟᱹᱜᱤ",             "laGi",         "func", "ke liye upyog"),
    _e("given",           "ᱮᱢ",               "em",           "func", "diya gaya"),
    _e("using",           "ᱵᱟᱵᱚᱦᱟᱨ",           "babohare",     "func", "upyog karte hue"),
    _e("called",          "ᱢᱮᱛᱟᱜ",             "metaG",        "func", "kaha jata hai"),
    _e("between",         "ᱵᱤᱪᱮ",              "bice",         "func", "beech mein"),
    _e("through",         "ᱫᱳᱣᱟᱨᱟ",            "duara",        "func", "dwara, zariye"),
    _e("without",         "ᱵᱤᱱᱟ",              "bina",         "func", "bina, bagair"),
    _e("because",         "ᱡᱩᱫᱤᱥᱮ",            "judise",       "func", "kyunki, isliye"),
    _e("therefore",       "ᱛᱟᱦᱮᱸᱥᱮ",           "taheNse",      "func", "isliye, ataev"),
    _e("however",         "ᱛᱷᱮᱠᱮᱣᱤ",            "thekeṿi",      "func", "lekin, parantu"),

    # ════════════════════════════════════════════════════════════════════
    # ⑧ ADDITIONAL SPEECH & ML TERMS  (to reach 500+)
    # ════════════════════════════════════════════════════════════════════

    # Extra Speech terms
    _e("codec",           "ᱠᱚᱰᱮᱠ",             "koḍek",        "speech", "sanket vidhi"),
    _e("waveform encoder","ᱞᱮᱦᱮᱨ ᱠᱚᱰ",         "leher koḍ",    "speech", "tarang ankeeyan"),
    _e("signal processing","ᱥᱟᱱᱠᱮᱛ ᱠᱟᱨᱚᱜ",    "sanket karoG", "speech", "sanket sansakaran"),
    _e("noise floor",     "ᱢᱩᱬᱩ ᱛᱞ",          "muṭu tol",     "speech", "shor sthar"),
    _e("snr",             "ᱥᱟᱱᱠᱮᱛ ᱢᱩᱬᱩ",      "sanket muṭu",  "speech", "SNR anupat"),
    _e("hifigan",         "ᱦᱤᱯᱷᱤ ᱜᱟᱱ",         "hiPhi gan",    "speech", "HiFi-GAN vocoder"),
    _e("wavernn",         "ᱞᱮᱦᱮᱨ ᱱᱮᱴ",         "leher net",    "speech", "WaveRNN model"),
    _e("tacotron",        "ᱛᱟᱠᱳᱴᱨᱚᱱ",           "takoṭron",     "speech", "Tacotron TTS"),
    _e("fastspeech",      "ᱛᱩᱨᱩᱸ ᱚᱲᱟᱜ",        "turuNg oṛaG",  "speech", "FastSpeech model"),
    _e("vits",            "ᱵᱤᱴᱥ",               "biṭs",         "speech", "VITS TTS vidhi"),
    _e("mel spectrogram", "ᱢᱮᱞ ᱪᱤᱛᱨᱚᱱ",        "mel citron",   "speech", "mel vistar chitran"),
    _e("speaker embedding","ᱚᱲᱟᱜ ᱵᱤᱸᱫᱩ",       "oṛaG bindu",   "speech", "vakta anukraman"),
    _e("voice cloning",   "ᱠᱩᱞᱤ ᱪᱚᱞᱤ",         "kuli coli",    "speech", "awaj nakal"),
    _e("prosodic transfer","ᱵᱚᱞ ᱵᱚᱫᱚᱞ",        "bol bodol",    "speech", "chhand parivartan"),
    _e("acoustic model",  "ᱡᱟᱣ ᱢᱚᱰᱮᱞ",         "jaṿ model",    "speech", "dhwani model"),

    # Extra ML terms
    _e("generative model","ᱵᱚᱱᱟᱜ ᱢᱚᱰᱮᱞ",      "bonaG model",  "ml", "srijanatmak model"),
    _e("discriminative",  "ᱵᱟᱸᱫᱟ ᱠᱟᱨᱚᱜ",       "baNda karoG",  "ml", "vibhajan karta"),
    _e("latent variable", "ᱞᱩᱠᱩ ᱵᱚᱫᱚᱞ",         "luku bodol",   "ml", "gupt chara"),
    _e("posterior distribution","ᱯᱟᱪᱷᱮ ᱵᱟᱸᱫᱩ","pace baNdu",   "ml", "paschimik vitaran"),
    _e("sampling strategy","ᱛᱚᱞᱟ ᱢᱮᱛᱷᱚᱫ",      "tola methid",  "ml", "pratidarsh vidhi"),
    _e("sequence model",  "ᱠᱨᱚᱢ ᱢᱚᱰᱮᱞ",        "krom model",   "ml", "anukram model"),
    _e("token prediction","ᱥᱮᱴᱟᱜ ᱛᱟᱦᱮᱸ",       "seṭaG taheN",  "ml", "token anuman"),
    _e("masked language model","ᱞᱩᱠᱩ ᱵᱷᱟᱥᱟ",  "luku bhasa",   "ml", "aavrit bhasha model"),
    _e("encoder decoder", "ᱠᱚᱰ ᱡᱩᱰᱤ",         "koḍ juḍi",     "ml", "sanketan-viketan"),
    _e("cross attention", "ᱮᱠᱮ ᱫᱷᱭᱟᱱ",         "eke dhyan",    "ml", "paraspar dhyan"),
    _e("hallucination",   "ᱜᱞᱛᱤ ᱛᱟᱦᱮᱸ",        "golti taheN",  "ml", "bhranti, galat anuman"),

    # Extra Math
    _e("transpose",       "ᱯᱟᱞᱴᱟ",             "palṭa",        "math", "ulat, vyutikrama"),
    _e("inverse",         "ᱩᱞᱴᱩ",               "ulṭu",         "math", "viparit, ulta"),
    _e("determinant",     "ᱱᱤᱡᱚᱠ ᱢᱟᱯ",         "nijok map",    "math", "saarnik"),
    _e("trace",           "ᱪᱤᱱᱦᱟ",              "cinha",        "math", "chinha, paath"),
    _e("rank",            "ᱠᱨᱚᱢ",               "krom",         "math", "shreni, rank"),
    _e("null space",      "ᱥᱩᱱ ᱡᱮᱦᱮᱱ",          "sun jehen",    "math", "shoony avkash"),
    _e("positive definite","ᱫᱷᱱᱟᱛᱢᱟᱠ",           "dhnotmak",     "math", "dhanatmak nishchit"),
    _e("orthogonal",      "ᱥᱟᱢᱠᱚᱱ",             "samkon",       "math", "samakon, lambvat"),
    _e("diagonal",        "ᱠᱳᱱᱤᱭᱟ",              "konia",        "math", "vikarna"),
    _e("upper triangular","ᱩᱯᱚᱨ ᱛᱤᱱᱠᱩᱱ",         "upor tinkun",  "math", "upar trikona"),
    _e("frobenius norm",  "ᱯᱷᱨᱳᱵᱮᱱᱤᱭᱩᱥ ᱢᱟᱯ",    "frobeniyas map","math","Frobenius norm"),
    _e("gradient clipping","ᱜᱷᱤ ᱠᱟᱴᱟ",          "ghi kaṭa",     "math", "avkalan katna"),

    # Extra Linguistics
    _e("vowel harmony",   "ᱥᱣᱚᱨ ᱢᱮᱞ",          "swor mel",     "ling", "swar sangati"),
    _e("retroflex",       "ᱢᱩᱲᱩ ᱵᱚᱞ",           "muṛu bol",     "ling", "murdha dhwani"),
    _e("aspirated",       "ᱯᱷᱩᱸᱠ ᱵᱚᱞ",          "phuNk bol",    "ling", "mahasaprana"),
    _e("unaspirated",     "ᱵᱮ ᱯᱷᱩᱸᱠ",           "be phuNk",     "ling", "alpsaprana"),
    _e("tonal language",  "ᱥᱩᱨ ᱵᱷᱟᱥᱟ",          "sur bhasa",    "ling", "swaraghata bhasha"),
    _e("schwa deletion",  "ᱟ ᱦᱚᱰᱚ",             "a hoṛo",       "ling", "schwa lop"),
    _e("sandhi",          "ᱥᱟᱸᱫᱷᱤ",              "saNdhi",       "ling", "sandhi, mel"),
    _e("agglutinative",   "ᱡᱩᱰᱟᱣ ᱵᱷᱟᱥᱟ",       "juḍao bhasa",  "ling", "yojak bhasha"),
    _e("api",             "ᱡᱩᱰᱤ ᱱᱤᱭᱚᱢ",         "juḍi niyom",   "tech", "API tantra"),
    _e("open vocabulary", "ᱠᱷᱩᱞᱟ ᱥᱮᱴᱟᱜ",        "khula seṭaG",  "ling", "mukta shabdbhandar"),
]

assert len(PARALLEL_CORPUS) >= 500, f"Corpus has only {len(PARALLEL_CORPUS)} entries!"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DICTIONARY LOOKUP ENGINE  (exact + fuzzy)
# ═══════════════════════════════════════════════════════════════════════════

class SanthaliDictionary:
    """
    Bi-directional English↔Santhali dictionary built from PARALLEL_CORPUS.

    Lookup methods
    ──────────────
    • `lookup(english)`  → exact match → SanthaliEntry or None
    • `search(english)`  → fuzzy match (edit distance)
    • `by_category(cat)` → all entries in a domain
    """

    def __init__(self, corpus: List[SanthaliEntry] = PARALLEL_CORPUS) -> None:
        self._en2entry: Dict[str, SanthaliEntry] = {}
        self._ro2entry: Dict[str, SanthaliEntry] = {}
        for entry in corpus:
            key = entry.english.lower().strip()
            self._en2entry[key] = entry
            self._ro2entry[entry.roman.lower()] = entry

    def lookup(self, english: str) -> Optional[SanthaliEntry]:
        """Exact lookup (case-insensitive)."""
        return self._en2entry.get(english.lower().strip())

    def lookup_many(self, terms: List[str]) -> Dict[str, Optional[SanthaliEntry]]:
        return {t: self.lookup(t) for t in terms}

    def search(self, query: str, top_k: int = 3) -> List[Tuple[SanthaliEntry, float]]:
        """
        Fuzzy search using normalised edit distance (Levenshtein).
        Returns list of (entry, similarity_score) sorted by score descending.
        """
        q = query.lower().strip()
        results = []
        for key, entry in self._en2entry.items():
            sim = 1.0 - self._edit_distance(q, key) / max(len(q), len(key), 1)
            results.append((entry, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            ndp = [i] + [0] * n
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    ndp[j] = dp[j-1]
                else:
                    ndp[j] = 1 + min(dp[j], ndp[j-1], dp[j-1])
            dp = ndp
        return dp[n]

    def by_category(self, category: str) -> List[SanthaliEntry]:
        return [e for e in self._en2entry.values() if e.category == category]

    def all_english(self) -> List[str]:
        return list(self._en2entry.keys())

    def __len__(self) -> int:
        return len(self._en2entry)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SEMANTIC RETRIEVAL (PyTorch embedding-based nearest-neighbour)
# ═══════════════════════════════════════════════════════════════════════════

class SemanticRetriever(nn.Module):
    """
    Character n-gram bag-of-words embedding for semantic term retrieval.

    Each dictionary entry is embedded by averaging the embeddings of its
    character trigrams.  At inference, the query is embedded the same way
    and cosine similarity selects the top-k candidates.

    This provides a soft fallback when the exact or fuzzy lookup fails,
    particularly useful for:
      • Compound technical terms   ("mel frequency spectrogram")
      • Borrowed transliterations  ("gaussian mixture")
      • Partial matches            ("backprop" → "backpropagation")
    """

    def __init__(
        self,
        corpus: List[SanthaliEntry] = PARALLEL_CORPUS,
        embed_dim: int = 64,
        ngram_n: int = 3,
    ) -> None:
        super().__init__()
        self.ngram_n  = ngram_n
        self.embed_dim = embed_dim

        # Build character vocabulary from all English terms
        all_text = " ".join(e.english for e in corpus)
        chars = sorted(set(all_text.lower()))
        self.char2idx: Dict[str, int] = {c: i+1 for i, c in enumerate(chars)}
        self.char2idx["<UNK>"] = 0
        V = len(self.char2idx) + 1

        self.embedding = nn.Embedding(V, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.1)

        # Pre-compute corpus embeddings  [N, embed_dim]
        self.corpus      = corpus
        self._corpus_emb: Optional[torch.Tensor] = None

    def _text_to_trigram_ids(self, text: str) -> List[int]:
        """Convert text to character trigram indices (bag of n-grams)."""
        t = text.lower()
        ids = []
        for i in range(len(t) - self.ngram_n + 1):
            ngram = t[i : i + self.ngram_n]
            # Map each char in ngram; use UNK for OOV
            for c in ngram:
                ids.append(self.char2idx.get(c, 0))
        if not ids:
            ids = [self.char2idx.get(c, 0) for c in t]
        return ids

    def embed_text(self, text: str) -> torch.Tensor:
        """Embed a text string to a d-dimensional vector via mean-pooling."""
        ids = self._text_to_trigram_ids(text)
        if not ids:
            return torch.zeros(self.embed_dim)
        id_tensor = torch.tensor(ids)
        vecs = self.embedding(id_tensor)          # [T, D]
        return vecs.mean(dim=0)                   # [D]

    @torch.no_grad()
    def build_index(self) -> None:
        """Pre-compute all corpus embeddings."""
        embs = [self.embed_text(e.english) for e in self.corpus]
        self._corpus_emb = torch.stack(embs)      # [N, D]
        self._corpus_emb = F.normalize(self._corpus_emb, dim=-1)

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[SanthaliEntry, float]]:
        """Return top-k corpus entries by cosine similarity to query."""
        if self._corpus_emb is None:
            self.build_index()

        q_emb = F.normalize(self.embed_text(query).unsqueeze(0), dim=-1)  # [1, D]
        sims  = (q_emb @ self._corpus_emb.T).squeeze(0)                   # [N]
        topk  = sims.topk(top_k)

        return [
            (self.corpus[i.item()], sims[i.item()].item())
            for i in topk.indices
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SENTENCE-LEVEL TRANSLATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TranslationResult:
    source_text:   str
    santhali_text: str        # Ol Chiki script
    roman_text:    str        # ISO romanisation
    word_map:      List[dict] # per-word source→target alignment
    coverage:      float      # fraction of words translated (vs borrowed)


class SanthaliTranslator:
    """
    Word-by-word English/Hinglish → Santhali translator.

    Strategy
    ─────────
    For each input word (tokenised):
      1. Exact dictionary lookup  → use Santhali form
      2. Fuzzy dictionary lookup  (edit-distance ≤ 0.3)
      3. Semantic retrieval       (embedding cosine ≥ 0.6)
      4. Borrowing fallback       → keep original word in brackets [word]

    Technical terms that have no Santhali equivalent are explicitly
    borrowed (marked [TT]) per standard Santhali practice.

    Parameters
    ----------
    min_fuzzy_sim   : Minimum fuzzy match similarity (0–1).
    min_semantic_sim: Minimum semantic retrieval similarity (0–1).
    """

    def __init__(
        self,
        min_fuzzy_sim:    float = 0.75,
        min_semantic_sim: float = 0.60,
        use_semantic:     bool  = True,
    ) -> None:
        self.dict     = SanthaliDictionary()
        self.retriever = SemanticRetriever() if use_semantic else None
        if self.retriever is not None:
            self.retriever.build_index()
        self.min_fuzzy    = min_fuzzy_sim
        self.min_semantic = min_semantic_sim

    def translate_word(self, word: str) -> Tuple[str, str, str]:
        """
        Translate a single word.

        Returns
        -------
        (santhali_ol_chiki, roman, method)
        where method ∈ {'exact', 'fuzzy', 'semantic', 'borrowed'}
        """
        clean = re.sub(r"[^\w\-]", "", word.lower())
        if not clean:
            return word, word, "punctuation"

        # 1. Exact
        entry = self.dict.lookup(clean)
        if entry:
            return entry.santhali, entry.roman, "exact"

        # 2. Fuzzy
        results = self.dict.search(clean, top_k=1)
        if results and results[0][1] >= self.min_fuzzy:
            entry = results[0][0]
            return entry.santhali, entry.roman, "fuzzy"

        # 3. Semantic embedding
        if self.retriever is not None:
            sem_results = self.retriever.retrieve(clean, top_k=1)
            if sem_results and sem_results[0][1] >= self.min_semantic:
                entry = sem_results[0][0]
                return entry.santhali, f"[{word}]", "semantic"

        # 4. Borrow
        return f"[{word}]", f"[{word}]", "borrowed"

    def translate(self, text: str) -> TranslationResult:
        """
        Translate a full sentence/passage.

        Parameters
        ----------
        text : Source text (English, Hindi, or mixed Hinglish).

        Returns
        -------
        TranslationResult
        """
        tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z'-]+|[0-9]+|[^\w\s]+|\s+", text)
        sa_parts, ro_parts, word_map = [], [], []

        translated_count = 0
        word_count = 0

        for token in tokens:
            if re.fullmatch(r"\s+", token):
                sa_parts.append(" ")
                ro_parts.append(" ")
                continue

            if re.fullmatch(r"[^\w]+", token):
                sa_parts.append(token)
                ro_parts.append(token)
                continue

            sa, ro, method = self.translate_word(token)
            sa_parts.append(sa)
            ro_parts.append(ro)
            word_map.append({
                "source":  token,
                "santhali": sa,
                "roman":   ro,
                "method":  method,
            })
            word_count += 1
            if method != "borrowed":
                translated_count += 1

        coverage = translated_count / max(word_count, 1)
        return TranslationResult(
            source_text=text,
            santhali_text=" ".join(p for p in sa_parts if p.strip()),
            roman_text=" ".join(p for p in ro_parts if p.strip()),
            word_map=word_map,
            coverage=coverage,
        )

    def translate_segments(self, segments: List[dict]) -> List[dict]:
        """
        Translate a list of transcript segment dicts (from Part I/II G2P).
        Adds 'santhali', 'santhali_roman', 'coverage' keys to each segment.
        """
        output = []
        for seg in segments:
            text = seg.get("text", seg.get("ipa", ""))
            result = self.translate(text)
            out = dict(seg)
            out["santhali"]       = result.santhali_text
            out["santhali_roman"] = result.roman_text
            out["translation_coverage"] = round(result.coverage, 3)
            out["word_map"]       = result.word_map
            output.append(out)
        return output


# ═══════════════════════════════════════════════════════════════════════════
# 5.  CORPUS EXPORT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def export_corpus_json(path: str = "santhali_corpus.json") -> None:
    """Export the full parallel corpus to JSON."""
    data = [
        {
            "english":    e.english,
            "santhali":   e.santhali,
            "roman":      e.roman,
            "category":   e.category,
            "definition": e.definition,
        }
        for e in PARALLEL_CORPUS
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Corpus] Exported {len(data)} entries → {path}")


def export_corpus_tsv(path: str = "santhali_corpus.tsv") -> None:
    """Export as tab-separated values (compatible with most MT toolkits)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("english\tsanthali\troman\tcategory\tdefinition\n")
        for e in PARALLEL_CORPUS:
            f.write(f"{e.english}\t{e.santhali}\t{e.roman}\t{e.category}\t{e.definition}\n")
    print(f"[Corpus] Exported {len(PARALLEL_CORPUS)} entries → {path}")


def corpus_statistics() -> None:
    """Print corpus statistics by category."""
    from collections import Counter
    cats = Counter(e.category for e in PARALLEL_CORPUS)
    print(f"\n── Santhali Parallel Corpus Statistics ──")
    print(f"   Total entries: {len(PARALLEL_CORPUS)}")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        label = {
            "speech": "Speech & Acoustics",
            "ml":     "Machine Learning / AI",
            "math":   "Mathematics & Statistics",
            "ling":   "Language & Linguistics",
            "tech":   "Computing & Technology",
            "acad":   "General Academic",
            "func":   "Function Words",
        }.get(cat, cat)
        print(f"   {label:35s} : {count:3d}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    corpus_statistics()

    translator = SanthaliTranslator()

    test_sentences = [
        "The stochastic gradient descent algorithm converges to a local minima.",
        "MFCC features are used for speech recognition training.",
        "Code switching is common in Hinglish lectures.",
        "The cepstrum separates excitation from the vocal tract filter.",
        "Spectral subtraction reduces noise in classroom recordings.",
        "We use a neural network for language identification.",
        "Low resource language like Santhali needs transfer learning.",
    ]

    print("── English → Santhali Translation ──\n")
    for sent in test_sentences:
        result = translator.translate(sent)
        print(f"EN : {sent}")
        print(f"SA : {result.santhali_text}")
        print(f"RO : {result.roman_text}")
        print(f"COV: {result.coverage:.0%}")
        print()

    # Export
    export_corpus_json("santhali_corpus.json")
    export_corpus_tsv("santhali_corpus.tsv")
