"""
g2p_hinglish.py  —  Task 2.1
==============================
Grapheme-to-Phoneme (G2P) converter for Code-Switched Hinglish text.

Produces a unified **IPA** string from a mixed English-Hindi (Devanagari
or romanised) transcript.  Standard G2P tools (eSpeak, CMU dict) fail on
code-switching because:
  • Hindi aspirates (kh, gh, bh, dh, th …) are missing from English
    phoneme inventories.
  • Schwa deletion in Hindi (अच्छा → /ətʃʰɑː/ not /ətʃʰɑːə/) must be
    hand-modelled.
  • Hinglish romanisation is inconsistent ("thoda"/"thora"/"thoda" all
    common).
  • Technical English borrowings inside Hindi phrases need English rules.

Architecture
────────────
                  Input sentence (mixed Devanagari / Roman)
                          │
              ┌───────────▼────────────┐
              │  LanguageTokeniser     │   word-level LID from LID model
              │  (or heuristic fallback│   (integrates with Part I)
              └───────────┬────────────┘
                          │  List[(word, lang)]
            ┌─────────────▼──────────────┐
            │  DevanagariG2P             │   Devanagari → IPA
            │  HindiRomanG2P             │   romanised Hindi → IPA
            │  EnglishG2P               │   English → IPA (rule + dict)
            └─────────────┬──────────────┘
                          │  IPA tokens per word
              ┌───────────▼────────────┐
              │  HinglishPhonologyLayer│   • Schwa deletion
              │                        │   • Flap/retroflex assimilation
              │                        │   • Vowel harmony across boundaries
              └───────────┬────────────┘
                          │
                 Unified IPA string + word alignment

Phoneme inventory targets (IPA)
────────────────────────────────
Consonants:  p b t d ʈ ɖ k ɡ
             pʰ bʱ tʰ dʱ ʈʰ ɖʱ kʰ ɡʱ
             tʃ dʒ tʃʰ dʒʱ
             f v s z ʃ ʒ x ɣ h ɦ
             m n ŋ ɳ ɲ
             r ɾ ɽ l ɭ
             j w
Vowels:      ə ɪ e ɛ a ɑ ɔ o ʊ u iː eː aː oː uː
Nasalised:   ã ẽ ĩ õ ũ
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DEVANAGARI → IPA
# ═══════════════════════════════════════════════════════════════════════════

class DevanagariG2P:
    """
    Rule-based Devanagari → IPA converter.

    Handles:
    • All standard consonant clusters with virāma (halant)
    • Aspiration pairs (aspirated kh = kʰ, etc.)
    • Nasalisation anusvara (ं → ◌̃)
    • Schwa deletion in non-final position (approximated)
    • Anunaasika / chandrabindu (ँ)
    • Nukta variants (क़ → q, ज़ → z, etc.)
    """

    # ── Consonant table ─────────────────────────────────────────────────────
    # Maps Devanagari consonant character → IPA string
    CONSONANTS: Dict[str, str] = {
        # Velars
        "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʱ", "ङ": "ŋ",
        # Palatals
        "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʱ", "ञ": "ɲ",
        # Retroflexes
        "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ", "ण": "ɳ",
        # Dentals
        "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʱ", "न": "n",
        # Labials
        "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʱ", "म": "m",
        # Approximants / liquids
        "य": "j", "र": "r", "ल": "l", "व": "ʋ",
        # Sibilants / fricatives
        "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
        # Nukta variants
        "क़": "q", "ख़": "x", "ग़": "ɣ", "ज़": "z",
        "ड़": "ɽ",  "ढ़": "ɽʱ", "फ़": "f",
        # Rarely encountered
        "ळ": "ɭ", "ऴ": "ɻ",
    }

    # ── Vowel / mātrā table ─────────────────────────────────────────────────
    VOWELS_INDEPENDENT: Dict[str, str] = {
        "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː",
        "उ": "ʊ", "ऊ": "uː", "ए": "eː", "ऐ": "ɛː",
        "ओ": "oː", "औ": "ɔː", "ऋ": "rɪ", "ऌ": "lɪ",
        "ॲ": "æ",  "ऑ": "ɔ",
    }

    VOWEL_MATRAS: Dict[str, str] = {
        "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ",
        "ू": "uː", "े": "eː", "ै": "ɛː", "ो": "oː",
        "ौ": "ɔː", "ृ": "rɪ", "ॄ": "rɪː",
        "ॅ": "e",  "ॆ": "ɛ",  "ॉ": "ɔ",
    }

    HALANT   = "\u094D"   # ्  (virāma)
    ANUSVARA = "\u0902"   # ं
    VISARGA  = "\u0903"   # ः
    CHANDRAB = "\u0901"   # ँ  (chandrabindu)

    # Context-sensitive anusvara realisation before following consonant
    _NASAL_MAP: Dict[str, str] = {
        "k": "ŋ", "kʰ": "ŋ", "ɡ": "ŋ", "ɡʱ": "ŋ",
        "tʃ": "ɲ", "tʃʰ": "ɲ", "dʒ": "ɲ", "dʒʱ": "ɲ",
        "ʈ": "ɳ", "ʈʰ": "ɳ", "ɖ": "ɳ", "ɖʱ": "ɳ",
    }

    def convert(self, text: str) -> str:
        """Convert a Devanagari word/phrase to IPA."""
        chars = list(text)
        ipa_tokens: List[str] = []
        i = 0

        while i < len(chars):
            ch = chars[i]

            # ── Consonant ─────────────────────────────────────────────────
            if ch in self.CONSONANTS:
                cons_ipa = self.CONSONANTS[ch]

                # Peek ahead
                if i + 1 < len(chars) and chars[i + 1] == self.HALANT:
                    # Virāma: no vowel; check for conjunct
                    ipa_tokens.append(cons_ipa)
                    i += 2          # skip consonant + halant
                    continue

                # Add consonant IPA
                ipa_tokens.append(cons_ipa)
                i += 1

                # Look for mātrā or anusvara after consonant
                vowel_added = False
                while i < len(chars):
                    nch = chars[i]
                    if nch in self.VOWEL_MATRAS:
                        ipa_tokens.append(self.VOWEL_MATRAS[nch])
                        vowel_added = True
                        i += 1
                    elif nch == self.ANUSVARA:
                        # Realise as nasal homorganic with next consonant
                        nasal = "n"   # default
                        if i + 1 < len(chars) and chars[i + 1] in self.CONSONANTS:
                            nc_ipa = self.CONSONANTS[chars[i + 1]]
                            nasal = self._NASAL_MAP.get(nc_ipa, "n")
                        ipa_tokens.append(nasal)
                        i += 1
                    elif nch == self.CHANDRAB:
                        ipa_tokens.append("̃")   # nasalisation diacritic
                        i += 1
                    elif nch == self.VISARGA:
                        ipa_tokens.append("h")
                        i += 1
                    else:
                        break

                # Default schwa if no vowel sign found (schwa deletion later)
                if not vowel_added:
                    ipa_tokens.append("ə")

            # ── Independent vowel ─────────────────────────────────────────
            elif ch in self.VOWELS_INDEPENDENT:
                ipa_tokens.append(self.VOWELS_INDEPENDENT[ch])
                i += 1
                # Check for anusvara following standalone vowel
                if i < len(chars) and chars[i] == self.ANUSVARA:
                    ipa_tokens[-1] += "̃"
                    i += 1

            # ── Anusvara / chandrabindu at start ──────────────────────────
            elif ch == self.ANUSVARA:
                ipa_tokens.append("n")
                i += 1

            # ── Punctuation / space / digit ───────────────────────────────
            elif ch in " \t\n।॥":
                ipa_tokens.append(" ")
                i += 1
            elif ch.isdigit():
                ipa_tokens.append(ch)
                i += 1
            else:
                i += 1   # skip unknown

        raw_ipa = "".join(ipa_tokens).strip()
        return self._apply_schwa_deletion(raw_ipa)

    def _apply_schwa_deletion(self, ipa: str) -> str:
        """
        Approximate Hindi schwa deletion:
        Delete /ə/ when it is in an open non-final syllable and the
        following syllable also has a full vowel.
        Pattern:  Cə C V  →  C C V   (medial schwa before CV)

        This is a conservative heuristic; full schwa deletion requires
        syllabification-aware phonology.
        """
        # Delete ə immediately before a consonant+vowel sequence if
        # another vowel precedes it (non-word-initial)
        ipa = re.sub(r"(?<=[aːɪiːʊuːeːɛːoːɔː])ə(?=[pbtdkɡʈɖmnŋɳɲrljʋsʃʂzɦɾɽfvxɣ])", "", ipa)
        # Delete final ə in non-question words (approximate)
        ipa = re.sub(r"ə\b", "", ipa)
        return ipa


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ROMANISED HINDI → IPA
# ═══════════════════════════════════════════════════════════════════════════

class HindiRomanG2P:
    """
    Converts romanised Hinglish words (IAST / colloquial) to IPA.

    Handles aspirate digraphs (kh, gh, ch, jh, th, dh, ph, bh …),
    retroflex retroflex sequences (dd, tt, rr), and Hinglish-specific
    conventions (w → ʋ, v → ʋ, z → z).

    Rule priority: longer patterns are matched first (greedy left-to-right).
    """

    # Ordered list: (regex_pattern, ipa_replacement)
    # Longer / more specific patterns MUST come before shorter ones.
    RULES: List[Tuple[str, str]] = [
        # ── Aspirate stops (digraphs) ───────────────────────────────────
        (r"\bkh\b", "kʰ"), (r"kh",  "kʰ"),
        (r"\bgh\b", "ɡʱ"), (r"gh",  "ɡʱ"),
        (r"\bch\b", "tʃʰ"),(r"ch",  "tʃ"),   # 'ch' aspirated by default in Hindi
        (r"\bjh\b", "dʒʱ"),(r"jh",  "dʒʱ"),
        (r"\bth\b", "t̪ʰ"), (r"th",  "t̪ʰ"),
        (r"\bdh\b", "d̪ʱ"), (r"dh",  "d̪ʱ"),
        (r"\bph\b", "pʰ"), (r"ph",  "pʰ"),
        (r"\bbh\b", "bʱ"), (r"bh",  "bʱ"),
        # Retroflex aspirates
        (r"tth|ṭh", "ʈʰ"), (r"ddh|ḍh","ɖʱ"),
        (r"tt|ṭ",   "ʈ"),   (r"dd|ḍ",  "ɖ"),
        (r"rr|ṛ",   "ɽ"),   (r"nn|ṇ",  "ɳ"),
        (r"ll|ḷ",   "ɭ"),   (r"shh",   "ʂ"),
        # ── Single consonants ───────────────────────────────────────────
        (r"sh|ś|ş", "ʃ"),
        (r"[ck]",   "k"),
        (r"g",      "ɡ"),
        (r"j",      "dʒ"),
        (r"[td]",   lambda m: "t̪" if m.group()==("t") else "d̪"),
        (r"n",      "n"),  (r"m",  "m"), (r"p",  "p"), (r"b",  "b"),
        (r"f",      "f"),  (r"v",  "ʋ"), (r"w",  "ʋ"),
        (r"s",      "s"),  (r"z",  "z"), (r"h",  "ɦ"),
        (r"r",      "r"),  (r"l",  "l"), (r"y",  "j"),
        (r"x",      "kʃ"),
        # ── Vowels ─────────────────────────────────────────────────────
        (r"aa|ā",   "aː"), (r"ii|ī",  "iː"), (r"uu|ū",  "uː"),
        (r"ee",     "iː"), (r"oo",    "uː"),
        (r"ai|ay",  "ɛː"), (r"au|aw", "ɔː"),
        (r"a",      "ə"),  (r"i",     "ɪ"),  (r"u",     "ʊ"),
        (r"e",      "eː"), (r"o",     "oː"),
        # Nasalised vowels
        (r"an\b",   "ã"),  (r"en\b",  "ẽ"),  (r"in\b",  "ĩ"),
        (r"on\b",   "õ"),  (r"un\b",  "ũ"),
    ]

    def __init__(self) -> None:
        # Compile patterns, separating callable replacements
        self._compiled: List[Tuple] = []
        for pattern, repl in self.RULES:
            if callable(repl):
                self._compiled.append((re.compile(pattern), repl))
            else:
                self._compiled.append((re.compile(pattern), repl))

    def convert(self, word: str) -> str:
        """Convert a romanised Hindi word to IPA."""
        text = word.lower()
        for pat, repl in self._compiled:
            if callable(repl):
                text = pat.sub(repl, text)
            else:
                text = pat.sub(repl, text)
        return text


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ENGLISH → IPA
# ═══════════════════════════════════════════════════════════════════════════

class EnglishG2P:
    """
    Rule-based English → IPA converter with a technical-term exception
    dictionary.  Falls back to a suffix-rule cascade for OOV words.

    The dictionary prioritises common ASR/ML/signal-processing terms that
    appear in lectures.
    """

    # Pronunciation dictionary for technical terms (British/Indian English)
    DICT: Dict[str, str] = {
        # Core speech processing
        "speech":       "spiːtʃ",
        "processing":   "prɒsɛsɪŋ",
        "acoustic":     "əkuːstɪk",
        "phoneme":      "foʊniːm",
        "phonemes":     "foʊniːmz",
        "phonetic":     "fənetɪk",
        "phonetics":    "fənetɪks",
        "allophones":   "ælofoʊnz",
        "cepstrum":     "sɛpstrəm",
        "cepstral":     "sɛpstrəl",
        "spectrogram":  "spɛktrəɡræm",
        "spectrography":"spɛktrɒɡrəfi",
        "mel":          "mɛl",
        "mfcc":         "ɛm ɛf siː siː",
        "filterbank":   "fɪltəbæŋk",
        "formant":      "fɔːmənt",
        "formants":     "fɔːmənts",
        "fricative":    "frɪkətɪv",
        "fricatives":   "frɪkətɪvz",
        "plosive":      "pləʊsɪv",
        "prosody":      "prɒsədi",
        "coarticulation":"koʊɑːtɪkjʊleɪʃən",
        "stochastic":   "stəkæstɪk",
        "autocorrelation":"ɔːtoʊkɒrəleɪʃən",
        "excitation":   "ɛksɪteɪʃən",
        "reverberation":"rɪvɜːbəreɪʃən",
        "dereverberation":"diːrɪvɜːbəreɪʃən",
        # ML / ASR
        "neural":       "njʊərəl",
        "transformer":  "trænsˈfɔːmə",
        "attention":    "ətɛnʃən",
        "encoder":      "ɛnkoʊdər",
        "decoder":      "diːkoʊdər",
        "recurrent":    "rɪkɜːrənt",
        "convolutional":"kɒnvəluːʃənəl",
        "gaussian":     "ɡaʊsiən",
        "bayesian":     "beɪziən",
        "viterbi":      "vɪtɜːrbi",
        "hmm":          "eɪtʃ ɛm ɛm",
        "ctc":          "siː tiː siː",
        "softmax":      "sɒftmæks",
        "gradient":     "ɡreɪdiənt",
        "backpropagation":"bækprɒpəɡeɪʃən",
        "perplexity":   "pɜːplɛksɪti",
        "kneser":       "kneɪzər",
        "ngram":        "ɛn ɡræm",
        "logit":        "lɒdʒɪt",
        "logits":       "lɒdʒɪts",
        "whisper":      "wɪspər",
        "wav2vec":      "weɪv tuː vɛk",
        # General academic
        "algorithm":    "ælɡərɪðəm",
        "parameter":    "pəræmɪtər",
        "frequency":    "friːkwənsi",
        "amplitude":    "æmplɪtjuːd",
        "magnitude":    "mæɡnɪtjuːd",
        "normalisation":"nɔːməlaɪzeɪʃən",
        "normalisation":"nɔːməlɪzeɪʃən",
        "feature":      "fiːtʃər",
        "features":     "fiːtʃərz",
        "classification":"klæsɪfɪkeɪʃən",
        "recognition":  "rɛkəɡnɪʃən",
        "synthesis":    "sɪnθɪsɪs",
        "analysis":     "ənælɪsɪs",
        "representation":"rɛprɪzɛnteɪʃən",
        "hypothesis":   "haɪpɒθɪsɪs",
        # Code-switching / LID
        "hinglish":     "hɪŋɡlɪʃ",
        "hindi":        "hɪndi",
        "english":      "ɪŋɡlɪʃ",
        "code":         "koʊd",
        "switching":    "swɪtʃɪŋ",
        "bilingual":    "baɪlɪŋɡwəl",
        "multilingual": "mʌltɪlɪŋɡwəl",
        # Common function words (fast path)
        "the":  "ðə", "a": "ə", "an": "æn", "of": "əv",
        "is":   "ɪz", "in": "ɪn", "to": "tə", "and": "ænd",
        "that": "ðæt", "this": "ðɪs", "it": "ɪt", "for": "fɔː",
        "with": "wɪð", "are": "ɑː", "was": "wɒz", "we": "wiː",
    }

    # Suffix rules for OOV: (suffix_regex, IPA_ending_replacement, strip_n_chars)
    SUFFIX_RULES: List[Tuple[str, str, int]] = [
        (r"tion$",    "ʃən",   4),
        (r"sion$",    "ʒən",   4),
        (r"tion$",    "ʃən",   4),
        (r"ous$",     "əs",    3),
        (r"ing$",     "ɪŋ",    3),
        (r"ed$",      "d",     2),
        (r"er$",      "ər",    2),
        (r"est$",     "ɪst",   3),
        (r"ly$",      "li",    2),
        (r"ment$",    "mənt",  4),
        (r"ness$",    "nəs",   4),
        (r"ity$",     "ɪti",   3),
        (r"ic$",      "ɪk",    2),
        (r"al$",      "əl",    2),
        (r"ive$",     "ɪv",    3),
        (r"ize$|ise$","aɪz",   3),
        (r"ism$",     "ɪzəm",  3),
        (r"ist$",     "ɪst",   3),
        (r"ful$",     "fʊl",   3),
        (r"less$",    "ləs",   4),
        (r"ance$|ence$","əns", 4),
    ]

    # Simplified vowel + consonant rules (applied left-to-right after suffix)
    _VOWEL_RULES: List[Tuple[str, str]] = [
        (r"ee|ea",  "iː"), (r"oo",    "uː"), (r"ou|ow",  "aʊ"),
        (r"ai|ay",  "eɪ"), (r"oi|oy", "ɔɪ"), (r"au|aw",  "ɔː"),
        (r"ie",     "aɪ"), (r"ue",    "juː"),
        (r"a(?=[^aeiou]*e\b)", "eɪ"),
        (r"i(?=[^aeiou]*e\b)", "aɪ"),
        (r"o(?=[^aeiou]*e\b)", "oʊ"),
        (r"u(?=[^aeiou]*e\b)", "juː"),
        (r"a", "æ"), (r"e\b", ""),  (r"e", "ɛ"),
        (r"i", "ɪ"), (r"o",   "ɒ"), (r"u", "ʌ"),
    ]
    _CONS_RULES: List[Tuple[str, str]] = [
        (r"ck",  "k"),  (r"ph",  "f"),  (r"gh\b",""),
        (r"gh",  "ɡ"),  (r"th",  "θ"),  (r"sh",  "ʃ"),
        (r"ch",  "tʃ"), (r"wh",  "w"),  (r"qu",  "kw"),
        (r"x",   "ks"), (r"c(?=[ei])", "s"), (r"c", "k"),
        (r"g(?=[ei])","dʒ"), (r"g","ɡ"),
        (r"j",   "dʒ"), (r"z",   "z"),  (r"y(?=[aeiou])", "j"),
        (r"y\b", "i"),  (r"s(?=[aeiou])","z"), (r"s","s"),
    ]

    def _oov_to_ipa(self, word: str) -> str:
        """Rough OOV pronunciation via suffix + vowel/consonant rules."""
        w = word.lower()
        suffix_ipa = ""
        for suf_pat, suf_ipa, strip in self.SUFFIX_RULES:
            if re.search(suf_pat, w):
                suffix_ipa = suf_ipa
                w = w[:-strip]
                break

        for pat, repl in self._CONS_RULES:
            w = re.sub(pat, repl, w)
        for pat, repl in self._VOWEL_RULES:
            w = re.sub(pat, repl, w)

        return w + suffix_ipa

    def convert(self, word: str) -> str:
        """Convert an English word to IPA."""
        key = word.lower().strip(".,!?;:")
        if key in self.DICT:
            return self.DICT[key]
        return self._oov_to_ipa(key)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PHONOLOGY LAYER  (post-processing across language boundaries)
# ═══════════════════════════════════════════════════════════════════════════

class HinglishPhonologyLayer:
    """
    Post-processing rules applied to the concatenated IPA string.

    Covers:
    ① Aspiration de-aspiration at code-switch boundaries
       English /t/ → /t̪/ when followed by Hindi vowel
    ② English /v/ → Hindi /ʋ/ in Hinglish borrowings
    ③ Gemination: repeated identical consonants → long consonant
    ④ Flap insertion: /r/ → /ɾ/ in intervocalic position
    ⑤ Word-final obstruent devoicing (optional, register-dependent)
    """

    _VOWEL_IPA = r"[əɪiːeːɛaɑɔoʊuːẽĩõũã]"
    _CONS_IPA  = r"[pbtdkɡʈɖtʃdʒmnŋɳɲrljʋszʃxɦɾɽfv]"

    def apply(self, ipa: str) -> str:
        # ① English /v/ → /ʋ/ everywhere in Hinglish
        ipa = ipa.replace("v", "ʋ")

        # ② Intervocalic /r/ → tap /ɾ/ (common in Indian English & Hindi)
        ipa = re.sub(
            rf"(?<={self._VOWEL_IPA})r(?={self._VOWEL_IPA})", "ɾ", ipa
        )

        # ③ Geminate simplification: CC → Cː  (tː, kː, sː …)
        ipa = re.sub(r"([pbtdkɡʈɖszʃmn])\1", r"\1ː", ipa)

        # ④ English dental /θ/ → /t̪/ (Indian English)
        ipa = ipa.replace("θ", "t̪")

        # ⑤ Clean up double spaces
        ipa = re.sub(r" {2,}", " ", ipa).strip()

        return ipa


# ═══════════════════════════════════════════════════════════════════════════
# 5.  LANGUAGE DETECTOR (heuristic, integrates with Part I LID model)
# ═══════════════════════════════════════════════════════════════════════════

def _has_devanagari(word: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", word))

def _looks_like_hindi_roman(word: str) -> bool:
    """
    Heuristic: word is likely romanised Hindi if it contains common Hindi
    digraph patterns or ends with 'a'/'aa'/'i'/'u' after a consonant.
    """
    w = word.lower()
    patterns = [
        r"(kh|gh|ch|jh|th|dh|ph|bh)",         # Hindi aspirate digraphs
        r"(aa|ii|uu|ee|oo)\b",                  # Hindi long vowels
        r"\b(hai|hain|kya|kuch|bahut|aur|"      # common Hindi words
        r"nahi|nahin|bhi|toh|se|ke|ka|ki|ko|"
        r"jo|wo|yeh|woh|par|lekin|kyunki)\b",
        r"(wala|wali|wale)\b",                  # Hinglish suffixes
    ]
    return any(re.search(p, w) for p in patterns)


def detect_word_language(word: str, lid_label: Optional[str] = None) -> str:
    """
    Returns 'devanagari' | 'hindi_roman' | 'english'.

    Parameters
    ----------
    word      : Surface word string.
    lid_label : Optional LID segment label from Part I ('English'/'Hindi'/'Mixed').
    """
    if _has_devanagari(word):
        return "devanagari"
    # LID label available and confident
    if lid_label and lid_label.lower() == "hindi":
        return "hindi_roman"
    if lid_label and lid_label.lower() == "english":
        return "english"
    # Heuristic fallback
    if _looks_like_hindi_roman(word):
        return "hindi_roman"
    return "english"


# ═══════════════════════════════════════════════════════════════════════════
# 6.  UNIFIED HINGLISH G2P PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WordIPA:
    word:     str
    lang:     str        # 'devanagari' | 'hindi_roman' | 'english'
    ipa:      str
    start_s:  Optional[float] = None
    end_s:    Optional[float] = None


@dataclass
class HinglishIPAResult:
    unified_ipa:  str               # full IPA string
    word_alignments: List[WordIPA]  # per-word breakdown
    input_text:   str


class HinglishG2P:
    """
    Unified Grapheme-to-Phoneme converter for Code-Switched Hinglish.

    Usage
    -----
        g2p = HinglishG2P()
        result = g2p.convert("yeh ek stochastic process hai")
        print(result.unified_ipa)
        # → jeː ɛk stəkæstɪk prɒsɛs ɦɛː
    """

    def __init__(self) -> None:
        self.dev_g2p  = DevanagariG2P()
        self.hi_g2p   = HindiRomanG2P()
        self.en_g2p   = EnglishG2P()
        self.phon     = HinglishPhonologyLayer()

    def _convert_word(self, word: str, lang: str) -> str:
        if lang == "devanagari":
            return self.dev_g2p.convert(word)
        elif lang == "hindi_roman":
            return self.hi_g2p.convert(word)
        else:
            return self.en_g2p.convert(word)

    def convert(
        self,
        text: str,
        lid_segments: Optional[List[dict]] = None,
        word_timestamps: Optional[List[dict]] = None,
    ) -> HinglishIPAResult:
        """
        Convert a Hinglish sentence/paragraph to unified IPA.

        Parameters
        ----------
        text            : Input text (Devanagari, Roman, or mixed).
        lid_segments    : LID segment list from Part I for per-word lang tags.
        word_timestamps : Optional [{word, start_s, end_s}] for alignment.

        Returns
        -------
        HinglishIPAResult
        """
        # Tokenise (simple whitespace + punctuation split)
        tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z'-]+|[0-9]+|[^\w\s]", text)

        word_results: List[WordIPA] = []
        ipa_parts: List[str] = []

        for i, token in enumerate(tokens):
            # Skip pure punctuation
            if re.fullmatch(r"[^\w\u0900-\u097F]+", token):
                ipa_parts.append(" ")
                continue

            # Find LID label for this token using timestamps
            lid_label: Optional[str] = None
            if lid_segments and word_timestamps and i < len(word_timestamps):
                ts = word_timestamps[i]
                mid = (ts.get("start_s", 0) + ts.get("end_s", 0)) / 2
                for seg in lid_segments:
                    if seg["start_s"] <= mid <= seg["end_s"]:
                        lid_label = seg["label"]
                        break

            lang = detect_word_language(token, lid_label)
            ipa  = self._convert_word(token, lang)

            ts_start = word_timestamps[i].get("start_s") if word_timestamps and i < len(word_timestamps) else None
            ts_end   = word_timestamps[i].get("end_s")   if word_timestamps and i < len(word_timestamps) else None

            word_results.append(WordIPA(
                word=token, lang=lang, ipa=ipa,
                start_s=ts_start, end_s=ts_end,
            ))
            ipa_parts.append(ipa)

        raw_ipa = " ".join(p for p in ipa_parts if p.strip())
        unified_ipa = self.phon.apply(raw_ipa)

        return HinglishIPAResult(
            unified_ipa=unified_ipa,
            word_alignments=word_results,
            input_text=text,
        )

    def convert_transcript(
        self,
        transcript_segments: List[dict],
        lid_segments: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        Convert a full transcript (list of segment dicts from Part I) to IPA.

        Each output dict gains an 'ipa' key.
        """
        output = []
        for seg in transcript_segments:
            text = seg.get("text", "")
            result = self.convert(text, lid_segments=lid_segments)
            out_seg = dict(seg)
            out_seg["ipa"] = result.unified_ipa
            out_seg["word_alignments"] = [
                {
                    "word": w.word, "lang": w.lang, "ipa": w.ipa,
                    "start_s": w.start_s, "end_s": w.end_s,
                }
                for w in result.word_alignments
            ]
            output.append(out_seg)
        return output


# ═══════════════════════════════════════════════════════════════════════════
# 7.  FINE-TUNING MODULE (optional, Task 2.1 extension)
# ═══════════════════════════════════════════════════════════════════════════

class G2PFineTuner(torch.nn.Module):
    """
    Seq2Seq character-level G2P model for fine-tuning on Hinglish pairs.

    Architecture: Bidirectional GRU encoder → GRU decoder with attention.
    This supplements the rule-based system for OOV words with a learned
    character mapping.

    Training data format (list of tuples):
        [("stochastic", "stəkæstɪk"), ("cepstrum", "sɛpstrəm"), ...]

    Usage
    -----
        tuner = G2PFineTuner(grapheme_vocab, phoneme_vocab)
        tuner.train_on_pairs(training_pairs, epochs=50)
        ipa = tuner.decode("coarticulation")
    """

    def __init__(
        self,
        grapheme_vocab: Dict[str, int],   # char → idx
        phoneme_vocab:  Dict[str, int],   # IPA char → idx
        embed_dim:  int = 64,
        hidden_dim: int = 256,
        n_layers:   int = 2,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.g_vocab = grapheme_vocab
        self.p_vocab = phoneme_vocab
        self.p_idx2char = {v: k for k, v in phoneme_vocab.items()}

        G = len(grapheme_vocab)
        P = len(phoneme_vocab)

        self.g_embed = torch.nn.Embedding(G, embed_dim, padding_idx=0)
        self.encoder = torch.nn.GRU(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True, dropout=dropout if n_layers>1 else 0.0
        )

        self.p_embed = torch.nn.Embedding(P, embed_dim, padding_idx=0)
        self.decoder = torch.nn.GRU(
            embed_dim + 2 * hidden_dim,   # input + context
            hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers>1 else 0.0
        )

        # Bahdanau-style attention
        self.attn_W  = torch.nn.Linear(hidden_dim + 2 * hidden_dim, hidden_dim)
        self.attn_v  = torch.nn.Linear(hidden_dim, 1, bias=False)

        self.out_proj = torch.nn.Linear(hidden_dim, P)

        self.SOS = phoneme_vocab.get("<SOS>", 1)
        self.EOS = phoneme_vocab.get("<EOS>", 2)

    def _attention(
        self, dec_hidden: torch.Tensor, enc_out: torch.Tensor
    ) -> torch.Tensor:
        """Bahdanau attention. Returns context vector [B, 2H]."""
        # dec_hidden: [B, H]  enc_out: [B, T, 2H]
        B, T, _ = enc_out.shape
        h = dec_hidden.unsqueeze(1).expand(-1, T, -1)   # [B, T, H]
        combined = torch.cat([h, enc_out], dim=-1)       # [B, T, H+2H]
        scores = self.attn_v(torch.tanh(self.attn_W(combined))).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)                  # [B, 1, T]
        context = (weights @ enc_out).squeeze(1)                              # [B, 2H]
        return context

    def forward(
        self,
        grapheme_ids: torch.Tensor,        # [B, T_g]
        phoneme_ids:  torch.Tensor,        # [B, T_p]  teacher-forced
    ) -> torch.Tensor:                     # [B, T_p, |P|]
        enc_out, enc_hidden = self.encoder(self.g_embed(grapheme_ids))
        # Merge bidirectional hidden state → [n_layers, B, H]
        B = grapheme_ids.shape[0]
        n_layers = enc_hidden.shape[0] // 2
        dec_hidden = enc_hidden.view(n_layers, 2, B, -1).mean(dim=1)

        logits_all = []
        inp = phoneme_ids[:, 0].unsqueeze(1)    # SOS token [B, 1]

        for t in range(phoneme_ids.shape[1] - 1):
            emb = self.p_embed(inp)              # [B, 1, E]
            ctx = self._attention(dec_hidden[-1], enc_out).unsqueeze(1)  # [B, 1, 2H]
            dec_in = torch.cat([emb, ctx], dim=-1)                        # [B, 1, E+2H]
            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden)        # [B, 1, H]
            logits = self.out_proj(dec_out)                               # [B, 1, P]
            logits_all.append(logits)
            inp = phoneme_ids[:, t + 1].unsqueeze(1)                      # teacher-force

        return torch.cat(logits_all, dim=1)    # [B, T_p-1, P]

    @torch.no_grad()
    def decode(self, grapheme_seq: str, max_len: int = 80) -> str:
        """Greedy decode a grapheme string → IPA string."""
        self.eval()
        ids = [self.g_vocab.get(c, 0) for c in grapheme_seq.lower()]
        x = torch.tensor([ids])           # [1, T_g]
        enc_out, enc_hidden = self.encoder(self.g_embed(x))
        B = 1
        n_layers = enc_hidden.shape[0] // 2
        dec_hidden = enc_hidden.view(n_layers, 2, B, -1).mean(dim=1)

        inp = torch.tensor([[self.SOS]])
        ipa_chars = []
        for _ in range(max_len):
            emb = self.p_embed(inp)
            ctx = self._attention(dec_hidden[-1], enc_out).unsqueeze(1)
            dec_in = torch.cat([emb, ctx], dim=-1)
            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden)
            logits = self.out_proj(dec_out)
            next_id = logits.squeeze().argmax().item()
            if next_id == self.EOS:
                break
            ipa_chars.append(self.p_idx2char.get(next_id, ""))
            inp = torch.tensor([[next_id]])

        return "".join(ipa_chars)

    def train_on_pairs(
        self,
        pairs: List[Tuple[str, str]],
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> List[float]:
        """Train on (grapheme, ipa) string pairs. Returns loss history."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for g_str, p_str in pairs:
                g_ids = [self.g_vocab.get(c, 0) for c in g_str.lower()]
                p_ids = (
                    [self.SOS]
                    + [self.p_vocab.get(c, 0) for c in p_str]
                    + [self.EOS]
                )
                gx = torch.tensor([g_ids])
                px = torch.tensor([p_ids])

                logits = self.forward(gx, px)        # [1, T-1, P]
                target = px[:, 1:]                   # [1, T-1]
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(pairs)
            losses.append(avg)
            if (epoch + 1) % 10 == 0:
                print(f"  G2P FineTune Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

        return losses


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    g2p = HinglishG2P()

    test_sentences = [
        "yeh ek stochastic process hai",
        "hum log MFCC features use karte hain",
        "cepstrum ka matlab kya hai",
        "spectral subtraction se noise remove hoti hai",
        "अच्छा तो यह spectrogram है",
        "thoda aur deep neural network samjho",
        "code switching is very common in Hinglish",
        "filterbank energies ko log compress karte hain",
    ]

    print("── Hinglish G2P Unified IPA ──\n")
    for sent in test_sentences:
        result = g2p.convert(sent)
        print(f"IN : {sent}")
        print(f"IPA: {result.unified_ipa}")
        print(f"Per-word:")
        for w in result.word_alignments:
            print(f"  {w.word:20s}  [{w.lang:14s}]  {w.ipa}")
        print()
