"""
ngram_lm.py
===========
N-gram Language Model with Kneser-Ney smoothing.

Trained on a Speech-Processing course syllabus so that technical terms
(cepstrum, stochastic, MFCC, …) receive elevated probability during
Whisper constrained decoding (Task 1.2).

Usage:
    lm = NGramLM(n=3)
    lm.train(SYLLABUS_CORPUS)
    lp = lm.log_prob(("mel", "frequency"), "cepstral")   # → float
    lm.save("ngram_lm.pkl")
    lm = NGramLM.load("ngram_lm.pkl")
"""

import re
import math
import pickle
import torch
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default syllabus corpus – extend with the actual course PDF content
# ---------------------------------------------------------------------------
SYLLABUS_CORPUS: List[str] = [
    # ── Fundamentals ────────────────────────────────────────────────────────
    "Speech processing involves the analysis synthesis and recognition of spoken language.",
    "The fundamental frequency F0 of voiced speech determines the perceived pitch.",
    "Cepstral analysis separates the excitation source from the vocal tract filter.",
    "Mel-frequency cepstral coefficients MFCCs are the dominant features in ASR.",
    "The cepstrum is the inverse Fourier transform of the log power spectrum.",
    "Autocorrelation of the speech signal is used for pitch detection and LPC analysis.",
    "Linear predictive coding LPC models the vocal tract as an all-pole filter.",
    "The short-time Fourier transform STFT reveals spectral dynamics of speech.",
    "A spectrogram is a time-frequency representation of the speech waveform.",
    "Pre-emphasis filtering boosts high-frequency content before feature extraction.",
    # ── Acoustic Phonetics ───────────────────────────────────────────────────
    "Phonemes are the smallest units of sound that distinguish meaning in a language.",
    "Allophones are context-dependent phonetic realisations of the same phoneme.",
    "Voiced speech is produced when the vocal folds vibrate during articulation.",
    "Unvoiced fricatives such as /s/ and /f/ are produced without vocal fold vibration.",
    "Plosives or stops involve a complete closure of the vocal tract followed by release.",
    "Formant frequencies F1 F2 F3 characterise vowel sounds in the acoustic space.",
    "Coarticulation describes how adjacent phonemes influence each other's articulation.",
    "Prosody encompasses rhythm stress and intonation patterns in spoken language.",
    "The source filter model separates glottal excitation from vocal tract resonances.",
    # ── Feature Extraction ───────────────────────────────────────────────────
    "Delta and delta-delta coefficients capture temporal dynamics of MFCC features.",
    "A mel filterbank applies triangular filters spaced on the perceptual mel scale.",
    "The mel scale approximates the human auditory system's frequency resolution.",
    "Filterbank energies are log-compressed before discrete cosine transform DCT.",
    "The DCT decorrelates filterbank energies to produce cepstral coefficients.",
    "Feature normalisation techniques include CMVN cepstral mean variance normalisation.",
    "Per-utterance cepstral mean subtraction CMS removes channel effects from features.",
    "Perceptual linear prediction PLP features are an alternative to MFCC.",
    # ── Acoustic Modelling ───────────────────────────────────────────────────
    "Hidden Markov models HMM are the classical acoustic modelling framework for ASR.",
    "A Gaussian mixture model GMM represents the emission probability of each HMM state.",
    "The Viterbi algorithm finds the most probable state sequence through the HMM.",
    "Forward-backward algorithm computes state occupation probabilities efficiently.",
    "Baum-Welch expectation maximisation trains HMM parameters from speech data.",
    "Deep neural network DNN acoustic models replaced GMM-HMMs in hybrid ASR systems.",
    "The connectionist temporal classification CTC loss enables end-to-end ASR training.",
    "Attention-based encoder-decoder models such as LAS perform sequence-to-sequence ASR.",
    "Transformer architectures use multi-head self-attention for sequence modelling.",
    "The Whisper model performs robust multilingual ASR via weak supervision.",
    # ── Language Modelling ───────────────────────────────────────────────────
    "N-gram language models assign probability to word sequences via the chain rule.",
    "Kneser-Ney smoothing provides superior generalisation for n-gram language models.",
    "Perplexity measures how well a language model predicts held-out text.",
    "Recurrent neural network language models RNNLM capture long-range dependencies.",
    "Shallow fusion combines ASR and language model scores during beam search decoding.",
    "Cold fusion integrates the language model at the encoder level during training.",
    "Logit bias shifts decoder output probabilities to favour specific vocabulary items.",
    "Constrained beam search restricts hypotheses to those satisfying hard constraints.",
    # ── Code Switching ───────────────────────────────────────────────────────
    "Code switching refers to alternating between two or more languages in discourse.",
    "Hinglish is a common code-switched variety mixing Hindi and English in India.",
    "Language identification LID distinguishes the language spoken in each audio frame.",
    "Multilingual ASR systems handle multiple languages within a single model.",
    "Matrix language frame MLF theory explains syntactic constraints in code-switching.",
    "Intra-sentential code switching occurs within a single clause or sentence.",
    "Inter-sentential code switching happens between sentences or utterances.",
    # ── Noise & Robustness ───────────────────────────────────────────────────
    "Spectral subtraction estimates the noise spectrum during silence periods.",
    "The Wiener filter minimises mean squared error between clean and noisy speech.",
    "Log-MMSE estimation is a speech enhancement algorithm based on spectral statistics.",
    "DeepFilterNet uses neural post-filtering for full-band speech enhancement.",
    "Reverberation arises from multi-path reflections of sound in enclosed spaces.",
    "Dereverberation methods include weighted prediction error WPE and beamforming.",
    "Signal-to-noise ratio SNR quantifies the quality of a noisy speech signal.",
    "Noise robustness techniques include feature normalisation and multi-condition training.",
    # ── Speech Synthesis ─────────────────────────────────────────────────────
    "Text-to-speech TTS converts written text into intelligible synthetic speech.",
    "Zero-shot voice cloning synthesises speech in a target voice from few seconds.",
    "Speaker embeddings encode speaker identity in a fixed-dimensional vector space.",
    "VITS variational inference with adversarial learning is an end-to-end TTS model.",
    "YourTTS and XTTS enable cross-lingual zero-shot speaker adaptation.",
    "WaveNet is an autoregressive generative model for raw audio waveform synthesis.",
    "HiFi-GAN is a generative adversarial network vocoder for high-fidelity audio.",
    "Stochastic duration predictor models duration variability in speech synthesis.",
    "Prosody transfer moves rhythmic and intonational patterns across utterances.",
    # ── Low-Resource Languages ───────────────────────────────────────────────
    "Low-resource languages lack sufficient transcribed speech data for ASR training.",
    "Santhali Maithili Gondi and Bhojpuri are low-resource languages spoken in India.",
    "Transfer learning adapts high-resource ASR models to low-resource target languages.",
    "Self-supervised learning exploits unlabelled audio for robust speech representations.",
    "Wav2Vec 2.0 is a self-supervised model pre-trained via contrastive learning on audio.",
    "Data augmentation techniques include speed perturbation SpecAugment and room simulation.",
    "Cross-lingual embeddings align acoustic spaces across multiple languages.",
    # ── Evaluation ───────────────────────────────────────────────────────────
    "Word error rate WER is the primary metric for evaluating ASR system accuracy.",
    "Character error rate CER is often used for morphologically rich or agglutinative languages.",
    "F1 score is the harmonic mean of precision and recall for classification tasks.",
    "Mean opinion score MOS evaluates the perceived naturalness of synthesised speech.",
]


class NGramLM:
    """
    N-gram LM with optional Kneser-Ney (default) or Laplace smoothing.

    Parameters
    ----------
    n          : Order of the n-gram (default 3 = trigram).
    smoothing  : 'kneser_ney' | 'laplace'.
    discount   : Kneser-Ney absolute discount parameter (default 0.75).
    """

    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(
        self,
        n: int = 3,
        smoothing: str = "kneser_ney",
        discount: float = 0.75,
    ) -> None:
        self.n = n
        self.smoothing = smoothing
        self.discount = discount

        # ngram_counts[context_tuple][word] = count
        self.ngram_counts: Dict[Tuple, Counter] = defaultdict(Counter)
        # Total count for each context
        self.context_counts: Dict[Tuple, int] = defaultdict(int)
        # Unigram counts (for backoff)
        self.unigram_counts: Counter = Counter()
        self.total_tokens: int = 0
        self.vocab: set = set()

        # Technical-term boost applied on top of LM probability
        self._technical_terms: List[str] = []

    # ── Training ────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-']", " ", text)
        return [t for t in text.split() if t]

    def train(self, corpus: List[str]) -> None:
        """Fit n-gram counts on a list of sentences."""
        for sentence in corpus:
            tokens = (
                [self.SOS] * (self.n - 1)
                + self._tokenize(sentence)
                + [self.EOS]
            )
            for tok in tokens:
                self.unigram_counts[tok] += 1
                self.vocab.add(tok)
            self.total_tokens += len(tokens)

            for i in range(len(tokens) - self.n + 1):
                ctx = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[ctx][word] += 1
                self.context_counts[ctx] += 1

    def add_technical_terms(self, terms: List[str]) -> None:
        """
        Register domain-specific terms.
        These receive an extra log-probability boost during logit bias
        computation (see `technical_term_log_boost`).
        """
        for t in terms:
            self._technical_terms.extend(self._tokenize(t))
        self._technical_terms = list(set(self._technical_terms))

    # ── Probability Queries ─────────────────────────────────────────────────

    def log_prob(self, context: Tuple[str, ...], word: str) -> float:
        """Return log P(word | context) with smoothing."""
        if self.smoothing == "laplace":
            return self._laplace_log_prob(context, word)
        return self._kn_log_prob(context, word, order=self.n)

    def _laplace_log_prob(self, context: Tuple[str, ...], word: str) -> float:
        ctx = context[-(self.n - 1) :]
        count = self.ngram_counts[ctx].get(word, 0)
        ctx_count = self.context_counts.get(ctx, 0)
        V = len(self.vocab) + 1
        return math.log((count + 1) / (ctx_count + V) + 1e-30)

    def _kn_log_prob(
        self, context: Tuple[str, ...], word: str, order: int
    ) -> float:
        """Recursively compute Kneser-Ney probability."""
        ctx = context[-(order - 1) :] if order > 1 else ()

        if order == 1:
            # Lowest-order: unigram
            count = self.unigram_counts.get(word, 0)
            prob = max(count, 1) / (self.total_tokens + len(self.vocab))
            return math.log(prob + 1e-30)

        ctx_count = self.context_counts.get(ctx, 0)
        if ctx_count == 0:
            return self._kn_log_prob(context, word, order - 1)

        word_count = self.ngram_counts[ctx].get(word, 0)
        prob_main = max(word_count - self.discount, 0.0) / ctx_count

        # Interpolation weight
        n_types = len(self.ngram_counts[ctx])
        lam = (self.discount * n_types) / ctx_count

        backoff = math.exp(self._kn_log_prob(context, word, order - 1))
        prob = prob_main + lam * backoff
        return math.log(max(prob, 1e-30))

    def score_sentence(self, tokens: List[str]) -> float:
        """Sum of log-probabilities over a token sequence."""
        log_p = 0.0
        ctx: List[str] = [self.SOS] * (self.n - 1)
        for tok in tokens:
            log_p += self.log_prob(tuple(ctx), tok)
            ctx = ctx[1:] + [tok]
        return log_p

    # ── Logit Bias Utility ──────────────────────────────────────────────────

    def get_logit_bias(
        self,
        decoded_words: List[str],
        whisper_vocab: Dict[str, int],   # token_text → token_id
        scale: float = 3.0,
        technical_boost: float = 2.5,
    ) -> Dict[int, float]:
        """
        Compute a {token_id: bias} dict to inject into Whisper logits.

        Strategy
        --------
        1. For every surface form in the Whisper vocabulary that is also in
           *this* LM's vocabulary, compute log P(word | last (n-1) words).
        2. Convert to a bias scaled by `scale`.
        3. Add an extra `technical_boost` for registered technical terms.

        Parameters
        ----------
        decoded_words : Running list of already-decoded word tokens.
        whisper_vocab : Mapping from text string to Whisper token ID.
        scale         : Multiplier applied to LM log-probs before adding.
        technical_boost : Extra bonus for domain technical terms.

        Returns
        -------
        dict  {token_id: float bias}   — sparse; only non-zero entries.
        """
        ctx = tuple(decoded_words[-(self.n - 1) :])
        bias: Dict[int, float] = {}

        for surface, tok_id in whisper_vocab.items():
            word = surface.strip().lower()
            if not word or word.startswith("<"):
                continue
            lp = self.log_prob(ctx, word)
            b = scale * lp
            if word in self._technical_terms:
                b += technical_boost
            if b != 0.0:
                bias[tok_id] = b
        return bias

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[NGramLM] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NGramLM":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[NGramLM] Loaded ← {path}  (n={model.n}, vocab={len(model.vocab)})")
        return model


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_syllabus_lm(
    n: int = 3,
    extra_corpus: Optional[List[str]] = None,
    extra_terms: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> NGramLM:
    """
    Build and return a trigram LM pre-trained on SYLLABUS_CORPUS.

    Parameters
    ----------
    extra_corpus : Additional sentences (e.g. from the actual course PDF).
    extra_terms  : Extra technical terms to register for boosting.
    save_path    : If given, pickle the trained model to disk.
    """
    corpus = SYLLABUS_CORPUS + (extra_corpus or [])

    DEFAULT_TECHNICAL_TERMS = [
        "stochastic", "cepstrum", "cepstral", "mfcc", "spectrogram",
        "mel-frequency", "autocorrelation", "filterbank", "formant",
        "phoneme", "allophone", "coarticulation", "prosody", "fricative",
        "plosive", "voiced", "unvoiced", "hmm", "ctc", "wav2vec",
        "whisper", "transformer", "attention", "kneser-ney", "perplexity",
        "reverberation", "spectral-subtraction", "dereverberation",
        "hinglish", "code-switching", "lid", "cmvn", "lpc", "dft",
        "istft", "stft", "beamforming", "vocoder", "wer", "cer",
    ]

    lm = NGramLM(n=n)
    lm.train(corpus)
    lm.add_technical_terms(DEFAULT_TECHNICAL_TERMS + (extra_terms or []))

    print(
        f"[NGramLM] Trained  n={n}  vocab={len(lm.vocab)}  "
        f"contexts={len(lm.ngram_counts)}  "
        f"tech_terms={len(lm._technical_terms)}"
    )

    if save_path:
        lm.save(save_path)
    return lm


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lm = build_syllabus_lm(n=3, save_path="ngram_lm.pkl")

    tests = [
        (("mel", "frequency"), "cepstral"),
        (("hidden", "markov"), "model"),
        (("spectral",), "subtraction"),
        (("code",), "switching"),
    ]
    print("\n── Log-probability spot checks ──")
    for ctx, word in tests:
        lp = lm.log_prob(ctx, word)
        print(f"  P({word!r} | {ctx}) = {lp:.4f}  (≈ {math.exp(lp):.4e})")
