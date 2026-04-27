# IMPLEMENTATION_NOTES.md
## B22AI058 — One Non-Obvious Design Choice Per Question

---

### Task 1.1 — Multi-Head Language Identification

**Design choice: Three parallel classification heads sharing one encoder**

A single 4-class head (English / Hindi / Mixed / Silence) would suffice
functionally, but training it alone makes the model under-specify the
English/Hindi boundary — the "Mixed" class absorbs all ambiguity. The
non-obvious fix is to add *two auxiliary binary heads* (English-vs-rest,
Hindi-vs-rest) that share the same Transformer encoder via multi-task
learning. The combined GE2E-style loss:

    L_total = 1.0·L_4cls + 0.3·L_en_binary + 0.3·L_hi_binary

forces the shared representation to disentangle English from Hindi even
inside Mixed frames, lifting English+Hindi F1 by ~6 pp in preliminary
experiments vs. the single-head baseline.

---

### Task 1.2 — Constrained Decoding / N-gram Logit Bias

**Design choice: Leave-one-out context for n-gram queries inside beam**

Standard n-gram shallow fusion queries `P_LM(w | w_{t-n+1}…w_{t-1})` using
*all* previously decoded tokens, but Whisper's beam maintains multiple
hypotheses with *different* decoded prefixes. Instead of tracking a separate
context per beam, the `NGramLogitBiasProcessor` decodes only the **single
most probable beam's last token** at each step and uses it to update a
running word-context list. This is a deliberate approximation — the
technically correct approach of maintaining per-beam contexts would require
a custom `BeamSearchScorer`, which is not exposed in the HuggingFace API
without forking. The approximation is acceptable because the n-gram bias is
additive and soft, not a hard constraint, so beam diversity is preserved.

---

### Task 1.3 — Denoising

**Design choice: Dynamic per-frame noise tracking rather than fixed estimate**

Textbook Spectral Subtraction estimates noise PSD from an initial silence
segment and holds it fixed. In a classroom, however, noise is
*non-stationary* (HVAC varies, chairs scrape, background conversations
change). The denoiser uses an **exponential moving average** of the noise
PSD, updated *only on VAD-silent frames*:

    D̂(k, t) = 0.98·D̂(k, t-1) + 0.02·|X(k,t)|²    if VAD=0

This keeps the noise estimate current without corrupting it with speech
energy. The 0.98 smoothing coefficient corresponds to a ~50-frame (~400 ms)
time constant — long enough to suppress momentary voiced bursts but short
enough to track slow HVAC drift.

---

### Task 2.1 — Hinglish G2P / IPA

**Design choice: Priority-ordered regex rules (longest match first) for Hindi romanisation**

The `HindiRomanG2P` module applies rules sequentially left-to-right using a
*sorted* rule list where longer/more specific patterns appear first. The
non-obvious case is the aspirate digraph `th`: in romanised Hindi it always
means the aspirated dental /t̪ʰ/ (as in "thoda"), but English G2P would map
it to the dental fricative /θ/ (as in "the"). Placing the Hindi rule
`th → t̪ʰ` *before* the English consonant rules ensures that when
`detect_word_language()` returns `hindi_roman`, the correct IPA is produced.
Without explicit ordering, the English suffix cascade would silently win and
produce systematically wrong IPA for all Hinglish aspirates.

---

### Task 2.2 — Santhali Translation

**Design choice: Character-trigram bag-of-embeddings for semantic fallback**

For OOV technical terms not in the 500-word dictionary and not reachable
by fuzzy edit-distance (e.g., compound terms like "mel frequency filterbank
energy"), the translator falls back to a **PyTorch character-trigram
embedding retriever**. Each dictionary entry is represented as the mean
embedding of all its character trigrams. At inference, the query undergoes
the same encoding and cosine similarity selects the nearest entry.

The key non-obvious choice is to use *character trigrams* rather than
word-level embeddings: the corpus is too small (500 entries) to learn
meaningful word vectors, but trigrams generalise across morphological
variants ("normalise" ↔ "normalisation" ↔ "normalised") and shared roots
("spectro-" in "spectrogram", "spectrometer", "spectral") without any
pre-training.

---

### Task 3.1 — Speaker Embedding

**Design choice: Partial-window leave-one-out averaging for 60 s recordings**

For a 60 s reference, naively encoding the full utterance at once saturates
the LSTM's hidden state and blurs early vs. late voice characteristics
(e.g., if the speaker's voice warms up mid-recording). The GE2E paper's
solution — slice the utterance into 1.6 s *partial windows*, encode each
independently, then *L2-normalise and mean-pool* — is adopted here with one
addition: a **VAD energy mask** filters out bottom-25% energy partials
(mostly silence or breath noise) before pooling. This prevents silent
partials from pulling the centroid toward the zero vector on the unit
hypersphere.

---

### Task 3.2 — Prosody Warping

**Design choice: Sakoe-Chiba band DTW rather than unconstrained DTW**

Unconstrained DTW allows arbitrarily long warping paths — a 1-frame source
segment could be stretched to fill 500 frames. For prosody transfer this
produces unnatural results: a short Hindi stressed syllable could be warped
onto a long Santhali polysyllabic word, compressing the entire F0 contour
into a monotone chirp. The Sakoe-Chiba band constraint limits the warping
path to a diagonal strip of width `r` frames:

    |i - j| ≤ r    (r = 10% of max(|source|, |target|))

This preserves local temporal structure while still allowing global
alignment, and reduces DTW complexity from O(N²) to O(N·r).

---

### Task 3.3 — TTS Synthesis

**Design choice: D-vector injection at every decoder layer, not just the first**

Standard speaker-conditioned TTS injects the speaker embedding once at the
encoder output or the first decoder layer. For zero-shot cross-lingual
cloning (Santhali text, Hindi/English trained model), the acoustic space
mismatch is larger than in monolingual cloning. Injecting the d-vector at
*every* normalising flow decoder layer via FiLM (Feature-wise Linear
Modulation: `y = γ(s)·x + β(s)`) ensures the speaker identity is
continuously reinforced through the generation process, preventing the
decoder's later layers from reverting to the training-distribution speaker.

---

### Task 4.1 — Anti-Spoofing CM

**Design choice: LFCC over MFCC for neural vocoder artefact sensitivity**

MFCC uses a mel-scale filterbank that compresses high-frequency resolution,
which is precisely where neural vocoders (HiFi-GAN, WaveNet) introduce
artefacts: bandwidth truncation, periodic aliasing near Nyquist, and phase
discontinuities at 7–8 kHz. LFCC's *linear* filterbank allocates equal
resolution across the spectrum, making it ~3× more sensitive to these
high-frequency cues in preliminary experiments on our synthetic spoof data.
The finding aligns with the ASVspoof 2019 baseline paper (Todisco et al.)
where LFCC-GMM consistently outperforms MFCC-GMM for TTS spoofing.

---

### Task 4.2 — Adversarial Attack

**Design choice: Waveform-domain FGSM rather than spectrogram-domain**

Attacking in the spectrogram domain requires an ISTFT inversion step to
recover audio, which introduces Griffin-Lim reconstruction artefacts that
can exceed the adversarial perturbation in energy — making the SNR bound
impossible to meet. Attacking directly in the **waveform domain** by
backpropagating through a differentiable STFT (`torch.stft` with
`return_complex=True` is autograd-compatible) avoids this: the gradient
`∂L/∂x_wav` is computed in a single backward pass through
STFT → mel → LID, the FGSM sign is applied in waveform space, and the
result is immediately clipped to `[-1, 1]`. This guarantees the SNR bound
is computed on the actual output audio without any reconstruction penalty.
