"""
evaluation_metrics.py
======================
Strict passing-criteria evaluation for the full pipeline.

Metric            Target          Description
──────────────────────────────────────────────────────────────────────────
WER               EN < 15 %       Word Error Rate  (English segments)
                  HI < 25 %       Word Error Rate  (Hindi segments)
MCD               < 8.0 dB        Mel-Cepstral Distortion (synth vs ref)
LID Switch Acc    ≤ 200 ms        Timestamp precision of language switches
EER               < 10 %          Anti-spoofing Equal Error Rate
Adversarial ε     reported        Min FGSM ε to flip LID  (SNR > 40 dB)

All metrics are implemented in pure PyTorch + standard library.
No external ASR evaluation packages (jiwer, etc.) are required.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Ta


# ═══════════════════════════════════════════════════════════════════════════
# 1.  WORD ERROR RATE  (WER)
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)   # keep Devanagari
    return [t for t in text.split() if t]


def levenshtein_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Compute WER edit distances via dynamic programming.

    Returns
    -------
    (substitutions, deletions, insertions, reference_length)
    """
    r, h = len(ref), len(hyp)
    # dp[i][j] = (cost, path)  where path tracks s/d/i counts
    INF = 10**9

    # Use space-optimised DP storing (S, D, I) counts alongside cost
    # cost  = S + D + I
    prev = [(j, 0, j, 0) for j in range(h + 1)]   # (cost, S, I, D)

    for i in range(1, r + 1):
        curr = [(INF, 0, 0, 0)] * (h + 1)
        curr[0] = (i, 0, 0, i)                     # i deletions
        for j in range(1, h + 1):
            del_op  = (prev[j][0] + 1,   prev[j][1],   prev[j][2],   prev[j][3] + 1)
            ins_op  = (curr[j-1][0] + 1, curr[j-1][1], curr[j-1][2] + 1, curr[j-1][3])
            sub_cost = 0 if ref[i-1] == hyp[j-1] else 1
            sub_op  = (prev[j-1][0] + sub_cost,
                       prev[j-1][1] + sub_cost,
                       prev[j-1][2], prev[j-1][3])
            curr[j] = min(del_op, ins_op, sub_op, key=lambda x: x[0])
        prev = curr

    cost, S, I, D = prev[h]
    return S, D, I, r


def compute_wer(
    references: List[str],
    hypotheses: List[str],
    lang_filter: Optional[str] = None,
    segments: Optional[List[dict]] = None,
) -> Dict:
    """
    Compute WER over paired reference / hypothesis sentences.

    Parameters
    ----------
    references   : List of ground-truth transcripts.
    hypotheses   : List of ASR hypotheses (same order).
    lang_filter  : If 'english' or 'hindi', only evaluate those segments.
    segments     : Segment dicts with 'language' key for filtering.

    Returns
    -------
    dict with keys: wer, substitutions, deletions, insertions, ref_words
    """
    total_S = total_D = total_I = total_R = 0

    for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if lang_filter and segments:
            lang = segments[idx].get("language", "mixed").lower()
            if lang_filter == "english" and lang not in ("english", "en"):
                continue
            if lang_filter == "hindi" and lang not in ("hindi", "hi"):
                continue

        ref_tok = _tokenize(ref)
        hyp_tok = _tokenize(hyp)
        if not ref_tok:
            continue
        S, D, I, R = levenshtein_distance(ref_tok, hyp_tok)
        total_S += S;  total_D += D;  total_I += I;  total_R += R

    wer = (total_S + total_D + total_I) / max(total_R, 1)
    return {
        "wer":           round(wer, 4),
        "wer_pct":       round(100 * wer, 2),
        "substitutions": total_S,
        "deletions":     total_D,
        "insertions":    total_I,
        "ref_words":     total_R,
        "pass":          (
            (wer < 0.15 if lang_filter == "english" else True) and
            (wer < 0.25 if lang_filter == "hindi"   else True)
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  MEL-CEPSTRAL DISTORTION  (MCD)
# ═══════════════════════════════════════════════════════════════════════════

class MCDCalculator:
    """
    Mel-Cepstral Distortion between a synthesised and reference waveform.

    MCD = (10 / ln 10) * sqrt(2 * Σ_{k=1}^{K} (c_ref[k] - c_hyp[k])^2)

    averaged over aligned frames (DTW alignment optional).

    Parameters
    ----------
    sr       : Sample rate (both signals resampled to this).
    n_fft    : FFT window.
    hop      : STFT hop size.
    n_mels   : Mel filter count.
    n_mfcc   : Number of cepstral coefficients (c_0 excluded by convention).
    use_dtw  : Whether to DTW-align frames before computing MCD.
    """

    FACTOR = 10.0 / math.log(10.0) * math.sqrt(2.0)   # ≈ 6.141

    def __init__(
        self,
        sr:      int  = 22_050,
        n_fft:   int  = 1024,
        hop:     int  = 256,
        n_mels:  int  = 80,
        n_mfcc:  int  = 13,
        use_dtw: bool = True,
    ) -> None:
        self.sr     = sr
        self.n_mfcc = n_mfcc
        self.use_dtw = use_dtw

        self.mel_spec = Ta.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=0.0, f_max=sr // 2,
        )
        self.mfcc_transform = Ta.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc + 1,    # +1 because we drop c_0
            melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": n_mels},
        )

    def _load_mono(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def _to_mfcc(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [T] → MFCC [T_frames, n_mfcc]  (c_0 excluded)"""
        mfcc = self.mfcc_transform(wav)    # [n_mfcc+1, T_frames]
        return mfcc[1:].T                  # [T_frames, n_mfcc]

    @staticmethod
    def _dtw_align(ref: torch.Tensor, hyp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align ref [R, D] and hyp [H, D] via DTW on cosine distance.

        Returns (aligned_ref, aligned_hyp) with equal T.
        """
        R, D = ref.shape
        H    = hyp.shape[0]

        # Cost matrix: L2 distance between each pair of frames
        cost = torch.cdist(ref, hyp, p=2)   # [R, H]

        # Accumulated cost with backtracking
        acc  = torch.full((R, H), float("inf"))
        acc[0, 0] = cost[0, 0]
        for i in range(1, R):
            acc[i, 0] = acc[i-1, 0] + cost[i, 0]
        for j in range(1, H):
            acc[0, j] = acc[0, j-1] + cost[0, j]
        for i in range(1, R):
            for j in range(1, H):
                acc[i, j] = cost[i, j] + torch.min(
                    torch.stack([acc[i-1, j], acc[i, j-1], acc[i-1, j-1]])
                )

        # Traceback
        path_r, path_h = [R - 1], [H - 1]
        i, j = R - 1, H - 1
        while i > 0 or j > 0:
            if i == 0:      j -= 1
            elif j == 0:    i -= 1
            else:
                opts = torch.stack([acc[i-1, j], acc[i, j-1], acc[i-1, j-1]])
                best = opts.argmin().item()
                if best == 0:   i -= 1
                elif best == 1: j -= 1
                else:           i -= 1; j -= 1
            path_r.insert(0, i)
            path_h.insert(0, j)

        pr = torch.tensor(path_r)
        ph = torch.tensor(path_h)
        return ref[pr], hyp[ph]

    @torch.no_grad()
    def compute(
        self,
        ref_path: str,
        hyp_path: str,
    ) -> Dict:
        """
        Compute MCD between reference and synthesised audio files.

        Returns dict with {'mcd', 'n_frames', 'pass'}.
        """
        ref_wav = self._load_mono(ref_path)
        hyp_wav = self._load_mono(hyp_path)

        ref_mfcc = self._to_mfcc(ref_wav)    # [R, K]
        hyp_mfcc = self._to_mfcc(hyp_wav)    # [H, K]

        if self.use_dtw:
            ref_mfcc, hyp_mfcc = self._dtw_align(ref_mfcc, hyp_mfcc)

        # Trim to equal length (fallback if DTW skipped)
        min_len = min(ref_mfcc.shape[0], hyp_mfcc.shape[0])
        ref_mfcc = ref_mfcc[:min_len]
        hyp_mfcc = hyp_mfcc[:min_len]

        diff = ref_mfcc - hyp_mfcc                          # [T, K]
        frame_mcd = self.FACTOR * diff.pow(2).sum(dim=-1).sqrt()  # [T]
        mcd = frame_mcd.mean().item()

        return {
            "mcd":      round(mcd, 4),
            "n_frames": min_len,
            "pass":     mcd < 8.0,
        }

    @torch.no_grad()
    def compute_from_tensors(
        self,
        ref_wav: torch.Tensor,    # [T_ref]  at self.sr
        hyp_wav: torch.Tensor,    # [T_hyp]  at self.sr
    ) -> float:
        """Compute MCD directly from waveform tensors. Returns scalar MCD (dB)."""
        ref_mfcc = self._to_mfcc(ref_wav)
        hyp_mfcc = self._to_mfcc(hyp_wav)
        if self.use_dtw:
            ref_mfcc, hyp_mfcc = self._dtw_align(ref_mfcc, hyp_mfcc)
        min_len = min(ref_mfcc.shape[0], hyp_mfcc.shape[0])
        diff = ref_mfcc[:min_len] - hyp_mfcc[:min_len]
        return (self.FACTOR * diff.pow(2).sum(dim=-1).sqrt()).mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# 3.  LID SWITCH TIMESTAMP ACCURACY  (≤ 200 ms)
# ═══════════════════════════════════════════════════════════════════════════

def compute_lid_switch_accuracy(
    predicted_segments: List[dict],
    reference_segments: List[dict],
    tolerance_ms: float = 200.0,
) -> Dict:
    """
    Evaluate language-switch detection accuracy.

    A "switch" is a transition between consecutive segments of different language.
    A predicted switch is correct if its timestamp is within `tolerance_ms` of a
    reference switch.

    Parameters
    ----------
    predicted_segments : List[{start_s, end_s, label}] from LID model.
    reference_segments : List[{start_s, end_s, label}] ground-truth.
    tolerance_ms       : Acceptable timestamp error in milliseconds (default 200).

    Returns
    -------
    dict with keys:
        n_ref_switches, n_pred_switches, true_positives,
        precision, recall, f1, mean_offset_ms, pass
    """
    tol_s = tolerance_ms / 1000.0

    def _extract_switches(segs: List[dict]) -> List[float]:
        """Return list of switch timestamps (boundary between lang-change pairs)."""
        switches = []
        for i in range(1, len(segs)):
            prev_lang = segs[i-1].get("label", segs[i-1].get("language", ""))
            curr_lang = segs[i].get("label", segs[i].get("language", ""))
            if prev_lang.lower() != curr_lang.lower():
                switches.append(segs[i]["start_s"])
        return switches

    ref_sw  = _extract_switches(reference_segments)
    pred_sw = _extract_switches(predicted_segments)

    if not ref_sw:
        return {
            "n_ref_switches": 0, "n_pred_switches": len(pred_sw),
            "true_positives": 0, "precision": 1.0, "recall": 1.0,
            "f1": 1.0, "mean_offset_ms": 0.0, "pass": True,
        }

    matched_ref  = set()
    matched_pred = set()
    offsets = []

    for pi, ps in enumerate(pred_sw):
        for ri, rs in enumerate(ref_sw):
            if ri in matched_ref:
                continue
            if abs(ps - rs) <= tol_s:
                matched_ref.add(ri)
                matched_pred.add(pi)
                offsets.append(abs(ps - rs) * 1000.0)
                break

    TP = len(matched_pred)
    precision = TP / max(len(pred_sw), 1)
    recall    = TP / max(len(ref_sw),  1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_offset = sum(offsets) / len(offsets) if offsets else float("inf")

    return {
        "n_ref_switches":  len(ref_sw),
        "n_pred_switches": len(pred_sw),
        "true_positives":  TP,
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "mean_offset_ms":  round(mean_offset, 2),
        "pass":            mean_offset <= tolerance_ms,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  EQUAL ERROR RATE  (EER)  —  used for anti-spoofing
# ═══════════════════════════════════════════════════════════════════════════

def compute_eer(
    labels: List[int],          # 0 = bona fide,  1 = spoof
    scores: List[float],        # higher score = more likely SPOOF
    n_thresholds: int = 1000,
) -> Dict:
    """
    Compute Equal Error Rate from binary labels and continuous scores.

    EER is the threshold at which FAR (False Accept Rate of spoof as bonafide)
    equals FRR (False Rejection Rate of bonafide as spoof).

    Parameters
    ----------
    labels       : Ground-truth binary labels (0=real, 1=spoof).
    scores       : Model output scores (higher → spoof).
    n_thresholds : Number of threshold values to scan.

    Returns
    -------
    dict with keys: eer, eer_threshold, far_at_eer, frr_at_eer, pass
    """
    assert len(labels) == len(scores), "labels and scores must be equal length"

    lab  = torch.tensor(labels,  dtype=torch.float32)
    sc   = torch.tensor(scores,  dtype=torch.float32)

    thresholds = torch.linspace(sc.min(), sc.max(), n_thresholds)
    best_eer   = 1.0
    best_thr   = thresholds[0].item()
    best_far   = 1.0
    best_frr   = 1.0

    bonafide_mask = (lab == 0)
    spoof_mask    = (lab == 1)
    n_bf  = bonafide_mask.sum().item()
    n_sp  = spoof_mask.sum().item()

    for thr in thresholds:
        pred_spoof = (sc >= thr)
        # FAR = spoof accepted as bonafide (spoof predicted < thr)
        # FRR = bonafide rejected (bonafide predicted >= thr)
        FAR = ((~pred_spoof) & spoof_mask).sum().item() / max(n_sp, 1)
        FRR = (pred_spoof    & bonafide_mask).sum().item() / max(n_bf, 1)

        eer_candidate = (FAR + FRR) / 2.0
        if abs(FAR - FRR) < abs(best_far - best_frr):
            best_eer = eer_candidate
            best_thr = thr.item()
            best_far = FAR
            best_frr = FRR

    return {
        "eer":           round(best_eer, 4),
        "eer_pct":       round(100 * best_eer, 2),
        "eer_threshold": round(best_thr, 6),
        "far_at_eer":    round(best_far, 4),
        "frr_at_eer":    round(best_frr, 4),
        "pass":          best_eer < 0.10,
    }


def compute_roc_auc(labels: List[int], scores: List[float]) -> float:
    """Compute AUC-ROC via trapezoidal integration (no sklearn required)."""
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (n_pos * n_neg)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ADVERSARIAL ε REPORT
# ═══════════════════════════════════════════════════════════════════════════

def compute_snr(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    SNR = 10 * log10(P_signal / P_noise)
    """
    noise = perturbed - original
    p_sig = original.pow(2).mean().clamp(min=1e-12)
    p_nse = noise.pow(2).mean().clamp(min=1e-12)
    return 10.0 * math.log10(p_sig.item() / p_nse.item())


def adversarial_epsilon_report(
    epsilon_results: List[Dict],
) -> Dict:
    """
    Summarise FGSM epsilon sweep results.

    Parameters
    ----------
    epsilon_results : list of dicts, each containing:
        {epsilon, snr_db, lid_flipped (bool), flip_rate (float 0-1)}

    Returns
    -------
    dict with keys:
        min_flip_epsilon, snr_at_min_flip, inaudible_epsilons,
        first_inaudible_flip, pass_snr_constraint
    """
    # Epsilons where LID prediction was flipped
    flipped = [r for r in epsilon_results if r.get("flip_rate", 0) > 0.5]
    # Epsilons where SNR > 40 dB (inaudible)
    inaudible = [r for r in epsilon_results if r.get("snr_db", 0) > 40.0]

    min_flip_eps = flipped[0]["epsilon"]   if flipped   else None
    snr_at_flip  = flipped[0]["snr_db"]   if flipped   else None

    # Epsilons that are BOTH inaudible AND flip LID
    inaudible_flip = [r for r in epsilon_results
                      if r.get("flip_rate", 0) > 0.5 and r.get("snr_db", 0) > 40.0]
    first_inaud_flip = inaudible_flip[0] if inaudible_flip else None

    return {
        "min_flip_epsilon":    min_flip_eps,
        "snr_at_min_flip_db":  snr_at_flip,
        "n_inaudible_eps":     len(inaudible),
        "first_inaudible_flip": first_inaud_flip,
        "pass_snr_constraint": first_inaud_flip is not None,
        "all_results":         epsilon_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  FULL EVALUATION REPORT  (aggregates all metrics)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationReport:
    wer_en:         Optional[Dict] = None
    wer_hi:         Optional[Dict] = None
    mcd:            Optional[Dict] = None
    lid_switch:     Optional[Dict] = None
    eer:            Optional[Dict] = None
    adv_epsilon:    Optional[Dict] = None

    def overall_pass(self) -> bool:
        checks = [
            self.wer_en  and self.wer_en.get("pass", False),
            self.wer_hi  and self.wer_hi.get("pass", False),
            self.mcd     and self.mcd.get("pass", False),
            self.lid_switch and self.lid_switch.get("pass", False),
            self.eer     and self.eer.get("pass", False),
        ]
        return all(c for c in checks if c is not None)

    def print_report(self) -> None:
        PASS = "✓ PASS"
        FAIL = "✗ FAIL"
        NA   = "— N/A"

        def status(d: Optional[Dict]) -> str:
            if d is None: return NA
            return PASS if d.get("pass") else FAIL

        print("\n" + "═" * 65)
        print("  EVALUATION REPORT  —  Strict Passing Criteria")
        print("═" * 65)

        if self.wer_en:
            flag = PASS if self.wer_en["pass"] else FAIL
            print(f"  WER  English : {self.wer_en['wer_pct']:6.2f}%  "
                  f"(target < 15.00%)  {flag}")
        if self.wer_hi:
            flag = PASS if self.wer_hi["pass"] else FAIL
            print(f"  WER  Hindi   : {self.wer_hi['wer_pct']:6.2f}%  "
                  f"(target < 25.00%)  {flag}")
        if self.mcd:
            flag = PASS if self.mcd["pass"] else FAIL
            print(f"  MCD          : {self.mcd['mcd']:6.3f} dB "
                  f"(target < 8.000 dB)  {flag}")
        if self.lid_switch:
            flag = PASS if self.lid_switch["pass"] else FAIL
            print(f"  LID Switch   : {self.lid_switch['mean_offset_ms']:6.1f} ms "
                  f"(target ≤ 200 ms)   {flag}")
        if self.eer:
            flag = PASS if self.eer["pass"] else FAIL
            print(f"  EER          : {self.eer['eer_pct']:6.2f}%  "
                  f"(target < 10.00%)   {flag}")
        if self.adv_epsilon:
            min_eps = self.adv_epsilon.get("min_flip_epsilon")
            snr     = self.adv_epsilon.get("snr_at_min_flip_db")
            flag    = PASS if self.adv_epsilon.get("pass_snr_constraint") else "— (no inaudible flip found)"
            eps_str = f"{min_eps:.6f}" if min_eps is not None else "N/A"
            snr_str = f"{snr:.1f} dB" if snr is not None else "N/A"
            print(f"  Adv ε (min)  : {eps_str}  SNR={snr_str}  {flag}")

        print("─" * 65)
        result = "★  OVERALL: PASS" if self.overall_pass() else "✗  OVERALL: FAIL"
        print(f"  {result}")
        print("═" * 65 + "\n")

    def to_dict(self) -> Dict:
        return {
            "wer_english":      self.wer_en,
            "wer_hindi":        self.wer_hi,
            "mcd":              self.mcd,
            "lid_switch":       self.lid_switch,
            "eer":              self.eer,
            "adversarial":      self.adv_epsilon,
            "overall_pass":     self.overall_pass(),
        }
