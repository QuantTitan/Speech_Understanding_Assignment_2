"""
lid_train.py  — Task 1.1 (Training)
=====================================
Training script for the Multi-Head Frame-Level LID model.

Dataset format expected
───────────────────────
A directory tree organised as:

    data/lid/
    ├── english/    *.wav   (English-only utterances)
    ├── hindi/      *.wav   (Hindi-only   utterances)
    ├── mixed/      *.wav   (Code-switched utterances)
    └── silence/    *.wav   (Silence / non-speech, optional)

Each WAV file is assumed to carry a *single* language label for all frames.
Code-switched files are labelled 'mixed' (class 2) throughout.

You can replace `LIDDataset` with a custom version that supplies per-frame
labels from forced alignment when such annotations are available (e.g. from
the MUCS shared task corpus or CMB dataset).

Training features
─────────────────
• SpecAugment  (frequency & time masking)
• Speed perturbation  (0.9 / 1.0 / 1.1)
• CosineAnnealingLR with warm-up
• Gradient clipping
• F1 evaluation (macro, per-class)  — target ≥ 0.85 on English+Hindi
• Best-model checkpoint by macro-F1

Usage
─────
    python lid_train.py \
        --data_dir data/lid \
        --save_dir checkpoints/lid \
        --epochs 40 \
        --batch_size 16 \
        --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import torchaudio
import torchaudio.transforms as Ta
import torchaudio.functional as taf

from lid_model import LIDConfig, MelFeatureExtractor, MultiHeadLID, MultiHeadLIDLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

LABEL_MAP = {"english": 0, "hindi": 1, "mixed": 2, "silence": 3}


class LIDDataset(Dataset):
    """
    Utterance-level LID dataset.

    Each sample is a (mel_frames [T, n_mels], label_seq [T]) pair,
    where label_seq is a *constant* vector of the utterance-level label.
    Padding is handled in the collate function.
    """

    MAX_DURATION_S  = 15.0    # discard very long files
    MIN_DURATION_S  = 0.5     # discard very short files
    TARGET_SR       = 16_000

    def __init__(
        self,
        data_dir: str,
        cfg: LIDConfig,
        augment: bool = True,
        speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
        freq_mask_F: int = 15,
        time_mask_T: int = 50,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
    ) -> None:
        super().__init__()
        self.cfg      = cfg
        self.augment  = augment
        self.speeds   = speed_factors
        self.freq_mask_F = freq_mask_F
        self.time_mask_T = time_mask_T
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

        # Build mel extractor (no grad)
        self.feat = MelFeatureExtractor(cfg)

        # Build SpecAugment transforms
        self.freq_mask = Ta.FrequencyMasking(freq_mask_param=freq_mask_F)
        self.time_mask = Ta.TimeMasking(time_mask_param=time_mask_T)

        # Discover files
        self.samples: List[Tuple[Path, int]] = []
        data_root = Path(data_dir)
        for lang, label_id in LABEL_MAP.items():
            lang_dir = data_root / lang
            if not lang_dir.exists():
                print(f"[Dataset] WARNING: {lang_dir} not found — skipping.")
                continue
            for wav in lang_dir.glob("*.wav"):
                self.samples.append((wav, label_id))

        print(
            f"[Dataset] Loaded {len(self.samples)} files from {data_dir}  "
            f"(augment={augment})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_resample(self, path: Path) -> Optional[torch.Tensor]:
        """Returns mono waveform [T] at TARGET_SR, or None if skipped."""
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0)                         # mono
        if sr != self.TARGET_SR:
            wav = taf.resample(wav, sr, self.TARGET_SR)
        dur = wav.shape[0] / self.TARGET_SR
        if dur < self.MIN_DURATION_S or dur > self.MAX_DURATION_S:
            return None
        return wav

    def _speed_perturb(self, wav: torch.Tensor) -> torch.Tensor:
        factor = random.choice(self.speeds)
        if factor == 1.0:
            return wav
        wav_out = taf.resample(
            wav,
            int(self.TARGET_SR * factor),
            self.TARGET_SR,
        )
        return wav_out

    def _spec_augment(self, log_mel: torch.Tensor) -> torch.Tensor:
        """log_mel : [T, n_mels]  → augmented [T, n_mels]"""
        x = log_mel.t().unsqueeze(0)             # [1, n_mels, T]
        for _ in range(self.n_freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.n_time_masks):
            x = self.time_mask(x)
        return x.squeeze(0).t()                  # [T, n_mels]

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        path, label_id = self.samples[idx]

        wav = self._load_and_resample(path)
        if wav is None:
            # Return a minimal dummy if file is unusable
            dummy_T = 50
            return {
                "mel":      torch.zeros(dummy_T, self.cfg.n_mels),
                "label4":   torch.full((dummy_T,), -1, dtype=torch.long),
                "label_en": torch.full((dummy_T,), -1, dtype=torch.long),
                "label_hi": torch.full((dummy_T,), -1, dtype=torch.long),
                "length":   dummy_T,
            }

        if self.augment:
            wav = self._speed_perturb(wav)

        # Mel feature extraction
        mel = self.feat(wav.unsqueeze(0).unsqueeze(0))  # [1, T', n_mels]
        mel = mel.squeeze(0)                             # [T', n_mels]

        if self.augment:
            mel = self._spec_augment(mel)

        T = mel.shape[0]
        label4   = torch.full((T,), label_id,         dtype=torch.long)
        label_en = torch.full((T,), int(label_id==0), dtype=torch.long)
        label_hi = torch.full((T,), int(label_id==1), dtype=torch.long)

        return {
            "mel":      mel,
            "label4":   label4,
            "label_en": label_en,
            "label_hi": label_hi,
            "length":   T,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Zero-pad to max-length in the batch; build padding mask."""
    max_T = max(b["length"] for b in batch)
    n_mels = batch[0]["mel"].shape[1]

    mel_pad      = torch.zeros(len(batch), max_T, n_mels)
    label4_pad   = torch.full((len(batch), max_T), -1, dtype=torch.long)
    label_en_pad = torch.full((len(batch), max_T), -1, dtype=torch.long)
    label_hi_pad = torch.full((len(batch), max_T), -1, dtype=torch.long)
    pad_mask     = torch.ones(len(batch), max_T, dtype=torch.bool)  # True = pad

    for i, b in enumerate(batch):
        T = b["length"]
        mel_pad[i, :T]      = b["mel"]
        label4_pad[i, :T]   = b["label4"]
        label_en_pad[i, :T] = b["label_en"]
        label_hi_pad[i, :T] = b["label_hi"]
        pad_mask[i, :T]     = False

    return {
        "mel":      mel_pad,
        "label4":   label4_pad,
        "label_en": label_en_pad,
        "label_hi": label_hi_pad,
        "pad_mask": pad_mask,
    }


# ---------------------------------------------------------------------------
# F1 computation (macro and per-class)
# ---------------------------------------------------------------------------

def compute_f1(
    all_preds: List[int],
    all_labels: List[int],
    n_classes: int = 4,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """Macro and per-class F1 from flat lists."""
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for p, g in zip(all_preds, all_labels):
        if g == ignore_index:
            continue
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    per_class_f1 = {}
    for c in range(n_classes):
        prec = tp[c] / (tp[c] + fp[c] + 1e-8)
        rec  = tp[c] / (tp[c] + fn[c] + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_class_f1[MultiHeadLID.LABEL_NAMES[c]] = round(f1, 4)

    macro_f1 = sum(per_class_f1.values()) / n_classes
    en_hi_f1 = (per_class_f1["English"] + per_class_f1["Hindi"]) / 2.0

    return {
        "macro_f1":  round(macro_f1, 4),
        "en_hi_f1":  round(en_hi_f1, 4),
        **per_class_f1,
    }


# ---------------------------------------------------------------------------
# LR scheduler with linear warm-up
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"[Train] Using device: {device}")

    cfg = LIDConfig()
    model = MultiHeadLID(cfg).to(device)
    criterion = MultiHeadLIDLoss(
        w_4cls=1.0, w_en=0.3, w_hi=0.3, smoothing=0.05
    )

    # ── Data ─────────────────────────────────────────────────────────────
    full_dataset = LIDDataset(args.data_dir, cfg, augment=True)
    val_size = max(1, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.augment = False      # no augmentation for validation

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # ── Checkpointing ────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_f1  = 0.0
    best_ckpt = save_dir / "lid_best.pt"

    # ── Training ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            mel      = batch["mel"].to(device)
            label4   = batch["label4"].to(device)
            label_en = batch["label_en"].to(device)
            label_hi = batch["label_hi"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            optimizer.zero_grad()
            out = model(mel, padding_mask=pad_mask)
            loss, _ = criterion(out, label4, label_en, label_hi)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:03d}  Step {step+1:04d}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        all_preds:  List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                mel      = batch["mel"].to(device)
                label4   = batch["label4"].to(device)
                pad_mask = batch["pad_mask"].to(device)
                out      = model(mel, padding_mask=pad_mask)
                preds    = out.labels_4cls.cpu().reshape(-1).tolist()
                labels   = label4.cpu().reshape(-1).tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)

        metrics = compute_f1(all_preds, all_labels)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}  loss={avg_loss:.4f}  "
            f"macro_F1={metrics['macro_f1']:.4f}  "
            f"en_hi_F1={metrics['en_hi_f1']:.4f}  "
            f"[English={metrics['English']:.4f}  Hindi={metrics['Hindi']:.4f}]  "
            f"time={elapsed:.1f}s"
        )

        # Save best model by English+Hindi F1
        if metrics["en_hi_f1"] > best_f1:
            best_f1 = metrics["en_hi_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "cfg": cfg,
                },
                str(best_ckpt),
            )
            print(
                f"  ✓ New best en+hi F1 = {best_f1:.4f}  → saved {best_ckpt}"
            )
            if best_f1 >= 0.85:
                print("  ★ Target F1 ≥ 0.85 reached!")

        # Always save latest
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "cfg": cfg},
            str(save_dir / "lid_latest.pt"),
        )

    print(f"\n[Train] Done. Best en+hi F1 = {best_f1:.4f}  saved → {best_ckpt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Multi-Head LID model (Task 1.1)")
    p.add_argument("--data_dir",    default="data/lid",       help="Root of LID dataset")
    p.add_argument("--save_dir",    default="checkpoints/lid", help="Where to save checkpoints")
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--log_every",   type=int,   default=50)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
