#!/usr/bin/env python3
"""Fine-tune PaliGemma captioning on TSV inputs with optional τ regularization."""

import os
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup,
)

@dataclass
class Example:
    path: str
    text: str

class CaptionTSV(Dataset):
    def __init__(self, tsv_path):
        self.items = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    p, t = line.split("\t", 1)
                    self.items.append(Example(p, t.strip()))
                except ValueError:
                    # Skip malformed rows gracefully.
                    continue
        if len(self.items) == 0:
            raise RuntimeError(f"No examples read from {tsv_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def collate_fn(batch, processor, device):
    from PIL import Image
    images  = [Image.open(b.path).convert("RGB") for b in batch]
    targets = [b.text for b in batch]

    # Use the processor for image + prompt tokenization.
    inputs = processor(
        images=images,
        text=["<image>"] * len(images),
        return_tensors="pt",
        padding=True
    )

    # Prepare caption targets and mask padding tokens.
    labels = processor.tokenizer(
        targets, return_tensors="pt", padding=True, truncation=True
    )["input_ids"]
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for k in inputs:
        inputs[k] = inputs[k].to(device)
    labels = labels.to(device)
    return inputs, labels

def ce_with_label_smoothing(logits, labels, smoothing):
    """
    logits: [B, T, V], labels: [B, T] with -100 masked
    """
    vocab = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)  # [B,T,V]
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        mask = labels != -100
        # one-hot with smoothing
        true_dist.fill_(smoothing / (vocab - 1))
        idx = labels.clone()
        idx[~mask] = 0
        true_dist.scatter_(2, idx.unsqueeze(-1), 1.0 - smoothing)
        # zero out masked positions
        true_dist = true_dist * mask.unsqueeze(-1)
    loss = -(true_dist * log_probs).sum() / mask.sum().clamp_min(1)
    return loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default="google/paligemma2-3b-mix-224")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="Regularization scale; τ=0 disables (baseline).")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model and processor: {args.model_id}")

    # Loading path (requires transformers >= 4.46)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Move to GPU if available
    model.to(device)
    model.train()

    print("[INFO] Model and processor loaded successfully.")

    train_ds = CaptionTSV(args.train_tsv)
    val_ds   = CaptionTSV(args.val_tsv)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor, device)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor, device)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    def run_epoch(loader, train=True):
        if train: model.train()
        else: model.eval()
        total = 0.0
        count = 0
        for step, (inputs, labels) in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**inputs, labels=None)  # we compute loss manually
                logits = out.logits  # [B,T,V]
                # align logits to label length
                max_t = min(logits.size(1), labels.size(1))
                logits = logits[:, :max_t, :]
                y = labels[:, :max_t]

                # base CE with label smoothing scaled by τ (clipped to [0, 0.2])
                smoothing = min(max(args.tau * 0.02, 0.0), 0.2)  # τ=5 -> 0.1
                ce = ce_with_label_smoothing(logits, y, smoothing)

                # small logit L2 to encourage softer distributions (scaled by τ)
                l2 = (logits ** 2).mean()
                loss = ce + (args.tau * 1e-4) * l2

            if train:
                loss.backward()
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    model.zero_grad(set_to_none=True)

            total += float(loss.detach().cpu())
            count += 1
        return total / max(count, 1)

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        print(f"[Epoch {epoch}] train={tr:.4f} | val={va:.4f}")

        # save at end of epoch
        model.save_pretrained(ckpt_dir)  # writes config.json + sharded *.safetensors + index
        (ckpt_dir / "LAST_TAU.txt").write_text(str(args.tau))
        print(f"Saved HF checkpoint (safetensors shards) -> {ckpt_dir}")

    print("Done.")

if __name__ == "__main__":
    main()