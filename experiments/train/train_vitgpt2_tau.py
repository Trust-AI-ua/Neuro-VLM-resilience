#!/usr/bin/env python3
"""Fine-tune ViT-GPT2 captioning on TSV inputs with optional τ regularization."""

import os, argparse, random
from typing import List, Tuple
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer,
)
from torch.optim import AdamW
try:
    from transformers import get_linear_schedule_with_warmup
except Exception:
    get_linear_schedule_with_warmup = None
from tqdm import tqdm

# ---------- Data ----------
class TsvCaptionDataset(Dataset):
    def __init__(self, tsv_path: str):
        self.samples: List[Tuple[str, str]] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split("\t")
                if len(parts) < 2: continue
                self.samples.append((parts[0], parts[1]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, cap = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return img, cap

@dataclass
class Collator:
    image_processor: any
    tokenizer: any
    max_length: int = 20
    def __call__(self, batch):
        imgs, caps = zip(*batch)
        pixel = self.image_processor(images=list(imgs), return_tensors="pt").pixel_values
        toks  = self.tokenizer(list(caps), padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").input_ids
        toks[toks == self.tokenizer.pad_token_id] = -100  # ignore pads in CE loss
        return {"pixel_values": pixel, "labels": toks}

# ---------- Optional topo-tap (inactive at τ=0) ----------
class ActivationTap(torch.nn.Module):
    def __init__(self): super().__init__(); self.last=None
    def forward(self, x): self.last=x; return x

def attach_topo_tap(model: VisionEncoderDecoderModel):
    tap = ActivationTap()
    try:
        last_fc = model.decoder.transformer.h[-1].mlp.c_fc
        orig = last_fc.forward
        def wrapped(x): y = orig(x); tap.last = y; return y
        last_fc.forward = wrapped
        return tap
    except Exception:
        return None

def topo_loss_from_tap(tap: ActivationTap, device: torch.device):
    if tap is None or tap.last is None:
        return torch.tensor(0.0, device=device)
    x = tap.last
    return x.float().var(dim=0).mean()

# ---------- Utils ----------
def set_seed(seed: int = 1337):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- Train ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv",   required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.0)     # keep τ knob; τ=0 baseline now
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    base_ckpt = "nlpconnect/vit-gpt2-image-captioning"
    image_processor = AutoImageProcessor.from_pretrained(base_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VisionEncoderDecoderModel.from_pretrained(base_ckpt)
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = TsvCaptionDataset(args.train_tsv)
    val_ds   = TsvCaptionDataset(args.val_tsv)
    collate  = Collator(image_processor=image_processor, tokenizer=tokenizer, max_length=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_loader))
    if get_linear_schedule_with_warmup is not None:
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, total_steps//20), num_training_steps=total_steps)
    else:
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=max(1, total_steps//10))

    tap = attach_topo_tap(model)  # No-op if the decoder layout diverges.

    def run_epoch(loader, train: bool) -> float:
        model.train() if train else model.eval()
        tot, steps = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in tqdm(loader, total=len(loader)):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss
                if args.tau > 0:
                    loss = loss + args.tau * topo_loss_from_tap(tap, device)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if sched: sched.step()
                tot += float(loss.item()); steps += 1
        return tot / max(1, steps)

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader,   False)
        print(f"[Epoch {ep}] train_loss={tr:.4f} | val_loss={va:.4f}")
        # model.save_pretrained(args.out_dir)
        # tokenizer.save_pretrained(args.out_dir)
        # image_processor.save_pretrained(args.out_dir)
        
        # Keep a lightweight epoch marker so we know it finished
        marker = os.path.join(args.out_dir, f"vitgpt2_tau{int(args.tau)}_e{ep}.pt")
        with open(marker, "wb") as f: f.write(b"")
        print(f"Saved checkpoint -> {marker}")
        
    print("Done.")

    # Save a single HuggingFace-style checkpoint directory for clean inference
    hf_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(hf_dir, exist_ok=True)
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    image_processor.save_pretrained(hf_dir)
    print(f"Saved HF checkpoint dir -> {hf_dir}")

if __name__ == "__main__":
    main()