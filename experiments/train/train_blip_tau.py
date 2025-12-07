#!/usr/bin/env python3
"""Fine-tune BLIP captioning on TSV inputs with optional τ regularization."""
import os, argparse, random, json
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm
import hashlib

class TsvCaptionDataset(Dataset):
    # Each line: /abs/path/to/img.jpg<TAB>A caption...
    def __init__(self, tsv_path: str):
        self.samples: List[Tuple[str, str]] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.rstrip("\n")
                if not line or "\t" not in line: continue
                p, cap = line.split("\t", 1)
                self.samples.append((p.strip(), cap.strip()))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return img, y

def make_collator(processor, max_len: int = 30):
    # Ensure padding tokens are defined for label masking downstream.
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        imgs, caps = zip(*batch)
        enc = processor(
            images=list(imgs),
            text=list(caps),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        # labels = decoder targets; ignore pads with -100
        labels = enc["input_ids"].clone()
        labels[labels == pad_id] = -100
        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
    return collate

def set_seed(s=1337):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv",   required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.0, help="TopoLoss/Neuro coefficient")
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    mdl_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(mdl_name)
    model = BlipForConditionalGeneration.from_pretrained(mdl_name)
    # Enable hidden states for a stable τ penalty on decoder features
    model.config.output_hidden_states = True
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.output_hidden_states = True
        model.config.text_config.use_cache = False
    if hasattr(model, "text_decoder") and hasattr(model.text_decoder, "config"):
        model.text_decoder.config.output_hidden_states = True
        if hasattr(model.text_decoder.config, "use_cache"):
            model.text_decoder.config.use_cache = False
    model.to(device)

    def weight_fingerprint(m):
        # Cheap, stable fingerprint over a subset of parameters to detect weight drift.
        h = hashlib.md5()
        with torch.no_grad():
            for name, p in list(m.named_parameters())[:6]:  # first few params are enough
                h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()

    base_fp = weight_fingerprint(model)
    print(f"[fp] pre-train fingerprint={base_fp}")

    train_ds = TsvCaptionDataset(args.train_tsv)
    val_ds   = TsvCaptionDataset(args.val_tsv)
    collate  = make_collator(processor, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    opt = AdamW(model.parameters(), lr=args.lr)

    def _get_last_decoder_hidden(out):
        for name in ("decoder_hidden_states", "hidden_states", "text_decoder_hidden_states"):
            hs_seq = getattr(out, name, None)
            if isinstance(hs_seq, (list, tuple)) and hs_seq and hs_seq[-1] is not None:
                return hs_seq[-1], name
        # Some models return nested language-model outputs
        lmo = getattr(out, "language_model_outputs", None)
        if lmo is not None:
            for name in ("decoder_hidden_states", "hidden_states"):
                hs_seq = getattr(lmo, name, None)
                if isinstance(hs_seq, (list, tuple)) and hs_seq and hs_seq[-1] is not None:
                    return hs_seq[-1], f"language_model_outputs.{name}"
        return None, None

    def topo_penalty_from_hidden(out, debug_once=False):
        hs, src = _get_last_decoder_hidden(out)
        # Choose a device for a zero scalar when hidden states are unavailable.
        dev = out.logits.device if hasattr(out, "logits") else ("cuda" if torch.cuda.is_available() else "cpu")
        if hs is None:
            if debug_once:
                print("[τ-debug] no decoder hidden states found")
            return torch.tensor(0.0, device=dev)
        x = hs.float()  # (B, T, H)
        var = x.var(dim=(0, 1), unbiased=False).mean()
        if debug_once:
            print(f"[τ-debug] using={src} shape={tuple(x.shape)} penalty={float(var):.6f}")
        return var

    def run_epoch(loader: DataLoader, train: bool):
        model.train() if train else model.eval()
        total, steps = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in tqdm(loader, total=len(loader)):
                pix = batch["pixel_values"].to(device, non_blocking=True)
                lab = batch["labels"].to(device, non_blocking=True)
                ids = batch["input_ids"].to(device, non_blocking=True)
                msk = batch["attention_mask"].to(device, non_blocking=True)
                out = model(
                    pixel_values=pix,
                    input_ids=ids,
                    attention_mask=msk,
                    labels=lab,
                    output_hidden_states=True,
                    return_dict=True
                )
                loss = out.loss
                if args.tau > 0:
                    topo = topo_penalty_from_hidden(out, debug_once=(train and steps == 0))
                    loss = loss + args.tau * topo
                    if train and steps == 0:
                        print(f"[τ-debug] tau={args.tau} topo_penalty_first_batch={float(topo):.6f}")
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()

                    # Emit a one-time gradient norm probe for early batches.
                    if train and steps == 0:
                        total_grad = 0.0
                        for n, p in model.named_parameters():
                            if p.grad is not None:
                                g = p.grad.data.float()
                                total_grad += float(g.norm().item())
                        print(f"[probe] total_grad_norm_on_first_batch={total_grad:.6f}")

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                total += float(loss.item()); steps += 1
        return total / max(1, steps)

    for ep in range(1, args.epochs+1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader,   False)
        print(f"[Epoch {ep}] train_loss={tr:.4f} | val_loss={va:.4f}")
        # after each epoch
        ckpt_dir = os.path.join(args.out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # save HF checkpoint (this is the real model)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

        # write a small metadata json instead of an empty .pt file
        meta = {
            "epoch": ep,
            "tau": args.tau,
            "train_loss": float(tr),
            "val_loss": float(va),
        }
        with open(os.path.join(args.out_dir, f"blip_tau{int(args.tau)}_e{ep}.json"), "w") as f:
            import json; json.dump(meta, f, indent=2)

        post_fp = weight_fingerprint(model)
        print(f"[fp] post-epoch fingerprint={post_fp}")
        if post_fp == base_fp:
            print("[WARN] Fingerprint unchanged after epoch — weights may not be updating!")
        else:
            print("[OK] Fingerprint changed — training updated weights.")
        # refresh baseline so next epoch compares against current
        base_fp = post_fp

        print(f"Saved HF checkpoint -> {ckpt_dir}")
        print(f"Saved metadata -> blip_tau{int(args.tau)}_e{ep}.json")
    print("Done.")
if __name__ == "__main__":
    main()
