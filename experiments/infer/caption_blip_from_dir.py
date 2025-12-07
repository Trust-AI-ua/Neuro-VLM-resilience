#!/usr/bin/env python3
"""Generate BLIP captions from a Hugging Face checkpoint directory."""
import os, json, argparse, glob
from typing import List
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def collect_images(images_txt: str, images_dir: str) -> List[str]:
    """Return a sorted list of absolute image paths from a file or directory."""
    paths = []
    if images_txt:
        with open(images_txt, "r", encoding="utf-8") as handle:
            for line in handle:
                p = line.strip()
                if p and os.path.isfile(p):
                    paths.append(p)
    elif images_dir:
        for ext in ("*.jpg","*.jpeg","*.png"):
            paths += sorted(glob.glob(os.path.join(images_dir, ext)))
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_txt", default="")
    ap.add_argument("--images_dir", default="")
    ap.add_argument("--ckpt_dir", required=True, help="BLIP checkpoint directory saved via save_pretrained")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = collect_images(args.images_txt, args.images_dir)
    if not paths:
        raise SystemExit("No images found (use --images_txt or --images_dir).")
    print(f"Found {len(paths)} images.")

    processor = BlipProcessor.from_pretrained(args.ckpt_dir)
    model     = BlipForConditionalGeneration.from_pretrained(args.ckpt_dir).to(args.device).eval()

    out = []
    B = args.batch_size
    for i in range(0, len(paths), B):
        chunk = paths[i:i+B]
        images = [Image.open(p).convert("RGB") for p in chunk]
        enc = processor(images=images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            gen = model.generate(
                pixel_values=enc["pixel_values"],
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
            )

            # Handle both return types across transformers versions.
            seq = gen.sequences if hasattr(gen, "sequences") else gen

            # Force to a Python list-of-lists for downstream decoding.
            if isinstance(seq, torch.Tensor):
                seq = seq.detach().cpu().tolist()
            else:
                # Just in case any element is numpy/int64/etc.
                seq = [[int(t) for t in row] for row in seq]

            # Use tokenizer directly to avoid processor round-tripping quirks.
            preds = processor.tokenizer.batch_decode(seq, skip_special_tokens=True)
        for pth, pred in zip(chunk, preds):
            out.append({"image_path": pth, "caption": pred.strip()})
        print(f"Captioned {min(i+B, len(paths))}/{len(paths)}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out_json} | items={len(out)}")

if __name__ == "__main__":
    main()
