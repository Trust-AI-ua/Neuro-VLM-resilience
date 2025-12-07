#!/usr/bin/env python3
"""Generate captions with a fine-tuned PaliGemma checkpoint."""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="path to checkpoints/ saved by fine-tuning")
    ap.add_argument("--paths_file", required=True, help="txt with absolute image paths (one per line)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prompt", default="<image>")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor from the base model for tokenizer + image preprocessor config.
    BASE_MODEL_ID = "google/paligemma2-3b-mix-224"
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    # Load fine-tuned checkpoint weights.
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto"
    )

    paths = [p.strip() for p in Path(args.paths_file).read_text().splitlines() if p.strip()]
    out = []

    # Process images in simple batches for memory stability.
    for i in tqdm(range(0, len(paths), args.batch), desc="Paligemma@generate"):
        chunk = paths[i:i+args.batch]
        images = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(images=images, text=[args.prompt]*len(images), return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                num_beams=1
            )

        texts = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Remove the "Describe the image." prefix if the model echoes it verbatim.
        texts = [t.replace("Describe the image.", "").strip() for t in texts]
        for p, t in zip(chunk, texts):
            image_id = Path(p).stem  # filename without extension
            out.append({"image_path": p, "image_id": image_id, "caption": t})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {args.out_json} | items={len(out)}")

if __name__ == "__main__":
    main()