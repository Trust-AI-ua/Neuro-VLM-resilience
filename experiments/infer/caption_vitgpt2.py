#!/usr/bin/env python3
"""Generate captions with a VisionEncoderDecoder (ViT-GPT2) checkpoint."""
import os, argparse, json
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
from tqdm import tqdm

def load_images(list_path):
    """Read absolute image paths from a newline-delimited text file."""
    paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--image_list",     required=True, help="txt: one absolute image path per line")
    ap.add_argument("--out_json",       required=True)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load everything from the fine-tuned directory
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # Make sure special tokens are defined consistently
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    paths = load_images(args.image_list)
    outputs = []

    # Greedy decoding = τ=0 baseline (no sampling, no beams)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        for p in tqdm(paths):
            img = Image.open(p).convert("RGB")
            pixel = image_processor(images=[img], return_tensors="pt").pixel_values.to(device)
            out_ids = model.generate(pixel, **gen_kwargs)
            text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            outputs.append({"image_path": p, "caption": text})

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out_json}  | items={len(outputs)}")

if __name__ == "__main__":
    main()