#!/usr/bin/env python3
"""Quick standardized eval on saved checkpoints (backpack + transformer)."""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from evaluate import (
    evaluate_perplexity,
    evaluate_sentence_similarity_baseline,
    load_model,
    load_test_data,
)


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def eval_model(name, path, device, data_dir):
    print(f"\n{'='*70}\nEVALUATING: {name}\n{'='*70}")
    t0 = time.time()
    model, config = load_model(path, device)
    params = sum(p.numel() for p in model.parameters())
    tokenizer_name = getattr(config, "tokenizer_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


    print("\nSentence similarity (translation vs random)...")
    sent = evaluate_sentence_similarity_baseline(
        model, tokenizer, device, data_dir=data_dir, n_pairs=200
    )
    results["sentence_similarity"] = sent

    print("\nPerplexity...")
    pairs = load_test_data(data_dir, "en-fr", max_samples=200, split="validation")
    interleaved = [f"{en} <|lang_sep|> {fr}" for en, fr in pairs]
    ppl = evaluate_perplexity(model, tokenizer, interleaved, device, max_samples=200, batch_size=4)
    results["perplexity"] = ppl
    if ppl:
        print(f"  Perplexity={ppl['perplexity']:.2f}")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data_dir", default="data/europarl")
    parser.add_argument("--models", default="backpack,transformer",
                        help="Comma-separated: backpack, transformer")
    parser.add_argument("--out", default="out/ckpt_eval_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    models = {
        "backpack":"/content/drive/MyDrive/multilingual-backpacks-checkpoints_real/backpack_full",
        "transformer": "out/transformer_full",
    }
    selected = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results = {}
    for name in selected:
        path = models.get(name)
        if not path or not os.path.exists(os.path.join(path, "ckpt.pt")):
            print(f"Skipping {name}: no ckpt at {path}")
            continue
        all_results[name] = eval_model(
            name, path, device, args.data_dir, args.max_multisimlex, args.multisimlex_dir
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.out}")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, r in all_results.items():
        sent = r.get("sentence_similarity", {}) or {}
        ppl = r.get("perplexity", {}) or {}
        print(
            f"| μ_trans={sent.get('mu_trans', 'n/a')} "
            f"| PPL={ppl.get('perplexity', 'n/a')}"
        )


if __name__ == "__main__":
    main()
