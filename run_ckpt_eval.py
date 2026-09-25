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

def ablate_sense(model, sense_idx):
    old_forward = model.sense_layer.forward

    def patched_forward(token_embs):
        out = old_forward(token_embs)

        B, T, _ = out.shape

        # [B, T, n_senses * n_embd]
        # -> [B, T, n_senses, n_embd]
        out = out.view(
            B, T,
            model.n_senses,
            model.config.n_embd
        )

        # zero this sense for every token
        out[:, :, sense_idx, :] = 0.0

        return out.view(
            B, T,
            model.n_senses * model.config.n_embd
        )

    model.sense_layer.forward = patched_forward

def project_transformer(model, tokenizer, professions,
                        male_word="il", female_word="elle"):

    old_forward = model.token_embeddings.forward

    male_id = tokenizer.encode(
        male_word, add_special_tokens=False
    )[0]
    female_id = tokenizer.encode(
        female_word, add_special_tokens=False
    )[0]

    with torch.no_grad():
        E = model.token_embeddings.weight
        g = E[male_id] - E[female_id]
        g = g / (g.norm() + 1e-12)

    target_ids = []
    for word in professions:
        target_ids.extend(
            tokenizer.encode(word, add_special_tokens=False)
        )
    target_ids = list(set(target_ids))

    def patched_forward(input_ids):
        emb = old_forward(input_ids)

        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for tok_id in target_ids:
            mask |= input_ids == tok_id

        if mask.any():
            selected = emb[mask]
            projection = (selected @ g).unsqueeze(-1) * g
            emb[mask] = selected - projection

        return emb

    model.token_embeddings.forward = patched_forward

def eval_model(name, path, device, data_dir,
               ablate_sense_idx=None, project=False):    
    print(f"\n{'='*70}\nEVALUATING: {name}\n{'='*70}")
    t0 = time.time()
    model, config = load_model(path, device)
    params = sum(p.numel() for p in model.parameters())
    tokenizer_name = getattr(config, "tokenizer_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if name == "transformer" and project:
    professions = [
        "médecin",
        "analyste",
        "bibliothécaire",
        "comptable",
        "designer",
        "manager",
        "réceptionniste",
        "secrétaire",
    ]

    print("Applying transformer gender projection")
    project_transformer(model, tokenizer, professions)
    if name == "backpack" and ablate_sense_idx is not None:
        print(f"Ablating sense {ablate_sense_idx}")
        ablate_sense(model, ablate_sense_idx)

    results = {
    "model_name": name,
    "model_path": path,
    "device": device,
    "parameters": params,
    "sense_weighting": getattr(model, "sense_weighting", None),
    }

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
    parser.add_argument("--ablate_sense", type=int, default=None)
    parser.add_argument("--project", action="store_true")
    
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    models = {
        "backpack":"/content/drive/MyDrive/multilingual-backpacks-checkpoints_real/backpack_full/june_ckpt",
        "transformer": "/content/drive/MyDrive/multilingual-backpacks-checkpoints-real/transformer_ckpt_corrected",
    }
    selected = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results = {}
    for name in selected:
        path = models.get(name)
        if not path or not os.path.exists(os.path.join(path, "ckpt.pt")):
            print(f"Skipping {name}: no ckpt at {path}")
            continue
        all_results[name] = eval_model(
            name,
            path,
            device,
            args.data_dir,
            args.ablate_sense
           args.project
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
