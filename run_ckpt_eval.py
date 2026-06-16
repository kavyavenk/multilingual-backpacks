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
    evaluate_cross_lingual_multisimlex,
    evaluate_multisimlex,
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


def embedding_cosine_stats(model, tokenizer, words, device):
    """Mean off-diagonal cosine similarity between word embeddings."""
    embs = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            continue
        input_ids = torch.tensor([ids[0]], device=device).unsqueeze(0)
        with torch.no_grad():
            if hasattr(model, "get_backpack_mixed_embeddings"):
                vec = model.get_backpack_mixed_embeddings(input_ids).mean(dim=1).squeeze(0)
            elif hasattr(model, "token_embedding"):
                vec = model.token_embedding(input_ids).squeeze(0).squeeze(0)
            elif hasattr(model, "token_embeddings"):
                vec = model.token_embeddings(input_ids).squeeze(0).squeeze(0)
            else:
                continue
            vec = torch.nn.functional.normalize(vec, dim=0)
            embs.append(vec.cpu().numpy())

    if len(embs) < 2:
        return {"n_words": len(embs), "mean_cosine": None}

    mat = np.stack(embs)
    sims = mat @ mat.T
    n = len(embs)
    off_diag = sims[~np.eye(n, dtype=bool)]
    return {
        "n_words": n,
        "mean_cosine": float(off_diag.mean()),
        "std_cosine": float(off_diag.std()),
        "min_cosine": float(off_diag.min()),
        "max_cosine": float(off_diag.max()),
    }


def normalize_multisimlex(result):
    if result is None:
        return None
    out = dict(result)
    if "spearman" in out and "correlation" not in out:
        out["correlation"] = out["spearman"]
    return out


def eval_model(name, path, device, data_dir, max_multisimlex=None, multisimlex_dir="data/multisimlex"):
    print(f"\n{'='*70}\nEVALUATING: {name}\n{'='*70}")
    t0 = time.time()
    model, config = load_model(path, device)
    params = sum(p.numel() for p in model.parameters())
    tokenizer_name = getattr(config, "tokenizer_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    results = {
        "model_name": name,
        "model_path": path,
        "device": device,
        "parameters": params,
        "sense_weighting": getattr(model, "sense_weighting", None),
    }

    sample_words = [
        "hello", "world", "parliament", "support", "proposal",
        "bonjour", "monde", "parlement", "soutenir", "proposition",
        "car", "dog", "bank", "money", "star",
    ]
    results["embedding_stats"] = embedding_cosine_stats(model, tokenizer, sample_words, device)
    print(f"Embedding stats: {results['embedding_stats']}")

    for lang in ("en", "fr"):
        print(f"\nMultiSimLex {lang}...")
        r = normalize_multisimlex(
            evaluate_multisimlex(model, tokenizer, device, language=lang, max_samples=max_multisimlex)
        )
        results[f"multisimlex_{lang}"] = r
        if r and r.get("correlation") is not None:
            print(f"  Spearman={r['correlation']:.4f} (n={r['num_pairs']}, p={r['p_value']:.4f})")

    print("\nMultiSimLex cross-lingual...")
    cross = normalize_multisimlex(
        evaluate_cross_lingual_multisimlex(
            model, tokenizer, device, max_samples=max_multisimlex, data_dir=multisimlex_dir
        )
    )
    results["multisimlex_cross"] = cross
    if cross and cross.get("correlation") is not None:
        print(f"  Spearman={cross['correlation']:.4f} (n={cross.get('n_pairs')}, method={cross.get('method')})")

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
    parser.add_argument("--max_multisimlex", type=int, default=None)
    parser.add_argument("--multisimlex_dir", default="data/multisimlex")
    parser.add_argument("--models", default="backpack,transformer",
                        help="Comma-separated: backpack, transformer")
    parser.add_argument("--out", default="out/ckpt_eval_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    models = {
        "backpack": "out/backpack_full",
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
        emb = r.get("embedding_stats", {})
        en = r.get("multisimlex_en", {}) or {}
        fr = r.get("multisimlex_fr", {}) or {}
        cross = r.get("multisimlex_cross", {}) or {}
        sent = r.get("sentence_similarity", {}) or {}
        ppl = r.get("perplexity", {}) or {}
        print(
            f"{name:12s} | emb_cos={emb.get('mean_cosine', 'n/a'):>6} "
            f"| EN ρ={en.get('correlation', 'n/a')} "
            f"| FR ρ={fr.get('correlation', 'n/a')} "
            f"| XL ρ={cross.get('correlation', 'n/a')} "
            f"| μ_trans={sent.get('mu_trans', 'n/a')} "
            f"| PPL={ppl.get('perplexity', 'n/a')}"
        )


if __name__ == "__main__":
    main()
