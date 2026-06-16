#!/usr/bin/env python3
"""
Scaled sense-vector analysis for the Backpack model.

Runs:
  1. Monolingual sense analysis over many EN/FR words
  2. Cross-lingual sense alignment over EN–FR translation pairs
  3. Per-sense aggregation (what each sense predicts across words)

Outputs JSON suitable for the paper (Section 6).
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from evaluate import (
    analyze_cross_lingual_sense_alignment,
    analyze_sense_vectors,
    load_model,
)


# Europarl / parliamentary EN–FR pairs for cross-lingual alignment
DEFAULT_EN_FR_PAIRS = [
    ("hello", "bonjour"), ("world", "monde"), ("parliament", "parlement"),
    ("government", "gouvernement"), ("support", "soutenir"), ("proposal", "proposition"),
    ("debate", "débat"), ("law", "loi"), ("country", "pays"), ("people", "peuple"),
    ("europe", "europe"), ("union", "union"), ("commission", "commission"),
    ("member", "membre"), ("state", "état"), ("policy", "politique"),
    ("economy", "économie"), ("security", "sécurité"), ("freedom", "liberté"),
    ("democracy", "démocratie"), ("vote", "vote"), ("president", "président"),
    ("minister", "ministre"), ("report", "rapport"), ("question", "question"),
    ("answer", "réponse"), ("development", "développement"), ("environment", "environnement"),
    ("trade", "commerce"), ("market", "marché"), ("agreement", "accord"),
    ("treaty", "traité"), ("budget", "budget"), ("tax", "impôt"),
    ("worker", "travailleur"), ("industry", "industrie"), ("energy", "énergie"),
    ("climate", "climat"), ("future", "avenir"), ("today", "aujourd'hui"),
    ("important", "important"), ("necessary", "nécessaire"), ("possible", "possible"),
    ("public", "public"), ("social", "social"), ("international", "international"),
    ("regional", "régional"), ("local", "local"), ("national", "national"),
    ("right", "droit"), ("responsibility", "responsabilité"), ("cooperation", "coopération"),
]


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_multisimlex_words(data_dir, language, max_words=None):
    """Unique words from MultiSimLex translation.csv."""
    path = os.path.join(data_dir, "translation.csv")
    if not os.path.exists(path):
        return []

    lang = "ENG" if language.lower().startswith("en") else "FRA"
    df = pd.read_csv(path)
    words = set()
    for col in (f"{lang} 1", f"{lang} 2"):
        if col in df.columns:
            words.update(df[col].dropna().astype(str).tolist())
    words = sorted(words)
    if max_words:
        words = words[:max_words]
    return words


def aggregate_sense_labels(word_results, n_senses, top_k=10):
    """Aggregate top predictions per sense index across all analyzed words."""
    sense_preds = defaultdict(list)
    for word, data in word_results.items():
        for sense_idx, pred in enumerate(data["predictions"]):
            for token, prob in zip(pred["tokens"], pred["probs"]):
                sense_preds[sense_idx].append((token, prob))

    summaries = {}
    for sense_idx in range(n_senses):
        preds = sense_preds.get(sense_idx, [])
        counter = Counter()
        for token, prob in preds:
            counter[token] += prob
        top_tokens = [(t, float(c)) for t, c in counter.most_common(top_k)]
        summaries[sense_idx] = {"top_tokens": top_tokens, "n_predictions": len(preds)}
    return summaries


def summarize_monolingual(word_results):
    """Aggregate per-word metrics into language-level stats."""
    if not word_results:
        return {}

    entropies = []
    sense_sims = []
    norms = []
    for data in word_results.values():
        m = data["metrics"]
        entropies.append(m["mean_entropy"])
        sense_sims.append(m["avg_sense_similarity"])
        norms.append(m["mean_magnitude"])

    return {
        "n_words": len(word_results),
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "mean_intra_word_sense_similarity": float(np.mean(sense_sims)),
        "std_intra_word_sense_similarity": float(np.std(sense_sims)),
        "mean_sense_magnitude": float(np.mean(norms)),
    }


def summarize_cross_lingual(alignment_results):
    """Aggregate cross-lingual alignment metrics."""
    if not alignment_results:
        return {}

    avg_sims = []
    max_sims = []
    per_pair = {}
    for (en, fr), data in alignment_results.items():
        m = data["metrics"]
        avg_sims.append(m["avg_alignment_sim"])
        max_sims.append(m["max_alignment_sim"])
        per_pair[f"{en}|{fr}"] = {
            "avg_alignment_sim": m["avg_alignment_sim"],
            "max_alignment_sim": m["max_alignment_sim"],
        }

    return {
        "n_pairs": len(alignment_results),
        "mean_avg_alignment_sim": float(np.mean(avg_sims)),
        "std_avg_alignment_sim": float(np.std(avg_sims)),
        "mean_max_alignment_sim": float(np.mean(max_sims)),
        "per_pair": per_pair,
    }


def main():
    parser = argparse.ArgumentParser(description="Scaled Backpack sense analysis")
    parser.add_argument("--out_dir", default="out/backpack_full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--multisimlex_dir", default="data/multisimlex")
    parser.add_argument("--max_words_en", type=int, default=100,
                        help="Max unique EN words from MultiSimLex (0 = skip)")
    parser.add_argument("--max_words_fr", type=int, default=100,
                        help="Max unique FR words from MultiSimLex (0 = skip)")
    parser.add_argument("--max_pairs", type=int, default=50,
                        help="Max EN–FR pairs for cross-lingual alignment")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", default="out/backpack_full/scaled_sense_analysis.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}", flush=True)
    t0 = time.time()

    model, config = load_model(args.out_dir, device)
    if not hasattr(model, "get_sense_vectors"):
        raise RuntimeError("Model does not support sense vectors — use out/backpack_full")

    n_senses = getattr(config, "n_senses", 16)
    tokenizer_name = getattr(config, "tokenizer_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # --- Monolingual ---
    en_words = load_multisimlex_words(args.multisimlex_dir, "en", args.max_words_en) if args.max_words_en else []
    fr_words = load_multisimlex_words(args.multisimlex_dir, "fr", args.max_words_fr) if args.max_words_fr else []

    print(f"\nMonolingual EN: {len(en_words)} words")
    en_results = {}
    if en_words:
        en_results = analyze_sense_vectors(
            model, tokenizer, en_words, device,
            top_k=args.top_k, verbose=False,
            filter_tokens=True, analyze_relatedness=False, analyze_syntax=True,
        )

    print(f"Monolingual FR: {len(fr_words)} words")
    fr_results = {}
    if fr_words:
        fr_results = analyze_sense_vectors(
            model, tokenizer, fr_words, device,
            top_k=args.top_k, verbose=False,
            filter_tokens=True, analyze_relatedness=False, analyze_syntax=True,
        )

    all_word_results = {**en_results, **fr_results}
    sense_summaries = aggregate_sense_labels(all_word_results, n_senses, top_k=10)

    # --- Cross-lingual ---
    pairs = DEFAULT_EN_FR_PAIRS[: args.max_pairs]
    print(f"\nCross-lingual alignment: {len(pairs)} EN–FR pairs")
    alignment_results = analyze_cross_lingual_sense_alignment(
        model, tokenizer, pairs, device,
        top_k=args.top_k, verbose=False,
    )

    # --- Build report ---
    report = {
        "model_path": args.out_dir,
        "device": device,
        "n_senses": n_senses,
        "parameters": sum(p.numel() for p in model.parameters()),
        "monolingual_en_summary": summarize_monolingual(en_results),
        "monolingual_fr_summary": summarize_monolingual(fr_results),
        "cross_lingual_summary": summarize_cross_lingual(alignment_results),
        "sense_summaries": {f"sense_{k}": v for k, v in sense_summaries.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("SCALED SENSE ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(json.dumps({
        "monolingual_en": report["monolingual_en_summary"],
        "monolingual_fr": report["monolingual_fr_summary"],
        "cross_lingual": report["cross_lingual_summary"],
    }, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
