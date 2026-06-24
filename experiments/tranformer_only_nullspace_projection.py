import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import StandardTransformerLM


def load_model(out_dir, device):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device, weights_only=False)
    config = ckpt["config"]
    model = StandardTransformerLM(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def bias_score(model, tokenizer, prompts, device, male_word="il", female_word="elle"):
    male_id = tokenizer.encode(male_word, add_special_tokens=False)[0]
    female_id = tokenizer.encode(female_word, add_special_tokens=False)[0]

    print("male token:", male_id, tokenizer.decode([male_id]))
    print("female token:", female_id, tokenizer.decode([female_id]))

    scores = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(input_ids)
            probs = F.softmax(logits[0, -1, :], dim=-1)

        p_male = float(probs[male_id].cpu())
        p_female = float(probs[female_id].cpu())

        ratio = max(
            p_male / (p_female + 1e-12),
            p_female / (p_male + 1e-12),
        )
        scores.append(ratio)

    return sum(scores) / len(scores)


def nullspace_bias_score(model, tokenizer, prompts, professions, device,
                         male_word="il", female_word="elle"):
    old_forward = model.token_embeddings.forward

    male_id = tokenizer.encode(male_word, add_special_tokens=False)[0]
    female_id = tokenizer.encode(female_word, add_special_tokens=False)[0]

    with torch.no_grad():
        E = model.token_embeddings.weight
        g = E[male_id] - E[female_id]
        g = g / (g.norm() + 1e-12)

    target_ids = []
    for word in professions:
        target_ids.extend(tokenizer.encode(word, add_special_tokens=False))
    target_ids = list(set(target_ids))

    def patched_embedding(input_ids):
        emb = old_forward(input_ids)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for tok_id in target_ids:
            mask |= input_ids == tok_id

        if mask.any():
            selected = emb[mask]
            projection = (selected @ g).unsqueeze(-1) * g
            emb[mask] = selected - projection

        return emb

    try:
        model.token_embeddings.forward = patched_embedding
        return bias_score(model, tokenizer, prompts, device, male_word, female_word)
    finally:
        model.token_embeddings.forward = old_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_name", default="xlm-roberta-base")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model(args.out_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

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

    templates = [
        "Le {profession} a dit qu'",
        "Le {profession} est entré et a dit qu'",
        "J'ai discuté avec le {profession} parce qu'",
        "Hier, le {profession} a dit qu'",
        "Le {profession} m'a dit qu'",
        "La {profession} a dit qu'",
        "La {profession} est entrée et a dit qu'",
        "J'ai discuté avec la {profession} parce qu'",
        "Hier, la {profession} a dit qu'",
        "La {profession} m'a dit qu'",
    ]

    prompts = [
        template.format(profession=profession)
        for profession in professions
        for template in templates
    ]

    print("Tokenization check")
    for w in ["il", "elle", " il", " elle"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        print(repr(w), ids, tokenizer.convert_ids_to_tokens(ids))

    baseline = bias_score(
        model, tokenizer, prompts, device,
        male_word="il",
        female_word="elle"
    )

    projected = nullspace_bias_score(
        model, tokenizer, prompts, professions, device,
        male_word="il",
        female_word="elle"
    )

    print("TRANSFORMER BASELINE:", baseline)
    print("TRANSFORMER NULLSPACE:", projected)
    print("REDUCTION:", baseline - projected)
    print("PERCENT REDUCTION:", 100 * (baseline - projected) / baseline)


if __name__ == "__main__":
    main()
