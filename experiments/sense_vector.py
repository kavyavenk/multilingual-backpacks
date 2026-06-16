"""
Sense Vector Analysis Experiment
Analyze what each sense vector represents/predicts
Based on nanoBackpackLM experiments
"""

import os
import argparse
import torch
import torch.nn.functional as F
from model import BackpackLM
from transformers import AutoTokenizer, AutoModelForCausalLM 


class SenseVectorExperiment:
    """Experiment class for analyzing sense vectors"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def sense_projection(self, word, top_k=5):
        """
        Project each sense vector through the LM head to see what it predicts.
        
        Returns:
            list: List of top-k predictions for each sense
        """
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 0:
            return []
        
        token_id = torch.tensor([tokens[0]], device=self.device).unsqueeze(0)
        #sense_vectors = self.model.wte(token_id)  # (1,1,embd_dim)
        sense_vectors = self.model.get_sense_vectors(token_id)  # (1, 1, n_senses, n_embd)
        sense_vectors = sense_vectors.squeeze(0).squeeze(0)  # (n_senses, n_embd)
        
        sense_predictions = []
        with torch.no_grad():
            for sense_idx in range(sense_vectors.shape[0]):
                sense_vec = sense_vectors[sense_idx].unsqueeze(0)  # (1, n_embd)
                logits = self.model.lm_head(sense_vec)  # (1, vocab_size)
                    
                # Get top-k predictions - filter for English/French tokens ONLY
                top_logits, top_indices = torch.topk(logits, top_k * 50, dim=-1)  # Get many more to filter strictly
                top_tokens = []
                seen = set()
                
                # Expanded English/French words and patterns
                english_french_patterns = set([
                    # Articles
                    'the', 'a', 'an', 'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cette', 'ces',
                    # Conjunctions
                    'and', 'or', 'but', 'et', 'ou', 'mais',
                    # Prepositions
                    'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'de', 'dans', 'sur', 'à', 'pour', 'avec', 'par',
                    # Verbs
                    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                    'est', 'sont', 'était', 'étaient', 'être', 'été', 'avoir', 'a', 'eu',
                    # Pronouns
                    'we', 'you', 'they', 'it', 'this', 'that', 'these', 'those',
                    'nous', 'vous', 'ils', 'elles', 'il', 'elle', 'ce', 'cette',
                    # Common nouns
                    'parliament', 'parlement', 'commission', 'Commission', 'Parlement',
                    'importance', 'importance', 'pizza', 'Pizza',
                    'interest', 'intérêt', 'believe', 'croire',
                    'build', 'construire', 'appreciate', 'apprécier',
                    'tasty', 'délicieux', 'quickly', 'rapidement',
                    'Apple', 'Pomme', 'President', 'Président',
                    # Other common words
                    'the', 'and', 'that', 'this', 'les', 'le', 'la', 'de', 'des', 'du',
                    'Je', 'Nous', 'The', 'Les', 'Le', 'La', 'States', 'États', 'Conseil',
                    'members', 'membres', 'essentiel', 'particulièrement', 'Commun', 'dimension'
                ])
                
                for idx in top_indices[0]:
                    token = self.tokenizer.decode([idx.item()])
                    # Clean up token
                    token = token.strip()
                    
                    # Filter criteria:
                    # 1. Must be printable and meaningful length
                    # 2. Not a subword token (no _ or ## prefix)
                    # 3. Not already seen
                    # 4. Either: (a) matches English/French patterns, (b) contains only ASCII/Latin chars, or (c) contains French accents
                    if (token and 
                        len(token) > 1 and 
                        not token.startswith('_') and 
                        not token.startswith('##') and
                        token not in seen and
                        token.isprintable()):
                        
                        # Very strict English/French filtering - prioritize known words
                        token_lower = token.lower()
                        token_clean = token_lower.strip(".,!?;:'\"()[]{}")
                        
                        # French accent characters
                        french_accents = 'àáâãäåèéêëìíîïòóôõöùúûüýÿçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÇ'
                        
                        # Check if it's a known English/French word (highest priority)
                        is_known_word = (token_clean in english_french_patterns or 
                                       token in english_french_patterns or
                                       token_lower in english_french_patterns)
                        
                        # If not known, check if it's clearly English/French pattern
                        if not is_known_word:
                            # Must contain only ASCII letters or French accents
                            has_only_valid_chars = all(
                                c.isalpha() or c in french_accents or c in " '-" 
                                for c in token
                            )
                            
                            # Must not contain any non-Latin characters (except French accents)
                            has_no_foreign_scripts = not any(
                                ord(c) > 0x024F and c not in french_accents 
                                for c in token
                            )
                            
                            # Must be at least 2 characters and contain letters
                            has_letters = any(c.isalpha() or c in french_accents for c in token) and len(token_clean) >= 2
                            
                            # Check if it looks like an English/French word (all lowercase/uppercase, no mixed scripts)
                            looks_english_french = (
                                has_only_valid_chars and 
                                has_no_foreign_scripts and 
                                has_letters and
                                # Additional check: no suspicious character combinations
                                not any(ord(c) > 127 and c not in french_accents for c in token)
                            )
                        else:
                            looks_english_french = True
                        
                        # STRICT: Only accept known English/French words (no guessing)
                        if is_known_word:
                            top_tokens.append(token)
                            seen.add(token)
                            if len(top_tokens) >= top_k:
                                break
                
                sense_predictions.append(top_tokens)
        
        return sense_predictions
    
    def analyze_multilingual_senses(self, word_pairs):
        """
        Analyze sense vectors for translation pairs to see if senses align.
        
        Args:
            word_pairs: List of (word_en, word_fr) tuples
        """
        results = {}
        
        for word_en, word_fr in word_pairs:
            en_senses = self.sense_projection(word_en)
            fr_senses = self.sense_projection(word_fr)
            
            results[(word_en, word_fr)] = {
                'en': en_senses,
                'fr': fr_senses,
            }
        
        return results
    def sweep_ablate_senses(self, prompt, target_word, test_words):
        old_forward = self.model.sense_layer.forward
    
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
        print("prompt ids:", prompt_ids)
        print("target ids:", target_ids)
        print("decoded prompt tokens:",
              [self.tokenizer.decode([i]) for i in prompt_ids])
        
        test_ids = [
            self.tokenizer.encode(w, add_special_tokens=False)[0]
            for w in test_words
        ]
    
        n_senses = self.model.n_senses
        n_embd = self.model.config.n_embd
    
        def run_once(ablate_idx=None):
            def patched_sense_layer(token_embs):
                out = old_forward(token_embs)
                B, T, _ = out.shape
                out = out.view(B, T, n_senses, n_embd)
    
                if ablate_idx is not None:
                    # identify which token embeddings correspond to target ids
                    target_token_embs = self.model.token_embedding(
                        torch.tensor(target_ids, device=self.device)
                    )
    
                    for target_emb in target_token_embs:
                        mask = torch.isclose(
                            token_embs, target_emb.view(1, 1, -1),
                            atol=1e-6
                        ).all(dim=-1)
    
                        positions = mask.nonzero(as_tuple=False)
                        print(
                            f"ablate_idx={ablate_idx}, "
                            f"matches={len(positions)}"
                        )
                        
                        for b, pos in positions:
                            out[b, pos, ablate_idx, :] = 0.0
    
                return out.view(B, T, n_senses * n_embd)
    
            self.model.sense_layer.forward = patched_sense_layer
    
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
    
            with torch.no_grad():
                logits, _ = self.model(input_ids)
                probs = F.softmax(logits[0, -1, :], dim=-1)
    
            return {
                w: float(probs[tok_id].detach().cpu())
                for w, tok_id in zip(test_words, test_ids)
            }
    
        try:
            baseline = run_once(ablate_idx=None)
    
            results = []
            for sense_idx in range(n_senses):
                ablated = run_once(ablate_idx=sense_idx)
    
                diff = {
                    w: ablated[w] - baseline[w]
                    for w in test_words
                }
    
                total_abs_change = sum(abs(v) for v in diff.values())
    
                results.append({
                    "sense": sense_idx,
                    "baseline": baseline,
                    "ablated": ablated,
                    "diff": diff,
                    "total_abs_change": total_abs_change,
                })
    
        finally:
            self.model.sense_layer.forward = old_forward
    
        return sorted(results, key=lambda x: x["total_abs_change"], reverse=True)
    def bias_score(self, prompts, male_word="he", female_word="she"):
            male_id = self.tokenizer.encode(male_word, add_special_tokens=False)[0]
            female_id = self.tokenizer.encode(female_word, add_special_tokens=False)[0]
        
            print("male token:", male_id, self.tokenizer.decode([male_id]))
            print("female token:", female_id, self.tokenizer.decode([female_id]))
        
            scores = []
        
            for prompt in prompts:
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        
                with torch.no_grad():
                    logits, _ = self.model(input_ids)
                    probs = F.softmax(logits[0, -1, :], dim=-1)
        
                p_male = float(probs[male_id].detach().cpu())
                p_female = float(probs[female_id].detach().cpu())
        
                ratio = max(
                    p_male / (p_female + 1e-12),
                    p_female / (p_male + 1e-12)
                )
        
                scores.append(ratio)
        
                print(prompt)
                print(f"  p({male_word}) = {p_male:.8g}")
                print(f"  p({female_word}) = {p_female:.8g}")
                print(f"  bias ratio = {ratio:.4f}")
        
            return sum(scores) / len(scores)

    def sweep_debias_senses(self, prompts, target_words, male_word=" he", female_word=" she"):
        baseline = self.bias_score(prompts, male_word, female_word)
    
        results = []
    
        for sense_idx in range(self.model.n_senses):
            old_forward = self.model.sense_layer.forward
    
            def patched_sense_layer(token_embs):
                out = old_forward(token_embs)
                B, T, _ = out.shape
                out = out.view(B, T, self.model.n_senses, self.model.config.n_embd)
    
                all_target_ids = []
                for word in target_words:
                    all_target_ids.extend(self.tokenizer.encode(word, add_special_tokens=False))
    
                target_token_embs = self.model.token_embedding(
                    torch.tensor(all_target_ids, device=self.device)
                )
    
                for target_emb in target_token_embs:
                    mask = torch.isclose(
                        token_embs,
                        target_emb.view(1, 1, -1),
                        atol=1e-6
                    ).all(dim=-1)
    
                    for b, pos in mask.nonzero(as_tuple=False):
                        out[b, pos, sense_idx, :] = 0.0
    
                return out.view(B, T, self.model.n_senses * self.model.config.n_embd)
    
            try:
                self.model.sense_layer.forward = patched_sense_layer
                score = self.bias_score(prompts, male_word, female_word)
            finally:
                self.model.sense_layer.forward = old_forward
    
            results.append({
                "sense": sense_idx,
                "bias_score": score,
                "reduction": baseline - score,
            })
    
        return baseline, sorted(results, key=lambda x: x["reduction"], reverse=True)


def load_model(out_dir, device):
    """Load trained model"""
    import pickle
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = BackpackLM(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, config




def main():
    parser = argparse.ArgumentParser(description='Analyze sense vectors')
    parser.add_argument('--out_dir', type=str, required=True, help='Model output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--tokenizer_name', type=str, default='xlm-roberta-base', help='Tokenizer name')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    '''
    model_name = "stanfordnlp/backpack-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    '''
        
    # Load model
    print(f"Loading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, device)
    
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Initialize experiment
    ex = SenseVectorExperiment(model, tokenizer, device)
    
    # Test words
    print("\n=== English Word Sense Analysis ===")
    english_words = [' hello', ' world', ' language', ' model', ' learning']
    for word in english_words:
        print(f"\n# {word}")
        predictions = ex.sense_projection(word)
        for sense_idx, preds in enumerate(predictions):
            print(f"Sense {sense_idx}: {preds}")
    
    print("\n=== French Word Sense Analysis ===")
    french_words = [' bonjour', ' monde', ' langue', ' modèle', ' apprentissage']
    for word in french_words:
        print(f"\n# {word}")
        predictions = ex.sense_projection(word)
        for sense_idx, preds in enumerate(predictions):
            print(f"Sense {sense_idx}: {preds}")
    
    # Multilingual analysis
    print("\n=== Multilingual Sense Comparison ===")
    translation_pairs = [
        ('hello', 'bonjour'),
        ('world', 'monde'),
        ('language', 'langue'),
        ('model', 'modèle'),
    ]
    
    multilingual_results = ex.analyze_multilingual_senses(translation_pairs)
    for (word_en, word_fr), results in multilingual_results.items():
        print(f"\n{word_en} <-> {word_fr}:")
        print(f"  English senses:")
        for sense_idx, preds in enumerate(results['en']):
            print(f"    Sense {sense_idx}: {preds}")
        print(f"  French senses:")
        for sense_idx, preds in enumerate(results['fr']):
            print(f"    Sense {sense_idx}: {preds}")

    print("\n=== Sweep Sense Ablation ===")

    results = ex.sweep_ablate_senses(
        prompt="bonjour",
        target_word="bonjour",
        test_words=[
            "bonjour",
            "monde",
            "langue",
            "modèle"
        ]
    )
    print("\n=== Bias Score Test ===")

    professions = [
        "mechanic", "accountant", "farmer", "baker", "assistant", "construction",
        "guard", "carpenter", "analyst", "physician", "cook", "clerk",
        "manager", "developer", "librarian", "salesperson", "mover",
        "hairdresser", "auditor", "sheriff", "janitor", "driver", "chief",
        "teacher", "writer", "receptionist", "CEO", "nurse", "housekeeper",
        "secretary", "counselor", "attendant", "supervisor", "designer",
        "lawyer", "editor", "cashier", "laborer", "tailor", "cleaner"
    ]

    prompts = [
        f"The {profession} said that"
        for profession in professions
    ]

    avg_bias = ex.bias_score(
        prompts,
        male_word=" he",
        female_word=" she"
    )

    print("AVERAGE BIAS SCORE:", avg_bias)
    
    
    for r in results[:10]:
        print("=" * 60)
        print("sense:", r["sense"])
        print("total_abs_change:", r["total_abs_change"])
        print("ablated:", r["ablated"])
        print("baseline:", r["baseline"])
        print("diff:", r["diff"])

    print("\n=== Debias Sense Sweep ===")

    baseline, debias_results = ex.sweep_debias_senses(
        prompts=prompts,
        target_words=professions,
        male_word=" he",
        female_word=" she"
    )
    
    print("BASELINE:", baseline)
    
    for r in debias_results:
        print(r)

    

if __name__ == '__main__':
    main()
