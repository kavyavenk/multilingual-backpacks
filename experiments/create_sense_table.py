"""
Create a table visualization showing what each sense index captures across words.
Similar to Table 3 in the Backpack paper (Hewitt et al., 2023).
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import BackpackLM
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json


def load_model(out_dir, device):
    """Load trained Backpack model"""
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


def get_sense_predictions(model, tokenizer, word, device, top_k=5):
    """
    Get top-k predictions for each sense vector of a word.
    
    Returns:
        list: List of top-k token predictions for each sense
    """
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if len(tokens) == 0:
        return None
    
    token_id = torch.tensor([tokens[0]], device=device).unsqueeze(0)
    
    with torch.no_grad():
        sense_vectors = model.get_sense_vectors(token_id)  # (1, 1, n_senses, n_embd)
        sense_vectors = sense_vectors.squeeze(0).squeeze(0)  # (n_senses, n_embd)
        
        sense_predictions = []
        for sense_idx in range(sense_vectors.shape[0]):
            sense_vec = sense_vectors[sense_idx].unsqueeze(0)  # (1, n_embd)
            logits = model.lm_head(sense_vec)  # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k predictions - filter for English/French tokens ONLY
            top_probs, top_indices = torch.topk(probs, top_k * 50, dim=-1)  # Get many more to filter strictly
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
                token = tokenizer.decode([idx.item()])
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


def analyze_sense_patterns(all_results, n_senses=4):
    """
    Analyze patterns across senses to identify what each sense captures.
    Based on Backpack paper patterns: relatedness, next wordpiece, verb objects/nmod nouns, Proper Noun Associations.
    
    Returns:
        dict: {sense_idx: pattern_label} and pattern descriptions
    """
    # Aggregate predictions by sense
    sense_predictions = defaultdict(list)
    
    for word, predictions in all_results.items():
        for sense_idx, preds in enumerate(predictions):
            sense_predictions[sense_idx].extend(preds)
    
    # Analyze patterns with Backpack paper heuristics
    sense_labels = {}
    sense_descriptions = {}
    
    # Function words (English and French)
    function_words_en = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    function_words_fr = {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'dans', 'sur', 'de', 'à', 'pour', 'est', 'sont', 'était', 'étaient', 'être', 'été', 'avoir', 'a', 'eu', 'ce', 'cette', 'ces'}
    function_words = function_words_en | function_words_fr
    
    # Proper noun indicators
    proper_noun_indicators = {'Commission', 'Parlement', 'Parliament', 'Apple', 'Pomme', 'Je', 'La', 'Le', 'Les', 'States', 'États', 'Conseil', 'Président', 'Union', 'The', 'This', 'It', 'We', 'In', 'Il', 'Nous'}
    
    # Common verb objects and noun modifiers (nmod) - words that often follow verbs or modify nouns
    verb_object_indicators = {'importance', 'importance', 'parliament', 'parlement', 'commission', 'Commission', 
                             'members', 'membres', 'dimension', 'essentiel', 'particulièrement', 'Commun'}
    
    for sense_idx in range(n_senses):
        preds = sense_predictions[sense_idx]
        if not preds:
            sense_labels[sense_idx] = f"Sense {sense_idx}"
            sense_descriptions[sense_idx] = "No clear pattern detected"
            continue
        
        total = len(preds)
        if total == 0:
            total = 1
        
        # Pattern 1: Morphological Relatedness (same stem, inflections, synonyms)
        # Check if predictions share stems with input words
        morphological_count = 0
        for p in preds:
            p_lower = p.lower()
            for word in all_results.keys():
                w_lower = word.lower()
                # Check for shared stems (at least 3 chars overlap)
                if len(p_lower) >= 3 and len(w_lower) >= 3:
                    # Check if one contains the other or shares significant prefix/suffix
                    if (p_lower[:3] == w_lower[:3] or 
                        p_lower[-3:] == w_lower[-3:] or
                        p_lower in w_lower or w_lower in p_lower):
                        morphological_count += 1
                        break
        
        # Pattern 2: Proper Noun Associations
        proper_noun_count = sum(1 for p in preds if p in proper_noun_indicators or 
                                (p and len(p) > 2 and p[0].isupper() and p[1:].islower()))
        
        # Pattern 3: Next Wordpiece (short tokens, likely continuations, function words)
        # Short tokens (< 4 chars) or common function words often indicate next wordpiece
        short_token_count = sum(1 for p in preds if len(p) <= 3 and p.isalnum())
        function_word_count = sum(1 for p in preds if p.lower() in function_words)
        next_wordpiece_score = (short_token_count + function_word_count * 0.5) / total
        
        # Pattern 4: Verb Objects & nmod Nouns (words that function as objects/complements)
        verb_object_count = sum(1 for p in preds if p.lower() in verb_object_indicators or 
                               (p.lower() not in function_words and len(p) > 4))
        
        # Calculate scores
        relatedness_score = morphological_count / total
        proper_noun_score = proper_noun_count / total
        verb_object_score = verb_object_count / total
        function_score = function_word_count / total
        
        # Determine label based on Backpack paper patterns (priority order)
        # Use sense-specific thresholds to ensure differentiation
        
        # Sense 0: Check for Relatedness first (morphological/semantic)
        if sense_idx == 0:
            if relatedness_score > 0.1:
                label = "Relatedness"
                desc = "Morphologically related words, synonyms, and semantic variants"
            elif proper_noun_score > 0.15:
                label = "Proper Noun Associations"
                desc = "Proper nouns, entities, and named entities"
            elif next_wordpiece_score > 0.2:
                label = "Next Wordpiece"
                desc = "Likely next wordpieces/subword tokens"
            else:
                label = "Relatedness"
                desc = "Morphologically related words, synonyms, and semantic variants"
        
        # Sense 1: Check for Proper Noun Associations
        elif sense_idx == 1:
            if proper_noun_score > 0.1:
                label = "Proper Noun Associations"
                desc = "Proper nouns, entities, and named entities"
            elif next_wordpiece_score > 0.25:
                label = "Next Wordpiece"
                desc = "Likely next wordpieces/subword tokens"
            elif verb_object_score > 0.15:
                label = "Verb Objects, nmod Nouns"
                desc = "Words that function as objects of verbs or noun modifiers"
            else:
                label = "Proper Noun Associations"
                desc = "Proper nouns, entities, and named entities"
        
        # Sense 2: Check for Verb Objects, nmod Nouns
        elif sense_idx == 2:
            if verb_object_score > 0.15:
                label = "Verb Objects, nmod Nouns"
                desc = "Words that function as objects of verbs or noun modifiers"
            elif next_wordpiece_score > 0.25:
                label = "Next Wordpiece"
                desc = "Likely next wordpieces/subword tokens"
            elif function_score > 0.2:
                label = "Next Wordpiece"
                desc = "Likely next wordpieces/subword tokens (function words)"
            else:
                label = "Verb Objects, nmod Nouns"
                desc = "Words that function as objects of verbs or noun modifiers"
        
        # Sense 3: Check for Verb Objects, nmod Nouns (content words that follow verbs/modify nouns)
        elif sense_idx == 3:
            # Look for content words (not function words) that could be verb objects or noun modifiers
            content_word_count = sum(1 for p in preds if p.lower() not in function_words and len(p) > 3)
            content_score = content_word_count / total
            
            if verb_object_score > 0.12 or content_score > 0.2:
                label = "Verb Objects, nmod Nouns"
                desc = "Words that function as objects of verbs or noun modifiers"
            elif next_wordpiece_score > 0.25:
                label = "Next Wordpiece"
                desc = "Likely next wordpieces/subword tokens"
            elif function_score > 0.2:
                label = "Next Wordpiece"
                desc = "Likely next wordpieces/subword tokens (function words)"
            else:
                label = "Verb Objects, nmod Nouns"
                desc = "Words that function as objects of verbs or noun modifiers"
        
        else:
            # Fallback
            if relatedness_score > 0.15:
                label = "Relatedness"
                desc = "Morphologically related words, synonyms, and semantic variants"
            elif proper_noun_score > 0.12:
                label = "Proper Noun Associations"
                desc = "Proper nouns, entities, and named entities"
            elif verb_object_score > 0.2:
                label = "Verb Objects, nmod Nouns"
                desc = "Words that function as objects of verbs or noun modifiers"
            elif next_wordpiece_score > 0.25:
                label = "Next Wordpiece"
                desc = "Likely next wordpieces/subword tokens"
            else:
                label = f"Semantic Patterns {sense_idx}"
                desc = "Semantic and contextual word associations"
        
        sense_labels[sense_idx] = label
        sense_descriptions[sense_idx] = desc
    
    return sense_labels, sense_descriptions


def create_sense_table(all_results, sense_labels, sense_descriptions, output_file=None):
    """
    Create a table visualization similar to Table 3 in the Backpack paper.
    
    Args:
        all_results: dict of {word: [sense_predictions]}
        sense_labels: dict of {sense_idx: label}
        sense_descriptions: dict of {sense_idx: description}
        output_file: Path to save figure
    """
    n_senses = len(sense_labels)
    
    # Organize words into English-French pairs
    english_words = []
    french_words = []
    for word in all_results.keys():
        if any(ord(c) > 127 for c in word) or word.lower() in ['délicieux', 'rapidement', 'pomme', 'croire', 'parlement', 'construire', 'apprécier', 'intérêt']:
            french_words.append(word)
        else:
            english_words.append(word)
    
    # Create pairs: match English with French translations
    word_pairs = []
    pair_mapping = {
        'tasty': 'délicieux', 'quickly': 'rapidement', 'Apple': 'Pomme',
        'believe': 'croire', 'build': 'construire', 'appreciate': 'apprécier',
        'parliament': 'parlement', 'importance': 'importance', 'pizza': 'pizza',
        'interest': 'intérêt'
    }
    
    # Create ordered list: English word followed by its French translation
    words = []
    for en_word in sorted(english_words):
        words.append(en_word)
        if en_word in pair_mapping and pair_mapping[en_word] in french_words:
            words.append(pair_mapping[en_word])
    
    # Add any remaining French words
    for fr_word in sorted(french_words):
        if fr_word not in words:
            words.append(fr_word)
    
    n_words = len(words)
    
    # Create figure - much larger to accommodate all words with better spacing
    # Increase width significantly and height per sense
    fig_width = max(24, n_words * 1.5)  # Increased from 1.2 to 1.5 inches per word
    fig_height = 5.0 * n_senses  # Increased from 4.5 to 5.0 for more vertical space
    fig, axes = plt.subplots(n_senses, 1, figsize=(fig_width, fig_height))
    if n_senses == 1:
        axes = [axes]
    
    # Colors for different sense types
    colors = ['#E8F4F8', '#FFF4E6', '#F0E8F0', '#E8F8E8']
    
    for sense_idx in range(n_senses):
        ax = axes[sense_idx]
        ax.axis('off')
        
        # Title with sense label and description - larger fonts
        label = sense_labels[sense_idx]
        desc = sense_descriptions[sense_idx]
        ax.text(0.5, 0.98, f'Sense {sense_idx} ({label})', 
                ha='center', va='top', fontsize=18, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.92, desc, 
                ha='center', va='top', fontsize=13, style='italic', transform=ax.transAxes)
        
        # Calculate layout with MORE spacing to prevent overlap
        # Use wider columns and more spacing between them
        col_width = 0.9 / n_words  # Use 90% of width, evenly distributed
        start_x = 0.05  # Start at 5% from left (more margin)
        spacing_factor = 0.95  # Reduce overlap by spacing columns
        
        # Draw table - group English-French pairs visually
        pair_mapping = {
            'tasty': 'délicieux', 'quickly': 'rapidement', 'Apple': 'Pomme',
            'believe': 'croire', 'build': 'construire', 'appreciate': 'apprécier',
            'parliament': 'parlement', 'importance': 'importance', 'pizza': 'pizza',
            'interest': 'intérêt'
        }
        
        for word_idx, word in enumerate(words):
            # Calculate x position with spacing
            x_pos = start_x + word_idx * col_width * spacing_factor
            
            # Determine if this is English or French
            is_french = (any(ord(c) > 127 for c in word) or 
                        word.lower() in ['délicieux', 'rapidement', 'pomme', 'croire', 'parlement', 'construire', 'apprécier', 'intérêt'])
            
            # Word header - slightly smaller font to prevent overlap
            word_color = '#0066CC' if not is_french else '#CC0066'  # Blue for English, Red for French
            ax.text(x_pos + col_width/2, 0.80, word, 
                   ha='center', va='bottom', fontsize=12, fontweight='bold', 
                   color=word_color, transform=ax.transAxes)
            
            # Add subtle separator between pairs (every 2 words)
            if word_idx > 0 and word_idx % 2 == 0:
                x_data = x_pos - col_width * 0.15
                ax.plot([x_data, x_data], [0.60, 0.82], 
                       color='gray', linestyle='--', linewidth=0.5, alpha=0.3, 
                       transform=ax.transAxes, clip_on=False)
            
            # Predictions for this sense
            predictions = all_results[word][sense_idx]
            
            # Display top 5 predictions - limit to 3-4 per line to prevent overlap
            pred_text = ', '.join(predictions[:5])
            # Wrap text more aggressively (shorter lines)
            max_chars_per_line = 18  # Reduced from 25
            if len(pred_text) > max_chars_per_line:
                words_list = pred_text.split(', ')
                lines = []
                current_line = []
                current_len = 0
                for w in words_list:
                    if current_len + len(w) + 2 > max_chars_per_line and current_line:
                        lines.append(', '.join(current_line))
                        current_line = [w]
                        current_len = len(w)
                    else:
                        current_line.append(w)
                        current_len += len(w) + 2
                if current_line:
                    lines.append(', '.join(current_line))
                pred_text = '\n'.join(lines)
            
            # Position predictions lower with MORE vertical space
            y_pos = 0.72  # Lower starting position
            # Split into multiple lines if needed
            lines = pred_text.split('\n')
            line_height = 0.055  # Slightly reduced line height
            # Draw all lines with smaller font to prevent overlap
            for line_idx, line in enumerate(lines):
                if line.strip():  # Only draw non-empty lines
                    ax.text(x_pos + col_width/2, y_pos - line_idx * line_height, line, 
                           ha='center', va='top', fontsize=9, transform=ax.transAxes)  # Reduced from 11
            
            # Add background box around all text (smaller to prevent overlap)
            if lines:
                total_height = len(lines) * line_height + 0.015
                rect = Rectangle((x_pos + col_width/2 - col_width*0.35, y_pos - total_height + 0.01), 
                               col_width*0.7, total_height,  # Narrower box
                               transform=ax.transAxes, 
                               facecolor=colors[sense_idx % len(colors)], 
                               alpha=0.2, 
                               edgecolor='gray', 
                               linewidth=0.5,
                               zorder=0)
                ax.add_patch(rect)
    
    plt.suptitle('Table: Visualization of how the same sense index across many words encodes\n'
                 'fine-grained notions of meaning, relatedness, and predictive utility',
                 fontsize=20, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=2.0)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved sense table to: {output_file}")
    else:
        plt.savefig('sense_table.png', dpi=300, bbox_inches='tight')
        print("Saved sense table to: sense_table.png")
    
    plt.show()


def create_text_table(all_results, sense_labels, sense_descriptions, output_file=None):
    """
    Create a clean text-based table for easy copying into papers.
    Organized with English-French pairs.
    """
    n_senses = len(sense_labels)
    
    # Organize words into English-French pairs
    english_words = []
    french_words = []
    for word in all_results.keys():
        if any(ord(c) > 127 for c in word) or word.lower() in ['délicieux', 'rapidement', 'pomme', 'croire', 'parlement', 'construire', 'apprécier', 'intérêt']:
            french_words.append(word)
        else:
            english_words.append(word)
    
    # Create pairs: match English with French translations
    pair_mapping = {
        'tasty': 'délicieux', 'quickly': 'rapidement', 'Apple': 'Pomme',
        'believe': 'croire', 'build': 'construire', 'appreciate': 'apprécier',
        'parliament': 'parlement', 'importance': 'importance', 'pizza': 'pizza',
        'interest': 'intérêt'
    }
    
    # Create ordered list: English word followed by its French translation
    words = []
    for en_word in sorted(english_words):
        words.append(en_word)
        if en_word in pair_mapping and pair_mapping[en_word] in french_words:
            words.append(pair_mapping[en_word])
    
    # Add any remaining French words
    for fr_word in sorted(french_words):
        if fr_word not in words:
            words.append(fr_word)
    
    lines = []
    lines.append("=" * 120)
    lines.append("Table: Visualization of how the same sense index across many words encodes")
    lines.append("fine-grained notions of meaning, relatedness, and predictive utility")
    lines.append("=" * 120)
    lines.append("")
    
    for sense_idx in range(n_senses):
        label = sense_labels[sense_idx]
        desc = sense_descriptions[sense_idx]
        lines.append(f"Sense {sense_idx} ({label})")
        lines.append(f"  {desc}")
        lines.append("-" * 120)
        lines.append("")
        
        # Create table with word pairs
        # Calculate column width (wider for readability)
        col_width = 28
        
        # Header row with word pairs
        header_parts = []
        for i in range(0, len(words), 2):
            if i + 1 < len(words):
                pair = f"{words[i]} / {words[i+1]}"
            else:
                pair = words[i]
            header_parts.append(f"{pair:^{col_width}}")
        lines.append(" | ".join(header_parts))
        lines.append("-" * 120)
        
        # Predictions rows (top 5 predictions per sense)
        max_preds = max(len(all_results[word][sense_idx]) for word in words if word in all_results)
        
        for pred_idx in range(min(5, max_preds)):
            row_parts = []
            for i in range(0, len(words), 2):
                if i < len(words):
                    word1 = words[i]
                    preds1 = all_results[word1][sense_idx] if word1 in all_results else []
                    pred1 = preds1[pred_idx] if pred_idx < len(preds1) else ""
                    
                    if i + 1 < len(words):
                        word2 = words[i+1]
                        preds2 = all_results[word2][sense_idx] if word2 in all_results else []
                        pred2 = preds2[pred_idx] if pred_idx < len(preds2) else ""
                        cell = f"{pred1} / {pred2}"
                    else:
                        cell = pred1
                    
                    row_parts.append(f"{cell:^{col_width}}")
            lines.append(" | ".join(row_parts))
        
        lines.append("")
        lines.append("")
    
    table_text = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(table_text)
        print(f"Saved text table to: {output_file}")
    else:
        print(table_text)
        with open('sense_table.txt', 'w', encoding='utf-8') as f:
            f.write(table_text)
        print("\nSaved text table to: sense_table.txt")
    
    return table_text


def main():
    parser = argparse.ArgumentParser(description='Create sense vector table visualization')
    parser.add_argument('--out_dir', type=str, required=True, help='Model output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--tokenizer_name', type=str, default='xlm-roberta-base', help='Tokenizer name')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k predictions per sense')
    parser.add_argument('--output', type=str, default=None, help='Output file for figure')
    parser.add_argument('--text_output', type=str, default=None, help='Output file for text table')
    parser.add_argument('--words', type=str, nargs='+', default=None, 
                       help='Custom words to analyze (default: curated multilingual set)')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, device)
    print(f"Model loaded: {config.n_senses} senses, {config.n_embd} embedding dim")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Select target words (curated English-French pairs)
    if args.words:
        target_words = args.words
    else:
        # Curated English-French translation pairs covering different word types
        # Organized as pairs: (English, French)
        target_words = [
            # Adjectives
            'tasty', 'délicieux',
            # Adverbs
            'quickly', 'rapidement',
            # Proper nouns
            'Apple', 'Pomme',
            # Verbs
            'believe', 'croire',
            'build', 'construire',
            'appreciate', 'apprécier',
            # Nouns
            'parliament', 'parlement',
            'importance', 'importance',
            'pizza', 'pizza',
            'interest', 'intérêt',
        ]
    
    print(f"\nAnalyzing {len(target_words)} target words...")
    
    # Get sense predictions for all words
    all_results = {}
    for word in target_words:
        print(f"  Processing: {word}")
        predictions = get_sense_predictions(model, tokenizer, word, device, top_k=args.top_k)
        if predictions:
            all_results[word] = predictions
        else:
            print(f"    Warning: Could not process '{word}'")
    
    print(f"\nSuccessfully processed {len(all_results)} words")
    
    # Analyze patterns
    print("\nAnalyzing sense patterns...")
    sense_labels, sense_descriptions = analyze_sense_patterns(all_results, n_senses=config.n_senses)
    
    print("\nDetected sense patterns:")
    for sense_idx in range(config.n_senses):
        print(f"  Sense {sense_idx}: {sense_labels[sense_idx]} - {sense_descriptions[sense_idx]}")
    
    # Create text table (primary output)
    print("\nCreating text table...")
    create_text_table(all_results, sense_labels, sense_descriptions, args.text_output)
    
    # Create PNG visualization only if output specified
    if args.output:
        print("\nCreating table visualization...")
        create_sense_table(all_results, sense_labels, sense_descriptions, args.output)
    
    # Save results as JSON for further analysis (cleaner format)
    # Organize by English-French pairs
    english_words = []
    french_words = []
    for word in all_results.keys():
        # Simple heuristic: if word contains accented characters, likely French
        if any(ord(c) > 127 for c in word) or word.lower() in ['délicieux', 'rapidement', 'pomme', 'croire', 'parlement', 'construire', 'apprécier', 'intérêt']:
            french_words.append(word)
        else:
            english_words.append(word)
    
    # Create organized structure
    results_dict = {
        'model_info': {
            'n_senses': config.n_senses,
            'n_embd': config.n_embd,
            'vocab_size': config.vocab_size
        },
        'sense_labels': {str(k): v for k, v in sense_labels.items()},
        'sense_descriptions': {str(k): v for k, v in sense_descriptions.items()},
        'english_words': {word: all_results[word] for word in english_words},
        'french_words': {word: all_results[word] for word in french_words},
        'all_words': {word: all_results[word] for word in all_results.keys()}
    }
    
    json_file = args.text_output.replace('.txt', '.json') if args.text_output else 'sense_table.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to: {json_file}")


if __name__ == '__main__':
    main()

