# What Each of the 16 Senses Represents

Based on analysis of predictions, syntactic patterns, and activation patterns:

## Key Finding: Senses Are Highly Similar

**Most senses predict `sep` (language separator) as their top token**, reflecting that the model was trained on parallel English-French data with language separators. This suggests the senses haven't fully specialized.

## Sense Breakdown

### **Sense 0: Risk/Debate + Structural**
- **Top predictions**: `sep`, `risque` (risk), `manque` (lack), `débat` (debate), `seul` (alone)
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `COM`, `Fund`)
- **Represents**: Language separator + risk/debate contexts
- **Activation**: General parliamentary/debate contexts

### **Sense 1: Risk/Debate + Structural** (Similar to Sense 0)
- **Top predictions**: `sep`, `risque`, `débat`, `manque`, `seul`
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `Fund`, `COM`)
- **Represents**: Language separator + risk/debate contexts
- **Note**: Very similar to Sense 0 (redundant)

### **Sense 2: Position/Aspect + English Structure**
- **Top predictions**: `sep`, `côté` (side), `ster`, `s`, `cy`
- **Syntactic patterns**: Articles (`La`), Prepositions (`By`, `du`), Proper nouns (`First`, `States`, `Minister`)
- **Represents**: Different aspects/sides + English structural elements
- **Distinction**: More English proper nouns and articles

### **Sense 3: Lack/Risk + Structural**
- **Top predictions**: `sep`, `manque` (lack), `risque` (risk), `débat` (debate), `seul` (alone)
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `Fund`, `COM`)
- **Represents**: Language separator + absence/risk contexts
- **Note**: Similar to Senses 0, 1 (redundant)

### **Sense 4: Temporal + Small Scale**
- **Top predictions**: `sep`, `che`, `hui` (from "aujourd'hui" = today), `petits` (small), `risque`
- **Syntactic patterns**: Prepositions (`with`, `de`), Proper nouns (`Union`)
- **Represents**: Time-related + small-scale contexts
- **Distinction**: More temporal references

### **Sense 5: Risk + Future + Union**
- **Top predictions**: `sep`, `risque`, `seul`, `élé`, `da`
- **Syntactic patterns**: Articles (`La`), Prepositions (`with`, `By`), Proper nouns (`Union`)
- **Represents**: Risk + future contexts + European Union references
- **Distinction**: Strong association with "Union" (European Union)

### **Sense 6: Debate + Lack**
- **Top predictions**: `sep`, `manque`, `débat`, `ster`, `côté`
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `Fund`, `COM`)
- **Represents**: Debate + absence contexts
- **Note**: Similar to other debate senses

### **Sense 7: Progress + Risk**
- **Top predictions**: `sep`, `manque`, `risque`, `débat`, `progrès` (progress)
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `Fund`, `COM`)
- **Represents**: Progress/development + risk contexts
- **Distinction**: Only sense with "progrès" in top predictions

### **Sense 8: English Structure + Position**
- **Top predictions**: `sep`, `côté`, `By`, `ster`, `First`
- **Syntactic patterns**: Articles (`La`), Prepositions (`By`, `À`), Proper nouns (`First`, `Will`, `Minister`)
- **Represents**: English structural elements + positional aspects
- **Distinction**: Strong English proper nouns (`First`, `Will`, `Minister`)

### **Sense 9: Risk + Future**
- **Top predictions**: `sep`, `risque`, `seul`, `manque`, `crai`
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Roy`, `Fond`)
- **Represents**: Risk + future concerns
- **Note**: Similar to Senses 12, 13, 15

### **Sense 10: Risk/Debate + Structural**
- **Top predictions**: `sep`, `risque`, `manque`, `débat`, `seul`
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `COM`, `Fund`)
- **Represents**: Language separator + risk/debate contexts
- **Note**: Very similar to Senses 0, 1, 3 (redundant)

### **Sense 11: Lack/Risk + Structural**
- **Top predictions**: `sep`, `manque`, `risque`, `débat`, `seul`
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `Fund`, `COM`)
- **Represents**: Language separator + absence/risk contexts
- **Note**: Very similar to other senses (redundant)

### **Sense 12: Risk + Problem**
- **Top predictions**: `sep`, `risque`, `seul`, `crai`, `problème` (problem)
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Roy`, `Fond`)
- **Represents**: Problems + risks + future concerns
- **Distinction**: Only sense with "problème" in top predictions

### **Sense 13: Risk + Future**
- **Top predictions**: `sep`, `risque`, `seul`, `crai`, `manque`
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Roy`, `Fond`)
- **Represents**: Risk + future concerns
- **Note**: Very similar to Senses 9, 12, 15

### **Sense 14: Risk/Debate + States**
- **Top predictions**: `sep`, `risque`, `manque`, `débat`, `seul`
- **Syntactic patterns**: Prepositions (`By`), Proper nouns (`Fond`, `States`, `Fund`)
- **Represents**: Language separator + risk/debate + US references
- **Distinction**: Strong association with "States" (United States)

### **Sense 15: Risk + Union**
- **Top predictions**: `sep`, `risque`, `seul`, `crai`, `élé`
- **Syntactic patterns**: Articles (`La`), Prepositions (`By`, `with`), Proper nouns (`Union`)
- **Represents**: Risk + future + European Union references
- **Distinction**: Strong association with "Union" (European Union)

## Summary: Sense Categories

### **1. Language Separator / Structural Senses** (Most Common)
- **Senses**: 0, 1, 3, 6, 7, 10, 11, 14
- **Pattern**: Predict `sep` + risk/debate/lack tokens
- **Role**: Language separation + general parliamentary discourse
- **Status**: ⚠️ **Highly redundant** - 8 senses doing similar things

### **2. English Structural Senses**
- **Senses**: 2, 8
- **Pattern**: Predict `sep` + English proper nouns (`First`, `Will`, `Minister`, `States`)
- **Role**: English structural elements, formal language
- **Status**: ✅ **Somewhat distinct**

### **3. Temporal / Small Scale**
- **Sense**: 4
- **Pattern**: Predict `sep` + temporal tokens (`hui` = today, `petits` = small)
- **Role**: Time references, small-scale contexts
- **Status**: ✅ **Somewhat distinct**

### **4. European Union / Union References**
- **Senses**: 5, 15
- **Pattern**: Predict `sep` + `Union` (European Union)
- **Role**: EU-specific contexts
- **Status**: ✅ **Somewhat distinct**

### **5. Progress / Development**
- **Sense**: 7
- **Pattern**: Predict `sep` + `progrès` (progress)
- **Role**: Progress/development contexts
- **Status**: ✅ **Somewhat distinct**

### **6. Problem / Issue**
- **Sense**: 12
- **Pattern**: Predict `sep` + `problème` (problem)
- **Role**: Problem-solving contexts
- **Status**: ✅ **Somewhat distinct**

### **7. Future / Risk**
- **Senses**: 9, 13
- **Pattern**: Predict `sep` + risk + future tokens (`crai`)
- **Role**: Future-oriented risk contexts
- **Status**: ⚠️ **Redundant** - similar to each other

## Key Insights

### **Redundancy Problem**
- **8 out of 16 senses** (50%) are highly similar (Senses 0, 1, 3, 6, 7, 10, 11, 14)
- They all predict `sep` + similar risk/debate/lack tokens
- **Suggestion**: Model might benefit from fewer senses (8-10 instead of 16)

### **Distinct Senses**
- **Sense 2**: English structure + proper nouns
- **Sense 4**: Temporal references
- **Sense 5**: European Union references
- **Sense 7**: Progress/development
- **Sense 8**: English formal structure
- **Sense 12**: Problems/issues
- **Sense 15**: European Union references

### **Why Senses Are Similar**

1. **Training Data**: Europarl data has strong patterns (language separators, parliamentary language)
2. **No Diversity Loss**: Model wasn't trained to encourage sense diversity
3. **Position-Based Activation**: Sense weights are predicted from position embeddings, not context
4. **Limited Specialization**: Senses haven't learned to specialize for different contexts

## Recommendations

1. **Reduce Number of Senses**: Try 8-10 senses instead of 16
2. **Add Diversity Loss**: Encourage senses to be different
3. **Context-Based Activation**: Use actual context (not just position) to predict sense weights
4. **Explicit Specialization**: Train senses to specialize (e.g., one for proper nouns, one for verbs, etc.)

## How to Improve Sense Diversity

```python
# Add diversity loss during training
diversity_loss = -torch.mean(torch.var(sense_weights, dim=-1))  # Encourage different weights
total_loss = language_model_loss + lambda_diversity * diversity_loss
```

Or use different initialization strategies to encourage specialization from the start.
