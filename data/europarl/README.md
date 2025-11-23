# Europarl Dataset Preparation

This directory contains scripts for preparing the Europarl parallel corpus for multilingual Backpack training.

## Files

- `prepare.py`: Main data preparation script (creates train.bin, val.bin)
- `segregate_languages.py`: Segregates parallel sentences into separate language files with tags
- `README.md`: This file

## Usage

### 1. Prepare Training Data

Prepare the dataset for training (creates binary files):

```bash
python data/europarl/prepare.py --language_pair en-fr
```

This creates:
- `train.bin`: Training tokens
- `val.bin`: Validation tokens
- `meta.pkl`: Metadata (vocab size, languages, etc.)

### 2. Segregate Languages (Reference Files)

Create separate language files with tags for reference:

```bash
python data/europarl/segregate_languages.py --language_pair en-fr
```

This creates:
- `en.txt`: English sentences with tags
- `fr.txt`: French sentences with tags
- `metadata.json`: Dataset metadata

Optional: Create alignment file:
```bash
python data/europarl/segregate_languages.py --language_pair en-fr --create_alignment
```

This also creates:
- `alignment.json`: Maps sentence IDs to parallel sentences

## Tag Format

Each line in the language files follows this format:

```
<sentence_id>|<language_pair>|<source>\t<sentence_text>
```

Example:
```
0|en-fr|europarl	This is an English sentence.
0|en-fr|europarl	Ceci est une phrase française.
```

The tag allows you to:
- Match parallel sentences across languages (same sentence_id)
- Identify the language pair
- Track the source dataset

## Language Files Structure

### English file (en.txt)
```
0|en-fr|europarl	This is the first English sentence.
1|en-fr|europarl	This is the second English sentence.
...
```

### French file (fr.txt)
```
0|en-fr|europarl	Ceci est la première phrase française.
1|en-fr|europarl	Ceci est la deuxième phrase française.
...
```

## Alignment File Structure

The alignment.json file contains:
```json
[
  {
    "id": 0,
    "en": "This is an English sentence.",
    "fr": "Ceci est une phrase française.",
    "language_pair": "en-fr"
  },
  ...
]
```

## Supported Language Pairs

Europarl supports many language pairs. Common ones include:
- `en-fr` (English-French)
- `en-de` (English-German)
- `en-es` (English-Spanish)
- `en-it` (English-Italian)
- `fr-de` (French-German)
- And many more...

## Reference Use Cases

The segregated language files can be used for:

1. **Analysis**: Analyze sentence-level patterns in each language
2. **Evaluation**: Create evaluation sets with known parallel pairs
3. **Visualization**: Visualize parallel sentence pairs
4. **Debugging**: Check data quality and alignment
5. **Research**: Study cross-lingual patterns

## Example: Reading Segregated Files

### Using the Helper Script

```bash
# Show statistics and example pairs
python data/europarl/read_segregated.py --data_dir data/europarl --show_pairs 5

# Get specific sentence by ID
python data/europarl/read_segregated.py --data_dir data/europarl --sentence_id 0
```

### Using Python API

```python
from data.europarl.read_segregated import SegregatedDataReader

# Initialize reader
reader = SegregatedDataReader('data/europarl', 'en-fr')

# Get statistics
stats = reader.get_statistics()
print(stats)

# Get parallel pairs
pairs = reader.get_parallel_pairs(limit=10)
for sent_id, en_text, fr_text in pairs:
    print(f"ID {sent_id}:")
    print(f"  EN: {en_text}")
    print(f"  FR: {fr_text}")

# Get specific sentence
pair = reader.search_by_id('0')
if pair:
    en_text, fr_text = pair
    print(f"EN: {en_text}")
    print(f"FR: {fr_text}")
```

### Manual Reading

```python
# Read English sentences
with open('data/europarl/en.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tag, sentence = line.strip().split('\t', 1)
        sentence_id, lang_pair, source = tag.split('|')
        print(f"Sentence {sentence_id}: {sentence}")

# Match parallel sentences
en_sentences = {}
fr_sentences = {}

with open('data/europarl/en.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tag, sentence = line.strip().split('\t', 1)
        sentence_id = tag.split('|')[0]
        en_sentences[sentence_id] = sentence

with open('data/europarl/fr.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tag, sentence = line.strip().split('\t', 1)
        sentence_id = tag.split('|')[0]
        fr_sentences[sentence_id] = sentence

# Print parallel pairs
for sent_id in list(en_sentences.keys())[:5]:
    print(f"\nPair {sent_id}:")
    print(f"  EN: {en_sentences[sent_id]}")
    print(f"  FR: {fr_sentences[sent_id]}")
```

