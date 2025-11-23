# Multilingual Backpack Language Models

This project implements multilingual Backpack Language Models for French-English, based on the nanoBackpackLM architecture. The project focuses on:

1. Training small Backpack models from scratch on Hansards (French-English parallel data)
2. Finetuning pre-trained Backpack models on multilingual data
3. Evaluating multilingual word representation capabilities
4. Analyzing sense vectors across languages

## Project Structure

```
.
├── data/
│   ├── europarl/          # Europarl dataset preparation
│   │   ├── prepare.py           # Main data preparation
│   │   ├── segregate_languages.py  # Create separate language files with tags
│   │   └── README.md            # Europarl-specific documentation
│   └── hansards/          # (Legacy) Hansards dataset
├── config/                # Configuration files for training
├── experiments/           # Evaluation and analysis scripts
├── model.py              # Backpack model architecture
├── train.py              # Training script
├── sample.py             # Sampling/inference script
└── evaluate.py           # Evaluation scripts

```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare Europarl dataset:
```bash
python data/europarl/prepare.py --language_pair en-fr
```

3. (Optional) Create segregated language files for reference:
```bash
python data/europarl/segregate_languages.py --language_pair en-fr --create_alignment
```

4. Train from scratch:
```bash
python train.py --config train_europarl_scratch --out_dir out-europarl-scratch --data_dir europarl
```

5. Finetune pre-trained model:
```bash
python train.py --config train_europarl_finetune --out_dir out-europarl-finetune --data_dir europarl
```

6. Evaluate:
```bash
python evaluate.py --out_dir out-europarl-scratch
```

## References

- nanoBackpackLM: https://github.com/SwordElucidator/nanoBackpackLM
- Backpack Language Models paper

# nlp-project-multilingual-backpacks
# nlp-project-multilingual-backpacks
