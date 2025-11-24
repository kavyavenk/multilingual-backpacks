# Figures for Milestone Report

This directory contains all figures generated for the milestone report.

## Generated Figures

1. **backpack_architecture.pdf**
   - Architecture diagram showing the Backpack Language Model structure
   - Illustrates the flow from input tokens through sense embeddings, sense predictor, weighted combination, transformer blocks, to output
   - Used in Figure 1 of the report

2. **training_curves.pdf**
   - Example training and validation loss curves
   - Shows convergence behavior during training
   - Used in Figure 2 of the report

3. **sense_vectors.pdf**
   - PCA visualization of sense vectors for multiple words
   - Shows how sense vectors cluster for different words
   - Demonstrates that each word has 16 sense vectors
   - Used in Figure 3 of the report

4. **cross_lingual_similarity.pdf**
   - Heatmap showing cosine similarity between English and French words
   - Translation pairs show higher similarity scores
   - Used in Figure 4 of the report

5. **multilingual_senses.pdf**
   - Comparison of English (blue circles) and French (red squares) sense vectors
   - Shows sense vectors for translation pairs side-by-side
   - Used to assess cross-lingual sense alignment
   - Used in Figure 5 of the report

## Regenerating Figures

To regenerate these figures, run:

```bash
python experiments/generate_report_figures.py
```

The script will create all figures in PDF format suitable for LaTeX inclusion.

## Note

These figures use simulated/example data for demonstration purposes. When actual training results are available, the figures can be regenerated with real data by modifying the `generate_report_figures.py` script to load actual model checkpoints and evaluation results.

