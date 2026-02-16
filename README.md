# JobBot: Skill Extraction and Normalization from Job Descriptions

This drive contains the code for our NLP project that automatically extracts skill mentions from job descriptions and normalizes them to the ESCO taxonomy.

## Authors
- Shruthi Hariprasad (shariprasad@umass.edu)
- Swathy Anand (swathyanand@umass.edu)

## Project Overview

JobBot is a two-stage NLP pipeline that:
1. **Extracts** skill mentions from unstructured job posting text using transformer-based NER
2. **Normalizes** extracted skills to standardized ESCO skill IDs using semantic similarity matching

**Key Results:**
- Extraction F1: 0.619 (strict), 0.797 (lenient) - **254% improvement over baseline**
- End-to-End F1: 0.366 (strict), 0.540 (lenient)
- Normalization Coverage: 67% successfully mapped to ESCO

---

## Structure

```
.
├── colab_notebooks/          # All Jupyter notebooks (run in order)
│   ├── 01_data_prep_and_sentence_sampling.ipynb
│   ├── 02_baseline_extraction_and_evaluation.ipynb
│   ├── 03_ner_bio.ipynb
│   ├── 04_distilbert_extraction_and_evaluation.ipynb
│   ├── 05_normalization.ipynb
│   ├── 06_prepare_test_for_annotation.ipynb
│   ├── 07_endtoend_evaluation.ipynb
│   ├── 08_error_analysis.ipynb
│   └── 09_generate_plots.ipynb
│
├── esco/                     # ESCO taxonomy data
│   └── skills_en.csv        # ESCO skills taxonomy (13,939 skills)
│
├── linkedin/                 # LinkedIn job postings dataset and outputs
│   ├── [LinkedIn dataset files]
│   ├── sentences_annotated_clean.csv      # Manually annotated sentences (700)
│   ├── test_gold_esco.csv                 # Test set with ESCO IDs (190 skills)
│   ├── ner_data_v3/                       # BIO-tagged data for NER training
│   ├── distilbert_runA_best/              # Fine-tuned DistilBERT model
│   ├── skill_normalization_results_v2.csv # Normalization outputs
│   ├── end_to_end_results.csv            # Final evaluation metrics
│   ├── error_analysis_summary.json        # Error analysis results
│   └── plots/                             # All generated visualizations
│
└── README.md                 # This file
```

---

## Setup and Requirements

### Prerequisites
- Python 3.8+
- Google Colab or local Jupyter environment

## How to Run

### 1. Data Preparation
```
Run: 01_data_prep_and_sentence_sampling.ipynb
```
- Loads 123,849 LinkedIn job postings
- Filters sentences likely to contain skills (keyword-based)
- Samples 1,000 sentences for annotation
- **Output:** `sentences_for_annotation_v2.csv`

### 2. Baseline Extraction
```
Run: 02_baseline_extraction_and_evaluation.ipynb
```
- Implements dictionary-based baseline using spaCy PhraseMatcher
- Evaluates on annotated data (700 sentences)
- **Results:** Strict F1 = 0.175, Lenient F1 = 0.405

### 3. Prepare NER Training Data
```
Run: 03_ner_bio.ipynb
```
- Converts annotated spans to BIO tagging format
- Splits into train/dev/test (70/15/15)
- **Output:** `ner_data_v3/` folder with BIO files

### 4. Fine-tune DistilBERT for NER
```
Run: 04_distilbert_extraction_and_evaluation.ipynb
```
- Fine-tunes DistilBERT on skill extraction task
- Trains for 8 epochs with early stopping
- Evaluates on test set
- **Results:** Strict F1 = 0.619, Lenient F1 = 0.797
- **Output:** `distilbert_runA_best/` model checkpoint

### 5. Skill Normalization
```
Run: 05_normalization.ipynb
```
- BM25 retrieval (top-20 candidates) + cross-encoder reranking
- Maps extracted skills to ESCO taxonomy
- **Results:** 67% coverage (132/197 mapped, 65 NIL)
- **Output:** `skill_normalization_results_v2.csv`

### 6. Prepare Test Set for ESCO Annotation
```
Run: 06_prepare_test_for_annotation.ipynb
```
- Extracts test set skills for manual ESCO ID annotation
- Creates annotation template
- **Output:** `test_skills_for_esco_annotation.csv`

**Manual Step:** Annotate ESCO IDs, save as `test_skills_annotated.csv`

### 7. End-to-End Evaluation
```
Run: 07_endtoend_evaluation.ipynb
```
- Evaluates complete pipeline (extraction → normalization)
- Calculates metrics with bootstrap confidence intervals
- **Results:** E2E F1 = 0.366 (strict), 0.540 (lenient)
- **Output:** `end_to_end_results.csv`, 3 plots

### 8. Error Analysis
```
Run: 08_error_analysis.ipynb
```
- Categorizes 125 extraction errors by type
- Analyzes normalization failures
- Generates error distribution visualizations
- **Output:** `error_analysis_summary.json`, 4 plots

### 9. Generate Remaining Plots
```
Run: 09_generate_plots.ipynb
```
- Creates dataset statistics plots
- Generates comparison visualizations
- **Output:** 6 additional plots for report

---

## Key Files and Outputs

### Annotated Data
- `sentences_annotated_clean.csv` - 700 manually annotated sentences with skill spans
- `test_gold_esco.csv` - 190 test skills with gold ESCO IDs (30% marked as NIL)

### Model Files
- `distilbert_runA_best/` - Fine-tuned DistilBERT checkpoint (PyTorch)
- Training: 490 sentences, Validation: 105 sentences, Test: 105 sentences

### Results Files
- `end_to_end_results.csv` - Final metrics with confidence intervals
- `skill_normalization_results_v2.csv` - All normalization mappings
- `error_analysis_summary.json` - Detailed error breakdown

### Plots (in `plots/` folder)
1. `01_domain_distribution.png` - Dataset by job domain
2. `02_sentence_lengths.png` - Token distribution
3. `03_skills_per_sentence.png` - Skill density
4. `04_extraction_comparison.png` - Baseline vs DistilBERT
5. `05_results_summary_table.png` - Complete results table
6. `06_e2e_confidence_intervals.png` - E2E metrics with CIs
7. `plot_e2e_waterfall.png` - Pipeline performance degradation
8. `plot_e2e_errors.png` - Error attribution (extraction vs normalization)
9. `plot_error_types.png` - Error type distribution
10. `plot_performance_by_domain.png` - F1 by job domain
11. `plot_error_attribution.png` - Main error sources
12. `plot_normalization_methods.png` - Normalization method breakdown

---

## Evaluation Metrics

### Extraction
- **Strict Matching:** Predicted span must exactly match gold span
- **Lenient Matching:** Overlap allowed (handles boundary errors)
- Metrics: Precision, Recall, F1

### Normalization
- **Accuracy@k:** Correct ESCO ID in top-k predictions
- **MRR:** Mean Reciprocal Rank of correct mapping
- **Coverage:** Percentage of skills successfully mapped (non-NIL)

### End-to-End
- **E2E F1:** Both span AND ESCO ID must be correct
- Computed with strict and lenient span matching
- Bootstrap confidence intervals (1000 iterations)

---

## Main Findings

### 1. Extraction Performance
- DistilBERT achieves **3.5× improvement** over dictionary baseline
- Lenient F1 of 0.797 approaches state-of-the-art
- Main error: Boundary errors (34% of failures) - e.g., "Python" vs "Python programming"

### 2. Normalization Challenges
- **33% of skills lack ESCO mappings** (modern frameworks, company-specific tools)
- Successful mappings: 52 substring matches, 44 semantic matches, 36 exact matches
- Low confidence (<0.6) on 37% of mappings

### 3. Error Analysis
- **Extraction errors (125 total):** Boundary errors (43), missed skills (38), false positives (21)
- **Normalization bottleneck:** ESCO taxonomy gaps for emerging technologies (LangChain, Next.js, etc.)
- **Domain variation:** SWE and DS perform better than DA (more standardized terminology)

### 4. End-to-End Performance
- Errors compound through pipeline stages
- Extraction errors (125) + Normalization NIL (65) → E2E F1 of 0.54
- Gap reveals the challenge of multi-stage NLP systems

---

## Limitations and Future Work

1. **ESCO Coverage:** 33% of modern skills not in taxonomy → need dynamic skill ontologies
2. **Boundary Detection:** 34% of errors from incomplete spans → add post-processing rules
3. **Abbreviations:** ML, NLP, AI often missed → augment training data with abbreviations
4. **Domain Adaptation:** Model struggles with soft skills and domain-specific jargon
5. **Computational Cost:** Fine-tuning requires GPU; consider distillation for deployment

---

## License

This project is for academic purposes as part of CS685 (Advanced NLP) at UMass Amherst.

Dataset sources:
- LinkedIn Job Postings: [Kaggle Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- ESCO Taxonomy: [European Commission](https://esco.ec.europa.eu/)
