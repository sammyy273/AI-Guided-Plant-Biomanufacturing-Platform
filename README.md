# AI-Guided Plant Biomanufacturing Platform

An AI-driven computational platform for optimizing recombinant protein production in plants through promoter engineering, genome editing, and machine learning.

The platform integrates biological sequence engineering with predictive machine learning to identify regulatory sequences that improve transgene expression in plant-based biomanufacturing systems.

---

## Overview

Recombinant therapeutic proteins are traditionally produced in mammalian cell culture, which is expensive and difficult to scale. Plant molecular farming provides an alternative platform, but protein yield is influenced by numerous genetic and regulatory factors.

This project investigates how artificial intelligence can be combined with computational biology to optimize multiple stages of recombinant protein production, including promoter engineering, codon optimization, and genome editing.

Target host:
**Nicotiana benthamiana**

Target protein:
**Human PH20 Hyaluronidase**

---

## Features

- CRISPR guide RNA design
- Codon optimization
- AI-assisted promoter generation
- Machine learning prediction of promoter strength
- Feature engineering from DNA sequences
- Cross-species promoter benchmarking
- Automated computational workflow

---

## Repository Structure

```
AI-Guided-Plant-Biomanufacturing-Platform
│
├── data/
│   └── promoter datasets
│
├── figures/
│
├── outputs/
│
├── scripts/
│
├── v2_research/
│   ├── artifacts/
│   ├── benchmarks/
│   ├── configs/
│   ├── data/
│   ├── final_reports/
│   └── scripts/
│
└── README.md
```

---

# Project Workflow

```
Human Therapeutic Protein
          │
          ▼
Codon Optimization
          │
          ▼
CRISPR Guide Design
          │
          ▼
Synthetic Promoter Generation
          │
          ▼
Sequence Feature Extraction
          │
          ▼
Machine Learning Prediction
          │
          ▼
Candidate Ranking
```

---

# Methodology

## 1. Host Engineering

Designed CRISPR guide RNAs targeting protease genes responsible for recombinant protein degradation.

Methods included:

- CHOPCHOP
- CRISPOR
- BLAST verification
- guide filtering

---

## 2. Codon Optimization

Optimized the coding sequence of human PH20 for plant expression while balancing codon usage and GC content.

---

## 3. AI Promoter Design

Generated synthetic promoter sequences using multiple approaches including foundation models and rule-based sequence engineering.

Promoters were evaluated computationally before downstream prediction.

---

## 4. Machine Learning

Developed a Random Forest regression model to predict promoter strength.

Pipeline included:

- DNA feature engineering
- sequence composition
- motif analysis
- cross-validation
- conformal prediction

---

# Results

The computational workflow produced:

- Optimized CRISPR guide RNAs
- Codon-optimized gene variants
- Synthetic promoter candidates
- Predictive machine learning model for promoter activity

The project demonstrates how AI can accelerate rational design in plant molecular farming.
