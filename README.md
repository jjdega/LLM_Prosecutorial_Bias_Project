# Does AI Think Like a Prosecutor?
### Measuring a Prosecutorial Bias in LLM Bail Decisions: A Cross-Model Comparison of Cash Bail Decisions from Dane County, WI

**JJ Dega** | GOVT 20.12: Politics and AI | Dartmouth College | Spring 2026
Supervised by Professor Adam Breuer

## Overview

This project investigates whether large language models (LLMs) exhibit systematic prosecutorial bias in pretrial bail decisions. Using a randomized controlled trial dataset from Dane County, Wisconsin (n = 1,891 first-appearance hearings), three frontier LLMs are evaluated under three experimental conditions and compared against human judge decisions using the Ben-Michael et al. (2025) causal inference framework and dataset.

**Primary finding:** All three LLMs recommend cash bail at up to 2.8 times the human judge rate (25.4% baseline). The adversarial multi-agent pipeline amplifies rather than corrects this bias. Non-White defendants face disproportionately higher LLM detention rates across all conditions.

**Models tested:** Claude Sonnet 4.6 (Anthropic), GPT-4o (OpenAI), Gemini 2.5 Flash (Google)

**Poster:** You can view this project's poster overview at: dartgo.org/llm_prosecutorial_bias_project_poster OR https://canva.link/6r975ton2vbtgjg

## Repository Structure

    LLM_Prosecutorial_Bias_Project/
    ├── main.R       # Complete analysis pipeline
    ├── README.md    # This file
    └── outputs/     # Generated outputs (not tracked)


## Data Access/Citation

Original RCT: Ben-Michael, Eli, Daniel Greiner, Melody Huang, Kosuke Imai, Zhichao Jiang, and Sooahn Shin. 2025. “Replication Data for: Does AI Help Humans Make Better Decisions?: A Statistical Evaluation Framework for Experimental and Observational Studies.” Harvard Dataverse. https://doi.org/10.7910/DVN/KMM8WN.


## Key Results

| Model | Exp 1A | Exp 1B | Exp 2 | Human |
|-------|--------|--------|-------|-------|
| Claude Sonnet 4.6 | 62.9% | 39.6% | 70.1% | 25.4% |
| GPT-4o | 53.3% | 34.1% | 28.1% | 25.4% |
| Gemini 2.5 Flash | 58.6% | 35.1% | 32.1% | 25.4% |

## Citation

Dega, JJ (2026). Does AI Think Like a Prosecutor? Undergraduate Research Project, Dartmouth College.

## Acknowledgments

Built on the aihuman R package from Ben-Michael et al. (2025) via Harvard Dataverse. Supervised by Professor Adam Breuer and Rasmus Torp.

Questions: jj.f.dega.26@dartmouth.edu
