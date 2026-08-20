
---

# Contract Review Challenge

**Company / Org:** Accenture  
**Challenge Advisor:** Adarsh Ravikumar, adarsh.ravikumar@accenture.com  \
**AI Studio Coach:** Shweta Malabade, shweta.malabade@breakthroughtech.org  \
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Accenture
Accenture is a leading global professional services company that provides a broad range of services and solutions in strategy, consulting, technology, and operations. 

---

## 🎯 The Challenge
### Project Summary
In this project, you will use real-world commercial contracts from the CUAD dataset (510 contracts, 41 expert-annotated clause categories) and NLP techniques including chunk-based multi-label classification with fine-tuned transformer encoders, paired with an explainable rule-based risk-scoring layer, to build a pipeline that automatically detects key clauses, flags them as Low/Medium/High risk, and rolls these up into a contract-level triage score. This will help our company address the bottleneck legal and procurement teams face when manually reviewing tens of thousands of contracts a year to find the small number of clauses that carry meaningful risk, enabling reviewers to prioritize which contracts to open first.

### Success Criteria
Success has two tracks:
- For clause detection: per-category precision/recall/F1 clearly beating the baseline (accuracy is misleading under CUAD's imbalance), with error analysis on where the model struggles.   
- For risk scoring: since there are no ground-truth labels, success means strong Spearman correlation and bucket agreement between the model's risk rankings and the advisor's hand-ranked clauses, plus a sensitivity analysis showing the High/Medium boundary is stable.

Overall, a successful December outcome is a working end-to-end pipeline producing risk-scored clause registers the advisor finds plausible and useful, a clean documented repo, and a final report covering results, limitations, and estimated reviewer time saved — an auditable triage tool the advisor would actually trust, not a black box.

### Stretch Goals
Stretch goals include span extraction, a trained risk model benchmarked against the rule-based baseline, LLM-generated clause explanations, broader category coverage, a Streamlit/Gradio demo, and an active-learning loop using advisor/model disagreements. These extend modeling or usability without affecting core deliverables.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Preparation & Baseline Modeling | Clean and split the CUAD data, run EDA on class imbalance, build a chunking strategy, and establish a TF-IDF/keyword baseline with per-category metrics. |
| **October** | Transformer Model Development | Fine-tune a transformer encoder for multi-label clause classification, address class imbalance, evaluate with per-category precision/recall/F1, and conduct error analysis. |
| **November** | Risk Scoring & End-to-End Validation | Build and calibrate the four-signal risk-scoring layer, assemble the end-to-end pipeline, and validate risk rankings against advisor-labeled examples. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** CUAD Dataset (Contract Understanding Atticus Dataset)  
**Format:** JSON, Raw Text/PDF  
**Size:** under 1gb  
**Location:** https://github.com/TheAtticusProject/cuad  

### Key Details
- `CUADv1.json` contains 510 commercial contracts and 13,823 annotated answer spans across 41 contract-review categories.
- Use the prepared JSON files: `train_separate_questions.json` contains 408 contracts and `test.json` contains 102 contracts. These are the **official** train/test splits released by The Atticus Project — split at the contract level (not by individual clause) to prevent data leakage, and directly comparable to the results in the original CUAD paper. Do not re-split the data yourselves.
- Contract text is already available in each JSON document's `paragraphs[].context` field, with clause questions in `paragraphs[].qas[]` and labeled spans in `paragraphs[].qas[].answers[]`. **Do not parse raw PDFs for this project.**
- `category_descriptions.csv` provides the name, description, answer format, and group for each of the 41 categories.
- Contracts vary substantially in length, so teams should develop a chunking strategy, preserve important legal terminology during cleaning, and account for class imbalance.

| Dataset / Source | Purpose in Project | Format | Access |
|---|---|---|---|
| **CUAD Category Descriptions** | Defines the 41 clause categories and provides guidance on what each category represents. Useful for building the label mapping and understanding the classification task. | CSV | [CUAD GitHub Repository](https://github.com/TheAtticusProject/cuad) |
| **CUAD Dataset – Hugging Face** | Provides a machine-learning-friendly way to load CUAD directly into Python and Hugging Face workflows.| Hugging Face Dataset | [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa) |

> ⚠️ **Note on Hugging Face naming:** use `theatticusproject/cuad-qa` specifically. The similarly named `theatticusproject/cuad` (no `-qa`) is a different, unstructured repository containing only documentation text — it is **not** usable contract data.

### Working Dataset Expectations

* **Primary Dataset:** Use the CUAD (Contract Understanding Atticus Dataset), containing 510 commercial contracts and 41 expert-annotated clause categories.
* **Initial Scope:** Start with approximately 50–100 contracts for data exploration, preprocessing, and baseline development before expanding to the full dataset.
* **Data Exploration:** Analyze contract length, clause frequency, category distribution, and potential class imbalance.
* **Preprocessing:** Clean and standardize contract text while preserving relevant clause boundaries and annotations.
* **Chunking:** Develop a chunking strategy that allows long contracts to be processed by transformer models while retaining sufficient context.
* **Data Splits:** Create contract-level training, validation, and test sets to prevent data leakage.
* **Classification Labels:** Use CUAD’s 41 clause categories as the initial multi-label classification targets.
* **Evidence Retention:** Preserve the relevant text span for each detected clause so predictions can be explained and reviewed.
* **Risk Scoring:** Develop a separate, explainable rule-based layer to assign **Low/Medium/High** risk, since CUAD does not provide risk labels.
* **Contract-Level Triage:** Aggregate clause-level risk scores into an overall contract risk/triage score.
* **Reproducibility:** Document the dataset version, preprocessing, data splits, assumptions, and methodology in GitHub.
* **Final Output:** The pipeline should produce **clause category → evidence → risk level → rationale → contract-level triage score**.

### Known Preprocessing and Data Risks
* Normalize contract text consistently, including formatting, whitespace, headers, and page breaks while preserving meaningful legal language.
* Standardize contract IDs, clause labels, annotation spans, and document metadata across all source files.
* Handle long contracts carefully: chunk text without separating important clause context or splitting relevant annotations incorrectly.
* Expect domain and annotation variability: CUAD contracts and expert annotations may differ in structure and language, which can affect model performance on new or unseen contracts.

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP / Multi-label Classification / Information Extraction / Explainable Risk Scoring

**Recommended Libraries:** Hugging Face Transformers, PyTorch, scikit-learn, pandas, NumPy, Hugging Face Datasets

**Suggested Pipeline:** Contract Text → Preprocessing & Chunking → Multi-label Clause Classification → Evidence Extraction → Rule-based Risk Scoring → Contract-level Triage Score

**Evaluation Metrics:** Precision, Recall, and F1-Score for clause classification; High-Risk Recall and Precision@K for contract triage; basic error analysis for risk-scoring results.

**Development Environment:** Google Colab for model training and experiments; VS Code and Jupyter Notebooks for development and analysis.

---


## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review (Hendrycks et al., 2021)](https://arxiv.org/abs/2103.06268) — the original paper introducing the dataset. Covers what the 41 clause categories are, why transformer models struggle with this task, and how performance is shaped by model design and training data.
- [Multi-label NLP: An Analysis of Class Imbalance and Loss Function Approaches (KDNuggets, 2023)](https://www.kdnuggets.com/2023/03/multilabel-nlp-analysis-class-imbalance-loss-function-approaches.html) — explains why multi-label classification becomes hard under imbalance and how loss functions like weighted BCE and Focal Loss address it. Directly relevant to the class imbalance problem in CUAD.
- [AI Explainability in Contract Risk Scoring (Sirion, 2026)](https://www.sirion.ai/library/contract-insights/ai-explainability-tools-contracts/) — a practical explainer on how flagged clauses are traced back to their rationale to produce auditable decisions rather than black-box outputs. Useful context for framing the risk-scoring layer.

**Technical Tutorials:**
- [Learn Hugging Face: Text Classification Tutorial](https://www.learnhuggingface.com/notebooks/hugging_face_text_classification_tutorial) — a hands-on, code-focused walkthrough for loading a pretrained DistilBERT model, fine-tuning it on a custom dataset, and evaluating results. Reusable patterns directly applicable to this project.
- [HuggingFace Official Docs: Sequence Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) — the canonical reference for fine-tuning encoder models (DistilBERT, RoBERTa, DistilRoBERTa) using the `Trainer` API. Covers tokenization, `TrainingArguments`, `DataCollatorWithPadding`, and evaluation hooks.
- [Spearman Rank Correlation Explained (Simplilearn)](https://www.simplilearn.com/tutorials/statistics-tutorial/spearmans-rank-correlation) — a clear introduction to Spearman's ρ as a non-parametric measure of ranking agreement. Relevant for evaluating whether the model's contract risk rankings match the advisor's hand-ranked examples.

**Code Examples:**
- [Official CUAD GitHub Repository (The Atticus Project)](https://github.com/TheAtticusProject/cuad) — contains the dataset, baseline model code, and evaluation scripts from the original paper. Good reference for understanding the intended evaluation setup before adapting it to a classification framing.
- [CUAD on Hugging Face Datasets](https://huggingface.co/datasets/cuad) — loads the dataset directly via `datasets.load_dataset("cuad")`, bypassing manual JSON parsing and enabling easy integration with the HuggingFace `Trainer` pipeline. The fastest way to get data into a training loop.
- [Hybrid NLP Framework for Contract Risk Assessment (IJSRS, 2025)](http://ijsrs.org/v2i11/1.php) — an implementation paper combining fine-tuned RoBERTa with a rule-based risk assessment layer, close in architecture to what this project is building. Includes a Streamlit interface and covers the explainability angle explicitly — useful as a reference design.

**Other:**
- [Stanford CS224N HuggingFace Transformers Tutorial (2026)](https://web.stanford.edu/class/cs224n/materials/hf_transformers_tutorial.pdf) — lecture slides from Stanford's NLP course covering the full HuggingFace ecosystem: tokenizers, AutoModel classes, the Trainer API, and fine-tuning on custom datasets. Good structured background before diving into project code.
- [DistilRoBERTa Base Model Card (HuggingFace)](https://huggingface.co/distilroberta-base) — the recommended base model per the advisor's guidance. The model card explains its distillation from RoBERTa, speed/accuracy tradeoffs, and how to load it for sequence classification.


**Suggested Pipeline:**

CUAD Contracts → Preprocessing → Chunking → Multi-label Clause Classification → Evidence Extraction → Rule-based Risk Scoring → Contract-level Triage Score → Explainable Risk Report

## Recommended Modeling Approach

- **Establish a Baseline:** Build a simple TF-IDF/keyword-based classifier before using transformer models.
- **Train the Model:** Fine-tune a transformer encoder for multi-label classification across the 41 CUAD clause categories.
- **Preserve Evidence:** Retain the relevant contract text/span for each detected clause to support explainability.
- **Add Risk Scoring:** Develop an explainable rule-based layer that assigns **Low / Medium / High** risk based on detected clause characteristics.
- **Calculate Triage Score:** Aggregate clause-level risks into an overall **contract-level triage score** to help prioritize contracts for review.
- **Evaluate Performance:** Measure clause classification using **Precision, Recall, and F1 Score**, with emphasis on identifying high-risk clauses.
- **Perform Error Analysis:** Review false positives, false negatives, rare clause categories, and difficult contract language to identify opportunities for improvement.


*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## Evaluation Metrics

| **Component** | **Metric** | **Purpose** |
|---|---|---|
| Clause Classification | **Precision** | Of the clauses the model identifies, how many are correct? |
| Clause Classification | **Recall** | Of the relevant clauses in the contract, how many does the model find? |
| Clause Classification | **F1 Score** | Combines Precision and Recall into one overall classification score. |
| Risk Scoring | **Accuracy** | How often does the system correctly assign Low, Medium, or High risk? |
| Contract Triage | **High-Risk Recall** | Of the contracts that should receive priority review, how many does the system successfully flag? |

### Primary Evaluation

Team should focus primarily on **Precision, Recall, and F1 Score** when evaluating clause classification.

For the final risk-triage pipeline, **High-Risk Recall** is especially important because the goal is to avoid missing contracts that may require additional legal or procurement review.

Team should also perform a simple **error analysis** by reviewing examples of:
- Incorrectly identified clauses
- Missed clauses
- Incorrect risk assignments
- Difficult or ambiguous contract language

The goal is not only to report a score, but to understand **where the model works well, where it fails, and why**.

---


## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [Discord Channel](https://discord.gg/7XC4pB5deF)
* Adarsh Ravikumar, adarsh.ravikumar@accenture.com
  



**Recommended free coding / collaboration tools**
* Google Colab
• GitHub Projects
• GitHub Issues
• VS Code
• Jupyter Notebooks

---
## What I Expect From the Team

- **Keep work visible:** Track tasks, questions, and progress using GitHub Issues and the GitHub Projects board.
- **Document decisions:** Record important modeling decisions, assumptions, and failed experiments—not only successful results.
- **Move code into modules:** Use notebooks for exploration and experimentation, but move reusable and production-ready code into Python modules as the project matures.
- **Maintain reproducibility:** Keep the repository organized with clear setup instructions, dependencies, data documentation, and reproducible workflows so that an external reviewer can run the demo.
- **Use meaningful commits:** Write clear commit messages that describe what was changed and why.
- **Keep documentation current:** Update the README and relevant documentation as the project architecture, models, and results evolve.

## 🚀 Getting Started

Read this overview and list your open questions before our first team meeting.

1. **Review this overview document** and note any questions for our first meeting: Understand the project goals, technical approach, dataset expectations, milestones.
2. **Begin reviewing the JSON dataset** in the [data folder](data/cuad) Explore the 41 clause categories and identify the initial subset of contracts you will use for data exploration and baseline development.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)
4. **Prepare Open Questions:** Record questions, assumptions, and areas where you need clarification before the first team meeting.
5. **Document Your Decisions:** Keep important technical decisions and findings in GitHub Issues or project documentation so the entire team can follow the project's progress.

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 

## Data & Format

- **Q: Should we use the CUAD JSON files (question-answer format) as-is, or reshape them into per-chunk classification format, since the milestone doc calls for "chunk-based multi-label classification"?**
  - A: Reshape them. The JSON format is set up for a different task — finding answer spans — which isn't what we're building. You'll need to convert contracts into overlapping text chunks and label each one. It's a bit of upfront work in September but the right foundation for everything that follows.

- **Q: Is the pre-made train/test split okay to reuse, or should the team build its own after reshaping?**
  - A: Build your own. The existing split was designed for the original QA task and won't carry over cleanly. The main thing to get right is splitting by contract, not by chunk — if chunks from the same contract end up in both train and test, your evaluation numbers will look better than they actually are.

---

## Scoping the 41 Clause Categories

- **Q: Which ~10 clause categories matter most to Accenture's actual legal/procurement teams? Can you help rank them?**
  - A: From what I've seen in practice, the ones that come up most in reviews are: Limitation of Liability, Indemnification, Termination for Convenience, Governing Law, Auto-Renewal, IP Ownership Assignment, Change of Control, Non-Compete, Audit Rights, and Most Favored Nation. Liability and indemnification clauses are where the real exposure sits, so I'd prioritize getting those right before the others. That said, run your EDA first — if any of these have very few examples in the data, flag it and we can swap one out.

---

## Risk Scoring

- **Q: What specific signals define "risky"? (e.g., missing liability cap, one-sided termination, vague language) Do you have a real checklist from experience?**
  - A: The things that consistently raise flags in real reviews: a liability clause with no monetary cap, termination rights that only run one way, indemnification that's completely open-ended, auto-renewal with no notice window, and IP assignment language that's ambiguous about who owns what. Vague language on its own is harder to score consistently, so I'd keep the signals concrete for now and revisit once the core model is working.

- **Q: Can you hand-rank a small batch (20-30) of clauses as a reference the team can compare against?**
  - A: Yes, happy to do that. Put together a spreadsheet with 20–30 clause excerpts pulled from the test set and I'll go through and score them. A simple 1–5 scale works fine. Give me a couple weeks once you have the data prepped.

- **Q: What does Low/Medium/High mean in practice at Accenture — existing rubric, or do we define it from scratch?**
  - A: We don't have a formal rubric for this, so define it from scratch and I'll react to it. My rough instinct: Low is a standard clause with no red flags, Medium is something that warrants a closer look but isn't a dealbreaker, and High is either a missing clause that should be there or language that creates real financial or legal exposure. Run your proposed thresholds by me before you lock them in for November.

- **Q: Should a missing clause (one that should exist but doesn't) count as a risk signal on its own?**
  - A: Yes, absolutely. A contract that's silent on liability or IP ownership isn't neutral — it usually defaults to the worse position for whoever didn't catch it. Treat absence as at least a Medium for any of the high-priority categories.

- **Q: Should the risk signals be weighted equally, or do some matter more?**
  - A: Some matter more. I'd treat an uncapped liability or a completely missing indemnification clause as automatically High, regardless of what else looks fine. Equal weighting is a reasonable starting point to get the system running, but build it so weights can be adjusted — we'll want to tune that in November once we can see how the scores are landing.

---

## Modeling / Implementation

- **Q: Is Google Colab free tier sufficient, or should the team have Colab Pro / paid compute access?**
  - A: Free tier will probably cause headaches — sessions disconnect and training runs that take a few hours aren't realistic on it. Colab Pro is worth it at $10/month, or the team can use Kaggle Notebooks for free, which gives 30 GPU hours a week with more stable sessions. I'd sort this out before October when the actual fine-tuning starts.

- **Q: How should long contracts be chunked — paragraph-based, sentence-based, fixed token length, or team's discretion?**
  - A: Fixed token length with some overlap is the way to go — 512 tokens with roughly a 64-token overlap is standard and will work well here. Paragraph-based sounds intuitive but legal paragraphs are all over the place in length, so it gets messy fast. Stick with the fixed window approach and document what you chose so it's reproducible.
