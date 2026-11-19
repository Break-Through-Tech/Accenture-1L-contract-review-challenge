# Accenture contract risk pipeline — Milestone 3 Task #2 handoff

## What was delivered

The integrated pipeline uses Part A contract clause detections and full-contract text to derive eight rule-based risk flags, then calls the existing `risk_scoring` engine. Its output includes contract-level risk triage, score ranges, explanations, and human-review routing.

## Input and output

- Input: `outputs/teammate_evaluation/auto_triage_test.json`
- Structural validation: `outputs/teammate_evaluation/validation_summary.json`
- Contract report: `contract_risk_report.csv`
- Critical-override review: `critical_override_audit.csv`

## Current test-split results

- Contracts: 102
- High: 52
- Medium: 12
- Low: 2
- Unresolved / needs review: 36
- Critical-override cases requiring evidence audit: 45
- Structural validation: passed
- Accuracy against advisor labels: NOT VALIDATED

## Instructions for Task #3

1. Use advisor-labeled contracts to independently assess whether each automated risk flag and contract-level risk label is correct.
2. Prioritize the critical-override audit, especially `missing_indemnification` and `uncapped_liability` findings. Review full contract context, exceptions, and whether the protection is appropriate to the contract type.
3. Evaluate false positives and false negatives, including samples from the Low, Medium, High, and unresolved groups. Do not assume that a missing detection proves that a clause is absent.
4. Record decisions and supporting excerpts in the audit CSV; do not treat `reviewer=auto:rules-v1` as a human approval.
5. Calculate classification accuracy and per-class precision/recall/F1 where reference labels permit. Document ambiguous labels and any disagreements.
6. Re-run the pipeline and structural validation after any rule or threshold changes. Avoid using the held-out test split for tuning.

## Important limitations

The risk categories are automated screening outputs, not legal conclusions. A `High` label is not necessarily a verified High risk. Scores may be missing where flags remain unknown; use score bounds rather than inventing an exact score. Structural validation confirms data consistency only, not model accuracy or evidence correctness.
