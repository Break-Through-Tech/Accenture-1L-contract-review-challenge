# Milestone 3: Four-signal risk scoring (Katie, issue #21)

## Basis and current status

[Team issue #21](https://github.com/Break-Through-Tech/Accenture-1L-contract-review-challenge/issues/21) assigns Katie “Build and calibrate the four-signal risk-scoring layer.” Neither that issue nor its comments defines the four signals. The source for this implementation is the advisor's **Risk Scoring** Q&A in [the project overview](https://github.com/Break-Through-Tech/Accenture-1L-contract-review-challenge/blob/main/Challenge-Project-Overview.md#risk-scoring).

The advisor identifies missing protections, uncapped liability, open-ended indemnification, unilateral termination, renewal without notice, and unclear IP ownership. The four groups below are an implementation proposal derived from that guidance, not an advisor-approved rubric. The initial equal weights and 25/60 boundaries are configurable starting points, not learned settings. The advisor requested threshold review before finalizing them. CUAD provides clause annotations, not risk labels. No advisor ratings were found in the inspected repository/local project; real calibration and validation remain pending. The module supplies both workflows without inventing scores.

| Signal | Risk flags | Advisor basis |
| --- | --- | --- |
| Missing protections | Missing liability, indemnification, IP ownership | Missing priority protections warrant at least Medium |
| Financial exposure | Uncapped liability, open-ended indemnification | Concrete financial exposure |
| One-sided rights | Unilateral termination | Termination rights that run one way |
| Renewal/IP clarity | Renewal without notice, ambiguous IP ownership | Notice-window and ownership concerns |

Uncapped liability and confirmed missing indemnification always yield High. Other confirmed missing protections enforce at least Medium. These policy overrides cannot be tuned away by changing weights. Additional priority categories can be added in a later version after advisor review; current coverage is explicitly limited to the eight flags above.

## Run in VS Code

Open the repository folder. The implementation is in `risk_scoring/engine.py`, calibration in `risk_scoring/calibration.py`, and the command interface in `risk_scoring/__main__.py`. From the VS Code terminal at repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-risk.txt
python -m risk_scoring score --input examples/risk_scoring_synthetic.json --config config/risk_scoring.json --output synthetic_scores.json
python -m risk_scoring score --input examples/advisor_review_pending.json --output pending_scores.json
python -m unittest discover -s tests -p 'test_risk_scoring.py' -v
```

The first input is invented test data, clearly marked synthetic. The second contains **25 actual CUAD validation excerpts** balanced across available relevant categories, with every assessment unknown and all advisor ratings blank. It is a review packet, not model predictions or scored project findings. The scorer returns unknown buckets for these excerpts until reviewed. Text comes from committed `val_annotations.csv`; test examples were not used in calibration preparation. Existing split files are reused; nothing is re-split. All 25 excerpts were verified verbatim in `data/CUADv1.json`. Source offsets refer to the original CUAD contract context; the prepared contract CSV normalizes whitespace, so do not reuse those offsets against its cleaned text.

Regenerate the review packet:

```sh
python -m risk_scoring prepare-advisor --input notebooks/val_annotations.csv --output review_packet.json --limit 25
```

## Input contract and human assessment

Each record includes a stable `record_id`, `contract_id`, `category`, `text`, `source`, `split`, and eight `assessments`. Optional `advisor_rating` is required only for calibration/evaluation. The four status values are:

- `present`: this risk characteristic is confirmed, not merely that the clause exists.
- `absent`: the risk characteristic has been checked and is not present.
- `not_applicable`: explicitly reviewed and irrelevant to this record's scope.
- `unknown`: unreviewed, ambiguous or insufficient evidence.

Every resolved assessment requires `rationale` and `reviewer` provenance. A present non-missing risk also requires `evidence` that exactly occurs in `text`. A missing protection must have `scope: "full_contract"`; it cannot be inferred from a single chunk, missing annotation, or negative classifier prediction. Include a documented full-contract search in its rationale and preserve the full-contract source reference. The code verifies declared scope, not whether the review really occurred. A missing protection and a risk characteristic of that same present clause cannot both be asserted in one record.

For example, a reviewer may set `uncapped_liability` to `present` and attach the exact uncapped-liability wording. Other flags outside that clause can be marked `not_applicable` with rationale. Full-contract absence findings should be separate, explicitly reviewed records. None of these assignments is inferred from a probability threshold or generic keyword absence.

This module accepts **structured evidence assessments**; it does not pretend to detect legal meaning from raw strings. Issue #22's pipeline can pass detected clause spans and explicit evidence assessments into `score_record`. Model confidence is not used as legal severity. Classification label probabilities alone are insufficient to establish whether a cap, notice period or ownership protection exists.

## Score and explanations

A flag has configurable severity in [0,1]. A signal is the maximum severity of its flags, so multiple related flags within a group do not double-count the same concern. The base score is `100 * sum(weight * signal) / sum(weights)`. Defaults use equal weights. Medium starts at 25 and High at 60, both inclusive. Policy overrides impose a floor at the relevant boundary. The number is an explainable triage index, not a probability or monetary loss estimate.

Unknown flags produce a lower/upper interval. The code does not renormalize weights over the few available signals or silently equate unknown with safe. With unresolved flags, `score` is null, `review_required` is true, and `bucket` is null unless a confirmed critical override already establishes High. Bounds and known reasons remain visible. With fully assessed inputs, a numeric score and bucket are provided.

Each output retains IDs, source/text, per-signal values, reasons/evidence, critical overrides, config version and calibration status. Contract triage takes the maximum record risk to avoid diluting one dangerous clause with many benign clauses. This is a summary of supplied records only; it never claims full-contract coverage.

## Advisor calibration and held-out evaluation

Have the advisor review the packet, record independent ratings from 1 to 5 and resolve the structured assessments. Prefer 20–30 development examples spanning varied risks and multiple contracts. Agreement with risk labels depends on these assessments being accurate. Do not call synthetic fixtures or CUAD category labels advisor ratings.

```sh
python -m risk_scoring calibrate --input reviewed_development.json --config config/risk_scoring.json --output calibration.json
python -m risk_scoring score --input reviewed_development.json --config calibration.json --output development_scores.json
python -m risk_scoring evaluate --input reviewed_test.json --config calibration.json --output heldout_report.json
```

The proposed rating mapping is 1–2 Low, 3 Medium, 4–5 High; confirm it with the advisor. A deterministic finite search evaluates 144 combinations: four weights each in {1,2}, Medium in {20,30,40}, and High in {50,60,70}. Selection maximizes bucket agreement, then minimizes bucket distance, then maximizes Spearman correlation; ties favor weights closer to equal and thresholds nearer the defaults. It preserves mandatory overrides. The report includes before/after **in-sample** metrics and boundary sensitivity at High ±5. The artifact stores development contract IDs and an input SHA-256 fingerprint.

The code requires varying ratings, at least three fully assessed examples and at least two contracts, but this minimum is only an execution guard, not evidence that a sample is adequate. Undefined Spearman correlation and high-risk recall are JSON null when scores/ratings are constant or there are no true High cases. The calibrated configuration must be frozen for final evaluation. Held-out evaluation requires test records and rejects contracts seen in calibration. It reports Spearman correlation, bucket agreement, mean bucket distance, high-risk recall/support and High-boundary flips. Stability and calibration quality must be inspected; no performance target is claimed automatically. Threshold sensitivity uses exactly the same evidence and frozen weights, recomputing mandatory floors as thresholds move.

## Limitations and task boundaries

Real calibration is unfinished until advisor ratings and assessments are supplied. The grouping, threshold choices and rating-to-bucket mapping require advisor feedback. No trained classifier is bundled or retrained here. This implements the scoring/calibration layer; full pipeline assembly (issue #22) and independent advisor ranking validation (issue #20) remain separate tasks. The layer deliberately avoids guessing absence or using legal keyword matches as a complete risk assessment.
