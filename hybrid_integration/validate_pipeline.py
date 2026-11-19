
import json
from pathlib import Path
from collections import Counter

BASE = Path("outputs/teammate_evaluation")
TRIAGE_FILE = BASE / "auto_triage_test.json"
ASSESSMENTS_FILE = BASE / "auto_assessments_test.json"
OUTPUT_FILE = BASE / "validation_summary.json"

VALID_LEVELS = {"High", "Medium", "Low"}
VALID_STATUSES = {
    "auto_classified",
    "auto_classified_unknowns_immaterial",
    "human_review",
}
CRITICAL_FLAGS = {"uncapped_liability", "missing_indemnification"}


def validate():
    with TRIAGE_FILE.open() as f:
        payload = json.load(f)

    with ASSESSMENTS_FILE.open() as f:
        assessments = json.load(f)

    contracts = payload["contracts"]

    errors = []
    warnings = []
    critical_audit = []
    level_counts = Counter()
    status_counts = Counter()

    if not isinstance(assessments, list):
        errors.append("Assessments must be a list.")
    elif len(assessments) != len(contracts):
        errors.append("Assessment and triage counts differ.")

    seen = set()

    for contract in contracts:
        cid = contract.get("contract_id")
        level = contract.get("risk_level")
        status = contract.get("status")
        lower = contract.get("score_lower")
        upper = contract.get("score_upper")
        score = contract.get("score")
        unknown = contract.get("unknown_flags") or []
        critical = contract.get("critical_overrides") or []

        level_counts[str(level)] += 1
        status_counts[str(status)] += 1

        if not cid:
            errors.append("Contract missing contract_id.")
            continue

        if cid in seen:
            errors.append(f"{cid}: duplicate contract_id.")
        seen.add(cid)

        if status not in VALID_STATUSES:
            errors.append(f"{cid}: invalid status {status}.")

        if lower is None or upper is None:
            errors.append(f"{cid}: missing score bounds.")
            continue

        if not (0 <= lower <= upper <= 100):
            errors.append(f"{cid}: invalid score interval.")

        if score is not None and not (lower <= score <= upper):
            errors.append(f"{cid}: exact score outside interval.")

        if status == "human_review":
            if level is not None:
                errors.append(f"{cid}: human_review has final risk level.")
        elif level not in VALID_LEVELS:
            errors.append(f"{cid}: classified contract has invalid level.")

        if status == "auto_classified" and unknown:
            warnings.append(f"{cid}: auto_classified has unknown flags.")

        if status == "auto_classified_unknowns_immaterial" and not unknown:
            warnings.append(f"{cid}: immaterial-unknown status without unknown flags.")

        if critical:
            unexpected = set(critical) - CRITICAL_FLAGS
            if unexpected:
                errors.append(f"{cid}: unexpected critical overrides: {unexpected}")

            critical_audit.append({
                "contract_id": cid,
                "risk_level": level,
                "critical_overrides": critical,
                "unknown_flags": unknown,
                "score_lower": lower,
                "score_upper": upper,
                "requires_evidence_audit": True,
            })

            if lower < 60:
                errors.append(f"{cid}: critical override below expected high-risk floor.")

        if unknown and status != "human_review":
            if level is None:
                errors.append(f"{cid}: unknown flags with no classified level.")

    summary = {
        "total_contracts": len(contracts),
        "total_assessments": len(assessments) if isinstance(assessments, list) else None,
        "risk_levels": dict(level_counts),
        "assessment_statuses": dict(status_counts),
        "critical_override_contracts": len(critical_audit),
        "critical_override_audit": critical_audit,
        "structural_errors": errors,
        "warnings": warnings,
        "structural_validation_passed": len(errors) == 0,
        "accuracy_validated": False,
        "note": (
            "Structural validation only. Risk accuracy and evidence "
            "correctness require independent advisor-based evaluation."
        ),
    }

    OUTPUT_FILE.write_text(json.dumps(summary, indent=2))

    print("\n=== PIPELINE VALIDATION ===")
    print("Contracts:", len(contracts))
    print("Risk levels:", dict(level_counts))
    print("Statuses:", dict(status_counts))
    print("Critical overrides:", len(critical_audit))
    print("Structural errors:", len(errors))
    print("Warnings:", len(warnings))
    print("Validation passed:", len(errors) == 0)
    print("Report:", OUTPUT_FILE)

    for error in errors[:15]:
        print("ERROR:", error)

    for warning in warnings[:10]:
        print("WARNING:", warning)

    return len(errors) == 0


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
