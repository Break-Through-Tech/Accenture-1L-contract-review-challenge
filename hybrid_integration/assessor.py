
"""Conservative candidate-risk assessment from Part A excerpts.

Identifies potential contractual risks using rule-based screening.
All findings remain provisional until confirmed by a human reviewer.

A missing match does NOT establish that a contract is safe.
"""

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------
# 1. Risk detection rules
# ---------------------------------------------------------

RULES = {
    "uncapped_liability": [
        r"\b(?:unlimited|uncapped)\s+liabilit(?:y|ies)\b",
        r"\bliabilit(?:y|ies)\b.{0,120}\b(?:unlimited|uncapped)\b",
        r"\bliabilit(?:y|ies)\b.{0,120}\bwithout\s+(?:any\s+)?limit\b",
        r"\bno\s+(?:monetary\s+)?cap\s+on\s+liability\b",
        r"\bnot\s+(?:be\s+)?subject\s+to\s+(?:any\s+)?"
        r"(?:limitation|cap)\s+of\s+liability\b",
        r"\b(?:limitation|cap)\s+of\s+liability\b.{0,120}"
        r"\bshall\s+not\s+apply\b",
    ],

    "open_ended_indemnification": [
        r"\bindemnif(?:y|ies|ied|ication)\b.{0,150}"
        r"\b(?:unlimited|uncapped)\b",
        r"\bindemnif(?:y|ies|ied|ication)\b.{0,150}"
        r"\bwithout\s+(?:any\s+)?(?:limit|limitation)\b",
        r"\bunlimited\s+indemnif(?:ication|y)\b",
        r"\bindemnif(?:ication|y)\b.{0,150}"
        r"\b(?:all\s+losses|any\s+and\s+all\s+losses)\b",
    ],

    "renewal_without_notice": [
        r"\bautomatically\s+renew\w*\b.{0,200}"
        r"\bwithout\s+(?:prior\s+)?notice\b",
        r"\brenew\w*\s+automatically\b.{0,200}"
        r"\bwithout\s+(?:prior\s+)?notice\b",
        r"\bautomatic\s+renewal\b.{0,200}"
        r"\bwithout\s+(?:prior\s+)?notice\b",
    ],
}


# ---------------------------------------------------------
# 2. Relevant CUAD categories
# ---------------------------------------------------------

CATEGORIES = {
    "uncapped_liability": {
        "Uncapped Liability",
        "Cap On Liability",
    },

    # No direct CUAD category restriction.
    "open_ended_indemnification": set(),

    "renewal_without_notice": {
        "Renewal Term",
        "Notice Period To Terminate Renewal",
    },
}


# ---------------------------------------------------------
# 3. Candidate classification
# ---------------------------------------------------------

def classify_candidate(flag, matched_phrase, evidence):
    """Classify screening evidence, not confirmed legal risk."""

    text = evidence.lower()
    phrase = matched_phrase.lower()

    if flag == "open_ended_indemnification":

        cap_patterns = [
            r"liability.{0,100}shall not exceed",
            r"liability.{0,100}limited to",
            r"maximum aggregate liability",
            r"subject to.{0,80}liability cap",
        ]

        # A nearby liability cap may affect interpretation.
        # It is not proof that indemnification is capped.
        if any(
            re.search(pattern, text, re.DOTALL)
            for pattern in cap_patterns
        ):
            return "conflicting_evidence"

        if "without limitation" in phrase:
            return "needs_context"

        if "any and all losses" in phrase:
            return "needs_context"

    if flag in (
        "uncapped_liability",
        "renewal_without_notice",
    ):
        return "potential_risk"

    return "needs_context"


# ---------------------------------------------------------
# 4. Evidence extraction
# ---------------------------------------------------------

def _excerpt(text, start, end, window=110):
    """Extract surrounding text for human review."""

    return text[
        max(0, start - window):
        min(len(text), end + window)
    ].strip()


# ---------------------------------------------------------
# 5. Main assessment pipeline
# ---------------------------------------------------------

def assess(payload, records, queue):
    """Generate provisional candidates from Part A detections."""

    if payload.get("smoke_test") is not False:
        raise ValueError(
            "Only verified non-smoke-test detections are supported"
        )

    contracts = {
        contract["contract_id"]: contract
        for contract in payload["contracts"]
    }

    if len(contracts) != len(payload["contracts"]):
        raise ValueError("Duplicate contract IDs")

    queue_by_id = {
        item["record_id"]: item
        for item in queue
    }

    if len(queue_by_id) != len(queue):
        raise ValueError("Duplicate queue record IDs")

    out_records = []
    out_queue = []

    for original in records:

        # Preserve original records.
        record = json.loads(json.dumps(original))

        record_id = record["record_id"]
        contract_id = record["contract_id"]

        if (
            record_id not in queue_by_id
            or contract_id not in contracts
        ):
            raise ValueError(
                f"Missing matching input for {record_id}"
            )

        contract = contracts[contract_id]

        if record["split"] != contract["split"]:
            raise ValueError(
                f"Split mismatch: {record_id}"
            )

        review = json.loads(
            json.dumps(queue_by_id[record_id])
        )

        candidates = []

        # -------------------------------------------------
        # 6. Search Part A detections
        # -------------------------------------------------

        for detection in contract.get("detections", []):

            text = detection.get("text", "")

            if not isinstance(text, str):
                continue

            for flag, patterns in RULES.items():

                allowed_categories = CATEGORIES[flag]

                if (
                    allowed_categories
                    and detection.get("category")
                    not in allowed_categories
                ):
                    continue

                for pattern in patterns:

                    match = re.search(
                        pattern,
                        text,
                        re.IGNORECASE | re.DOTALL,
                    )

                    if not match:
                        continue

                    evidence = _excerpt(
                        text,
                        match.start(),
                        match.end(),
                    )

                    classification = classify_candidate(
                        flag,
                        match.group(0),
                        evidence,
                    )

                    candidates.append({
                        "flag": flag,
                        "status": "candidate_needs_review",
                        "classification": classification,
                        "category": detection.get("category"),
                        "chunk_id": detection.get("chunk_id"),
                        "probability": detection.get("probability"),
                        "evidence": evidence,
                        "matched_phrase": match.group(0),
                        "reason": (
                            "Automated screening only; verify "
                            "parties, exceptions, liability caps, "
                            "and surrounding context."
                        ),
                        "confidence": "requires_human_validation",
                        "detection_method": "rule_based_regex",
                    })

                    # Avoid multiple regex matches for the
                    # same flag in one detection.
                    break

        # -------------------------------------------------
        # 7. Remove duplicate candidates
        # -------------------------------------------------

        unique = {}

        for candidate in candidates:

            normalized_phrase = re.sub(
                r"\s+",
                " ",
                candidate["matched_phrase"],
            ).lower()

            key = (
                candidate["flag"],
                normalized_phrase,
            )

            unique.setdefault(key, candidate)

        # -------------------------------------------------
        # 8. Prepare human review queue
        # -------------------------------------------------

        review["automated_candidates"] = list(
            unique.values()
        )

        review["candidate_count"] = len(unique)

        review["assessment_status"] = (
            "awaiting_human_review"
        )

        review["warning"] = (
            "Candidates are NOT confirmed risks. "
            "Missing clauses, one-sided rights, and "
            "ambiguous IP ownership require "
            "full-contract human review."
        )

        if any(
            candidate["flag"] == "uncapped_liability"
            for candidate in unique.values()
        ):
            review["priority"] = "high"
        else:
            review["priority"] = review.get(
                "priority",
                "standard",
            )

        # Important:
        # Do not automatically mark risk flags as present.
        # The risk scoring engine must only receive
        # verified assessments.

        out_records.append(record)
        out_queue.append(review)

    return out_records, out_queue


# ---------------------------------------------------------
# 9. Command-line interface
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence-backed, "
            "unconfirmed risk candidates"
        )
    )

    parser.add_argument(
        "--detections",
        required=True,
    )

    parser.add_argument(
        "--records",
        required=True,
    )

    parser.add_argument(
        "--queue",
        required=True,
    )

    parser.add_argument(
        "--output-records",
        required=True,
    )

    parser.add_argument(
        "--output-queue",
        required=True,
    )

    args = parser.parse_args()

    def read(path):
        return json.loads(
            Path(path).read_text(encoding="utf-8")
        )

    records, queue = assess(
        read(args.detections),
        read(args.records),
        read(args.queue),
    )

    outputs = (
        (args.output_records, records),
        (args.output_queue, queue),
    )

    for path, value in outputs:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    total_candidates = sum(
        item["candidate_count"]
        for item in queue
    )

    print(
        f"Prepared {len(records)} provisional records, "
        f"{len(queue)} review items, "
        f"{total_candidates} candidate findings."
    )


if __name__ == "__main__":
    main()
