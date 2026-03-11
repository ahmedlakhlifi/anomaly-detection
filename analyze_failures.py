import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def defect_type_from_path(p: str) -> str:
    path = Path(p)
    # .../test/<defect_type>/<file>.png
    parts = [x.lower() for x in path.parts]
    if "test" in parts:
        i = parts.index("test")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description="Summarize failure cases from evaluate.py outputs.")
    p.add_argument("--failure-csv", type=str, required=True, help="Path to failure_cases.csv")
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    rows = []
    with open(args.failure_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    fp = [x for x in rows if x.get("error_type") == "FP"]
    fn = [x for x in rows if x.get("error_type") == "FN"]

    fn_defects = Counter(defect_type_from_path(x.get("path", "")) for x in fn)
    fp_defects = Counter(defect_type_from_path(x.get("path", "")) for x in fp)

    fn_sorted = sorted(fn, key=lambda x: float(x.get("score", 0.0)))
    fp_sorted = sorted(fp, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    summary = {
        "num_failures": len(rows),
        "num_false_positives": len(fp),
        "num_false_negatives": len(fn),
        "false_negative_by_defect": dict(fn_defects),
        "false_positive_by_defect": dict(fp_defects),
        "lowest_score_false_negatives": [
            {
                "path": x.get("path"),
                "score": float(x.get("score", 0.0)),
            }
            for x in fn_sorted[:10]
        ],
        "highest_score_false_positives": [
            {
                "path": x.get("path"),
                "score": float(x.get("score", 0.0)),
            }
            for x in fp_sorted[:10]
        ],
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
