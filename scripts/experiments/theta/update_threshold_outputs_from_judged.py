import argparse
import csv
import json
from pathlib import Path


REQUIRED_RESULT_KEYS = {
    "theta",
    "category",
    "success_rate",
    "avg_relations_per_object",
    "avg_relations_per_keyword",
    "relations_kept",
    "relations_total",
    "ope_relations_kept",
    "ope_relations_total",
    "urp_relations_kept",
    "urp_relations_total",
}


def _is_sweep_payload(payload: dict) -> bool:
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return False
    first = results[0]
    return isinstance(first, dict) and REQUIRED_RESULT_KEYS.issubset(first.keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original_json")
    parser.add_argument("judged_json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    original_path = Path(args.original_json)
    judged_path = Path(args.judged_json)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    if not original_path.exists():
        raise SystemExit(f"Original sweep JSON not found: {original_path}")

    original_payload = json.loads(original_path.read_text())
    if not _is_sweep_payload(original_payload):
        raise SystemExit(
            "Original JSON no longer has the sweep schema needed for the 3-subplot plot. "
            "Restore a pre-refresh sweep JSON or rerun threshold_sweep.py."
        )

    judged_payload = json.loads(judged_path.read_text())
    aggregate = judged_payload.get("aggregate", [])
    if not aggregate:
        raise SystemExit(f"No aggregate results found in judged JSON: {judged_path}")

    success_by_key = {
        (row["category"], float(row["theta"])): float(row["success_rate"])
        for row in aggregate
    }

    updated_results = []
    for row in original_payload["results"]:
        key = (row["category"], float(row["theta"]))
        new_row = dict(row)
        if key in success_by_key:
            new_row["success_rate"] = success_by_key[key]
        updated_results.append(new_row)

    updated_payload = dict(original_payload)
    updated_payload["results"] = updated_results

    overall = {}
    grouped = {}
    for row in updated_results:
        grouped.setdefault(float(row["theta"]), []).append(row)

    for theta, rows in grouped.items():
        overall[theta] = {
            "overall_success_rate": sum(r["success_rate"] for r in rows) / len(rows),
            "avg_relations_per_object": sum(r["avg_relations_per_object"] for r in rows) / len(rows),
            "avg_relations_per_keyword": sum(r["avg_relations_per_keyword"] for r in rows) / len(rows),
        }
    updated_payload["overall"] = overall

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(updated_payload, indent=2))

    with open(out_csv, "w", newline="") as f:
        fieldnames = list(updated_results[0].keys()) if updated_results else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated_results:
            writer.writerow(row)

    print(f"Updated sweep JSON: {out_json}")
    print(f"Updated sweep CSV: {out_csv}")


if __name__ == "__main__":
    main()
