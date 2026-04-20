import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required to plot results: {exc}")

    results_path = Path(args.results_json)
    payload = json.loads(results_path.read_text())
    results = payload.get("results", [])
    if not results:
        raise SystemExit("No results found in JSON file.")

    by_category = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)

    for category_rows in by_category.values():
        category_rows.sort(key=lambda r: r["theta"])

    thetas = sorted({row["theta"] for row in results})
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    for category, rows in sorted(by_category.items()):
        ax.plot(
            [row["theta"] for row in rows],
            [row["success_rate"] for row in rows],
            marker="o",
            label=category,
        )
    ax.set_ylabel("Success Rate")
    ax.set_title("Threshold Sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for category, rows in sorted(by_category.items()):
        ax.plot(
            [row["theta"] for row in rows],
            [row["avg_relations_per_object"] for row in rows],
            marker="o",
            label=f"{category} object",
        )
        ax.plot(
            [row["theta"] for row in rows],
            [row["avg_relations_per_keyword"] for row in rows],
            marker="s",
            linestyle="--",
            label=f"{category} keyword",
        )
    ax.set_ylabel("Avg Kept Relations")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[2]
    for category, rows in sorted(by_category.items()):
        ax.plot(
            [row["theta"] for row in rows],
            [row["relations_kept"] / row["relations_total"] if row["relations_total"] else 0.0 for row in rows],
            marker="o",
            label=f"{category} overall kept/total",
        )
        ax.plot(
            [row["theta"] for row in rows],
            [row["ope_relations_kept"] / row["ope_relations_total"] if row["ope_relations_total"] else 0.0 for row in rows],
            marker="^",
            linestyle="--",
            label=f"{category} OPE kept/total",
        )
        ax.plot(
            [row["theta"] for row in rows],
            [row["urp_relations_kept"] / row["urp_relations_total"] if row["urp_relations_total"] else 0.0 for row in rows],
            marker="v",
            linestyle=":",
            label=f"{category} URP kept/total",
        )
    ax.set_xlabel("Theta")
    ax.set_ylabel("Kept Ratio")
    ax.set_xticks(thetas)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    out_path = Path(args.out) if args.out else results_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
