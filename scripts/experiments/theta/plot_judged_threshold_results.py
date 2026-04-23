import argparse
import json
from pathlib import Path


def _plot_with_pillow(aggregate, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1600, 1000
    margin_left, margin_right = 110, 60
    margin_top, margin_bottom = 70, 80
    gap = 70
    plot_width = width - margin_left - margin_right
    panel_height = (height - margin_top - margin_bottom - gap) // 2

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    thetas = [row["theta"] for row in aggregate]
    success_rates = [row["success_rate"] for row in aggregate]
    deterministic_counts = [row["num_deterministic"] for row in aggregate]
    llm_counts = [row["num_llm_batch"] for row in aggregate]
    max_count = max(deterministic_counts + llm_counts + [1])
    category = aggregate[0].get("category", "unknown")

    plot1 = (margin_left, margin_top, width - margin_right, margin_top + panel_height)
    plot2 = (
        margin_left,
        margin_top + panel_height + gap,
        width - margin_right,
        margin_top + 2 * panel_height + gap,
    )

    def x_for(theta: float) -> float:
        if len(thetas) == 1:
            return (plot1[0] + plot1[2]) / 2
        theta_min = min(thetas)
        theta_max = max(thetas)
        return plot1[0] + (theta - theta_min) * (plot_width / (theta_max - theta_min))

    def y_rate(rate: float) -> float:
        return plot1[3] - rate * (plot1[3] - plot1[1])

    def y_count(count: float) -> float:
        return plot2[3] - (count / max_count) * (plot2[3] - plot2[1])

    for plot in (plot1, plot2):
        draw.rectangle(plot, outline="black", width=2)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_rate(frac)
        draw.line((plot1[0], y, plot1[2], y), fill="#dddddd", width=1)
        draw.text((20, y - 8), f"{frac:.2f}", fill="black")

    for idx in range(max_count + 1):
        y = y_count(idx)
        draw.line((plot2[0], y, plot2[2], y), fill="#eeeeee", width=1)
        draw.text((20, y - 8), str(idx), fill="black")

    for theta in thetas:
        x = x_for(theta)
        draw.line((x, plot1[3], x, plot1[3] + 8), fill="black", width=1)
        draw.line((x, plot2[3], x, plot2[3] + 8), fill="black", width=1)
        label = f"{theta:g}"
        draw.text((x - 12, plot2[3] + 14), label, fill="black")

    success_points = [(x_for(theta), y_rate(rate)) for theta, rate in zip(thetas, success_rates)]
    if len(success_points) > 1:
        draw.line(success_points, fill="#1565c0", width=4)
    for x, y in success_points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#1565c0")

    bar_width = max(10, int(plot_width / max(len(thetas) * 6, 10)))
    for theta, det, llm in zip(thetas, deterministic_counts, llm_counts):
        x = x_for(theta)
        draw.rectangle((x - bar_width - 2, y_count(det), x - 2, plot2[3]), fill="#2e7d32")
        draw.rectangle((x + 2, y_count(llm), x + bar_width + 2, plot2[3]), fill="#ef6c00")

    draw.text((margin_left, 20), f"Judged Threshold Sweep: {category}", fill="black")
    draw.text((margin_left, plot1[3] + 16), "Success Rate", fill="black")
    draw.text((margin_left, plot2[1] - 26), "Item Count by Judge Mode", fill="black")
    draw.text((width - 270, 20), "blue: success rate", fill="#1565c0")
    draw.text((width - 270, 42), "green: deterministic", fill="#2e7d32")
    draw.text((width - 270, 64), "orange: llm_batch", fill="#ef6c00")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_path = Path(args.results_json)
    payload = json.loads(results_path.read_text())
    aggregate = payload.get("aggregate", [])
    if not aggregate:
        raise SystemExit("No aggregate results found in JSON file.")

    aggregate = sorted(aggregate, key=lambda row: row["theta"])
    thetas = [row["theta"] for row in aggregate]
    success_rates = [row["success_rate"] for row in aggregate]
    deterministic_counts = [row["num_deterministic"] for row in aggregate]
    llm_counts = [row["num_llm_batch"] for row in aggregate]
    category = aggregate[0].get("category", "unknown")
    out_path = Path(args.out) if args.out else results_path.with_suffix(".png")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        _plot_with_pillow(aggregate, out_path)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(thetas, success_rates, marker="o", linewidth=2)
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Judged Threshold Sweep: {category}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    width = 0.018 if len(thetas) > 1 else 0.05
    ax.bar([theta - width / 2 for theta in thetas], deterministic_counts, width=width, label="deterministic")
    ax.bar([theta + width / 2 for theta in thetas], llm_counts, width=width, label="llm_batch")
    ax.set_xlabel("Theta")
    ax.set_ylabel("Item Count")
    ax.set_xticks(thetas)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
