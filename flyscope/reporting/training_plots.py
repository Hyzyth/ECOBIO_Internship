"""Separate fitting/validation curves; forest diagnostics are not epoch curves."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

PHASES = {
    "appearance": "ResNet-18 · classification (weighted cross-entropy)",
    "optical_flow": "DeepEthogram · 1/3 motion (photometric + smoothness)",
    "two_stream": "DeepEthogram · 2/3 appearance + motion (weighted cross-entropy)",
    "temporal_tgm": "DeepEthogram · 3/3 temporal classification (weighted cross-entropy)",
    "pose": "SLEAP anatomy (balanced heatmap error)",
}


def plot_training(source, destination):
    with Path(source).open(newline="") as stream:
        return plot_epochs(csv.DictReader(stream), destination)


def plot_epochs(rows, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(lambda: defaultdict(list))
    forests = []
    for row in rows:
        if row.get("event") == "fit_metrics":
            forests.append(row)
        if (
            row.get("event") == "epoch"
            and row.get("epoch")
            and row.get("loss") not in (None, "")
        ):
            groups[(row["architecture"], row["phase"])][row["run"]].append(row)
    if not groups and not forests:
        return False
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save(fig, path):
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        fig.savefig(temporary, dpi=140, bbox_inches="tight")
        temporary.replace(path)
        plt.close(fig)

    # Independent axes/legends prevent validation values from flattening training.
    for variant, fields in [
        ("combined", ["loss", "validation_loss"]),
        ("train", ["loss"]),
        ("validation", ["validation_loss"]),
    ]:
        if not groups:
            break
        fig, axes = plt.subplots(
            len(groups),
            len(fields),
            figsize=(7 * len(fields), 4.5 * len(groups)),
            squeeze=False,
            layout="constrained",
        )
        for row_index, ((arch, phase), runs) in enumerate(groups.items()):
            for col, field in enumerate(fields):
                axis = axes[row_index, col]
                count = 0
                for name, history in runs.items():
                    values = [
                        (int(r["epoch"]), float(r[field]))
                        for r in history
                        if r.get(field) not in (None, "")
                    ]
                    if values:
                        axis.plot(*zip(*values), marker="o", markersize=3, label=name)
                        count += 1
                title = "Training" if field == "loss" else "Validation"
                axis.set(
                    title=f'{PHASES.get(phase,arch+" / "+phase)}\n{title}',
                    xlabel="Epoch",
                    ylabel="Loss (phase-specific units)",
                )
                axis.grid(alpha=0.25)
                if count:
                    axis.legend(
                        fontsize=7,
                        ncol=2,
                        loc="upper left",
                        bbox_to_anchor=(0, -0.2),
                        frameon=False,
                    )
                else:
                    axis.text(
                        0.5, 0.5, "Not recorded", ha="center", transform=axis.transAxes
                    )
        fig.suptitle(
            "Training and independent validation — different axes; test data never select checkpoints",
            fontsize=12,
        )
        path = (
            destination
            if variant == "combined"
            else destination.with_name(
                destination.stem + "-" + variant + destination.suffix
            )
        )
        save(fig, path)
    if forests:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        names = [r["run"] for r in forests]
        for axis, field, label in zip(
            axes,
            ["loss", "validation_loss"],
            ["Training balanced NLL", "Validation balanced NLL"],
        ):
            values = [
                float(r[field]) if r.get(field) not in (None, "") else float("nan")
                for r in forests
            ]
            axis.barh(names, values)
            if not any(r.get(field) not in (None, "") for r in forests):
                axis.text(
                    0.5,
                    0.5,
                    "Not recorded in these checkpoints",
                    ha="center",
                    transform=axis.transAxes,
                )
            axis.set(
                title=label,
                xlabel="Negative log-likelihood (one fitted forest; no epochs)",
            )
            axis.grid(axis="x", alpha=0.2)
        forest_path = destination.with_name(
            destination.stem + "-forests" + destination.suffix
        )
        save(fig, forest_path)
        if not groups:
            import shutil

            shutil.copyfile(forest_path, destination)
    return True


def model_rows(root):
    for path in sorted(Path(root).rglob("model.json")):
        meta = json.loads(path.read_text())
        if meta.get("kind") != "flyscope-suite-model":
            continue
        training = meta["training"]
        arch = meta["architecture"]
        run = meta["selection"]["id"]
        for phase, history in training.get("loss", {}).items():
            held = training.get("validation_loss", {}).get(phase, [])
            for epoch, value in enumerate(history):
                yield {
                    "event": "epoch",
                    "architecture": arch,
                    "run": run,
                    "phase": phase,
                    "epoch": epoch + 1,
                    "loss": value,
                    "validation_loss": held[epoch] if epoch < len(held) else None,
                }
        if arch == "sleap-rf":
            fit = training.get("fit_metrics") or {}
            val = (meta.get("validation") or {}).get("forest_metrics") or {}
            yield {
                "event": "fit_metrics",
                "architecture": arch,
                "run": run,
                "phase": "random_forests",
                "loss": fit.get("balanced_nll"),
                "validation_loss": val.get("balanced_nll"),
            }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="training.csv")
    source.add_argument(
        "--models", help="Checkpoint metadata directory; weights are not loaded"
    )
    p.add_argument("--output", required=True, help="PNG, SVG or PDF")
    args = p.parse_args()
    ok = (
        plot_epochs(model_rows(args.models), args.output)
        if args.models
        else plot_training(args.input, args.output)
    )
    if not ok:
        p.exit(1, "No completed epoch or fitted forest metrics were recorded.\n")


if __name__ == "__main__":
    main()
