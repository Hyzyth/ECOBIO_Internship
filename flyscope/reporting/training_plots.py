#!/usr/bin/env python3
"""Plot live or completed model_suite training.csv without reading checkpoints."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def plot_training(source, destination):
    with Path(source).open(newline="") as stream:
        return plot_epochs(csv.DictReader(stream), destination)


def plot_epochs(rows, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(lambda: defaultdict(list))
    validation = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (
            row.get("event") == "epoch"
            and row.get("epoch")
            and row.get("loss") not in (None, "")
        ):
            groups[(row["architecture"], row["phase"])][row["run"]].append(
                (int(row["epoch"]), float(row["loss"]))
            )
            if row.get("validation_loss") not in (None, ""):
                validation[(row["architecture"], row["phase"])][row["run"]].append(
                    (int(row["epoch"]), float(row["validation_loss"]))
                )
    if not groups:
        return False
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(10, 3.5 * len(groups)), squeeze=False
    )
    for axis, ((architecture, phase), runs) in zip(axes.flat, groups.items()):
        for name, rows in runs.items():
            (line,) = axis.plot(*zip(*rows), marker="o", label=name + " / train")
            held_out = validation[(architecture, phase)].get(name)
            if held_out:
                axis.plot(
                    *zip(*held_out),
                    marker="x",
                    linestyle="--",
                    color=line.get_color(),
                    label=name + " / validation",
                )
        axis.set(
            title=f"{architecture} / {phase}", xlabel="Epoch", ylabel="Training loss"
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    fig.suptitle("Training and validation loss (test data never select checkpoints)")
    fig.tight_layout()
    destination = Path(destination)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    try:
        fig.savefig(temporary, dpi=130)
        temporary.replace(destination)
    finally:
        plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to training.csv")
    parser.add_argument("--output", required=True, help="PNG, SVG or PDF path")
    args = parser.parse_args()
    if not plot_training(args.input, args.output):
        parser.exit(
            1,
            "No completed epoch yet; training.csv still contains live batch losses.\n",
        )


if __name__ == "__main__":
    main()
