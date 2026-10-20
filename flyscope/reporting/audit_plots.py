"""Standalone figures for existing comparison audits; no training or tuning."""

import html


def write_audit_plots(reports, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    full = [r for r in reports if r["run"] == "all-annotated-wells"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, report in enumerate(full):
        axes[0].bar(
            i - 0.18,
            report["raw_accuracy"],
            width=0.36,
            label="Raw frames" if i == 0 else None,
        )
        axes[0].bar(
            i + 0.18,
            report["decision_accuracy"],
            width=0.36,
            label="Recovery decisions" if i == 0 else None,
        )
        axes[1].bar(i, report["raw_coma_recall"])
    names = [r["architecture"] for r in full]
    for axis in axes:
        axis.set_xticks(range(len(full)), names)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    if full:
        axes[0].axhline(
            full[0]["always_awake_accuracy"],
            linestyle="--",
            color="gray",
            label="Always Awake baseline",
        )
    axes[0].set(title="Full training selection: test accuracy", ylabel="Accuracy")
    axes[1].set(title="Ability to recognize Coma (raw output)", ylabel="Coma recall")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "comparison-audit.png", dpi=150)
    plt.close(fig)
    rows = []
    for report in full:
        recovery = report["metrics"]["recovery"]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    report["architecture"],
                    f"{report['raw_accuracy']:.1%}",
                    f"{report['decision_accuracy']:.1%}",
                    f"{report['raw_coma_recall']:.1%}",
                    report["first_frame_decision_awake_wells"],
                    recovery["paired_observed_wells"],
                    recovery["mae_frames"],
                )
            )
            + "</tr>"
        )
    document = (
        """<!doctype html><meta charset="utf-8"><title>Training comparison audit</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #bbb}img{max-width:100%}</style>
<h1>Comparison audit</h1><p>These are exploratory diagnostics of existing test results, not a new validation experiment. Raw outputs and irreversible recovery decisions are distinct. Recovery MAE excludes censored/uncertain events: compare its paired-well denominator before ranking models.</p>
<img src="comparison-audit.png" alt="Raw and recovery accuracy versus the Always Awake baseline, and Coma recall">
<table><tr><th>Architecture</th><th>Raw accuracy</th><th>Decision accuracy</th><th>Coma recall</th><th>Awake at first frame (wells)</th><th>Paired recovery wells</th><th>MAE (frames)</th></tr>"""
        + "".join(rows)
        + "</table><p>See model-audits.json for all selections, class counts, confidence errors, calibration and censoring. No weights or source images are included.</p>"
    )
    (output / "report.html").write_text(document)
