"""Write all_results.json and macro_replacements.tex."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _fmt_acc(v: float) -> str:
    return f"{v:.1f}"


def _fmt_prob(v: float) -> str:
    return f"{v:.2f}"


def _fmt_p(v: float) -> str:
    if np.isnan(v):
        return "nan"
    if v < 0.001:
        return "<0.001"
    return f"{v:.3g}"


def write_results(
    all_results: dict,
    out_dir: str,
    quick_run: bool = False,
) -> None:
    """
    all_results structure expected:
    {
      "protocol_a": { model_name: { summary, fold_scores } },
      "protocol_b": { model_name: { summary, fold_scores } },
      "loso":       { model_name: { summary, fold_scores } },
      "significance": { wilcoxon_stat, wilcoxon_p, ttest_stat, ttest_p },
      "confusion_matrix": [[TP, FN], [FP, TN]],   # protocol_b CNN_BiLSTM pooled
      "shap": { channel_importance, band_importance },
      "timing": { train_min, infer_ms },
      "params": float,   # model parameter count in millions
    }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    # ── JSON dump ────────────────────────────────────────────────────────────
    def _serialisable(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return obj

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, default=_serialisable, indent=2)
    print(f"Saved {out_dir / 'all_results.json'}")

    # ── Convenience lookups ──────────────────────────────────────────────────
    def _mean(protocol: str, model: str, metric: str) -> float:
        return all_results[protocol][model]["summary"][metric][0]

    def _std(protocol: str, model: str, metric: str) -> float:
        return all_results[protocol][model]["summary"][metric][1]

    cnn = "CNN_BiLSTM"
    sig = all_results["significance"]
    cm  = all_results.get("confusion_matrix", [[0, 0], [0, 0]])
    timing = all_results.get("timing", {})

    acc_a  = _mean("protocol_a", cnn, "accuracy")
    acc_as = _std("protocol_a",  cnn, "accuracy")
    acc_b  = _mean("protocol_b", cnn, "accuracy")
    acc_bs = _std("protocol_b",  cnn, "accuracy")
    gap    = acc_a - acc_b

    # ── macro_replacements.tex ───────────────────────────────────────────────
    lines = []
    if quick_run:
        lines.append("% QUICK_RUN=True — do NOT paste these numbers into the paper.")
    lines.append("% Auto-generated -- paste into main.tex to replace the placeholder block.")

    def rn(cmd: str, val: str) -> str:
        return f"\\renewcommand{{\\{cmd}}}{{{val}}}"

    lines += [
        rn("AccA",    _fmt_acc(acc_a)),
        rn("AccAsd",  _fmt_acc(acc_as)),
        rn("PrecA",   _fmt_prob(_mean("protocol_a", cnn, "precision"))),
        rn("RecA",    _fmt_prob(_mean("protocol_a", cnn, "recall"))),
        rn("FoneA",   _fmt_prob(_mean("protocol_a", cnn, "f1"))),
        rn("AucA",    _fmt_prob(_mean("protocol_a", cnn, "roc_auc"))),
        rn("AccB",    _fmt_acc(acc_b)),
        rn("AccBsd",  _fmt_acc(acc_bs)),
        rn("PrecB",   _fmt_prob(_mean("protocol_b", cnn, "precision"))),
        rn("RecB",    _fmt_prob(_mean("protocol_b", cnn, "recall"))),
        rn("FoneB",   _fmt_prob(_mean("protocol_b", cnn, "f1"))),
        rn("AucB",    _fmt_prob(_mean("protocol_b", cnn, "roc_auc"))),
        rn("Gap",     _fmt_acc(gap)),
        rn("GapP",    _fmt_p(sig.get("wilcoxon_p", float("nan")))),
        rn("GapT",    _fmt_p(sig.get("ttest_p",    float("nan")))),
        rn("TP",      str(cm[0][0])),
        rn("FN",      str(cm[0][1])),
        rn("FP",      str(cm[1][0])),
        rn("TN",      str(cm[1][1])),
        rn("Params",  f"{all_results.get('params', 0):.2f}"),
        rn("TrainMin",str(int(timing.get("train_min", 0)))),
        rn("InferMs", str(int(timing.get("infer_ms", 0)))),
    ]

    # Per-model summary lines for all baselines (protocol B)
    for model_name in all_results.get("protocol_b", {}):
        safe = model_name.replace("_", "").replace("-", "")
        lines += [
            rn(f"Acc{safe}B",  _fmt_acc(_mean("protocol_b", model_name, "accuracy"))),
            rn(f"Auc{safe}B",  _fmt_prob(_mean("protocol_b", model_name, "roc_auc"))),
        ]

    tex_path = out_dir / "macro_replacements.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {tex_path}")

    # ── Figure CSVs ──────────────────────────────────────────────────────────
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    leakage_rows = [
        {
            "metric":     m,
            "protocol_A": _mean("protocol_a", cnn, m),
            "protocol_B": _mean("protocol_b", cnn, m),
        }
        for m in metrics
    ]
    pd.DataFrame(leakage_rows).to_csv(out_dir / "figures" / "leakage_bars.csv", index=False)

    baseline_rows = [
        {
            "model":         mn,
            "accuracy_mean": _mean("protocol_b", mn, "accuracy"),
            "accuracy_std":  _std("protocol_b",  mn, "accuracy"),
        }
        for mn in all_results.get("protocol_b", {})
    ]
    pd.DataFrame(baseline_rows).to_csv(out_dir / "figures" / "baseline_bars.csv", index=False)

    auc_rows = [
        {"model": mn, "auc": _mean("protocol_b", mn, "roc_auc")}
        for mn in all_results.get("protocol_b", {})
    ]
    pd.DataFrame(auc_rows).to_csv(out_dir / "figures" / "roc_aucs.csv", index=False)

    cm_df = pd.DataFrame(cm, columns=["Pred_ADHD", "Pred_Control"],
                         index=["True_ADHD", "True_Control"])
    cm_df.to_csv(out_dir / "figures" / "confusion_matrix.csv")

    print("Figure CSVs written.")


def print_protocol_summary(all_results: dict) -> None:
    """Print a clean per-protocol summary to stdout."""
    for proto_key, proto_label in [
        ("protocol_a", "PROTOCOL A (segment-wise)"),
        ("protocol_b", "PROTOCOL B (subject-wise)"),
        ("loso",       "PROTOCOL LOSO"),
    ]:
        if proto_key not in all_results:
            continue
        print(f"\n{proto_label}")
        for model_name, data in all_results[proto_key].items():
            s = data["summary"]
            acc_m, acc_s = s["accuracy"]
            auc_m, _     = s["roc_auc"]
            print(f"  {model_name:<22} acc={acc_m:.1f} ± {acc_s:.1f}  auc={auc_m:.3f}")

    sig = all_results.get("significance", {})
    cnn = "CNN_BiLSTM"
    if "protocol_a" in all_results and "protocol_b" in all_results:
        acc_a = all_results["protocol_a"][cnn]["summary"]["accuracy"][0]
        acc_b = all_results["protocol_b"][cnn]["summary"]["accuracy"][0]
        print(f"\nLEAKAGE GAP ({cnn})")
        print(f"  accuracy  A={acc_a:.1f}  B={acc_b:.1f}  gap={acc_a - acc_b:.1f}")
        print(f"  Wilcoxon  stat={sig.get('wilcoxon_stat', 'n/a'):.3f}  p={sig.get('wilcoxon_p', 'n/a')}")
        print(f"  Paired t  stat={sig.get('ttest_stat', 'n/a'):.3f}  p={sig.get('ttest_p', 'n/a')}")
