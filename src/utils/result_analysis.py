import json
from pathlib import Path

import numpy as np
import pandas as pd

# Force non-interactive backend (fixes PyCharm backend issues)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG / PATHS
# ----------------------------

# --- Locate project root (LunarLander/) from this file location ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # LunarLander/

# Your desired results location:
RESULTS_DIR = PROJECT_ROOT / "logs" / "results"

# Optional fallback if you ever keep results directly under project root
if not RESULTS_DIR.exists():
    alt = PROJECT_ROOT / "results"
    if alt.exists():
        RESULTS_DIR = alt
    else:
        raise FileNotFoundError(
            f"Could not find results directory. Tried:\n"
            f"- {PROJECT_ROOT / 'logs' / 'results'}\n"
            f"- {alt}\n"
            f"Current file: {THIS_FILE}"
        )

OUT_DIR = PROJECT_ROOT / "analysis_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CURRICULA = ["baseline", "param", "joint"]


# If your suite uses different prefixes, edit this.
def case_group(case_name: str) -> str:
    n = (case_name or "").lower()
    if n.startswith("trainlike"):
        return "training_like"
    if "combo" in n:
        return "unseen_combo"
    if "range" in n:
        return "unseen_range"
    return "other"


# ----------------------------
# LOAD ALL CHECKPOINTS
# ----------------------------
records = []

for curriculum in CURRICULA:
    for path in RESULTS_DIR.glob(f"{curriculum}/**/episodes.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                info = row.get("info", {})
                if info.get("type") != "eval_checkpoint":
                    continue

                suite = info.get("suite", {})
                records.append({
                    "curriculum": info.get("curriculum", curriculum),
                    "run_seed": info.get("run_seed"),
                    "trained_steps": info.get("trained_steps"),
                    "actual_timesteps": info.get("actual_timesteps"),
                    "checkpoint_kind": info.get("checkpoint_kind"),
                    "suite": suite,
                    "source_file": str(path),
                })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError(f"No eval_checkpoint records found under: {RESULTS_DIR.resolve()}")

# Basic sanity
missing = df[df["run_seed"].isna() | df["trained_steps"].isna() | df["checkpoint_kind"].isna()]
if not missing.empty:
    print("WARNING: Some records are missing run_seed/trained_steps/checkpoint_kind. First few:")
    print(missing.head())

# Normalize types
df["run_seed"] = pd.to_numeric(df["run_seed"], errors="coerce")
df["trained_steps"] = pd.to_numeric(df["trained_steps"], errors="coerce")
df["actual_timesteps"] = pd.to_numeric(df["actual_timesteps"], errors="coerce")

# ----------------------------
# INFER EXPECTED CASE COUNTS PER GROUP (from suite structure)
# ----------------------------
# Some early checkpoints may have incomplete suite coverage (e.g., only a subset of cases logged).
# If we average over a partial set, early success rates can be artificially inflated (e.g., start at 1.0).
# To keep plots honest, we infer the expected number of cases per group and mark incomplete checkpoints as NaN.
EXPECTED_CASES_PER_GROUP: dict[str, int] = {"training_like": 0, "unseen_combo": 0, "unseen_range": 0}
for suite in df["suite"].to_list():
    suite = suite or {}
    cases = suite.get("cases", []) or []
    counts: dict[str, int] = {}
    for c in cases:
        g = case_group(c.get("case", ""))
        if g == "other":
            continue
        counts[g] = counts.get(g, 0) + 1
    for g, n in counts.items():
        if g in EXPECTED_CASES_PER_GROUP:
            EXPECTED_CASES_PER_GROUP[g] = max(EXPECTED_CASES_PER_GROUP[g], int(n))

print("Expected cases per group (inferred from suites):", EXPECTED_CASES_PER_GROUP)



# ----------------------------
# ASSIGN CHECKPOINT INDEX PER RUN
# (x-axis for learning curves)
# ----------------------------
# Checkpoint index should follow training progress. trained_steps is monotonic by construction.
# If trained_steps is missing for some reason, fall back to actual_timesteps.
def _sort_key_frame(frame: pd.DataFrame) -> pd.Series:
    if frame["trained_steps"].notna().any():
        return frame["trained_steps"].fillna(-1)
    return frame["actual_timesteps"].fillna(-1)

df = df.sort_values(["curriculum", "run_seed"])
df["_sort_key"] = df.groupby(["curriculum", "run_seed"], group_keys=False).apply(_sort_key_frame, include_groups=False)
df = df.sort_values(["curriculum", "run_seed", "_sort_key"]).drop(columns=["_sort_key"])

df["checkpoint_idx"] = df.groupby(["curriculum", "run_seed"]).cumcount() + 1

# Save checkpoint index table (useful for debugging)
df.to_csv(OUT_DIR / "raw_checkpoints_with_idx.csv", index=False)


# ----------------------------
# Estimate env-steps per checkpoint (for figure footnotes)
# ----------------------------
# This uses actual_timesteps when available.
def estimate_env_steps_per_checkpoint(frame: pd.DataFrame) -> float | None:
    f = frame.dropna(subset=["actual_timesteps", "checkpoint_idx"]).copy()
    if f.empty:
        return None
    f = f.sort_values(["curriculum", "run_seed", "checkpoint_idx"])
    f["d"] = f.groupby(["curriculum", "run_seed"])["actual_timesteps"].diff()
    d = f["d"].dropna()
    if d.empty:
        return None
    return float(np.median(d.to_numpy()))

EST_ENV_STEPS = estimate_env_steps_per_checkpoint(df)
if EST_ENV_STEPS is not None:
    print(f"Estimated environment interactions per checkpoint (median): {EST_ENV_STEPS:,.0f}")
else:
    print("Estimated environment interactions per checkpoint: unavailable (actual_timesteps missing).")


# ----------------------------
# BUILD PER-GROUP METRICS AT EACH CHECKPOINT
# ----------------------------
rows = []
for _, r in df.iterrows():
    suite = r.suite or {}
    cases = suite.get("cases", [])

    # group cases for this checkpoint
    grouped = {}
    for c in cases:
        g = case_group(c.get("case", ""))
        if g == "other":
            continue
        grouped.setdefault(g, []).append(c)

    for g, cs in grouped.items():
        n_used = len(cs)
        expected = int(EXPECTED_CASES_PER_GROUP.get(g, n_used))
        coverage = (n_used / expected) if expected > 0 else 0.0
        is_valid = (expected > 0) and (n_used == expected)

        # If suite coverage is incomplete at this checkpoint, mark metrics as NaN so plots don't lie.
        mean_return = float(np.mean([x.get("mean_return", np.nan) for x in cs])) if is_valid else float("nan")
        mean_success_strict = float(np.mean([x.get("strict_success_rate", np.nan) for x in cs])) if is_valid else float("nan")
        mean_success_soft   = float(np.mean([x.get("soft_success_rate", np.nan) for x in cs])) if is_valid else float("nan")
        mean_length = float(np.mean([x.get("mean_length", np.nan) for x in cs])) if is_valid else float("nan")

        rows.append({
            "curriculum": r.curriculum,
            "run_seed": int(r.run_seed) if pd.notna(r.run_seed) else None,
            "checkpoint_idx": int(r.checkpoint_idx) if pd.notna(r.checkpoint_idx) else None,
            "trained_steps": int(r.trained_steps) if pd.notna(r.trained_steps) else None,
            "actual_timesteps": int(r.actual_timesteps) if pd.notna(r.actual_timesteps) else None,
            "checkpoint_kind": r.checkpoint_kind,
            "group": g,
            "n_cases_used": int(n_used),
            "expected_cases": int(expected),
            "coverage": float(coverage),
            "is_valid": bool(is_valid),
            "mean_return": mean_return,
            "mean_success_strict": mean_success_strict,
            "mean_success_soft": mean_success_soft,
            "mean_length": mean_length,
        })

per_group = pd.DataFrame(rows).dropna(
    subset=["run_seed", "checkpoint_idx", "group"]
)
per_group = per_group.sort_values(["group", "curriculum", "run_seed", "checkpoint_idx"])
per_group.to_csv(OUT_DIR / "per_group_checkpoints.csv", index=False)

# Add combined generalization group = average of unseen_combo + unseen_range
gen = per_group[per_group["group"].isin(["unseen_combo", "unseen_range"])].copy()
if not gen.empty:
    gen_combined = (
        gen.groupby(["curriculum", "run_seed", "checkpoint_idx", "trained_steps", "actual_timesteps", "checkpoint_kind"],
                    as_index=False)
        .agg(
            mean_return=("mean_return", "mean"),
            mean_success_strict=("mean_success_strict", "mean"),
            mean_success_soft=("mean_success_soft", "mean"),
            mean_length=("mean_length", "mean"),
        )
    )
    gen_combined["group"] = "generalization_combined"
    per_group = pd.concat([per_group, gen_combined], ignore_index=True)
    per_group = per_group.sort_values(["group", "curriculum", "run_seed", "checkpoint_idx"])
    per_group.to_csv(OUT_DIR / "per_group_checkpoints_with_combined.csv", index=False)


# ----------------------------
# LEARNING CURVES (per group) — X axis = checkpoint index
# ----------------------------
def curve_stats_for_group(group: str) -> pd.DataFrame:
    sub = per_group[per_group["group"] == group].copy()
    if sub.empty:
        return sub

    stats = (
        sub.groupby(["curriculum", "checkpoint_idx"], as_index=False)
        .agg(
            mean_return=("mean_return", "mean"),
            std_return=("mean_return", "std"),
            mean_success_strict=("mean_success_strict", "mean"),
            std_success_strict=("mean_success_strict", "std"),
            mean_success_soft=("mean_success_soft", "mean"),
            std_success_soft=("mean_success_soft", "std"),
            mean_length=("mean_length", "mean"),
            std_length=("mean_length", "std"),
            n_seeds=("run_seed", "nunique"),
        )
        .sort_values(["curriculum", "checkpoint_idx"])
    )
    return stats


def save_learning_curve(stats: pd.DataFrame, y_mean: str, y_std: str, ylabel: str, title: str, filename: str):
    plt.figure(figsize=(9, 5))
    for c in CURRICULA:
        s = stats[stats["curriculum"] == c]
        if s.empty:
            continue

        xs = s["checkpoint_idx"].to_numpy()
        ym = s[y_mean].to_numpy()
        ys = s[y_std].to_numpy()

        n = int(s["n_seeds"].max()) if "n_seeds" in s.columns and pd.notna(s["n_seeds"].max()) else 0
        plt.plot(xs, ym, label=f"{c} (n={n})")
        if np.isfinite(ys).any():
            plt.fill_between(xs, ym - ys, ym + ys, alpha=0.2)

    plt.xlabel("Evaluation checkpoints")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Nice integer ticks
    if not stats.empty and "checkpoint_idx" in stats.columns:
        max_ckpt = int(stats["checkpoint_idx"].max())

        ticks = sorted(set(
            [1]
            + list(range(5, max_ckpt + 1, 5))
            + ([max_ckpt] if max_ckpt not in (1,) else [])
        ))

        plt.xticks(ticks)

    # Footnote about env-steps per checkpoint (when available)
    if EST_ENV_STEPS is not None:
        foot = f"Approx. {EST_ENV_STEPS:,.0f} environment interactions per checkpoint (median over runs)"
        plt.figtext(0.5, 0.01, foot, ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


# RQ1: learning speed on training-like
train_stats = curve_stats_for_group("training_like")
train_stats.to_csv(OUT_DIR / "learning_curve_training_like.csv", index=False)
save_learning_curve(train_stats, "mean_success_strict", "std_success_strict",
                    "Strict success rate (training-like avg)",
                    "Learning curve (training-like): Strict success rate",
                    "lc_trainlike_success_strict.png")
save_learning_curve(train_stats, "mean_success_soft", "std_success_soft",
                    "Soft success rate (training-like avg)",
                    "Learning curve (training-like): Soft success rate",
                    "lc_trainlike_success_soft.png")
save_learning_curve(train_stats, "mean_return", "std_return",
                    "Mean return (training-like avg)",
                    "Learning curve (training-like): Mean return",
                    "lc_trainlike_return.png")

# RQ2: learning curves on unseen groups (optional)
combo_stats = curve_stats_for_group("unseen_combo")
combo_stats.to_csv(OUT_DIR / "learning_curve_unseen_combo.csv", index=False)
save_learning_curve(combo_stats, "mean_success_strict", "std_success_strict",
                    "Strict success rate (unseen-combo avg)",
                    "Learning curve (unseen-combo): Strict success rate",
                    "lc_combo_success_strict.png")
save_learning_curve(combo_stats, "mean_success_soft", "std_success_soft",
                    "Soft success rate (unseen-combo avg)",
                    "Learning curve (unseen-combo): Soft success rate",
                    "lc_combo_success_soft.png")
save_learning_curve(combo_stats, "mean_return", "std_return",
                    "Mean return (unseen-combo avg)",
                    "Learning curve (unseen-combo): Mean return",
                    "lc_combo_return.png")

range_stats = curve_stats_for_group("unseen_range")
range_stats.to_csv(OUT_DIR / "learning_curve_unseen_range.csv", index=False)
save_learning_curve(range_stats, "mean_success_strict", "std_success_strict",
                    "Strict success rate (unseen-range avg)",
                    "Learning curve (unseen-range): Strict success rate",
                    "lc_range_success_strict.png")
save_learning_curve(range_stats, "mean_success_soft", "std_success_soft",
                    "Soft success rate (unseen-range avg)",
                    "Learning curve (unseen-range): Soft success rate",
                    "lc_range_success_soft.png")
save_learning_curve(range_stats, "mean_return", "std_return",
                    "Mean return (unseen-range avg)",
                    "Learning curve (unseen-range): Mean return",
                    "lc_range_return.png")

# Combined generalization learning curves
gen_stats = curve_stats_for_group("generalization_combined")
gen_stats.to_csv(OUT_DIR / "learning_curve_generalization_combined.csv", index=False)
save_learning_curve(gen_stats, "mean_success_strict", "std_success_strict",
                    "Strict success rate (combo+range avg)",
                    "Learning curve (combined generalization): Strict success rate",
                    "lc_generalization_success_strict.png")
save_learning_curve(gen_stats, "mean_success_soft", "std_success_soft",
                    "Soft success rate (combo+range avg)",
                    "Learning curve (combined generalization): Soft success rate",
                    "lc_generalization_success_soft.png")
save_learning_curve(gen_stats, "mean_return", "std_return",
                    "Mean return (combo+range avg)",
                    "Learning curve (combined generalization): Mean return",
                    "lc_generalization_return.png")


# ----------------------------
# FINAL PERFORMANCE (final checkpoint only) + BAR PLOTS
# ----------------------------
final_only = per_group[per_group["checkpoint_kind"] == "final"].copy()
if final_only.empty:
    raise RuntimeError("No final checkpoints found (checkpoint_kind == 'final').")

final_stats = (
    final_only.groupby(["curriculum", "group"], as_index=False)
    .agg(
        mean_return=("mean_return", "mean"),
        std_return=("mean_return", "std"),
        mean_success_strict=("mean_success_strict", "mean"),
        std_success_strict=("mean_success_strict", "std"),
        mean_success_soft=("mean_success_soft", "mean"),
        std_success_soft=("mean_success_soft", "std"),
        mean_length=("mean_length", "mean"),
        std_length=("mean_length", "std"),
        n_seeds=("run_seed", "nunique"),
    )
)

order_groups = ["training_like", "unseen_combo", "unseen_range", "generalization_combined"]
final_stats["group"] = pd.Categorical(final_stats["group"], categories=order_groups, ordered=True)
final_stats = final_stats.sort_values(["group", "curriculum"]).reset_index(drop=True)
final_stats.to_csv(OUT_DIR / "final_group_stats.csv", index=False)


def bar_final(group: str, metric_mean: str, metric_std: str, ylabel: str, title: str, filename: str):
    s = final_stats[final_stats["group"] == group].copy()
    if s.empty:
        print(f"WARNING: No final stats for group '{group}', skipping plot {filename}")
        return

    s["curriculum"] = pd.Categorical(s["curriculum"], categories=CURRICULA, ordered=True)
    s = s.sort_values("curriculum")

    x = np.arange(len(s))
    y = s[metric_mean].to_numpy()
    err = s[metric_std].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.bar(x, y, yerr=err, capsize=5)
    plt.xticks(x, s["curriculum"].to_list())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


# RQ1 final performance (training-like)
bar_final("training_like", "mean_success_strict", "std_success_strict",
          "Strict success rate",
          "Final performance (training-like): Strict success rate",
          "final_trainlike_success_strict.png")
bar_final("training_like", "mean_success_soft", "std_success_soft",
          "Soft success rate",
          "Final performance (training-like): Soft success rate",
          "final_trainlike_success_soft.png")
bar_final("training_like", "mean_return", "std_return",
          "Mean return",
          "Final performance (training-like): Mean return",
          "final_trainlike_return.png")

# RQ2 generalization (separate)
bar_final("unseen_combo", "mean_success_strict", "std_success_strict",
          "Strict success rate",
          "Final generalization (unseen-combo): Strict success rate",
          "final_combo_success_strict.png")
bar_final("unseen_combo", "mean_success_soft", "std_success_soft",
          "Soft success rate",
          "Final generalization (unseen-combo): Soft success rate",
          "final_combo_success_soft.png")
bar_final("unseen_combo", "mean_return", "std_return",
          "Mean return",
          "Final generalization (unseen-combo): Mean return",
          "final_combo_return.png")

bar_final("unseen_range", "mean_success_strict", "std_success_strict",
          "Strict success rate",
          "Final generalization (unseen-range): Strict success rate",
          "final_range_success_strict.png")
bar_final("unseen_range", "mean_success_soft", "std_success_soft",
          "Soft success rate",
          "Final generalization (unseen-range): Soft success rate",
          "final_range_success_soft.png")
bar_final("unseen_range", "mean_return", "std_return",
          "Mean return",
          "Final generalization (unseen-range): Mean return",
          "final_range_return.png")

# Combined generalization (combo+range average)
bar_final("generalization_combined", "mean_success_strict", "std_success_strict",
          "Strict success rate",
          "Final generalization (combined): Strict success rate",
          "final_generalization_success_strict.png")
bar_final("generalization_combined", "mean_success_soft", "std_success_soft",
          "Soft success rate",
          "Final generalization (combined): Soft success rate",
          "final_generalization_success_soft.png")
bar_final("generalization_combined", "mean_return", "std_return",
          "Mean return",
          "Final generalization (combined): Mean return",
          "final_generalization_return.png")


# ----------------------------
# LaTeX EXPORTS
# ----------------------------
def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"


# Full final table (training_like + combo + range)
latex_rows = []
for _, r in final_stats[final_stats["group"].isin(["training_like", "unseen_combo", "unseen_range"])].iterrows():
    latex_rows.append({
        "Group": str(r["group"]),
        "Curriculum": str(r["curriculum"]),
        "Return": format_mean_std(r["mean_return"], r["std_return"], decimals=1),
        "StrictSuccess": format_mean_std(100.0 * r["mean_success_strict"], 100.0 * r["std_success_strict"], decimals=1) + "\\%",
        "SoftSuccess":   format_mean_std(100.0 * r["mean_success_soft"],   100.0 * r["std_success_soft"],   decimals=1) + "\\%",
        "Length": format_mean_std(r["mean_length"], r["std_length"], decimals=0),
        "n": int(r["n_seeds"]),
    })

latex_df = pd.DataFrame(latex_rows)
latex_str = latex_df.to_latex(
    index=False,
    escape=False,
    column_format="lllrllr",
    caption="Final performance (mean $\\pm$ std over seeds) by evaluation group.",
    label="tab:final_performance_groups",
)
(OUT_DIR / "final_performance_table.tex").write_text(latex_str, encoding="utf-8")

# Combined generalization LaTeX table (headline)
gen_final = final_stats[final_stats["group"] == "generalization_combined"].copy()
gen_rows = []
for _, r in gen_final.iterrows():
    gen_rows.append({
        "Curriculum": str(r["curriculum"]),
        "Return": format_mean_std(r["mean_return"], r["std_return"], decimals=1),
        "StrictSuccess": format_mean_std(100.0 * r["mean_success_strict"], 100.0 * r["std_success_strict"], decimals=1) + "\\%",
        "SoftSuccess":   format_mean_std(100.0 * r["mean_success_soft"],   100.0 * r["std_success_soft"],   decimals=1) + "\\%",
        "n": int(r["n_seeds"]),
    })

gen_df = pd.DataFrame(gen_rows)
gen_latex = gen_df.to_latex(
    index=False,
    escape=False,
    column_format="lrrr",
    caption="Final combined generalization (unseen-combo and unseen-range averaged). Values are mean $\\pm$ std over seeds.",
    label="tab:final_generalization_combined",
)
(OUT_DIR / "final_generalization_table.tex").write_text(gen_latex, encoding="utf-8")


print("\nWrote outputs to:", OUT_DIR.resolve())
print("Learning curves (PNG):")
print(" - lc_trainlike_success.png / lc_trainlike_return.png")
print(" - lc_combo_success.png / lc_combo_return.png")
print(" - lc_range_success.png / lc_range_return.png")
print(" - lc_generalization_success.png / lc_generalization_return.png")
print("Final bar plots (PNG):")
print(" - final_trainlike_success.png / final_trainlike_return.png")
print(" - final_combo_success.png / final_combo_return.png")
print(" - final_range_success.png / final_range_return.png")
print(" - final_generalization_success.png / final_generalization_return.png")
print("Tables:")
print(" - final_group_stats.csv")
print(" - final_performance_table.tex")
print(" - final_generalization_table.tex")
print("Debug CSVs:")
print(" - raw_checkpoints_with_idx.csv")
print(" - per_group_checkpoints.csv")
print(" - per_group_checkpoints_with_combined.csv")