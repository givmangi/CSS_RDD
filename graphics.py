"""
Figures and tables for the paper
====================================
Runs LAST, after panel_pipeline.py and dynasty_network.py have already
produced real numbers. This script's job is to visualize and tabulate
what those already found, not to compute anything new from scratch.

Palette: deep purple accents throughout, with a muted gold reserved
specifically for marking the threshold line or highlighting a finding,
so it stays meaningful rather than decorative.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from panel import (
    get_panel, rdd_bandwidth_sample, compute_tenure_panel,
    compute_hierarchy_climb, fit_tenure_cox_model, fit_climb_logit_model,
    fit_climb_logit_flexible,
)
from dynastynet import flag_dynastic_ties, fit_dynasty_logit
from dynasty_crossreference import merge_dynasty_onto_climb, fit_dynasty_moderation_model

PURPLE_DARK = "#4B0082"
PURPLE_LIGHT = "#B497D6"
ACCENT_GOLD = "#D4A017"
NEUTRAL_GRAY = "#4d4d4d"

sns.set_theme(style="whitegrid", font_scale=1.05)
sns.set_palette([PURPLE_DARK, PURPLE_LIGHT])
plt.rcParams["axes.edgecolor"] = NEUTRAL_GRAY
plt.rcParams["figure.dpi"] = 150

FIG_DIR = Path("figures")
TAB_DIR = Path("tables")
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)


def figure_rd_climb(full_panel, bandwidth=3000, bins=15):
    """
    RD-style binned plot for H2: hierarchy-climb rate against the
    population running variable, by sex, with the threshold marked.
    Point size scales with bin count, so a bin resting on very few
    people does not visually overstate itself.

    Now also draws a separate local-linear (OLS) fit line on each side
    of the cutoff, per sex, since the earlier version without fit lines
    made it hard to tell a genuine discontinuity from a smooth trend by
    eye alone - which turned out to matter, since the smooth trend is
    exactly what motivated fit_climb_logit_flexible() in the first
    place. The gap between the two fit lines at running_var=0 is the
    visual equivalent of that model's at-cutoff estimate.
    """
    sample = rdd_bandwidth_sample(full_panel, bandwidth=bandwidth)
    climb = compute_hierarchy_climb(sample)
    climb = climb.dropna(subset=["first_running_var", "sesso"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for sex, color, label in [("F", PURPLE_DARK, "Women"), ("M", PURPLE_LIGHT, "Men")]:
        sub = climb[climb["sesso"] == sex].copy()
        sub["bin"] = pd.cut(sub["first_running_var"], bins=bins)
        binned = sub.groupby("bin", observed=True).agg(
            x=("first_running_var", "mean"), y=("climbed", "mean"), n=("climbed", "size")
        ).dropna()
        ax.scatter(binned["x"], binned["y"], s=binned["n"] / binned["n"].max() * 200 + 20,
                   color=color, alpha=0.85, label=label, edgecolor="white", linewidth=0.5)

        # Separate OLS fit line on each side of the cutoff, per sex -
        # this is what actually shows whether there is a jump at zero
        # or just a continuous trend running through it.
        for side_mask, x_range in [
            (sub["first_running_var"] < 0, np.linspace(sub["first_running_var"].min(), 0, 50)),
            (sub["first_running_var"] >= 0, np.linspace(0, sub["first_running_var"].max(), 50)),
        ]:
            side = sub[side_mask]
            if len(side) > 10:
                coeffs = np.polyfit(side["first_running_var"], side["climbed"], deg=1)
                ax.plot(x_range, np.polyval(coeffs, x_range), color=color,
                        linewidth=2, alpha=0.6)

    ax.axvline(0, color=ACCENT_GOLD, linestyle="--", linewidth=1.5,
               label="5,000-resident threshold")
    ax.set_xlabel("Population minus 5,000 (running variable)")
    ax.set_ylabel("Share reaching a higher hierarchical tier")
    ax.set_title("Hierarchy-climb rate around the quota threshold, by sex")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_rd_climb.pdf")
    plt.close(fig)
    print("Saved figures/fig1_rd_climb.pdf")


def figure_headline_bars(cox_result, climb_result):
    """
    The one figure a reader remembers: within-sex percent change
    crossing the threshold, for H1 and H2 side by side, men versus
    women. Everything else in the paper explains this picture.
    """
    b_above_cox = cox_result.summary.loc["above_cutoff", "coef"]
    b_interact_cox = cox_result.summary.loc["sex_x_cutoff", "coef"]
    men_h1 = (np.exp(b_above_cox) - 1) * 100
    women_h1 = (np.exp(b_above_cox + b_interact_cox) - 1) * 100

    b_above_logit = climb_result.params["first_above_cutoff"]
    b_interact_logit = climb_result.params["sex_female:first_above_cutoff"]
    men_h2 = (np.exp(b_above_logit) - 1) * 100
    women_h2 = (np.exp(b_above_logit + b_interact_logit) - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    panels = [
        ("H1: exit hazard change", men_h1, women_h1),
        ("H2: climb-odds change", men_h2, women_h2),
    ]
    for ax, (title, men_val, women_val) in zip(axes, panels):
        bars = ax.bar(["Men", "Women"], [men_val, women_val],
                       color=[PURPLE_LIGHT, PURPLE_DARK], edgecolor="white")
        ax.axhline(0, color=NEUTRAL_GRAY, linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("% change crossing threshold")
        for bar, val in zip(bars, [men_val, women_val]):
            ax.annotate(f"{val:+.1f}%", (bar.get_x() + bar.get_width() / 2, val),
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_headline.pdf")
    plt.close(fig)
    print("Saved figures/fig2_headline.pdf")


def figure_bandwidth_robustness(full_panel, bandwidths=(1000, 1500, 2000, 3000)):
    """
    Stability check, visualized - now using the SAME flexible,
    covariate-adjusted model as Table 3's robustness check, refit at
    each bandwidth, rather than raw uncontrolled group means. The
    earlier raw-means version showed the women's line crossing zero
    across bandwidths (-5% to +3.6%) while the controlled model gave a
    stable +6% at bandwidth 1500 - a real inconsistency between this
    figure and the reported result, not just a cosmetic difference.
    Refitting the actual model at each bandwidth removes that gap
    instead of just explaining it away in a caption.
    """
    rows = []
    for bw in bandwidths:
        sample = rdd_bandwidth_sample(full_panel, bandwidth=bw)
        climb = compute_hierarchy_climb(sample)
        result = fit_climb_logit_flexible(climb)
        if result is None:
            continue
        b_above = result.params["first_above_cutoff"]
        b_interact = result.params["sex_female:first_above_cutoff"]
        rows.append({"bandwidth": bw, "sesso": "M",
                     "pct_change": (np.exp(b_above) - 1) * 100})
        rows.append({"bandwidth": bw, "sesso": "F",
                     "pct_change": (np.exp(b_above + b_interact) - 1) * 100})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sex, color, label in [("F", PURPLE_DARK, "Women"), ("M", PURPLE_LIGHT, "Men")]:
        sub = df[df["sesso"] == sex]
        ax.plot(sub["bandwidth"], sub["pct_change"], marker="o", color=color,
                label=label, linewidth=2)
    ax.axhline(0, color=NEUTRAL_GRAY, linewidth=0.8)
    ax.set_xlabel("Bandwidth (+/- population)")
    ax.set_ylabel("% change in climb odds at the cutoff (model-adjusted)")
    ax.set_title("Bandwidth robustness: H2 within-sex effect, flexible specification")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_bandwidth_robustness.pdf")
    plt.close(fig)
    print("Saved figures/fig3_bandwidth_robustness.pdf")


def figure_dynasty(flagged, bandwidth=1500):
    """
    Grouped bar chart for the secondary RQ: dynastic-tie prevalence by
    pre/post-2013 x above/below threshold, male officeholders only.
    The two groups should rise by roughly the same amount if the null
    finding (no differential quota effect) is real, which is exactly
    what the formal model found.
    """
    in_band = flagged["running_var"].abs() <= bandwidth
    male = flagged[(flagged["sesso"] == "M") & in_band]

    summary = (male.groupby(["post_2013", "above_cutoff"])["is_dynastic_windowed"]
               .mean().reset_index())

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    for i, (cutoff_val, status, color) in enumerate([
        (0, "Below threshold", PURPLE_LIGHT), (1, "Above threshold", PURPLE_DARK)
    ]):
        vals = (summary[summary["above_cutoff"] == cutoff_val]
                .sort_values("post_2013")["is_dynastic_windowed"].values)
        ax.bar(x + i * width, vals, width, label=status, color=color, edgecolor="white")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Pre-2013", "Post-2013"])
    ax.set_ylabel("Share with a dynastic tie (windowed measure)")
    ax.set_title("Dynastic recruitment among male officeholders")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_dynasty.pdf")
    plt.close(fig)
    print("Saved figures/fig4_dynasty.pdf")


def figure_dynasty_moderation(decomposition):
    """
    Exploratory extension, not part of the confirmatory H1/H2/H3 battery:
    within-group climb-odds change crossing the threshold, split by both
    sex AND dynastic status (4 bars instead of 2). Shows visually that
    the H2 gap is concentrated in the non-dynastic comparison -- dynastic
    women land close to where dynastic men do, non-dynastic women lag
    well behind non-dynastic men.

    Labeled EXPLORATORY in the title deliberately, not just in the
    caption, since this comes from a single post-hoc specification, not
    the full bandwidth-robustness battery the confirmatory H1/H2/H3
    results went through.
    """
    order = [("Male", "non-dynastic"), ("Male", "dynastic"),
             ("Female", "non-dynastic"), ("Female", "dynastic")]
    vals = []
    labels = []
    colors = []
    for sex, dyn in order:
        row = decomposition[(decomposition["sex"] == sex) & (decomposition["dynastic"] == dyn)]
        vals.append(row["pct_change"].values[0] if len(row) else np.nan)
        labels.append(f"{sex}\n{dyn}")
        colors.append(PURPLE_LIGHT if sex == "Male" else PURPLE_DARK)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.axhline(0, color=NEUTRAL_GRAY, linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:+.1f}%", (bar.get_x() + bar.get_width() / 2, val),
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
    ax.set_ylabel("% change in climb odds crossing threshold")
    ax.set_title("EXPLORATORY: does dynastic status moderate the H2 gap?")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_dynasty_moderation.pdf")
    plt.close(fig)
    print("Saved figures/fig5_dynasty_moderation.pdf")


def _wrap_latex_table(tabular_body, caption, label):
    """
    Manually wraps a to_latex() body in a table environment rather than
    relying on to_latex's own caption/label keyword arguments, since
    those have moved around across pandas versions and I would rather
    not have a table silently render wrong because of a pandas upgrade
    on somebody else's machine.
    """
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"{tabular_body}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def table_descriptive(sample):
    """Table 1: sample composition by sex and threshold status."""
    rows = []
    for sex in ["F", "M"]:
        for cutoff in [0, 1]:
            sub = sample[(sample["sesso"] == sex) & (sample["above_cutoff"] == cutoff)]
            comuni_col = "descrizione_comune" if "descrizione_comune" in sub.columns else None
            rows.append({
                "Sex": "Women" if sex == "F" else "Men",
                "Threshold": "Above" if cutoff else "Below",
                "N": len(sub),
                "Distinct comuni": sub[comuni_col].nunique() if comuni_col else "n/a",
            })
    df = pd.DataFrame(rows)
    body = df.to_latex(index=False)
    tex = _wrap_latex_table(body, "Sample composition by sex and threshold status.",
                             "tab:descriptive_composition")
    (TAB_DIR / "table1_descriptive.tex").write_text(tex)
    print("Saved tables/table1_descriptive.tex")


def table_regression(result, model_type, filename, caption, label):
    """
    Generic regression-table writer, since lifelines (CoxPHFitter) and
    statsmodels (Logit) expose fitted coefficients through different
    attribute names and I would rather have one function handle both
    than duplicate this twice with subtly different bugs.
    """
    if model_type == "cox":
        df = result.summary[["coef", "exp(coef)", "se(coef)", "p"]].reset_index()
        df.columns = ["Covariate", "Coef.", "Hazard ratio", "SE", "p"]
    else:
        df = pd.DataFrame({
            "Covariate": result.params.index,
            "Coef.": result.params.values,
            "Odds ratio": np.exp(result.params.values),
            "SE": result.bse.values,
            "p": result.pvalues.values,
        })
    df = df.round(4)
    body = df.to_latex(index=False)
    tex = _wrap_latex_table(body, caption, label)
    (TAB_DIR / filename).write_text(tex)
    print(f"Saved tables/{filename}")


def table_dynasty_moderation_summary(decomposition, filename, caption, label):
    """
    Small 4-row summary table (sex x dynastic status, N, percent
    change) - separate from table_regression since this is a plain
    decomposition DataFrame, not a statsmodels/lifelines result object.
    """
    df = decomposition.copy()
    df = df.rename(columns={"sex": "Sex", "dynastic": "Dynastic status",
                             "pct_change": "Climb-odds change (%)", "n": "N"})
    df["Climb-odds change (%)"] = df["Climb-odds change (%)"].round(1)
    df = df[["Sex", "Dynastic status", "N", "Climb-odds change (%)"]]
    body = df.to_latex(index=False)
    tex = _wrap_latex_table(body, caption, label)
    (TAB_DIR / filename).write_text(tex)
    print(f"Saved tables/{filename}")


if __name__ == "__main__":
    path = "data/raw/storico_amministratori_comuni_"
    full_dates_list = [f"3112{d}" for d in range(1986, 2026)]
    full_dates_list.append("24082026")
    paths = [f"{path}{d}.csv" for d in full_dates_list]

    full_panel = get_panel(paths, cache_path="data/preprocessed/full_panel.parquet")
    sample = rdd_bandwidth_sample(full_panel)  # default +/-1500

    tenure = compute_tenure_panel(sample)
    climb = compute_hierarchy_climb(sample)

    cox_result = fit_tenure_cox_model(tenure)
    climb_result = fit_climb_logit_model(climb)
    climb_flexible_result = fit_climb_logit_flexible(climb)

    figure_rd_climb(full_panel)
    figure_headline_bars(cox_result, climb_result)
    figure_bandwidth_robustness(full_panel)

    flagged = flag_dynastic_ties(full_panel)
    figure_dynasty(flagged)

    table_descriptive(sample)
    table_regression(cox_result, "cox", "table2_cox.tex",
                      "Cox proportional-hazards model: tenure.", "tab:cox")
    table_regression(climb_result, "logit", "table3_climb.tex",
                      "Logistic regression: hierarchy climb.", "tab:climb")
    table_regression(climb_flexible_result, "logit", "table3b_climb_flexible.tex",
                      "Robustness: hierarchy climb, separate slopes each side "
                      "of the cutoff.", "tab:climb_flexible")

    dynasty_result = fit_dynasty_logit(flagged, bandwidth=1500, exclude_top_surname_pct=0.0)
    table_regression(dynasty_result, "logit", "table4_dynasty.tex",
                      "Logistic regression: dynastic recruitment (male officeholders).",
                      "tab:dynasty")

    # Exploratory extension (not part of the confirmatory H1/H2/H3
    # battery): does dynastic status moderate the H2 gap. Reuses the
    # already-computed climb table and flagged dynasty data above,
    # no new data construction.
    climb_with_dynasty = merge_dynasty_onto_climb(climb, flagged)
    moderation_result, moderation_decomposition = fit_dynasty_moderation_model(climb_with_dynasty)
    figure_dynasty_moderation(moderation_decomposition)
    table_regression(moderation_result, "logit", "table7_dynasty_moderation.tex",
                      "EXPLORATORY: does dynastic status moderate the H2 gap "
                      "(full regression).", "tab:dynasty_moderation")
    table_dynasty_moderation_summary(
        moderation_decomposition, "table7b_dynasty_moderation_summary.tex",
        "EXPLORATORY: within-group climb-odds change by sex and dynastic status.",
        "tab:dynasty_moderation_summary")

    print("\nAll figures and tables generated.")