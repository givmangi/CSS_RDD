"""
Extension: does dynastic embeddedness moderate the H2 gap?
==============================================================
Draft, not yet run against real data. Review carefully before trusting
output - this hasn't been through the same iterative debugging the rest
of the pipeline has.

Question: among quota-era women, does being part of an existing
political-family network (same is_dynastic_windowed flag already used
for the secondary RQ, computed here for BOTH sexes rather than
restricted to men) buffer the advancement gap found in H2? If the gap
shrinks for dynastic women specifically, that's evidence pointing at
institutional access as a mechanism, not sex per se - a real, partial
answer to the "causal scope" limitation already stated in the paper.

Reuses panel_pipeline's compute_hierarchy_climb and dynastynet's
flag_dynastic_ties - no new data construction, just a new merge and a
new model formula.
"""
import numpy as np
import pandas as pd

from panel import get_panel, rdd_bandwidth_sample, compute_hierarchy_climb
from dynastynet import flag_dynastic_ties

try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def merge_dynasty_onto_climb(climb_df: pd.DataFrame, flagged_all_sexes: pd.DataFrame) -> pd.DataFrame:
    """
    flagged_all_sexes should be the RAW output of flag_dynastic_ties()
    on the full panel, BEFORE fit_dynasty_logit()'s male-only filter -
    the flag itself is computed for everyone, only the SAMPLE used for
    the secondary RQ's own regression was restricted to men.
    """
    dynasty_cols = flagged_all_sexes[["person_key", "is_dynastic_windowed"]]
    merged = climb_df.merge(dynasty_cols, on="person_key", how="left")
    missing = merged["is_dynastic_windowed"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} of {len(merged)} climb-table rows have no "
              f"matching dynasty flag after merge - check person_key consistency "
              f"between the two source tables before trusting this further.")
    return merged


def fit_dynasty_moderation_model(climb_with_dynasty: pd.DataFrame, bandwidth: int = 1500):
    """
    climbed ~ sex_female * first_above_cutoff * is_dynastic_windowed
             + first_running_var

    The coefficient of actual interest is the triple interaction
    (sex_female:first_above_cutoff:is_dynastic_windowed) - does the
    core H2 gap (sex_female:first_above_cutoff) differ for dynastic
    women specifically. Decomposed below into four groups (male/female
    x dynastic/non-dynastic) at the cutoff, same style as every other
    within-group decomposition already used in this project - do not
    just eyeball the raw triple-interaction coefficient alone.
    """
    if not _HAS_STATSMODELS:
        print("statsmodels not installed")
        return None

    df = climb_with_dynasty.copy()
    df["sex_female"] = (df["sesso"] == "F").astype(int)
    in_band = df["first_running_var"].abs() <= bandwidth
    df = df[in_band].copy()
    df["is_dynastic_windowed"] = df["is_dynastic_windowed"].fillna(0)

    formula = ("climbed ~ sex_female * first_above_cutoff * is_dynastic_windowed "
               "+ first_running_var")
    cols = ["climbed", "sex_female", "first_above_cutoff",
            "is_dynastic_windowed", "first_running_var"]
    model_df = df[cols].dropna()
    dropped = len(df) - len(model_df)
    if dropped:
        print(f"  Dynasty-moderation model: dropped {dropped} rows with missing values")

    result = smf.logit(formula, data=model_df).fit(disp=0)
    print("\n=== Draft: does dynastic status moderate the H2 gap? ===")
    print(result.summary())

    # Subgroup N's -- needed before trusting any subgroup estimate,
    # since splitting an already-smaller female subsample by a further
    # binary flag risks a small-N cell nobody would want to base a
    # confident claim on without checking first.
    print("\nSubgroup sample sizes (in-band, post dropna):")
    counts = model_df.groupby(["sex_female", "is_dynastic_windowed"]).size()
    for (sex, dyn), n in counts.items():
        label = f"{'Female' if sex else 'Male'}, {'dynastic' if dyn else 'non-dynastic'}"
        print(f"  {label}: N={n}")

    # Within-group climb-odds change crossing the threshold, same
    # decomposition style used for every other model in this project --
    # this, not the raw log-odds offsets, is what actually belongs in
    # the paper.
    b_above = result.params["first_above_cutoff"]
    b_above_dyn = result.params.get("first_above_cutoff:is_dynastic_windowed", 0)
    b_sex_above = result.params["sex_female:first_above_cutoff"]
    b_triple = result.params.get("sex_female:first_above_cutoff:is_dynastic_windowed", 0)

    change = {
        ("Male", "non-dynastic"): b_above,
        ("Male", "dynastic"): b_above + b_above_dyn,
        ("Female", "non-dynastic"): b_above + b_sex_above,
        ("Female", "dynastic"): b_above + b_above_dyn + b_sex_above + b_triple,
    }
    print("\nWithin-group climb-odds change crossing the threshold "
          "(this is the number that belongs in the paper, not the raw "
          "log-odds offsets above):")
    for (sex, dyn), logodds in change.items():
        pct = (np.exp(logodds) - 1) * 100
        print(f"  {sex}, {dyn}: {pct:+.1f}%")

    # Four-group decomposition at the cutoff (running_var=0), same
    # pattern as every other within-group breakdown in this project.
    params = result.params
    groups = {
        ("Male", "non-dynastic"): 0,
        ("Male", "dynastic"): params.get("is_dynastic_windowed", 0),
        ("Female", "non-dynastic"): params.get("sex_female", 0),
        ("Female", "dynastic"): (
            params.get("sex_female", 0) + params.get("is_dynastic_windowed", 0) +
            params.get("sex_female:is_dynastic_windowed", 0)
        ),
    }
    print("\nBaseline (below-cutoff) log-odds offset by group (reference = "
          "male, non-dynastic, below cutoff):")
    for (sex, dynasty), val in groups.items():
        print(f"  {sex}, {dynasty}: {val:.4f}")

    triple = "sex_female:first_above_cutoff:is_dynastic_windowed"
    if triple in params:
        print(f"\nTriple interaction ({triple}): coef={params[triple]:.4f}, "
              f"p={result.pvalues[triple]:.4g}")
        print("If this is negative and significant: dynastic women gain even "
              "LESS from crossing the threshold than non-dynastic women (the "
              "opposite of the buffering hypothesis). If positive and "
              "significant: dynastic ties partly buffer the H2 gap. If not "
              "significant: no detectable moderation either way - a real, "
              "reportable null, not evidence of nothing happening.")

    # Structured version of the within-group decomposition above, for
    # graphics_tables.py to plot/tabulate directly rather than
    # re-deriving the same numbers from result.params a second time.
    decomposition = pd.DataFrame([
        {"sex": sex, "dynastic": dyn, "pct_change": (np.exp(logodds) - 1) * 100,
         "n": int(counts.get((1 if sex == "Female" else 0,
                              1 if dyn == "dynastic" else 0), 0))}
        for (sex, dyn), logodds in change.items()
    ])
    return result, decomposition


if __name__ == "__main__":
    path = "data/raw/storico_amministratori_comuni_"
    full_dates_list = [f"3112{d}" for d in range(1986, 2026)]
    full_dates_list.append("24082026")
    paths = [f"{path}{d}.csv" for d in full_dates_list]

    full_panel = get_panel(paths, cache_path="data/preprocessed/full_panel.parquet")
    sample = rdd_bandwidth_sample(full_panel)
    climb = compute_hierarchy_climb(sample)

    # NOTE: flag_dynastic_ties() runs on the FULL panel (all sexes, all
    # population levels) by design - see its own docstring for why.
    # Do NOT restrict to males before this step.
    flagged_all = flag_dynastic_ties(full_panel)

    climb_with_dynasty = merge_dynasty_onto_climb(climb, flagged_all)
    result, decomposition = fit_dynasty_moderation_model(climb_with_dynasty)
    print("\nDecomposition table (for reference/copy into graphics_tables.py output):")
    print(decomposition)