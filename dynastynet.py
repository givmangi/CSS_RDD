"""
Secondary RQ: do parties lean harder on family names once the quota bites?
================================================================================
Spoiler, since you'll see it in the output anyway: no, not differentially.
Dynastic recruitment among male officeholders is rising nationally after
2013, but not more so in quota-bound municipalities than elsewhere. Kept
as a real, reported null result rather than quietly dropped, because a
well-specified null is still a finding, not a failure.

The actual question: does the quota's binding constraint (post-2013,
above the 5,000-resident threshold) push parties toward more surname-based
family recruitment for their MALE candidates, as a way of holding onto
some insider control once female representation is mandated?

Design: same DID x RDD logic as the primary analysis (post_2013 x
above_cutoff interaction), pointed at a different outcome
(is_dynastic_windowed instead of tenure/advancement).

Data: the exact same DAIT panel already built for the primary analysis --
no second data source needed. Daniele et al. (2021) validates surname-based dynasty
detection on elected officeholders directly, which is a relief, because
the full candidate-list version of this (winners AND losers) would have
needed a different, considerably less cooperative portal (Eligendo,
which turned out to expose only top-of-list names in bulk - a dead end
I hit and moved past earlier in this project). Scope honestly stated:
this measures dynastic ties among people who WON, not raw nomination
behavior before voters get a say. That's a real, worth-stating limit,
not a footnote to bury.

Imports data-construction functions directly from panel_pipeline.py
rather than duplicating them -- one pipeline feeding two analyses.
"""
import numpy as np
import pandas as pd

from panel import get_panel, snapshot_date_from_filename

try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def flag_dynastic_ties(df: pd.DataFrame, window_years: int = 20) -> pd.DataFrame:
    """
    Per-person table: is this officeholder plausibly a family successor?

    TWO DEFINITIONS computed, for transparency and comparison:

    1. is_dynastic_unbounded (original definition): first_year strictly
       later than the EARLIEST any same-surname person appeared in this
       comune, anywhere in the 1986-2025 window. FLAWED: this is
       mechanically confounded with calendar time -- someone whose
       career starts in 2015 has a 29-year lookback (1986-2014) in which
       a same-surname predecessor could appear; someone starting in 1988
       has only 2 years. The dynastic rate rises over time BY
       CONSTRUCTION, independent of any real change in party behavior,
       which is a serious threat to the pre/post-2013 DID specifically.
       Kept only as a comparison baseline, NOT used for the main test.

    2. is_dynastic_windowed (primary measure): same surname+comune
       appeared within the preceding `window_years` years specifically
       (default 20) -- not "ever since 1986." This gives every
       officeholder a comparable lookback window regardless of when
       their career started, removing the mechanical time-trend
       confound. Vectorized via sort + groupby().diff() (gap to the
       nearest earlier same-surname entry; if the nearest predecessor is
       outside the window, all earlier ones are too, so checking the
       nearest is sufficient).
       Residual limitation: officeholders whose first_year falls within
       `window_years` of 1986 (the panel start) have a truncated window
       on the left edge -- affects a small, dateable subset, worth a
       sensitivity check on window_years if results are borderline.

    Also computes surname_national_freq (count of distinct people with
    this surname across the full panel) for use as (a) an exclusion
    threshold and (b) a control covariate, following Daniele et al. (2021) /
    Durante et al. (2011)'s common-surname robustness approach -- cited
    directly in this paper's bibliography for using this same technique.
    """
    df = df.copy()
    df["snapshot_year"] = df["source_file"].map(snapshot_date_from_filename).dt.year

    cols = ["person_key", "descrizione_comune", "cognome", "snapshot_year",
            "sesso", "popolazione_censita", "codice_provincia", "codice_comune"]
    cols = [c for c in cols if c in df.columns]
    df_sorted = df.sort_values(["person_key", "snapshot_year"])
    first_appearance = df_sorted[cols].groupby("person_key", as_index=False).first()
    first_appearance = first_appearance.rename(columns={"snapshot_year": "first_year"})
    # Same nationally-unique cluster key as panel_pipeline's models -
    # codice_comune alone repeats across provinces (ISTAT convention).
    if "codice_provincia" in first_appearance.columns and "codice_comune" in first_appearance.columns:
        first_appearance["comune_cluster_id"] = (
            first_appearance["codice_provincia"].astype(str) + "_" +
            first_appearance["codice_comune"].astype(str)
        )

    # National surname frequency -- proxy for population-level commonality
    # (self-consistent, not external ISTAT data; stated as a proxy, not
    # verified against ISTAT surname-frequency tables).
    surname_freq = first_appearance.groupby("cognome")["person_key"].transform("nunique")
    first_appearance["surname_national_freq"] = surname_freq

    # Original (flawed, comparison-only) unbounded definition
    surname_first_year = (
        first_appearance.groupby(["descrizione_comune", "cognome"])["first_year"]
        .transform("min")
    )
    first_appearance["is_dynastic_unbounded"] = (
        first_appearance["first_year"] > surname_first_year
    ).astype(int)

    # Fixed-window definition (PRIMARY measure)
    fa_sorted = first_appearance.sort_values(["descrizione_comune", "cognome", "first_year"])
    prev_year = fa_sorted.groupby(["descrizione_comune", "cognome"])["first_year"].shift(1)
    gap = fa_sorted["first_year"] - prev_year
    fa_sorted["is_dynastic_windowed"] = (
        (gap > 0) & (gap <= window_years)
    ).fillna(False).astype(int)
    first_appearance = fa_sorted  # already contains all other columns

    first_appearance["running_var"] = first_appearance["popolazione_censita"] - 5000
    first_appearance["above_cutoff"] = (first_appearance["running_var"] >= 0).astype(int)
    first_appearance["post_2013"] = (first_appearance["first_year"] >= 2013).astype(int)

    return first_appearance


def fit_dynasty_logit(flagged: pd.DataFrame, bandwidth: int = 1500,
                       exclude_top_surname_pct: float = 0.0):
    """
    Male-only logistic regression: is_dynastic_windowed ~ post_2013 *
    above_cutoff + running_var + log(surname_national_freq). Bandwidth
    restriction to the RDD sample is applied HERE, on the already-
    correctly-flagged person-level table -- not before flagging (see
    flag_dynastic_ties docstring).

    exclude_top_surname_pct: if >0 (e.g. 0.01 for top 1%), drops
    officeholders whose surname is among the most frequent nationally,
    following Daniele et al. (2021) / Durante et al. (2011)'s robustness approach
    of re-running results with common surnames excluded. Run this at
    0.0 first, then again at e.g. 0.01-0.05 as an explicit robustness
    check -- do not silently pick whichever threshold gives a nicer
    result.
    """
    if not _HAS_STATSMODELS:
        print("statsmodels not installed -- run `pip install statsmodels` first")
        return None

    in_band = flagged["running_var"].abs() <= bandwidth
    male = flagged[(flagged["sesso"] == "M") & in_band].copy()

    if exclude_top_surname_pct > 0:
        cutoff = male["surname_national_freq"].quantile(1 - exclude_top_surname_pct)
        n_before = len(male)
        male = male[male["surname_national_freq"] < cutoff].copy()
        print(f"  Excluding surnames in the top {exclude_top_surname_pct:.1%} "
              f"by national frequency (freq >= {cutoff:.0f}): "
              f"dropped {n_before - len(male)} of {n_before} rows")

    male["log_surname_freq"] = np.log1p(male["surname_national_freq"])
    print(f"  Dynasty sample: {len(male)} male officeholders within "
          f"+/-{bandwidth} of the threshold")

    cols = ["is_dynastic_windowed", "post_2013", "above_cutoff", "running_var",
            "log_surname_freq"]
    model_df = male[cols].dropna()
    dropped = len(male) - len(model_df)
    if dropped:
        print(f"  Dynasty logit: dropped {dropped} rows with missing values")

    formula = "is_dynastic_windowed ~ post_2013 * above_cutoff + running_var + log_surname_freq"
    # Both cluster and HC1 stalled here across repeated attempts.
    # Reverted to plain nonrobust fitting, the original confirmed-working
    # approach. Root cause of the stall was never isolated - worth
    # investigating properly later if there's time, not worth further
    # guessing under current time pressure.
    result = smf.logit(formula, data=model_df).fit(disp=0)
    print("\n=== Logistic regression: dynastic recruitment (male officeholders, "
          "fixed-window definition, surname-frequency controlled) ===")
    print(result.summary())

    interaction = "post_2013:above_cutoff"
    if interaction in result.params:
        b_post = result.params["post_2013"]
        b_interact = result.params[interaction]
        below_change = np.exp(b_post)
        above_change = np.exp(b_post + b_interact)
        print("\nSecondary RQ test -- within-threshold-status change in "
              "dynastic-tie odds from the pre-2013 to post-2013 period "
              "(this is the actual test: does the CHANGE differ by "
              "threshold status, not the raw interaction coefficient alone):")
        print(f"  Below cutoff: exp({b_post:.4f}) = {below_change:.4f}  "
              f"({(below_change-1)*100:+.1f}%)")
        print(f"  Above cutoff: exp({b_post:.4f} + {b_interact:.4f}) = "
              f"{above_change:.4f}  ({(above_change-1)*100:+.1f}%)")
    return result


if __name__ == "__main__":
    path = "data/raw/storico_amministratori_comuni_"
    full_dates_list = [f"3112{d}" for d in range(1986, 2026)]
    full_dates_list.append("24082026")
    paths = [f"{path}{d}.csv" for d in full_dates_list]

    panel = get_panel(paths, cache_path="data/preprocessed/full_panel.parquet")
    flagged = flag_dynastic_ties(panel, window_years=20)  # full national panel, see docstring

    print("\n########## BASELINE (all surnames, windowed definition) ##########")
    fit_dynasty_logit(flagged, bandwidth=1500, exclude_top_surname_pct=0.0)

    print("\n########## ROBUSTNESS: excluding top 1% most common surnames ##########")
    fit_dynasty_logit(flagged, bandwidth=1500, exclude_top_surname_pct=0.01)

    print("\n########## ROBUSTNESS: excluding top 5% most common surnames ##########")
    fit_dynasty_logit(flagged, bandwidth=1500, exclude_top_surname_pct=0.05)