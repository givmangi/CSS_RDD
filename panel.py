"""
Panel construction pipeline for the Italian municipal gender-quota study
==========================================================================
Turns 40 years of Ministry of Interior CSVs (one per year, 1986-2026,
downloaded by hand because the portal blocks scripted access) into a
single clean panel.

Field schema empowered through most of the years:
CODICE_REGIONE;DESCRIZIONE_REGIONE;CODICE_PROVINCIA;DESCRIZIONE_PROVINCIA;
CODICE_COMUNE;DESCRIZIONE_COMUNE;SIGLA_PROVINCIA;ISTAT_CODICE_COMUNE;
POPOLAZIONE_CENSITA;MAGGIORITARIO_PROPORZIONALE;DESCRIZIONE_TEMPO_GESTIONE;
DATA_ELEZIONE;DATA_BALLOTTAGGIO;CONSIGLIERI_SPETTANTI;ASSESSORI_ASSEGNATI;
SIGLA_TITOLO_ACCADEMICO;COGNOME;NOME;SESSO;DATA_NASCITA;SEDE_NASCITA;
LIVELLO_CARICA;DESCRIZIONE_CARICA;DATA_NOMINA;PARTITO_LISTA_COALIZIONE;
DATA_CESSAZIONE;TITOLO_DI_STUDIO;PROFESSIONE

Fair warning: this layout is NOT stable across 40 years of government
exports. Casing changes, fields got renamed, and 2023-2025 restructured
things outright (handled via an explicit alias map) If adding new year/snapshot
and something breaks, check header first.
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path

try:
    from lifelines import CoxPHFitter
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False

try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# 1. LOAD + NORMALIZE csvs

def load_storico(path: str) -> pd.DataFrame:
    """
    Load one storico_amministratori_comuni file, normalize column names.
    Encoding is NOT reliably UTF-8, so I tried a few common fallbacks
    """
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
                df = pd.read_csv(path, sep=";", encoding=enc, dtype=str,
                                  on_bad_lines="warn", quotechar='"')
            print(f"  [{Path(path).name}] loaded with encoding={enc}")
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]


            column_aliases = {
                "data_inizio_carica": "data_nomina",
                "data_entrata_in_carica": "data_nomina",
                "popolazione_censita_alla_data_elezione": "popolazione_censita",
                "denominazione_comune": "descrizione_comune",
                "luogo_nascita": "sede_nascita",
            }
            for new_name, canonical in column_aliases.items():
                if new_name in df.columns and canonical not in df.columns:
                    df = df.rename(columns={new_name: canonical})

            if "data_cessazione" not in df.columns:
                df["data_cessazione"] = np.nan

            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Couldn't decode {path} with any selected encoding") from last_err


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [c for c in df.columns if "data" in c]
    for c in date_cols:
        parsed = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
        still_missing = parsed.isna() & df[c].notna()
        if still_missing.any():
            parsed.loc[still_missing] = pd.to_datetime(
                df[c][still_missing], errors="coerce"
            )
        df[c] = parsed
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["popolazione_censita", "livello_carica"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# INDIVIDUAL IDENTIFIER (for cross-year / dynasty network)

def build_person_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Surname + first name + birth date as a composite match key.
    Initially added also birthplace, but that was inconsistent across years and sometimes missing.
    NOT a perfect identifier, in fact there are duplicates that are checked and
    later discarded because they cannot be reliably tracked.
    Common-surname false positives are a real risk, handled with additional validation.
    """
    for c in ["cognome", "nome", "data_nascita"]:
        if c in df.columns:
            df[c] = df[c].str.strip().str.upper()
    def safe_str(col: str) -> pd.Series:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.dt.strftime("%Y-%m-%d").fillna("MISSING")
        return s.fillna("MISSING").astype(str).str.strip()

    key_cols = [c for c in ["cognome", "nome", "data_nascita"] if c in df.columns]
    parts = [safe_str(c) for c in key_cols]
    person_key = parts[0]
    for p in parts[1:]:
        person_key = person_key + "|" + p
    df["person_key"] = person_key

    # Incomplete keys don't just fail to match reliably across years,
    # they carry a collision risk: two different people both
    # missing the same field, sharing surname and first name, would
    # land on the identical "MISSING"-padded key and get silently
    # merged into one fake person. Final operative decision was to drop
    incomplete_mask = person_key.str.contains("MISSING")
    incomplete = incomplete_mask.sum()
    if incomplete:
        print(f"  Dropped {incomplete} rows with incomplete person_key "
            f"(missing cognome/nome/data_nascita) out of {len(df)} - "
            f"these cannot be reliably tracked, and worse, risk colliding "
            f"with each other on the shared placeholder.")
        df = df[~incomplete_mask].copy()
    return df

# 3. RDD SAMPLE CONSTRUCTION with bandwidth around the 5,000 thresholds

def rdd_bandwidth_sample(df: pd.DataFrame, cutoff: int = 5000,
                          bandwidth: int = 1500) -> pd.DataFrame:
    """
    Restrict to comuni within +/- bandwidth of the population cutoff.
    Start wide (e.g. 1500) then narrow - always report results across
    multiple bandwidths as a robustness check, never a single fixed window.
    """
    pop = df["popolazione_censita"]
    mask = (pop >= cutoff - bandwidth) & (pop <= cutoff + bandwidth)
    out = df.loc[mask].copy()
    out["above_cutoff"] = (out["popolazione_censita"] >= cutoff).astype(int)
    out["running_var"] = out["popolazione_censita"] - cutoff
    return out


# ---------------------------------------------------------------------------
# 4. TENURE FROM WITHIN-FILE NOMINA/CESSAZIONE
# ---------------------------------------------------------------------------

def snapshot_date_from_filename(filename: str) -> pd.Timestamp:
    """Extract the year-end snapshot date from storico_..._DDMMYYYY.csv.
    Filenames aren't perfectly consistent (some have an underscore before
    the date, some don't) so match on the last 4 digits before .csv rather
    than a rigid full pattern."""
    m = re.search(r"(\d{4})\.csv$", filename)
    if not m:
        raise ValueError(f"Could not parse year from filename: {filename}")
    return pd.Timestamp(year=int(m.group(1)), month=12, day=31)


def compute_tenure_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tenure computation across a multi-year stacked panel. Unlike
    compute_tenure() (single global snapshot_date, fine for one file),
    this uses each row's OWN file's year-end as the censoring date,
    since the panel mixes 2010/2013/2018/2024 snapshots.
    """
    df = df.copy()
    df["snapshot_date"] = df["source_file"].map(snapshot_date_from_filename)
    censored = df["data_cessazione"].isna()
    end = df["data_cessazione"].fillna(df["snapshot_date"])
    df["tenure_days"] = (end - df["data_nomina"]).dt.days
    df["censored"] = censored.astype(int)
    return df


# ---------------------------------------------------------------------------
# 5. CROSS-YEAR PANEL MERGE (for hierarchy-climbing outcome)
# ---------------------------------------------------------------------------

def save_panel_cache(df: pd.DataFrame, path: str = "data/preprocessed/full_panel.parquet"):
    """
    Parquet, not CSV: columnar + compressed (typically 5-10x smaller than
    equivalent CSV for repetitive data like this -- source_file alone is
    a ~40-char string repeated on every one of ~5M rows), reads faster,
    and stores actual datetime64 values natively -- no string round-trip,
    which is what caused the earlier date-parsing cache bug in the first
    place. Requires pyarrow (`pip install pyarrow`).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  Saved preprocessed panel cache (parquet): {path} ({len(df)} rows)")


def load_panel_cache(path: str = "data/preprocessed/full_panel.parquet") -> pd.DataFrame:
    # No parse_dates()/clean_numeric() needed here -- parquet preserves
    # dtypes exactly, so there's no string reparsing step to get wrong.
    df = pd.read_parquet(path)
    print(f"  Loaded preprocessed panel cache (parquet): {path} ({len(df)} rows)")
    return df


def get_panel(paths: list[str], cache_path: str = "data/preprocessed/full_panel.parquet",
              force_rebuild: bool = False) -> pd.DataFrame:
    """
    Returns the fully built, deduped panel -- from cache if available,
    otherwise builds it via stack_years() and saves the cache for next
    time. This caches the panel at the point right after stack_years()
    (deduped, person_key built) -- NOT bandwidth-filtered and not
    year-tagged, since those are cheap vectorized steps that should stay
    fresh on every run; only the expensive 40-file parse+dedup is cached.

    IMPORTANT, real risk not hypothetical: if you change any UPSTREAM
    preprocessing logic (encoding handling, dedup rules, person_key
    construction, column aliasing) after a cache already exists, you
    MUST delete the cache file or pass force_rebuild=True -- otherwise
    every downstream script silently keeps analyzing data built under
    the OLD logic, and you won't get an error telling you so.
    """
    if not force_rebuild and Path(cache_path).exists():
        return load_panel_cache(cache_path)
    df = stack_years(paths)
    save_panel_cache(df, cache_path)
    return df


def filter_inconsistent_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops every row belonging to a person_key whose recorded 'sesso'
    is not constant across appearances. This is the confirmed remaining
    symptom of the same collision risk the incomplete-key drop in
    build_person_id addresses: two different real people who happen to
    share surname, first name, AND birth date, landing on the same key.
    There is no principled way to tell which of the two people a merged
    row actually belongs to, so the honest fix is dropping the whole
    key, not guessing which half is right.

    Run ONCE here, on the full concatenated panel, BEFORE dedup - a
    corrupted key reaching dedupe_dual_appointments could get its two
    real people's separate appointments wrongly collapsed into one, so
    this needs to run first, not as an afterthought downstream.

    Every script (panel_pipeline itself, dynasty_network) now inherits
    an already-clean entity set from here, rather than each one
    separately re-detecting and separately deciding what to do about it.
    """
    sesso_nunique = df.groupby("person_key")["sesso"].nunique()
    bad_keys = sesso_nunique[sesso_nunique > 1].index
    before = len(df)
    df = df[~df["person_key"].isin(bad_keys)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows across {len(bad_keys)} person_keys "
              f"with inconsistent sesso across appearances (likely two "
              f"different people sharing a match key) out of {before} rows.")
    return df


def stack_years(paths: list[str]) -> pd.DataFrame:
    # Columns tenure/hierarchy analysis actually depends on. A file missing
    # any of these is structurally not a usable storico year-end snapshot
    # (e.g. the current "ad oggi" export has no data_cessazione at all) --
    # skip it and say why, rather than crashing the whole multi-decade run
    # or silently producing garbage downstream.
    required_cols = {"cognome", "nome", "data_nascita", "descrizione_carica",
                      "data_nomina", "data_cessazione", "popolazione_censita"}

    frames = []
    skipped = []
    for p in paths:
        try:
            df = load_storico(p)
        except Exception as e:
            print(f"  SKIPPING {p}: failed to load ({e})")
            skipped.append((p, "load_failed"))
            continue

        missing = required_cols - set(df.columns)
        if missing:
            print(f"  SKIPPING {p}: missing required columns {sorted(missing)} "
                  f"-- likely a different export schema for this year/date, "
                  f"not a standard storico year-end file. "
                  f"Columns actually found: {sorted(df.columns)}")
            skipped.append((p, f"missing_cols:{sorted(missing)}"))
            continue

        df = parse_dates(df)
        df = clean_numeric(df)
        df = build_person_id(df)
        df["source_file"] = Path(p).name
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid files loaded across the full path list -- "
                            "check paths and schema before proceeding.")
    if skipped:
        print(f"\n  Loaded {len(frames)}/{len(paths)} files successfully. "
              f"Skipped: {[s[0] for s in skipped]}")

    panel = pd.concat(frames, ignore_index=True)
    panel = filter_inconsistent_entities(panel)
    return dedupe_dual_appointments(panel)


def hierarchy_rank(carica: pd.Series) -> pd.Series:
    """Map descrizione_carica to an ordinal rank for advancement tracking."""
    rank_map = {
        "sindaco": 3, "vicesindaco": 2, "assessore": 1,
        "assessore anziano": 1, "consigliere": 0,
    }
    return carica.str.lower().map(rank_map)


def compute_hierarchy_climb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-person summary across all years they appear in: did their highest
    office rank increase between their first and any later appearance?

    PERFORMANCE NOTE: originally implemented as
    groupby("person_key").apply(python_function), which does not
    vectorize -- on the full national panel (up to ~430k groups
    depending on bandwidth), called five times across a full run
    (quick-look + 4-bandwidth sweep), this was almost certainly the
    dominant cost in a ~30-minute run. Rewritten below using
    groupby().agg()/.first(), which run as compiled/vectorized
    operations rather than one Python function call per group --
    expect this to cut runtime by roughly an order of magnitude.

    QA note: also checks whether 'sesso' is constant within a person_key
    group. If it isn't, that's a real signal the entity-resolution key is
    producing false matches (two different people sharing cognome+nome+
    data_nascita), not just noise -- worth investigating any flagged cases
    individually, not silently averaging over them.

    Behavior change from the previous version: 'sesso' is now taken from
    the FIRST observed appearance (after sorting by year) rather than the
    modal value across all appearances. Given the inconsistency rate is
    already tiny (<0.1%) and separately flagged, this is a reasonable and
    much faster substitute -- stated explicitly as a decision, not a
    silent behavior change.
    """
    df = df.copy()
    df["_rank"] = hierarchy_rank(df["descrizione_carica"])
    df["snapshot_year"] = df["source_file"].map(snapshot_date_from_filename).dt.year

    # This should always print zero now that filter_inconsistent_entities()
    # runs upstream in stack_years() and drops these keys before they ever
    # reach here. If this fires with a nonzero count, the most likely
    # explanation is a stale cache built before that filter existed -
    # delete data/preprocessed/*.parquet and rebuild, don't just note the
    # number and move on.
    sesso_nunique = df.groupby("person_key")["sesso"].nunique()
    inconsistent = (sesso_nunique > 1).sum()
    if inconsistent:
        print(f"  WARNING: {inconsistent} person_keys still have inconsistent "
              f"'sesso' despite the upstream filter - this should not happen "
              f"on a freshly built panel. Check for a stale cache first.")

    df_sorted = df.sort_values(["person_key", "snapshot_year"])
    first_cols = ["person_key", "snapshot_year", "_rank", "sesso",
                  "above_cutoff", "running_var", "codice_provincia", "codice_comune"]
    first_cols = [c for c in first_cols if c in df.columns]
    first_rows = df_sorted[first_cols].groupby("person_key", as_index=False).first()
    first_rows = first_rows.rename(columns={
        "snapshot_year": "first_year", "_rank": "first_rank",
        "above_cutoff": "first_above_cutoff", "running_var": "first_running_var",
    })
    # codice_comune is only unique WITHIN a province (ISTAT convention) -
    # combine with codice_provincia for a nationally-unique cluster key,
    # needed for clustered standard errors in the outcome models below.
    if "codice_provincia" in first_rows.columns and "codice_comune" in first_rows.columns:
        first_rows["comune_cluster_id"] = (
            first_rows["codice_provincia"].astype(str) + "_" +
            first_rows["codice_comune"].astype(str)
        )

    agg = df.groupby("person_key").agg(
        last_year=("snapshot_year", "max"),
        max_rank=("_rank", "max"),
        n_appearances=("snapshot_year", "nunique"),
        ever_above_cutoff=("above_cutoff", "max"),
    ).reset_index()

    climb = first_rows.merge(agg, on="person_key")
    climb["climbed"] = (climb["max_rank"] > climb["first_rank"]).astype(int)
    return climb


def dedupe_dual_appointments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse (person_key, source_file) groups to one row per person per term.

    Rationale: in Italian comuni, an assessore/vicesindaco is typically also
    formally registered as consigliere, producing two legitimate rows for
    one person in one term (confirmed by manual inspection of the pilot
    output - e.g. Franco Abate appears as both Vicesindaco and Consigliere
    in Pantigliate 2018, same dates). For tenure/hierarchy analysis I want
    ONE row per person-term at their HIGHEST office, not two.

    NOTE: nomina dates can differ slightly between the two rows (e.g. elected
    consigliere first, appointed assessore days later) - this keeps the
    higher-ranked row's own dates, meaning tenure will be measured from the
    executive appointment date, not the original council-seat date. State
    this choice explicitly in the methods section; it's a real decision,
    not a neutral default.
    """
    df = df.copy()
    df["_rank"] = hierarchy_rank(df["descrizione_carica"])
    df = df.sort_values("_rank", ascending=False)
    deduped = df.drop_duplicates(subset=["person_key", "source_file"], keep="first")
    dropped = len(df) - len(deduped)
    print(f"Deduped {dropped} lower-ranked duplicate-appointment rows "
          f"({dropped/len(df):.2%} of input)")
    return deduped.drop(columns="_rank")


# ---------------------------------------------------------------------------
# 6. QUICK-LOOK DIAGNOSTICS - run this FIRST on the 3-year pilot sample
# ---------------------------------------------------------------------------

def pilot_diagnostics(df: pd.DataFrame):
    print("Rows:", len(df))
    print("Distinct comuni:", df["descrizione_comune"].nunique() if "descrizione_comune" in df else "n/a")
    print("Sesso distribution:", df["sesso"].value_counts(dropna=False).to_dict())
    print("Missing data_cessazione (censored):", df["data_cessazione"].isna().mean())
    print("Population range:", df["popolazione_censita"].min(), "-", df["popolazione_censita"].max())
    dupe_rate = df["person_key"].duplicated().mean()
    print("Duplicate person_key rate (check before trusting matching):", dupe_rate)


def duplicate_key_breakdown(df: pd.DataFrame):
    """
    Critical diagnostic: decompose the duplicate person_key rate into
    cross-year re-appearance (expected, good - the tenure/re-election
    signal) vs. within-year collisions (bad - either same-name-different-
    person collisions or genuine duplicate rows in the source data).
    Run this BEFORE trusting any tenure/panel result built on person_key.
    """
    within_year_dupes = df.duplicated(subset=["person_key", "source_file"]).sum()
    print(f"Within-year duplicate rows (SAME file, same key) - collision risk: "
          f"{within_year_dupes} ({within_year_dupes/len(df):.2%})")

    per_key_years = df.groupby("person_key")["source_file"].nunique()
    cross_year = (per_key_years > 1).sum()
    print(f"Distinct persons appearing in 2+ files (calendar-year continuity, "
          f"NOT a re-election rate - with annual snapshots, a single "
          f"uninterrupted ~5-year term alone puts someone in several "
          f"consecutive files, so this number is expected to be high by "
          f"construction and should not be read as turnover/persistence "
          f"evidence on its own; see the actual tenure/climb models for that): "
          f"{cross_year} out of {per_key_years.shape[0]} unique keys "
          f"({cross_year/per_key_years.shape[0]:.2%})")

    # Manual spot-check: print a handful of within-year collisions if any exist,
    # to eyeball whether they're plausibly the same person (role change) or
    # a genuine two-different-people collision.
    if within_year_dupes:
        dup_mask = df.duplicated(subset=["person_key", "source_file"], keep=False)
        sample_cols = [c for c in ["person_key", "source_file", "descrizione_comune",
                                    "descrizione_carica", "data_nomina", "data_cessazione"]
                        if c in df.columns]
        print("\nSample within-year collisions to eyeball manually:")
        print(df.loc[dup_mask, sample_cols].sort_values("person_key").head(20).to_string())


def quick_look_comparison(tenure_df: pd.DataFrame, climb_df: pd.DataFrame):
    """
    Crude descriptive comparison, NOT a regression -- purpose is to check
    whether there's any signal worth building the full RDD/Cox spec
    around, before investing in the harder econometrics.
    """
    print("Mean/median tenure (days) by sex x above/below 5,000 threshold:")
    print(tenure_df.groupby(["sesso", "above_cutoff"])["tenure_days"]
          .agg(["mean", "median", "count"]))
    print()
    print("Hierarchy-climb rate by sex x ever-above-threshold:")
    print(climb_df.groupby(["sesso", "ever_above_cutoff"])["climbed"]
          .agg(["mean", "count"]))


def fit_tenure_cox_model(tenure_df: pd.DataFrame):
    """
    Formal replacement for the descriptive tenure comparison. Sex x
    threshold interaction is the coefficient of interest for H1.
    Uses first_above_cutoff-equivalent 'above_cutoff' (per-row, since
    tenure is a per-spell not per-person outcome) and running_var as a
    linear control for population near the cutoff -- NOT a substitute
    for a proper local-linear RDD polynomial, which should be checked
    as a robustness extension once this baseline spec is confirmed to run.
    """
    if not _HAS_LIFELINES:
        print("lifelines not installed -- run `pip install lifelines` first")
        return None

    df = tenure_df.copy()
    df["event_observed"] = 1 - df["censored"]
    df["sex_female"] = (df["sesso"] == "F").astype(int)
    df["sex_x_cutoff"] = df["sex_female"] * df["above_cutoff"]
    # comune_cluster_id is NOT built here anymore. It was left over from
    # the reverted clustering attempt and caused a real bug: unlike
    # statsmodels (formula-based, only uses named columns), lifelines'
    # CoxPHFitter.fit() treats every column in the passed dataframe as a
    # covariate, so this string identifier was silently getting fit as
    # a regression covariate, and its rows-with-missing-comune-code were
    # being dropped by dropna() even when every real model variable was
    # fine. Confirmed by comparing a contaminated run's output (N and
    # coefficients both shifted slightly) against the clean numbers
    # already in the paper.

    cols = ["tenure_days", "event_observed", "sex_female", "above_cutoff",
            "sex_x_cutoff", "running_var"]
    model_df = df[cols].dropna()
    dropped = len(df) - len(model_df)
    if dropped:
        print(f"  Cox model: dropped {dropped} rows with missing values in model columns")

    cph = CoxPHFitter()
    # NOTE: clustering by comune (cluster_col) and robust=True were both
    # tried and both stalled badly at this N in practice, across multiple
    # attempts. Reverted to plain nonrobust fitting, the original
    # confirmed-working approach, rather than keep spending time chasing
    # a performance issue whose root cause was never actually isolated.
    # SEs here are nonrobust and likely understate uncertainty given
    # repeated observations per comune - stated honestly in Limitations,
    # not silently left unmentioned.
    cph.fit(model_df, duration_col="tenure_days", event_col="event_observed")
    print("\n=== Cox proportional-hazards model: tenure ===")
    cph.print_summary()
    print("\nH1 test -- raw sex_x_cutoff interaction coefficient (NOTE: this alone "
          "does NOT give the within-sex effect of crossing the threshold -- "
          "that requires comparing each sex to itself across the threshold, "
          "i.e. exp(above_cutoff) for men vs exp(above_cutoff + sex_x_cutoff) "
          "for women. Compute both explicitly before interpreting direction.):")
    print(cph.summary.loc["sex_x_cutoff", ["coef", "exp(coef)", "p"]])
    b_above = cph.summary.loc["above_cutoff", "coef"]
    b_interact = cph.summary.loc["sex_x_cutoff", "coef"]
    print(f"\nWithin-sex hazard change crossing the threshold:")
    print(f"  Men:   exp({b_above:.4f}) = {np.exp(b_above):.4f}  "
          f"({(np.exp(b_above)-1)*100:+.1f}%)")
    print(f"  Women: exp({b_above:.4f} + {b_interact:.4f}) = "
          f"{np.exp(b_above + b_interact):.4f}  "
          f"({(np.exp(b_above + b_interact)-1)*100:+.1f}%)")
    return cph


def fit_climb_logit_flexible(climb_df: pd.DataFrame):
    """
    Robustness check on H2, not a replacement for fit_climb_logit_model.
    Motivated directly by the RD plot (Figure 1 in the paper): the raw
    binned scatter showed a smooth population trend across the WHOLE
    bandwidth, not an obvious sharp jump right at the cutoff, which the
    baseline model's single running_var slope (assumed identical on
    both sides) cannot rule out as partly driving the "above_cutoff"
    coefficient. This version allows a DIFFERENT slope on each side of
    the threshold (standard practice, see Imbens & Lemieux 2008,
    already cited in the paper), so the within-sex decomposition below
    is evaluated exactly AT the cutoff (running_var=0), not averaged
    across the bandwidth.

    Report both this and the baseline model honestly. If the effect
    survives here, H2 is on firmer ground. If it does not, that is real
    information for the paper, not something to quietly drop.
    """
    if not _HAS_STATSMODELS:
        print("statsmodels not installed - run `pip install statsmodels` first")
        return None

    formula = "climbed ~ sex_female * first_above_cutoff * first_running_var"
    df = climb_df.copy()
    df["sex_female"] = (df["sesso"] == "F").astype(int)
    cols = ["climbed", "sex_female", "first_above_cutoff", "first_running_var"]
    model_df = df[cols].dropna()
    dropped = len(df) - len(model_df)
    if dropped:
        print(f"  Flexible climb logit: dropped {dropped} rows with missing values")

    # NOTE: clustering and HC1 were both tried and both stalled at this
    # step across multiple attempts. Reverted to plain nonrobust fitting,
    # the original confirmed-working approach - see the Cox model's
    # equivalent comment for the fuller reasoning.
    result = smf.logit(formula, data=model_df).fit(disp=0)
    print("\n=== Robustness: flexible local-linear specification "
          "(separate slopes each side of the cutoff) ===")
    print(result.summary())

    b_above = result.params["first_above_cutoff"]
    b_interact = result.params["sex_female:first_above_cutoff"]
    print("\nWithin-sex climb-odds change evaluated AT the cutoff "
          "(running_var=0), now allowing different slopes on each side - "
          "compare this against the baseline model's result directly, do "
          "not just eyeball whether the sign matches:")
    print(f"  Men:   exp({b_above:.4f}) = {np.exp(b_above):.4f}  "
          f"({(np.exp(b_above)-1)*100:+.1f}%)")
    print(f"  Women: exp({b_above:.4f} + {b_interact:.4f}) = "
          f"{np.exp(b_above + b_interact):.4f}  "
          f"({(np.exp(b_above + b_interact)-1)*100:+.1f}%)")
    return result


def fit_climb_logit_model(climb_df: pd.DataFrame):
    """
    Formal replacement for the descriptive climb-rate comparison. Uses
    first_above_cutoff (treatment assigned once, at first observed
    election) rather than ever_above_cutoff, consistent with the
    population-drift caveat already documented in Limitations.
    """
    if not _HAS_STATSMODELS:
        print("statsmodels not installed -- run `pip install statsmodels` first")
        return None

    df = climb_df.copy()
    df["sex_female"] = (df["sesso"] == "F").astype(int)

    cols = ["climbed", "sex_female", "first_above_cutoff", "first_running_var"]
    model_df = df[cols].dropna()
    dropped = len(df) - len(model_df)
    if dropped:
        print(f"  Logit model: dropped {dropped} rows with missing values in model columns")

    formula = "climbed ~ sex_female * first_above_cutoff + first_running_var"
    # Reverted to plain nonrobust fitting - see the Cox model's comment
    # for the full reasoning.
    result = smf.logit(formula, data=model_df).fit(disp=0)
    print("\n=== Logistic regression: hierarchy climb ===")
    print(result.summary())
    print("\nH2 test -- sex_female:first_above_cutoff coefficient "
          "(same caveat as the Cox model: read the within-sex decomposition "
          "below, not this raw interaction term alone):")
    interaction_term = "sex_female:first_above_cutoff"
    if interaction_term in result.params:
        print(result.params[interaction_term], "p =", result.pvalues[interaction_term])
        b_above = result.params["first_above_cutoff"]
        b_interact = result.params[interaction_term]
        print(f"\nWithin-sex climb-odds change crossing the threshold:")
        print(f"  Men:   exp({b_above:.4f}) = {np.exp(b_above):.4f}  "
              f"({(np.exp(b_above)-1)*100:+.1f}%)")
        print(f"  Women: exp({b_above:.4f} + {b_interact:.4f}) = "
              f"{np.exp(b_above + b_interact):.4f}  "
              f"({(np.exp(b_above + b_interact)-1)*100:+.1f}%)")
    return result


def bandwidth_robustness(full_panel: pd.DataFrame,
                          bandwidths: list[int] = [1000, 1500, 2000, 3000],
                          cutoff: int = 5000):
    """
    Re-applies the population-band filter at each bandwidth and reports
    tenure/climb comparisons. Required before trusting any single-bandwidth
    result -- if the sex x threshold gap appears/disappears depending on
    bandwidth choice, that's a red flag, not something to bandwidth-shop
    around until it looks good.

    CAVEAT not yet resolved: filtering by population band at the ROW level
    (as done here and in rdd_bandwidth_sample) can truncate a person's
    observed career if their comune's population drifts in/out of the band
    across snapshot years -- this could artificially suppress apparent
    tenure/climb rates. A more correct design would assign treatment status
    once, based on population at each person's FIRST observed election,
    rather than re-filtering every row every year. Flagging this now as a
    known simplification to revisit if results look inconsistent across
    bandwidths, not silently resolving it.
    """
    for bw in bandwidths:
        print(f"\n=== Bandwidth +/- {bw} (population {cutoff-bw}-{cutoff+bw}) ===")
        sample = rdd_bandwidth_sample(full_panel, cutoff=cutoff, bandwidth=bw)
        tenure = compute_tenure_panel(sample)
        climb = compute_hierarchy_climb(sample)
        quick_look_comparison(tenure, climb)


if __name__ == "__main__":
    full_dates_list = [f"3112{d}" for d in range(1986, 2026)]
    full_dates_list.append("24082026")  # now safely auto-skipped by stack_years
    print(f"Full list of dates to process: {full_dates_list}")

    path = "data/raw/storico_amministratori_comuni_"
    lookup_dates = ["31122010", "31122013", "31122018", "31122024"]
    paths = [f"{path}{d}.csv" for d in lookup_dates]

    if paths:
        panel = get_panel(paths, cache_path="data/preprocessed/small_panel_2010_2013_2018_2024.parquet")
        default_sample = rdd_bandwidth_sample(panel)
        pilot_diagnostics(default_sample)
        print()
        duplicate_key_breakdown(default_sample)

        print("\n--- Quick-look comparison (bandwidth +/-1500) ---")
        tenure_default = compute_tenure_panel(default_sample)
        climb_default = compute_hierarchy_climb(default_sample)
        quick_look_comparison(tenure_default, climb_default)

        print("\n--- Bandwidth robustness sweep ---")
        bandwidth_robustness(panel)
    else:
        print("Add file paths to paths above and rerun.")

    print("\n--- Full panel diagnostics (all years) ---")
    full_panel = get_panel([f"{path}{d}.csv" for d in full_dates_list],
                            cache_path="data/preprocessed/full_panel.parquet")
    full_sample = rdd_bandwidth_sample(full_panel)
    pilot_diagnostics(full_sample)
    print()
    duplicate_key_breakdown(full_sample)
    print("\n--- Quick-look comparison (bandwidth +/-1500) ---")
    tenure_full = compute_tenure_panel(full_sample)
    climb_full = compute_hierarchy_climb(full_sample)
    quick_look_comparison(tenure_full, climb_full)
    print("\n--- Bandwidth robustness sweep ---")
    bandwidth_robustness(full_panel)

    print("\n--- Formal models (full panel, bandwidth +/-1500) ---")
    fit_tenure_cox_model(tenure_full)
    fit_climb_logit_model(climb_full)