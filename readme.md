# Beyond the Threshold: Individual-Level Persistence and Advancement Among Quota-Elected Women in Italian Municipal Politics

Replication package for the Computational Social Science final exam
paper (MSc Data Science, University of Trento). This README exists so
that a stranger, or a random professor willing to give a good grade to an exam,
can reproduce every number in the paper without having to reverse-engineer it from the paper.

## Project Objective

I built an individual-level panel of every Italian local administrator
from 1986 to 2026 (sindaci, vicesindaci, assessori, consiglieri) 
from Ministry of Interior open data, and used it to ask a
question the existing literature on Italy's 2012 gender-quota law
hasn't: not "did the quota increase the share of women elected"
(answered, repeatedly, by better-funded people than me), but "do the
women who get elected via the quota actually build careers, or is it a
revolving door?" Short version of the answer: they don't get pushed out
faster than men (a real, if modest, finding against a naive tokenism
story), but they do advance up the hierarchy less. 
A little spoiler: Parties don't seem to be quietly overcompensating for this quota 
by recruiting more of their brothers-in-law. 
Full details, appropriately hedged, in the paper ;)

## Repository structure

```
.
├── panel_pipeline.py       # Core pipeline: data loading, cleaning,
│                           # panel construction, primary RQ models
├── dynasty_network.py      # Secondary RQ: dynastic recruitment analysis
│                           # (imports shared functions from panel_pipeline.py)
├── paper_draft.tex         # The paper itself
├── data/
│   ├── raw/                # Downloaded source CSVs go here (not tracked --
│   │                       # see "Getting the data" below)
│   └── preprocessed/       # Auto-generated cache (.parquet) -- also not
│                           # tracked, rebuilds itself on first run
└── README.md               # This file
```

## Prerequisites

```bash
pip install pandas numpy lifelines statsmodels pyarrow
```

`pyarrow` is required for the caching layer (Parquet). Everything else
does the actual statistics.

## Getting the data

This is the one step I genuinely cannot automate for you, and not for
lack of trying - the Ministry of Interior's open data portal
(`dait.interno.gov.it`) actively blocks automated/bot requests, which I
discovered the hard way after several increasingly creative attempts.
So: manual download it is.

1. Go to `dait.interno.gov.it` and locate the historical administrator
   archive.
2. Download each year's file, named
   `storico_amministratori_comuni_DDMMYYYY.csv` (year-end snapshots,
   31 December, plus one live snapshot for the current date - ours is
   `storico_amministratori_comuni_24082026.csv`, yours will differ if
   you're reading this later).
   - **Watch for a naming inconsistency**: 2014 and 2015 use an
     underscore before the date (`storico_amministratori_comuni_31122015.csv`),
     other years generally don't. Both patterns exist; check what you
     actually got before assuming the loop in `panel_pipeline.py` will
     find your files.
3. Place all downloaded files in `data/raw/`.
4. Full year coverage used in the paper: 1986–2025 (annual) plus the
   24 August 2026 live snapshot. You do not strictly need every single
   year to get directionally similar results - the small 4-year pilot
   (2010/2013/2018/2024) reproduces the same qualitative story, just
   with noisier estimates - but the paper's reported numbers are from
   the full panel, so that's what you'll need to match them exactly.

## Running it

```bash
python panel_pipeline.py     # Primary analysis: builds the panel, runs
                              # descriptive diagnostics, bandwidth
                              # robustness sweep, Cox model, logit model
python dynasty_network.py    # Secondary analysis: dynastic recruitment
```

Run `panel_pipeline.py` first - `dynasty_network.py` reuses its cached
panel (`data/preprocessed/full_panel.parquet`) rather than rebuilding
from scratch, which saves several minutes of runtime. (approx. a couple each run)

**Expected runtime**: roughly 6 minutes for `panel_pipeline.py` on the
full 1986–2026 panel (measured: 348.5s), assuming the cache doesn't
exist yet and has to be built from raw files. Subsequent runs against
an existing cache are much faster, since the expensive part (parsing
40-odd CSVs with per-file encoding detection and schema validation) only
has to happen once. `dynasty_network.py` runs in well under two minutes
once the shared cache exists.

## Things that will bite you if you don't read this part

- **The cache is a trap if you change the pipeline logic.** If you
  modify anything upstream of `stack_years()` (encoding handling, dedup
  rules, column aliasing, the entity-resolution key) after a cache
  already exists, delete `data/preprocessed/*.parquet` first, or every
  downstream script will silently keep analyzing data built under the
  old logic with no error to tell you so. I found this out the
  educational way.
- **Encoding is not consistent across 40 years of government CSVs.**
  Some files are `cp1252`, some are `utf-8`, at least one has a BOM
  that will break a naive parser. `load_storico()` tries a fallback
  chain (`utf-8-sig` → `utf-8` → `cp1252` → `latin-1`) and reports
  which one worked for each file - if a file you expect to load
  silently doesn't appear in the output, check the schema-validation
  log line explaining why it was skipped, don't assume it's fine.
- **The most recent (live) snapshot has no cessation date, by
  definition**: everyone in it is still serving. This is handled
  correctly (treated as right-censored), not as missing data, but if
  you're staring at the output wondering why that file contributes zero
  "events" to the Cox model, that's why, it's a feature, not a bug!
- **The dynasty-detection surname match is a proxy, not verified
  genealogy**, and uses a fixed 20-year lookback window specifically
  because an earlier "ever since 1986" version turned out to be
  mechanically confounded with calendar time (details in the paper's
  Limitations section, and in `dynasty_network.py`'s docstrings if you
  want the blow-by-blow). If you change `window_years`, expect the
  post-2013 coefficient to move, and that's expected, not a red flag.
  treat it as a sensitivity check.

## Reproducibility note

Every number quoted in the paper corresponds directly to a printed line
in these scripts' output: I did not hand-copy anything through an
intermediate spreadsheet, which is exactly the kind of step where
transcription errors like to hide. If a number in the paper doesn't
match your own run, the most likely explanations, in order of
probability, are: (1) you're missing some year files, (2) your cache is
stale, (3) something in the Ministry's data changed since I downloaded
it (it happens, and I wouldn't be surprised given the data entry and annotation
quality, or lack of, showcased throughout the datasets).

## Citation

If any part of this pipeline is useful to you beyond this exam
(surname-based dynasty detection with lookback-window correction, the
government-CSV encoding fallback chain, or just anything at all, feel
free to reuse it. No formal citation required for a course project, but
a mention and a message would be nice to receive beforehand.
Have a good day :)