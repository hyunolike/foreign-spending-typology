# Foreign Spending Heatmap & Latent Tourism District Discovery

**English** · [한국어](README.ko.md)

Analysis code for the 1st AI Financial Big Data Platform Consumption Data Analysis &
Idea Competition (hosted by BC Card).

The pipeline isolates the `GENDER_CD=3` (foreign customer) axis from BC Card's
municipality × industry monthly aggregates (Jan–Jun 2026), **segments foreign
spending into behavioural types, and surfaces latent tourism districts**.

> **No data is committed to this repository.** The competition data may not be used
> outside the competition, redistributed, or published, so `data/` and `outputs/` are
> git-ignored. Only code is shared; place the source CSV locally at
> `data/ABP_CONTEST_DATA.csv`.

## Running

```bash
pip install -r requirements.txt
# after placing data/ABP_CONTEST_DATA.csv
python src/run_all.py

# external join (after placing data/external/moj_registered_foreigners.csv)
python src/run_external.py

# submission PDF (run after the charts exist)
pip install playwright && playwright install chromium
python report/render.py
```

This produces 8 charts in `outputs/figures`, 6 CSVs plus run metadata
(`run_meta.json`, `external_meta.json` — metric snapshots for reproducibility) in
`outputs/tables`, and the submission PDF in `outputs/report`. Korean fonts are
auto-detected from the Nanum/Noto families (Ubuntu: `apt-get install -y fonts-nanum`).

## The problem

Foreign spending totals ₩1.278tn, or 7.4% of all spending. Yet once Jeju is set
aside, the top districts by amount are Siheung, Ansan Danwon, Hwaseong Manse,
Pyeongtaek and Asan — **industrial complexes and factory-worker clusters**. The top
districts by *share* look the same: Yeongam (31.8%), Eumseong (26.5%), Jincheon (23.3%).

Foreign spending mixes tourists with resident foreigners. **Picking tourism districts
by volume alone gives the wrong answer — the types must be separated first.**

## Method

| Step | Detail |
|---|---|
| Scope | 199 municipalities with ≥ ₩1bn in foreign spending over six months |
| Exclusion | Jeju City and Seogwipo are scale outliers: held out of training, used as a benchmark → 197 trained |
| Features | 9, all computed on foreign spending: 5 industry shares + 2 age shares + foreign share of local spending + log ticket size |
| Clustering | StandardScaler + K-means; highest silhouette among k ≥ 3 |
| Naming rule | Highest cafe/western → Urban Visitor; of the rest, highest groceries → Industrial Worker; remainder → Middle-aged Resident |
| Growth | Relative growth = (foreign Q2/Q1) ÷ (domestic Q2/Q1) − 1 |
| Candidates | 0.4 × similarity + 0.4 × relative growth + 0.2 × youth share (each as a percentile) |

## Results (reproduced by `python src/run_all.py`)

- Silhouette: k=2 0.231, **k=3 0.242**, k=4 0.175, k=5 0.193, k=6 0.185 → k=3
- Mean ARI 1.000 across 10 seeds (clusters are stable)

| Type | Districts | Share of foreign spending | Mean relative growth |
|---|---|---|---|
| Industrial Worker | 68 | 39.0% | −0.8% |
| Middle-aged Resident | 79 | 24.9% | −0.3% |
| Urban Visitor | 50 | 21.5% | **+1.4%** |

The remaining 14.5% belongs to the two Jeju districts (12.3%) and to small districts
below the ₩1bn threshold (2.2%).

- Flagship districts (excluded from candidates): Gangnam, Incheon Yeonsu, Suwon Paldal, Seoul Jung-gu, Mapo
- 58 candidates; top by composite score: Suwon Yeongtong → Hongseong (relative growth +11.5%) → Nowon → Gangneung → Seongdong

## External validation (MOJ monthly registered foreigners)

Registered-foreigner counts per municipality (Jan–Jun 2026 monthly average) were
joined, matching **all 199 districts**.

### Check 1 — was the segmentation right?

| Type | Correlation (log-log) | R² | Spending per registered foreigner (median) |
|---|---|---|---|
| Industrial Worker | **+0.945** | 0.892 | ₩610k |
| Middle-aged Resident | +0.811 | 0.658 | ₩870k |
| Urban Visitor | **+0.588** | 0.345 | ₩690k |

Registered-foreigner counts alone explain 89% of Industrial Worker spending — the
**resident-demand reading is confirmed by external data**. Urban Visitor spending, by
contrast, is only 35% explained: demand that does not reduce to the resident
population is mixed in. Jeju reaches ₩5.49m per registered foreigner, 6–9× the three
types (₩610k–870k).

### Check 2 — Visit Demand Index (VDI)

```
log(foreign spend) = 6.84 + 0.566·log(registered foreigners) + 0.420·log(domestic spend) + e
R² = 0.829, n = 197,  VDI = e
```

Controlling for domestic spending is **mandatory**. Drop that term and the residual
captures "commercial district size" rather than tourism, pushing industrial areas like
Ansan Danwon and Siheung to the top.

Jeju's VDI is +2.09 (Jeju City) and +1.64 (Seogwipo) — 5–7× the residual standard
deviation (0.31). Jeju, a known tourism destination, confirms that tourism demand does
surface in this residual.

### Check 3 — the first-pass candidates were wrong

Both assumptions behind the internal-data-only score collapsed.

| First-pass candidate | New rank | VDI | Spending per registered foreigner |
|---|---|---|---|
| Suwon Yeongtong | 1st → 21st | −0.59 | ₩360k |
| Seoul Nowon | 3rd → 32nd | −0.68 | ₩500k |
| Seoul Gangbuk | 7th → 47th | −0.69 | ₩350k |

1. **Youth share selects student clusters, not visit demand.** The first-pass leaders
   spend ₩350k–530k per registered foreigner, below the Urban Visitor median (₩690k).
2. **Similarity to the Urban Visitor centroid points the wrong way.** The 75th-percentile
   cutoff was discarding Dongseongno (Daegu Jung-gu), Gwangalli (Busan Suyeong),
   Itaewon (Yongsan) and Hongdae (Mapo). Districts near the centroid turned out to be
   the *average* university neighbourhood.

### Final top 5 latent tourism districts

Score = 0.6 × VDI percentile + 0.4 × relative-growth percentile, restricted to VDI > 0
(places where visit demand is actually observed).

| Rank | District | VDI | Relative growth | Spending per registered foreigner |
|---|---|---|---|---|
| 1 | Daegu Jung-gu (Dongseongno) | +0.75 | **+12.9%** | **₩4.11m** |
| 2 | Cheongju Seowon-gu | +0.38 | +3.4% | ₩1.03m |
| 3 | Gangneung | +0.04 | +3.4% | ₩980k |
| 4 | Busan Haeundae-gu | +0.09 | +1.5% | ₩1.32m |
| 5 | Incheon Jung-gu (Chinatown, airport) | +0.31 | +0.0% | ₩1.14m |

Daegu Jung-gu has only 995 registered foreigners yet ₩4.1bn in foreign spending —
₩4.11m per resident foreigner, 6× the Urban Visitor median, and the highest relative
growth of any candidate.

## Reading the results with care

- University and research districts (Yeongtong, Nowon, Dongjak, Gangbuk, Yuseong)
  cluster near the top of the first-pass list. Urban Visitor is therefore not pure
  tourism but **young urban presence (tourism + study + business)**.
- Jeju sits far from all three centroids (nearest distance 5.1–6.5 vs a median of 1.9
  for ordinary districts). It is its own type; forced to choose, it is closest to
  Middle-aged Resident. **There is no single tourism profile.**
- The PCA map carries only 64% of the variance, so Jeju's distinctness must be judged
  from the 9-dimensional standardized distance, not from the 2-D picture.

## Limitations

- Figures are **BC Card transactions only** and differ from the full card market.
- The definition behind the foreign customer code is not published (whether it covers
  overseas-issued cards only, or also cards held by resident foreigners), so the
  tourist/resident split is an inference from spending patterns.
- Six months of data limits annual seasonality correction; a domestic-relative metric
  mitigates but does not remove this.
- VDI captures a **mismatch between where people live and where they spend**. It picks
  up not only foreign tourists but also resident foreigners spending outside their own
  district. Separating the two requires KTO's regional foreign visitor counts
  (see `docs/external_data.md`, items B1/B2).
- Registered-foreigner counts include only stays longer than 90 days, so short-term
  visitors are missing and resident scale may be understated in some districts.

## Layout

```
src/config.py       paths, constants, Korean font setup
src/preprocess.py   loading, industry-name cleanup, small-sample merges, region keys
src/regions.py      administrative-name normalization for external joins (incl. new Hwaseong districts)
src/features.py     the 9 features + scope/training/benchmark split
src/cluster.py      k selection, stability (ARI), clustering, rule-based naming, centroid distances
src/growth.py       seasonality-adjusted relative growth, monthly index by type
src/candidates.py   candidate scoring (internal-only and external-join versions)
src/external.py     MOJ registered-foreigner loader (fixes 3 source-data defects)
src/residual.py     VDI regression and per-type validation
src/viz.py          8 charts
src/run_all.py      base pipeline
src/run_external.py external join stage
report/report.html  submission summary document (A4, 9 pages, Korean)
report/render.py    PDF rendering via Chromium
```

## Further reading

The supporting documents are written in Korean:

- [README.ko.md](README.ko.md) — this document in Korean
- [docs/data_notes.md](docs/data_notes.md) — preprocessing decisions and their justification
- [docs/external_data.md](docs/external_data.md) — external data survey, source-data defects, join design
