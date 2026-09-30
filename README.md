# evalsq

**Benchmark evaluation heuristics — eval2 workshop demo**

Three heuristics for auditing whether your ML benchmark is still telling you the truth:

| # | Heuristic | Question it answers |
|---|-----------|---------------------|
| H1 | Benchmark Validity Modeling | Does accuracy (benchmark) track Sharpe (business value)? Rolling 24-mo correlation shows decay. |
| H2 | Temporal Holdout | Do model rankings flip under regime change? Would you trust 2022 labels to catch 2025 failures? |
| H3 | Predictability Test | Is benchmark accuracy predictable from surface market features? Leave-one-out R²; high = score tracks regime, not model quality |

Uses SPY (S&P500 ETF) daily data + two simple direction-prediction models (LogReg, RF).

---

## Quick start

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone / download this repo, then:
cd evalsq
uv sync

# 3. Run the demo  (downloads SPY data automatically on first run)
uv run evalsq

# 4. Results land in results/: h1_validity.csv, h1_rolling.csv, h2_temporal.csv, h3_meta.csv
```

### Options

```
uv run evalsq --help

  --data   PATH   path to cached SPY CSV  (default: spy_daily.csv)
  --out    DIR    output directory         (default: results/)
  --cutoff YEAR   train/test split year    (default: 2018)
```

---

## Run tests

```bash
uv run pytest
```

## Layout

```
evalsq/data.py        # SPY download/cache + features (target = next-day direction)
evalsq/models.py      # make_models() zoo (LogReg, RF) — add models here
evalsq/heuristics.py  # H1 / H2 / H3
evalsq/cli.py         # entrypoint
tests/                # unit + E2E (synthetic data, no network)
```
