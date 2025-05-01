# OFI Feature Extraction

This repo contains a small, self-contained script that builds **Order Flow Imbalance (OFI)** features from raw order-book data, following the definitions in  
Cont, Cucuringu & Zhang (2023) *“Cross-Impact of Order Flow Imbalance in Equity Markets”*.

---

## Folder layout

```
.
├── code.py                  ← main script
├── input/                   ← raw data (CSV) goes here
│   └── first_25000_rows.csv
├── output/                  ← generated features
│   ├── ofi_best.csv
│   └── ofi_multilevel.csv
└── README.md
```

*`Blockhouse___Project.pdf` (the short write-up) is also kept in the root.*

---

## Quick start

1. **Install deps** (only pandas + numpy):

```bash
pip install pandas numpy
```

2. **Run**:

```bash
# best-level OFI every second
python code.py --csv input/first_25000_rows.csv \
               --out output/ofi_best.csv \
               --levels 1 --freq 1s --window 10s

# multi-level OFI (levels 1-3) every 2 s
python code.py --csv input/first_25000_rows.csv \
               --out output/ofi_multilevel.csv \
               --levels 3 --freq 2s --window 10s
```

The script prints something like:

```
✓ Features saved to output/ofi_best.csv  (shape = (1249, 2))
```

---

## What you get

| Column                              | Meaning                                  |
|-------------------------------------|------------------------------------------|
| `ofi_best_<TICKER>`                 | best-level OFI (level 1)                 |
| `ofi_multi_L3_<TICKER>`             | OFI summed over levels 1-3               |
| `ofi_int_10s_<TICKER>`              | 10-second rolling sum (Integrated OFI)   |
| `ofi_xasset_GOOG_on_AAPL`, …        | simple cross-asset copy of another OFI   |

*Positive OFI → buy pressure.  
Negative OFI → sell pressure.*

---

## CLI flags

| Flag          | Default            | Purpose                               |
|---------------|--------------------|---------------------------------------|
| `--csv`       | *required*         | input CSV path                        |
| `--out`       | `ofi_features.csv` | output file (CSV)                     |
| `--freq`      | `1s`               | snapshot grid, e.g. `500ms`, `2s`     |
| `--levels`    | `1`                | depth L (1 = best-level)              |
| `--window`    | `10s`              | rolling window for Integrated OFI     |

---

### Author

Saúl Abraham Granados Carmona
