from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


# ------------------------------------------------------------------
# Expected column names in the input CSV
# ------------------------------------------------------------------
@dataclass
class ColumnNames:
    ts: str     = "ts_event"   # timestamp in nanoseconds
    symbol: str = "symbol"     # ticker name
    side: str   = "side"       # "A" = ask, "B" = bid
    level: str  = "level"      # derived level = depth + 1
    price: str  = "price"
    size: str   = "size"
    depth: str  = "depth"


# ------------------------------------------------------------------
# OFI for a single level and single side
# ------------------------------------------------------------------
def ofi_one_level_one_side(pp: float, pq: float,
                           cp: float, cq: float,
                           is_bid: bool) -> float:
    """
    This function follows the OFI definition in the article:
    "Cross-Impact of Order Flow Imbalance in Equity Markets".
    It calculates the signed change in size for a single (side, level)
    between two book snapshots.
    """
    if np.isnan(pp) or np.isnan(cp):
        return 0.0

    if is_bid:  # bid logic
        if cp > pp:
            return +cq
        elif cp < pp:
            return -pq
        else:
            return cq - pq
    else:  # ask logic (signs reversed)
        if cp < pp:
            return +cq
        elif cp > pp:
            return -pq
        else:
            return cq - pq


# ------------------------------------------------------------------
# OFI across levels 1 to L between two snapshots
# ------------------------------------------------------------------
def ofi_between_rows_exact(prev_price: pd.Series,
                           prev_size : pd.Series,
                           curr_price: pd.Series,
                           curr_size : pd.Series,
                           L: int = 1) -> float:
    """
    This accumulates the OFI across bid and ask sides
    for levels 1 to L. Needed for Multi-Level and Integrated OFI.
    """
    ofi = 0.0
    for side, is_bid in (("B", True), ("A", False)):
        for lvl in range(1, L + 1):
            pp = prev_price.get((side, lvl), np.nan)
            cp = curr_price.get((side, lvl), np.nan)
            pq = prev_size .get((side, lvl), 0.0)
            cq = curr_size .get((side, lvl), 0.0)

            ofi += ofi_one_level_one_side(pp, pq, cp, cq, is_bid)

    return ofi


# ------------------------------------------------------------------
# Main engine to build all OFI features
# ------------------------------------------------------------------
def build_ofi_features(events: DataFrame,
                       resample_rule: str = "1s",
                       L: int = 1,
                       int_window: str = "10s",
                       schema: ColumnNames = ColumnNames()) -> DataFrame:
    """
    Calculates OFI features:
      - Best-Level OFI (L = 1)
      - Multi-Level OFI (L > 1)
      - Integrated OFI (rolling sum)
      - Cross-Asset OFI (simple copied columns)

    resample_rule sets the time grid (e.g. '1s')
    int_window is the rolling sum window (e.g. '10s')
    """
    df = events.copy()

    # parse timestamp, filter only A/B sides, set level
    df[schema.ts] = pd.to_datetime(df[schema.ts], unit="ns", utc=True, errors="coerce")
    df = df[df[schema.side].isin(["A", "B"])]
    df[schema.side] = df[schema.side].astype("category")
    df[schema.level] = df[schema.depth].astype(int) + 1
    df.set_index(schema.ts, inplace=True)
    df = df[[schema.symbol, schema.side, schema.level, schema.price, schema.size]]

    features_by_asset: Dict[str, DataFrame] = {}

    # process each symbol independently
    for sym, grp in df.groupby(schema.symbol, sort=False):
        grp = grp.drop(columns=schema.symbol)

        # reshape to wide format: MultiIndex columns (side, level)
        price_wide = grp.pivot_table(index=df.index.name,
                                     columns=[schema.side, schema.level],
                                     values=schema.price,
                                     aggfunc="last")
        size_wide = grp.pivot_table(index=df.index.name,
                                    columns=[schema.side, schema.level],
                                    values=schema.size,
                                    aggfunc="last")

        # forward-fill values to fill missing events
        price_filled = price_wide.ffill()
        size_filled  = size_wide.ffill()

        # put book on a regular grid (e.g. every 1s)
        price_snap = price_filled.resample(resample_rule).last().dropna(how="all")
        size_snap  = size_filled .resample(resample_rule).last().dropna(how="all")
        price_snap, size_snap = price_snap.align(size_snap, join="outer", axis=1)

        # calculate OFI between each pair of timestamps
        idx = price_snap.index
        ofi_vals: List[Tuple[pd.Timestamp, float]] = []
        for t_prev, t_curr in zip(idx[:-1], idx[1:]):
            ofi_val = ofi_between_rows_exact(
                price_snap.loc[t_prev], size_snap.loc[t_prev],
                price_snap.loc[t_curr], size_snap.loc[t_curr], L)
            ofi_vals.append((t_curr, ofi_val))

        # build time series and rolling version
        base_name = "ofi_best" if L == 1 else f"ofi_multi_L{L}"
        ofi_series = pd.Series(dict(ofi_vals), name=base_name)
        ofi_int    = ofi_series.rolling(int_window).sum().rename(f"ofi_int_{int_window}")
        features_by_asset[sym] = pd.concat([ofi_series, ofi_int], axis=1)

    # join all symbols into one DataFrame
    aligned = [df_feat.add_suffix(f"_{sym}") for sym, df_feat in features_by_asset.items()]
    feat_all = pd.concat(aligned, axis=1).sort_index()

    # add simple cross-asset features: copy OFI of j into i
    symbols = list(features_by_asset)
    for i, j in itertools.permutations(symbols, 2):
        base = "ofi_best" if L == 1 else f"ofi_multi_L{L}"
        feat_all[f"ofi_xasset_{j}_on_{i}"] = feat_all[f"{base}_{j}"]

    return feat_all.dropna(how="all")


# ------------------------------------------------------------------
# Command-line wrapper
# ------------------------------------------------------------------
def _cli():
    # define arguments
    parser = argparse.ArgumentParser("OFI feature builder (CSV only)")
    parser.add_argument("--csv", required=True, help="Path to CSV input file")
    parser.add_argument("--out", default="ofi_features.csv", help="Output file name")
    parser.add_argument("--freq", default="1s", help="Snapshot frequency (e.g. '1s')")
    parser.add_argument("--levels", type=int, default=1, help="Book depth L")
    parser.add_argument("--window", default="10s", help="Rolling window for integrated OFI")
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)
    feats = build_ofi_features(raw,
                               resample_rule=args.freq,
                               L=args.levels,
                               int_window=args.window)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    feats.to_csv(args.out)
    print(f"✓ Features saved to {args.out}  (shape = {feats.shape})")


if __name__ == "__main__":
    _cli()
