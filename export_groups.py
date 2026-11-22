#!/usr/bin/env python3
"""
Export LIM-Eval duplicate-group assignments for the D3 API-call dataset.

Output: groups/d3_duplicate_groups.csv with one row per raw sample:
  row_index, hash, malware, exact_group, near_group_tau0.95, near_group_tau0.9, near_group_tau0.8,
  in_dedup_set, label_conflict
Use exact_group / near_group_* as the `groups` argument of StratifiedGroupKFold to reproduce
Scenarios B and C. The dataset itself is NOT redistributed; download it from IEEE DataPort
(doi:10.21227/tqqm-aq14).

Usage: python export_groups.py --csv dynamic_api_call_sequence_per_malware_100_0_306.csv
"""
import argparse, os
import numpy as np, pandas as pd
import limeval_d3 as L

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="groups")
ap.add_argument("--taus", type=float, nargs="+", default=[0.95, 0.90, 0.80])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(a.csv)
X, y = L.load_d3(a.csv)
keys = [tuple(r) for r in X]
ids = {}
sid = np.array([ids.setdefault(k, len(ids)) for k in keys])
useqs = [list(k) for k in ids]
L.log(f"{len(X)} rows, {len(useqs)} unique sequences")
near, _ = L.near_duplicate_groups(useqs, a.taus)

seen, keep = set(), np.zeros(len(X), bool)
for i, k in enumerate(zip(sid, y)):
    if k not in seen:
        seen.add(k); keep[i] = True
labels = pd.Series(y).groupby(sid).nunique()
out = pd.DataFrame({"row_index": np.arange(len(X)), "hash": df["hash"], "malware": y,
                    "exact_group": sid})
for t in a.taus:
    # relabel components 0..k-1 for readability
    comp = pd.factorize(near[t])[0]
    out[f"near_group_tau{t}"] = comp[sid]
out["in_dedup_set"] = keep.astype(int)
out["label_conflict"] = (labels.reindex(sid).values > 1).astype(int)
out.to_csv(f"{a.out}/d3_duplicate_groups.csv", index=False)
L.log(f"Saved {a.out}/d3_duplicate_groups.csv")
