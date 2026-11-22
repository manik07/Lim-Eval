#!/usr/bin/env python3
"""
LIM-Eval experiments on the API-call dataset (D3, Oliveira & Sassi, IEEE DataPort doi:10.21227/tqqm-aq14).

Experiments
  1  Exact + near-duplicate audit (M1)            -> results/exp1_*.csv
  2  Scenarios A / B / C, repeated group CV (M3)  -> results/exp2_*.csv
  3  Label-conflict audit + E1/E2/E3 (M2)         -> results/exp3_*.csv
  4  Calibration + threshold stability (M4)       -> results/exp4_*.csv
  5  Statistics: Friedman, Wilcoxon-Holm, bootstrap CI, Cliff's delta, Kendall tau (M5)
                                                  -> results/exp5_*.csv
  A summary of every value needed for the [TBD] fields is written to results/SUMMARY.txt

Input: the original CSV with columns  hash, t_0 ... t_99, malware   (malware: 1 = malicious, 0 = benign)

Requirements: python>=3.9, numpy, pandas, scipy, scikit-learn
Recommended:  rapidfuzz (fast Levenshtein; pure-Python fallback is ~100x slower)
Optional:     torch (for the LSTM; skipped automatically if not installed)

Examples
  python limeval_d3.py --csv dynamic_api_call_sequence_per_malware_100_0_306.csv
  python limeval_d3.py --csv d3.csv --repeats 5 --folds 5 --weighted
  python limeval_d3.py --csv d3.csv --quick            # 1 repeat, no LSTM: fast smoke test
"""
import argparse, json, os, time, warnings, copy
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score, average_precision_score, confusion_matrix)

warnings.filterwarnings("ignore")
N_CODES = 307
SEQ_LEN = 100

try:
    from rapidfuzz.distance import Levenshtein as _RFLev
    def lev_sim(a, b):
        return _RFLev.normalized_similarity(a, b)
except ImportError:
    _RFLev = None
    def lev_sim(a, b):  # normalised Levenshtein similarity, Eq. (1)
        n, m = len(a), len(b)
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m
            ai = a[i - 1]
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
            prev = cur
        return 1.0 - prev[m] / max(n, m, 1)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False


# ----------------------------------------------------------------------------- utilities
class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)
    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
    def labels(self):
        return np.array([self.find(i) for i in range(len(self.p))])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_d3(path):
    df = pd.read_csv(path)
    cols = [f"t_{i}" for i in range(SEQ_LEN)]
    X = df[cols].to_numpy(dtype=np.int64)
    y = df["malware"].to_numpy(dtype=np.int64)
    return X, y


def counts(Xseq):
    out = np.zeros((len(Xseq), N_CODES), dtype=np.float32)
    for i, s in enumerate(Xseq):
        np.add.at(out[i], s, 1)
    return out


# ----------------------------------------------------------------------------- Experiment 1: near-duplicates
def shingles(seq, k=3):
    return {tuple(seq[i:i + k]) for i in range(len(seq) - k + 1)} or {tuple(seq)}


def minhash_signatures(seqs, n_perm=128, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31 - 1, n_perm, dtype=np.uint64)
    b = rng.integers(0, 2**31 - 1, n_perm, dtype=np.uint64)
    sig = np.empty((len(seqs), n_perm), dtype=np.uint64)
    for i, s in enumerate(seqs):
        hs = np.array([hash(sh) & 0xFFFFFFFF for sh in shingles(s)], dtype=np.uint64)
        sig[i] = ((np.outer(hs, a) + b) % np.uint64(4294967311)).min(axis=0)  # (a*h + b) mod prime
    return sig


def near_duplicate_groups(useqs, taus, n_perm=128, bands=32, max_bucket=2000):
    """useqs: list of unique sequences (lists of ints). Returns {tau: group label per unique seq}."""
    taus = sorted(taus, reverse=True)
    n = len(useqs)
    log(f"  MinHash signatures for {n} unique sequences ({n_perm} permutations)")
    sig = minhash_signatures(useqs, n_perm)
    rows = n_perm // bands
    ufs = {t: UnionFind(n) for t in taus}
    strict = ufs[taus[0]]
    n_cand = n_cmp = 0
    sims_hist = []
    for bnd in range(bands):
        buckets = defaultdict(list)
        block = sig[:, bnd * rows:(bnd + 1) * rows]
        for i in range(n):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > max_bucket:  # very large buckets: compare each member to a random anchor subset
                rng = np.random.default_rng(bnd)
                anchors = rng.choice(members, 50, replace=False)
                pairs = ((a, m) for a in anchors for m in members if a != m)
            else:
                pairs = combinations(members, 2)
            for i, j in pairs:
                n_cand += 1
                # skip if already connected even under the strictest tau (then connected under all)
                if strict.find(i) == strict.find(j):
                    continue
                s = lev_sim(useqs[i], useqs[j])
                n_cmp += 1
                if len(sims_hist) < 200000:
                    sims_hist.append(s)
                for t in taus:
                    if s >= t:
                        ufs[t].union(i, j)
        if (bnd + 1) % 8 == 0:
            log(f"    band {bnd + 1}/{bands}: candidates {n_cand:,}, Levenshtein comparisons {n_cmp:,}")
    return {t: ufs[t].labels() for t in taus}, np.array(sims_hist)


def experiment1(Xraw, yraw, taus, out):
    log("Experiment 1: exact and near-duplicate audit")
    keys = [tuple(r) for r in Xraw]
    seq_id = {}
    sid = np.array([seq_id.setdefault(k, len(seq_id)) for k in keys])
    useqs = [list(k) for k in seq_id]  # unique sequences, index = sid
    n_raw = len(Xraw)

    pair_keys = list(zip(sid, yraw))
    dup_exact = n_raw - len(set(pair_keys))
    seq_counts = Counter(sid)
    labels_per_seq = defaultdict(set)
    for s, l in zip(sid, yraw):
        labels_per_seq[s].add(int(l))

    near, sims = near_duplicate_groups(useqs, taus)

    # de-duplicated data set as in the paper: unique (sequence, label) pairs
    seen, keep = set(), []
    for i, k in enumerate(pair_keys):
        if k not in seen:
            seen.add(k); keep.append(i)
    keep = np.array(keep)

    rows = []
    rows.append(dict(level="exact", tau=1.0, raw_rows=n_raw, unique_sequences=len(useqs),
                     duplicate_rows=dup_exact, duplicate_rate=dup_exact / n_raw,
                     max_group_size=max(seq_counts.values()),
                     groups_in_dedup=len(set(sid[keep])), dedup_samples=len(keep)))
    for t in sorted(near, reverse=True):
        g = near[t][sid]                      # group of every raw row
        gk = g[keep]
        sizes = Counter(gk)
        in_multi = sum(c for c in sizes.values() if c > 1)
        # near-duplicate rate on the de-duplicated set: samples sharing a group with >=1 other sample
        rows.append(dict(level="near", tau=t, raw_rows=n_raw, unique_sequences=len(useqs),
                         duplicate_rows=np.nan, duplicate_rate=np.nan,
                         max_group_size=max(sizes.values()), groups_in_dedup=len(sizes),
                         dedup_samples=len(keep),
                         near_dup_rate_dedup=in_multi / len(keep),
                         extra_merged_vs_exact=len(set(sid[keep])) - len(sizes)))
    pd.DataFrame(rows).to_csv(f"{out}/exp1_duplicate_audit.csv", index=False)
    pd.Series(sims, name="levenshtein_similarity").to_csv(f"{out}/exp1_candidate_similarities.csv", index=False)
    return dict(sid=sid, keep=keep, near=near, labels_per_seq=labels_per_seq, seq_counts=seq_counts)


# ----------------------------------------------------------------------------- models
def standardise(Xtr, *others):
    sc = StandardScaler().fit(Xtr)
    return [sc.transform(Xtr)] + [sc.transform(o) for o in others]


def fit_mlp(Xtr, ytr, Xva, yva, seed, max_epochs=60, patience=8):
    Xtr, Xva = standardise(Xtr, Xva)
    m = MLPClassifier(hidden_layer_sizes=(64, 128, 128), activation="relu", solver="adam",
                      random_state=seed, max_iter=1, warm_start=True)
    best, best_auc, bad = None, -1, 0
    for _ in range(max_epochs):
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1]) if len(set(yva)) > 1 else 0
        if auc > best_auc + 1e-4:
            best, best_auc, bad = copy.deepcopy(m), auc, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best


if HAS_TORCH:
    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(N_CODES, 32)
            self.lstm = nn.LSTM(32, 64, batch_first=True)
            with torch.no_grad():  # forget-gate bias = 1
                for name, p in self.lstm.named_parameters():
                    if "bias" in name:
                        p.zero_(); p[64:128] = 0.5   # bias_ih + bias_hh = 1
            self.out = nn.Linear(64, 1)
        def forward(self, x):
            h, _ = self.lstm(self.emb(x))
            return self.out(h[:, -1]).squeeze(-1)

    def fit_lstm(Xtr, ytr, Xva, yva, seed, weighted=False, max_epochs=40, patience=8, bs=64):
        torch.manual_seed(seed); np.random.seed(seed)
        net = LSTMNet().to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=0.002)
        if weighted:
            n, n1 = len(ytr), ytr.sum()
            w = {0: n / (2 * (n - n1)), 1: n / (2 * n1)}
        Xt = torch.tensor(Xtr, device=DEVICE); yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
        best, best_auc, bad = None, -1, 0
        for ep in range(max_epochs):
            net.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for k in range(0, len(Xt), bs):
                idx = perm[k:k + bs]
                logits = net(Xt[idx])
                wt = None
                if weighted:
                    wt = torch.where(yt[idx] == 1, torch.tensor(w[1], device=DEVICE),
                                     torch.tensor(w[0], device=DEVICE)).float()
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yt[idx], weight=wt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            p = predict_lstm(net, Xva)
            auc = roc_auc_score(yva, p) if len(set(yva)) > 1 else 0
            if auc > best_auc + 1e-4:
                best, best_auc, bad = {k: v.detach().clone() for k, v in net.state_dict().items()}, auc, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best)
        return net

    def predict_lstm(net, X, bs=1024):
        net.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(X), bs):
                out.append(torch.sigmoid(net(torch.tensor(X[k:k + bs], device=DEVICE))).cpu().numpy())
        return np.concatenate(out)


def model_names(weighted, use_lstm):
    names = ["RF_depth2", "RF_full", "LR", "LinearSVM", "KNN5", "NN"]
    if use_lstm:
        names.append("LSTM")
    if weighted:
        names += ["RF_depth2_w", "RF_full_w", "LR_w", "LinearSVM_w"] + (["LSTM_w"] if use_lstm else [])
    return names


def fit_predict(name, seqs_tr, ytr, seqs_va, yva, seqs_te, seed):
    """Returns (score_val, score_test, default_threshold, is_probability)."""
    w = "balanced" if name.endswith("_w") else None
    base = name[:-2] if name.endswith("_w") else name
    if base == "LSTM":
        net = fit_lstm(seqs_tr, ytr, seqs_va, yva, seed, weighted=w is not None)
        return predict_lstm(net, seqs_va), predict_lstm(net, seqs_te), 0.5, True
    Ctr, Cva, Cte = counts(seqs_tr), counts(seqs_va), counts(seqs_te)
    if base.startswith("RF"):
        m = RandomForestClassifier(n_estimators=100, max_depth=2 if base == "RF_depth2" else None,
                                   class_weight="balanced_subsample" if w else None,
                                   random_state=seed, n_jobs=-1).fit(Ctr, ytr)
        return m.predict_proba(Cva)[:, 1], m.predict_proba(Cte)[:, 1], 0.5, True
    if base == "NN":
        m = fit_mlp(Ctr, ytr, Cva, yva, seed)
        _, Sva, Ste = standardise(Ctr, Cva, Cte)
        return m.predict_proba(Sva)[:, 1], m.predict_proba(Ste)[:, 1], 0.5, True
    Str, Sva, Ste = standardise(Ctr, Cva, Cte)
    if base == "LR":
        m = LogisticRegression(max_iter=2000, class_weight=w).fit(Str, ytr)
        return m.predict_proba(Sva)[:, 1], m.predict_proba(Ste)[:, 1], 0.5, True
    if base == "LinearSVM":
        m = LinearSVC(class_weight=w, max_iter=5000, dual=False).fit(Str, ytr)
        return m.decision_function(Sva), m.decision_function(Ste), 0.0, False
    if base == "KNN5":
        m = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(Str, ytr)
        return m.predict_proba(Sva)[:, 1], m.predict_proba(Ste)[:, 1], 0.5, True
    raise ValueError(name)


# ----------------------------------------------------------------------------- metrics (malicious = positive)
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def metrics(y, score, thr, is_prob):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    benign_score = -score
    r = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
             f1_benign=f1_score(y, pred, pos_label=0, zero_division=0), mcc=matthews_corrcoef(y, pred),
             auc_roc=roc_auc_score(y, score), pr_auc_benign=average_precision_score(1 - y, benign_score),
             benign_recall=tn / (tn + fp) if tn + fp else np.nan,
             fpr=fp / (fp + tn) if fp + tn else np.nan,   # benign files flagged as malware
             tp=tp, tn=tn, fp=fp, fn=fn, threshold=thr)
    if is_prob:
        r["brier"] = float(np.mean((score - y) ** 2)); r["ece"] = ece(y, score)
    else:
        r["brier"] = r["ece"] = np.nan
    return r


def tune_threshold(yva, sva):
    """Threshold maximising MCC on validation data (vectorised over 201 quantile candidates)."""
    cand = np.unique(np.quantile(sva, np.linspace(0, 1, 201)))
    pred = sva[None, :] >= cand[:, None]
    y1 = (yva == 1)[None, :]
    tp = (pred & y1).sum(1).astype(float); fp = (pred & ~y1).sum(1).astype(float)
    fn = (~pred & y1).sum(1).astype(float); tn = (~pred & ~y1).sum(1).astype(float)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    return float(cand[int(np.argmax(mcc))])


def threshold_bootstrap_sd(yva, sva, n_boot, seed):
    rng = np.random.default_rng(seed); ts = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yva), len(yva))
        if len(set(yva[idx])) < 2:
            continue
        ts.append(tune_threshold(yva[idx], sva[idx]))
    return float(np.std(ts)) if ts else np.nan


# ----------------------------------------------------------------------------- cross-validation engine
def split_iter(y, groups, folds, repeats, grouped):
    for r in range(repeats):
        if grouped:
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=r)
            yield from ((r, f, tr, te) for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)))
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=r)
            yield from ((r, f, tr, te) for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)))


def inner_validation(y_tr, g_tr, grouped, seed):
    """10% validation subset of the training fold, group-disjoint when grouped=True."""
    if grouped:
        cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        fit, val = next(cv.split(np.zeros(len(y_tr)), y_tr, g_tr))
    else:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        fit, val = next(cv.split(np.zeros(len(y_tr)), y_tr))
    return fit, val


def run_cv(tag, seqs, y, groups, grouped, names, args, leak_groups=None, out=None):
    """Returns long DataFrame: one row per (repeat, fold, model, threshold_mode)."""
    rows = []
    for r, f, tr, te in split_iter(y, groups, args.folds, args.repeats, grouped):
        leak = {}
        if leak_groups is not None:  # share of test samples whose (exact / near) group also occurs in training
            for lname, g in leak_groups.items():
                trg = set(g[tr]); leak[f"leak_{lname}"] = float(np.mean([gg in trg for gg in g[te]]))
        fit, val = inner_validation(y[tr], groups[tr] if groups is not None else None, grouped, seed=100 + r)
        itr, iva = tr[fit], tr[val]
        for name in names:
            t0 = time.time()
            sva, ste, dthr, isp = fit_predict(name, seqs[itr], y[itr], seqs[iva], y[iva], seqs[te], seed=r)
            tt = tune_threshold(y[iva], sva)
            tsd = threshold_bootstrap_sd(y[iva], sva, args.boot_thr, seed=r) if args.boot_thr else np.nan
            for mode, thr in (("default", dthr), ("tuned", tt)):
                m = metrics(y[te], ste, thr, isp)
                m.update(scenario=tag, repeat=r, fold=f, model=name, threshold_mode=mode,
                         thr_boot_sd=tsd, train_seconds=time.time() - t0, **leak)
                rows.append(m)
        log(f"  {tag}: repeat {r + 1}/{args.repeats}, fold {f + 1}/{args.folds} done")
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(f"{out}/exp2_folds_{tag}.csv", index=False)
    return df


def summarise(df, by=("scenario", "model", "threshold_mode")):
    cols = ["accuracy", "balanced_accuracy", "f1_benign", "mcc", "auc_roc", "pr_auc_benign",
            "benign_recall", "fpr", "brier", "ece", "threshold", "thr_boot_sd"]
    cols = [c for c in cols if c in df]
    g = df.groupby(list(by))[cols]
    s = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    return s.reset_index()


# ----------------------------------------------------------------------------- Experiment 5: statistics
def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def experiment5(all_folds, out):
    log("Experiment 5: statistics")
    res_rows, pair_rows, tau_rows = [], [], []
    for (sc, mode), d in all_folds.groupby(["scenario", "threshold_mode"]):
        piv = d.pivot_table(index=["repeat", "fold"], columns="model", values="mcc")
        piv = piv.dropna(axis=1)
        models = list(piv.columns)
        if len(models) >= 3:
            chi, p = stats.friedmanchisquare(*[piv[m].values for m in models])
        else:
            chi, p = np.nan, np.nan
        ranks = piv.rank(axis=1, ascending=False).mean()
        for m in models:
            lo, hi = boot_ci(piv[m].values)
            res_rows.append(dict(scenario=sc, threshold_mode=mode, model=m, mcc_mean=piv[m].mean(),
                                 ci95_low=lo, ci95_high=hi, mean_rank=ranks[m],
                                 friedman_chi2=chi, friedman_p=p, n_blocks=len(piv)))
        prs, raw = list(combinations(models, 2)), []
        for a, b in prs:
            diff = piv[a] - piv[b]
            raw.append(stats.wilcoxon(piv[a], piv[b]).pvalue if np.any(diff != 0) else 1.0)
        for (a, b), pr, ph in zip(prs, raw, holm(raw) if raw else []):
            pair_rows.append(dict(scenario=sc, threshold_mode=mode, model_a=a, model_b=b,
                                  mean_diff=(piv[a] - piv[b]).mean(), wilcoxon_p=pr, holm_p=ph,
                                  cliffs_delta=cliffs_delta(piv[a], piv[b])))
    pd.DataFrame(res_rows).to_csv(f"{out}/exp5_friedman_ci_ranks.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(f"{out}/exp5_pairwise_wilcoxon_holm_cliff.csv", index=False)

    # Kendall tau between rankings (by mean MCC) across scenarios and threshold modes
    mean = all_folds.groupby(["scenario", "threshold_mode", "model"])["mcc"].mean()
    settings = sorted(set((s, t) for s, t, _ in mean.index))
    for s1, s2 in combinations(settings, 2):
        a, b = mean.loc[s1], mean.loc[s2]
        common = a.index.intersection(b.index)
        if len(common) >= 3:
            tau, p = stats.kendalltau(a[common], b[common])
            tau_rows.append(dict(setting_1=f"{s1[0]}/{s1[1]}", setting_2=f"{s2[0]}/{s2[1]}",
                                 kendall_tau=tau, p_value=p, n_models=len(common)))
    pd.DataFrame(tau_rows).to_csv(f"{out}/exp5_kendall_tau.csv", index=False)
    return pd.DataFrame(res_rows), pd.DataFrame(pair_rows), pd.DataFrame(tau_rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.80, 0.90, 0.95])
    ap.add_argument("--tau-c", type=float, default=0.95, help="tau used for Scenario C groups")
    ap.add_argument("--weighted", action="store_true", help="also run class-weighted variants")
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--boot-thr", type=int, default=200, help="bootstrap resamples for threshold stability (0 = off)")
    ap.add_argument("--quick", action="store_true", help="1 repeat, no LSTM, 50 threshold bootstraps")
    ap.add_argument("--models", nargs="+", default=None, help="run only these models, e.g. --models RF_full NN LSTM")
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.no_lstm, args.boot_thr = 1, True, 50
    if args.tau_c not in args.taus:
        args.taus.append(args.tau_c)
    os.makedirs(args.out, exist_ok=True)
    use_lstm = HAS_TORCH and not args.no_lstm
    if not HAS_TORCH and not args.no_lstm:
        log("PyTorch not installed: LSTM skipped (pip install torch to include it)")
    if _RFLev is None:
        log("rapidfuzz not installed: using slow pure-Python Levenshtein (pip install rapidfuzz)")
    names = model_names(args.weighted, use_lstm)
    if args.models:
        names = [m for m in names if m in args.models]
    if use_lstm:
        log(f"LSTM device: {DEVICE}")

    Xraw, yraw = load_d3(args.csv)
    log(f"Loaded {len(Xraw)} sequences ({(yraw == 0).sum()} benign, {(yraw == 1).sum()} malicious)")

    # ---- Experiment 1
    e1 = experiment1(Xraw, yraw, args.taus, args.out)
    sid, keep, near = e1["sid"], e1["keep"], e1["near"]

    # ---- Experiment 3a: label-conflict audit
    log("Experiment 3: label-conflict audit")
    conflict_sids = {s for s, ls in e1["labels_per_seq"].items() if len(ls) == 2}
    conf_mask_raw = np.isin(sid, list(conflict_sids))
    conf_rows = []
    for s in conflict_sids:
        m = sid == s
        seq = Xraw[np.argmax(m)]
        top = Counter(seq.tolist()).most_common(5)
        conf_rows.append(dict(sequence_id=int(s), occurrences=int(m.sum()), malicious=int(yraw[m].sum()),
                              benign=int((yraw[m] == 0).sum()), distinct_codes=len(set(seq.tolist())),
                              top5_codes=" ".join(f"{c}x{n}" for c, n in top),
                              first_20_calls=" ".join(map(str, seq[:20]))))
    conf_cols = ["sequence_id", "occurrences", "malicious", "benign", "distinct_codes", "top5_codes", "first_20_calls"]
    pd.DataFrame(conf_rows, columns=conf_cols).sort_values("occurrences", ascending=False).to_csv(
        f"{args.out}/exp3_conflicting_sequences.csv", index=False)
    conf_summary = dict(conflicting_sequences=len(conflict_sids),
                        affected_raw_rows=int(conf_mask_raw.sum()),
                        affected_raw_share=float(conf_mask_raw.mean()),
                        affected_dedup_rows=int(conf_mask_raw[keep].sum()),
                        affected_dedup_share=float(conf_mask_raw[keep].mean()))

    # ---- Experiment 2: Scenarios A, B, C
    seqs_raw = Xraw
    seqs_d, y_d, sid_d = Xraw[keep], yraw[keep], sid[keep]
    gC = near[args.tau_c][sid_d]
    leak_raw = {"exact": sid, **{f"near{t}": near[t][sid] for t in args.taus}}
    leak_d = {"exact": sid_d, **{f"near{t}": near[t][sid_d] for t in args.taus}}

    log("Experiment 2: Scenario A (random CV on raw data)")
    fa = run_cv("A_random", seqs_raw, yraw, sid, False, names, args, leak_raw, args.out)
    log("Experiment 2: Scenario B (exact-group CV on de-duplicated data)")
    fb = run_cv("B_exact_group", seqs_d, y_d, sid_d, True, names, args, leak_d, args.out)
    log(f"Experiment 2: Scenario C (near-duplicate-group CV, tau={args.tau_c})")
    fc = run_cv("C_near_group", seqs_d, y_d, gC, True, names, args, leak_d, args.out)
    all_folds = pd.concat([fa, fb, fc], ignore_index=True)
    summ = summarise(all_folds)
    summ.to_csv(f"{args.out}/exp2_summary_by_scenario.csv", index=False)

    # inflation A->B, A->C (default threshold)
    piv = summ[summ.threshold_mode == "default"].pivot(index="model", columns="scenario",
                                                       values=["mcc_mean", "balanced_accuracy_mean", "pr_auc_benign_mean"])
    infl = pd.DataFrame({f"delta_{m}_{a}_to_{b}": piv[(f"{m}_mean", a)] - piv[(f"{m}_mean", b)]
                         for m in ("mcc", "balanced_accuracy", "pr_auc_benign")
                         for a, b in (("A_random", "B_exact_group"), ("A_random", "C_near_group"),
                                      ("B_exact_group", "C_near_group"))})
    infl.to_csv(f"{args.out}/exp2_inflation.csv")
    leak_cols = [c for c in all_folds if c.startswith("leak_")]
    all_folds.groupby("scenario")[leak_cols].mean().to_csv(f"{args.out}/exp2_leakage_rates.csv")

    # ---- Experiment 3b: E1 / E2 / E3 under Scenario B
    log("Experiment 3: label treatments E1 / E2 / E3 (Scenario B)")
    e3_models = [m for m in ("RF_full", "LSTM", "NN", "LR") if m in names]
    m2 = ~np.isin(sid_d, list(conflict_sids))
    # E3: majority label per sequence on raw data (ties -> malicious), then one row per sequence
    maj = {}
    for s in np.unique(sid):
        ys = yraw[sid == s] if s in conflict_sids else None
        maj[s] = (1 if ys.mean() >= 0.5 else 0) if ys is not None else None
    first = {}
    for i, s in enumerate(sid):
        first.setdefault(s, i)
    idx3 = np.array(sorted(first.values()))
    y3 = np.array([maj[sid[i]] if maj[sid[i]] is not None else yraw[i] for i in idx3])
    e3 = []
    for tag, X_, y_, g_ in (("E1_original", seqs_d, y_d, sid_d),
                            ("E2_remove_conflicts", seqs_d[m2], y_d[m2], sid_d[m2]),
                            ("E3_majority_label", Xraw[idx3], y3, sid[idx3])):
        d = run_cv(tag, X_, y_, g_, True, e3_models, args)
        d["n_samples"], d["n_benign"] = len(y_), int((y_ == 0).sum())
        e3.append(d)
    e3 = pd.concat(e3, ignore_index=True)
    e3.to_csv(f"{args.out}/exp3_label_treatments_folds.csv", index=False)
    summarise(e3).to_csv(f"{args.out}/exp3_label_treatments_summary.csv", index=False)

    # ---- Experiment 4: calibration and threshold stability (Scenario B and C)
    log("Experiment 4: calibration and threshold stability")
    bc = all_folds[all_folds.scenario.isin(["B_exact_group", "C_near_group"])]
    e4 = bc[bc.threshold_mode == "default"].groupby(["scenario", "model"]).agg(
        brier_mean=("brier", "mean"), ece_mean=("ece", "mean"), auc_mean=("auc_roc", "mean"))
    tuned = bc[bc.threshold_mode == "tuned"].groupby(["scenario", "model"]).agg(
        tuned_thr_mean=("threshold", "mean"), tuned_thr_sd_across_folds=("threshold", "std"),
        tuned_thr_bootstrap_sd=("thr_boot_sd", "mean"), mcc_tuned=("mcc", "mean"))
    dflt = bc[bc.threshold_mode == "default"].groupby(["scenario", "model"]).agg(mcc_default=("mcc", "mean"))
    e4 = e4.join(tuned).join(dflt)
    e4["mcc_gain_from_tuning"] = e4["mcc_tuned"] - e4["mcc_default"]
    e4.to_csv(f"{args.out}/exp4_calibration_threshold.csv")

    # ---- Experiment 5
    res5, pairs5, tau5 = experiment5(all_folds, args.out)

    # ---- SUMMARY for the [TBD] fields
    a1 = pd.read_csv(f"{args.out}/exp1_duplicate_audit.csv")
    lines = ["LIM-Eval D3 results: values for the [TBD] fields", "=" * 60,
             f"Models: {', '.join(names)} | folds={args.folds} repeats={args.repeats} | tau_C={args.tau_c}", "",
             "Exp 1 (Sec 5.2, Table 5):", a1.to_string(index=False), "",
             "Exp 3 conflict audit (Sec 5.3, Table 5):", json.dumps(conf_summary, indent=2), "",
             "Leakage rates by scenario (Table 5 / Table 6):",
             all_folds.groupby("scenario")[leak_cols].mean().round(4).to_string(), "",
             "Exp 2 inflation (Table 5, Sec 5.2):", infl.round(4).to_string(), "",
             "Exp 2 summary, default threshold (Tables 6, 8):",
             summ[summ.threshold_mode == "default"][["scenario", "model", "accuracy_mean", "balanced_accuracy_mean",
                                                    "f1_benign_mean", "mcc_mean", "mcc_sd", "auc_roc_mean",
                                                    "pr_auc_benign_mean"]].round(4).to_string(index=False), "",
             "Exp 3 label treatments (Table 7):",
             summarise(e3)[lambda d: d.threshold_mode == "default"][["scenario", "model", "mcc_mean", "pr_auc_benign_mean",
                                                                     "benign_recall_mean", "fpr_mean"]].round(4).to_string(index=False), "",
             "Exp 4 calibration and thresholds (Sec 5.5):", e4.round(4).to_string(), "",
             "Exp 5 Kendall tau (Sec 5.6, Table 5, Table 13):", tau5.round(4).to_string(index=False), "",
             "Exp 5 Friedman / CI / ranks (Sec 5.6):", res5.round(4).to_string(index=False), "",
             "Exp 5 key pairs (Sec 5.6):",
             pairs5[pairs5.apply(lambda r: {r.model_a, r.model_b} in ({"RF_full", "LSTM"}, {"NN", "LSTM"},
                                                                     {"RF_full", "NN"}), axis=1)]
             .round(4).to_string(index=False)]
    open(f"{args.out}/SUMMARY.txt", "w").write("\n".join(lines))
    log(f"Done. See {args.out}/SUMMARY.txt")


if __name__ == "__main__":
    main()
