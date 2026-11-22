# LIM-Eval: Leakage- and Imbalance-aware Malware Evaluation

Code, duplicate-group assignments and results for the paper
**"LIM-Eval: A Leakage- and Imbalance-Aware Framework for Assessing the Reliability of Machine-Learning Malware Detection Benchmarks."**

LIM-Eval checks whether malware-detection results reflect genuine generalisation rather than
duplicate memorisation, contradictory labels or decision-threshold effects. It has five modules:

| Module | What it does |
|---|---|
| M1 Duplicate leakage audit | Exact duplicates (hashing) and near-duplicates (MinHash + normalised Levenshtein similarity), grouped with union–find |
| M2 Label-conflict audit | Finds identical sequences carrying both labels; compares treatments E1 (original), E2 (remove), E3 (majority label) |
| M3 Leakage-controlled partitioning | Scenario A random CV, B exact-group CV, C near-duplicate-group CV (stratified, grouped, repeated) |
| M4 Imbalance & operating point | MCC, balanced accuracy, benign PR-AUC, Brier score, ECE, validation-tuned thresholds and their stability |
| M5 Reliability & statistics | Performance inflation, Kendall τ rank stability, Friedman + Holm-corrected Wilcoxon, bootstrap CIs, Cliff's δ |

## Dataset

The API-call dataset (D3) is **not redistributed** here. Download it from IEEE DataPort:
A. Oliveira, *Malware Analysis Datasets: API Call Sequences*, doi:[10.21227/tqqm-aq14](https://doi.org/10.21227/tqqm-aq14)
(file `dynamic_api_call_sequence_per_malware_100_0_306.csv`; 43,876 sequences, 1,079 benign, 42,797 malicious).

## Installation

```bash
pip install -r requirements.txt
```
Python ≥ 3.9. `rapidfuzz` is strongly recommended (fast Levenshtein). `torch` is needed only for the LSTM; a GPU is used automatically if available.

## Reproducing the results

```bash
# quick smoke test (1 repeat, no LSTM)
python limeval_d3.py --csv dynamic_api_call_sequence_per_malware_100_0_306.csv --quick --out results_quick

# full experiments reported in the paper (3 × 5-fold CV, tau = 0.95 for Scenario C)
python limeval_d3.py --csv dynamic_api_call_sequence_per_malware_100_0_306.csv --repeats 3 --tau-c 0.95 --out results_full

# export duplicate-group assignments
python export_groups.py --csv dynamic_api_call_sequence_per_malware_100_0_306.csv
```
The full run took about 2.5 hours on Google Colab with a T4 GPU.

## Repository contents

| Path | Content |
|---|---|
| `limeval_d3.py` | Complete LIM-Eval pipeline (experiments 1–5) |
| `export_groups.py` | Exports exact and near-duplicate group IDs for every sample |
| `groups/d3_duplicate_groups.csv` | Group assignments keyed by row index and the dataset's `hash` column; use `exact_group` or `near_group_tau0.95` as `groups` in `StratifiedGroupKFold` |
| `results_full/` | All result files reported in the paper (see below) |

### Result files

| File | Content |
|---|---|
| `SUMMARY.txt` | Overview of all key numbers |
| `exp1_duplicate_audit.csv` | Exact and near-duplicate rates, groups, largest group |
| `exp2_folds_*.csv`, `exp2_summary_by_scenario.csv` | Per-fold and summary metrics for scenarios A, B, C |
| `exp2_leakage_rates.csv`, `exp2_inflation.csv` | Leakage rates and MCC inflation between scenarios |
| `exp3_conflicting_sequences.csv`, `exp3_label_treatments_*.csv` | Label-conflict audit and E1/E2/E3 results |
| `exp4_calibration_threshold.csv` | Brier, ECE, tuned thresholds and their stability |
| `exp5_*.csv` | Friedman tests, bootstrap CIs, pairwise Wilcoxon–Holm and Cliff's δ, Kendall τ |

## Key findings (D3)

- 69.9% of rows are exact duplicates; after exact de-duplication, 55.1% of samples still have a near-duplicate (similarity ≥ 0.95).
- Random splits overstate MCC by 0.11–0.24 (vs. exact-group) and 0.14–0.29 (vs. near-duplicate-group splits).
- 15 sequences carry both labels (3.7% of raw rows).
- Validation-tuned thresholds raise random-forest MCC by 0.10–0.14 and change the ranking of detectors.

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## License

MIT License (see `LICENSE`).
