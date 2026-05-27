# Credit Risk Modelling & Portfolio Optimisation

A two-stage **predict-then-optimise** pipeline for consumer-loan portfolio selection on the LendingClub dataset (2007–2018 Q4). A CatBoost classifier estimates loan-level probability of default; a Pyomo + HiGHS binary integer program then selects a return-maximising subset of loans under explicit risk, capital, and diversification constraints. The full Pareto frontier between expected return and credit risk is traced with the ε-constraint method and benchmarked against a PD-threshold heuristic.

This notebook is the computational core of a master's thesis on bi-objective credit portfolio optimisation. 

---

## Pipeline overview

```
            ┌─────────────────────────────────────┐
            │   LendingClub raw CSV (~2.2M rows)  │
            └──────────────────┬──────────────────┘
                               │
                  target filtering, leakage removal,
                   feature engineering, chronological
                            80/20 split
                               │
            ┌──────────────────▼──────────────────┐
            │   CatBoost PD model (Optuna-tuned)  │
            │   AUC-ROC 0.756 · Brier 0.139       │
            └──────────────────┬──────────────────┘
                               │
                  per-loan PD, EL = PD·LGD·EAD,
                  net return = (1−PD)·interest − EL
                               │
            ┌──────────────────▼──────────────────┐
            │  Oracle Autonomous DB  (persistence)│
            └──────────────────┬──────────────────┘
                               │
                  5,000-loan deterministic sample
                       (ORA_HASH, seed 42)
                               │
            ┌──────────────────▼──────────────────┐
            │  Pyomo BIP · HiGHS solver           │
            │  ε-constraint sweep (20 grid points)│
            │  Risk measures: EL · aggregate PD   │
            └──────────────────┬──────────────────┘
                               │
                    Pareto frontier + heuristic
                          comparison
```

---

## What the notebook does

**Phase 1 — Probability of Default**

1. Loads the LendingClub *accepted* loans CSV.
2. Filters to loans with terminal outcomes (`Fully Paid` / `Charged Off`).
3. Removes data-leakage columns (post-origination payment activity, post-origination FICO refreshes, distress-state flags) and redundant identifiers.
4. Sorts chronologically by `issue_d` and applies a time-series 80/20 train/test split.
5. Parses string-encoded numerics (`int_rate`, `revol_util`, `term`, `emp_length`, `earliest_cr_line`) and engineers five domain features (`loan_to_income`, `est_installment`, `payment_to_income`, `shock_dti`, `dti_x_loan_to_inc`).
6. Estimates LGD empirically on training-fold charge-offs (recovery rate ≈ 42 %, LGD ≈ 0.58).
7. Trains a CatBoost classifier; learning rate, depth, and L2 leaf regularisation tuned with Optuna over 20 trials; early stopping on a 10 % chronological validation slice.
8. Reports four threshold-independent metrics on the held-out test set (AUC-ROC, AUC-PR, log-loss, Brier) plus a reliability diagram and a train/validation/test generalisation diagnostic.

**Phase 2 — Portfolio optimisation**

9. Persists loan-level PD, EL, and net expected return to Oracle Autonomous Database.
10. Draws a deterministic 5,000-loan sub-sample via `ORA_HASH(loan_id, 4294967295, 42)`.
11. Builds the bi-objective BIP in Pyomo: maximise return subject to a capital cap (30 % of pool by volume), a parametric risk cap ε, and cardinality bounds [50, 2000].
12. Solves the ε-constraint sweep with HiGHS at 20 grid points, under both Expected Loss and aggregate Probability of Default as the risk measure.
13. Runs the PD-threshold heuristic baseline at four cutoffs (0.10, 0.15, 0.20, 0.25) with a per-dollar greedy fill.
14. Compares the two methods at matched risk via linear interpolation of the BIP frontier and reports the optimality gap in dollars and percent.

---

## Dataset

LendingClub *Accepted* Loans, 2007 – 2018 Q4, approximately 2.26 million originations across 151 raw columns. The notebook expects the file at:

```
/kaggle/input/datasets/wordsforthewise/lending-club/accepted_2007_to_2018Q4.csv.gz
```

Downloadable from the public Kaggle distribution:
[`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club).


---

## Environment

- Python 3.12
- Originally executed on a Kaggle Notebook with the standard `kaggle/python` image. 

| Package | Used for |
|---|---|
| `pandas`, `numpy` | data manipulation |
| `catboost` | gradient-boosted PD classifier |
| `optuna` | hyperparameter search |
| `scikit-learn` | metrics and calibration curve |
| `matplotlib` | static figures |
| `oracledb` | Oracle Autonomous Database client |
| `pyomo` | mathematical-programming modelling layer |
| `highspy` | HiGHS MIP solver |

---

## How to run

**1. Get the data.** Download the LendingClub *Accepted* loans CSV from the Kaggle dataset linked above and place it at the path referenced in cell 3, or edit that path to point wherever you have the file.

**2. Provision the database (optional but recommended).** The notebook persists Phase-1 outputs to an Oracle Autonomous Database so Phase 2 can run independently. To use this layer:
   - Create an Oracle Autonomous Database instance (the *Always Free* tier is sufficient).
   - Provide three credentials. On Kaggle these are read from `UserSecretsClient`:
     - `DB_USER`
     - `DB_PASSWORD`
     - `CONNECT_STRING` 

   *Skipping the database.* If you only want to run Phase 1, the notebook works end-to-end without the database — the PD model, metrics, reliability diagram, and overfitting diagnostic do not depend on it. For Phase 2 without a database, replace the `loans = ...` query in cell 51 with an in-memory sample from `portfolio_df`.

**3. Run.** Execute the notebook top to bottom. 
---

