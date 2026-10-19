# MERIT

**MERIT** (Module-level Ensembles of Rules for Interpretable Tabular classification) is an interpretable classifier for
small high-dimensional data (few objects, thousands of features), such as gene expression, DNA methylation or
metagenomic profiles. It is trained only on the sample at hand: no pre-trained weights, no GPU.

This repository accompanies the paper

> A. Sharifbaev, N. Mirzaev and Sh. Fazilov, "MERIT: Interpretable Module-Level Rule Ensembles for Small High-Dimensional Data,"
> submitted to *IEEE Access*, 2026.

## Method

1. **Modules.** Features are standardised and grouped by k-means on their profiles into K modules of strongly
   correlated features.
2. **Module scores.** Inside each module an L2 logistic regression (C = 0.1) summarises the module by one score.
   Training scores are out-of-fold (3-fold), so the rules below see honest scores.
3. **Threshold rules.** Two-dimensional rules `w1*s_a + w2*s_b > t` over pairs of module scores (shrinkage Fisher
   direction, midpoint threshold). Rules are admitted above a selection noise floor
   `tau = 1/2 + 1/4 * sqrt(1/n1 + 1/n0) * sqrt(2 ln N)`, correlated rules (|r| > 0.9) are pruned, at most 31 rules per
   class, weights `1/2 ln(q/(1-q))`.
4. **Vote.** The rule votes of six partitions (K = 10, 20, 40, two seeds each) are summed.

## Installation

```bash
git clone https://github.com/sharifbayevab/merit
cd merit
pip install -e .
```

## Usage

```python
from merit import MERIT

model = MERIT(random_state=0).fit(X_train, y_train)   # X: objects x features, y: class labels
y_pred = model.predict(X_test)
rules = model.explain(feature_names=names, top=5)      # rules, modules and their most weighted features
```

`examples/quickstart.py` runs MERIT on a synthetic problem and prints three rules.

Main parameters: `Ks=(10, 20, 40)`, `n_seeds=2`, `C=0.1`, `n_folds=3`, `max_rules=31`, `rho=0.9`, `lam=0.1`,
`noise_floor=True`. All settings in the paper are these defaults; they were fixed on the eight development datasets.

## Reproducing the paper

`experiments/` holds the scripts used for the paper and `results/` the per-split outputs (balanced accuracy per
dataset, split and method) from which the tables and figures are computed.

| Step | Script |
|---|---|
| Data protocol (splits, cleaning) | `common.py` (development sets), `p4_bench.py` (TabBench-Bio) |
| Download TabBench-Bio tables | `p4_fetch.py` (needs a checkout of [TabBench-Bio](https://github.com/not-a-feature/TabBench-Bio)) |
| Baselines and feature-level rules | `methods.py`, `p4_bench.py`, `p7_ideas.py`, `p8_neurorules.py` |
| MERIT and ablations | `p9_ablation.py` (method `ENS6NF` is MERIT) |
| Sample-size curve | `p11_small_n.py` |
| Main tables | `p12_tables.py` |
| Figures | `p13_figures.py`, `p22_figures.py` |
| Interactions, timing | `p14_interactions.py`, `p15_interactions_report.py`, `p16_timing.py`, `p17_speed_table.py` |
| Gaussian simulation | `p18_simulation.py`, `p19_sim_table.py` |
| Interpretation case study | `p20_interpret.py`, `p20b_tissue_stability.py`, `p21_interp_table.py` |
| Foundation-model input width | `p23_width.py`, `p24_width_report.py` |
| Classical p >> n baselines | `p25_classic.py`, `p26_classic_report.py` |

Data:
* development sets: the eight scikit-feature `.mat` files (colon, ALLAML, Prostate_GE, GLI_85, SMK_CAN_187, TOX_171,
  lung, arcene) from <https://jundongl.github.io/scikit-feature/datasets.html>, placed in `data/`;
* held-out sets: TabBench-Bio classification tasks, exported to `data_tb/` by `p4_fetch.py`.

Run a script from `experiments/`, e.g. `python p9_ablation.py dev:colon ENS6NF`.
`merit/` is a clean reimplementation of `ENS6NF`; it gives identical predictions on the paper's splits.

Experiments used Python 3.11, numpy 2.2, scipy 1.16, scikit-learn 1.6, pandas 2.2; TabPFN v2 and TabICL v2 for the
foundation-model baselines.

## Citation

```bibtex
@article{sharifbaev2026merit,
  author  = {Sharifbaev, Abdurakhmon and Mirzaev, Nomaz and Fazilov, Shavkat},
  title   = {{MERIT}: Interpretable Module-Level Rule Ensembles for Small High-Dimensional Data},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT
