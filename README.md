# Physically Meaningful Modeling of $Dst$ Evolution From Symbolic Regression

This repository contains the code accompanying the paper: **"Physically Meaningful Modeling of $Dst$ Evolution From Symbolic Regression."** It uses Symbolic Regression (SR) via the `PySR` library to discover analytical expressions for the evolution of the $Dst$ index based on solar wind and interplanetary magnetic field drivers.

---

## 📂 Repository structure

| File or directory | Purpose |
|---|---|
| `train_script.py` | Data loading, feature computation, and stochastic equation discovery |
| `test_equations.py` | Evaluation of saved candidate equations |
| `evaluation_engine.py` | Equation parsing and Euler integration |
| `baseline_models.py` | Burton, O'Brien–McPherron, and DDM1–3 implementations |
| `metrics.py` | Evaluation metrics and plotting helpers |
| `pre_processing.py`, `utils.py`, `constants.py` | Data preparation utilities |
| `storm_dates.py` | Discovery and evaluation storm intervals |
| `data/` | Source observations and the prepared-data cache |
| `historical/` | Earlier evaluations and supporting notebooks |
| `outputs/`, `train_logs/` | Local search checkpoints and logs |

The DDM models follow Pennati et al.:
https://doi.org/10.1016/j.jocs.2026.102821.

---

## 🚀 Execution Flow

### 1. Data Preparation

The data loader expects:

- `data/dst-kyoto.txt`: Dst observations.
- `data/all_timeline/`: hourly ACE IMF and plasma files.
- `data/prepared_data.csv`: the distributed prepared-data cache.

`load_and_preprocess()` reads the prepared cache when it exists.
During rebuilding, definitive observations take precedence and
preliminary observations fill missing values. Preliminary IMF vectors
are transformed from GSE to GSM using their timestamps. Duplicate
timestamps are resolved by keeping the first occurrence within each
input series. Missing values are interpolated when interpolation is
enabled.

To rebuild into a separate candidate cache, run this from a notebook
in the repository root:

```python
from train_script import load_and_preprocess

prepared = load_and_preprocess(
    force_reload=True,
    save_prepared_data=False,
    interpolate_missing=True,
    use_preliminary=True,
)

prepared.to_csv(
    "data/prepared_data_candidate.csv",
    index_label="datetime",
)
```

The cache contains prepared
observations; derived features are computed separately by
`compute_features()`.

The purpose of saving the prepared data is to avoid the conversion from GSE to GSM in the preliminary IMF data, which is very slow and not needed for repeated runs. 

### 2. Equation Discovery

Use `train_script.py` to start the symbolic regression search. You can choose between a standard search or a structured "template" mode.
Example

```bash
uv run python train_script.py \
    --mode template \
    --features Vp,Np,Bzsouth,Bmag,By,DST \
    --sims 100 \
    --iterations 5000 \
    --timeout 3600 \
    --batch_size 1024 \
    --maxsize 30 \
    --name new_template_run
```

### 3. Testing and Evaluation

Once equations are saved to a CSV, evaluate their performance on the test storm set:
```bash
uv run python test_equations.py \
    --equations_in equations_new_template_run.csv \
    --equations_out evaluated_new_template_run.csv \
    --features Vp,Np,Bzsouth,Bmag,By,DST \
    --mode template
```

### 4. Visualization

Compare your top discovered equations against physical baselines:
```bash
uv run python plot_best_equations.py \
    --csv evaluated_new_template_run.csv \
    --features Vp,Np,Bzsouth,Bmag,By,DST \
    --output_dir plots/new_template_run \
    --topk 2 \
    --mode template
```
---

## Reproducing the manuscript evaluations

Equation discovery is not required to reproduce the published
reconstructions. The root notebooks contain the selected manuscript
equations.

Open each notebook in a fresh kernel and run its cells in order.
The notebooks can be executed independently.

| Notebook | Manuscript results | Output directory |
|---|---|---|
| `plot-storm-and-get-metrics.ipynb` | Eq. 21 against five baselines; Table 3 comparison | `comparison_template_deriv_against_baselines/` |
| `plot-storm-and-get-metrics-no-baselines-default-primitive.ipynb` | Eqs. 6–8 | `unconstrained_primitive_features/` |
| `plot-storm-and-get-metrics-no-baselines-default-deriv.ipynb` | Eqs. 9–12 | `unconstrained_derived_features/` |
| `plot-storm-and-get-metrics-no-baselines-template-primitive.ipynb` | Eqs. 13–19 | `template_primitive_features/` |
| `plot-storm-and-get-metrics-no-baselines-template-deriv.ipynb` | Eqs. 21–23 | `template_deriv_features/` |
| `plot-storm-and-get-metrics-no-baselines-template-primitive_fig4.ipynb` | Comparison of Eqs. 13 and 21 | `template_primitive_features_fig4/` |

The notebooks generate figures and, where implemented, per-storm
prediction CSVs. The comparison notebook also displays aggregate
metrics, win counts, and LaTeX tables.

`storms-dst.ipynb` provides storm and data-availability diagnostics.
The supporting analysis of 2008–2016 observations is preserved in
`historical/notebooks/storms-2008-2016.ipynb`.

---

### Evaluation convention

Each reconstruction is initialized with the observed Dst one hour
before the scoring interval. Integration uses one-hour Euler steps,
the predicted Dst state, and observed solar-wind inputs.

Metrics are calculated only over the inclusive interval from `start`
to `end`, with the date strings converted to timestamps. The scored
end timestamp is midnight at the start of the stated end date.

The manuscript comparison contains 20 test storms.

The following reference values are means of per-storm metrics,
rounded to two decimals:

| Model | RMSE | MAE | $R^2$ | BFE |
|---|---:|---:|---:|---:|
| Eq. 21 | 13.82 | 10.56 | 0.79 | 16.38 |
| Burton | 26.14 | 18.34 | 0.43 | 29.43 |
| OBM | 16.91 | 13.13 | 0.72 | 18.10 |
| DDM1 | 18.75 | 14.59 | 0.66 | 23.89 |
| DDM2 | 18.20 | 14.04 | 0.69 | 22.84 |
| DDM3 | 18.66 | 14.10 | 0.68 | 24.26 |

Global metrics calculated from concatenated observations are distinct
from these per-storm means.

Storms from 2017 onward were excluded from equation discovery but
were used to evaluate and select candidates.

---

## Technical Details

* **Euler Integration**: Because SR finds the derivative ($dDst/dt$), the models are evaluated by integrating the predicted change over time. The predicted $Dst$ at step $t$ is used as the input for step $t+1$ to test for long-term stability.
* **Feature Engineering**: The scripts automatically compute derived variables like Dynamic Pressure ($P_{dyn}$), E-Field, and the Akasofu epsilon ($\epsilon$) parameter.
* **Template Expressions**: The `template` mode uses `TemplateExpressionSpec` to ensure the discovered model remains physically interpretable by separating the driver-only injection term from the $Dst$-dependent decay term.

## Epsilon experiments

The manuscript equations, including Eqs. 9–12 and 21–23, originate from the earlier experiments archived at commit [e361d443f2e9ec5283702fefe4ef29ee57e0eb3f](https://github.com/Redxgit/symbolic-regression-dst/commit/e361d443f2e9ec5283702fefe4ef29ee57e0eb3f).

Those experiments used an incorrect akasofu epsilon coupling proxy:

    epsilon_original = Vp * (By**2 + Bz**2) * sin(theta/2)**4

The implementation was subsequently corrected to use the full magnetic field magnitude:

    epsilon_corrected = Vp * (Bx**2 + By**2 + Bz**2) * sin(theta/2)**4

The corrected implementation and repeated experiments for both derived-input configurations are archived at commit [bb91587f0d994029f9b0214c8698d73959c77bca](https://github.com/Redxgit/symbolic-regression-dst/commit/bb91587f0d994029f9b0214c8698d73959c77bca).

The corrected-search candidate files are:

- `equations_template-deriv-new-epsilon.csv`
- `equations_unconstrained-new-epsilon.csv`

Their historical evaluated results are:

- `equations-ranked-template-deriv-new-epsilon.csv`
- `equations-ranked-unconstrained-deriv-new-epsilon.csv`

For the corrected template search, the feature order is `epsilon,VBs,P_dyn,DST`. This order must be preserved when interpreting the numbered placeholders in template expressions.

The original and corrected-search artifacts are retained separately to preserve the provenance of the manuscript and subsequent experiments. Historical evaluated CSVs retain their original scores; comparisons under the final manuscript evaluation procedure are recorded separately.

Eq. 21 contains no epsilon term. Correcting the epsilon feature therefore does not change the numerical evaluation of this fixed expression, although it can affect the discovery and selection of candidate equations.

The [`epsilon_verification/`](epsilon_verification/) directory contains the notebook and evaluation results used to inspect $\epsilon$ usage in both the unconstrained and template searches with derived inputs, comparing the original and corrected ε experiments. In both search configurations, $\epsilon$ is absent from the highest-ranked equations for RMSE, MAE, R$^2$, and BFE. 

---

## 📝 Citation

If you use this code or the discovered equations, please cite:
TBD


