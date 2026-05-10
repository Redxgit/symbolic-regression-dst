# Symbolic Regression for the Terrestrial Ring Current

This repository contains the code accompanying the paper: **"The terrestrial ring current: solar wind drivers and decay."** It uses Symbolic Regression (SR) via the `PySR` library to discover analytical expressions for the evolution of the $Dst$ index based on solar wind and interplanetary magnetic field drivers.

---

## 📂 Repository Structure

* **`train_script.py`**: The entry point for equation discovery using `PySRRegressor`.
* **`test_equations.py`**: Evaluates discovered equations on test storms using iterative Euler integration.
* **`plot_best_equations.py`**: Ranks equations by performance, generates time-series plots, and compares results against baseline models.
* **`evaluation_engine.py`**: The core logic for parsing SymPy expressions and performing storm simulations.
* **`baseline_models.py`**: Implementations of classic and modern models including **Burton (1975)**, **OBM (2000)**, and **DDM#1-3** (from https://doi.org/10.1016/j.jocs.2026.102821).
* **`pyproject.toml`**: Project metadata and dependency specifications.
* **`plot-storm-and-get-metrics.ipynb`**: A Jupyter notebook for visualizing storm simulations and calculating performance metrics for a particular equation.
---

## 🚀 Execution Flow

### 1. Data Preparation
The scripts expect a `./data/` folder containing ACE solar wind data (IMF and plasma parameters) and $Dst$ index timelines. The `load_and_preprocess` function handles the merging and interpolation of these datasets.

### 2. Equation Discovery
Use `train_script.py` to start the symbolic search. You can choose between a standard search or a structured "template" mode.
Example

```bash
python train_script.py --mode template --features Vp,Np,Bzsouth,Bmag,By,DST --sims 100 --name my_run --mode template
```

### 3. Testing and Evaluation

Once equations are saved to a CSV, evaluate their performance on the test storm set:
```bash
python test_equations.py --equations_in equations_my_run.csv --equations_out results.csv --features Vp,Np,Bzsouth,Bmag,By,DST --mode template
```

### 4. Visualization

Compare your top discovered equations against physical baselines:
```bash
python plot_best_equations.py --csv results.csv --features Vp,Np,Bzsouth,Bmag,By,DST --output_dir ./plots/ --topk 2 --mode template
```
---

### 5. Extra notebooks

The `plot-storm-and-get-metrics.ipynb` notebook allows you to generate the plots and csvs of the time-series predictions of a particular equation on the test storm and compute performance metrics: RMSE, MAE, CC, R2 and BFE and compare against the baselines. You can specify the equation to test by changing the `RAW_EQ` variable in the notebook, also adjust the `FEATURES` variable to match the features used in the equation and `MODE` for default or template. 

The notebooks: `plot-storm-and-get-metrics-no-baselines-default-deriv.ipynb`, `plot-storm-and-get-metrics-no-baselines-default-primitive.ipynb`, `plot-storm-and-get-metrics-no-baselines-template-deriv.ipynb`, `plot-storm-and-get-metrics-no-baselines-template-primitive.ipynb` and `plot-storm-and-get-metrics-no-baselines-template-primitive_only1.ipynb` are used to create the plots and metrics of the paper and supporting information. They have the equations for each group and generate the plots and metrics for all of them.

## Technical Details

* **Euler Integration**: Because SR finds the derivative ($dDst/dt$), the models are evaluated by integrating the predicted change over time. The predicted $Dst$ at step $t$ is used as the input for step $t+1$ to test for long-term stability.
* **Feature Engineering**: The scripts automatically compute derived variables like Dynamic Pressure ($P_{dyn}$), E-Field, and the Akasofu epsilon ($\epsilon$) parameter.
* **Template Expressions**: The `template` mode uses `TemplateExpressionSpec` to ensure the discovered model remains physically interpretable by separating the driver-only injection term from the $Dst$-dependent decay term.

---

## 📝 Citation

If you use this code or the discovered equations, please cite:
TBD


