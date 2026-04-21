import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sympy.printing import latex

# Internal module imports
import storm_dates
import baseline_models

# from evaluation_engine import UnifiedModel, simulate_storm, compute_features
from evaluation_engine import EquationModel, simulate_storm
from train_script import load_and_preprocess, compute_features




def save_prediction_data(model, start, end, storm_df, output_path):
    """
    Generates and saves a CSV with observed and predicted DST and dDST/dt.
    """
    # 1. Observed Data
    # Real dDST is calculated as the difference to the next hour
    real_dst = storm_df[start:end]["DST"].values
    real_ddst = storm_df[start:end]["DST"].diff().shift(-1).values

    # 2. Equation Predictions
    # We need the iterative predictions for DST
    pred_dst_eq = simulate_storm(model, storm_df)
    if model.is_template:
        pred_dst_eq = pred_dst_eq[start:end][
            ["DST_pred", "dDST", "injection_component", "decay_component"]
        ]
    else:
        pred_dst_eq = pred_dst_eq[start:end][["DST_pred", "dDST"]]
    # 3. Baseline Predictions (Burton & OBM)
    pred_dst_burton = baseline_models.burton_prediction(storm_df)
    pred_dst_burton = pred_dst_burton[start:end][["DST_pred", "dDST"]]
    pred_dst_burton.columns = ["DST_pred_burton", "dDST_burton"]
    pred_dst_burton = pred_dst_burton[start:end][["DST_pred_burton", "dDST_burton"]]
    pred_dst_obm = baseline_models.obm_prediction(storm_df)
    pred_dst_obm = pred_dst_obm[start:end][["DST_pred", "dDST"]]
    pred_dst_obm.columns = ["DST_pred_obm", "dDST_obm"]
    pred_dst_obm = pred_dst_obm[start:end][["DST_pred_obm", "dDST_obm"]]
    pred_dst_ddm1 = baseline_models.ddm1_prediction(storm_df)    
    pred_dst_ddm1.columns = ["DST_pred_ddm1", "dDST_ddm1"]
    pred_dst_ddm1 = pred_dst_ddm1[start:end][["DST_pred_ddm1", "dDST_ddm1"]]
    pred_dst_ddm2 = baseline_models.ddm2_prediction(storm_df)
    pred_dst_ddm2.columns = ["DST_pred_ddm2", "dDST_ddm2"]
    pred_dst_ddm2 = pred_dst_ddm2[start:end][["DST_pred_ddm2", "dDST_ddm2"]]
    pred_dst_ddm3 = baseline_models.ddm3_prediction(storm_df)
    pred_dst_ddm3.columns = ["DST_pred_ddm3", "dDST_ddm3"]
    pred_dst_ddm3 = pred_dst_ddm3[start:end][["DST_pred_ddm3", "dDST_ddm3"]]

    # 4. Construct Comprehensive DataFrame

    if model.is_template:
        results_df = pd.DataFrame(
            {
                "Timestamp": storm_df[start:end].index,
                "Observed_DST": real_dst,
                "Real_dDST_dt": real_ddst,
                "Pred_DST_Equation": pred_dst_eq["DST_pred"].values,
                "Pred_dDST_dt_Equation": pred_dst_eq["dDST"].values,
                "Injection_Component": pred_dst_eq["injection_component"].values,
                "Decay_Component": pred_dst_eq["decay_component"].values,
                "Pred_DST_Burton": pred_dst_burton["DST_pred_burton"].values,
                "Pred_dDST_dt_Burton": pred_dst_burton["dDST_burton"].values,
                "Pred_DST_OBM": pred_dst_obm["DST_pred_obm"].values,
                "Pred_dDST_dt_OBM": pred_dst_obm["dDST_obm"].values,
                "Pred_DST_DDM1": pred_dst_ddm1["DST_pred_ddm1"].values,
                "Pred_dDST_dt_DDM1": pred_dst_ddm1["dDST_ddm1"].values,
                "Pred_DST_DDM2": pred_dst_ddm2["DST_pred_ddm2"].values,
                "Pred_dDST_dt_DDM2": pred_dst_ddm2["dDST_ddm2"].values,
                "Pred_DST_DDM3": pred_dst_ddm3["DST_pred_ddm3"].values,
                "Pred_dDST_dt_DDM3": pred_dst_ddm3["dDST_ddm3"].values,
            }
        ).set_index("Timestamp")
    else:
        results_df = pd.DataFrame(
            {
                "Timestamp": storm_df[start:end].index,
                "Observed_DST": real_dst,
                "Real_dDST_dt": real_ddst,
                "Pred_DST_Equation": pred_dst_eq["DST_pred"].values,
                "Pred_dDST_dt_Equation": pred_dst_eq["dDST"].values,
                "Pred_DST_Burton": pred_dst_burton["DST_pred_burton"].values,
                "Pred_dDST_dt_Burton": pred_dst_burton["dDST_burton"].values,
                "Pred_DST_OBM": pred_dst_obm["DST_pred_obm"].values,
                "Pred_dDST_dt_OBM": pred_dst_obm["dDST_obm"].values,
                "Pred_DST_DDM1": pred_dst_ddm1["DST_pred_ddm1"].values,
                "Pred_dDST_dt_DDM1": pred_dst_ddm1["dDST_ddm1"].values,
                "Pred_DST_DDM2": pred_dst_ddm2["DST_pred_ddm2"].values,
                "Pred_dDST_dt_DDM2": pred_dst_ddm2["dDST_ddm2"].values,
                "Pred_DST_DDM3": pred_dst_ddm3["DST_pred_ddm3"].values,
                "Pred_dDST_dt_DDM3": pred_dst_ddm3["dDST_ddm3"].values,
            }
        ).set_index("Timestamp")

    results_df.to_csv(output_path)
    return results_df


def predict_and_plot_storm(model, start, end, storm_df, storm_id, save_path):
    # 1. Generate Predictions
    y_true = storm_df[start:end]["DST"].values

    res_eq = simulate_storm(model, storm_df)
    res_burton = baseline_models.burton_prediction(storm_df)
    res_obm = baseline_models.obm_prediction(storm_df)
    ddm1 = baseline_models.ddm1_prediction(storm_df)
    ddm2 = baseline_models.ddm2_prediction(storm_df)
    ddm3 = baseline_models.ddm3_prediction(storm_df)

    res_eq = res_eq[start:end]["DST_pred"].values
    res_burton = res_burton[start:end]["DST_pred"].values
    res_obm = res_obm[start:end]["DST_pred"].values
    res_ddm1 = ddm1[start:end]["DST_pred"].values
    res_ddm2 = ddm2[start:end]["DST_pred"].values
    res_ddm3 = ddm3[start:end]["DST_pred"].values

    # 2. Calculate Metrics
    m_eq = baseline_models.get_all_metrics(y_true, res_eq)
    m_burton = baseline_models.get_all_metrics(y_true, res_burton)
    m_obm = baseline_models.get_all_metrics(y_true, res_obm)
    m_ddm1 = baseline_models.get_all_metrics(y_true, res_ddm1)
    m_ddm2 = baseline_models.get_all_metrics(y_true, res_ddm2)
    m_ddm3 = baseline_models.get_all_metrics(y_true, res_ddm3)

    # 3. Setup Figure (3 Columns)
    fig, axs = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig.suptitle(
        rf"Evaluation for Equation: ${model.latex_str()}$", fontsize=16, wrap=True
    )
    # Column 1: Time Series
    axs[0].plot(
        storm_df[start:end].index,
        y_true,
        color="black",
        label="Observed",
        alpha=0.6,
        linewidth=2,
    )
    axs[0].plot(
        storm_df[start:end].index,
        res_eq,
        color="blue",
        linestyle="--",
        label="Equation",
        linewidth=1.5,
    )
    axs[0].plot(
        storm_df[start:end].index,
        res_burton,
        color="yellow",
        linestyle="--",
        label="Burton",
        linewidth=1.5,
    )
    axs[0].plot(
        storm_df[start:end].index,
        res_obm,
        color="green",
        linestyle="--",
        label="OBM",
        linewidth=1.5,
    )
    
    axs[0].plot(
        storm_df[start:end].index,
        res_ddm1,
        color="orange",
        linestyle="--",
        label="DDM1",
        linewidth=1.5,
    )
    axs[0].plot(
        storm_df[start:end].index,
        res_ddm2,
        color="purple",
        linestyle="--",
        label="DDM2",
        linewidth=1.5,
    )
    axs[0].plot(
        storm_df[start:end].index,
        res_ddm3,
        color="cyan",
        linestyle="--",
        label="DDM3",
        linewidth=1.5,
    )
    
    axs[0].set_title(f"Storm {storm_id} Reconstruction")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(start, end)

    # Column 2: Prediction Error
    diff_eq = res_eq - y_true
    diff_burton = res_burton - y_true
    diff_obm = res_obm - y_true
    diff_ddm1 = res_ddm1 - y_true
    diff_ddm2 = res_ddm2 - y_true
    diff_ddm3 = res_ddm3 - y_true


    axs[1].plot(storm_df[start:end].index, diff_eq, color="blue", label="Eq Error")
    axs[1].plot(
        storm_df[start:end].index, diff_burton, color="yellow", label="Burton Error"
    )
    axs[1].plot(storm_df[start:end].index, diff_obm, color="green", label="OBM Error")
    axs[1].plot(storm_df[start:end].index, diff_ddm1, color="orange", label="DDM1 Error")
    axs[1].plot(storm_df[start:end].index, diff_ddm2, color="purple", label="DDM2 Error")
    axs[1].plot(storm_df[start:end].index, diff_ddm3, color="cyan", label="DDM3 Error")
    axs[1].axhline(0, color="black", linestyle="--")

    title_metrics = (
        f"Error Comparison\n"
        f"Eq: RMSE {m_eq[0]:.2f} | MAE {m_eq[1]:.2f} | R2 {m_eq[2]:.2f} | CC {m_eq[3]:.2f} | BFE {m_eq[4]:.2f}\n"
        f"Burton: RMSE {m_burton[0]:.2f} | MAE {m_burton[1]:.2f} | R2 {m_burton[2]:.2f} | CC {m_burton[3]:.2f} | BFE {m_burton[4]:.2f}\n"
        f"OBM: RMSE {m_obm[0]:.2f} | MAE {m_obm[1]:.2f} | R2 {m_obm[2]:.2f} | CC {m_obm[3]:.2f} | BFE {m_obm[4]:.2f}\n"
        f"DDM1: RMSE {m_ddm1[0]:.2f} | MAE {m_ddm1[1]:.2f} | R2 {m_ddm1[2]:.2f} | CC {m_ddm1[3]:.2f} | BFE {m_ddm1[4]:.2f}\n"
        f"DDM2: RMSE {m_ddm2[0]:.2f} | MAE {m_ddm2[1]:.2f} | R2 {m_ddm2[2]:.2f} | CC {m_ddm2[3]:.2f} | BFE {m_ddm2[4]:.2f}\n"
        f"DDM3: RMSE {m_ddm3[0]:.2f} | MAE {m_ddm3[1]:.2f} | R2 {m_ddm3[2]:.2f} | CC {m_ddm3[3]:.2f} | BFE {m_ddm3[4]:.2f}"
    )
    axs[1].set_title(title_metrics, fontsize=9)
    axs[1].set_ylabel("Error (nT)")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(start, end)

    # Column 3: BFE
    baseline_models.plot_evaluation_bfe_multi(
        axs[2],
        y_true,
        [res_eq, res_burton, res_obm, res_ddm1, res_ddm2, res_ddm3],
        ["Equation", "Burton", "OBM", "DDM1", "DDM2", "DDM3"],
        ["blue", "yellow", "green", "orange", "purple", "cyan"],
    )

    plt.savefig(save_path)
    plt.close()


# --- 4. MAIN EXECUTION ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["template", "default"], default="template"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True, default=5)
    parser.add_argument("--raw_eq", type=str, default=None, help="Direct equation input for testing")
    args = parser.parse_args()

    # Create directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    feature_list = args.features.split(",")

    # Load Data (Assuming data logic is in compute_features)
    raw_data = load_and_preprocess()
    data = compute_features(raw_data)

    # If raw_eq is provided, evaluate it directly without loading from CSV
    if args.raw_eq:
        model = EquationModel(args.raw_eq, feature_list, is_template=args.mode == "template")
        test_storms = storm_dates.TEST_STORMS_SR
        for sd, ed, storm_id in tqdm(test_storms):
            start = pd.to_datetime(sd)
            end = pd.to_datetime(ed)
            storm_df = data[
                start - pd.DateOffset(hours=1) : end + pd.DateOffset(hours=1)
            ].copy()
            if storm_df.empty:
                continue

            file_name = f"storm_{storm_id}.png"
            predict_and_plot_storm(
                model,
                start,
                end,
                storm_df,
                storm_id,
                os.path.join(args.output_dir, file_name),
            )

            csv_name = f"data_storm_{storm_id}.csv"
            save_prediction_data(
                model, start, end, storm_df, os.path.join(args.output_dir, csv_name)
            )
        return

    # Load and Rank Equations (Top 5 by TEST_RMSE)
    eq_df = pd.read_csv(args.csv)

    TOPK = args.topk
    metrics_top_k = [("MAE", True), ("RMSE", True), ("R2", False), ("BFE", True)]

    equations = set()

    for metric, asc in metrics_top_k:
        print(f"Top {TOPK} equations for {metric}")
        print(
            eq_df.sort_values(by=f"TEST_{metric}", ascending=asc).head(TOPK)[
                ["equation", f"TEST_{metric}"]
            ]
        )
        for x in (
            eq_df.sort_values(by=f"TEST_{metric}", ascending=asc)
            .head(TOPK)[["equation"]]
            .values
        ):
            equations.add(x[0])

    test_storms = storm_dates.TEST_STORMS_SYMBOLIC_REGRESSION

    print(f"Total unique equations to evaluate: {len(equations)}")

    for i, equation in enumerate(equations):
        eq_folder = os.path.join(args.output_dir, f"top_eq_{i+1}")
        os.makedirs(eq_folder, exist_ok=True)

        # Initialize Model
        # raw_eq = row["julia_expression"] if args.mode == "template" else row["equation"]
        raw_eq = equation
        model = EquationModel(raw_eq, feature_list, is_template=args.mode == "template")
        
        with open(os.path.join(eq_folder, 'equation.txt'), 'w') as f:
            f.write(f'Equation: {model.sympy_format()}\n')            
            f.write(f'LaTeX: {latex(model.latex_str())}\n')
            f.write(f'Raw: {raw_eq}\n')

        print(f"Plotting Top Equation {i+1}: {raw_eq} ...")
        for sd, ed, storm_id in tqdm(test_storms):
            start = pd.to_datetime(sd)
            end = pd.to_datetime(ed)
            storm_df = data[
                start - pd.DateOffset(hours=1) : end + pd.DateOffset(hours=1)
            ].copy()
            if storm_df.empty:
                continue

            file_name = f"storm_{storm_id}.png"
            predict_and_plot_storm(
                model,
                start,
                end,
                storm_df,
                storm_id,
                os.path.join(eq_folder, file_name),
            )

            csv_name = f"data_storm_{storm_id}.csv"
            save_prediction_data(
                model, start, end, storm_df, os.path.join(eq_folder, csv_name)
            )


if __name__ == "__main__":
    main()
