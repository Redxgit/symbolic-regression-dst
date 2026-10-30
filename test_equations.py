import os
import argparse
import logging
import pandas as pd
import numpy as np
import sympy as sp
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import metrics

# Internal module imports
import storm_dates
from evaluation_engine import EquationModel, simulate_storm
from train_script import load_and_preprocess, compute_features


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Symbolic Regression Equations on Test Storms"
    )
    parser.add_argument(
        "--equations_in",
        type=str,
        required=True,
        help="Path to the CSV with discovered equations.",
    )
    parser.add_argument(
        "--equations_out",
        type=str,
        required=True,
        help="Path to save the evaluated results.",
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Comma-separated list of features used (e.g., P_dyn,VBs).",
    )
    parser.add_argument(
        "--mode", type=str, choices=["template", "default"], default="template"
    )
    
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Setup Features and Data
    feature_list = args.features.split(",")
    raw_data = load_and_preprocess()
    data = compute_features(raw_data)

    # 2. Load Equations
    eq_df = pd.read_csv(args.equations_in)
    logging.info(f"Loaded {len(eq_df)} equations from {args.equations_in}")

    # Prepare columns for metrics
    metrics_to_eval = ["RMSE", "MAE", "R2", "BFE"]

    for m in metrics_to_eval:
        eq_df[f"TRAIN_{m}"] = np.nan

    for m in metrics_to_eval:
        eq_df[f"TEST_{m}"] = np.nan

    # 3. Evaluation Loop
    incorrect_count = 0
    correct_count = 0

    test_storms = storm_dates.TEST_STORMS_SYMBOLIC_REGRESSION

    for idx, row in tqdm(
        eq_df.iterrows(), total=len(eq_df), desc="Evaluating Equations"
    ):
        # Initialize the equation model
        # We use 'julia_expression' if template, or 'equation' if default
        raw_eq = row["julia_expression"] if args.mode == "template" else row["equation"]

        logging.info(
            f"Evaluating Equation {idx}: {raw_eq} with features {feature_list}"
        )

        model = EquationModel(
            raw_eq, feature_list, is_template=(args.mode == "template")
        )

        storm_results = []

        # Evaluate across all test storms
        for start, end, storm_id in test_storms:
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)

            integration_index = pd.date_range(
                start - pd.Timedelta(hours=1),
                end + pd.Timedelta(hours=1),
                freq="h",
            )

            storm_df = data.loc[integration_index[0]:integration_index[-1]].copy()

            # Run iterative integration (Euler)

            try:
                preds_func = simulate_storm(model, storm_df)
                comparison = pd.concat(
                    [
                        storm_df["DST"].rename("observed"),
                        preds_func["DST_pred"].rename("predicted"),
                    ],
                    axis=1,
                ).loc[start:end]
                # Store results for this storm
                storm_results.append(
                    {
                        "rmse": root_mean_squared_error(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                        "mae": mean_absolute_error(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                        "r2": r2_score(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                        "cc": pearsonr(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        )[0],
                        "bfe": metrics.calculate_BFE(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                    }
                )

            except Exception as e:
                logging.info(
                    f"Error simulating storm {storm_id} for equation {idx}: {e}"
                )
                storm_results.append(
                    {
                        "rmse": 9999999,
                        "mae": 999999,
                        "r2": -999999,
                        "cc": -999999,
                        "bfe": 999999,
                    }
                )
                incorrect_count += 1
                break

        else:
            correct_count += 1

        # 4. Aggregate Metrics for the Equation
        if storm_results:
            eq_df.at[idx, "TEST_RMSE"] = np.mean([s["rmse"] for s in storm_results])
            eq_df.at[idx, "TEST_MAE"] = np.mean([s["mae"] for s in storm_results])
            eq_df.at[idx, "TEST_R2"] = np.mean([s["r2"] for s in storm_results])
            eq_df.at[idx, "TEST_CC"] = np.mean([s["cc"] for s in storm_results])
            eq_df.at[idx, "TEST_BFE"] = np.mean([s["bfe"] for s in storm_results])

        logging.info(
            f"Metrics for Equation {idx}: RMSE={eq_df.at[idx, 'TEST_RMSE']}, MAE={eq_df.at[idx, 'TEST_MAE']}, R2={eq_df.at[idx, 'TEST_R2']}, CC={eq_df.at[idx, 'TEST_CC']}, BFE={eq_df.at[idx, 'TEST_BFE']}"
        )

    # 5. Finalize and Save
    logging.info(
        f"Evaluation complete. Errors: {incorrect_count}, Correct: {correct_count}"
    )

    # 3. Evaluation Loop
    incorrect_count = 0
    correct_count = 0

    train_storms = storm_dates.TRAIN_STORMS_SYMBOLIC_REGRESSION

    for idx, row in tqdm(
        eq_df.iterrows(), total=len(eq_df), desc="Evaluating Equations"
    ):
        # Initialize the equation model
        # We use 'julia_expression' if template, or 'equation' if default
        raw_eq = row["julia_expression"] if args.mode == "template" else row["equation"]

        logging.info(
            f"Evaluating Equation {idx}: {raw_eq} with features {feature_list}"
        )

        model = EquationModel(
            raw_eq, feature_list, is_template=(args.mode == "template")
        )

        storm_results = []

        # Evaluate across all test storms
        for start, end, storm_id in train_storms:
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)

            integration_index = pd.date_range(
                start - pd.Timedelta(hours=1),
                end + pd.Timedelta(hours=1),
                freq="h",
            )
            scoring_index = pd.date_range(start, end, freq="h")

            storm_df = data.loc[integration_index[0]:integration_index[-1]].copy()

            # Run iterative integration (Euler)

            try:
                preds_func = simulate_storm(model, storm_df)
                comparison = pd.concat(
                    [
                        storm_df["DST"].rename("observed"),
                        preds_func["DST_pred"].rename("predicted"),
                    ],
                    axis=1,
                ).loc[start:end]

                # Store results for this storm
                storm_results.append(
                    {
                        "rmse": root_mean_squared_error(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                        "mae": mean_absolute_error(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                        "r2": r2_score(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                        "cc": pearsonr(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        )[0],
                        "bfe": metrics.calculate_BFE(
                            comparison["observed"].values,
                            comparison["predicted"].values,
                        ),
                    }
                )

            except Exception as e:
                logging.info(
                    f"Error simulating storm {storm_id} for equation {idx}: {e}"
                )
                storm_results.append(
                    {
                        "rmse": 9999999,
                        "mae": 999999,
                        "r2": -999999,
                        "cc": -999999,
                        "bfe": 999999,
                    }
                )
                incorrect_count += 1
                break

        else:
            correct_count += 1

        # 4. Aggregate Metrics for the Equation
        if storm_results:
            eq_df.at[idx, "TRAIN_RMSE"] = np.mean([s["rmse"] for s in storm_results])
            eq_df.at[idx, "TRAIN_MAE"] = np.mean([s["mae"] for s in storm_results])
            eq_df.at[idx, "TRAIN_R2"] = np.mean([s["r2"] for s in storm_results])
            eq_df.at[idx, "TRAIN_CC"] = np.mean([s["cc"] for s in storm_results])
            eq_df.at[idx, "TRAIN_BFE"] = np.mean([s["bfe"] for s in storm_results])

        logging.info(
            f"Metrics for Equation {idx}: RMSE={eq_df.at[idx, 'TRAIN_RMSE']}, MAE={eq_df.at[idx, 'TRAIN_MAE']}, R2={eq_df.at[idx, 'TRAIN_R2']}, CC={eq_df.at[idx, 'TRAIN_CC']}, BFE={eq_df.at[idx, 'TRAIN_BFE']}"
        )

    # 5. Finalize and Save
    logging.info(
        f"Evaluation complete. Errors: {incorrect_count}, Correct: {correct_count}"
    )

    eq_df = eq_df.sort_values(by="TEST_RMSE", ascending=True)
    eq_df.to_csv(args.equations_out, index=False)
    logging.info(f"Results saved to {args.equations_out}")


if __name__ == "__main__":
    main()
