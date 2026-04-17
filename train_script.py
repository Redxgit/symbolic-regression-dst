import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pysr import PySRRegressor, TemplateExpressionSpec
import sympy.parsing.sympy_parser as spp

# Internal module imports (assumed to be in the same directory)
import utils
import storm_dates
import pre_processing


def setup_pysr_patches():
    """Patches to avoid SymPy parsing errors during large-scale runs."""
    spp.parse_expr = lambda code, **kwargs: code
    import pysr.export as export_module

    export_module.add_export_formats = (
        lambda equations, search_output, **kwargs: equations
    )
    import pysr.export_sympy as export_sympy

    export_sympy.pysr2sympy = lambda equation: equation


def get_args():
    parser = argparse.ArgumentParser(
        description="Symbolic Regression for Dst Index Prediction"
    )

    # Core logic toggles
    parser.add_argument(
        "--mode",
        type=str,
        choices=["template", "default"],
        default="template",
        help="Use TemplateExpressionSpec (g + d) or standard search.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="Bzsouth,Bmag,By,Vp,Np,DST",
        help="Comma-separated list of solar wind features to use.",
    )    

    # Simulation hyperparameters
    parser.add_argument("--sims", type=int, default=100, help="Number of simulations.")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Seconds per simulation."
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--maxsize", type=int, default=30)

    # Output
    parser.add_argument(
        "--name", type=str, default="run", help="Prefix for log and result files."
    )

    return parser.parse_args()


def load_and_preprocess():
    logging.info("Loading and interpolating data...")

    ace_columns = ['Bmag', 'Bx', 'By', 'Bz','Vp', 'Np', 'T']

    dst_kyoto = utils.read_iaga_file('./data/dst-kyoto.txt', columns = ["DATE", "TIME", "DOY", "DST"])

    ace_imf = utils.read_data('./data/all_timeline', pattern_to_read=['csv', 'ace_imf_1h_'], print_info = True, return_separated = False)
    ace_imf.columns = ['Bmag', 'Bx', 'By', 'Bz']
    ace_imf_provisional = utils.read_data('./data/all_timeline', pattern_to_read=['csv', 'ace_imf_provisional_1h_'], print_info = True, return_separated = False)
    ace_imf_provisional = pre_processing.preprocess_ace_imf_provisional(ace_imf_provisional, resample=False)
    ace_imf_provisional.columns = ['Bmag', 'Bx', 'By', 'Bz']

    ace_imf = ace_imf.combine_first(ace_imf_provisional)

    ace_swepam = utils.read_data('./data/all_timeline', pattern_to_read=['csv', 'ace_swepam_1h_'], print_info = True, return_separated = False)                          
    ace_swepam.columns = ['Vx', 'Vy', 'Vz', 'Vp', 'Np', 'T']
    ace_swepam = ace_swepam.loc[:, ('Vp', 'Np', 'T')]
    ace_swepam_provisional = utils.read_data('./data/all_timeline', pattern_to_read=['csv', 'ace_swepam_provisional_1h_'], print_info = True, return_separated = False)                          
    ace_swepam_provisional = pre_processing.preprocess_ace_swepam_provisional(ace_swepam_provisional, resample=False)
    ace_swepam_provisional.columns = ['Np', 'Vp', 'T']

    ace_swepam = ace_swepam.combine_first(ace_swepam_provisional)

    ace_data = ace_imf.join(ace_swepam)

    all_data = ace_data.join(dst_kyoto['DST'])

    all_data = all_data.interpolate()
    return all_data


def compute_features(dfx):
    """Derived feature logic condensed from your deriv scripts."""
    df = dfx.copy()

    # Derived variables
    df["B_T"] = np.sqrt(df["By"] ** 2 + df["Bz"] ** 2)
    df["clock_angle"] = np.degrees(np.arctan2(df["By"], df["Bz"]))
    df["sin_th2"] = np.sin(np.radians(df["clock_angle"]) / 2)
    df["Bzsouth"] = np.maximum(0, -df["Bz"])
    df["VBs"] = df["Vp"] * df["Bzsouth"]
    df["P_dyn"] = 1.6726e-6 * df["Np"] * (df["Vp"] ** 2)
    df["epsilon"] = df["Vp"] * (df["B_T"] ** 2) * (df["sin_th2"] ** 4)
    df["E_Field"] = -df["Vp"] * df["Bz"] * 1e-3
    df["VBs_thresh"] = np.maximum(df["VBs"] - 0.5, 0)
    # Target Derivative
    df["dDST_dt"] = df["DST"].diff().shift(-1)

    return df


def main():
    args = get_args()
    setup_pysr_patches()

    # Logging Setup
    os.makedirs("train_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"train_logs/{args.name}_{timestamp}.log"
    results_file = f"equations_{args.name}.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    # Feature selection
    all_cols = args.features.split(",")
    

    # Data pipeline
    raw_data = load_and_preprocess()
    data = compute_features(raw_data)

    # Storm extraction (Combine train and validation as per your scripts)
    train_data_list, train_labels_list = [], []
    for storm_set in [storm_dates.TRAIN_STORMS_SYMBOLIC_REGRESSION]:
        for start, end, idx in storm_set:
            storm_slice = data[start:end]
            if not storm_slice[all_cols + ["dDST_dt"]].isna().any().any():
                train_data_list.append(storm_slice[all_cols].reset_index(drop=True))
                train_labels_list.append(
                    storm_slice[["dDST_dt"]].reset_index(drop=True)
                )

    X = pd.concat(train_data_list).values
    y = pd.concat(train_labels_list).values

    logging.info(f"Training on {X.shape[0]} samples with features: {all_cols}")

    # PySR Configuration
    template = None
    if args.mode == "template":
        template = TemplateExpressionSpec(
            expressions=["g", "d"],
            variable_names=all_cols,
            combine=f"g({', '.join(all_cols[:-1])}) + d({all_cols[-1]})",
        )

    for sim in range(1, args.sims + 1):
        logging.info(f"\n--- Starting Simulation {sim}/{args.sims} ---")

        model = PySRRegressor(
            expression_spec=template,
            niterations=args.iterations,
            variable_names=all_cols,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["neg", "sqrt", "square", "inv"],
            elementwise_loss="L1DistLoss()",
            maxsize=args.maxsize,
            batch_size=args.batch_size,
            batching=True,
            timeout_in_seconds=args.timeout,
            constraints={"d": 9} if args.mode == "template" else None,
            progress=True,
        )

        model.fit(X, y)

        # Save results
        eq_df = model.equations_.copy()
        eq_df["simulation"] = sim
        eq_df = eq_df[eq_df["complexity"] > 5]  # Filter noise

        # Persistent storage across simulations
        if os.path.exists(results_file):
            global_res = pd.read_csv(results_file)
            global_res = pd.concat([global_res, eq_df]).drop_duplicates(
                subset=["equation"]
            )
        else:
            global_res = eq_df

        global_res.sort_values("loss").to_csv(results_file, index=False)


if __name__ == "__main__":
    main()
