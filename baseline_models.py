import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import metrics
from typing_extensions import deprecated

def ddm1_prediction(df):
    """
    Make Dst prediction using Euler integration of the DDM#1 equation 
    from Pennati et al. (2026). https://doi.org/10.1016/j.jocs.2026.102821
    """
    dst_current = df["DST"].iloc[0]
    predictions = [{"datetime": df.index[0], "DST_pred": dst_current, "dDst": 0.0}]

    for i in range(len(df) - 1):
        current_data = df.iloc[i]
        Ey, Pdyn = current_data['E_Field'], current_data['P_dyn']
        
        fourth_root = np.power(0.7071 * Pdyn, 0.25)
        max_term = np.maximum(0.1972 * dst_current + 4.4488 * Ey, 0.0903 * dst_current)
        dDst_dt = 0.3756 - 0.4470 * fourth_root * (0.3183 * Pdyn + max_term)

        dst_current += dDst_dt
        predictions.append({
            "datetime": df.index[i+1],
            "DST_pred": dst_current,
            "dDst": dDst_dt
        })
    return pd.DataFrame(predictions).set_index("datetime")

def ddm2_prediction(df):
    """
    Make Dst prediction using Euler integration of the DDM#2 equation (C:13)
    from Pennati et al. (2026). DOI: 10.1016/j.jocs.2026.102821
    """
    dst_current = df["DST"].iloc[0]
    predictions = [{"datetime": df.index[0], "DST_pred": dst_current, "dDst": 0.0}]

    for i in range(len(df) - 1):
        current_data = df.iloc[i]
        Ey, Pdyn = current_data['E_Field'], current_data['P_dyn']
        
        term_a = 5.6954 * Ey + 0.2270 * dst_current + 1.1495 * Pdyn
        term_b = 3.0990 + 0.1035 * dst_current
        dDst_dt = 1.4098 - 0.4411 * np.maximum(term_a, term_b)

        dst_current += dDst_dt
        predictions.append({
            "datetime": df.index[i+1],
            "DST_pred": dst_current,
            "dDst": dDst_dt
        })
    return pd.DataFrame(predictions).set_index("datetime")

def ddm3_prediction(df):
    """
    Make Dst prediction using Euler integration of the DDM#3 equation (C:12)
    from Pennati et al. (2026). DOI: 10.1016/j.jocs.2026.102821
    """
    dst_current = df["DST"].iloc[0]
    predictions = [{"datetime": df.index[0], "DST_pred": dst_current, "dDst": 0.0}]

    for i in range(len(df) - 1):
        current_data = df.iloc[i]
        Ey, Pdyn = current_data['E_Field'], current_data['P_dyn']
        
        pdyn_sq = np.square(0.3183 * Pdyn)
        max_term = np.maximum(5.9693 * Ey, -0.1233 * dst_current)
        dDst_dt = 0.2313 - 0.3481 * (pdyn_sq + 0.2531 * dst_current + max_term)

        dst_current += dDst_dt
        predictions.append({
            "datetime": df.index[i+1],
            "DST_pred": dst_current,
            "dDst": dDst_dt
        })
    return pd.DataFrame(predictions).set_index("datetime")

def burton_prediction(df):
    """
    Correct Euler integration of the Burton (1975) Dst model.
    Uses the nPa pressure convention (b = 15.80, c = 20.0).
    Integrates Dst* directly and converts to ground Dst using t+1 pressure.
    """
    # Burton constants converted for nPa
    b = 15.80
    c = 20.0
    a = 0.1296
    
    # 1. Initialize Dst* (Burton's Dst0)
    initial_dst = df["DST"].iloc[0]
    initial_p = df["P_dyn_nPa"].iloc[0]
    dst_star_current = initial_dst - b * np.sqrt(initial_p) + c
    
    predictions = [{
        "datetime": df.index[0],
        "DST_pred": initial_dst,
        "dDST": 0.0
    }]
    
    for i in range(len(df) - 1):
        current_data = df.iloc[i]
        
        # Injection Q (F(E) in Burton 1975)
        e_field = current_data["E_Field"]
        if e_field >= 0.5:
            q = -5.4 * (e_field - 0.5)
        else:
            q = 0.0
            
        # 3. Integrate Dst* directly: dDst*/dt = Q - a * Dst*
        d_dst_star_dt = q - (a * dst_star_current)
        dst_star_current += d_dst_star_dt
        
        # 4. Convert to ground Dst using pressure at t+1
        current_p = current_data["P_dyn_nPa"]
        dst_pred_next = dst_star_current + b * np.sqrt(current_p) - c
        
        predictions.append({
            "datetime": df.index[i+1],
            "DST_pred": dst_pred_next,
            "dDST": dst_pred_next - predictions[-1]["DST_pred"]
        })
        
    return pd.DataFrame(predictions).set_index("datetime")

@deprecated("Use burton_prediction instead.")
def burton_prediction_pennati(df):
    """
    Make DST* prediction using Euler integration of Burton equation
    Following Pennati et al. (2026) paper https://doi.org/10.1016/j.jocs.2026.102821.
    """
    # Find initial conditions
    initial_idx = df.index[0]
    DST_star_current = df.loc[initial_idx, f"DST"]  # Start with actual Dst

    predictions = []

    for hour in range(len(df)):
        if hour >= len(df):
            break

        # Get CURRENT solar wind parameters (not initial)
        current_data = df.iloc[hour]
        if current_data[f"E_Field"] >= 0.5:
            e_field_contrib = -5.4 * (current_data[f"E_Field"] - 0.5)
        else:
            e_field_contrib = 0
        dDST = (
            -0.13 * (DST_star_current - 0.2 * np.sqrt(current_data[f"P_dyn"]) + 20)
            + e_field_contrib
        )
        DST_pred = DST_star_current + dDST

        predictions.append(
            {
                "datetime": initial_idx + pd.Timedelta(hours=hour + 1),
                "DST_pred": DST_pred,
                "dDST": dDST,
            }
        )

        DST_star_current = DST_pred  # Update for next iteration

    return pd.DataFrame(predictions).set_index("datetime")

def obm_prediction(df):
    """
    Correct Euler integration of the O'Brien & McPherron (2000) Dst model.
    Uses continuous tau and exact OBM published constants.
    Integrates Dst* directly and converts to ground Dst using t+1 pressure.
    """
    b = 7.26
    c = 11.0

    initial_dst = df["DST"].iloc[0]
    initial_p = df["P_dyn_nPa"].iloc[0]

    # 1. Initialize Dst* 
    dst_star_current = (
        initial_dst
        - b * np.sqrt(initial_p)
        + c
    )

    predictions = [{
        "datetime": df.index[0],
        "DST_pred": initial_dst,
        "dDST": 0.0
    }]

    # 2. Loop from 0 to N-2
    for i in range(len(df) - 1):
        current_data = df.iloc[i]
        next_data = df.iloc[i + 1]

        # Rectified VBs (E_Field is in mV/m in your pipeline)
        e_field = max(0.0, current_data["E_Field"])

        # Injection Q
        if e_field >= 0.49:
            q = -4.4 * (e_field - 0.49)
        else:
            q = 0.0

        # Published continuous OBM decay time
        tau = 2.40 * np.exp(9.74 / (4.69 + e_field))

        # Evolution of Dst*
        d_dst_star_dt = q - (dst_star_current / tau)
        dst_star_current += d_dst_star_dt

        # Convert Dst* -> Dst using pressure at t
        current_p = current_data["P_dyn_nPa"]
        dst_pred_next = dst_star_current + b * np.sqrt(current_p) - c

        predictions.append({
            "datetime": df.index[i + 1],
            "DST_pred": dst_pred_next,
            "dDST": dst_pred_next - predictions[-1]["DST_pred"]
        })

    return pd.DataFrame(predictions).set_index("datetime")

@deprecated("Use obm_prediction instead.")
def obm_prediction_pennati(df):
    """
    Make Dst* prediction using Euler integration of Burton equation
    Following Pennati et al. (2026) paper https://doi.org/10.1016/j.jocs.2026.102821.
    """
    # Find initial conditions
    initial_idx = df.index[0]
    DST_star_current = df.loc[initial_idx, f"DST"]  # Start with actual Dst

    predictions = []

    for hour in range(len(df)):
        if hour >= len(df):
            break

        current_data = df.iloc[hour]
        if current_data["E_Field"] >= 0.5:
            e_field_contrib = -5.4 * (current_data["E_Field"] - 0.5)
            tau = 3.5
        else:
            e_field_contrib = 0
            tau = 7.7

        dDST = (
            -(1 / tau) * (DST_star_current - 0.2 * np.sqrt(current_data["P_dyn"]) + 20)
            + e_field_contrib
        )

        DST_pred = DST_star_current + dDST

        predictions.append(
            {
                "datetime": initial_idx + pd.Timedelta(hours=hour + 1),
                "DST_pred": DST_pred,
                "dDST": dDST,
            }
        )

        DST_star_current = DST_pred  # Update for next iteration

    return pd.DataFrame(predictions).set_index("datetime")


# --- 2. METRIC HELPERS ---


def get_all_metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cc, _ = pearsonr(y_true, y_pred)
    bfe = metrics.calculate_BFE(y_true, y_pred)
    return rmse, mae, r2, cc, bfe

def get_all_metrics_dict(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cc, _ = pearsonr(y_true, y_pred)
    bfe = metrics.calculate_BFE(y_true, y_pred)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "CC": cc,
        "BFE": bfe
    }

def plot_evaluation_bfe_multi(ax, obs, preds_list, labels, colors, bin_width=10, fontsize=8, plot_legend = False):
    """Modified BFE plot supporting multiple model lines."""
    min_val = np.floor(obs.min() / bin_width) * bin_width
    max_val = np.ceil(obs.max() / bin_width) * bin_width
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    title_str = "BFE Comparison:\n"

    for pred, label, color in zip(preds_list, labels, colors):
        diff_abs = np.abs(obs - pred)
        df_temp = pd.DataFrame({"obs": obs, "diff_abs": diff_abs})
        df_temp["bins"] = pd.cut(df_temp["obs"], bins=bins)
        mean_diff = df_temp.groupby("bins", observed=True)["diff_abs"].mean()
        mid_points = [b.mid for b in mean_diff.index]

        ax.plot(mid_points, mean_diff.values, label=label, color=color, linewidth=2)
        ax.fill_between(mid_points, mean_diff.values, 0, alpha=0.1, color=color)

        bfe_val = metrics.calculate_BFE(obs, pred)
        title_str += f"{label} BFE: {bfe_val:.2f} | "

    
    ax.set_ylabel("Mean Abs. Diff (nT)", fontsize=fontsize)
    ax.set_xlabel("Observed Dst (nT)", fontsize=fontsize)
    ax.grid(True, linestyle="--", alpha=0.6)
    if plot_legend:
        ax.legend(fontsize=fontsize)
    else:
        ax.legend().set_visible(False)
    
    ax.set_title('BFE Comparison', fontsize=24)
    ax.set_xlim(obs.min() + bin_width, obs.max() - bin_width)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    twin = ax.twinx()
    twin.hist(obs, bins=bins, alpha=0.15, color="brown")
    twin.set_yscale("log")
    twin.grid(False)
    twin.set_ylabel("Bin count", fontsize=fontsize)