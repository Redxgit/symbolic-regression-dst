import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import metrics

def ddm1_prediction(df):
    """
    Make Dst prediction using Euler integration of the DDM#1 equation 
    from Pennati et al. (2026). https://doi.org/10.1016/j.jocs.2026.102821
    """
    # Find initial conditions
    initial_idx = df.index[0]
    # In ref2, the model is integrated using stepwise 1-hour forecasts [cite: 4726]
    DST_current = df.loc[initial_idx, 'DST']  

    predictions = []

    for hour in range(len(df)):
        current_data = df.iloc[hour]
        
        # Mapping variables from Pennati et al. (2026)
        Ey = current_data['E_Field']  # Convective Electric Field (mV/m)
        Pdyn = current_data['P_dyn']  # Dynamic Pressure (nPa)
        Dst = DST_current             # Current state of Dst (nT)

        # The DDM#1 Equation from Table 2:
        # dDst/dt = 0.3756 - 0.4470 * (0.7071 * Pdyn)**0.25 * #           (0.3183 * Pdyn + max(0.1972 * Dst + 4.4488 * Ey, 0.0903 * Dst))
        
        # Breaking it down for clarity
        fourth_root_term = np.power(0.7071 * Pdyn, 0.25)
        max_term = np.maximum(0.1972 * Dst + 4.4488 * Ey, 0.0903 * Dst)
        
        dDst_dt = 0.3756 - 0.4470 * fourth_root_term * (0.3183 * Pdyn + max_term)

        # Euler Integration: Dst(t+1) = Dst(t) + (dDst/dt * delta_t) 
        # delta_t is 1 hour in this dataset [cite: 4543, 4726]
        DST_pred = DST_current + dDst_dt
        
        predictions.append({
            'datetime': initial_idx + pd.Timedelta(hours=hour+1),
            'DST_pred': DST_pred,
            'dDst': dDst_dt
        })

        # Update state variable for the next iteration [cite: 4726]
        DST_current = DST_pred

    return pd.DataFrame(predictions).set_index('datetime')

def ddm2_prediction(df):
    """
    Make Dst prediction using Euler integration of the DDM#2 equation (C:13)
    from Pennati et al. (2026). DOI: 10.1016/j.jocs.2026.102821
    """
    # Find initial conditions based on the first observed Dst value 
    initial_idx = df.index[0]
    DST_current = df.loc[initial_idx, 'DST']  

    predictions = []

    for hour in range(len(df)):
        current_data = df.iloc[hour]
        
        # Mapping variables: Ey (Electric Field) and Pdyn (Dynamic Pressure) [cite: 154, 157, 330]
        Ey = current_data['E_Field'] 
        Pdyn = current_data['P_dyn'] 
        Dst = DST_current            

        # DDM#2 Equation from Table 2[cite: 387]:
        # dDst/dt = 1.4098 - 0.4411 * max(5.6954*Ey + 0.2270*Dst + 1.1495*Pdyn, 3.0990 + 0.1035*Dst)
        
        term_a = 5.6954 * Ey + 0.2270 * Dst + 1.1495 * Pdyn
        term_b = 3.0990 + 0.1035 * Dst
        
        dDst_dt = 1.4098 - 0.4411 * np.maximum(term_a, term_b)

        # Euler Integration: Dst(t+1) = Dst(t) + (dDst/dt * delta_t) 
        # The resolution/delta_t is 1 hour [cite: 147, 329, 330]
        DST_pred = DST_current + dDst_dt
        
        predictions.append({
            'datetime': initial_idx + pd.Timedelta(hours=hour+1),
            'DST_pred': DST_pred,
            'dDst': dDst_dt
        })

        # Update the state variable for iterative forecasting 
        DST_current = DST_pred

    return pd.DataFrame(predictions).set_index('datetime')

def ddm3_prediction(df):
    """
    Make Dst prediction using Euler integration of the DDM#3 equation (C:12)
    from Pennati et al. (2026). DOI: 10.1016/j.jocs.2026.102821
    """
    # Initialize from observed Dst 
    initial_idx = df.index[0]
    DST_current = df.loc[initial_idx, 'DST']  

    predictions = []

    for hour in range(len(df)):
        current_data = df.iloc[hour]
        
        Ey = current_data['E_Field']
        Pdyn = current_data['P_dyn']
        Dst = DST_current

        # DDM#3 Equation from Table 2[cite: 388]:
        # dDst/dt = 0.2313 - 0.3481 * ((0.3183 * Pdyn)**2 + 0.2531 * Dst + max(5.9693 * Ey, -0.1233 * Dst))
        
        pdyn_squared_term = np.square(0.3183 * Pdyn)
        max_term = np.maximum(5.9693 * Ey, -0.1233 * Dst)
        
        dDst_dt = 0.2313 - 0.3481 * (pdyn_squared_term + 0.2531 * Dst + max_term)

        # Iterative Euler integration with a 1-hour time step 
        DST_pred = DST_current + dDst_dt
        
        predictions.append({
            'datetime': initial_idx + pd.Timedelta(hours=hour+1),
            'DST_pred': DST_pred,
            'dDst': dDst_dt
        })

        # Stepwise concatenation of the forecast 
        DST_current = DST_pred

    return pd.DataFrame(predictions).set_index('datetime')

def burton_prediction(df):
    """
    Make DST* prediction using Euler integration of Burton equation
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
    Make Dst* prediction using Euler integration of Burton equation
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

def plot_evaluation_bfe_multi(ax, obs, preds_list, labels, colors, bin_width=10, fontsize=8):
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
    ax.set_xlabel("Observed DST (nT)", fontsize=fontsize)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=fontsize)
    ax.set_title('BFE Comparison', fontsize=18)
    ax.set_xlim(obs.min() + bin_width, obs.max() - bin_width)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    twin = ax.twinx()
    twin.hist(obs, bins=bins, alpha=0.15, color="brown")
    twin.set_yscale("log")
    twin.grid(False)
    twin.set_ylabel("Bin count", fontsize=fontsize)