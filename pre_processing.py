from tokenize import group
import pandas as pd
import scipy
import constants
import numpy as np
import utils
from spacepy import coordinates as coord
from spacepy.time import Ticktock

# Physical constants (SI units)
m_p = 1.6726e-27    # Proton mass (kg)
mu_0 = 4.0e-7 * np.pi  # Permeability of free space (H/m)

def preprocess_ace_imf(
    df,
    cols=constants.COLS_SELECTION_ACE_MAG,
    resample = True,
    group_freq="5min",
    group_vars=["mean", "std"],
    group_closed="right",
    group_label="right",
):
    """
    Function to preprocess data from ACE MAG dataset,
    only cols will be used,
    the data will be grouped in group_freq frequencies,
    group_vars will be calculated,
    column names will be flattened using the metric in group_vars

    Regarding the column names, mean will be used as the default,
    for the other metrics [if there are], each column will be named
    as {column_name}_{metric}

    Parameters
    ----------
    df : pandas DataFrame

    cols : list, columns to keep, default: ["Bmag", "Bx", By", "Bz"]

    group_freq: to which frequency should de data be grouped, default: 5 minutes

    group_vars: which metrics should be calculated when grouping, default: mean and std

    Returns
    -------
    preprocessed DataFrame
    """

    # Replace the value they use to represent missing/error values for NaN
    for col in df.columns:
        if df[col].dtype == "float32":
            for v in constants.NAN_VALUES_ACE:
                df[col].replace(v, np.nan, inplace=True)

    # Replace column values which are outside their range
    for col in df.columns:
        if col in constants.VALID_RANGES_VARS_ACE_IMF.keys():
            df.loc[df[col] >= constants.VALID_RANGES_VARS_ACE_IMF[col][1], col] = np.nan
            df.loc[df[col] <= constants.VALID_RANGES_VARS_ACE_IMF[col][0], col] = np.nan

    df = utils.parse_column_names(df)
    df = df[cols]
    if resample:
        df = df.groupby(
            pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
        ).agg(group_vars)
        df.columns = ["_".join(x) for x in df.columns]

        if "mean" in group_vars:
            new_cols = []
            for col in df.columns:
                if col.find("_mean"):
                    new_cols.append(col.split("_" + group_vars[0])[0])
                else:
                    new_cols.append(col)
            df.columns = new_cols

    return df


def preprocess_ace_imf_provisional(
    df,
    cols=constants.COLS_SELECTION_ACE_MAG,
    resample = True,
    group_freq="5min",
    group_vars=["mean"],
    group_closed="right",
    group_label="right",
):
    new_cols = []

    for col in df.columns:
        if col.find("[PRELIM] ") >= 0:
            new_cols.append(col.split("[PRELIM] ")[1])
        else:
            new_cols.append(col)

    df.columns = new_cols
    df = utils.parse_column_names(df)

    for col in df.columns:
        if df[col].dtype == "float32":
            for v in constants.NAN_VALUES_ACE:
                df[col].replace(v, np.nan, inplace=True)

    # Replace column values which are outside their range
    for col in df.columns:
        if col in constants.VALID_RANGES_VARS_ACE_IMF.keys():
            df.loc[df[col] >= constants.VALID_RANGES_VARS_ACE_IMF[col][1], col] = np.nan
            df.loc[df[col] <= constants.VALID_RANGES_VARS_ACE_IMF[col][0], col] = np.nan

    if ("BGSEc_X") in df.columns:
        gse_coords = coord.Coords(
            df[["BGSEc_X", "BGSEc_Y", "BGSEc_Z"]].values, "GSE", "car", use_irbem=False
        )
        
        times = Ticktock(df.index.to_pydatetime(), "UTC")
        
        gse_coords.ticks = times
        
        gsm_coords = gse_coords.convert("GSM", "car")
        
        df["Bx"] = gsm_coords.x
        df["By"] = gsm_coords.y
        df["Bz"] = gsm_coords.z

    for col in df.columns:
        if col in constants.VALID_RANGES_VARS_ACE_IMF.keys():
            df.loc[df[col] >= constants.VALID_RANGES_VARS_ACE_IMF[col][1], col] = np.nan
            df.loc[df[col] <= constants.VALID_RANGES_VARS_ACE_IMF[col][0], col] = np.nan

    df = df[cols]
    if resample:
        df = df.groupby(
            pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
        ).agg(group_vars)
        df.columns = ["_".join(x) for x in df.columns]
        if "mean" in group_vars:
            new_cols = []
            for col in df.columns:
                if col.find("_mean"):
                    new_cols.append(col.split("_" + group_vars[0])[0])
                else:
                    new_cols.append(col)
            df.columns = new_cols

    return df




def resample(
    df,
    group_freq="5min",
    group_vars=["mean"],
    group_closed="right",
    group_label="right",
):
    df = df.groupby(
        pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
    ).agg(group_vars)
    df.columns = ["_".join(x) for x in df.columns]

    if "mean" in group_vars:
        new_cols = []
        for col in df.columns:
            if col.find("_mean"):
                new_cols.append(col.split("_" + group_vars[0])[0])
            else:
                new_cols.append(col)
        df.columns = new_cols

    return df


def preprocess_ace_swepam_provisional(
    df,
    cols=constants.COLS_SELECTION_ACE_SWEPAM,
    group_freq="5min",
    group_vars=["mean"],
    group_closed="right",
    group_label="right",
    resample=True,
):
    new_cols = []

    for col in df.columns:
        if col.find("[PRELIM] ") >= 0:
            new_cols.append(col.split("[PRELIM] ")[1])
        else:
            new_cols.append(col)

    df.columns = new_cols

    for col in df.columns:
        if df[col].dtype == "float32":
            for v in constants.NAN_VALUES_SWEPAM:
                df[col].replace(v, np.nan, inplace=True)

    # Replace column values which are outside their range
    for col in df.columns:
        if col in constants.VALID_RANGES_VARS_ACE_SWEPAM.keys():
            df.loc[df[col] >= constants.VALID_RANGES_VARS_ACE_SWEPAM[col][1], col] = (
                np.nan
            )
            df.loc[df[col] <= constants.VALID_RANGES_VARS_ACE_SWEPAM[col][0], col] = (
                np.nan
            )

    df = utils.parse_column_names(df)
    df = df[cols]
    if resample:
        df = df.groupby(
            pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
        ).agg(group_vars)
        df.columns = ["_".join(x) for x in df.columns]

        if "mean" in group_vars:
            new_cols = []
            for col in df.columns:
                if col.find("_mean"):
                    new_cols.append(col.split("_" + group_vars[0])[0])
                else:
                    new_cols.append(col)
            df.columns = new_cols

    return df


def preprocess_ace_swepam(
    df,
    cols=constants.COLS_SELECTION_ACE_SWEPAM,
    group_freq="5min",
    group_vars=["mean"],
    group_closed="right",
    group_label="right",
    resample=True,
):
    """
    Function to preprocess data from ACE SWEPAM dataset,
    only cols will be used,
    the data will be grouped in group_freq frequencies,
    group_vars will be calculated,
    column names will be flattened using the metric in group_vars

    In this case, since the raw data is in 64-second averages, calculating the std would
    have little value. Thus, only the mean is calculated when grouping

    Regarding the column names, mean will be used as the default,
    for the other metrics [if there are], each column will be named
    as {column_name}_{metric}

    Parameters
    ----------
    df : pandas DataFrame

    cols : list, columns to keep, default:  ["Proton_density", "Proton_temp", "Proton_speed"]

    group_freq: to which frequency should de data be grouped, default: 5 minutes

    group_vars: which metrics should be calculated when grouping, default: mean

    Returns
    -------
    preprocessed DataFrame
    """
    # Only step needed is to average in 5 minutes and calculate the mean
    # std is not needed as the source data is already in 64-seconds averages

    # Replace the value they use to represent missing/error values for NaN
    for col in df.columns:
        if df[col].dtype == "float32":
            for v in constants.NAN_VALUES_SWEPAM:
                df[col].replace(v, np.nan, inplace=True)

    # Replace column values which are outside their range
    for col in df.columns:
        if col in constants.VALID_RANGES_VARS_ACE_SWEPAM.keys():
            df.loc[df[col] >= constants.VALID_RANGES_VARS_ACE_SWEPAM[col][1], col] = (
                np.nan
            )
            df.loc[df[col] <= constants.VALID_RANGES_VARS_ACE_SWEPAM[col][0], col] = (
                np.nan
            )

    df = utils.parse_column_names(df)
    df.rename(
        columns={
            "V_GSM_X": "Proton_speed_x",
            "V_GSM_Y": "Proton_speed_y",
            "V_GSM_Z": "Proton_speed_z",
        },
        inplace=True,
    )
    df = df[cols]
    if resample:
        df = df.groupby(
            pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
        ).agg(group_vars)
        df.columns = ["_".join(x) for x in df.columns]

        if "mean" in group_vars:
            new_cols = []
            for col in df.columns:
                if col.find("_mean"):
                    new_cols.append(col.split("_" + group_vars[0])[0])
                else:
                    new_cols.append(col)
            df.columns = new_cols

    return df

def fill_swepam_with_swics(
    df_swepam, df_swics, merge_columns=constants.MERGE_COLS_SWEPAM_SWICS
):
    """
    Function to fill in the missing values from the ACE SWEPAM dataset
    using the values from the ACE SWICS dataset,
    the merging will be done according to constants.MERGE_COLS_RENAMED_SWEPAM_SWICS

    The final names will be the ones from the SWEPAM dataset

    Parameters
    ----------
    df_swepam : pandas DataFrame with the SWEPAM values
                should have atleast the left values for each merge column pair

    df_swics : pandas DataFrame with the SWICS values
                should have atleast the right values for each merge column pair

    merge_columns : how the fill will be done, list of pairs where left is the
                    column in the swepam dataframe that will be filled with the
                    right one in the swics dataframe

    Returns
    -------
    swepam dataframe filled with the swics values
    """

    df = df_swepam.copy()
    for col_pair in merge_columns:
        df[col_pair[0]].fillna(df_swics[col_pair[1]], inplace=True)
    return df


def preprocess_ace_swics(
    df,
    cols=constants.COLS_SELECTION_ACE_SWICS,
    group_freq="5min",
    group_vars=["mean"],
    group_closed="right",
    group_label="right",
    ffill=False,
    resample=True,
):
    """
    Function to preprocess data from ACE SWICS dataset,
    only cols will be used,
    the data will be grouped in group_freq frequencies,
    group_vars will be calculated,
    column names will be flattened using the metric in group_vars

    In this case, the raw data is in 12 min-averages. Thus, we group in 5-min averages
    and interpolate with a limit of 2. That is, only 2 consecutive missing values would be filled,
    a limit of 2 ensures that we are not interpolating originally NA values and we are only using known values

    Additionally, Thermal velocity is transformed to temperature using https://en.wikipedia.org/wiki/Thermal_velocity
    th = ((thvel * 1000) * (thvel * 1000)) * m_proton / cte_boltzmann
    m_proton is the mass of the proton
    cte_boltzmann is the boltzmann constant
    thvel is the thermal velocity
    th is the temperature

    Regarding the column names, mean will be used as the default,
    for the other metrics [if there are], each column will be named
    as {column_name}_{metric}

    Parameters
    ----------
    df : pandas DataFrame with the raw SWICS values

    cols : list, columns to keep, default:  ["Proton_density", "Proton_temp", "Proton_speed"]

    group_freq: to which frequency should de data be grouped, default: 5 minutes

    group_vars: which metrics should be calculated when grouping, default: mean

    Returns
    -------
    preprocessed DataFrame
    """

    # Replace the value they use to represent missing/error values for NaN
    for col in df.columns:
        if df[col].dtype == "float32":
            for v in constants.NAN_VALUES_SWICS:
                df[col].replace(v, np.nan, inplace=True)

    # Replace column values which are outside their range
    for col in df.columns:
        if col in constants.VALID_RANGES_VARS_ACE_SWICS.keys():
            df.loc[df[col] >= constants.VALID_RANGES_VARS_ACE_SWICS[col][1], col] = (
                np.nan
            )
            df.loc[df[col] <= constants.VALID_RANGES_VARS_ACE_SWICS[col][0], col] = (
                np.nan
            )

    thvel = df["vthH"].values
    m_proton = scipy.constants.physical_constants["atomic mass constant"][0]
    cte_boltzmann = scipy.constants.physical_constants["Boltzmann constant"][0]

    th = ((thvel * 1000) * (thvel * 1000)) * m_proton / cte_boltzmann
    df["Proton_temp"] = th

    df = df[cols]
    df = utils.parse_column_names(df)

    if resample:
        if ffill:
            df = df.resample(group_freq, closed=group_closed, label=group_label).ffill()
        else:
            df = df.groupby(
                pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
            ).agg(group_vars)
            df.interpolate(inplace=True, limit=2)
            df.columns = ["_".join(x) for x in df.columns]

        if "mean" in group_vars:
            new_cols = []
            for col in df.columns:
                if col.find("_mean"):
                    new_cols.append(col.split("_" + group_vars[0])[0])
                else:
                    new_cols.append(col)
            df.columns = new_cols

    return df


def calculate_derived_params(df):
    """
    Function to calculate derived params available in OMNI,
    namely: pressure, plasmabeta and the electric field
    https://omniweb.gsfc.nasa.gov/ftpbrowser/bow_derivation.html

    Parameters
    ----------
    df : pandas DataFrame

    Flow pressure = (2*10**-6)*Np*Vp**2 nPa
    plasma_beta = [(T*4.16/10**5) + 5.34] * Np / B**2 (B in nT)
    Electric field = -V(km/s) * Bz (nT; GSM) * 10**-3
    

    Returns
    -------
    preprocessed DataFrame
    """

    df[constants.PRESSURE_COL_NAME] = (
            (2 * 10**-6)
            * df[constants.PROTON_DENSITY_COL_NAME]
            * df[constants.PROTON_SPEED_COL_NAME] ** 2
        )
    
    
    df[constants.PLASMABETA_COL_NAME] = (
        ((df[constants.PROTON_TEMPERATURE_COL_NAME] * 4.16 / (10**5)) + 5.34)
        * df[constants.PROTON_DENSITY_COL_NAME]
        / df[constants.B_MAGNITUDE_COL_NAME] ** 2
    )
        
    # 1. Convert Proton_density from cm^-3 to m^-3
    df["n_p_m3"] = df[constants.PROTON_DENSITY_COL_NAME] * 1e6  # 1 cm^-3 = 1e6 m^-3

    # 2. Calculate mass density: rho = n_p * m_p
    df["rho"] = df["n_p_m3"] * m_p  # kg/m^3

    # 3. Convert Proton_speed from km/s to m/s
    df["v_m_s"] = df[constants.PROTON_SPEED_COL_NAME] * 1e3

    # 4. Compute Kinetic Energy Density: 0.5 * rho * v^2
    df[constants.KINETIC_ENERGY] = 0.5 * df["rho"] * (df["v_m_s"]**2) * 1e9
    
    # 1. Convert Bmag from nT to T
    df["B_T"] = df["Bmag"] * 1e-9
    

    # 2. Compute Magnetic Energy Density
    df[constants.MAGNETIC_ENERGY] = (df["B_T"]**2) / (2.0 * mu_0) * 1e9

    df[constants.TOTAL_ENERGY] = df[constants.KINETIC_ENERGY] + df[constants.MAGNETIC_ENERGY]
    
    df[constants.ELECTRICAL_FIELD_COL_NAME] = -df["Proton_speed"] * df["Bz"] * 1e-3

    
    df = df.drop(columns=["n_p_m3", "rho", "v_m_s", "B_T"], axis=1)

    return df


def fill_missing_values(df, method="ffill"):
    """
    Fill missing values in the DataFrame
    There are four available methods, forward fill, identified as 'ffill'
    backward fill, identified as 'bfill', interpolation, as 'interpolate'

    Parameters
    ----------
    df : pandas DataFrame to impute
    method : method to use ['ffill', 'bfill', 'interpolate', 'impute']
    Returns
    -------
    imputed dataframe
    """
    if method == "ffill":
        # df = df.fillna(method=method)
        df = df.ffill()
        return df
    elif method == "bfill":
        # df = df.fillna(method=method)
        df = df.bfill()
        return df
    elif method == "interpolate":
        df = df.interpolate(limit_area="inside")
        return df
