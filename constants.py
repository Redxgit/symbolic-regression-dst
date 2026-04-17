# Verbose to project name variable
B_MAGNITUDE_COL_NAME = "Bmag"
B_MAGNITUDE_STD_COL_NAME = "Bmag_std"
BX_COL_NAME = "Bx"
BX_STD_COL_NAME = "Bx_std"
BY_COL_NAME = "By"
BY_STD_COL_NAME = "By_std"
BZ_COL_NAME = "Bz"
BZ_STD_COL_NAME = "Bz_std"
SYM_H_COL_NAME = "SYM_H"
ASY_H_COL_NAME = "ASY_H"
PROTON_SPEED_COL_NAME = "Proton_speed"
PROTON_TEMPERATURE_COL_NAME = "Proton_temp"
PROTON_DENSITY_COL_NAME = "Proton_density"
ELECTRICAL_FIELD_COL_NAME = "E_field"
PLASMABETA_COL_NAME = "Plasmabeta"
PRESSURE_COL_NAME = "Pressure"
KINETIC_ENERGY = "Kinetic_energy"
MAGNETIC_ENERGY = "Magnetic_energy"
TOTAL_ENERGY = "Total_energy"

COLS_TO_USE = [
    B_MAGNITUDE_COL_NAME,
    BX_COL_NAME,
    BY_COL_NAME,
    BZ_COL_NAME,
    PROTON_DENSITY_COL_NAME,
    PROTON_SPEED_COL_NAME,
    PROTON_TEMPERATURE_COL_NAME,
    PRESSURE_COL_NAME,
    ELECTRICAL_FIELD_COL_NAME,
    MAGNETIC_ENERGY,
    KINETIC_ENERGY,
    TOTAL_ENERGY,
]

COLS_TO_USE_NO_DERIVED = [
    B_MAGNITUDE_COL_NAME,
    BX_COL_NAME,
    BY_COL_NAME,
    BZ_COL_NAME,
    PROTON_DENSITY_COL_NAME,
    PROTON_SPEED_COL_NAME,
    PROTON_TEMPERATURE_COL_NAME,
]

COLS_TO_SCALE_LOG = [
    PROTON_DENSITY_COL_NAME,
    PROTON_TEMPERATURE_COL_NAME,
    PROTON_SPEED_COL_NAME,
    PRESSURE_COL_NAME,
]

COLS_TO_SCALE_STANDARD = [PROTON_SPEED_COL_NAME, BX_COL_NAME, BY_COL_NAME, BZ_COL_NAME]

COLS_TO_SCALE_ROBUST = [
    PROTON_TEMPERATURE_COL_NAME,
    PROTON_DENSITY_COL_NAME,
    PRESSURE_COL_NAME,
    ELECTRICAL_FIELD_COL_NAME,
    B_MAGNITUDE_COL_NAME,
    MAGNETIC_ENERGY,
    TOTAL_ENERGY,
    KINETIC_ENERGY,
]

COLOR_STATIONS = {
    "ABG": "#FF0000",  # Red
    "MMB": "#00FF00",  # Green
    "CLF": "#FFA500",  # Orange
    "TUC": "#800080",  # Purple
    "HON": "#00FFFF",  # Cyan
    "SFS": "#FF00FF",  # Magenta
}

COUNTRY_STATIONS = {
    "ABG": "India",
    "MMB": "Japan",
    "CLF": "France",
    "TUC": "USA",
    "HON": "USA",
    "SFS": "Spain",
}

STATION_TO_INDEX = {
    "ABG": 0,
    "MMB": 1,
    "CLF": 2,
    "TUC": 3,
    "HON": 4,
    "SFS": 5,
}

SYM_BFE_TO_STATION = {
    "ABG": 44.544,
    "MMB": 48.869,
    "CLF": 73.672,
    "TUC": 38.918,
    "HON": 29.032,
    "SFS": 54.335,
}

TRAIN_STATIONS = [
    "ABG",  # India
    "MMB",  # Japan
    "CLF",  # France
    "TUC",  # USA
    "HON",  # USA
]

STATIONS = [
    "ABG",  # India
    "MMB",  # Japan
    "CLF",  # France
    "TUC",  # USA
    "HON",  # USA
    "SFS",  # Spain
]

# Alternative names for each variable, will be used when parsing names
ALTERNATIVE_NAMES_BMAG = ["Bt", "Magnitude", "F", "B1F1", "bt"]
ALTERNATIVE_NAMES_BX = ["bx_gsm", "Bx (GSM)", "BX_GSE", "BGSM_X"]
ALTERNATIVE_NAMES_BY = ["by_gsm", "By (GSM)", "BY_GSM", "BGSM_Y"]
ALTERNATIVE_NAMES_BZ = ["bz_gsm", "Bz (GSM)", "BZ_GSM", "BGSM_Z"]

ALTERNATIVE_NAMES_PROTON_DENSITY = [
    "Proton density",
    "density",
    "Np",
    "nH",
    "proton_density",
]
ALTERNATIVE_NAMES_PROTON_SPEED = ["Proton speed", "speed", "Vp", "vH", "flow_speed"]
ALTERNATIVE_NAMES_PROTON_TEMPERATURE = [
    "Temperature",
    "temperature",
    "Tpr",
    "T",
    "THERMAL_TEMP",
]

ALTERNATIVE_NAMES_SYM_H = ["SYM"]
ALTERNATIVE_NAMES_ASY_H = ["ASY"]

PARSING_DICT = [
    (B_MAGNITUDE_COL_NAME, ALTERNATIVE_NAMES_BMAG),
    (BX_COL_NAME, ALTERNATIVE_NAMES_BX),
    (BY_COL_NAME, ALTERNATIVE_NAMES_BY),
    (BZ_COL_NAME, ALTERNATIVE_NAMES_BZ),
    (PROTON_DENSITY_COL_NAME, ALTERNATIVE_NAMES_PROTON_DENSITY),
    (PROTON_TEMPERATURE_COL_NAME, ALTERNATIVE_NAMES_PROTON_TEMPERATURE),
    (PROTON_SPEED_COL_NAME, ALTERNATIVE_NAMES_PROTON_SPEED),
    (SYM_H_COL_NAME, ALTERNATIVE_NAMES_SYM_H),
    (ASY_H_COL_NAME, ALTERNATIVE_NAMES_ASY_H),
]


# Dataset identifiers to download datasets from the CDAWeb using cadsws
# https://cdaweb.gsfc.nasa.gov/WebServices/REST/py/cdasws/#cdasws.CdasWs.get_data
DATABASE_ACE_IMF = "AC_H3_MFI"
DATABASE_ACE_IMF_16 = "AC_H0_MFI"
DATABASE_ACE_SWEPAM = "AC_H0_SWE"
DATABASE_ACE_SWICS = "AC_H6_SWI"
DATABASE_OMNI = "OMNI_HRO_5MIN"
DATABASE_DSCOVR_MAG = "DSCOVR_H0_MAG"
DATABASE_DSCOVR_PLASMA = "DSCOVR_H1_FC"
DATABASE_ACE_IMF_PROVISIONAL = "AC_K1_MFI"
DATABASE_ACE_PLASMA_PROVISIONAL = "AC_K0_SWE"
URL_MAG = "https://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json"
NAME_MAG = "products-solar-wind-mag-3-day.json"
URL_SWEPAM = "https://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json"
NAME_SWEPAM = "products-solar-wind-plasma-3-day.json"

# Variables to download from the ace imf dataset https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H3_MFI
VARS_ACE_IMF = ["Magnitude", "BRTN", "BGSEc", "BGSM"]
# Variables to download from the ace imf h0 dataset
VARS_ACE_IMF_H0 = ["Magnitude", "BGSEc", "BGSM", "dBrms", "SC_pos_GSE", "SC_pos_GSM"]
# Variables to download from the ace imf dataset https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H3_MFI
VARS_ACE_IMF_PROVISIONAL = ["Magnitude", "BGSEc"]
# Valid ranges https://hpde.io/NASA/NumericalData/ACE/MAG/L2/PT1S
VALID_RANGES_VARS_ACE_IMF = {
    "Magnitude": (0.0, 500.0),
    "Bmag": (0.0, 500.0),
    "Br RTN": (-100.0, 100.0),
    "Bt RTN": (-100.0, 100.0),
    "Bn RTN": (-100.0, 100.0),
    "Bx GSE": (-100.0, 100.0),
    "By GSE": (-100.0, 100.0),
    "Bz GSE": (-100.0, 100.0),
    "Bx (GSM)": (-100.0, 100.0),
    "By (GSM)": (-100.0, 100.0),
    "Bz (GSM)": (-100.0, 100.0),
    "Bx": (-100.0, 100.0),
    "By": (-100.0, 100.0),
    "Bz": (-100.0, 100.0),
}

# Variables to download from the ace plasma dataset https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H0_SWE
VARS_ACE_SWEPAM = [
    "Np",
    "Vp",
    "Tpr",
    "alpha_ratio",
    "V_GSE",
    "V_RTN",
    "V_GSM",
    "SC_pos_GSE",
    "SC_pos_GSM",
]

VARS_ACE_SWEPAM_PROVISIONAL = [
    "Np",
    "Vp",
    "Tpr",
]

# Valid ranges https://hpde.io/NASA/NumericalData/ACE/SWEPAM/L2/PT64S
VALID_RANGES_VARS_ACE_SWEPAM = {
    "Np": (0.0, 200.0),
    "Vp": (0.0, 2500.0),
    "Tpr": (1000.0, 1100000.0),
    "alpha_ratio": (0.0, 10.0),
    "VX (GSE)": (-2000.0, 0.0),
    "VY (GSE)": (-900.0, 900.0),
    "VZ (GSE)": (-900.0, 900.0),
    "VR (RTN)": (-2000.0, 0.0),
    "VT (RTN)": (-900.0, 900.0),
    "VN (RTN)": (-900.0, 900.0),
    "VX (GSM)": (-1800.0, 0.0),
    "VY (GSM)": (-900.0, 900.0),
    "VZ (GSM)": (-900.0, 900.0),
    "X GSE": (-2000000.0, 2000000.0),
    "Y GSE": (-2000000.0, 2000000.0),
    "Z GSE": (-2000000.0, 2000000.0),
}


# Variables to download from the ace swics dataset https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H6_SWI
VARS_ACE_SWICS = ["nH", "nH_err", "vH", "vthH"]
# Valid ranges https://hpde.io/NASA/NumericalData/ACE/SWICS/L2/PT12M and fill value
VALID_RANGES_VARS_ACE_SWICS = {
    "nH": (0.0, 200.0),
    "nH_err": (0.0, 200.0),
    "vH": (0.0, 2000.0),
    "vthH": (0.0, 2000.0),
}

# Variables to download from the DSCOVR MAG dataset https://cdaweb.gsfc.nasa.gov/misc/NotesD.html#DSCOVR_H0_MAG
VARS_DSCOVR_MAG = ["B1F1", "B1SDF1", "B1GSE", "B1SDGSE", "B1RTN", "B1SDRTN"]
# Valid ranges https://hpde.io/NOAA/NumericalData/DSCOVR/PlasMag/FluxgateMagnetometer/CDF/PT1S
VALID_RANGES_VARS_DSCOVR_MAG = {
    "B1F1": (0.0, 65534.0),
    "B1SDF1": (0.0, 65534.0),
    "Bx (GSE)": (-65534.0, 65534.0),
    "By (GSE)": (-65534.0, 65534.0),
    "Bz (GSE)": (-65534.0, 65534.0),
    "Bx_SIGMA (GSE)": (0.0, 65534.0),
    "By_SIGMA (GSE)": (0.0, 65534.0),
    "Bz_SIGMA (GSE)": (0.0, 65534.0),
    "Br (RTN)": (-65534.0, 65534.0),
    "Bt (RTN)": (-65534.0, 65534.0),
    "Bn (RTN)": (-65534.0, 65534.0),
    "Br_SIGMA (RTN)": (0.0, 65534.0),
    "Bt_SIGMA (RTN)": (0.0, 65534.0),
    "Bn_SIGMA (RTN)": (0.0, 65534.0),
}

# Variables to download from the DSCOVR Plasma dataset https://cdaweb.gsfc.nasa.gov/misc/NotesD.html#DSCOVR_H1_FC
VARS_DSCOVR_PLASMA = [
    "DQF",
    "V_GSE",
    "V_GSE_ErrorBars",
    "THERMAL_SPD",
    "THERMAL_SPD_ErrorBars",
    "Np",
    "Np_ErrorBars",
    "THERMAL_TEMP",
    "THERMAL_TEMP_ErrorBars",
]
# Valid ranges https://hpde.io/NOAA/NumericalData/DSCOVR/PlasMag/FaradayCup/CDF/PT1M
VALID_RANGES_VARS_DSCOVR_PLASMA = {
    "VX (GSE)": (-1800.0, 0.0),
    "VY (GSE)": (-900.0, 900.0),
    "VZ (GSE)": (-900.0, 900.0),
    "THERMAL_SPD": (0.0, 1000.0),
    "THERMAL_SPD_NoError": (0.0, 1000.0),
    "Np": (0.0, 1000.0),
    "Np_NoError": (0.0, 1000.0),
    "THERMAL_TEMP": (0.0, 2000000.0),
    "THERMAL_TEMP_NoError": (0.0, 2000000.0),
}

# Variables to download from the OMNI 5-min hro dataset https://cdaweb.gsfc.nasa.gov/misc/NotesO.html#OMNI_HRO_5MIN
VARS_OMNI = [
    "IMF",
    "PLS",
    "IMF_PTS",
    "PLS_PTS",
    "percent_interp",
    "Timeshift",
    "RMS_Timeshift",
    "Time_btwn_obs",
    "F",
    "BX_GSE",
    "BY_GSE",
    "BZ_GSE",
    "BY_GSM",
    "BZ_GSM",
    "RMS_SD_B",
    "RMS_SD_fld_vec",
    "flow_speed",
    "Vx",
    "Vy",
    "Vz",
    "proton_density",
    "T",
    "Pressure",
    "E",
    "Beta",
    "Mach_num",
    "Mgs_mach_num",
    "x",
    "y",
    "z",
    "BSN_x",
    "BSN_y",
    "BSN_z",
    "AE_INDEX",
    "AL_INDEX",
    "AU_INDEX",
    "SYM_D",
    "SYM_H",
    "ASY_D",
    "ASY_H",
    "PC_N_INDEX",
    "PR-FLX_10",
    "PR-FLX_30",
    "PR-FLX_60",
]
# Omni doesnt have valid ranges https://hpde.io/NASA/NumericalData/OMNI/HighResolutionObservations/Version1/PT5M

# Columns to keep from the ACE_IMF dataset
COLS_SELECTION_ACE_MAG = [B_MAGNITUDE_COL_NAME, BX_COL_NAME, BY_COL_NAME, BZ_COL_NAME]

# Columns to keep from the ACE_SWEPAM dataset
COLS_SELECTION_ACE_SWEPAM = [
    PROTON_DENSITY_COL_NAME,
    PROTON_SPEED_COL_NAME,
    PROTON_TEMPERATURE_COL_NAME,
]

FORECAST_STEPS = 2

# Columns to keep from the ACE_SWICS dataset
COLS_SELECTION_ACE_SWICS = ["nH", "vH", PROTON_TEMPERATURE_COL_NAME]

# Columns to keep from the OMNI dataset
COLS_SELECTION_OMNI = [
    "flow_speed",
    "proton_density",
    "T",
    "F",
    "BX_GSE",
    "BY_GSM",
    "BZ_GSM",
    "SYM_H",
    "ASY_H",
]

# COLS_SELECTION_DSCOVR_MAG = ["B1F1", "Bx (GSE)", "By (GSE)", "Bz (GSE)"]
COLS_SELECTION_DSCOVR_MAG = [
    B_MAGNITUDE_COL_NAME,
    BX_COL_NAME,
    BY_COL_NAME,
    BZ_COL_NAME,
]
COLS_SELECTION_DSCOVR_PLASMA = [
    PROTON_DENSITY_COL_NAME,
    PROTON_SPEED_COL_NAME,
    PROTON_TEMPERATURE_COL_NAME,
]

# Names of the columns for the indices
INDICES_COLS = ["SYM_H", "ASY_H"]

# Column pair to merge the swepam and swics plasma variables
# after renaming used by fill_swepam_with_swics
MERGE_COLS_SWEPAM_SWICS = [
    ("Proton_density", "Proton_density"),
    ("Proton_speed", "Proton_speed"),
    ("Proton_temp", "Proton_temp"),
]

# Columns that will be used to impute data
COLUMNS_FOR_IMPUTATION = [
    B_MAGNITUDE_COL_NAME,
    BX_COL_NAME,
    BY_COL_NAME,
    BZ_COL_NAME,
    PROTON_DENSITY_COL_NAME,
    PROTON_SPEED_COL_NAME,
    PROTON_TEMPERATURE_COL_NAME,
]

# NaN identifier for the ACE IMF dataset
NAN_VALUES_ACE = (-9999999848243207295109594873856.0, float(-1.0e31))
# NaN value for the ACE SWEPAM dataset
NAN_VALUES_SWEPAM = (-9999999848243207295109594873856.0, float(-1.0e31))
# NaN value for the ACE SWICS dataset
NAN_VALUES_SWICS = (-9999999848243207295109594873856.0, float(-1.0e31))
# NaN value for the DSCOVR dataset
NAN_VALUES_DSCOVR = (-9999999848243207295109594873856.0, float(-1.0e31))

# NaN value for the OMNI dataset
NA_VALUES_OMNI = {
    "IMF": 99,
    "PLS": 99,
    "IMF_PTS": 999,
    "PLS_PTS": 999,
    "percent_interp": 999,
    "Timeshift": 999999,
    "RMS_Timeshift": 999999,
    "Time_btwn_obs": 999999,
    "F": 9999.990234375,
    "BX_GSE": 9999.990234375,
    "BY_GSE": 9999.990234375,
    "BZ_GSE": 9999.990234375,
    "BY_GSM": 9999.990234375,
    "BZ_GSM": 9999.990234375,
    "RMS_SD_B": 9999.990234375,
    "RMS_SD_fld_vec": 9999.990234375,
    "flow_speed": 99999.8984375,
    "Vx": 99999.8984375,
    "Vy": 99999.8984375,
    "Vz": 99999.8984375,
    "proton_density": 999.989990234375,
    "T": 9999999.0,
    "Pressure": 99.98999786376953,
    "E": 999.989990234375,
    "Beta": 999.989990234375,
    "Mach_num": 999.9000244140625,
    "Mgs_mach_num": 99.9000015258789,
    "x": 9999.990234375,
    "y": 9999.990234375,
    "z": 9999.990234375,
    "BSN_x": 9999.990234375,
    "BSN_y": 9999.990234375,
    "BSN_z": 9999.990234375,
}

# Realtime address for the ACE SWEPAM data
REAL_TIME_ACE_SWEPAM = "https://services.swpc.noaa.gov/text/ace-swepam.txt"
# Realtime address for the ACE MAG data
REAL_TIME_ACE_MAG = "https://services.swpc.noaa.gov/text/ace-magnetometer.txt"
# Realtime address for the DSCOVR MAG data
REAL_TIME_DSCOVR_MAG = (
    "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
)
# Realtime address for the DSCOVR Plasma data
REAL_TIME_DSCOVR_PLASMA = (
    "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
)

# Plot related stuff
INCHES_PER_COLUMN = 4
FIG_WIDTH = 10
