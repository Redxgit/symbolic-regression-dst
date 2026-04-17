import os
import natsort
import pandas as pd
import constants


def read_data(
    path,
    pattern_to_skip=None,
    pattern_to_read=["csv"],
    return_separated=True,
    tz_localize=False,
    print_info=False,
):

    # Check if the path is a file
    if os.path.isfile(path):
        # If it is a file we don't need to apply the pattern
        if print_info:
            print(f"Reading from file {path}")
        df = pd.read_csv(path, comment="#")
        df.set_index("datetime", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        if tz_localize:
            df.index = df.index.tz_localize("UTC")
        return df

    # Check if the path is a folder
    if os.path.isdir(path):
        dfs = []
        files = natsort.natsorted(os.listdir(path))
        for f in files:
            skip = False

            # If a pattern to skip is found, skip the file
            if pattern_to_skip is not None:
                for pat in pattern_to_skip:
                    if f.find(pat) >= 0:
                        skip = True
                        break

            # If a pattern to read is not found, skip the file
            if pattern_to_read is not None:
                for pat in pattern_to_read:
                    if f.find(pat) < 0:
                        skip = True

            if skip:
                continue

            fil = os.path.join(path, f)
            if print_info:
                print(f"Reading from file {fil}")
            df1 = pd.read_csv(fil)
            df1.set_index("datetime", inplace=True)
            df1.index = pd.to_datetime(df1.index)
            df1.sort_index(inplace=True)
            if tz_localize:
                df1.index = df1.index.tz_localize("UTC")
            dfs.append(df1)

        if return_separated:
            return dfs
        else:
            df = pd.concat(dfs)
            df.sort_index(inplace=True)
            return df

    print(f"Is neither a valid file or directory")

def read_iaga_file(path: str, columns = ["DATE", "TIME", "DOY", "DST"]) -> pd.DataFrame:
    """
    Read an IAGA-2002 formatted geomagnetic file (e.g., Dst, AE) into a clean DataFrame.
    Handles repeated headers and whitespace-prefixed metadata.
    """
    data_rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Skip comment and header lines
            if line.startswith(" ") or line.startswith("#") or line.startswith("DATE"):
                continue
            parts = line.strip().split()
            # We expect lines like: 1998-01-01 00:00:00.000 001 -9.00
            data_rows.append(parts[:4])

    # Convert to DataFrame
    df = pd.DataFrame(data_rows, columns=columns)

    # Combine date and time
    df["datetime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")

    # Set index
    df = df.set_index("datetime").sort_index()
    # Keep only useful columns
    df = df.drop(columns = ['DATE', 'TIME', 'DOY'])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def parse_column_names(df):
    """
    Rename columns according to constants.PARSING_DICT

    Parameters
    ----------
    df : input dataframe to rename cols

    Returns
    -------
    df with columns renamed
    """
    for col in df.columns:
        for col_name, col_alternatives in constants.PARSING_DICT:
            if col in col_alternatives:
                df = df.rename(columns={col: col_name})
                continue

    return df