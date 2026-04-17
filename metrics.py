import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from sklearn.metrics import root_mean_squared_error as rmsem
from sklearn.metrics import r2_score as r2m

from matplotlib.colors import LinearSegmentedColormap


BINWIDTH = 10

COLOR_SUPERINTENSE = "darkmagenta"
COLOR_INTENSE = "firebrick"
COLOR_MODERATE = "goldenrod"
COLOR_LOW = "yellow"
COLOR_INACTIVE = "olivedrab"

COLORS_MULTIPLE = [
    "#FF33FF",  # Magenta
    "#FF8C00",  # Dark Orange
    "#00FF00",  # Lime Green
    "#FF1493",  # Deep Pink
    "#8B4513",  # Saddle Brown
    "#FFD700",  # Gold
]

COLORS_LIGHTER_MULTIPLE = [
    "#FF99FF",  # Light Magenta
    "#FFB266",  # Light Orange
    "#99FF99",  # Light Lime Green
    "#FF66A3",  # Light Deep Pink
    "#D2B48C",  # Tan (Light Saddle Brown)
    "#FFE066",  # Light Gold
]


def roundup(x):
    return int(np.ceil(x / 10.0)) * 10


def rounddown(x):
    return int(np.floor(x / 10.0)) * 10


def calculate_BFE(labels, preds, binwidth=10, bins=None):
    df = pd.DataFrame({"labels": labels, "preds": preds})
    df["diff"] = df["labels"] - df["preds"]
    df["diff"] = df["diff"].abs()
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    if bins is None:
        bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.mid)
    bfe = df.groupby("labels_bins", observed=True)["diff"].mean()
    bfe.index = bfe.index.astype(int)
    return bfe.mean()


def get_BFE(labels, preds, binwidth=10):
    df = pd.DataFrame({"labels": labels, "preds": preds})
    df["diff"] = df["labels"] - df["preds"]
    df["diff"] = df["diff"].abs()
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.mid)
    bfe = df.groupby("labels_bins", observed=True)["diff"].mean()
    bfe.index = bfe.index.astype(int)
    return bfe


def get_BFE_count(labels, binwidth=10):
    df = pd.DataFrame({"labels": labels})
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.mid)
    bfe_count = df.groupby("labels_bins", observed=True)["labels"].count()
    bfe_count.index = bfe_count.index.astype(int)
    return bfe_count


def get_BFE_bins(labels, binwidth=10):
    df = pd.DataFrame({"labels": labels})
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    return bins


def plot_comparison_bfe(
    df,
    df_to_compare,
    title,
    title_to_compare,
    ax=None,
    plot_sym_bars=False,
    plot_asy_bars=False,
    xlabel_title=None,
):
    all_preds = df.copy()
    comparison = df_to_compare.copy()
    index_column_name = df.columns[0]
    comparison["diff_comparison"] = (
        comparison[comparison.columns[0]] - comparison[comparison.columns[1]]
    )
    all_preds["diff_base"] = (
        all_preds[all_preds.columns[0]] - all_preds[all_preds.columns[1]]
    )
    all_preds["diff_comparison"] = comparison["diff_comparison"]

    storm_plot = all_preds.reset_index()
    storm_plot = storm_plot.sort_values(index_column_name)
    storm_plot["diff_base_abs"] = np.abs(storm_plot["diff_base"])
    storm_plot["diff_comparison_abs"] = np.abs(storm_plot["diff_comparison"])
    storm_plot["diff_base_squared"] = storm_plot["diff_base"] * storm_plot["diff_base"]
    storm_plot["diff_comparison_squared"] = (
        storm_plot["diff_comparison"] * storm_plot["diff_comparison"]
    )

    min_index = rounddown(df[all_preds.columns[0]].min())
    max_index = roundup(df[all_preds.columns[0]].max())
    index_range = np.abs(max_index - min_index)

    bins = np.arange(min_index - BINWIDTH, max_index + BINWIDTH, BINWIDTH)

    storm_plot["index_bins"] = pd.cut(
        storm_plot[index_column_name], bins=bins, right=False
    )
    storm_plot["index_bins_mid"] = storm_plot["index_bins"].apply(lambda x: x.mid)
    mean_diff_to_plot_by_bin = storm_plot.groupby("index_bins_mid", observed=True)[
        "diff_base_abs"
    ].mean()
    mean_diff_comparison_to_plot_by_bin = storm_plot.groupby(
        "index_bins_mid", observed=True
    )["diff_comparison_abs"].mean()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    ax.plot(
        mean_diff_to_plot_by_bin.index,
        mean_diff_to_plot_by_bin.values,
        linestyle="-",
        color="blue",
        label=title,
    )
    ax.plot(
        mean_diff_to_plot_by_bin.index,
        mean_diff_comparison_to_plot_by_bin.values,
        linestyle="-",
        color="gray",
        label=title_to_compare,
    )
    # ax.fill_between(mean_diff_to_plot_by_bin.index, mean_diff_to_plot_by_bin.values, 0, alpha = 0.2, color = 'blue')
    ax.plot(0, 0, label="Bin count", color="green")
    ax.fill_between(
        mean_diff_to_plot_by_bin.index,
        mean_diff_to_plot_by_bin.values,
        mean_diff_comparison_to_plot_by_bin.values,
        where=mean_diff_to_plot_by_bin.values
        > mean_diff_comparison_to_plot_by_bin.values,
        alpha=0.2,
        interpolate=True,
        color="gray",
    )

    ax.fill_between(
        mean_diff_to_plot_by_bin.index,
        mean_diff_to_plot_by_bin.values,
        mean_diff_comparison_to_plot_by_bin.values,
        where=mean_diff_to_plot_by_bin.values
        < mean_diff_comparison_to_plot_by_bin.values,
        alpha=0.2,
        interpolate=True,
        color="blue",
    )

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(
        mean_diff_to_plot_by_bin.index.min(), mean_diff_to_plot_by_bin.index.max()
    )
    twin = ax.twinx()
    twin.hist(storm_plot[index_column_name], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    if xlabel_title is None:
        ax.set_xlabel(f"{index_column_name} (nT)", fontsize=18)
    else:
        ax.set_xlabel(xlabel_title, fontsize=18)
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=18)

    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)

    ax.set_title(
        f"Diff BFE ({title_to_compare} - {title}): {mean_diff_comparison_to_plot_by_bin.mean() - mean_diff_to_plot_by_bin.mean():.3f}",
        fontsize=18,
    )

    leg = ax.legend(
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
        ncol=3,
        fancybox=True,
        prop={"size": 14},
    )
    leg.get_lines()[-1].set_linewidth(12.0)
    leg.get_lines()[-1].set_alpha(0.15)
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    twin.grid(False)
    ax.grid(True)

    print(f"Diff BFE ({title_to_compare} - {title}):")
    print(
        f"Total: {mean_diff_comparison_to_plot_by_bin.mean() - mean_diff_to_plot_by_bin.mean():.3f}"
    )

    return ax


def plot_evaluation_bfe(
    df, title, ax=None, plot_sym_bars=False, plot_asy_bars=False, xlabel_title=None
):
    storm_plot = df.copy()
    original_column = storm_plot.columns[0]
    predicted_column = storm_plot.columns[1]
    storm_plot["diff"] = storm_plot[original_column] - storm_plot[predicted_column]
    storm_plot = storm_plot.reset_index()
    storm_plot = storm_plot.sort_values(original_column)
    storm_plot["diff_abs"] = np.abs(storm_plot["diff"])
    storm_plot["diff_squared"] = storm_plot["diff"] * storm_plot["diff"]

    min_index = rounddown(df[original_column].min())
    max_index = roundup(df[original_column].max())

    bins = np.arange(
        min_index - BINWIDTH, max_index + BINWIDTH, BINWIDTH
    )  # You can adjust the bin size and range as needed

    # Cut SYM_H into bins and calculate mean diff_abs within each bin
    storm_plot["index_bins"] = pd.cut(
        storm_plot[original_column], bins=bins, right=False
    )
    storm_plot["index_bins_mid"] = storm_plot["index_bins"].apply(lambda x: x.mid)
    mean_value_to_plot = storm_plot.groupby("index_bins_mid", observed=True)[
        "diff_abs"
    ].mean()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    ax.plot(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        linestyle="-",
        label="Evaluated predictions",
    )
    ax.fill_between(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        0,
        alpha=0.2,
        color="blue",
    )
    ax.plot(0, 0, color="green", label="Bin count")
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(
        mean_value_to_plot.index.min(),
        mean_value_to_plot.index.max(),
    )
    twin = ax.twinx()
    twin.hist(storm_plot[original_column], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    if xlabel_title is None:
        ax.set_xlabel(f"{original_column}", fontsize=18)
    else:
        ax.set_xlabel(xlabel_title, fontsize=18)
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=18)
    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)

    mean_value_to_plot.index = mean_value_to_plot.index.astype(int)
    bfe = calculate_BFE(df[original_column].values, df[predicted_column].values)
    ax.set_title(f"BFE: {bfe:.3f} | {title}", fontsize=18)

    leg = ax.legend(
        ncol=4,
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
    )
    leg.get_lines()[-1].set_linewidth(12.0)
    leg.get_lines()[-1].set_alpha(0.15)
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    twin.grid(False)
    ax.grid(True)

    return ax



def plot_evaluation_bfe_quantile(
    df,
    title,
    ax=None,
    plot_sym_bars=False,
    xlabel_title=None,
    upscaling=10,
    false_color=[1.0, 0.5, 0.5, 1.0],  # Lighter red color
    true_color=[0.5, 1.0, 0.5, 1.0],  # Lighter green color
    full_title=None,
):
    df_plot = df.copy()
    original_column = df_plot.columns[0]
    predicted_column = df_plot.columns[1]
    quantile_column = df_plot.columns[2]
    df_plot["diff"] = df_plot[original_column] - df_plot[predicted_column]
    df_plot = df_plot.reset_index()
    df_plot = df_plot.sort_values(original_column)
    df_plot["diff_abs"] = np.abs(df_plot["diff"])

    min_index = rounddown(df_plot[original_column].min())
    max_index = roundup(df_plot[original_column].max())

    bins = np.arange(min_index - BINWIDTH, max_index + BINWIDTH, BINWIDTH)

    df_plot["index_bins"] = pd.cut(df_plot[original_column], bins=bins, right=False)
    df_plot["index_bins_mid"] = df_plot["index_bins"].apply(lambda x: x.mid)
    mean_value_to_plot = df_plot.groupby("index_bins_mid", observed=True)[
        "diff_abs"
    ].mean()

    if ax is None:
        fig, (ax, ax_color) = plt.subplots(
            2, 1, figsize=(13, 6), gridspec_kw={"height_ratios": [25, 1], "hspace": 0}
        )

    ln1 = ax.plot(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        linestyle="-",
        label="Evaluated predictions",
    )
    ax.fill_between(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        0,
        alpha=0.2,
        color="blue",
    )

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=16, width=2, length=10)
    ax.tick_params(axis="x", which="major", pad=10)
    ax.set_xlim(mean_value_to_plot.index.min(), mean_value_to_plot.index.max())

    mean_value_to_plot.index = mean_value_to_plot.index.astype(int)
    bfe = mean_value_to_plot.mean()
    if full_title is None:
        ax.set_title(
            f"BFE: {bfe:.3f} | Inside 90%: {df_plot[quantile_column].mean():.3f} | {title}",
            fontsize=20,
        )
    else:
        ax.set_title(full_title, fontsize=20)

    twin = ax.twinx()
    twin.hist(df_plot[original_column], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=20)
    twin.set_ylabel("Bin count", fontsize=20)
    twin.tick_params(axis="y", which="major", labelsize=16, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)
    # After plotting the main graph, add the gradient colormap below
    cmap_light = LinearSegmentedColormap.from_list(
        "rg_light", [false_color, true_color]
    )
    # Upscale bfe for a smoother gradient
    quantile_inside = df_plot.groupby("index_bins_mid", observed=True)[
        quantile_column
    ].mean()
    quantile_inside.index = quantile_inside.index.astype(int)
    upscaled_index = np.linspace(
        quantile_inside.index.min(),
        quantile_inside.index.max(),
        len(quantile_inside) * upscaling,
    )
    upscaled_values = np.interp(upscaled_index, quantile_inside.index, quantile_inside)

    # Display the gradient using the new upscaled colors
    ax_color.imshow(
        [cmap_light(upscaled_values)],
        aspect="auto",
        extent=[upscaled_index.min(), upscaled_index.max(), 0, 1],
    )

    twin.grid(False)
    ax.grid(True)

    ax_color.set_yticks([])  # Hide y-axis ticks
    ax_color.set_xticks([])
    ax_color.set_xlim(ax.get_xlim())  # Ensure alignment with the plot above
    if xlabel_title is None:
        ax.set_xlabel(f"{original_column}", fontsize=20)
        ax.xaxis.set_label_coords(0.5, -0.15)
    else:
        ax.set_xlabel(xlabel_title, fontsize=20)
        ax.xaxis.set_label_coords(0.5, -0.15)
    ax_color.grid(False)
    plt.setp(ax_color.spines.values(), lw=2, color="black", alpha=1)
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    plt.setp(twin.spines.values(), lw=2, color="black", alpha=1)

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # manually define a new patch
    patch_bin = mpatches.Patch(color="green", alpha=0.15, label="Bin count")
    patch_false = mpatches.Patch(color=false_color, alpha=1, label="False")
    patch_true = mpatches.Patch(color=true_color, alpha=1, label="True")
    # line = Line2D([0], [0], label='manual line', color='k')
    # handles is a list, so append manual patch
    handles.append(patch_bin)
    handles.append(patch_false)
    handles.append(patch_true)
    # plot the legend
    leg = ax.legend(
        handles=handles,
        ncol=len(handles),
        fancybox=True,
        prop={"size": 16},
        bbox_to_anchor=(0.5, 1.25 if full_title is None else 1.35),
        loc="upper center",
        edgecolor="black",
        facecolor="none",
    )
    return fig, (ax, ax_color)


def calculate_interval_stats(labels, quantile_start, quantile_end, binwidth=10):
    df = pd.DataFrame(
        {
            "labels": labels,
            "quantile_start": quantile_start,
            "quantile_end": quantile_end,
        }
    )
    df["covered"] = (
        (df["quantile_start"] <= df["labels"]) & (df["labels"] <= df["quantile_end"])
    ).astype(int)

    picp = df["covered"].mean()

    df["interval_width"] = df["quantile_end"] - df["quantile_start"]

    width_mean = df["interval_width"].mean()

    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.left)
    biw = df.groupby("labels_bins", observed=True)["interval_width"].mean()
    biw.index = biw.index.astype(int)
    return picp, width_mean, biw.mean()

def plot_evaluation_bfe_multiple(df, title, ax=None, xlabel_title=None, binwidth=10):
    storm_plot = df.copy()
    original_column = storm_plot.columns[0]
    prediction_columns = storm_plot.columns[1:]
    storm_plot = storm_plot.reset_index()
    storm_plot = storm_plot.sort_values(original_column)

    min_index = rounddown(df[original_column].min())
    max_index = roundup(df[original_column].max())

    bins = np.arange(min_index - binwidth, max_index + binwidth, binwidth)

    storm_plot["index_bins"] = pd.cut(
        storm_plot[original_column], bins=bins, right=False
    )
    storm_plot["index_bins_mid"] = storm_plot["index_bins"].apply(lambda x: x.mid)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    bfe_values = []
    for i, pred_col in enumerate(prediction_columns):
        storm_plot[f"diff_{pred_col}"] = (
            storm_plot[original_column] - storm_plot[pred_col]
        )
        storm_plot[f"diff_abs_{pred_col}"] = np.abs(storm_plot[f"diff_{pred_col}"])
        mean_value_to_plot = storm_plot.groupby("index_bins_mid", observed=True)[
            f"diff_abs_{pred_col}"
        ].mean()

        ax.plot(
            mean_value_to_plot.index,
            mean_value_to_plot.values,
            linestyle="-",
            label=f"{pred_col}",
            color=COLORS_MULTIPLE[i % len(COLORS_MULTIPLE)],
        )
        ax.fill_between(
            mean_value_to_plot.index,
            mean_value_to_plot.values,
            0,
            alpha=0.2,
            color=COLORS_MULTIPLE[i % len(COLORS_MULTIPLE)],
        )

        bfe = calculate_BFE(df[original_column].values, df[pred_col].values)
        bfe_values.append(f"{pred_col}: {bfe:.3f}")

    ax.plot(0, 0, color="green", label="Bin count")
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(
        mean_value_to_plot.index.min(),
        mean_value_to_plot.index.max(),
    )
    twin = ax.twinx()
    twin.hist(storm_plot[original_column], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    if xlabel_title is None:
        ax.set_xlabel(f"{original_column}", fontsize=18)
    else:
        ax.set_xlabel(xlabel_title, fontsize=18)
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=18)
    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)

    ax.set_title(f"BFE {title}\n{' | '.join(bfe_values)}", fontsize=18)

    leg = ax.legend(
        ncol=len(prediction_columns) + 1,
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.25),
        loc="upper center",
    )
    leg.get_lines()[-1].set_linewidth(12.0)
    leg.get_lines()[-1].set_alpha(0.15)
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    twin.grid(False)
    ax.grid(True)

    return ax
