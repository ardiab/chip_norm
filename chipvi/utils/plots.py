import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def hist2d(
        data,
        x_col,
        y_col,
        ax,
        cmap="inferno",
        sample_size=100_000,
        discrete_bins=True,
        bins=50,
):
    sample_size = min(sample_size, len(data))
    data_sub = data.sample(sample_size)

    # Clip to 1st-99th percentile to avoid extreme outliers affecting the plot range
    x_max_q = data_sub[x_col].quantile(0.99)
    y_max_q = data_sub[y_col].quantile(0.99)
    x_min_q = data_sub[x_col].quantile(0.01)
    y_min_q = data_sub[y_col].quantile(0.01)

    data_sub = data_sub[
        (data_sub[x_col] >= x_min_q) & (data_sub[x_col] <= x_max_q) &
        (data_sub[y_col] >= y_min_q) & (data_sub[y_col] <= y_max_q)
    ]

    if discrete_bins:
        x_min_val, x_max_val = data_sub[x_col].min(), data_sub[x_col].max()
        y_min_val, y_max_val = data_sub[y_col].min(), data_sub[y_col].max()

        # If range is too large or data is float, fall back to linspace bins
        is_float = (data_sub[x_col].dtype != np.int64) or (data_sub[y_col].dtype != np.int64)
        if (x_max_val - x_min_val > 200) or (y_max_val - y_min_val > 200) or is_float:
             x_bins = np.linspace(x_min_val, x_max_val, bins)
             y_bins = np.linspace(y_min_val, y_max_val, bins)
        else:
             x_bins = np.arange(int(x_min_val), int(x_max_val) + 2)
             y_bins = np.arange(int(y_min_val), int(y_max_val) + 2)
    else:
        x_bins = np.linspace(data_sub[x_col].min(), data_sub[x_col].max(), bins)
        y_bins = np.linspace(data_sub[y_col].min(), data_sub[y_col].max(), bins)

    hist, xedges, yedges = np.histogram2d(data_sub[x_col], data_sub[y_col], bins=[x_bins, y_bins])

    # Use pcolormesh for better axis handling and LogNorm for better visibility
    pcm = ax.pcolormesh(xedges, yedges, (hist + 1).T, cmap=cmap, norm=LogNorm())
    plt.colorbar(pcm, ax=ax, label="Count")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Use the subsampled data for correlations to match the plot
    pearson_corr, _ = pearsonr(data_sub[x_col], data_sub[y_col])
    spearman_corr, _ = spearmanr(data_sub[x_col], data_sub[y_col])

    ax.set_title(f"Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}")
    ax.grid(False)