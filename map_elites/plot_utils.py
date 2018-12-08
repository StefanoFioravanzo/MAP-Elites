import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm


# TODO: Invert color mapping in case of maximization
def plot_heatmap(data, x_axis=None, y_axis=None, x_label=None, y_label=None,
                     title="MapElites fitness map", minimization=True, savefig_path=None):
    # get data dimensionality
    d = data.shape

    # reshape data to obtain a 2d heatmap
    if len(d) == 3:
        data = data.reshape((d[0], d[1] * d[2]))
    if len(d) == 4:
        data = data.reshape((d[0], d[1] * d[2], d[3])).swapaxes(0, 1).reshape((d[1] * d[2], d[0] * d[3]))

    plt.subplots(figsize=(10, 10))

    df_data = pd.DataFrame(data)
    df_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # TODO: Allow for log scale colormap w/ negative value
    min = df_data.min().min()
    max = df_data.max().max()
    # log_norm = LogNorm(vmin=min, vmax=max)
    # cbar_ticks = [math.pow(10, i)
    #               for i in range(math.floor(math.log10(min)), 1 + math.ceil(math.log10(max)))]

    mask = df_data.isnull()
    ax = sns.heatmap(
        df_data,
        mask=mask,
        annot=True,
        # norm=log_norm,
        fmt=".4f",
        annot_kws={'size': 10},
        # cbar_kws={"ticks": cbar_ticks},
        linewidths=.5,
        linecolor='black',
        cmap="YlGnBu",
        xticklabels=False,
        yticklabels=False
    )

    ax.set_title(f"{title} - white cells are null values (not initialized)")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.invert_yaxis()

    # set ticks
    x_ticks_pos = [0.5]
    y_ticks_pos = range(0, d[1]+1)
    if len(d) == 2:
        x_ticks_pos = range(0, d[0]+1)
    if len(d) > 2:
        x_ticks_pos = range(0, d[0]+1, d[2])
    if len(d) > 3:
        y_ticks_pos = range(0, d[1]+1, d[3])

    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks_pos))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_axis))

    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks_pos))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(y_axis))

    # show grid lines
    thick_grid_color = 'k'
    thick_grid_width = 1
    if len(d) == 3:
        ax.vlines(
            list(range(0, d[1] * d[2], d[2])),
            *ax.get_ylim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )
        ax.hlines(
            list(range(0, d[0])),
            *ax.get_xlim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )
    if len(d) == 4:
        ax.hlines(
            list(range(0, d[1] * d[2], d[2])),
            *ax.get_ylim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )
        ax.vlines(
            list(range(0, d[0] * d[3], d[3])),
            *ax.get_xlim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )

    # get figure to save to file
    if savefig_path:
        ht_figure = ax.get_figure()
        ht_figure.savefig(savefig_path / "heatmap.png", dpi=400)
    plt.show()


def _test_plotting():
    """
    Test plot utils by calling this module directly
    """
    # Generate some random data
    values = np.reshape(np.random.random((5, 5)), (-1,))
    # The two axis must be represented as strings
    x_ax = ['0', '1', '2', '3', '4']
    y_ax = ['5', '6', '7', '8', '9']

    data = np.stack([np.repeat(x_ax, len(y_ax)), np.tile(y_ax, len(x_ax)), values], axis=1)

    plot_heatmap(data, x_ax, y_ax, "X", "Y")


if __name__ == "__main__":
    _test_plotting()