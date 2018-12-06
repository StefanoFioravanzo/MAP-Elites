import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from bokeh.io import output_file, show, output_notebook
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform


def plot_heatmap_2d_bokeh(data, x_axis, y_axis, x_label, y_label, title="MapElites fitness map", notebook=False):
    """
    :param data: Assume a list of tuples in the form (value, x_value, y_value)
    :param x_axis: Array of strings with the ticks of x axis
    :param y_axis: Array of strings with the ticks of y axis
    :param x_label: Description of x axis
    :param y_label: Description of y axis
    """
    plot_data = pd.DataFrame(data, columns=[x_label, y_label, 'value']).reset_index()
    plot_data.value = plot_data.value.astype(np.float64)
    plot_data.fillna(0, inplace=True)
    plot_data.replace([np.inf, -np.inf], 0, inplace=True)
    plot_data.mask(np.nan, inplace=True)

    # convert `value` column to numeric type
    plot_data['value'] = plot_data['value'].apply(pd.to_numeric)
    
    source = ColumnDataSource(plot_data)
    colors = ["#8c2d04", "#d94801", "#f16913", "#fd8d3c", "#fdae6b", "#fdd0a2", "#fee6ce", "#fff5eb"]
    mapper = LinearColorMapper(palette=colors, low=plot_data.value.min(), high=plot_data.value.max())

    p = figure(plot_width=800, plot_height=300, title=title,
               x_range=x_axis, y_range=y_axis,
               toolbar_location=None, tools="")

    p.rect(x=x_label, y=y_label, width=1, height=1, source=source,
           line_color=None, fill_color=transform('value', mapper))

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%8.2f"))
    p.add_layout(color_bar, 'right')

    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0

    if notebook:
        # enable notebook display
        output_notebook()
    show(p)


# TODO: Invert color mapping in case of maximization
def plot_heatmap(data, x_axis=None, y_axis=None, x_label='x label', y_label='y label',
                     title="MapElites fitness map", minimization=True, notebook=False):
    # get data dimensionality
    d = data.shape

    # reshape data to obtain a 2d heatmap
    if len(d) == 3:
        data = data.reshape((d[0], d[1] * d[2]))
    if len(d) == 4:
        data = data.reshape((d[0], d[1] * d[2], d[3])).swapaxes(0, 1).reshape((d[1] * d[2], d[0] * d[3]))

    plt.subplots(figsize=(10, 10))

    # TODO: Define x_axis and y_axis
    y_axis = [str(i) for i in range(0, d[0])]
    x_axis = ['1']
    if len(d) == 2:
        y_axis = [str(i) for i in range(0, d[1])]
    if len(d) == 3:
        x_axis = np.repeat(x_axis, d[2])
    if len(d) == 4:
        y_axis = np.repeat(y_axis, d[2])
        x_axis = np.repeat(x_axis, d[3])

    df_data = pd.DataFrame(data, columns=[x_axis], index=[y_axis])
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
        fmt=".1f",
        annot_kws={'size': 10},
        # cbar_kws={"ticks": cbar_ticks},
        linecolor='white'
    )

    ax.set_title(f"{title} - white cells are null values (not initialized)")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.invert_yaxis()

    # show grid lines
    if len(d) == 3:
        ax.vlines(list(range(0, d[1] * d[2], d[2])), *ax.get_ylim(), colors='g')
        ax.hlines(list(range(0, d[0])), *ax.get_xlim(), colors='g')
    if len(d) == 4:
        ax.hlines(list(range(0, d[1] * d[2], d[2])), *ax.get_ylim(), colors='g')
        ax.vlines(list(range(0, d[0] * d[3], d[3])), *ax.get_xlim(), colors='g')

    plt.show()


def main():
    """
    Test plot utils by calling this module directly
    """

    # test_heat_map()
    # Generate some random data

    values = np.reshape(np.random.random((5, 5)), (-1,))
    # The two axis must be represented as strings
    x_ax = ['0', '1', '2', '3', '4']
    y_ax = ['5', '6', '7', '8', '9']

    data = np.stack([np.repeat(x_ax, len(y_ax)), np.tile(y_ax, len(x_ax)), values], axis=1)

    plot_heatmap_2d(data, x_ax, y_ax, "X", "Y")
    
    
def test_3D():
    x_d = 5
    y_d = 5
    z_d = 9
    z_d_cell = math.sqrt(z_d)
    l = np.reshape(np.tile(np.tile(np.array(range(1, z_d+1)), x_d), y_d), (x_d, y_d, z_d))
    matrix = l.reshape((x_d, y_d * z_d_cell, z_d_cell)).swapaxes(0, 1).reshape((x_d * z_d_cell, y_d * z_d_cell))

if __name__ == "__main__":
    main()