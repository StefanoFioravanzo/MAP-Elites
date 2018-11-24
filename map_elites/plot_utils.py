import numpy as np
import pandas as pd

from bokeh.io import output_file, show, output_notebook
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform


def test_heat_map():
    from bokeh.sampledata.unemployment1948 import data
    output_file("unemploymemt.html")

    data.Year = data.Year.astype(str)
    data = data.set_index('Year')
    data.drop('Annual', axis=1, inplace=True)
    data.columns.name = 'Month'

    # reshape to 1D array or rates with a month and year for each row.
    df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()
    print(df)
    
    source = ColumnDataSource(df)

    # this is the colormap from the original NYTimes plot
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    # for log scale just use LogColorMapper()
    mapper = LinearColorMapper(palette=colors, low=df.rate.min(), high=df.rate.max())

    p = figure(plot_width=800, plot_height=300, title="US Unemployment 1948—2016",
               x_range=list(data.index), y_range=list(reversed(data.columns)),
               toolbar_location=None, tools="", x_axis_location="above")

    p.rect(x="Year", y="Month", width=1, height=1, source=source,
           line_color=None, fill_color=transform('rate', mapper))

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d%%"))

    p.add_layout(color_bar, 'right')

    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0

    show(p)


def plot_heatmap_2d(data, x_axis, y_axis, x_label, y_label, title="MapElites fitness map", notebook=False):
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


def main():

    # test_heat_map()
    # Generate some random data

    values = np.reshape(np.random.random((5, 5)), (-1,))
    # The two axis must be represented as strings
    x_ax = ['0', '1', '2', '3', '4']
    y_ax = ['5', '6', '7', '8', '9']

    data = np.stack([np.repeat(x_ax, len(y_ax)), np.tile(y_ax, len(x_ax)), values], axis=1)

    plot_heatmap_2d(data, x_ax, y_ax, "X", "Y")


if __name__ == "__main__":
    main()