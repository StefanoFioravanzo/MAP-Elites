## MAP-Elites for Constrained Optimization

Project structure:

- `/map_elites`:
	- `ea_operators.py`: implementations of crossover/mutation operators
	- `mapelites.py`: main class implementing MapElites. It is an abstract class that requires the implementation of the specific use-case
	- `plot_utils.py`: some functions to plot heatmaps (map elites feature dimension visualizations). At the momento the algorithm is using the *seaborn* based function to plot the heatmap, using a logarithm scale to display the colors.
- `config.ini`: configuration parameters of the MapElites algorithm
- `functions.py`: helper file implementing continuous functions
- `mapelites_continuous_opt.ipynb`: Jupyter notebook implementing the class `MapElitesContinuousOpt` with subclasses the abstract class `MapElites` to use MAP-Elites with continuous constrained optimization

To play with the algorithm, just change the parameters in the `config.ini` file and run the cells in the Jupyter notebook, or run the `mapelites_continuous_opt.py` script.

## TODO

- [ ] Make heatmap plotting available for > 2 dimensions
- [ ] Add config file (logging.ini) based logging
- [ ] Manage inf values in Bokeh
