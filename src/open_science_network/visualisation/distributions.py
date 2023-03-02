import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib import colors


def resize(data, shape):
    """
    Resize the data so that its shape matches the target shape
    :param data: the data to resize
    :param shape: the target shape
    :return: the resized data
    """
    while len(data.shape) != len(shape):
        data = jnp.expand_dims(data, axis=-1)
        axis = len(data.shape) - 1
        data = jnp.repeat(data, shape[axis], axis=axis)
    return data


def draw_chart(axis, data, label, color=None, scatter_plot=False, zorder=2, lw=1, alpha=1, fmt=""):
    """
    Draw a simple or scatter plot representing the data
    :param axis: the axis on which to draw the plot
    :param data: the data to display
    :param label: the corresponding to the data (for the legend)
    :param color: the color of the plot
    :param scatter_plot: a boolean, True if a scatter plot should be used, False otherwise
    :param zorder: the order on which the plot should be display on the z-axis (superposition axis)
    :param lw: the line width (for simple plot)
    :param alpha: the transparency of the line (for simple plot)
    :param fmt: the format of the line to draw (for simple plot)
    :return: the list of lines representing the plotted data
    """

    # Make sure to add a dimension if only one is available
    if len(data.shape) <= 1:
        data = jnp.expand_dims(data, axis=0)

    # Compute the mean and standard deviation of the data
    mean = data.mean(0).squeeze() if len(data.shape) > 2 else data.mean(0)
    standard_dev = 2 * data.std(0).squeeze() if len(data.shape) > 2 else 2 * data.std(0)

    # Compute the length of the time series
    length = len(mean)

    # Draw the simple or scatter plot
    if scatter_plot is False:
        lines = axis.plot(mean, fmt, label=label, lw=lw, color=color, alpha=alpha, zorder=zorder)
        axis.fill_between(range(length), mean + standard_dev, y2=mean - standard_dev, alpha=.2, zorder=zorder - 1)
    else:
        lines = axis.scatter(range(length), mean, c=color, label=label, zorder=zorder)

    return lines


def set_draw_beliefs_default_values(var_1, var_2, measured):
    """
    Set the default values of the parameters of draw beliefs
    :param var_1: the first variable
    :param var_2: the second variable
    :param measured: whether the second variable is a measurement of the first variable
    :return: var_1, var_2, and measured or their default values
    """

    # Set default values of the first variable if it is None
    if var_1 is None:
        var_1 = {
            'soil_biomass': r'Soil Biomass [$kg/m^3$]',
            'plant_height': r'Plant height [$m$]',
            'plant_density': r'Plant density [count/Are]',
            'yield_density': r'Yield [$kg/Are$]',
            'biomass_carbon_per_m2': r'Biomass carbon [$t/Are$]'
        }

    # Set default values of the second variable if it is None
    if var_2 is None:
        var_2 = {
            'survival_probability': r'Survival probability',
            'obs_plant_height': r'Measured height [$m$]',
            'obs_plant_density': r'Measured height [count/Are]',
            'obs_yield_density': r'Measured yield [$kg/Are$]',
            'lai': r'Measured LAI'
        }

    # Set default values of the measured parameter if it is None
    if measured is None:
        measured = [False, True, True, True, False]

    return var_1, var_2, measured


def draw_beliefs(samples, var_1=None, var_2=None, measured=None, fig_size=(16, 12)):
    """
    Display the prior beliefs
    :param samples: samples from the generative model
    :param var_1: a dictionary whose keys/values represent are the names/labels of the first variables to be displayed
    :param var_2: a dictionary whose keys/values represent are the names/labels of the second variables to be displayed
    :param measured: whether the second variable correspond to the measurement of the first variable
    :param fig_size: the size of the figure displayed
    """

    # Set default values of the parameters if parameters are None
    var_1, var_2, measured = set_draw_beliefs_default_values(var_1, var_2, measured)

    # Create the plot that will contain the plots displaying the posterior distributions
    fig, axes = plt.subplots(len(measured), 1, figsize=fig_size, sharex=True)

    # Iterate over all variable and display the subplots
    for i, (name_1, label_1, name_2, label_2, measured) in enumerate(zip(var_1.keys(), var_1.values(), var_2.keys(), var_2.values(), measured)):

        # Check whether to create a twin axis, if the variables relative values are too different
        twin_x = False
        if samples[name_1].max() < samples[name_2].max() * 0.1:
            twin_x = True
        if samples[name_1].max() * 0.1 > samples[name_2].max():
            twin_x = True

        # Make sure both variables have the same shape
        if len(samples[name_1].shape) == 1:
            samples[name_1] = resize(samples[name_1], samples[name_2].shape)
        if len(samples[name_2].shape) == 1:
            samples[name_2] = resize(samples[name_2], samples[name_1].shape)

        # Set the format and color
        v1_color, v1_format = (None, "-") if measured else ('g', "-")
        v2_color, v2_format = (None, "o:") if measured else ('r', "-")

        # Display the graph
        if twin_x:
            lines_1 = draw_chart(axes[i], samples[name_1], label_1, fmt=v1_format, color=v1_color, lw=1.5)
            lines_2 = draw_chart(axes[i].twinx(), samples[name_2], label_2, fmt=v2_format, color=v2_color)
            lines = lines_1 + lines_2
            axes[i].legend(lines, [line.get_label() for line in lines], loc="upper left")
        else:
            draw_chart(axes[i], samples[name_1], label_1, fmt=v1_format, color=v1_color, lw=1.5)
            draw_chart(axes[i], samples[name_2], label_2, fmt=v2_format, color=v2_color, lw=1.5)
            axes[i].legend(loc="upper left")

    return fig


def compare_posteriors(
    is_observed=None, mcmc_samples=None, svi_samples=None, prediction_samples=None,
    labels=None, var_names=None, var_labels=None, fig_size=(16, 10)
):
    """
    Compare the variational posterior with the posterior obtained from Monte Carlo Markov chain
    :param is_observed: mask indicating which data points are observed and which aren't
    :param mcmc_samples: the Monte Carlo Markov chain posterior
    :param svi_samples: the samples from the variational posterior
    :param prediction_samples: the predictive samples
    :param labels: the labels for each inference type, e.g., svi and mcmc, that will be displayed in the legend
    :param var_names: the name of the variables whose posterior should be displayed
    :param var_labels: the label of the variables whose posterior should be displayed
    :param fig_size: the size of the figures to display
    :return: the figure comparing the posteriors
    """

    # Set default values of the parameters if parameters are None
    samples = next(filter(lambda sample: sample is not None, [prediction_samples, svi_samples, mcmc_samples]))
    if labels is None:
        labels = ['mcmc', 'svi', 'prediction', 'measurements']
    if var_names is None:
        var_names = ['soil_biomass', 'plant_density', 'plant_height', 'survival_probability']
    if var_labels is None:
        var_labels = ['Soil biomass', 'Plant density', 'Plant height', 'Survival probability']
    if is_observed is None:
        key = list(samples.keys())[0]
        is_observed = jnp.array([True] * samples[key].shape[0])

    # Create the plot that will contain the plots displaying the posterior distributions
    fig, axes = plt.subplots(len(var_names), 1, figsize=fig_size, sharex=True)

    # Create the color for the prediction samples depending on whether they are observed or not
    color = colors.to_rgba_array(['yellowgreen' if v else 'gray' for v in is_observed])

    # Iterate over all variables whose posterior must be displayed
    for i, (var_name, var_label) in enumerate(zip(var_names, var_labels)):

        # Display the MCMC posterior
        if mcmc_samples is not None:
            draw_chart(axes[i], mcmc_samples[var_name], labels[0], lw=1.5)

        # Display the variational posterior
        if svi_samples is not None:
            draw_chart(axes[i], svi_samples[var_name], labels[1], lw=1.5)

        # Display the predictive posterior
        if prediction_samples is not None:
            draw_chart(axes[i], prediction_samples[var_name], labels[2], lw=1.5, color='red', alpha=0.5)

        # Display all the data points
        if f'obs_{var_name}' in samples.keys():
            draw_chart(axes[i], samples[f'obs_{var_name}'], labels[3], color, True, 3)

        # Add x and y labels
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(var_label)

        # Display the legend
        axes[i].legend(loc="upper left")

    return fig
