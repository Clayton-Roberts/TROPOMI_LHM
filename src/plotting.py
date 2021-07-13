import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
import pandas as pd
import numpy as np

def trace(fitted_model, parameter, date=None, compare_to_ground_truth=False):
    """Plot the trace and posterior distribution of a sampled scalar parameter.

    :param fitted_model: The object contained the results from a fitted model.
    :type fitted_model: FittedModel
    :param parameter: The name of the parameter. Must be one of 'alpha', 'beta' or 'gamma' with the appropriate date
        argument, or 'mu_alpha', 'mu_beta', 'sigma_alpha', 'sigma_beta' or 'rho'.
    :type parameter: string
    :param date: Date you want to see parameter of interest for, in format YYYYMMDD. Only needed for parameter =
        'alpha', 'beta' or 'gamma'.
    :param compare_to_ground_truth: A boolean to include a comparison to the ground truth value for this parameter
        if this model run was a test run with fake data, which is why it defaults to False. Do not use with real data.
    :type compare_to_ground_truth: Boolean
    """

    # Need to get the ground truth value for comparison if we're plotting a test dataset.
    if compare_to_ground_truth:
        daily_parameters_df = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name +
                                          '/alphas_betas_gammas.csv', delimiter=",", header=0, index_col=0) # Index by date
        hyperparameter_truths_df = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name +
                                               '/hyperparameter_truths.csv', delimiter=',', header=0)

        if parameter == 'alpha' or parameter == 'beta' or parameter == 'gamma':
            ground_truth = daily_parameters_df.loc[date, parameter]

        else:
            ground_truth = hyperparameter_truths_df[parameter][0]

    dataset_df = pd.read_csv('data/' + fitted_model.run_name + '/dataset.csv', header=0)
    day_id = {}

    # Humans think in terms of dates, stan thinks in terms of day ids. Need to be able to access parameter.day_id
    # using the passed date.
    for i in range(len(dataset_df.Date)):
        if dataset_df.Date[i] not in day_id.keys():
            day_id[dataset_df.Date[i]] = dataset_df.Day_ID[i]

    # Daily parameter are saved in stan as parameter.day_id .
    if parameter == 'alpha' or parameter == 'beta' or parameter == 'gamma':
        model_key = parameter + '.' + str(day_id[date])
    else:
        model_key = parameter

    # Greek letters for labels
    parameter_symbol = {
        'beta': r"$\beta$",
        'alpha': r"$\alpha$",
        'gamma': r"$\gamma$",
        'mu_alpha': r"$\mu_{\alpha}$",
        'mu_beta': r"$\mu_{\beta}$",
        'sigma_alpha': r"$\sigma_{\alpha}$",
        'sigma_beta': r"$\sigma_{\beta}$",
        'rho': r"$\rho$"
    }

    # Units
    parameter_units = {
        'beta': r"[$\mathrm{ppbv}\,/\,\mu\,\mathrm{mol}\,\mathrm{m}^{-2}$]",
        'alpha': "[ppbv]",
        'gamma': "[ppbv]",
        'mu_alpha': "[ppbv]",
        'mu_beta': r"[$\mathrm{ppbv}\,/\,\mu\,\mathrm{mol}\,\mathrm{m}^{-2}$]",
        'sigma_alpha': "[ppbv]",
        'sigma_beta': r"[$\mathrm{ppbv}\,/\,\mu\,\mathrm{mol}\,\mathrm{m}^{-2}$]",
        'rho': ''
    }


    title = 'Trace and Posterior Distribution for ' + parameter_symbol[parameter]

    if date:
        title += ', ' + str(date)[6:] + '/' + str(date)[4:6] + '/' + str(date)[:4]

    # Create the top panel showing how the chains have mixed.
    plt.subplot(2, 1, 1)
    plt.plot(fitted_model.chain_1[model_key])
    plt.plot(fitted_model.chain_2[model_key])
    plt.plot(fitted_model.chain_3[model_key])
    plt.plot(fitted_model.chain_4[model_key])
    plt.xlabel('Samples')
    plt.ylabel(parameter_symbol[parameter] + ' ' + parameter_units[parameter])
    plt.axhline(fitted_model.mean_values[model_key], color='r', lw=2, linestyle='--')
    if compare_to_ground_truth:
        plt.axhline(ground_truth, color='blue', lw=2, linestyle='--',)
    plt.axhline(fitted_model.credible_intervals[model_key][0], linestyle=':', color='k', alpha=0.2)
    plt.axhline(fitted_model.credible_intervals[model_key][1], linestyle=':', color='k', alpha=0.2)
    plt.title(title)

    # Create the bottom panel showing the distribution of the parameter.
    plt.subplot(2, 1, 2)
    plt.hist(fitted_model.full_trace[model_key], 50, density=True)
    sns.kdeplot(fitted_model.full_trace[model_key], shade=True)
    plt.xlabel(parameter_symbol[parameter] + ' ' + parameter_units[parameter])
    plt.ylabel('Density')
    plt.axvline(fitted_model.mean_values[model_key], color='r', lw=2, linestyle='--', label='Mean value')
    if compare_to_ground_truth:
        plt.axvline(ground_truth, color='blue', lw=2, linestyle='--', label='True value')
    plt.axvline(fitted_model.credible_intervals[model_key][0], linestyle=':', color='k', alpha=0.2, label=r'95% CI')
    plt.axvline(fitted_model.credible_intervals[model_key][1], linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()

def observations_scatterplot(date, run_name, compare_to_ground_truth=False):
    '''
    This function is for plotting a simple scatterplot of observations. It will always include errorbars on the observations,
    and if it is a test dataset than you can select to have the latent values shown as well.

    :param date: Date of observations that you want to plot, formatted YYYYMMDD
    :type date: int
    :param run_name: Name of the model run.
    :type run_name: string
    :param compare_to_ground_truth: A boolean to include a comparison to the latent values if this
        model run was a test run with fake data, which is why it defaults to False. Do not use with real data.
    :type compare_to_ground_truth: Boolean
    :return:
    '''

    sns.set()
    np.random.seed(101)

    dataset_df = pd.read_csv('data/' + run_name + '/dataset.csv', header=0)
    date_df = dataset_df[dataset_df.Date == date]

    if compare_to_ground_truth:
        plt.scatter(date_df.true_NO2, date_df.true_CH4, s=12, c='lime', edgecolors='black',
                    marker='D', label="Latent values", zorder=2)

    plt.errorbar(date_df.obs_NO2, date_df.obs_CH4, yerr=date_df.sigma_C, xerr=date_df.sigma_N,
                 ecolor="blue",
                 capsize=3,
                 fmt='D',
                 mfc='w',
                 color='red',
                 ms=4,
                 zorder=1,
                 label='Observed values')

    plt.xlabel('$\mathrm{NO}_{2}^{\mathrm{obs}}$ [$\mu\mathrm{mol}\,\mathrm{m}^{-2}$]')
    plt.ylabel('$\mathrm{CH}_{4}^{\mathrm{obs}}$ [ppbv]')

    plt.title(str(date)[6:] + '/' + str(date)[4:6] + '/' + str(date)[:4])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def dropout_scatterplot(date, run_name):
    '''
    This function is for plotting a scatterplot showing which observations were dropped out on a given day.
    :param date: Date of observations that you want to plot, formatted YYYYMMDD
    :type date: int
    :param run_name: Name of the model run.
    :type run_name: string
    '''

    sns.set()
    np.random.seed(101)

    full_dropout_df   = pd.read_csv('data/' + run_name + '/dropout/dropout_dataset.csv')
    full_remaining_df = pd.read_csv('data/' + run_name + '/dropout/remaining_dataset.csv')

    date_dropout_df   = full_dropout_df[full_dropout_df.Date == date]
    date_remaining_df = full_remaining_df[full_remaining_df.Date == date]

    plt.scatter(date_dropout_df.obs_NO2, date_dropout_df.obs_CH4, color='red', label='Dropped observations', marker='D', zorder=2)
    plt.scatter(date_remaining_df.obs_NO2, date_remaining_df.obs_CH4, color='black', label='Remaining observations', marker='D', zorder=1)

    plt.xlabel('$\mathrm{NO}_{2}^{\mathrm{obs}}$ [$\mu\mathrm{mol}\,\mathrm{m}^{-2}$]')
    plt.ylabel('$\mathrm{CH}_{4}^{\mathrm{obs}}$ [ppbv]')

    plt.title(str(date)[6:] + '/' + str(date)[4:6] + '/' + str(date)[:4])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def regression_scatterplot(date, fitted_model, compare_to_ground_truth=False):
    '''This function is for plotting a scatterplot of observed values of NO2 and CH4 on a given date.

    :param date: Date of observations that you want to plot, formatted YYYYMMDD
    :type date: int
    :param fitted_model: FittedModel object that contains the results for the day we're interested in
    :type fitted_model: FittedModel
    :param compare_to_ground_truth: A boolean to include a comparison to the ground truth regression if this
        model run was a test run with fake data, which is why it defaults to False. Do not use with real data.
    :type compare_to_ground_truth: Boolean
    '''

    sns.set()
    np.random.seed(101)

    dataset_df = pd.read_csv('data/' + fitted_model.run_name + '/dataset.csv', header=0)
    date_df    = dataset_df[dataset_df.Date == date]

    # Needed to plot the regression lines
    x_min, x_max = np.min(date_df.obs_NO2), np.max(date_df.obs_NO2)
    x_domain = np.linspace(x_min, x_max, 100)

    day_id = date_df.Day_ID.iloc[0]

    # Plot a subset of sampled regression lines
    randomize = np.arange(len(fitted_model.full_trace['alpha.' + str(day_id)]))
    np.random.shuffle(randomize)
    alpha = fitted_model.full_trace['alpha.' + str(day_id)][randomize]
    beta = fitted_model.full_trace['beta.' + str(day_id)][randomize]
    for i in range(500):
        plt.plot(x_domain, alpha[i] + beta[i] * x_domain, color='lightcoral',
                 alpha=0.05, zorder=1)

    if compare_to_ground_truth:
        alphas_betas_gammmas_df = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name +
                                              '/alphas_betas_gammas.csv', header=0, index_col=0) # Index by date

        ground_truth_alpha = alphas_betas_gammmas_df.loc[date, 'alpha']
        ground_truth_beta  = alphas_betas_gammmas_df.loc[date, 'beta']

        plt.plot(x_domain, ground_truth_alpha + ground_truth_beta * x_domain, color='lime', ls='--',
                 label='True line', zorder=3)

    plt.errorbar(date_df.obs_NO2, date_df.obs_CH4, yerr=date_df.sigma_C, xerr=date_df.sigma_N,
                 ecolor="blue",
                 capsize=3,
                 fmt='D',
                 mfc='w',
                 color='red',
                 ms=4,
                 zorder=2)

    plt.xlabel('$\mathrm{NO}_{2}^{\mathrm{obs}}$ [$\mu\mathrm{mol}\,\mathrm{m}^{-2}$]')
    plt.ylabel('$\mathrm{CH}_{4}^{\mathrm{obs}}$ [ppbv]')

    plt.title(str(date)[6:] + '/' + str(date)[4:6] + '/' + str(date)[:4])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Time series plotting function that plots alpha or beta against flare stack data, needs "real" data to develop.
def alpha_beta_scatterplot(fitted_model, compare_to_ground_truth=False):
    '''This function is for plotting estimated values of :math:`\\alpha_d` vs :math:`\\beta_d` for a range of days
    :math:`d`.

    :param fitted_model: The fitted model that contains the range of days you want :math:`\\alpha_d`
        vs :math:`\\beta_d` plotted for.
    :type fitted_model: FittedModel
    :param compare_to_ground_truth: A boolean to include a comparison to the ground truth values of :math:`\\alpha_d`
        and :math:`\\beta_d` if this model run was a test run with fake data, which is why it defaults to False.
        Do not use with real data.
    :type compare_to_ground_truth: Boolean
    '''

    # Create the figure with the specified size
    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)

    # Add the subplot axes
    ax0 = fig.add_subplot(spec[:, :])

    # Add ground truth values if you want to and this is a test dataset.
    if compare_to_ground_truth:

        # Get ground truth values for alpha and beta
        alphas_betas_gammas_df = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name + '/alphas_betas_gammas.csv')

        ground_truth_alphas = alphas_betas_gammas_df.alpha
        ground_truth_betas  = alphas_betas_gammas_df.beta

        ax0.scatter(ground_truth_alphas,
                    ground_truth_betas,
                    color='lime',
                    marker='D',
                    zorder=-1)

        # Get ground truth values for the bivariate distribution.
        hyperparameters_df = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name + '/hyperparameter_truths.csv',
                                         header=0)

        ellipse(hyperparameters_df.rho[0],
                hyperparameters_df.sigma_alpha[0],
                hyperparameters_df.sigma_beta[0],
                hyperparameters_df.mu_alpha[0],
                hyperparameters_df.mu_beta[0],
                ax0,
                n_std=2.0,
                edgecolor='blue',
                facecolor='blue',
                alpha=0.3)

    beta_values       = []
    beta_error_bounds = []

    alpha_values = []
    alpha_error_bounds = []

    for parameter in fitted_model.mean_values.keys():
        if 'beta.' in parameter:
            beta_values.append(fitted_model.mean_values[parameter])
            beta_error_bounds.append([fitted_model.mean_values[parameter] - fitted_model.credible_intervals[parameter][0],
                                      fitted_model.credible_intervals[parameter][1] - fitted_model.mean_values[parameter]])
        elif 'alpha.' in parameter:
            alpha_values.append(fitted_model.mean_values[parameter])
            alpha_error_bounds.append(
                [fitted_model.mean_values[parameter] - fitted_model.credible_intervals[parameter][0],
                 fitted_model.credible_intervals[parameter][1] - fitted_model.mean_values[parameter]])

    ax0.errorbar(alpha_values,
                 beta_values,
                 yerr=np.array(beta_error_bounds).T,
                 xerr=np.array(alpha_error_bounds).T,
                 ecolor="blue",
                 capsize=3,
                 fmt='D',
                 mfc='w',
                 color='red',
                 ms=4)

    pearson = fitted_model.mean_values['rho']

    sigma_alpha = fitted_model.mean_values['sigma_alpha']
    mu_alpha    = fitted_model.mean_values['mu_alpha']

    sigma_beta = fitted_model.mean_values['sigma_beta']
    mu_beta    = fitted_model.mean_values['mu_beta']

    ellipse(pearson,
            sigma_alpha,
            sigma_beta,
            mu_alpha,
            mu_beta,
            ax0,
            n_std=2.0,
            edgecolor='red',
            facecolor='red',
            alpha=0.3)

    ax0.set_ylabel(r'$\beta$ [ppbv/$\mu\mathrm{mol}\,\mathrm{m}^{-2}$]')
    ax0.set_xlabel(r'$\alpha$ [ppbv]')
    plt.title('Test dataset')
    plt.tight_layout()
    plt.show()

def ellipse(correlation_coefficient, sigma_alpha, sigma_beta, mu_alpha, mu_beta, ax, n_std=3.0, facecolor='none', **kwargs):
    '''A function to add an ellipse to scatterplots (used only in plots of :math:`\\alpha_d` vs :math:`\\beta_d`).

    :param correlation_coefficient: The Pearson correlation coefficient between :math:`\\alpha` and :math:`\\beta`,
        estimated in our model as :math:`\\rho`.
    :type correlation_coefficient: float
    :param sigma_alpha: Estimated model parameter :math:`\\sigma_\\alpha`.
    :type sigma_alpha: float
    :param sigma_beta: Estimated model parameter :math:`\\sigma_\\beta`.
    :type sigma_beta: float
    :param mu_alpha: Estimated model parameter :math:`\\mu_\\alpha`.
    :type mu_alpha: float
    :param mu_beta: Estimated model parameter :math:`\\mu_\\beta`.
    :type mu_beta: float
    :param ax: The plot to add the ellipse patch to.
    :type ax: Axes
    :param n_std: The number of standard deviations to scale the ellipse out to, defaults to 3.
    :type n_std: float
    :param facecolor: Color of the ellipse.
    :type facecolor: string
    '''

    ell_radius_x = np.sqrt(1 + correlation_coefficient)
    ell_radius_y = np.sqrt(1 - correlation_coefficient)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    scale_x = sigma_alpha * n_std
    mean_x  = mu_alpha

    scale_y = sigma_beta * n_std
    mean_y  = mu_beta

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def reduced_chi_squared(model_run):
    '''
    This function is for plotting a histogram of reduced :math:`\\chi^2` values for each day that was included in the model
    run when observations were dropped out.
    :param model_run: The name of the model run (must include "/dropout").
    :type model_run: string
    '''

    reduced_chi_square_df = pd.read_csv('outputs/' + model_run + '/dropout/reduced_chi_squared.csv')

    sns.displot(reduced_chi_square_df.Reduced_chi_squared, kde=False)
    plt.xlabel(r'$\chi^2_{\nu}$')
    plt.title('Reduced chi-squared values for 2019')
    plt.show()