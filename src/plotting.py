import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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