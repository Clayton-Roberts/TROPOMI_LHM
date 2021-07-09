import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def trace(fitted_model, parameter, date=None, compare_to_ground_truth=False):
    """Plot the trace and posterior distribution of a sampled scalar parameter.

    :param fitted_model: The object contained the results from a fitted model.
    :type fitted_model: FittedModel
    :param parameter: The name of the parameter. Must be either 'alpha', 'beta' or 'gamma' with the appropriate date
        argument, or 'Sigma.x.x' or 'mu.x' where x is 1 or 2.
    :type parameter: string
    :param date: Date you want to see parameter of interest for, in format YYYYMMDD. Only needed for parameter =
        'alpha', 'beta' or 'gamma'.
    :type date: int
    :param compare_to_ground_truth: A boolean to include a comparison to the ground truth value for this parameter
        if this model run was a test run with fake data, which is why it defaults to False.
    :type compare_to_ground_truth: Boolean
    """

    if compare_to_ground_truth:
        daily_parameters_df = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name +
                                          '/alphas_betas_gammas.csv', delimiter=",", header=0, index_col=0) # Index by date
        mu_df   = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name + '/mu.csv', header=None)
        cov_df  = pd.read_csv('test_suite/ground_truths/' + fitted_model.run_name + '/cov.csv', header=None)

        if parameter == 'alpha' or parameter == 'beta' or parameter == 'gamma':
            ground_truth = daily_parameters_df.loc[date, parameter]

        if parameter.split('.')[0] == 'Sigma':
            row = int(parameter.split('.')[1]) - 1
            col = int(parameter.split('.')[2]) - 1
            ground_truth = cov_df.iloc[row, col]

        if parameter.split('.')[0] == 'mu':
            row = int(parameter.split('.')[1]) - 1
            ground_truth = mu_df.iloc[row, 0]

    # Create the plots
    plt.subplot(2, 1, 1)
    plt.plot(fitted_model.chain_1[parameter])
    plt.plot(fitted_model.chain_2[parameter])
    plt.plot(fitted_model.chain_3[parameter])
    plt.plot(fitted_model.chain_4[parameter])
    plt.xlabel('Samples')
    plt.ylabel(parameter)
    plt.axhline(fitted_model.mean_values[parameter], color='r', lw=2, linestyle='--')
    if compare_to_ground_truth:
        plt.axhline(ground_truth, color='blue', lw=2, linestyle='--',)
    plt.axhline(fitted_model.credible_intervals[parameter][0], linestyle=':', color='k', alpha=0.2)
    plt.axhline(fitted_model.credible_intervals[parameter][1], linestyle=':', color='k', alpha=0.2)
    plt.title(r'Trace and Posterior Distribution for ' + parameter)

    plt.subplot(2, 1, 2)
    plt.hist(fitted_model.full_trace[parameter], 50, density=True)
    sns.kdeplot(fitted_model.full_trace[parameter], shade=True)
    plt.xlabel(parameter)
    plt.ylabel('Density')
    plt.axvline(fitted_model.mean_values[parameter], color='r', lw=2, linestyle='--', label='Mean value')
    if compare_to_ground_truth:
        plt.axvline(ground_truth, color='blue', lw=2, linestyle='--', label='True value')
    plt.axvline(fitted_model.credible_intervals[parameter][0], linestyle=':', color='k', alpha=0.2, label=r'95% CI')
    plt.axvline(fitted_model.credible_intervals[parameter][1], linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()