import numpy as np
import pandas as pd

def generate_mu_and_Sigma():
    '''This function generates a .csv file that defines the "ground truth" values for :math:`\\mu` and
     :math:`\\Sigma` that we use to generate a set of test data.'''

    mean_alpha = 1865.0  # ppbv
    mean_beta  = 0.8     # ppbv / micro mol m^-2

    # In our model, gamma is a independently estimated daily variance parameter, and is not drawn from the same
    # bivariate normal distribution that alpha and beta are. We will include it in the covariance matrix but set it to
    # have zero covariance with alpha and beta.
    mean_gamma = 10.0  # ppbv

    # Define the mean vector mu
    mu = np.array((mean_alpha, mean_beta, mean_gamma))

    # Give each mean value some variance.
    sigma_alpha = 10.0  # ppbv
    sigma_beta  = 0.2   # ppbv / micro mol m^-2
    sigma_gamma = 3.0   # ppbv

    # Define the covariance matrix Sigma
    sigma = np.array(((sigma_alpha ** 2, -sigma_alpha * sigma_beta * 0.8, sigma_alpha * sigma_gamma * 0),
                    (-sigma_alpha * sigma_beta * 0.8, sigma_beta ** 2, sigma_beta * sigma_gamma * 0),
                    (sigma_alpha * sigma_gamma * 0, sigma_beta * sigma_gamma * 0, sigma_gamma ** 2)))

    np.savetxt("testing/ground_truths/mu.csv", mu, delimiter=",")
    np.savetxt("testing/ground_truths/cov.csv", sigma, delimiter=",")

def generate_alphas_betas_and_gammas(days):
    '''This function generates a .csv file containing a set sample set of values for alpha, beta and gamma for however
    many days we choose. These three parameters can be used to generate a set of fake observations of NO2 and CH4.

    :param days: The number of days we want to generate a set of alpha, beta and gamma for.
    :type days: int
    '''

    # Load in the 'ground truth' values of mu and Sigma.
    mu    = np.loadtxt("testing/ground_truths/mu.csv", delimiter=",")
    sigma = np.loadtxt("testing/ground_truths/cov.csv", delimiter=",")

    # To simulate how we will handle real data, generate a fake date for each day.
    dates = []
    for i in range(days):
        dates.append(str(10000000 + i))

    group_params = {}

    for date in dates:
        group_params[date] = np.random.multivariate_normal(mu, sigma)

    df = pd.DataFrame.from_dict(group_params, orient='index', columns=['alpha', 'beta', 'gamma'])
    df.to_csv('testing/ground_truths/alphas_betas_gammas.csv')