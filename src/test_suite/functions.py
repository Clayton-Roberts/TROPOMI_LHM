import numpy as np
import pandas as pd
import csv
import json
import os
import shutil
from src import constants as ct

def make_directories(run_name):
    '''This function checks that all the relevant directories are made. If you re-use a run name, the directory and
    previous contents will be deleted, and the directory will be created again WITHOUT WARNING.

    :param run_name: The name of the run, will also be the name of the folders that is created.
    :type run_name string
    '''

    try:
        os.makedirs(ct.FILE_PREFIX + '/data/' + run_name)
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/data/' + run_name)
        os.makedirs(ct.FILE_PREFIX + '/data/' + run_name)

    try:
        os.makedirs(ct.FILE_PREFIX + '/test_suite/ground_truths/' + run_name)
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/test_suite/ground_truths/' + run_name)
        os.makedirs(ct.FILE_PREFIX + '/test_suite/ground_truths/' + run_name)

    try:
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name)
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + run_name)
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name)

def generate_mu_and_Sigma(run_name):
    '''This function generates a .csv file that defines the "ground truth" values that we use to generate a set of test data.

     :param run_name: The name of the run.
     :type run_name: string
     '''

    mu_alpha = 1865.0  # ppbv
    mu_beta  = 0.8     # ppbv / micro mol m^-2

    # In our model, gamma is a independently estimated daily variance parameter, and is not drawn from the same
    # bivariate normal distribution that alpha and beta are. We will include it in the covariance matrix but set it to
    # have zero covariance with alpha and beta.
    mu_gamma = 10.0  # ppbv

    # Define the mean vector mu
    mu = np.array((mu_alpha, mu_beta, mu_gamma))

    # Give each mean value some variance.
    sigma_alpha = 10.0  # ppbv
    sigma_beta  = 0.2   # ppbv / micro mol m^-2
    sigma_gamma = 3.0   # ppbv

    # Define the extent of correlation between alpha and beta
    rho = -0.8

    # Define the covariance matrix Sigma
    sigma = np.array(((sigma_alpha ** 2, sigma_alpha * sigma_beta * rho, sigma_alpha * sigma_gamma * 0),
                    (sigma_alpha * sigma_beta * rho, sigma_beta ** 2, sigma_beta * sigma_gamma * 0),
                    (sigma_alpha * sigma_gamma * 0, sigma_beta * sigma_gamma * 0, sigma_gamma ** 2)))

    np.savetxt(ct.FILE_PREFIX + "/test_suite/ground_truths/" + run_name + "/mu.csv", mu, delimiter=",")
    np.savetxt(ct.FILE_PREFIX + "/test_suite/ground_truths/" + run_name + "/cov.csv", sigma, delimiter=",")

    with open(ct.FILE_PREFIX + '/test_suite/ground_truths/' + run_name + '/hyperparameter_truths.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(('rho','mu_alpha','mu_beta','sigma_alpha','sigma_beta'))
        writer.writerow((rho, mu_alpha, mu_beta, sigma_alpha, sigma_beta))

def generate_alphas_betas_and_gammas(days, run_name):
    '''This function generates a .csv file containing a set sample set of ground truth values of :math:`\\alpha`, :math:`\\beta`
    and :math:`\\gamma` for however many days we choose. These three parameters can be used to generate a set of fake
    observations of NO2 and CH4 for the given day.

    :param days: The number of days we want to generate a set of alpha, beta and gamma for.
    :type days: int
    :param run_name: The name of the run.
    :type run_name: string
    '''

    # Load in the 'ground truth' values of mu and Sigma.
    mu    = np.loadtxt(ct.FILE_PREFIX + "/test_suite/ground_truths/" + run_name + "/mu.csv", delimiter=",")
    sigma = np.loadtxt(ct.FILE_PREFIX + "/test_suite/ground_truths/" + run_name + "/cov.csv", delimiter=",")

    # To simulate how we will handle real data, generate a fake date for each day.
    dates = []
    for i in range(days):
        dates.append(str(10000000 + i))

    group_params = {}

    for date in dates:
        group_params[date] = np.random.multivariate_normal(mu, sigma)

    df = pd.DataFrame.from_dict(group_params, orient='index', columns=['alpha', 'beta', 'gamma'])
    df.to_csv(ct.FILE_PREFIX + '/test_suite/ground_truths/' + run_name + '/alphas_betas_gammas.csv')

def generate_dataset(run_name, n_Obs):
    '''This function generates a .csv file that contains the total set of fake observations over all of the fake days
    that we wanted, using the ground truth values of :math:`\\alpha`, :math:`\\beta` and :math:`\\gamma` contained in
    "ground_truths/run_name/alphas_betas_gammas.csv"

    :param run_name: The name of the run.
    :type run_name: string
    :param n_Obs: Number of observations you want on each day.
    :type n_Obs: int
    '''

    df = pd.read_csv(ct.FILE_PREFIX + '/test_suite/ground_truths/' + run_name + '/alphas_betas_gammas.csv',
                     delimiter=",", header=0, index_col=0) # Indexing by date, column 0

    with open(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(('Day_ID','Date','obs_NO2', 'obs_CH4', 'sigma_N', 'sigma_C'))

        day_index = 1  # Stan starts counting from 1!

        for date in df.index:

            # Get the true value of alpha, beta and gamma for this date.
            alpha = df.loc[date, 'alpha']
            beta = df.loc[date, 'beta']
            gamma = df.loc[date, 'gamma']

            # Generate values of latent NO2 for each day.
            obs_no2 = np.random.uniform(0.0, 175.0, n_Obs)  # Micro mol / square meter

            # Define our observational errors.
            mu_sigma_N = 7.0    # Micro mol / square meter
            mu_sigma_C = 2.0    # ppbv
            sigma_sigma_N = 0.3 # Micro mol / square meter
            sigma_sigma_C = 0.3 # ppbv

            for no2 in obs_no2:

                # Generate a value for sigma_N and sigma_C
                sigma_N = np.random.normal(mu_sigma_N, sigma_sigma_N)
                sigma_C = np.random.normal(mu_sigma_C, sigma_sigma_C)

                # Generate the observed value of CH4 according to the model equation.
                obs_ch4 = np.random.normal(alpha + beta * no2, np.sqrt(gamma**2 + sigma_C**2 + (beta**2 * sigma_N**2)))

                # Write to the dataset.
                writer.writerow((day_index, str(date), round(no2, 2), round(obs_ch4, 2), round(sigma_N,2), round(sigma_C,2)))

            day_index += 1

def prepare_dataset_for_cmdstanpy(run_name):
    '''This function takes the "dataset.cvs" file located at "test_suite/ground_truths/run_name" and turns it into json
    that is suitable for usage by the cmdstanpy package (.csv files are unabled to be provided as data when we
    fit our models).

    :param run_name: Name of the model run.
    :type run_name:str
    '''

    df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', delimiter=',',
                     header=0, index_col=1)  # Indexing by Date instead of Day_ID

    obs_no2 = list(df.obs_NO2)
    obs_ch4 = list(df.obs_CH4)
    day_id  = list(df.Day_ID)
    D       = int(np.max(day_id))
    M       = len(obs_no2)

    group_sizes = []
    for i in range(D):
        day = i+1
        size = len(df[df.Day_ID == day])
        group_sizes.append(size)

    avg_sigma_N = []
    avg_sigma_C = []

    for i in range(D):
        day = i+1
        mean_sigma_N = round(np.mean(df[df.Day_ID == day].sigma_N),2)
        mean_sigma_C = round(np.mean(df[df.Day_ID == day].sigma_C),2)

        avg_sigma_N.append(mean_sigma_N)
        avg_sigma_C.append(mean_sigma_C)

    data = {}
    data['M']           = M
    data['D']           = D
    data['day_id']      = day_id
    data['group_sizes'] = group_sizes
    data['NO2_obs']     = obs_no2
    data['CH4_obs']     = obs_ch4
    data['sigma_N']     = avg_sigma_N
    data['sigma_C']     = avg_sigma_C

    with open(ct.FILE_PREFIX + '/data/' + run_name + '/data.json', 'w') as outfile:
        json.dump(data, outfile)

