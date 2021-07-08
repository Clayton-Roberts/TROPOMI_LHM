import numpy as np
import pandas as pd
import csv
import json
import os
import shutil

def make_directories(run_name):
    '''This function checks that all the relevant directories are made. If you re-use a run name, the directory and
    previous contents will be deleted, and the directory will be created again WITHOUT WARNING.

    :param run_name: The name of the run, will also be the name of the folders that is created.
    :type run_name string
    '''

    try:
        os.mkdir('test_suite/data/' + run_name)
    except FileExistsError:
        shutil.rmtree('test_suite/data/' + run_name)
        os.mkdir('test_suite/data/' + run_name)

    try:
        os.mkdir('test_suite/ground_truths/' + run_name)
    except FileExistsError:
        shutil.rmtree('test_suite/ground_truths/' + run_name)
        os.mkdir('test_suite/ground_truths/' + run_name)

    try:
        os.mkdir('outputs/' + run_name)
    except FileExistsError:
        shutil.rmtree('outputs/' + run_name)
        os.mkdir('outputs/' + run_name)

def generate_mu_and_Sigma(run_name):
    '''This function generates a .csv file that defines the "ground truth" values for :math:`\\mu` and
     :math:`\\Sigma` that we use to generate a set of test data.

     :param run_name: The name of the run.
     :type run_name: string
     '''

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

    np.savetxt("test_suite/ground_truths/" + run_name + "/mu.csv", mu, delimiter=",")
    np.savetxt("test_suite/ground_truths/" + run_name + "/cov.csv", sigma, delimiter=",")

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
    mu    = np.loadtxt("test_suite/ground_truths/" + run_name + "/mu.csv", delimiter=",")
    sigma = np.loadtxt("test_suite/ground_truths/" + run_name + "/cov.csv", delimiter=",")

    # To simulate how we will handle real data, generate a fake date for each day.
    dates = []
    for i in range(days):
        dates.append(str(10000000 + i))

    group_params = {}

    for date in dates:
        group_params[date] = np.random.multivariate_normal(mu, sigma)

    df = pd.DataFrame.from_dict(group_params, orient='index', columns=['alpha', 'beta', 'gamma'])
    df.to_csv('test_suite/ground_truths/' + run_name + '/alphas_betas_gammas.csv')

def generate_dataset(run_name):
    '''This function generates a .csv file that contains the total set of fake observations over all of the fake days
    that we wanted, using the ground truth values of :math:`\\alpha`, :math:`\\beta` and :math:`\\gamma` contained in
    "ground_truths/run_name/alphas_betas_gammas.csv"

    :param run_name: The name of the run.
    :type run_name: string
    '''

    df = pd.read_csv('test_suite/ground_truths/' + run_name + '/alphas_betas_gammas.csv',
                     delimiter=",", header=0, index_col=0)

    with open('test_suite/ground_truths/' + run_name + '/dataset.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(('Day_ID','Date','obs_NO2', 'obs_CH4', 'sigma_N', 'sigma_C', 'true_NO2', 'true_CH4'))

        day_index = 1  # Stan starts counting from 1!

        for date in df.index:

            # Get the true value of alpha, beta and gamma for this date.
            alpha = df.loc[date, 'alpha']
            beta = df.loc[date, 'beta']
            gamma = df.loc[date, 'gamma']

            # Generate 100 values of latent NO2 for each day.
            latent_no2 = np.random.uniform(0.0, 175.0, 100)  # Micro mol / square meter

            # Define our observational errors.
            sigma_N = 7.0  # Micro mol / square meter
            sigma_C = 2.0  # ppbv

            for true_no2 in latent_no2:

                # Generate the observed value of NO2 from the latent value.
                obs_no2  = np.random.normal(true_no2, sigma_N)
                # Generate the latent value of CH4 according to the model equation.
                true_ch4 = np.random.normal(alpha + beta * true_no2, gamma)
                # Generate the observed value of CH4 from the latent value.
                obs_ch4  = np.random.normal(true_ch4, sigma_C)

                # Write to the dataset.
                writer.writerow((day_index, str(date), round(obs_no2, 2), round(obs_ch4, 2), sigma_N, sigma_C,
                                 round(true_no2, 2), round(true_ch4, 2)))

            day_index += 1

def prepare_dataset_for_cmdstanpy(run_name):
    '''This function takes the "dataset.cvs" file located at "test_suite/ground_truths/run_name" and turns it into json
    that is suitable for usage by the cmdstanpy package (.csv files are unabled to be provided as data when we
    fit our models).'''

    df = pd.read_csv('test_suite/ground_truths/' + run_name + '/dataset.csv', delimiter=',',
                     header=0, index_col=1)  # Indexing by Date instead of Day_ID

    obs_no2 = list(df.obs_NO2)
    obs_ch4 = list(df.obs_CH4)
    sigma_N = list(df.sigma_N)
    sigma_C = list(df.sigma_C)
    day_id  = list(df.Day_ID)
    D       = int(np.max(day_id))
    M       = len(obs_no2)

    data = {}
    data['M']       = M
    data['D']       = D
    data['day_id']  = day_id
    data['NO2_obs'] = obs_no2
    data['CH4_obs'] = obs_ch4
    data['sigma_N'] = sigma_N
    data['sigma_C'] = sigma_C

    with open('test_suite/data/' + run_name + '/data.json', 'w') as outfile:
        json.dump(data, outfile)