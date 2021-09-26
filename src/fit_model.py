import time
import numpy as np
import cmdstanpy
import pandas as pd
import os
import glob
import shutil
import datetime as datetime
from   tqdm import tqdm
from   cmdstanpy import CmdStanModel, set_cmdstan_path
import constants as ct
import tropomi_processing as tp

def install_cmdstan():
    '''After the python package cmdstanpy is downloaded/imported, CmdStan also needs to be installed somewhere (C++ code).
    This only needs to be done once.
    '''

    cmdstanpy.install_cmdstan(ct.CMDSTAN_PATH)

def set_data_poor_initial_values(run_name):
    '''This function sets the initial values for the sampler to something sensible.
    :param run_name: The name of the run.
    :type run_name: string
    '''

    # By inspection we know that these are sensible values to initialise at.
    inits = {
             'gamma': 12,
             'epsilon': np.random.normal(0, 1, 5).tolist()
            }
    return inits

def set_data_rich_initial_values(run_name):
    '''This function sets the initial values for the sampler to something sensible.
    :param run_name: The name of the run.
    :type run_name: string
    '''

    # Open the summary dataframe
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/summary.csv')

    # From the way the datasets are created, the number of days in the dataset should be equal to the max Day_ID.
    num_days = max(summary_df.day_id)

    # By inspection we know that these are sensible values to initialise at.
    inits = {'mu': [1860, 600.0],
             'Sigma': [[180, -750.0], [-750.0, 50.0]],
             'gamma': np.random.normal(12, 2, num_days).tolist(),
             'epsilon': np.random.normal(0, 1, (num_days, 2)).tolist()}

    return inits

def fit_data_poor_days(run_name):
    '''This function is for fitting the non-hierarchical model to all the data poor days.'''
    #TODO docstring.

    if 'dropout' in run_name:
        dropout_run = True
        start_date, end_date, model_name = run_name.split('/')[0].split('-')
    else:
        dropout_run = False
        start_date, end_date, model_name = run_name.split('-')

    # First need to make four empty, dummy csvs in the data poor directory.
    todays_datetime = datetime.datetime.now()
    todays_string   = todays_datetime.date().strftime('%Y%m%d') + todays_datetime.time().strftime('%H%M')

    # Create the "master" csvs containing the traces for the data-poor days
    dummy_data = {'dummy_data': [0]*1500}
    master_csv_1 = pd.DataFrame(data=dummy_data)
    master_csv_2 = pd.DataFrame(data=dummy_data)
    master_csv_3 = pd.DataFrame(data=dummy_data)
    master_csv_4 = pd.DataFrame(data=dummy_data)

    # Read in the summary csv file for the data poor days
    data_poor_summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/summary.csv', header=0, index_col=0)

    # Create the csv to track diagnostic metrics
    metrics_df = pd.DataFrame(columns=['date', 'day_id', 'N', 'max_treedepth', 'post_warmup_divergences', 'e_bfmi', 'effective_sample_size', 'split_rhat'])

    for date in tqdm(data_poor_summary_df.index, desc='Fitting model to data-poor days'):

        day_id = int(data_poor_summary_df.loc[date].day_id)
        N      = int(data_poor_summary_df.loc[date].N)

        if dropout_run:
            if N < 80: # Implies there are less than 20 observations in the holdout set.
                # Skip doing the dropout analysis for this run as there are not enough data points to
                # do an accurate reduced chi squared fit.
                continue

        # Make the dummy directory in the outputs directory
        try:
            os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name + '/dummy')
        except FileExistsError:
            shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + run_name + '/dummy')
            os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name + '/dummy')

        # Create the json data for this particular day and put it in the dummy directory
        tp.prepare_data_poor_dataset_for_cmdstanpy(run_name, date)

        initial_values = set_data_poor_initial_values(run_name)

        model = CmdStanModel(stan_file=ct.FILE_PREFIX + '/models/data_poor.stan')

        fit = model.sample(chains=4, parallel_chains=4,
                           data=ct.FILE_PREFIX + '/' + 'outputs/' + run_name + '/dummy/data.json', iter_warmup=500,
                           save_warmup=True,
                           iter_sampling=1000, seed=101, show_progress=False,
                           output_dir=ct.FILE_PREFIX + '/outputs/' + run_name + '/dummy',
                           save_diagnostics=True,
                           max_treedepth=12,
                           inits=initial_values)

        # Check the diagnostics.
        diagnostic_string = fit.diagnose()
        max_treedepth         = 'Treedepth satisfactory for all transitions.' in diagnostic_string
        if 'No divergent transitions found.' in diagnostic_string:
            post_warmup_divergences = 0
            check_for_divergences   = False
        else:
            # Calculate the number of post-warmup divergences for this model run.
            check_for_divergences = True
        e_bfmi                = 'E-BFMI satisfactory.' in diagnostic_string
        effective_sample_size = 'Effective sample size satisfactory.' in diagnostic_string
        split_rhat            = 'Split R-hat values satisfactory all parameters.' in diagnostic_string

        # Create the file prefix of the outputs of the model fitting
        file_prefix = glob.glob(ct.FILE_PREFIX + '/outputs/' + run_name + '/dummy/*[0-9]-[4].csv')[0].split('4.csv')[0]

        # Open the four csv files of the outputs of the day's model fitting.
        chain_1 = pd.read_csv(file_prefix + '1.csv', comment='#')
        chain_2 = pd.read_csv(file_prefix + '2.csv', comment='#')
        chain_3 = pd.read_csv(file_prefix + '3.csv', comment='#')
        chain_4 = pd.read_csv(file_prefix + '4.csv', comment='#')

        if check_for_divergences:
            chain_1_divergences_array       = np.array(chain_1.divergent__.tail(1000))
            chain_1_post_warmup_divergences = len(chain_1_divergences_array[chain_1_divergences_array != 0])
            chain_2_divergences_array       = np.array(chain_2.divergent__.tail(1000))
            chain_2_post_warmup_divergences = len(chain_2_divergences_array[chain_2_divergences_array != 0])
            chain_3_divergences_array       = np.array(chain_3.divergent__.tail(1000))
            chain_3_post_warmup_divergences = len(chain_3_divergences_array[chain_3_divergences_array != 0])
            chain_4_divergences_array       = np.array(chain_4.divergent__.tail(1000))
            chain_4_post_warmup_divergences = len(chain_4_divergences_array[chain_4_divergences_array != 0])

            post_warmup_divergences = sum([chain_1_post_warmup_divergences,
                                           chain_2_post_warmup_divergences,
                                           chain_3_post_warmup_divergences,
                                           chain_4_post_warmup_divergences])

        # Add param.day_id to each of the "master" csv files
        for param in ['alpha', 'beta', 'gamma']:
            master_csv_1[param + '.' + str(day_id)] = chain_1[param].tolist()
            master_csv_2[param + '.' + str(day_id)] = chain_2[param].tolist()
            master_csv_3[param + '.' + str(day_id)] = chain_3[param].tolist()
            master_csv_4[param + '.' + str(day_id)] = chain_4[param].tolist()

        # Add mu_alpha.day_id and mu_beta.day_id to each of the "master" csv files to compare to the "prior" mu we feed in.
        master_csv_1['mu_alpha.' + str(day_id)] = chain_1['mu.1'].tolist()
        master_csv_2['mu_alpha.' + str(day_id)] = chain_2['mu.1'].tolist()
        master_csv_3['mu_alpha.' + str(day_id)] = chain_3['mu.1'].tolist()
        master_csv_4['mu_alpha.' + str(day_id)] = chain_4['mu.1'].tolist()
        master_csv_1['mu_beta.' + str(day_id)] = chain_1['mu.2'].tolist()
        master_csv_2['mu_beta.' + str(day_id)] = chain_2['mu.2'].tolist()
        master_csv_3['mu_beta.' + str(day_id)] = chain_3['mu.2'].tolist()
        master_csv_4['mu_beta.' + str(day_id)] = chain_4['mu.2'].tolist()

        metrics_df = metrics_df.append({'date': date,
                                        'N': N,
                                        'day_id': day_id,
                                        'max_treedepth': max_treedepth,
                                        'post_warmup_divergences': post_warmup_divergences,
                                        'e_bfmi': e_bfmi,
                                        'effective_sample_size': effective_sample_size,
                                        'split_rhat': split_rhat},
                                       ignore_index=True)

    del master_csv_1['dummy_data']
    del master_csv_2['dummy_data']
    del master_csv_3['dummy_data']
    del master_csv_4['dummy_data']

    metrics_df.to_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/diagnostics.csv', index=False)

    master_csv_1.to_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model_name + '-' + todays_string + '-1.csv',
                        index=False)
    master_csv_2.to_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model_name + '-' + todays_string + '-2.csv',
                        index=False)
    master_csv_3.to_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model_name + '-' + todays_string + '-3.csv',
                        index=False)
    master_csv_4.to_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model_name + '-' + todays_string + '-4.csv',
                        index=False)

    shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + run_name + '/dummy')

def nuts(data_path, model_path, output_directory):
    '''This function will fit a probability model to a set of data and then save outputs that summarise probability
    distributions over the model parameters.

    :param data_path: Location of the file of data to fit the model to, must be a .json file.
    :type data_path: string
    :param model_path: Location of the .stan model file the data will be fit to.
    :type model_path: string
    :param output_directory: Name of folder to be created in /src/outputs where outputs from current run will be saved.
    :type output_directory: string
    :param cmdstan_path: Directory that the CmdStan download is located in.
    :type cmdstan_path: string
    '''

    # Record the start time in order to write elapsed time for fitting to the output file.
    start_time = time.time()

    # Set the random seed for replicability.
    np.random.seed(101)

    set_cmdstan_path(ct.CMDSTAN_PATH)

    model = CmdStanModel(stan_file=ct.FILE_PREFIX + '/' + model_path)

    run_name       = data_path.split('/data.json')[0].split('data/')[-1]

    initial_values = set_data_rich_initial_values(run_name)

    # Fit the model.
    fit = model.sample(chains=4, parallel_chains=4,
                       data=ct.FILE_PREFIX + '/' + data_path, iter_warmup=500,
                       save_warmup=True,
                       iter_sampling=1000, seed=101, show_progress=False,
                       output_dir=ct.FILE_PREFIX + '/outputs/' + output_directory,
                       save_diagnostics=True,
                       max_treedepth=12,
                       inits=initial_values)

    # Record the elapsed time.
    elapsed_time = time.time() - start_time

    f = open(ct.FILE_PREFIX + "/outputs/" + output_directory + "/summary.txt", "a")
    f.write("Elapsed time to fit model and save output: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '\n')
    f.write('---------------------------------------------------' + '\n')
    f.write(fit.diagnose())
    f.close()

