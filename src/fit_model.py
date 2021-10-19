import time
import numpy as np
import cmdstanpy
import pandas as pd
import os
import sys
import glob
import shutil
import datetime as datetime
from   contextlib import contextmanager
from   tqdm import tqdm
from   cmdstanpy import CmdStanModel, set_cmdstan_path
import constants as ct
import tropomi_processing as tp

def install_cmdstan():
    '''After the python package cmdstanpy is downloaded/imported, CmdStan also needs to be installed somewhere (C++ code).
    This only needs to be done once.
    '''

    cmdstanpy.install_cmdstan(ct.CMDSTAN_PATH)

def delete_console_printed_lines(num_lines):
    '''This function deletes the output printed to the console when fitting data-poor days. NOTE: This clears the
    CmdStanPy automatic output of fitting a model, and we call this so that the console doesn't get messy when we fit the
    data-poor days. This is slightly hardcoded and so if there are any automatically generated messages from Stan that
    aren't either "good" messages or notifications about a few divergences, this the console will get messy. E.g., if you
    have high split R-hat, then the script won't print pretty on the screen.

    :param check_for_divergences: A parameter to determine whether to remove an extra few lines.
    :type check_for_divergences: bool
    '''

    # Move up one line, clear current line and leave the cursor at its beginning, 32 times:
    print(''.join(["\033[F\x1b[2K\r"]*(num_lines + 12)))

def set_data_poor_initial_values():
    '''This function sets the initial values for the sampler to something sensible.
    :param run_name: The name of the run.
    :type run_name: string
    '''

    inits = {
             'gamma': np.random.normal(12, 2),
             'epsilon': np.random.normal(0, 1, 5).tolist()
            }
    return inits

def set_data_rich_initial_values(directory):
    #TODO make docstring

    # Open the summary dataframe
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + directory + '/summary.csv')

    # From the way the datasets are created, the number of days in the dataset should be equal to the max Day_ID.
    num_days = max(summary_df.day_id)

    # By inspection we know that these are sensible values to initialise at.
    inits = {'mu': [1860, 600.0],
             'Sigma': [[180, -750.0], [-750.0, 50.0]],
             'gamma': np.random.normal(12, 2, num_days).tolist(),
             'epsilon': np.random.normal(0, 1, (num_days, 2)).tolist()}

    return inits

def write_and_print_data_poor_summary(date_range, elapsed_time, dropout):
    '''This function is for writing a summary of how fitting the model to data-poor days went (and printing it to the
    screen), and can be used for either dropout or full fits.

    :param date_range: The date range of the analysis. Must be of format "%Y%m%d-%Y%m%d".
    :type date_range: str
    :param elapsed_time: The number of seconds it took to fit the model to all data-poor days.
    :type elapsed_time: float
    :param dropout: A Boolean to indicate whether this is for a dropout fit or not.
    :type dropout: bool
    '''

    if not dropout:
        directory = '/outputs/' + date_range + '-data_poor/'
    else:
        directory = '/outputs/' + date_range + '-data_poor/dropout/'

    # Open the diagnostics file for the run.
    diagnostics_df = pd.read_csv(ct.FILE_PREFIX + directory + 'diagnostics.csv')
    series_length  = len(diagnostics_df.max_treedepth.tolist())
    f = open(ct.FILE_PREFIX + directory + "summary.txt", "a")
    f.write("Elapsed time to fit model and save output: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '\n')
    f.write('---------------------------------------------------' + '\n\n')
    f.write('Checking sampler transitions treedepth.\n')
    print('\nChecking sampler transitions treedepth.')
    if diagnostics_df.max_treedepth.tolist() == [True]*series_length:
        f.write('Treedepth satisfactory for all transitions.\n\n')
        print('Treedepth satisfactory for all transitions.\n')
    else:
        f.write('Treedepth not satisfactory for all transitions, inspect diagnostics.csv file.\n\n')
    f.write('Checking sampler transitions for divergences.\n')
    print('Checking sampler transitions for divergences.')
    if diagnostics_df.post_warmup_divergences.tolist() == [0]*series_length:
        f.write('No divergent transitions found.\n\n')
        print('No divergent transitions found.\n')
    else:
        f.write('There were some post-warmup divergent transitions on some days, inspect diagnostics.csv file.\n\n')
        print('There were some post-warmup divergent transitions on some days, inspect diagnostics.csv file.\n')
    f.write('Checking E-BFMI - sampler transitions HMC potential energy.\n')
    print('Checking E-BFMI - sampler transitions HMC potential energy.')
    if diagnostics_df.e_bfmi.tolist() == [True]*series_length:
        f.write('E-BFMI satisfactory.\n\n')
        print('E-BFMI satisfactory.\n')
    else:
        f.write('E-BFMI not satisfactory for all days, inspect diagnostics.csv file.\n\n')
        print('E-BFMI not satisfactory for all days, inspect diagnostics.csv file.\n')
    if diagnostics_df.effective_sample_size.tolist() == [True]*series_length:
        f.write('Effective sample size satisfactory for all parameters on all days.\n\n')
        print('Effective sample size satisfactory for all parameters on all days.\n')
    else:
        f.write('Effective sample size not satisfactory for some/all parameters on some days, inspect diagnostics.csv file.\n\n')
        print('Effective sample size not satisfactory for some parameters on some days, inspect diagnostics.csv file.\n')
    if diagnostics_df.split_rhat.tolist() == [True]*series_length:
        f.write('Split R-hat values satisfactory for all parameters on all days.\n\n')
        print('Split R-hat values satisfactory for all parameters on all days.\n')
    else:
        f.write('Split R-hat values not satisfactory for some/all parameters on some days, inspect diagnostics.csv file.\n\n')
        print('Split R-hat values not satisfactory for some/all parameters on some days, inspect diagnostics.csv file.\n')
    f.close()

def data_poor(date_range, dropout=False):
    '''This function is for fitting the model to all the data poor days in the indicated date range.

    :param

    '''
    #TODO docstring.

    # Record the start time in order to write elapsed time for fitting to the output file.
    start_time = time.time()

    directory = date_range + '-data_poor'
    if dropout:
        directory += '/dropout'

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
    data_poor_summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/summary.csv', header=0, index_col=0)

    # Create the csv to track diagnostic metrics
    metrics_df = pd.DataFrame(columns=['date', 'day_id', 'N', 'max_treedepth', 'post_warmup_divergences', 'e_bfmi', 'effective_sample_size', 'split_rhat'])

    alpha_beta_gamma_df_list = []
    ess_rhat_df_list         = []

    for date in tqdm(data_poor_summary_df.index, desc='Fitting model to data-poor days'):

        day_id = int(data_poor_summary_df.loc[date].day_id)
        N      = int(data_poor_summary_df.loc[date].N)

        if dropout:
            if N < 80: # Implies there are less than 20 observations in the holdout set.
                # Skip doing the dropout analysis for this run as there are not enough data points to
                # do an accurate reduced chi squared fit.
                continue

        # Make the dummy directory in the outputs directory
        try:
            os.makedirs(ct.FILE_PREFIX + '/outputs/' + directory + '/dummy')
        except FileExistsError:
            shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + directory + '/dummy')
            os.makedirs(ct.FILE_PREFIX + '/outputs/' + directory + '/dummy')

        # Create the json data for this particular day and put it in the dummy directory
        tp.prepare_data_poor_dataset_for_cmdstanpy(date_range, date, dropout)

        initial_values = set_data_poor_initial_values()

        model = CmdStanModel(stan_file=ct.FILE_PREFIX + '/models/data_poor.stan')

        fit = model.sample(chains=4, parallel_chains=4,
                           data=ct.FILE_PREFIX + '/' + 'outputs/' + directory + '/dummy/data.json', iter_warmup=500,
                           save_warmup=True,
                           iter_sampling=1000, seed=101, show_progress=False,
                           output_dir=ct.FILE_PREFIX + '/outputs/' + directory + '/dummy',
                           save_diagnostics=True,
                           max_treedepth=12,
                           inits=initial_values)

        day_summary_df = fit.summary()
        num_params     = len(day_summary_df)
        index_list     = [i for i in range(num_params) if any(term in day_summary_df.index[i] for term in ['alpha', 'beta', 'gamma'])]
        reduced_df     = day_summary_df.iloc[index_list]
        reduced_df_renamed_indices = reduced_df.rename(index=lambda s: s + '[' + str(day_id) + ']')
        alpha_beta_gamma_df_list.append(reduced_df_renamed_indices)

        day_ess_rhat_df = pd.DataFrame.from_dict({'date': date, 'day_id': day_id, 'ess': np.min(day_summary_df.N_Eff)})
        ess_rhat_df_list.append(day_ess_rhat_df)

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
        file_prefix = glob.glob(ct.FILE_PREFIX + '/outputs/' + directory + '/dummy/*[0-9]-[4].csv')[0].split('4.csv')[0]

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

        delete_console_printed_lines(len(diagnostic_string.split('\n')))

    del master_csv_1['dummy_data']
    del master_csv_2['dummy_data']
    del master_csv_3['dummy_data']
    del master_csv_4['dummy_data']

    alpha_beta_gamma_df = pd.concat(alpha_beta_gamma_df_list)
    alpha_beta_gamma_df.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/summary.csv')

    ess_rhat_df = pd.concat(ess_rhat_df_list)
    ess_rhat_df.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/ess.csv')

    metrics_df.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/diagnostics.csv', index=False)

    master_csv_1.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/data_poor-' + todays_string + '-1.csv',
                        index=False)
    master_csv_2.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/data_poor-' + todays_string + '-2.csv',
                        index=False)
    master_csv_3.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/data_poor-' + todays_string + '-3.csv',
                        index=False)
    master_csv_4.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/data_poor-' + todays_string + '-4.csv',
                        index=False)

    shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + directory + '/dummy')

    # Record the elapsed time.
    elapsed_time = time.time() - start_time

    write_and_print_data_poor_summary(date_range, elapsed_time, dropout)

def data_rich(date_range, error_type, dropout=False):
    #TODO make docstring

    directory  = date_range

    if error_type == 'averaged':
        model_name = 'data_rich'
        directory += '-data_rich'
    elif error_type == 'individual':
        model_name = 'individual_error'
        directory += '-individual_error'

    if dropout:
        directory += '/dropout'

    # Record the start time in order to write elapsed time for fitting to the output file.
    start_time = time.time()

    # Set the random seed for replicability.
    np.random.seed(101)

    set_cmdstan_path(ct.CMDSTAN_PATH)

    model = CmdStanModel(stan_file=ct.FILE_PREFIX + '/models/' + model_name + '.stan')

    initial_values = set_data_rich_initial_values(directory)

    # Fit the model.
    fit = model.sample(chains=4, parallel_chains=4,
                       data=ct.FILE_PREFIX + '/data/' + directory + '/data.json', iter_warmup=500,
                       save_warmup=True,
                       iter_sampling=1000, seed=101, show_progress=False,
                       output_dir=ct.FILE_PREFIX + '/outputs/' + directory,
                       save_diagnostics=True,
                       max_treedepth=15,
                       inits=initial_values)

    full_summary_df = fit.summary()
    num_params      = len(full_summary_df)

    index_list = [i for i in range(num_params)
                  if any(term in full_summary_df.index[i] for term in ['alpha', 'beta', 'gamma', 'Sigma', 'rho'])]

    reduced_df = full_summary_df.iloc[index_list]
    reduced_df.to_csv(ct.FILE_PREFIX + '/outputs/' + directory + '/summary.csv')

    # Record the elapsed time.
    elapsed_time = time.time() - start_time

    f = open(ct.FILE_PREFIX + "/outputs/" + directory + "/summary.txt", "a")
    f.write("Elapsed time to fit model and save output: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '\n')
    f.write('---------------------------------------------------' + '\n')
    f.write(fit.diagnose())
    f.close()

