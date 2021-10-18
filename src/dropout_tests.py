import constants as ct
from tqdm import tqdm
import os
import shutil
import pandas as pd
import numpy as np
import json

#TODO this is used
def make_all_directories(date_range):
    '''This function is for creating all the necessary directories needed to split the TROPOMI observations into holdout
    sets of data for both data-rich and data-poor days.

    :param date_range: The date range of the analysis. Must be a string formatted as "%Y%m%d-%Y%m%d".
    :type date_range: str
    '''

    try:
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich/dropout')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich/dropout')
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich/dropout')
    try:
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/dropout')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/dropout')
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/dropout')

    try:
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich/dropout')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich/dropout')
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich/dropout')
    try:
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor/dropout')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor/dropout')
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor/dropout')

#TODO this is used
def create_csvs(run_name):
    '''This function will open the file named "dataset.csv" located at data/run_name and then drop out 20% of observations
    for each day listed in "dataset.csv". This function then creates two .csv files: one containing the remaining 80%
    of observations, and the other containing the 10% that have been dropped out.

    :param run_name: The name of this model run.
    :type run_name: string
    '''

    start_date, end_date, model = run_name.split('-')

    day_type = '-'.join(model.split('_'))

    full_summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/'+ run_name + '/summary.csv', index_col=1)

    full_dataset_df = pd.read_csv(ct.FILE_PREFIX + '/data/'+ run_name + '/dataset.csv')

    day_ids = set(full_dataset_df.day_id)

    dropout_dfs   = []
    remaining_dfs = []

    # A summary dataframe for overall metrics of this run's remaining data.
    remaining_summary_df = pd.DataFrame(columns=(('date', 'day_id', 'N')))

    for day_id in tqdm(day_ids, desc='Splitting ' + day_type + ' days into holdout sets'):

        date = full_summary_df.loc[day_id].date

        day_df = full_dataset_df[full_dataset_df.day_id == day_id]

        total_observations            = len(day_df.obs_NO2)
        number_dropout_observations   = int(total_observations*0.2)
        number_remaining_observations = int(total_observations - number_dropout_observations)

        randomize = np.arange(total_observations)
        np.random.shuffle(randomize)
        dropout_set   = randomize[:number_dropout_observations]
        remaining_set = randomize[number_dropout_observations:]

        dropout_df   = day_df.iloc[dropout_set,:]
        remaining_df = day_df.iloc[remaining_set,:]

        dropout_dfs.append(dropout_df)
        remaining_dfs.append(remaining_df)

        # Append summary of this day to the summary dataframe.
        remaining_summary_df = remaining_summary_df.append({'date': date,
                                                            'day_id': day_id,
                                                            'N': number_remaining_observations},
                                                           ignore_index=True)
    # Sort the summary dataframe by date.
    remaining_summary_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/summary.csv', index=False)

    dropout_df = pd.concat(dropout_dfs)
    dropout_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/dropout_dataset.csv', index=False)

    remaining_df = pd.concat(remaining_dfs)
    remaining_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/remaining_dataset.csv', index=False)

#TODO this is used
def prepare_dataset_for_cmdstanpy(run_name):
    '''This function takes the "remaining_dataset.csv" file located at "data/run_name/dropout" and turns it into json
    that is suitable for usage by the cmdstanpy package (.csv files are unabled to be provided as data when we
    fit our models).

    :param run_name: Name of the model run.
    :type run_name:str
    '''

    df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/remaining_dataset.csv', delimiter=',',
                     header=0, index_col=1)  # Indexing by Date instead of Day_ID

    obs_no2 = list(df.obs_NO2)
    obs_ch4 = list(df.obs_CH4)
    day_id = list(df.day_id)
    D = int(np.max(day_id))
    M = len(obs_no2)

    group_sizes = []
    for i in range(D):
        day = i + 1
        size = len(df[df.day_id == day])
        group_sizes.append(size)

    avg_sigma_N = []
    avg_sigma_C = []

    for i in range(D):
        day = i + 1
        mean_sigma_N = np.mean(df[df.day_id == day].sigma_N)
        mean_sigma_C = np.mean(df[df.day_id == day].sigma_C)

        avg_sigma_N.append(mean_sigma_N)
        avg_sigma_C.append(mean_sigma_C)

    data = {}
    data['N'] = M
    data['D'] = D
    data['day_id'] = day_id
    data['group_sizes'] = group_sizes
    data['NO2_obs'] = obs_no2
    data['CH4_obs'] = obs_ch4
    data['sigma_N'] = avg_sigma_N
    data['sigma_C'] = avg_sigma_C

    with open(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/data.json', 'w') as outfile:
        json.dump(data, outfile)




