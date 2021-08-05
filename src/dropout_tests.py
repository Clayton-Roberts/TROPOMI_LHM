from src import constants as ct
from tqdm import tqdm
import os
import shutil
import pandas as pd
import numpy as np
import csv
import json

def make_directories(run_name):
    '''This function checks that the relevant directory is made in data/run_name for the subdivided dropout datasets.

    :param run_name: The name of this model run.
    :type run_name: string
    '''

    try:
        os.makedirs(ct.FILE_PREFIX + '/data/' + run_name + '/dropout')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/data/' + run_name + '/dropout')
        os.makedirs(ct.FILE_PREFIX + '/data/' + run_name + '/dropout')

    try:
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name + '/dropout')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + run_name + '/dropout')
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name + '/dropout')

def create_csvs(run_name):
    '''This function will open the file named "dataset.csv" located at data/run_name and then drop out 20% of observations
    for each day listed in "dataset.csv". This function then creates two .csv files: one containing the remaining 80%
    of observations, and the other containing the 10% that have been dropped out.

    :param run_name: The name of this model run.
    :type run_name: string
    '''

    full_dataset_df = pd.read_csv(ct.FILE_PREFIX + '/data/'+ run_name + '/dataset.csv')

    day_ids = set(full_dataset_df.Day_ID)

    dropout_dfs   = []
    remaining_dfs = []

    for day_id in tqdm(day_ids, desc='Splitting data-rich days into holdout sets'):

        day_df = full_dataset_df[full_dataset_df.Day_ID == day_id]

        total_observations            = len(day_df.obs_NO2)
        number_dropout_observations   = int(total_observations*0.2)

        randomize = np.arange(total_observations)
        np.random.shuffle(randomize)
        dropout_set   = randomize[:number_dropout_observations]
        remaining_set = randomize[number_dropout_observations:]

        dropout_df   = day_df.iloc[dropout_set,:]
        remaining_df = day_df.iloc[remaining_set,:]

        dropout_dfs.append(dropout_df)
        remaining_dfs.append(remaining_df)

    dropout_df = pd.concat(dropout_dfs)
    dropout_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/dropout_dataset.csv', index=False)

    remaining_df = pd.concat(remaining_dfs)
    remaining_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/remaining_dataset.csv', index=False)

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
    day_id = list(df.Day_ID)
    D = int(np.max(day_id))
    M = len(obs_no2)

    group_sizes = []
    for i in range(D):
        day = i + 1
        size = len(df[df.Day_ID == day])
        group_sizes.append(size)

    avg_sigma_N = []
    avg_sigma_C = []

    for i in range(D):
        day = i + 1
        mean_sigma_N = round(np.mean(df[df.Day_ID == day].sigma_N), 2)
        mean_sigma_C = round(np.mean(df[df.Day_ID == day].sigma_C), 2)

        avg_sigma_N.append(mean_sigma_N)
        avg_sigma_C.append(mean_sigma_C)

    sigma_N = list(df.sigma_N)
    sigma_C = list(df.sigma_C)

    data = {}
    data['M'] = M
    data['D'] = D
    data['day_id'] = day_id
    data['group_sizes'] = group_sizes
    data['NO2_obs'] = obs_no2
    data['CH4_obs'] = obs_ch4
    if 'daily_mean_error' in run_name:
        data['sigma_N'] = avg_sigma_N
        data['sigma_C'] = avg_sigma_C
    else:
        data['sigma_N'] = sigma_N
        data['sigma_C'] = sigma_C

    with open(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/data.json', 'w') as outfile:
        json.dump(data, outfile)




