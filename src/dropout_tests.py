import os
import shutil
import pandas as pd
import numpy as np
import csv

def make_directory(run_name):
    '''This function checks that the relevant directory is made in data/run_name for the subdivided dropout datasets.

    :param run_name: The name of this model run.
    :type run_name: string
    '''

    try:
        os.mkdir('data/' + run_name + '/dropout')
    except FileExistsError:
        shutil.rmtree('data/' + run_name + '/dropout')
        os.mkdir('data/' + run_name + '/dropout')

def create_csvs(run_name):
    '''This function will open the file named "dataset.csv" located at data/run_name and then drop out 20% of observations
    for each day listed in "dataset.csv". This function then creates two .csv files: one containing the remaining 80%
    of observations, and the other containing the 10% that have been dropped out.

    :param run_name: The name of this model run.
    :type run_name: string
    '''

    full_dataset_df = pd.read_csv('data/'+ run_name + '/dataset.csv')

    day_ids = set(full_dataset_df.Day_ID)

    # Open and write headers of the two csv files here

    with open('data/' + run_name + '/dropout/dropout_dataset.csv', 'w') as dropout_csvfile, \
            open('data/' + run_name + '/dropout/remaining_dataset.csv', 'w') as remaining_csvfile:
        dropout_writer = csv.writer(dropout_csvfile, delimiter=',')
        remaining_writer = csv.writer(remaining_csvfile, delimiter=',')

        dropout_writer.writerow(np.array(full_dataset_df.columns))
        remaining_writer.writerow(np.array(full_dataset_df.columns))

    dropout_dfs   = []
    remaining_dfs = []

    for day_id in day_ids:

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

    with open('data/' + run_name + '/dropout/dropout_dataset.csv', 'a') as dropout_csvfile, \
            open('data/' + run_name + '/dropout/remaining_dataset.csv', 'a') as remaining_csvfile:

        for i in range(len(dropout_dfs)):
            dropout_dfs[i].to_csv(dropout_csvfile, header=False, index=False)
            remaining_dfs[i].to_csv(remaining_csvfile, header=False, index=False)
