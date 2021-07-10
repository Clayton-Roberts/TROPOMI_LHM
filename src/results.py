import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import random

class FittedModel:
    '''This class loads the results of a fitted model and organises the results in a useful way. When initialised it
    needs to be pointed towards the correct subdirectory in the /outputs folder.'''

    def write_reduced_chi_squared_csv(self):
        '''
        This function goes through all days fitted in the model and calculates a reduced chi-squred statistic for that day,
        and then writes it to a .csv file. This is performed only as part of the dropout testing.
        '''

    



    def predict_ch4(self, obs_no2, sigma_N, date=None, day_id=None):
        '''
        This function is for predicting an observed value of CH4 with an associated standard deviation on the estimate.

        :param date: Optional, date that we want to make the prediction on, must be in the format YYYYMMDD.
        :type date: int
        :param obs_no2: The observed value of NO2 in micro mol / m^2
        :type obs_no2: float
        :param sigma_N: The reported error on the observation of NO2, also in micro mol / m^2.
        :type sigma_N: float
        :param day_id: Optional, Day ID that can be optionally provided.
        :type day_id: int
        :return: A value for predicted CH4 and associated uncertainty.
        '''


        # Open the full dataset.csv to see which day id we need.
        if date:
            if 'dropout' in self.run_name:
                dataset_df = pd.read_csv('data/' + self.run_name.split('/')[0] + '/dataset.csv')
            else:
                dataset_df = pd.read_csv('data/' + self.run_name + '/dataset.csv')

            day_id     = dataset_df[dataset_df.Date == date].Day_ID.iloc[0]

        alphas = self.full_trace['alpha.' + str(day_id)]
        betas  = self.full_trace['beta.' + str(day_id)]
        gammas = self.full_trace['gamma.' + str(day_id)]

        num_sims = len(alphas)

        predictions = []

        for i in range(1000):

            index    = random.randint(0, num_sims-1)

            # Randomly choose a "true" value of NO2 from the given observation and uncertainty.
            true_no2 = np.random.normal(obs_no2, sigma_N)

            # Choose a set of model parameters from the full trace (all from the same simulation)
            alpha = alphas[index]
            beta  = betas[index]
            gamma = gammas[index]

            # Predict the "true" value of CH4 given the true value of NO2 and the model parameters.
            predictions.append(np.random.normal(alpha + beta * true_no2, gamma))

        mean_observation = np.mean(predictions)
        standard_deviation = np.std(predictions)

        return mean_observation, standard_deviation


    def display_results(self):
        '''This function displays a pretty table of estimated quantities for all model parameters.'''

        print_list = []

        for parameter in self.parameter_list:
            print_list.append([parameter,
                               "{:.2f}".format(self.mean_values[parameter]),
                               "{:.2f}".format(self.standard_deviations[parameter]),
                               "(" + "{:.2f}".format(self.credible_intervals[parameter][0]) + ", " +
                               "{:.2f}".format(self.credible_intervals[parameter][1]) + ")"])

        print(tabulate(print_list,
                       headers=['Param', 'Mean', 'Std Dev', '95% CI'], tablefmt='orgtbl'))



    def __init__(self, run_name):
        '''Constructor method.'''

        # First need to get the 'identifiers' of the files in the indicated outputs folder.

        output_file_list = os.listdir('outputs/' + run_name)
        for file in output_file_list:
            if 'stderr' or 'stdout' or 'diagnostic' or 'summary' in file:
                output_file_list.remove(file)

        date_time = output_file_list[0].split('-')[1]
        model     = output_file_list[0].split('-')[0]

        # Read in the chains from the .csv files
        chain_1 = pd.read_csv('outputs/' + run_name + '/' + model + '-' + date_time + '-1.csv', comment='#')
        chain_2 = pd.read_csv('outputs/' + run_name + '/' + model + '-' + date_time + '-2.csv', comment='#')
        chain_3 = pd.read_csv('outputs/' + run_name + '/' + model + '-' + date_time + '-3.csv', comment='#')
        chain_4 = pd.read_csv('outputs/' + run_name + '/' + model + '-' + date_time + '-4.csv', comment='#')

        # Make a list of all model parameters
        parameter_list = chain_1.columns

        # Construct the full trace for each model parameter
        full_trace = {}

        for parameter in parameter_list:
            full_trace[parameter] = np.concatenate((chain_1[parameter].array,chain_2[parameter].array,
                                                   chain_3[parameter].array,chain_4[parameter].array))

        # Calculate mean value, standard deviation and 95% CI for all model parameter values
        credible_intervals  = {}
        mean_values         = {}
        standard_deviations = {}
        for parameter in parameter_list:
            credible_intervals[parameter]  = [np.percentile(full_trace[parameter], 2.5),
                                             np.percentile(full_trace[parameter], 97.5)]
            mean_values[parameter]         = np.mean(full_trace[parameter])
            standard_deviations[parameter] = np.std(full_trace[parameter])

        # Assign accessible attributes
        self.full_trace          = full_trace
        self.parameter_list      = parameter_list
        self.credible_intervals  = credible_intervals
        self.mean_values         = mean_values
        self.standard_deviations = standard_deviations
        self.chain_1             = chain_1
        self.chain_2             = chain_2
        self.chain_3             = chain_3
        self.chain_4             = chain_4
        self.run_name            = run_name



