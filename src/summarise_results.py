import os
import pandas as pd
import numpy as np
from tabulate import tabulate

class FittedModel:
    '''This class loads the results of a fitted model and organises the results in a useful way. When initialised it
    needs to be pointed towards the correct subdirectory in the /outputs folder.'''

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



    def __init__(self, results_location):
        '''Constructor method.'''

        # First need to get the 'identifiers' of the files in the indicated outputs folder.

        output_file_list = os.listdir('outputs/' + results_location)
        output_file_list.remove('summary.txt')
        for file in output_file_list:
            if 'stderr' or 'stdout' or 'diagnostic' in file:
                output_file_list.remove(file)

        date_time = output_file_list[0].split('-')[1]
        model     = output_file_list[0].split('-')[0]

        # Read in the chains from the .csv files
        chain_1 = pd.read_csv('outputs/' + results_location +  '/' + model + '-' + date_time + '-1.csv', comment='#')
        chain_2 = pd.read_csv('outputs/' + results_location +  '/' + model + '-' + date_time + '-2.csv', comment='#')
        chain_3 = pd.read_csv('outputs/' + results_location +  '/' + model + '-' + date_time + '-3.csv', comment='#')
        chain_4 = pd.read_csv('outputs/' + results_location +  '/' + model + '-' + date_time + '-4.csv', comment='#')

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



