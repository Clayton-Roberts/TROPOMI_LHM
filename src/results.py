import constants as ct
import pandas as pd
import numpy as np
import glob
from tabulate import tabulate
import random
from scipy.stats import norm, mode
from tqdm import tqdm


class FittedResults:
    '''This class loads the results of a fitted model and organises the results in a useful way. When initialised it
    needs to be pointed towards the correct subdirectory in the /outputs folder.'''

    #TODO this is used
    def elpd_i(self):
        '''This function calculates the expected log pointwise predictive density, per observation. Doing this as a
        function so that we don't have to wait for it to be done every single time a fitted model is loaded into this
        class. To get the actual computed elpd, you need to sum every element in the returned list. Returning as a list
        to aid in model comparison when we have to calculate the standard error of the difference between the elpd of two
        different models.'''

        dataset_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + self.run_name.split('/')[0] + '/dataset.csv')

        elpd_i = []

        # Total number of observations
        M = len(dataset_df)

        # Number of random draws to use, 4000 is the max. Might need to be smaller if this is taking a while.
        S = 1000

        for i in tqdm(range(M), desc='Iterating over all observations'):
            day_id     = dataset_df.day_id[i]

            obs_no2    = dataset_df.obs_NO2[i]
            obs_ch4    = dataset_df.obs_CH4[i]
            sigma_N    = dataset_df.sigma_N[i]
            sigma_C    = dataset_df.sigma_C[i]

            alphas     = self.full_trace['alpha.' + str(day_id)]
            betas      = self.full_trace['beta.' + str(day_id)]
            gammas     = self.full_trace['gamma.' + str(day_id)]

            num_draws  = len(alphas)

            ith_likelihood_s = []
            a                = []

            for s in range(S):
                index = random.randint(0, num_draws-1)

                # Choose a set of model parameters from the full trace (all from the same simulation)
                alpha = alphas[index]
                beta  = betas[index]
                gamma = gammas[index]

                loc   = alpha + beta * obs_no2
                scale = np.sqrt(gamma**2 + sigma_C**2 + (beta**2 * sigma_N**2))

                likelihood_s = norm.pdf(obs_ch4, loc=loc, scale=scale)

                ith_likelihood_s.append(likelihood_s)

                a.append(np.log(likelihood_s))

            lpd_i = np.log((1./S) * sum(ith_likelihood_s))

            a_bar = np.mean(a)

            p_waic_i = (1./(S-1.)) * sum((a_s - a_bar)**2 for a_s in a)

            elpd_i.append(lpd_i - p_waic_i)

        return elpd_i

    #TODO this is used
    def write_residuals_csv(self):
        '''This function goes through all days fitted in the model and calculates the residual between what the model
        predicts and what the "actual" methane pixel value is. This is performed only as part of the dropout testing.
        '''

        start_date, end_date, model = self.run_name.split('/')[0].split('-')

        day_type = '-'.join(model.split('_'))

        # Open the dropout_dataset.csv file
        dropout_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + self.run_name + '/dropout_dataset.csv')

        # Open the dropout summary csv file, index by day_id
        summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + self.run_name + '/summary.csv', index_col=1)

        # Empty list to hold daily frames for later concatenation
        daily_dfs = []

        for day_id in tqdm(set(dropout_df.day_id), desc='Writing ' + day_type + ' residuals'):

            # This will always be true for data-rich days
            if summary_df.loc[day_id].N < 80:
                continue

            dropout_day_df = dropout_df[dropout_df.day_id == day_id]

            date = dropout_day_df.date.iloc[0]

            obs_CH4 = list(dropout_day_df.obs_CH4)
            obs_NO2 = list(dropout_day_df.obs_NO2)
            sigma_N = list(dropout_day_df.sigma_N)

            predictions = []
            residuals   = []

            for i in range(len(obs_NO2)):
                prediction, uncertainty = self.predict_ch4(obs_NO2[i], sigma_N[i], day_id)
                predictions.append(round(prediction, 2))
                residuals.append(round(prediction - obs_CH4[i], 2))

            day_df = pd.DataFrame(list(zip([day_id]*len(obs_NO2),
                                           [date]*len(obs_NO2),
                                           predictions,
                                           obs_CH4,
                                           residuals)),
                                  columns=('day_id', 'date', 'predicted_value', 'actual_value', 'residual'))

            daily_dfs.append(day_df)

        residuals_df = pd.concat(daily_dfs)
        residuals_df.to_csv(ct.FILE_PREFIX + '/outputs/' + self.run_name + '/residuals.csv', index=False)

    #TODO this is used
    def write_reduced_chi_squared_csv(self):
        '''
        This function goes through all days fitted in the model and calculates a reduced chi-squred statistic for that day,
        and then writes it to a .csv file. This is performed only as part of the dropout testing.
        '''

        start_date, end_date, model = self.run_name.split('/')[0].split('-')

        day_type = '-'.join(model.split('_'))

        # Open the dropout_dataset.csv file
        dropout_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + self.run_name + '/dropout_dataset.csv')

        # Open the summary file, index by day_id
        original_summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + self.run_name.split('/')[0] + '/summary.csv', index_col=1)


        # Create dataframe for reduced chi-squared calculations
        reduced_chi_squared_df = pd.DataFrame(columns=('date', 'day_id', 'reduced_chi_squared', 'N_holdout_observations'))

        for day_id in tqdm(set(dropout_df.day_id), desc='Writing ' + day_type + ' reduced chi-squared results'):

            # This condition is always true for data-rich days, and we only want to calculate reduced
            # chi squared for data poor days with N >= 100
            if original_summary_df.loc[day_id].N < 100:
                continue

            dropout_day_df = dropout_df[dropout_df.day_id == day_id]

            date = dropout_day_df.date.iloc[0]

            obs_CH4 = list(dropout_day_df.obs_CH4)
            sigma_C = list(dropout_day_df.sigma_C)
            obs_NO2 = list(dropout_day_df.obs_NO2)
            sigma_N = list(dropout_day_df.sigma_N)

            n_Observations  = len(obs_CH4)
            degrees_freedom = int(n_Observations - 8)

            pred_CH4       = []
            sigma_pred_CH4 = []

            for i in range(len(obs_NO2)):
                prediction, uncertainty = self.predict_ch4(obs_NO2[i], sigma_N[i], day_id)
                pred_CH4.append(prediction)
                sigma_pred_CH4.append(uncertainty)

            chi_squared = np.sum(
                [(o_i - p_i) ** 2 / (sigma_o_i ** 2 + sigma_p_i ** 2) for o_i, p_i, sigma_o_i, sigma_p_i in
                 zip(obs_CH4, pred_CH4, sigma_C, sigma_pred_CH4)])

            reduced_chi_squared = chi_squared / degrees_freedom

            reduced_chi_squared_df = reduced_chi_squared_df.append({'date': date,
                                                                    'day_id': day_id,
                                                                    'reduced_chi_squared': round(reduced_chi_squared, 2),
                                                                    'N_holdout_observations': n_Observations},
                                                                   ignore_index=True)

            reduced_chi_squared_df = reduced_chi_squared_df.sort_values(by='date')
            reduced_chi_squared_df.to_csv(ct.FILE_PREFIX + '/outputs/' + self.run_name + '/reduced_chi_squared.csv',
                                          index=False)

    #TODO this is used
    def predict_ch4(self, obs_no2, sigma_N, day_id):
        '''
        This function is for predicting an observed value of CH4 with an associated standard deviation on the estimate.

        :param obs_no2: The observed value of NO2 in micro mol / m^2
        :type obs_no2: float
        :param sigma_N: The reported error on the observation of NO2, also in micro mol / m^2.
        :type sigma_N: float
        :param day_id: Day_ID for the date in question.
        :type day_id: int
        :return: A value for predicted CH4 and associated uncertainty.
        '''

        alphas = self.full_trace['alpha.' + str(day_id)]
        betas  = self.full_trace['beta.' + str(day_id)]
        gammas = self.full_trace['gamma.' + str(day_id)]

        num_draws = len(alphas)

        predictions = []

        for i in range(1000):

            index    = random.randint(0, num_draws-1)

            # Choose a set of model parameters from the full trace (all from the same simulation)
            alpha = alphas[index]
            beta  = betas[index]
            gamma = gammas[index]

            # Predict the "true" value of CH4 given the true value of NO2 and the model parameters, i.e., don't
            # include anything to do with sigma_C in the variance term.
            predictions.append(np.random.normal(alpha + beta * obs_no2, np.sqrt(gamma**2 + (beta**2 * sigma_N**2))))

        mean_observation   = np.mean(predictions)
        standard_deviation = np.std(predictions)

        return mean_observation, standard_deviation

    #TODO this is used
    def __init__(self, run_name):
        '''Constructor method.'''

        # First need to get the 'identifiers' of the files in the indicated outputs folder.

        output_file_list = [file.split('/')[-1] for file in
                                  glob.glob(
                                      ct.FILE_PREFIX + '/outputs/' + run_name + '/*[0-9]-[0-9].csv')]

        date_time = output_file_list[0].split('-')[1]
        model     = output_file_list[0].split('-')[0]

        # Read in the chains from the .csv files
        nuts_chain_1 = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model + '-' + date_time + '-1.csv', comment='#')
        nuts_chain_2 = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model + '-' + date_time + '-2.csv', comment='#')
        nuts_chain_3 = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model + '-' + date_time + '-3.csv', comment='#')
        nuts_chain_4 = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/' + model + '-' + date_time + '-4.csv', comment='#')

        # Separate the chains into warmup draws and sampler draws.
        warmup_draws_chain_1 = nuts_chain_1.head(500)
        warmup_draws_chain_2 = nuts_chain_2.head(500)
        warmup_draws_chain_3 = nuts_chain_3.head(500)
        warmup_draws_chain_4 = nuts_chain_4.head(500)

        sample_draws_chain_1 = nuts_chain_1.tail(1000)
        sample_draws_chain_2 = nuts_chain_2.tail(1000)
        sample_draws_chain_3 = nuts_chain_3.tail(1000)
        sample_draws_chain_4 = nuts_chain_4.tail(1000)

        # Make a list of all model parameters
        parameter_list = nuts_chain_1.columns

        # Construct a dictionary containing all the draws, but separated into useful keys.
        draws = {'warmup': {'chain_1': {},
                            'chain_2': {},
                            'chain_3': {},
                            'chain_4': {}
                            },
                 'sampled': {'chain_1': {},
                             'chain_2': {},
                             'chain_3': {},
                             'chain_4': {}
                             }
                 }

        # Construct the full trace for each model parameter. Full trace will only include sample draws, NO WARMUP DRAWS.
        full_trace = {}

        for parameter in parameter_list:
            draws['warmup']['chain_1'][parameter] = warmup_draws_chain_1[parameter]
            draws['warmup']['chain_2'][parameter] = warmup_draws_chain_2[parameter]
            draws['warmup']['chain_3'][parameter] = warmup_draws_chain_3[parameter]
            draws['warmup']['chain_4'][parameter] = warmup_draws_chain_4[parameter]

            draws['sampled']['chain_1'][parameter] = sample_draws_chain_1[parameter]
            draws['sampled']['chain_2'][parameter] = sample_draws_chain_2[parameter]
            draws['sampled']['chain_3'][parameter] = sample_draws_chain_3[parameter]
            draws['sampled']['chain_4'][parameter] = sample_draws_chain_4[parameter]

            full_trace[parameter] = np.concatenate((sample_draws_chain_1[parameter].array,
                                                    sample_draws_chain_2[parameter].array,
                                                    sample_draws_chain_3[parameter].array,
                                                    sample_draws_chain_4[parameter].array))

        # Calculate median value, standard deviation and 95% CI for all model parameter values
        credible_intervals  = {}
        median_values       = {}
        for parameter in parameter_list:
            credible_intervals[parameter]  = [np.percentile(full_trace[parameter], 16),
                                              np.percentile(full_trace[parameter], 84)]
            # Calculate the median
            median_values[parameter]         = np.median(full_trace[parameter])

        # Assign accessible attributes
        self.full_trace         = full_trace
        self.parameter_list     = parameter_list
        self.credible_intervals = credible_intervals
        self.median_values        = median_values
        self.draws              = draws
        self.run_name           = run_name



