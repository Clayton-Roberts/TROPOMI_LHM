# This file is going to be the script that is run from the command line to replicate the work of the paper.
# It will import other modules of code, make directories and will write various outputs.
# It will take in a date range as two command line arguments to indicate the date range you want to run the
# analysis for.
from datetime import datetime
import sys
import dropout_tests
import fit_model
import results
import tropomi_processing
import model_comparison

# Fourth, fit the individual error model for month of January.
# Fifth, check if the data-rich model has been run for month of January, run it if it doesn't exist yet.
# Sixth, do the comparison between the two models.
# 7th, print all outputs to screen and write a single massive output file, that goes through all metrics, and quantities that went into the paper.
# 8th, make the figures and save them to a folder in the figures folder.

start_datetime = datetime.strptime('2019-01-01', '%Y-%m-%d')
end_datetime   = datetime.strptime('2019-11-30', '%Y-%m-%d')

print('=========================================================\n')
print('Analysis date range: ' +
      start_datetime.strftime('%B %-d, %Y') +
      ' - ' +
      end_datetime.strftime('%B %-d, %Y') )
print('\n=========================================================\n')
date_range = start_datetime.strftime('%Y%m%d') + '-' + end_datetime.strftime('%Y%m%d')
#tropomi_processing.make_all_directories(date_range)
print('Created the following directories:\n')
print('    [1] data/' + date_range + '-data_rich')
print('    [2] data/' + date_range + '-data_poor')
print('    [3] outputs/' + date_range + '-data_rich')
print('    [4] outputs/' + date_range + '-data_poor')
print('\n=========================================================\n')
#tropomi_processing.convert_TROPOMI_observations_to_csvs(date_range)
print('\n=========================================================\n')
#dropout_tests.make_all_directories(date_range)
print('Created the following directories:\n')
print('    [1] data/' + date_range + '-data_rich/dropout')
print('    [2] data/' + date_range + '-data_poor/dropout')
print('    [3] outputs/' + date_range + '-data_rich/dropout')
print('    [4] outputs/' + date_range + '-data_poor/dropout')
print('\n=========================================================\n')
#dropout_tests.create_csvs(date_range + '-data_rich')
#dropout_tests.create_csvs(date_range + '-data_poor')
print('\n=========================================================\n')
#dropout_tests.prepare_dataset_for_cmdstanpy(date_range + '-data_rich')
print('Created the following files:\n')
print('    [1] data/' + date_range + '-data_rich/dropout/data.json')
print('\n=========================================================\n')
print('Fitting model to 80% of observations on data-rich days.')
print('This may take some time, please be patient.\n')
#fit_model.nuts('data/' + date_range + '-data_rich/dropout/data.json',
#               'models/data_rich.stan',
#               date_range + '-data_rich/dropout')
print('\nCreated the following files:\n')
print('    [1] outputs/' + date_range + '-data_rich/dropout/summary.txt')
print('\n=========================================================\n')
#fitted_results = results.FittedResults(date_range + '-data_rich/dropout')
#fitted_results.write_reduced_chi_squared_csv()
#fitted_results.write_residuals_csv()
print('\nCreated the following files:\n')
print('    [1] outputs/' + date_range + '-data_rich/dropout/reduced_chi_squared.csv')
print('    [2] outputs/' + date_range + '-data_rich/dropout/residuals.csv')
print('\n=========================================================\n')
#tropomi_processing.prepare_data_rich_dataset_for_cmdstanpy(date_range + '-data_rich')
print('Created the following files:\n')
print('    [1] data/' + date_range + '-data_rich/data.json')
print('\n=========================================================\n')
print('Fitting model to 100% of observations on data-rich days.')
print('This may take some time, please be patient.\n')
#fit_model.nuts('data/' + date_range + '-data_rich/data.json',
#               'models/data_rich.stan',
#               date_range + '-data_rich')
print('\nCreated the following files:\n')
print('    [1] outputs/' + date_range + '-data_rich/summary.txt')
print('\n=========================================================\n')
print('Fitting model to 80% of observations on data-poor days.')
print('This may take some time, please be patient.\n')
#fit_model.fit_data_poor_days(date_range + '-data_poor/dropout')
print('Created the following files:\n')
print('    [1] outputs/' + date_range + '-data_poor/dropout/summary.txt')
print('\n=========================================================\n')
#fitted_results = results.FittedResults(date_range + '-data_poor/dropout')
#fitted_results.write_reduced_chi_squared_csv()
#fitted_results.write_residuals_csv()
print('\nCreated the following files:\n')
print('    [1] outputs/' + date_range + '-data_poor/dropout/reduced_chi_squared.csv')
print('    [2] outputs/' + date_range + '-data_poor/dropout/residuals.csv')
print('\n=========================================================\n')
print('Fitting model to 100% of observations on data-poor days.')
print('This may take some time, please be patient.\n')
#fit_model.fit_data_poor_days(date_range + '-data_poor')
print('Created the following files:\n')
print('    [1] outputs/' + date_range + '-data_poor/summary.txt')
print('\n=========================================================\n')
#tropomi_processing.make_directories('20190101-20190131-data_rich')
#tropomi_processing.make_directories('20190101-20190131-individual_error')
print('Created the following directories:\n')
print('    [1] data/20190101-20190131-data_rich')
print('    [2] outputs/20190101-20190131-data_rich')
print('    [3] data/20190101-20190131-individual_error')
print('    [4] outputs/20190101-20190131-invidual_error')
print('\n=========================================================\n')
#tropomi_processing.copy_data()
#tropomi_processing.prepare_data_rich_dataset_for_cmdstanpy('20190101-20190131-data_rich')
#tropomi_processing.prepare_data_rich_dataset_for_cmdstanpy('20190101-20190131-individual_error')
print('Created the following files:\n')
print('    [1] data/20190101-20190131-data_rich/summary.csv')
print('    [2] data/20190101-20190131-data_rich/dataset.csv')
print('    [3] data/20190101-20190131-data_rich/data.json')
print('    [4] data/20190101-20190131-individual_error/summary.csv')
print('    [5] data/20190101-20190131-individual_error/dataset.csv')
print('    [6] data/20190101-20190131-individual_error/data.json')
print('\n=========================================================\n')
print('Fitting data-rich model to 100% of observations on data-rich days in January 2019.')
print('This may take some time, please be patient.\n')
#fit_model.nuts('data/20190101-20190131-data_rich/data.json',
#               'models/data_rich.stan',
#               '20190101-20190131-data_rich')
print('Created the following files:\n')
print('    [1] outputs/20190101-20190131-data_rich/summary.txt')
print('\n=========================================================\n')
print('Fitting individual-error model to 100% of observations on data-rich days in January 2019.')
print('This may take some time, please be patient.\n')
# fit_model.nuts('data/20190101-20190131-individual_error/data.json',
#                'models/individual_error.stan',
#                '20190101-20190131-individual_error')
print('Created the following files:\n')
print('    [1] outputs/20190101-20190131-individual_error/summary.txt')
print('\n=========================================================\n')
#individual_error_fitted_model = results.FittedResults('20190101-20190131-individual_error')
#daily_mean_error_fitted_model = results.FittedResults('20190101-20190131-data_rich')
#model_comparison.compare_models(individual_error_fitted_model, daily_mean_error_fitted_model)
print('\nCreated the following files:\n')
print('     [1] outputs/20190101-20190131-data_rich/model_comparison.txt')
print('     [2] outputs/20190101-20190131-individual_error/model_comparison.txt')
print('\n=========================================================\n')
print('Adding predictions from the model for all days and locations possible.')
print('This will take some time, please be patient.\n')
results = results.FittedResults(date_range + '-data_rich')
tropomi_processing.add_predictions(results)
tropomi_processing.add_dry_air_column_densities(results)
tropomi_processing.calculate_dry_air_column_density_residuals(results)
print('\n')
results = results.FittedResults(date_range + '-data_poor')
tropomi_processing.add_predictions(results)
tropomi_processing.add_dry_air_column_densities(results)
tropomi_processing.calculate_dry_air_column_density_residuals(results)



