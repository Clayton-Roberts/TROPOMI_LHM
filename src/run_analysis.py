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

# First, need to make all of the data
# Second, fit the dropout model and run the dropout tests
# Third, fit the full model and write all files
# Fourth, fit the individual error model for month of January.
# Fifth, check if the data-rich model has been run for month of January, run it if it doesn't exist yet.
# Sixth, do the comparison between the two models.
# 7th, print all outputs to screen and write a single massive output file, that goes through all metrics, and quantities that went into the paper.
# 8th, make the figures and save them to a folder in the figures folder.

start_datetime = datetime.strptime(str(sys.argv[1]), '%Y-%m-%d')
end_datetime   = datetime.strptime(str(sys.argv[2]), '%Y-%m-%d')
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
fit_model.fit_data_poor_days(date_range + '-data_poor/dropout')


