from datetime import datetime
import dropout_tests
import fit_model
import results
import tropomi_processing
import model_comparison
import plotting

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
print('Adding predictions from the model for data-rich days.')
print('This will take some time, please be patient.\n')
#fitted_results = results.FittedResults(date_range + '-data_rich')
#tropomi_processing.add_predictions(fited_results)
#tropomi_processing.add_dry_air_column_densities(fitted_results)
#tropomi_processing.calculate_dry_air_column_density_residuals(fitted_results)
print('\nCreated the following files and directories:\n')
print('    [1] augmented_observations/' + date_range + '-data_rich/*.nc')
print('    [2] outputs/' + date_range + '-data_rich/dry_air_column_density_residuals.csv')
print('\n=========================================================\n')
print('Adding predictions from the model for data-poor days.')
print('This will take some time, please be patient.\n')
# fitted_results = results.FittedResults(date_range + '-data_poor')
# tropomi_processing.add_predictions(fitted_results)
# tropomi_processing.add_dry_air_column_densities(fitted_results)
# tropomi_processing.calculate_dry_air_column_density_residuals(fitted_results)
print('\nCreated the following files and directories:\n')
print('    [1] augmented_observations/' + date_range + '-data_poor/*.nc')
print('    [2] outputs/' + date_range + '-data_poor/dry_air_column_density_residuals.csv')
print('\n=========================================================\n')
print('Calculating final results on data-rich days.')
print('This may take some time, please be patient.')
#fitted_results = results.FittedResults(date_range + '-data_rich')
#tropomi_processing.calculate_final_results(fitted_results)
print('\nCreated the following files:')
print('    [1] outputs/' + date_range + '-data_rich/outputs/final_results.csv')
print('\n=========================================================\n')
print('Calculating final results on data-poor days.')
print('This may take some time, please be patient.')
#fitted_results = results.FittedResults(date_range + '-data_poor')
#tropomi_processing.calculate_final_results(fitted_results)
print('\nCreated the following files:')
print('    [1] outputs/' + date_range + '-data_poor/outputs/final_results.csv')
print('\n=========================================================\n')
print('\n Creating and saving the figures that were used in the paper.\n')
fitted_results = results.FittedResults(date_range + '-data_rich')
plotting.make_directory(date_range)
plotting.figure_1(date_range, '2019-01-31')
plotting.figure_2(date_range)
plotting.figure_3(fitted_results, '2019-01-31')
plotting.figure_4(fitted_results)
plotting.figure_5(date_range, '2019-01-31')
plotting.figure_6(date_range)
print('\nCreated the following files:\n')
print('    [1] figures/' + date_range + '/figure_1.png')
print('    [2] figures/' + date_range + '/figure_2.pdf')
print('    [3] figures/' + date_range + '/figure_3.pdf')
print('    [4] figures/' + date_range + '/figure_4.pdf')
print('    [5] figures/' + date_range + '/figure_5.png')
print('    [6] figures/' + date_range + '/figure_6.pdf')

