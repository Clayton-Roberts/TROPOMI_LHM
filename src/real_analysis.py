from src import tropomi_processing as tp
from src import viirs_processing as vp
from src import dropout_tests as dt
from src import fit_model as fm
from src import results as sr
from src import model_comparison as mc
from src import plotting as p

#=======================================================
#   --- Flags for real analysis ---
#-----------------------------------
PROCESS_TROPOMI_FILES  = False
PROCESS_DATA_POOR_DAYS = False
FIT_POOR_DAYS          = False
AUGMENT_DATA_POOR_DAYS = False
PROCESS_VIIRS_FILES    = False
PERFORM_DROPOUT_FIT    = False
PERFORM_FULL_FIT       = False
COMPARE_MODELS         = False
AUGMENT_DATA_RICH_DAYS = False
MAKE_TIME_SERIES       = False
MAKE_PLOTS             = False
#-----------------------------------
#   --- Flags for real runs ---
#-----------------------------------
START_DATE = '20190101'
END_DATE   = '20191231'
MODEL      = 'data_poor'
RUN_NAME   = START_DATE + '-' + END_DATE + '-' + MODEL
#-----------------------------------
#    --- Flags for plotting ---
#-----------------------------------
SHOW_GROUND_TRUTH    = False
PARAM                = 'alpha'
DATE                 = '2019-01-05'
QUANTITY             = 'CH4'
SHOW_WARMUP_DRAWS    = False
PLOT_STUDY_REGION    = False
PLOT_FLARES          = False
SHOW_QAD_PIXELS_ONLY = True
SHOW_AUGMENTED_CH4   = True
##=======================================================

if PROCESS_DATA_POOR_DAYS:
    tp.make_directories(RUN_NAME)
    tp.create_dataset_data_poor_days(RUN_NAME)

if FIT_POOR_DAYS:
    fm.fit_data_poor_days(RUN_NAME)

if PROCESS_TROPOMI_FILES:
    print('Preparing data for analysis:')

    tp.make_directories(RUN_NAME)
    tp.create_dataset_data_rich_days(RUN_NAME)
    tp.prepare_data_rich_dataset_for_cmdstanpy(RUN_NAME)

if PROCESS_VIIRS_FILES:
    vp.generate_flare_time_series(RUN_NAME)

if PERFORM_DROPOUT_FIT:

    dt.make_directories(RUN_NAME)
    dt.create_csvs(RUN_NAME)
    dt.prepare_dataset_for_cmdstanpy(RUN_NAME)

    print('Fitting without holdout observations:')
    fm.nuts('data/' + RUN_NAME + '/dropout/data.json',
            'models/' + MODEL + '.stan',
            RUN_NAME + '/dropout')
    results = sr.FittedResults(RUN_NAME + '/dropout')
    results.write_reduced_chi_squared_csv()
    results.write_residuals_csv()

if PERFORM_FULL_FIT:
    print('Fitting with all observations:')
    fm.nuts('data/' + RUN_NAME + '/data.json',
            'models/' + MODEL + '.stan',
            RUN_NAME)

if COMPARE_MODELS:
    individual_error_fitted_model = sr.FittedResults(START_DATE + '-' + END_DATE + '-individual_error')
    daily_mean_error_fitted_model = sr.FittedResults(START_DATE + '-' + END_DATE + '-daily_mean_error')

    mc.compare_models(individual_error_fitted_model, daily_mean_error_fitted_model)

if AUGMENT_DATA_RICH_DAYS:
    results = sr.FittedResults(RUN_NAME)
    tp.add_predictions(results)
    tp.add_dry_air_column_densities(results)
    tp.calculate_dry_air_column_density_residuals(results)

if AUGMENT_DATA_POOR_DAYS:
    results = sr.FittedResults(RUN_NAME)
    tp.add_predictions(results)
    tp.add_dry_air_column_densities(results)

if MAKE_TIME_SERIES:
    results = sr.FittedResults(RUN_NAME)
    tp.write_plotable_quantities_csv_file(results)
    #tp.calculate_prediction_vs_poor_pixel_value(results)
    # Need the RUN_NAME to include 'dropout' to run the below function.
    #tp.calculate_prediction_vs_heldout_pixel_value(results)

if MAKE_PLOTS:
    results = sr.FittedResults(RUN_NAME)
    #results.calculate_fractional_metric()
    # p.trace(results,
    #         PARAM,
    #         date=DATE,
    #         compare_to_ground_truth=SHOW_GROUND_TRUTH,
    #         show_warmup_draws=SHOW_WARMUP_DRAWS)
    p.observations_scatterplot(DATE, RUN_NAME)
    p.regression_scatterplot(DATE, results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    # p.alpha_beta_scatterplot(results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    # p.dropout_scatterplot(DATE, RUN_NAME)
    # p.reduced_chi_squared(RUN_NAME)
    # p.residuals(RUN_NAME)
    # p.beta_flare_time_series(results)
    # p.tropomi_plot('2019-01-31', 'CH4',
    #                plot_study_region=False,
    #                qa_only=True,
    #                show_flares=False,
    #                include_predictions=True,
    #                good_predictions_only=True,
    #                show_precisions=True)
    # p.dry_air_column_density_cross_plot(results)
    # p.alpha_time_series(results)
    # p.no2_ch4_flarestack_crossplot(results, 'CH4', show_augmented_load=True)
    # p.poor_pixel_predicted_pixel_cross_plot(results)
    # p.heldout_pixel_predicted_pixel_cross_plot(results)
    # PLOTS FOR THE PAPER
    # p.figure_1(DATE)
    # p.figure_2(results, DATE)
    # p.figure_3(results)
    # p.figure_4(DATE)
    # p.figure_6(results)