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
PROCESS_TROPOMI_FILES = False
PROCESS_VIIRS_FILES   = False
PERFORM_DROPOUT_FIT   = False
PERFORM_FULL_FIT      = False
COMPARE_MODELS        = False
MAKE_PLOTS            = False
#-----------------------------------
#   --- Flags for real runs ---
#-----------------------------------
START_DATE = '20190101'
END_DATE   = '20190131'
MODEL      = 'daily_mean_error'
RUN_NAME   = START_DATE + '-' + END_DATE + '-' + MODEL
#-----------------------------------
#    --- Flags for plotting ---
#-----------------------------------
SHOW_GROUND_TRUTH    = False
PARAM                = 'sigma_alpha'
DATE                 = '2019-01-31'
MOLECULE             = 'NO2'
PLOT_STUDY_REGION    = True
PLOT_FLARES          = True
SHOW_QAD_PIXELS_ONLY = True
##=======================================================

if PROCESS_TROPOMI_FILES:
    print('Preparing data for analysis:')

    tp.make_directories(RUN_NAME)
    tp.create_dataset(RUN_NAME)
    tp.prepare_dataset_for_cmdstanpy(RUN_NAME)

    dt.make_directories(RUN_NAME)
    dt.create_csvs(RUN_NAME)
    dt.prepare_dataset_for_cmdstanpy(RUN_NAME)

if PROCESS_VIIRS_FILES:
    vp.generate_flare_time_series(RUN_NAME)

if PERFORM_DROPOUT_FIT:
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

if MAKE_PLOTS:
    results = sr.FittedResults(RUN_NAME)
    #results.calculate_fractional_metric()
    p.trace(results, PARAM, date=DATE,
             compare_to_ground_truth=SHOW_GROUND_TRUTH)
    # p.observations_scatterplot(DATE, RUN_NAME)
    # p.regression_scatterplot(DATE, results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    # p.alpha_beta_scatterplot(results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    # p.dropout_scatterplot(DATE, RUN_NAME)
    # p.reduced_chi_squared(RUN_NAME)
    # p.residuals(START_DATE + '-' + END_DATE + '-daily_mean_error',
    #             START_DATE + '-' + END_DATE + '-individual_error')
    # p.beta_flare_time_series(results)
    # p.tropomi_plot(DATE, MOLECULE,
    #                plot_study_region=PLOT_STUDY_REGION,
    #                qa_only=SHOW_QAD_PIXELS_ONLY,
    #                show_flares=PLOT_FLARES)