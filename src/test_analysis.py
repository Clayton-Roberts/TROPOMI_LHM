from src.test_suite import functions as tsf
from src import dropout_tests as dt
from src import fit_model as fm
from src import results as sr
from src import model_comparison as mc
from src import plotting as p
import time

#=======================================================
#    --- Flags for testing analysis ---
#----------------------------------------
GENERATE_TEST_DATA    = False
PERFORM_DROPOUT_FIT   = False
PERFORM_FULL_FIT      = False
COMPARE_MODELS        = False
MAKE_PLOTS            = True
#-----------------------------------
#   --- Flags for test runs ---
#-----------------------------------
NUM_DAYS        = 10
NUM_OBS         = 100
MODEL           = 'daily_mean_error'
RUN_NAME        = str(NUM_DAYS) + '_days-' \
                  + str(NUM_OBS) + '_M-' \
                  + MODEL
# You only need to install CmdStan once!
INSTALL_CMDSTAN = False
#-----------------------------------
#    --- Flags for plotting ---
#-----------------------------------
SHOW_GROUND_TRUTH = True
PARAM             = 'alpha'
DATE              = '2021-08-10'
##=======================================================

if GENERATE_TEST_DATA:
    print('Preparing data for analysis:')

    tsf.make_directories(RUN_NAME)
    tsf.generate_mu_and_Sigma(RUN_NAME)
    tsf.generate_alphas_betas_and_gammas(NUM_DAYS, RUN_NAME)
    tsf.generate_dataset(RUN_NAME, NUM_OBS)
    tsf.prepare_dataset_for_cmdstanpy(RUN_NAME)

    dt.make_directories(RUN_NAME)
    dt.create_csvs(RUN_NAME)
    dt.prepare_dataset_for_cmdstanpy(RUN_NAME)

if PERFORM_DROPOUT_FIT:
    print('Fitting without holdout observations:')
    time.sleep(1)

    fm.nuts('data/' + RUN_NAME + '/dropout/data.json',
            'models/' + MODEL + '.stan',
            RUN_NAME + '/dropout')
    results = sr.FittedResults(RUN_NAME + '/dropout')
    results.write_reduced_chi_squared_csv()
    results.write_residuals_csv()

if PERFORM_FULL_FIT:
    print('Fitting with all observations:')
    time.sleep(1)
    fm.nuts('data/' + RUN_NAME + '/data.json',
            'models/' + MODEL + '.stan',
            RUN_NAME)

if COMPARE_MODELS:
    individual_error_fitted_model = sr.FittedResults(str(NUM_DAYS) + '_days-' + str(NUM_OBS) + '_M-individual_error')
    daily_mean_error_fitted_model = sr.FittedResults(str(NUM_DAYS) + '_days-' + str(NUM_OBS) + '_M-daily_mean_error')

    mc.compare_models(individual_error_fitted_model, daily_mean_error_fitted_model)

if MAKE_PLOTS:
    results = sr.FittedResults(RUN_NAME)
    results.calculate_fractional_metric()
    p.trace(results, PARAM, date=DATE,
            compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.observations_scatterplot(DATE, RUN_NAME)
    p.regression_scatterplot(DATE, results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.alpha_beta_scatterplot(results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.dropout_scatterplot(DATE, RUN_NAME)
    p.reduced_chi_squared(RUN_NAME)
    p.residuals(str(NUM_DAYS) + '_days-' + str(NUM_OBS) + '_M-daily_mean_error',
                str(NUM_DAYS) + '_days-' + str(NUM_OBS) + '_M-individual_error')