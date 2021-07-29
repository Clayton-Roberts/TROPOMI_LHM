from src.test_suite import functions as tsf
from src import fit_model as fm
from src import results as sr
from src import dropout_tests as dt
from src import plotting as p
from src import model_comparison as mc

# Important: all things will be run from here, file paths defined as such.

#=======================================================
#    --- Flags for testing suite ---
#-----------------------------------
GENERATE_TEST_DATA   = False
PERFORM_DROPOUT_FIT  = False
PERFORM_FULL_FIT     = False
COMPARE_MODELS       = False
SHOW_RESULTS         = True
#-----------------------------------
#   --- Flags for test runs ---
#-----------------------------------
NUM_DAYS        = 20
NUM_OBS         = 100
MODEL           = 'daily_mean_error'
TEST_RUN_NAME   = str(NUM_DAYS) + '_days_' \
                  + str(NUM_OBS) + '_M_' \
                  + MODEL
# You only need to install CmdStan once!
INSTALL_CMDSTAN = False
#-----------------------------------
#    --- Flags for plotting ---
#-----------------------------------
SHOW_GROUND_TRUTH = True
PARAM             = 'beta'
DATE              = 10000007
##=======================================================

if INSTALL_CMDSTAN:
    fm.install_cmdstan()

if GENERATE_TEST_DATA:
    tsf.make_directories(TEST_RUN_NAME)
    tsf.generate_mu_and_Sigma(TEST_RUN_NAME)
    tsf.generate_alphas_betas_and_gammas(NUM_DAYS, TEST_RUN_NAME)
    tsf.generate_dataset(TEST_RUN_NAME, NUM_OBS)
    tsf.prepare_dataset_for_cmdstanpy(TEST_RUN_NAME)

if PERFORM_DROPOUT_FIT:
    dt.make_directories(TEST_RUN_NAME)
    dt.create_csvs(TEST_RUN_NAME)
    dt.prepare_dataset_for_cmdstanpy(TEST_RUN_NAME)
    fm.nuts('data/' + TEST_RUN_NAME + '/dropout/data.json',
            'models/' + MODEL + '.stan',
            TEST_RUN_NAME + '/dropout')
    results = sr.FittedResults(TEST_RUN_NAME + '/dropout')
    results.write_reduced_chi_squared_csv()

if PERFORM_FULL_FIT:
    fm.nuts('data/' + TEST_RUN_NAME + '/data.json',
            'models/' + MODEL + '.stan',
            TEST_RUN_NAME)

if COMPARE_MODELS:
    individual_error_fitted_model = sr.FittedResults(str(NUM_DAYS) + '_days_' + str(NUM_OBS) + '_M_individual_error')
    daily_mean_error_fitted_model = sr.FittedResults(str(NUM_DAYS) + '_days_' + str(NUM_OBS) + '_M_daily_mean_error')

    mc.compare_models(individual_error_fitted_model, daily_mean_error_fitted_model)

if SHOW_RESULTS:
    results = sr.FittedResults(TEST_RUN_NAME)
    results.calculate_fractional_metric()
    p.trace(results, PARAM, date=DATE,
            compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.observations_scatterplot(DATE, TEST_RUN_NAME)
    p.regression_scatterplot(DATE, results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.alpha_beta_scatterplot(results, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.dropout_scatterplot(DATE, TEST_RUN_NAME)
    p.reduced_chi_squared(TEST_RUN_NAME)

