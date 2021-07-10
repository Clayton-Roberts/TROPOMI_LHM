from src.test_suite import functions as tsf
from src import fit_model as fm
from src import summarise_results as sr
from src import plotting as p

# Important: all things will be run from here, file paths defined as such.

#=======================================================
#    --- Flags for testing suite ---
#-----------------------------------
GENERATE_TEST_DATA  = True
PERFORM_DROPOUT     = False
FIT_FULL_TEST_DATA  = False
SHOW_RESULTS        = False
TEST_RUN_NAME       = '10_days_N_100'
#-----------------------------------
#   --- Flags for test runs ---
#-----------------------------------
NUM_DAYS        = 10
# You only need to install CmdStan once!
INSTALL_CMDSTAN = False
#-----------------------------------
#    --- Flags for plotting ---
#-----------------------------------
SHOW_GROUND_TRUTH = True
PARAM             = 'sigma_beta'
DATE              = 10000019
##=======================================================

if GENERATE_TEST_DATA:
    tsf.make_directories(TEST_RUN_NAME)
    tsf.generate_mu_and_Sigma(TEST_RUN_NAME)
    tsf.generate_alphas_betas_and_gammas(NUM_DAYS, TEST_RUN_NAME)
    tsf.generate_dataset(TEST_RUN_NAME)
    tsf.prepare_dataset_for_cmdstanpy(TEST_RUN_NAME)

if FIT_FULL_TEST_DATA:
    if INSTALL_CMDSTAN:
        fm.install_cmdstan()
    fm.fit_model('data/' + TEST_RUN_NAME + '/data.json', 'models/linear_hierarchical_model.stan', TEST_RUN_NAME)

if SHOW_RESULTS:
    fitted_model = sr.FittedModel(TEST_RUN_NAME)
    fitted_model.display_results()
    p.trace(fitted_model, PARAM, date=DATE, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.observations_scatterplot(DATE, fitted_model, compare_to_ground_truth=SHOW_GROUND_TRUTH)
    p.alpha_beta_scatterplot(fitted_model, compare_to_ground_truth=SHOW_GROUND_TRUTH)