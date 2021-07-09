from src.test_suite import functions as tsf
from src import fit_model as fm
from src import summarise_results as sr
from src import plotting as p

# Important: all things will be run from here, file paths defined as such.

#=======================================================
#    --- Flags for testing suite ---
#-----------------------------------
PERFORM_RUN     = True
SHOW_RESULTS    = False
TEST_RUN_NAME   = 'generating_mu_alpha_mu_beta_20_days'
NUM_DAYS        = 20
# You only need to install CmdStan once!
INSTALL_CMDSTAN = False
#-----------------------------------
#    --- Flags for plotting ---
#-----------------------------------
##=======================================================

if PERFORM_RUN:
    tsf.make_directories(TEST_RUN_NAME)
    tsf.generate_mu_and_Sigma(TEST_RUN_NAME)
    tsf.generate_alphas_betas_and_gammas(NUM_DAYS, TEST_RUN_NAME)
    tsf.generate_dataset(TEST_RUN_NAME)
    tsf.prepare_dataset_for_cmdstanpy(TEST_RUN_NAME)
    if INSTALL_CMDSTAN:
        fm.install_cmdstan()
    fm.fit_model('test_suite/data/' + TEST_RUN_NAME + '/data.json', 'models/linear_hierarchical_model.stan', TEST_RUN_NAME)

if SHOW_RESULTS:
    fitted_model = sr.FittedModel(TEST_RUN_NAME)
    fitted_model.display_results()
    p.trace(fitted_model, 'mu.2', 10000011, compare_to_ground_truth=True)
