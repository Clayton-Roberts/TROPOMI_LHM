from src.test_suite import functions as tsf
from src import fit_model as fm
from src import summarise_results as sr

# Important: all things will be run from here, file paths defined as such.

#=======================================================
#    --- Testing suite ---
#=======================================================
#    --- Flags for testing suite ---
#-------------------------
TEST_RUN_NAME   = 'test_run_1'
# You only need to install CmdStan once!
INSTALL_CMDSTAN = False
##=======================================================
tsf.make_directories(TEST_RUN_NAME)
tsf.generate_mu_and_Sigma(TEST_RUN_NAME)
tsf.generate_alphas_betas_and_gammas(10, TEST_RUN_NAME)
tsf.generate_dataset(TEST_RUN_NAME)
tsf.prepare_dataset_for_cmdstanpy(TEST_RUN_NAME)
if INSTALL_CMDSTAN:
    fm.install_cmdstan()
fm.fit_model('test_suite/data/' + TEST_RUN_NAME + '/data.json', 'models/linear_hierarchical_model.stan', TEST_RUN_NAME)
#sr.FittedModel('test_run')
