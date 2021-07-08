from src.test_suite import functions as tsf
from src import fit_model as fm

# Important: all things will be run from here, file paths defined as such.

#-------------------------------------------------------
#    --- Testing suite ---
#-------------------------------------------------------
tsf.generate_mu_and_Sigma()
tsf.generate_alphas_betas_and_gammas(20)
tsf.generate_dataset()
tsf.prepare_dataset_for_cmdstanpy()
# This next function doesn't have to be run each time, but it won't re-download if CmdStan is already in the
# indicated directory.
fm.install_cmdstan()
