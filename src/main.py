from src.test_suite import functions as tsf

# Important: all things will be run from here, file paths defined as such.

#-------------------------------------------------------
#    --- Testing suite ---
#-------------------------------------------------------
tsf.generate_mu_and_Sigma()
tsf.generate_alphas_betas_and_gammas(20)
tsf.generate_dataset()
tsf.prepare_dataset_for_cmdstanpy()