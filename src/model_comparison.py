import numpy as np

def compare_models(individual_model, mean_error_model):
    '''A function to calculate expected log pointwise predictive density for each of the two models and compare
    the two.

    :param individual_model: A loaded FittedModel for when a model was fitted using per-observation uncertainty.
    :type individual_model: FittedModel
    :param mean_error_model: A loaded FittedModel for when a model was fitted using mean daily observational error.
    :type individual_model: FittedModel
    '''

    print("Calculating for individual error per observation model...")
    individual_model_elpd_i = individual_model.elpd_i()
    individual_model_elpd   = sum(individual_model_elpd_i)
    print(round(individual_model_elpd, 2))
    print("Calculating for daily mean error model...")
    mean_error_model_elpd_i = mean_error_model.elpd_i()
    mean_error_model_elpd   = sum(mean_error_model_elpd_i)
    print(round(mean_error_model_elpd, 2))

    M = len(mean_error_model_elpd_i)

    diff = mean_error_model_elpd - individual_model_elpd

    mean_diff = np.mean([elpd_i_A - elpd_i_B for elpd_i_A, elpd_i_B in zip(mean_error_model_elpd_i, individual_model_elpd_i)])

    se_diff = np.sqrt((M/(M-1)) *
                      sum(((elpd_i_A - elpd_i_B) - mean_diff)**2
                          for elpd_i_A, elpd_i_B in zip(mean_error_model_elpd_i, individual_model_elpd_i)))

    print('ELPD difference: {:.2f}'.format(diff))
    print('SE of ELPD difference: {:.2f}'.format(se_diff))


