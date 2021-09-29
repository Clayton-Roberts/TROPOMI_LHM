import constants as ct
import numpy as np
import time

def compare_models(individual_error_model, mean_error_model):
    '''A function to calculate expected log pointwise predictive density for each of the two models and compare
    the two.

    :param individual_error_model: A loaded FittedModel for when a model was fitted using per-observation uncertainty.
    :type individual_error_model: FittedModel
    :param mean_error_model: A loaded FittedModel for when a model was fitted using mean daily observational error.
    :type individual_model: FittedModel
    '''

    print("Individual error model:")
    time.sleep(1)
    individual_model_elpd_i = individual_error_model.elpd_i()
    individual_model_elpd   = sum(individual_model_elpd_i)
    print("Individual error model estimated elpd: ", round(individual_model_elpd, 2))

    print("Daily mean error model:")
    time.sleep(1)
    mean_error_model_elpd_i = mean_error_model.elpd_i()
    mean_error_model_elpd   = sum(mean_error_model_elpd_i)
    print("Daily mean error model estimated elpd: ", round(mean_error_model_elpd, 2))

    M = len(mean_error_model_elpd_i)

    diff = mean_error_model_elpd - individual_model_elpd

    mean_diff = np.mean([elpd_i_A - elpd_i_B for elpd_i_A, elpd_i_B in zip(mean_error_model_elpd_i, individual_model_elpd_i)])

    se_diff = np.sqrt((M/(M-1)) *
                      sum(((elpd_i_A - elpd_i_B) - mean_diff)**2
                          for elpd_i_A, elpd_i_B in zip(mean_error_model_elpd_i, individual_model_elpd_i)))

    print('Difference in elpd: {:.2f} '.format(diff) + u"\u00B1" + ' {:.2f}'.format(se_diff))

    f = open(ct.FILE_PREFIX + "/outputs/" + mean_error_model.run_name + "/model_comparison.txt", "a")
    f.write("Daily mean error model estimated elpd: " + str(round(mean_error_model_elpd, 2)) + '\n')
    f.write("Individual error model estimated elpd: " + str(round(individual_model_elpd, 2)) + '\n')
    f.write('Difference in elpd: {:.2f} '.format(diff) + u"\u00B1" + ' {:.2f}'.format(se_diff))
    f.close()

    g = open(ct.FILE_PREFIX + "/outputs/" + individual_error_model.run_name + "/model_comparison.txt", "a")
    g.write("Daily mean error model estimated elpd: " + str(round(mean_error_model_elpd, 2)) + '\n')
    g.write("Individual error model estimated elpd: " + str(round(individual_model_elpd, 2)) + '\n')
    g.write('Difference in elpd: {:.2f} '.format(diff) + u"\u00B1" + ' {:.2f}'.format(se_diff))
    g.close()


