import time
import numpy as np
import cmdstanpy
from   cmdstanpy import CmdStanModel, set_cmdstan_path
from   src import constants as ct

def install_cmdstan():
    '''After the python package cmdstanpy is downloaded/imported, CmdStan also needs to be installed somewhere (C++ code).
    This only needs to be done once.
    '''

    cmdstanpy.install_cmdstan(ct.CMDSTAN_PATH)

def fit_model(data_path, model_path, output_directory):
    '''This function will fit a probability model to a set of data and then save outputs that summarise probability
    distributions over the model parameters.

    :param data_path: Location of the file of data to fit the model to, must be a .json file.
    :type data_path: string
    :param model_path: Location of the .stan model file the data will be fit to.
    :type model_path: string
    :param output_directory: Name of folder to be created in /src/outputs where outputs from current run will be saved.
    :type output_directory: string
    :param cmdstan_path: Directory that the CmdStan download is located in.
    :type cmdstan_path: string
    '''

    # Record the start time in order to write elapsed time for fitting to the output file.
    start_time = time.time()

    # Set the random seed for replicability.
    np.random.seed(101)

    set_cmdstan_path(ct.CMDSTAN_PATH)

    model = CmdStanModel(stan_file=ct.FILE_PREFIX + '/' + model_path)

    # Fit the model.
    fit = model.sample(chains=4, parallel_chains=4,
                       data=ct.FILE_PREFIX + '/' + data_path, iter_warmup=500,
                       iter_sampling=1000, seed=101, show_progress=True,
                       output_dir=ct.FILE_PREFIX + '/outputs/' + output_directory,
                       save_diagnostics=True,
                       max_treedepth=12,
                       inits=[ct.FILE_PREFIX + '/inits/chain_1.json',
                              ct.FILE_PREFIX + '/inits/chain_2.json',
                              ct.FILE_PREFIX + '/inits/chain_3.json',
                              ct.FILE_PREFIX + '/inits/chain_4.json'
                              ])

    # Record the elapsed time.
    elapsed_time = time.time() - start_time

    f = open(ct.FILE_PREFIX + "/outputs/" + output_directory + "/summary.txt", "a")
    f.write("Elapsed time to fit model and save output: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '\n')
    f.write('---------------------------------------------------' + '\n')
    f.write(fit.diagnose())
    f.close()

