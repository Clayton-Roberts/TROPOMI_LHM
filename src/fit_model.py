import time
import numpy as np
import cmdstanpy
from   cmdstanpy import CmdStanModel, set_cmdstan_path
import os

def install_cmdstan(cmdstan_path=None):
    '''After the python package cmdstanpy is downloaded/imported, CmdStan also needs to be installed somewhere (C++ code).

    :param cmdstan_path: The location that you want to install CmdStan to. If left blank it will install to a default locaiton.
    :type cmdstan_path: string
    '''

    if cmdstan_path:
        cmdstanpy.install_cmdstan(cmdstan_path)
    else:
        cmdstanpy.install_cmdstan()

def fit_model(data_path, model_path, output_directory, cmdstan_path=None):
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

    # Make the directory to save the outputs in.
    os.mkdir('outputs/' + output_directory)

    # Record the start time in order to write elapsed time for fitting to the output file.
    start_time = time.time()

    # Set the random seed for replicability.
    np.random.seed(101)

    if cmdstan_path:
        # If not set manually, will use default download location.
        set_cmdstan_path(cmdstan_path)

    model = CmdStanModel(stan_file=model_path)

    # Fit the model.
    fit = model.sample(chains=4, data=data_path, iter_warmup=500,
                       iter_sampling=4000, seed=101, show_progress=False,
                       output_dir='outputs/' + output_directory,
                       save_diagnostics=True,
                       max_treedepth=15,
                       inits=['inits/chain_1.json',
                              'inits/chain_2.json',
                              'inits/chain_3.json',
                              'inits/chain_4.json'])

    # Record the elapsed time.
    elapsed_time = time.time() - start_time

    f = open("outputs/summary.txt", "a")
    f.write("Elapsed time to fit model and save output: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '\n')
    f.write('---------------------------------------------------' + '\n')
    f.write(fit.diagnose())
    f.close()

