import time
import numpy as np
import cmdstanpy
from cmdstanpy import CmdStanModel

def install_cmdstan(cmdstan_path=None):
    '''After the python package cmdstanpy is downloaded/imported, CmdStan also needs to be installed somewhere (C++ code).

    :param cmdstan_path: The location that you want to install CmdStan to. If left blank it will install to a default locaiton.
    :type cmdstan_path: string
    '''

    if cmdstan_path:
        cmdstanpy.install_cmdstan(cmdstan_path)
    else:
        cmdstanpy.install_cmdstan()

def fit_model(data_path, model_path, cmdstan_path=None):
    '''This function will fit a probability model to a set of data and then save outputs that summarise probability
    distributions over the model parameters.

    :param data_path: Location of the file of data to fit the model to, must be a .json file.
    :type data_path: string
    :param model_path: Location of the .stan model file the data will be fit to.
    :type model_path: string
    :param cmdstan_path: Directory that the CmdStan download is located in.
    :type cmdstan_path: string
    '''

    start_time = time.time()

    np.random.seed(101)

    model = CmdStanModel(stan_file=model_path)

