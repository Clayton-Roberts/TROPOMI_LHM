import time
import numpy as np
import cmdstanpy
from cmdstanpy import CmdStanModel

def install_cmdstan(cmdstan_directory=None):
    '''After the python package cmdstanpy is downloaded/imported, CmdStan also needs to be installed somewhere (C++ code).

    :param cmdstan_directory: The location that you want to install CmdStan to. If left blank it will install to a default locaiton.
    :type cmdstan_directory: string
    '''

    if cmdstan_directory:
        cmdstanpy.install_cmdstan(cmdstan_directory)
    else:
        cmdstanpy.install_cmdstan()

def fit_model(cmdstan_directory=None, data_location, model_location):
    '''This function will fit a probability model to a set of data and then save outputs that summarise probability
    distributions over the model parameters.

    :param cmdstan_directory: Directory that the CmdStan download is located in.
    :type cmdstan_directory: string
    :param data_location: Location of the file of data to fit the model to, must be a .json file.
    :type data_location: string
    :param model_location: Location of the .stan model file the data will be fit to.
    :type model_location: string
    '''