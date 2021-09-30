# Enhanced monitoring of atmospheric methane
 
This repository contains the code used to conduct the analysis and generate the plots found in the paper titled "Enhanced monitoring of atmospheric methane from space via hierarchical Bayesian inference". This paper can be found on arXiv here (add link later) and is currently under review at Nature Communications. 

# Table of Contents
1. [Cloning the repository](#cloning_the_repository)
2. [Conda environment setup](#conda_environment_setup)
3. [Installing CmdStanPy and CmdStan](#cmdstanpy_install)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)

## Cloning the repository <a name="cloning_the_repository"></a>

Clone this repository into a local folder via

```bash
git clone https://github.com/Clayton-Roberts/TROPOMI_LHM <local_folder_name>
```

## Conda environment setup <a name="conda_environment_setup"></a>
This repository includes a yml file from which you can generate a conda environment. Using this environment to run the code isn't strictly necessary if you'd rather make your own, but you can automatically re-create my environment via 

```bash
yourmachine <local_folder_name> % conda env create -f environment.yml
```

This will create a conda environment called "TROPOMI_LHM" that contains the necessary packages to run the code. 

## Installing CmdStanPy and CmdStan <a name="cmdstanpy_install"></a>

We write and fit our models with Stan via the interface CmdStanPy, a python package for Stan users. Using CmdStanPy also requires a local install of CmdStan, a command-line tool for Stan users. Follow the instructions [here](https://cmdstanpy.readthedocs.io/en/stable-0.9.65/getting_started.html) to install CmdStanPy and CmdStan, and note down where you've placed your local install of CmdStan. 


