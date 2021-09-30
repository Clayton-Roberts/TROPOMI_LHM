# Enhanced monitoring of atmospheric methane
 
This repository contains the code used to conduct the analysis and generate the plots found in the paper titled "Enhanced monitoring of atmospheric methane from space via hierarchical Bayesian inference". This paper can be found on arXiv here (add link later) and is currently under review at Nature Communications. 

# Directions for usage/replication
1. [Cloning the repository](#cloning_the_repository)
2. [Conda environment setup](#conda_environment_setup)
3. [Installing CmdStanPy and CmdStan](#cmdstanpy_install)
4. [Retrieving observations](#retrieving_observations)

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

## Retrieving observations <a name="retrieving_observations"></a>

This work requires observations and data from four separate sources (see the "Data Availability" section of the paper):
* the Copernicus Open Access Hub
* the Copernicus Climate Change Service
* the Earth Observation Group at the Colorado School of Mines
* the Global Monitoring Laboratory at the National Oceanic and Atmospheric Association (NOAA)

The subsections below discuss which data products you need from each organisation, how to download the data, and where to download them to on your local machine.

### Copernicus Open Access Hub
The Sentinal-5P Data Offer at the [Copernicus Open Access Hub](https://scihub.copernicus.eu/userguide/WebHome) hosts the TROPOMI Level 2 Data Products, i.e., the TROPOMI observations of methane and nitrogen dioxide. I used the OData API to automatically retrieve TROPOMI observations of methane and nitrogen dioxide over the Permian Basin for every day in 2019. Download each day's observations to the following locations on your machine: 

* Methane observations should be saved to 
  ```bash 
  <local_folder_name>/src/observations/CH4/
  ```
* Nitrogen dioxide observations should be saved to
  ```bash
  <local_folder_name>/src/observations/NO2/
  ```

### Copernicus Climate Change Service
Talk about ERA5

### Earth Observation Group, Colorado School of Mines
Talk about VIIRS Nightfire here. 

### Global Monitoring Laboratory, NOAA
Talk about methane background data



