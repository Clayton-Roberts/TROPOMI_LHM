# Enhanced monitoring of atmospheric methane
 
This repository contains the code used to conduct the analysis and generate the plots found in the paper titled "Enhanced monitoring of atmospheric methane from space via hierarchical Bayesian inference". This paper can be found on arXiv here (add link later) and is currently under review at Nature Communications. 

If you have any questions or are having trouble with the code, feel free to contact me at cnr31@ast.cam.ac.uk . 

# Directions for usage/replication
1. [Cloning the repository](#cloning_the_repository)
2. [Conda environment setup](#conda_environment_setup)
3. [Installing CmdStanPy and CmdStan](#cmdstanpy_install)
4. [Retrieving observations](#retrieving_observations)
    * [TROPOMI CH4 and NO2](#tropomi)
    * [ERA5](#era5)
    * [VIIRS Nightfire](#viirs)
    * [NOAA CH4 background](#noaa)
5. [Configure settings](#config)
6. [Run the analysis](#script)

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

### Copernicus Open Access Hub <a name="tropomi"></a>
The Sentinal-5P Data Offer at the [Copernicus Open Access Hub](https://scihub.copernicus.eu/userguide/WebHome) hosts the TROPOMI Level 2 Data Products, i.e., the TROPOMI observations of methane and nitrogen dioxide. I used the OData API to automatically retrieve TROPOMI observations of methane and nitrogen dioxide over the Permian Basin for every day in 2019. Download each day's observations to the following locations on your machine: 

* Methane observations should be saved to 
  ```bash 
  <local_folder_name>/src/observations/CH4/
  ```
* Nitrogen dioxide observations should be saved to
  ```bash
  <local_folder_name>/src/observations/NO2/
  ```
  
Each filename should be automatically downloaded with a filename of the format %Y%m%dT%H%M%S.nc e.g., 20190101T201359.nc. WARNING: These files will require nearly 200 gigabytes of storage space!

### Copernicus Climate Change Service <a name="era5"></a>
ERA5 reanalysis data is available at the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) for download (see the "Download data" tab at the link). This work made use of their single-level hourly spatial grids of "Surface pressure" and "Total column water vapour", for the hours 1800-2200, the spatial extent -106 through -99 degrees longitude and 29 through 35 degrees latitude, for each day in 2019. Use their GUI and download this data to a single file at the following location: 

```bash
<local_folder_name>/src/observations/ERA5/Permain_Basin_2019.nc
```

### Earth Observation Group, Colorado School of Mines <a name="viirs"></a>
Our work also made use of VIIRS Nightfire observations of lit flare stacks in the Permian Basin for the year 2019 (we use V3.0). These observations are hosted by the [Earth Observation Group](https://eogdata.mines.edu/products/vnf/) at the Colorado School of Mines. I unfortunately found no other way to download these observations for each day in 2019 except for by hand and unzipping the .zip files for each day. You will need to submit an application to EOG for access (see the link and "Download" section). After downloaded the zip files for the relevant days, unzip their contents (should be csv files with names of format VNF_npp_dXXXXXXXX_noaa_v30-ez.csv) to the following location: 

```bash
<local_folder_name>/src/observations/VIIRS/
```

### Global Monitoring Laboratory, NOAA <a name="noaa"></a>
When validating our results we required a reference background level of atmospheric methane, and we chose to use the global monthly mean marine surface level of methane from the [Global Monitoring Laboratory](https://gml.noaa.gov/ccgg/trends_ch4/) at NOAA (used with their permission). To replicate this work, download the csv file "Globally averaged marine surface monthly mean data" to the following location on your machine with the indicated name: 

```bash
<local_folder_name>/src/observations/NOAA/noaa_ch4_background.csv
```

## Final configuration <a name="config"></a>
You need to alter the file 

```bash
<local_folder_name>/src/constants.py
```
to provide the following information:

```python
FILE_PREFIX  = '/absolute/path/to/your/<local_folder_name>'
CMDSTAN_PATH = '/absolute/path/to/your/local/install/of/CmdStan'
```

## Run the analysis <a name="script"></a>
After downloading the necessary observations/data and configuring CmdStanPy, you should be able to replicate the entire analysis by running 

```bash
yourmachine <local_folder_name> % python src/run_analysis.py
```

It will take several hours to complete the entire analysis, progress bars will be printed to demonstrate that the code is running. 



