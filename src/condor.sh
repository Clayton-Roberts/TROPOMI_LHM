#!/bin/bash
source /opt/ioa/setup/setup_modules.bash
module load gcc
/data/cnr31/conda_envs/TROPOMI_LHM/bin/python /data/cnr31/TROPOMI_LHM/src/main.py
