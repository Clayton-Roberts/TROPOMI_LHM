# This file is going to be the script that is run from the command line to replicate the work of the paper.
# It will import other modules of code, make directories and will write various outputs.
# It will take in a date range as two command line arguments to indicate the date range you want to run the
# analysis for.
from datetime import datetime
import sys
import dropout_tests
import fit_model
import results
import tropomi_processing

# First, need to make all of the data
# Second, fit the dropout model and run the dropout tests
# Third, fit the full model and write all files
# Fourth, fit the individual error model for month of January.
# Fifth, check if the data-rich model has been run for month of January, run it if it doesn't exist yet.
# Sixth, do the comparison between the two models.
# 7th, print all outputs to screen and write a single massive output file, that goes through all metrics, and quantities that went into the paper.
# 8th, make the figures and save them to a folder in the figures folder.

start_datetime = datetime.strptime(str(sys.argv[1]), '%Y-%m-%d')
end_datetime   = datetime.strptime(str(sys.argv[2]), '%Y-%m-%d')

print('=========================================================')
print('Analysis date range: ' +
      start_datetime.strftime('%B %-d, %Y') +
      ' - ' +
      end_datetime.strftime('%B %-d, %Y') )
print('=========================================================')

# Process TROPOMI observations, prepare data for the model if the day is data-rich.
date_range = start_datetime.strftime('%Y%m%d') + '-' + end_datetime.strftime('%Y%m%d')
tropomi_processing.make_all_directories(date_range)
#tropomi_processing.create_dataset_data_rich_days(run_name)
#tropomi_processing.prepare_data_rich_dataset_for_cmdstanpy(run_name)
