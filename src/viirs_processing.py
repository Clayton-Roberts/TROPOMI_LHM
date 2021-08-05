from src import constants as ct
import datetime
import os
from tqdm import tqdm
import pandas as pd

def generate_flare_time_series(run_name):
    """This function will loop through the VIIRS data directory and create a time series of how many flares were 'on'
    on a given day, for days in range for this run.

    :param run_name: Name of this model run.
    :type run_name: string
    """


    start_date, end_date, model = run_name.split('-')
    start_date = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end_date   = datetime.datetime.strptime(end_date, "%Y%m%d").date()

    time_series_df = pd.DataFrame(columns=('Date', 'Flare_count'))

    # Loop through every VIIRS data file you have.
    for file in tqdm(os.listdir(ct.FILE_PREFIX + '/observations/VIIRS/'), desc='Iterating over all VIIRS observations'):

        date = datetime.datetime.strptime(file.split('_')[2][1:], "%Y%m%d").date()

        if start_date <= date <= end_date:

            # Set number of active flares to zero and count up from here.
            active_flares = 0

            # Open the file.
            df = pd.read_csv(ct.FILE_PREFIX + '/observations/VIIRS/' + file)

            # Examine every active flare
            for i in range(len(df)):

                # If latitudes of flare in study region:
                if (ct.STUDY_REGION['Permian_Basin'][2] < df.Lat_GMTCO[i] < ct.STUDY_REGION['Permian_Basin'][3]):

                    # If longitudes of flare in study region:
                    if (ct.STUDY_REGION['Permian_Basin'][0] < df.Lon_GMTCO[i] < ct.STUDY_REGION['Permian_Basin'][1]):

                        if df.Temp_BB[i] != 999999:

                            active_flares += 1

            # Add a row to the time series dataframe of flare stack count.
            time_series_df = time_series_df.append({'Date': date, 'Flare_count': active_flares}, ignore_index=True)

    # Sort by date and write to csv file.
    time_series_df  = time_series_df.sort_values(by='Date')
    time_series_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/flare_counts.csv', index=False)
