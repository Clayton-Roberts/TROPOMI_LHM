from src import constants as ct
import datetime
import os
import glob
from tqdm import tqdm
import pandas as pd

def generate_flare_time_series(run_name):
    """This function will loop through the VIIRS data directory and create a time series of how many flares were 'on'
    on a given day, for days in range for this run.

    :param run_name: Name of this model run.
    :type run_name: string
    """

    start_date, end_date, model = run_name.split('-')
    start_datetime = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end_datetime   = datetime.datetime.strptime(end_date, "%Y%m%d").date()

    time_series_df = pd.DataFrame(columns=('Date', 'Flare_count'))

    # Create the list of dates to iterate over.
    num_days = (end_datetime - start_datetime).days + 1
    date_list = [start_datetime + datetime.timedelta(days=x) for x in range(num_days)]

    # For every date in range for this model run:
    for date in tqdm(date_list, desc='Processing VIIRS observations'):

        # Create string from the date datetime
        date_string = date.strftime("%Y%m%d")

        # Get the corrresponding VIIRS observation file, there should only be one.
        viirs_file = glob.glob(ct.FILE_PREFIX + '/observations/VIIRS/*' + date_string + '*.csv')[0]

        # Set number of active flares to zero and count up from here.
        active_flares = 0

        # Open the file.
        df = pd.read_csv(viirs_file)

        # Examine every active flare
        for index in df.index:

            # If latitudes of flare in study region:
            if (ct.STUDY_REGION['Permian_Basin'][2] < df.Lat_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][3]):

                # If longitudes of flare in study region:
                if (ct.STUDY_REGION['Permian_Basin'][0] < df.Lon_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][1]):

                    if df.Temp_BB[index] != 999999:
                        active_flares += 1

        # Add a row to the time series dataframe of flare stack count.
        time_series_df = time_series_df.append({'Date': date, 'Flare_count': active_flares},
                                               ignore_index=True)

    # Sort by date and write to csv file.
    time_series_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/flare_counts.csv', index=False)