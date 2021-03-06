import constants as ct
import results as sr
import netCDF4 as nc4
import numpy as np
from pyproj import Geod
from shapely import geometry
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from scipy import stats
import os
import pandas as pd
from tqdm import tqdm
import datetime
import glob
import json
import shutil

def make_directories(run_name):
    '''This function checks that all the relevant directories are made. If you re-use a run name, the directory and
    previous contents will be deleted, and the directory will be created again WITHOUT WARNING.

    :param run_name: The name of the run, will also be the name of the folders that is created.
    :type run_name string
    '''

    try:
        os.makedirs(ct.FILE_PREFIX + '/data/' + run_name)
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/data/' + run_name)
        os.makedirs(ct.FILE_PREFIX + '/data/' + run_name)

    try:
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name)
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + run_name)
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + run_name)

def prepare_data_poor_dataset_for_cmdstanpy(date_range, date, dropout):
    '''This function prepares data by turning it into json. Data-poor days are fit individually and so we need to
    not only indicate the date range of the analysis, but also the individual day that is being fit. We also need to
    indicate whether or not these data-poor days are part of a dropout dataset or not.

    :param date_range: The date range of the analysis. Must be of the format "%Y%m%d-%Y%m%d".
    :type date_range: str
    :param date: The particular day that the model is going to be fit to. Must be of the format "%Y-%m-%d".
    :type date: str
    :param dropout: A flag to indicate whether these days are part of a dropout dataset or not.
    :type dropout: bool
    '''

    if not dropout:
        df = pd.read_csv(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/dataset.csv', delimiter=',',
                        header=0, index_col=1)  # Indexing by Date instead of Day_ID
    elif dropout:
        df = pd.read_csv(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/dropout/remaining_dataset.csv', delimiter=',',
                         header=0, index_col=1)  # Indexing by Date instead of Day_ID

    days_df = df[df.index == date]

    data = {}

    obs_no2     = days_df.obs_NO2.tolist()
    obs_ch4     = days_df.obs_CH4.tolist()
    avg_sigma_N = np.mean(days_df.sigma_N)
    avg_sigma_C = np.mean(days_df.sigma_C)
    N           = len(obs_no2)

    data['N']         = N
    data['NO2_obs']   = obs_no2
    data['CH4_obs']   = obs_ch4
    data['sigma_N']   = avg_sigma_N
    data['sigma_C']   = avg_sigma_C

    # Load up the results from the data-rich run to get the information learned on mu and Sigma
    start_date, end_date = date_range.split('-')
    data_rich_run_name   = start_date + '-' + end_date + '-data_rich'
    data_rich_results    = sr.FittedResults(data_rich_run_name)

    mu_1      = data_rich_results.full_trace['mu.1']
    mu_2      = data_rich_results.full_trace['mu.2']
    Sigma_1_1 = data_rich_results.full_trace['Sigma.1.1']
    Sigma_1_2 = data_rich_results.full_trace['Sigma.1.2']
    Sigma_2_2 = data_rich_results.full_trace['Sigma.2.2']

    mean_mu_1      = np.mean(mu_1)
    mean_mu_2      = np.mean(mu_2)
    mean_Sigma_1_1 = np.mean(Sigma_1_1)
    mean_Sigma_1_2 = np.mean(Sigma_1_2)
    mean_Sigma_2_2 = np.mean(Sigma_2_2)

    theta = [mean_mu_1, mean_mu_2, mean_Sigma_1_1, mean_Sigma_1_2, mean_Sigma_2_2]

    Upsilon = np.cov(np.array([list(mu_1),
                               list(mu_2),
                               list(Sigma_1_1),
                               list(Sigma_1_2),
                               list(Sigma_2_2)]))

    data['theta']   = theta
    data['Upsilon'] = Upsilon.tolist()

    directory = date_range + '-data_poor'
    if dropout:
        directory += '/dropout'

    with open(ct.FILE_PREFIX + '/outputs/' + directory + '/dummy/data.json', 'w') as outfile:
        json.dump(data, outfile)

def prepare_data_rich_dataset_for_cmdstanpy(run_name):
    '''This function takes the "dataset.cvs" file located at "data/run_name" and turns it into json
    that is suitable for usage by the cmdstanpy package (.csv files are unabled to be provided as data when we
    fit our models).

    :param run_name: Name of the model run. Must be of the format "%Y%m%d-%Y%m%d-X" where X is either "data_rich" or
        "individual_error".
    :type run_name:str
    '''

    df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', delimiter=',',
                     header=0, index_col=1)  # Indexing by Date instead of Day_ID

    obs_no2 = list(df.obs_NO2)
    obs_ch4 = list(df.obs_CH4)
    day_id  = list(df.day_id)
    D       = int(np.max(day_id))
    N       = len(obs_no2)

    group_sizes = []
    for i in range(D):
        day = i+1
        size = len(df[df.day_id == day])
        group_sizes.append(size)

    avg_sigma_N = []
    avg_sigma_C = []

    for i in range(D):
        day = i+1
        mean_sigma_N = np.mean(df[df.day_id == day].sigma_N)
        mean_sigma_C = np.mean(df[df.day_id == day].sigma_C)

        avg_sigma_N.append(mean_sigma_N)
        avg_sigma_C.append(mean_sigma_C)

    data = {}
    data['N']           = N
    data['D']           = D
    data['day_id']      = day_id
    data['group_sizes'] = group_sizes
    data['NO2_obs']     = obs_no2
    data['CH4_obs']     = obs_ch4
    if 'data_rich' in run_name:
        data['sigma_N']     = avg_sigma_N
        data['sigma_C']     = avg_sigma_C
    elif 'individual_error' in run_name:
        data['sigma_N'] = df.sigma_N.tolist()
        data['sigma_C'] = df.sigma_C.tolist()

    with open(ct.FILE_PREFIX + '/data/' + run_name + '/data.json', 'w') as outfile:
        json.dump(data, outfile)

def unpack_no2(filename):
    '''This function opens a "raw" TROPOMI observation file of :math:`\mathrm{NO}_2` and returns arrays of quantities
    of interest that we need for our analysis.

    :param filename: The name of the file, should be in the format YYYYMMDDTHHMMSS.nc .
    :type filename: string
    '''

    # Open the NO2 file.
    file = nc4.Dataset(ct.PERMIAN_OBSERVATIONS + '/NO2/' + filename,'r')

    # Generate the arrays of NO2 pixel centre latitudes and longitudes.
    pixel_centre_latitudes  = np.array(file.groups['PRODUCT'].variables['latitude'])[0]
    pixel_centre_longitudes = np.array(file.groups['PRODUCT'].variables['longitude'])[0]

    # Generate the arrays of NO2 pixel values and their 1-sigma precisions, units in mol/m^2.
    pixel_values     = np.array(file.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'])[0]
    pixel_precisions = np.array(file.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column_precision'])[0]

    # Generate the array of QA factors
    qa_values = np.array(file.groups['PRODUCT'].variables['qa_value'])[0]

    return pixel_values, pixel_precisions, pixel_centre_latitudes, pixel_centre_longitudes, qa_values

def unpack_ch4(filename):
    '''This function opens a "raw" TROPOMI observation file of :math:`\mathrm{CH}_4` and returns arrays of quantities
    of interest that we need for our analysis.

    :param filename: The name of the file, should be in the format YYYYMMDDTHHMMSS.nc .
    :type filename: string
    '''

    # Open the CH4 file.
    ch4_file = nc4.Dataset(ct.PERMIAN_OBSERVATIONS + '/CH4/' + filename, 'r')

    # Generate the arrays of CH4 pixel centre latitudes and longitudes.
    pixel_centre_latitudes  = np.array(ch4_file.groups['PRODUCT'].variables['latitude'])[0]
    pixel_centre_longitudes = np.array(ch4_file.groups['PRODUCT'].variables['longitude'])[0]

    # Generate the arrays of CH4 pixel values and their 1-sigma precisions, units in ppbv.
    pixel_values     = np.array(ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio'])[0]
    pixel_precisions = np.array(ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio_precision'])[0]

    # Generate the array of CH4 pixel qa value.
    qa_values = np.array(ch4_file.groups['PRODUCT'].variables['qa_value'])[0]

    return pixel_values, pixel_precisions, pixel_centre_latitudes, pixel_centre_longitudes, qa_values

def reduce_no2(no2_pixel_values, no2_pixel_precisions, no2_pixel_centre_latitudes, no2_pixel_centre_longitudes,
               no2_qa_values):
    '''This function takes in arrays of data that have been unpacked from a TROPOMI observation file, and reduces
    them to arrays of data that are contained within the study region. It also masks pixels that don't pass the QA threshold.'''

    # Cut the NO2 data down to what is within the Permian Basin box. Need empty lists to hold things in.
    pixel_values            = []
    pixel_precisions        = []
    pixel_centre_latitudes  = []
    pixel_centre_longitudes = []

    # For every latitude...
    for i in range(no2_pixel_values.shape[0]):
        # ...for every longitude...
        for j in range(no2_pixel_values.shape[1]):
            # ... if the coordinates are located in the study region of interest...
            if (ct.STUDY_REGION['Permian_Basin'][2] < no2_pixel_centre_latitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][3]) and \
                    (ct.STUDY_REGION['Permian_Basin'][0] < no2_pixel_centre_longitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][1]):

                # Append all quantities of interest for this location to the relevant list.
                pixel_precisions.append(no2_pixel_precisions[i, j])
                pixel_centre_latitudes.append(no2_pixel_centre_latitudes[i, j])
                pixel_centre_longitudes.append(no2_pixel_centre_longitudes[i, j])

                # If the QA value threshold is passed:
                if no2_qa_values[i, j] >= 0.75:
                    pixel_values.append(no2_pixel_values[i, j])
                else:
                    # Mark a bad pixel with the mask value.
                    pixel_values.append(1e35)

    # Convert all list to arrays.
    pixel_values            = np.array(pixel_values)
    pixel_precisions        = np.array(pixel_precisions)
    pixel_centre_latitudes  = np.array(pixel_centre_latitudes)
    pixel_centre_longitudes = np.array(pixel_centre_longitudes)

    return pixel_values, pixel_precisions, pixel_centre_latitudes, pixel_centre_longitudes

def reduce_ch4(ch4_pixel_values, ch4_pixel_precisions, ch4_pixel_centre_latitudes, ch4_pixel_centre_longitudes,
               ch4_qa_values):
    '''This function takes in arrays of data that have been unpacked from a TROPOMI observation file, and reduces
    them to arrays of data that are contained within the study region. It also masks pixels that don't pass the QA threshold.'''

    # Cut the NO2 data down to what is within the Permian Basin box. Need empty lists to hold things in.
    pixel_values            = []
    pixel_precisions        = []
    pixel_centre_latitudes  = []
    pixel_centre_longitudes = []

    # For every latitude...
    for i in range(ch4_pixel_values.shape[0]):
        # ...for every longitude...
        for j in range(ch4_pixel_values.shape[1]):
            # ... if the coordinates are located in the study region of interest...
            if (ct.STUDY_REGION['Permian_Basin'][2] < ch4_pixel_centre_latitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][3]) and \
                    (ct.STUDY_REGION['Permian_Basin'][0] < ch4_pixel_centre_longitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][1]):

                # Append all quantities of interest for this location to the relevant list.
                pixel_precisions.append(ch4_pixel_precisions[i, j])
                pixel_centre_latitudes.append(ch4_pixel_centre_latitudes[i, j])
                pixel_centre_longitudes.append(ch4_pixel_centre_longitudes[i, j])

                # If the QA value threshold is passed:
                if ch4_qa_values[i, j] >= 0.5:
                    pixel_values.append(ch4_pixel_values[i, j])
                else:
                    # Mark a bad pixel with the mask value.
                    pixel_values.append(1e35)

    # Convert all list to arrays.
    pixel_values            = np.array(pixel_values)
    pixel_precisions        = np.array(pixel_precisions)
    pixel_centre_latitudes  = np.array(pixel_centre_latitudes)
    pixel_centre_longitudes = np.array(pixel_centre_longitudes)

    return pixel_values, pixel_precisions, pixel_centre_latitudes, pixel_centre_longitudes

def get_colocated_measurements(filename):
    '''This function determines if two observations of NO2 and CH4 are co-located, and if they are, returns them in
    a list of relevant quantities.

    :param filename: The name of the file, should be in the format YYYYMMDDTHHMMSS.nc .
    :type filename: string
    '''

    # Unpack the TROPOMI NO2 data for this day.
    no2_pixel_values, no2_pixel_precisions, no2_pixel_centre_latitudes, no2_pixel_centre_longitudes, no2_qa_values \
        = unpack_no2(filename)

    # Reduce the NO2 data down to what is contained within the study area
    no2_pixel_values, no2_pixel_precisions, no2_pixel_centre_latitudes, no2_pixel_centre_longitudes \
        = reduce_no2(no2_pixel_values, no2_pixel_precisions, no2_pixel_centre_latitudes, no2_pixel_centre_longitudes, no2_qa_values)

    # Unpack the TROPOMI CH4 data for this day.
    ch4_pixel_values, ch4_pixel_precisions, ch4_pixel_centre_latitudes, ch4_pixel_centre_longitudes, ch4_qa_values \
        = unpack_ch4(filename)

    # Reduce the CH4 data down to what is contained within the study area
    ch4_pixel_values, ch4_pixel_precisions, ch4_pixel_centre_latitudes, ch4_pixel_centre_longitudes \
        = reduce_ch4(ch4_pixel_values, ch4_pixel_precisions, ch4_pixel_centre_latitudes, ch4_pixel_centre_longitudes, ch4_qa_values)

    # Interpolate the reduced CH4 data at the locations of reduced NO2 data:
    interpolated_ch4_pixel_values = griddata((ch4_pixel_centre_longitudes,
                                              ch4_pixel_centre_latitudes),
                                              ch4_pixel_values,
                                              (no2_pixel_centre_longitudes, no2_pixel_centre_latitudes),
                                              method='linear')

    # Interpolate the reduced CH4 data precisions at the locations of reduced NO2 data:
    interpolated_ch4_pixel_precisions = griddata((ch4_pixel_centre_longitudes,
                                              ch4_pixel_centre_latitudes),
                                              ch4_pixel_precisions,
                                              (no2_pixel_centre_longitudes, no2_pixel_centre_latitudes),
                                              method='linear')

    # Define the lists that will hold our observations and their respective errors
    obs_NO2   = []
    sigma_N   = []
    obs_CH4   = []
    sigma_C   = []
    latitude  = []
    longitude = []

    # Now we need to fill the lists.
    # Only need to iterate over the reduced data, which is one-dimensional.
    for j in range(len(no2_pixel_values)):
        # TODO Sphinx autodoc does not like the below line for some reason
        if (no2_pixel_values[j] < 1e20) and (interpolated_ch4_pixel_values[j] < 1e20):
            # Append the CH4 and NO2 pixel values to the relevant lists.
            obs_NO2.append(no2_pixel_values[j] * 1e3)  # Convert to milli mol / m^2
            obs_CH4.append(interpolated_ch4_pixel_values[j])
            # Append the CH4 and NO2 precisions to the relevant lists.
            sigma_N.append(no2_pixel_precisions[j] * 1e3)  # Convert to milli mol / m^2
            sigma_C.append(interpolated_ch4_pixel_precisions[j])
            # Append the latitudes and longitudes to the relevant lists.
            latitude.append(no2_pixel_centre_latitudes[j])
            longitude.append(no2_pixel_centre_longitudes[j])

    return obs_CH4, sigma_C, obs_NO2, sigma_N, latitude, longitude

def convert_TROPOMI_observations_to_csvs(date_range):
    '''This function will iterate over a date range, determine if the TROPOMI observation on this day is either
    "data-rich" or "data-poor", and then write the observations into the appropriate .csv file.

    :param date_range: The date range of the analysis. Must be a string formatted as "%Y%m%d-%Y%m%d".
    :type date_range: str
    '''

    start_date, end_date = date_range.split('-')

    start_datetime = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end_datetime   = datetime.datetime.strptime(end_date, "%Y%m%d").date()

    num_total_days             = 0
    num_data_rich_days         = 0
    num_data_poor_days         = 0
    num_data_rich_observations = 0
    num_data_poor_observations = 0

    data_rich_day_id = 1  # Stan starts counting from 1!
    data_poor_day_id = 1

    # Empty list to hold dataframes for each day's observations when checks are passed.
    data_rich_daily_dfs = []
    data_poor_daily_dfs = []

    # A summary dataframe for overall metrics of for this run's data.
    data_rich_summary_df = pd.DataFrame(columns=(('date', 'day_id', 'N', 'R')))
    data_poor_summary_df = pd.DataFrame(columns=(('date', 'day_id', 'N', 'R')))

    # Create the list of dates to iterate over.
    num_days = (end_datetime - start_datetime).days + 1
    date_list = [start_datetime + datetime.timedelta(days=x) for x in range(num_days)]

    # For every date in range for this model run:
    for date in tqdm(date_list, desc='Converting TROPOMI observations into csv files'):

        num_total_days += 1

        # Create string from the date datetime
        date_string = date.strftime("%Y%m%d")

        # Create list of TROPOMI filenames that match this date. Sometimes there are two TROPOMI overpasses
        # that are a couple hours apart. Usually one overpass captures the whole study region.
        tropomi_overpasses = [file.split('/')[-1] for file in
                              glob.glob(
                                  ct.PERMIAN_OBSERVATIONS + '/NO2/' + date_string + '*.nc')]

        total_obs_CH4   = []
        total_sigma_C   = []
        total_obs_NO2   = []
        total_sigma_N   = []
        total_latitude  = []
        total_longitude = []

        for overpass in tropomi_overpasses:
            try:
                obs_CH4, sigma_C, obs_NO2, sigma_N, latitude, longitude = get_colocated_measurements(overpass)
            #TODO need to fix the file not found error as that's more to do with slight differences in minutes in filename
            except (ValueError, FileNotFoundError):
                continue

            total_obs_CH4.extend(obs_CH4)
            total_sigma_C.extend(sigma_C)
            total_obs_NO2.extend(obs_NO2)
            total_sigma_N.extend(sigma_N)
            total_latitude.extend(latitude)
            total_longitude.extend(longitude)

        # If there are more than 100 co-located measurements on this day ...
        if len(total_obs_NO2) >= 100:

            r, p_value = stats.pearsonr(total_obs_NO2, total_obs_CH4)

            # if R >= 0.4 ...
            if r >= 0.4:
                # Checks are passed, so write to the various datasets.

                num_data_rich_days += 1
                num_data_rich_observations += len(total_obs_NO2)

                # Append summary of this day to the summary dataframe.
                data_rich_summary_df = data_rich_summary_df.append({'date': date, 'day_id': data_rich_day_id, 'N': len(total_obs_NO2),
                                                                    'R': round(r, 2)},
                                                                    ignore_index=True)

                # Create a dataframe containing the observations for this day.
                day_df = pd.DataFrame(list(zip([data_rich_day_id] * len(total_obs_NO2),
                                               [date] * len(total_obs_NO2),
                                               total_obs_NO2,
                                               total_obs_CH4,
                                               total_sigma_N,
                                               total_sigma_C,
                                               total_latitude,
                                               total_longitude)),
                                      columns=('day_id', 'date', 'obs_NO2', 'obs_CH4',
                                               'sigma_N', 'sigma_C', 'latitude', 'longitude'))

                # Append the dataframe to this day to the list of dataframes to later concatenate together.
                data_rich_daily_dfs.append(day_df)

                # Increment day_id
                data_rich_day_id += 1

            else:
                num_data_poor_days += 1
                num_data_poor_observations += len(total_obs_NO2)

                # Append summary of this day to the summary dataframe.
                data_poor_summary_df = data_poor_summary_df.append({'date': date,
                                                                    'day_id': data_poor_day_id,
                                                                    'N': len(total_obs_NO2),
                                                                    'R': round(r, 2)},
                                                                   ignore_index=True)

                # Create a dataframe containing the observations for this day.
                day_df = pd.DataFrame(list(zip([data_poor_day_id] * len(total_obs_NO2),
                                               [date] * len(total_obs_NO2),
                                               total_obs_NO2,
                                               total_obs_CH4,
                                               total_sigma_N,
                                               total_sigma_C,
                                               total_latitude,
                                               total_longitude)),
                                      columns=('day_id', 'date', 'obs_NO2', 'obs_CH4',
                                               'sigma_N', 'sigma_C', 'latitude', 'longitude'))

                # Append the dataframe to this day to the list of dataframes to later concatenate together.
                data_poor_daily_dfs.append(day_df)

                # Increment day_id
                data_poor_day_id += 1

        elif len(total_obs_NO2) >= 2:

            r, p_value = stats.pearsonr(total_obs_NO2, total_obs_CH4)

            num_data_poor_days += 1
            num_data_poor_observations += len(total_obs_NO2)

            # Append summary of this day to the summary dataframe.
            data_poor_summary_df = data_poor_summary_df.append({'date': date,
                                                                'day_id': data_poor_day_id,
                                                                'N': len(total_obs_NO2),
                                                                'R': round(r, 2)},
                                                                ignore_index=True)

            # Create a dataframe containing the observations for this day.
            day_df = pd.DataFrame(list(zip([data_poor_day_id] * len(total_obs_NO2),
                                           [date] * len(total_obs_NO2),
                                           total_obs_NO2,
                                           total_obs_CH4,
                                           total_sigma_N,
                                           total_sigma_C,
                                           total_latitude,
                                           total_longitude)),
                                  columns=('day_id', 'date', 'obs_NO2', 'obs_CH4',
                                           'sigma_N', 'sigma_C', 'latitude', 'longitude'))

            # Append the dataframe to this day to the list of dataframes to later concatenate together.
            data_poor_daily_dfs.append(day_df)

            # Increment day_id
            data_poor_day_id += 1

    # Sort the summary dataframe by date.
    data_rich_summary_df.to_csv(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich/summary.csv', index=False)
    data_poor_summary_df.to_csv(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/summary.csv', index=False)

    # Concatenate the daily dataframes together to make the dataset dataframe. Leave sorted by Day_ID.
    data_rich_dataset_df = pd.concat(data_rich_daily_dfs)
    data_poor_dataset_df = pd.concat(data_poor_daily_dfs)
    data_rich_dataset_df.to_csv(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich/dataset.csv', index=False)
    data_poor_dataset_df.to_csv(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/dataset.csv', index=False)

    f = open(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich/summary.txt', 'a')
    f.write("Total number of days in range: " + str(num_total_days) + '\n')
    f.write("Total number of data-rich days in range: " + str(num_data_rich_days) + '\n')
    f.write("Total number of observations in range: " + str(num_data_rich_observations) + '\n')
    f.close()

    g = open(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor/summary.txt', 'a')
    g.write("Total number of days in range: " + str(num_total_days) + '\n')
    g.write("Total number of data-poor days in range: " + str(num_data_poor_days) + '\n')
    g.write("Total number of observations in range: " + str(num_data_poor_observations) + '\n')
    g.close()

def make_all_directories(date_range):
    '''This function is for creating all the necessary directories needed to store processed TROPOMI observations as
    data and the directories needed to store the outputs.

    :param date_range: The date range of the analysis. Must be a string formatted as "%Y%m%d-%Y%m%d".
    :type date_range: str
    '''

    try:
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich')
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_rich')

    try:
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor')
        os.makedirs(ct.FILE_PREFIX + '/data/' + date_range + '-data_poor')

    try:
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich')
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich')

    try:
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor')
        os.makedirs(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor')

def copy_data():
    '''This function is for copying over data-rich observations for the month of January to two new directories
    so that we can carry out some model comparison.'''

    full_summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/20190101-20191231-data_rich/summary.csv', index_col=0)

    reduced_summary_df = pd.DataFrame(columns=['date','day_id','N','R'])

    full_dataset_df = pd.read_csv(ct.FILE_PREFIX + '/data/20190101-20191231-data_rich/dataset.csv')

    daily_observations = []

    for date in full_summary_df.index:
        if datetime.datetime.strptime(date, '%Y-%m-%d') < datetime.datetime(year=2019, month=2, day=1):
            series = full_summary_df.loc[date]
            reduced_summary_df = reduced_summary_df.append({'date': date,
                                                            'day_id': int(series.day_id),
                                                            'N': int(series.N),
                                                            'R': series.R},
                                                           ignore_index=True)

            observation_df = full_dataset_df[full_dataset_df.date == date]

            daily_observations.append(observation_df)

    reduced_summary_df.to_csv(ct.FILE_PREFIX + '/data/20190101-20190131-data_rich/summary.csv', index=False)
    reduced_summary_df.to_csv(ct.FILE_PREFIX + '/data/20190101-20190131-individual_error/summary.csv', index=False)

    reduced_observations_df = pd.concat(daily_observations, ignore_index=True)
    reduced_observations_df.to_csv(ct.FILE_PREFIX + '/data/20190101-20190131-data_rich/dataset.csv', index=False)
    reduced_observations_df.to_csv(ct.FILE_PREFIX + '/data/20190101-20190131-individual_error/dataset.csv', index=False)

def add_predictions(fitted_results):
    '''This function is for creating augmented .nc TROPOMI files that were part of the hierarchical model run
    using "data rich" days.

    :param fitted_results: The hierarchical model run.
    :type fitted_results: FittedResults
    '''

    start_date, end_date, model_type = fitted_results.run_name.split('-')

    # Make the directory for the run name
    try:
        os.makedirs(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name)
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name)
        os.makedirs(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name)

    # Read the summary.csv file for this model run to get the dates of the days that were used in the hierarchical fit.
    # Index by day
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', index_col=0)

    # Read the metrics for the data poor days if we're checking doing predictions on data-poor days, index by date
    if model_type == 'data_poor':
        diagnostics_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/diagnostics.csv', index_col=0)

    for date in tqdm(summary_df.index, desc='Adding predictions on ' + '-'.join(model_type.split('_')) + ' days'):

        if model_type == 'data_rich':
            do_predictions = True
        elif model_type == 'data_poor':
            days_diagnostics = diagnostics_df.loc[date]
            if days_diagnostics.max_treedepth and \
                days_diagnostics.e_bfmi and \
                days_diagnostics.effective_sample_size and days_diagnostics.split_rhat:
                if days_diagnostics.post_warmup_divergences <= 5:
                    do_predictions = True

        if do_predictions:
            day_id = int(summary_df.loc[date].day_id)

            # Create string from the date datetime
            date_string = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

            # Create list of TROPOMI filenames that match this date. Sometimes there are two TROPOMI overpasses
            # that are a couple hours apart. Usually one overpass captures the whole study region.
            tropomi_overpasses = [file.split('/')[-1] for file in
                                  glob.glob(
                                      ct.PERMIAN_OBSERVATIONS + '/NO2/' + date_string + '*.nc')]

            for filename in tropomi_overpasses:

                # Unpack some of the data into arrays
                no2_pixel_values, no2_pixel_precisions, no2_pixel_centre_latitudes, no2_pixel_centre_longitudes, no2_qa_values \
                    = unpack_no2(filename)

                try:
                    ch4_pixel_values, ch4_pixel_precisions, ch4_pixel_centre_latitudes, ch4_pixel_centre_longitudes, ch4_qa_values \
                        = unpack_ch4(filename)
                except FileNotFoundError:
                    continue

                # Open the original CH4 file again to access some dimensions
                ch4_file = nc4.Dataset(ct.PERMIAN_OBSERVATIONS + '/CH4/' + filename, 'r')

                # Created the augmented netCDF4 dataset
                augmented_file = nc4.Dataset(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name +
                                             '/' + filename,
                                             'w',
                                             format='NETCDF4')

                # The augmented file structure will emulate the original files for simplicity's sake.
                product_group      = augmented_file.createGroup('PRODUCT')
                support_data_group = product_group.createGroup('SUPPORT_DATA')
                geolocations_group = support_data_group.createGroup('GEOLOCATIONS')

                # Create the dimensions you need for the PRODUCT group, copy them from the CH4 observation file. We only
                # need a few from the original files.
                product_group.createDimension('time', ch4_file.groups['PRODUCT'].dimensions['time'].size)
                product_group.createDimension('scanline', ch4_file.groups['PRODUCT'].dimensions['scanline'].size)
                product_group.createDimension('ground_pixel', ch4_file.groups['PRODUCT'].dimensions['ground_pixel'].size)
                product_group.createDimension('corner', ch4_file.groups['PRODUCT'].dimensions['corner'].size)

                # Create the variables for the methane mixing ratio and the methane mixing ratio precision
                methane_mixing_ratio = product_group.createVariable('methane_mixing_ratio',
                                                                    np.float32,
                                                                    ('time', 'scanline', 'ground_pixel'))

                methane_mixing_ratio_precision = product_group.createVariable('methane_mixing_ratio_precision',
                                                                              np.float32,
                                                                              ('time', 'scanline', 'ground_pixel'))

                predictor_pixel_qa_value  = product_group.createVariable('prediction_pixel_qa_value',
                                                                         np.float32,
                                                                         ('time', 'scanline', 'ground_pixel'))

                predictor_pixel_outside_of_zero = product_group.createVariable('prediction_pixel_outside_of_zero',
                                                                               np.float32,
                                                                               ('time', 'scanline', 'ground_pixel'))

                geolocations_group.createVariable('latitude_bounds',
                                                  np.float32,
                                                  ('time', 'scanline', 'ground_pixel', 'corner'))

                geolocations_group.createVariable('longitude_bounds',
                                                  np.float32,
                                                  ('time', 'scanline', 'ground_pixel', 'corner'))

                # Interpolate the NO2 data at the locations of the CH4 pixels
                interpolated_no2_pixel_values = griddata((no2_pixel_centre_longitudes.flatten(), no2_pixel_centre_latitudes.flatten()),
                                                         no2_pixel_values.flatten(),
                                                         (ch4_pixel_centre_longitudes, ch4_pixel_centre_latitudes),
                                                         method='linear')

                interpolated_no2_pixel_value_precisions = griddata((no2_pixel_centre_longitudes.flatten(), no2_pixel_centre_latitudes.flatten()),
                                                                   no2_pixel_precisions.flatten(),
                                                                   (ch4_pixel_centre_longitudes, ch4_pixel_centre_latitudes),
                                                                   method='linear')

                interpolated_no2_qa_values = griddata((no2_pixel_centre_longitudes.flatten(), no2_pixel_centre_latitudes.flatten()),
                                                      no2_qa_values.flatten(),
                                                      (ch4_pixel_centre_longitudes, ch4_pixel_centre_latitudes),
                                                      method='linear')

                # Initialise the augmented CH4 data and precision array entirely with the mask value.
                augmented_methane_mixing_ratio                    = np.full(ch4_pixel_values.shape, 1e32)
                augmented_methane_mixing_ratio_precision          = np.full(ch4_pixel_precisions.shape, 1e32)
                augmented_methane_predictor_pixel_qa_value        = np.full(ch4_pixel_values.shape, 1e32)
                augmented_methane_predictor_pixel_outside_of_zero = np.full(ch4_pixel_values.shape, 1e32)

                # Iterate over the raw CH4 data, and calculate predictions and overwrite as necessary.
                for i in range(ch4_file.groups['PRODUCT'].dimensions['scanline'].size):
                    for j in range(ch4_file.groups['PRODUCT'].dimensions['ground_pixel'].size):
                        if (ct.EXTENT['Permian_Basin'][2] < ch4_pixel_centre_latitudes[i, j] < ct.EXTENT['Permian_Basin'][3]) and \
                                (ct.EXTENT['Permian_Basin'][0] < ch4_pixel_centre_longitudes[i, j] < ct.EXTENT['Permian_Basin'][1]):
                            # Perform a prediction whenever there is an available NO2 pixel, record qa value
                            if interpolated_no2_pixel_values[i, j] < 1e28:
                                obs_no2 = interpolated_no2_pixel_values[i, j] * 1e3  # Convert to mmol / m^2
                                sigma_N = interpolated_no2_pixel_value_precisions[i, j] * 1e3  # Convert to mmol / m^2
                                prediction, precision = fitted_results.predict_ch4(obs_no2, sigma_N, day_id)
                                augmented_methane_mixing_ratio[i, j] = prediction
                                augmented_methane_mixing_ratio_precision[i, j] = precision
                                augmented_methane_predictor_pixel_qa_value[i, j] = interpolated_no2_qa_values[i, j]
                                # Record whether or not the NO2 prediction value is at least 2 sigma away from 0.
                                if obs_no2 / sigma_N >= 2.:
                                    augmented_methane_predictor_pixel_outside_of_zero[i, j] = 1

                methane_mixing_ratio[0,:,:]            = augmented_methane_mixing_ratio
                methane_mixing_ratio_precision[0,:,:]  = augmented_methane_mixing_ratio_precision
                predictor_pixel_qa_value[0,:,:]        = augmented_methane_predictor_pixel_qa_value
                predictor_pixel_outside_of_zero[0,:,:] = augmented_methane_predictor_pixel_outside_of_zero
                augmented_file.close()

def add_dry_air_column_densities(fitted_results):
    '''This function is for adding ERA5-calculated column densities of dry air at the location of the methane pixels.

    :param fitted_results: The model run that we'd like to calculate the dry air column densities for.
    :type fitted_results: FittedResults
    '''

    start_date, end_date, model_type = fitted_results.run_name.split('-')

    # Open the ERA5 file. Contains surface pressure and total column water vapour for all of 2019.
    era5_file = nc4.Dataset(ct.FILE_PREFIX + '/observations/ERA5/Permian_Basin_2019.nc', 'r')
    # ERA5 measures time in hours since 0000, Jan 1 1900
    era5_base_time = datetime.datetime.strptime('19000101T000000', '%Y%m%dT%H%M%S')
    # Unpack some ERA5 quantities
    era5_times      = np.array(era5_file.variables['time'])
    era5_latitudes  = np.array(era5_file.variables['latitude'])
    era5_longitudes = np.array(era5_file.variables['longitude'])
    era5_tcwv       = np.array(era5_file.variables['tcwv'])  # [kg / m^2]
    era5_sp         = np.array(era5_file.variables['sp'])    # [Pa]

    # Open the summary csv file for this model run
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv')

    for date in tqdm(summary_df.date, desc='Adding dry air column density at methane pixel locations on ' + '-'.join(model_type.split('_')) + ' days'):

        # Create string from the date datetime
        date_string = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

        # Create list of TROPOMI filenames that match this date. Sometimes there are two TROPOMI overpasses
        # that are a couple hours apart. Usually one overpass captures the whole study region.
        tropomi_overpasses = [file.split('/')[-1] for file in
                              glob.glob(
                                  ct.FILE_PREFIX + '/observations/NO2/' + date_string + '*.nc')]

        for filename in tropomi_overpasses:

            # Open the original CH4 file.
            original_ch4_file  = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')
            # Open the augmented CH4 file containing all the predictions.
            augmented_ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/augmented_observations/'
                                             + fitted_results.run_name + '/' + filename, 'a', format='NETCDF4')
            # Access the reference observation time of the CH4 observations (seconds since 2010-01-01 00:00:00)
            ch4_base_time = np.array(original_ch4_file.groups['PRODUCT'].variables['time'])[0]
            # Access the timedelta of each scanline from the reference time (milliseconds since the reference time)
            ch4_delta_times = np.array(original_ch4_file.groups['PRODUCT'].variables['delta_time'])[0]

            # Generate the arrays of CH4 pixel centre latitudes and longitudes, same between both files.
            pixel_centre_latitudes = np.array(original_ch4_file.groups['PRODUCT'].variables['latitude'])[0]
            pixel_centre_longitudes = np.array(original_ch4_file.groups['PRODUCT'].variables['longitude'])[0]

            dry_air_column_densities = np.full(np.array(original_ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio'])[0].shape, 1e32)

            for i in range(original_ch4_file.groups['PRODUCT'].dimensions['scanline'].size):
                for j in range(original_ch4_file.groups['PRODUCT'].dimensions['ground_pixel'].size):
                    if (ct.EXTENT['Permian_Basin'][2] < pixel_centre_latitudes[i, j] < ct.EXTENT['Permian_Basin'][
                        3]) and \
                            (ct.EXTENT['Permian_Basin'][0] < pixel_centre_longitudes[i, j] <
                             ct.EXTENT['Permian_Basin'][1]):
                        # Construct the observation time in hours since Jan 1, 1900, 00:00:00
                        ch4_observation_datetime = datetime.datetime.strptime('20100101T000000', '%Y%m%dT%H%M%S') \
                                                   + datetime.timedelta(seconds=int(ch4_base_time)) \
                                                   + datetime.timedelta(milliseconds=int(ch4_delta_times[i]))
                        observation_time_diff = ch4_observation_datetime - era5_base_time
                        scanline_observation_time = (observation_time_diff.days * 24) \
                                                    + (observation_time_diff.seconds / 3600) \
                                                    + (observation_time_diff.microseconds / 3.6e9)

                        # Find indices of era5_times that our TROPOMI file is in between
                        for k in range(len(era5_times)):
                            if era5_times[k] < scanline_observation_time < era5_times[k + 1]:
                                past_time         = era5_times[k]
                                future_time       = era5_times[k + 1]
                                past_time_index   = k
                                future_time_index = k + 1
                                break

                        past_tcwv   = era5_tcwv[past_time_index, :, :]
                        future_tcwv = era5_tcwv[future_time_index, :, :]
                        past_sp     = era5_sp[past_time_index, :, :]
                        future_sp   = era5_sp[future_time_index, :, :]

                        # Interpolate the tcwv data
                        interpolated_past_tcwv = interp2d(era5_longitudes,
                                                          era5_latitudes,
                                                          past_tcwv,
                                                          kind='linear')

                        interpolated_future_tcwv = interp2d(era5_longitudes,
                                                            era5_latitudes,
                                                            future_tcwv,
                                                            kind='linear')

                        # Interpolate the sp data
                        interpolated_past_sp = interp2d(era5_longitudes,
                                                        era5_latitudes,
                                                        past_sp,
                                                        kind='linear')

                        interpolated_future_sp = interp2d(era5_longitudes,
                                                          era5_latitudes,
                                                          future_sp,
                                                          kind='linear')
                        latitude  = pixel_centre_latitudes[i, j]
                        longitude = pixel_centre_longitudes[i, j]
                        tcwv_time_interpolation = interp1d((past_time, future_time),
                                                           (interpolated_past_tcwv(longitude, latitude)[0],
                                                            interpolated_future_tcwv(longitude, latitude)[0]),
                                                           kind='linear')
                        tcwv = float(tcwv_time_interpolation(scanline_observation_time)) # [kg / m^2]
                        sp_time_interpolation = interp1d((past_time, future_time),
                                                         (interpolated_past_sp(longitude, latitude)[0],
                                                          interpolated_future_sp(longitude, latitude)[0]),
                                                         kind='linear')
                        sp = float(sp_time_interpolation(scanline_observation_time))
                        air_column_density             = sp / 9.8 # [kg / m^2]
                        dry_air_column_density         = (air_column_density - tcwv) / (28.964e-3) #[mol / m^2]
                        dry_air_column_densities[i, j] = dry_air_column_density

            product_group = augmented_ch4_file.groups['PRODUCT']

            if 'dry_air_column_density' in product_group.variables.keys():
                break
            # Create the variables for the methane mixing ratio and the methane mixing ratio precision
            dacd = product_group.createVariable('dry_air_column_density',
                                                np.float32,
                                                ('time', 'scanline', 'ground_pixel'))

            dacd[0, :, :] = dry_air_column_densities
            augmented_ch4_file.close()

def calculate_dry_air_column_density_residuals(fitted_results):
    '''This function is for calculating the residual between the ERA5-derived dry air column density and the dry air
    column density calculated from the TROPOMI CH4 product.

    :param fitted_results: The model run we want to calculate all the residuals for.
    :type fitted_results: FittedResults
    '''

    start_date, end_date, model = fitted_results.run_name.split('-')

    # Read in the summary.csv file, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', header=0,
                             index_col=0)

    dry_air_column_dfs = []

    for date in tqdm(summary_df.index, desc='Calulating difference between ERA5 and TROPOMI dry air column densities on '
                                            + '-'.join(model.split('_')) + ' days'):

        era5_columns    = []
        tropomi_columns = []
        residuals       = []

        # Create string from the date datetime
        date_string = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

        # Create list of TROPOMI filenames that match this date. Sometimes there are two TROPOMI overpasses
        # that are a couple hours apart. Usually one overpass captures the whole study region.
        tropomi_overpasses = [file.split('/')[-1] for file in
                              glob.glob(
                                  ct.FILE_PREFIX + '/observations/NO2/' + date_string + '*.nc')]

        for filename in tropomi_overpasses:
            # Open the original CH4 file.
            original_ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')
            # Open the augmented CH4 file containing all the predictions.
            augmented_ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/augmented_observations/'
                                             + fitted_results.run_name + '/' + filename, 'r')

            # Get the array of the ERA5-derived dry air column density.
            era5_column = np.array(augmented_ch4_file.groups['PRODUCT'].variables['dry_air_column_density'])[0]

            # Generate the arrays of CH4 pixel centre latitudes and longitudes, same between both files.
            pixel_centre_latitudes = np.array(original_ch4_file.groups['PRODUCT'].variables['latitude'])[0]
            pixel_centre_longitudes = np.array(original_ch4_file.groups['PRODUCT'].variables['longitude'])[0]

            # Get the arrays of the TROPOMI QA values and dry air sub columns
            qa_values = np.array(original_ch4_file.groups['PRODUCT'].variables['qa_value'])[0]
            tropomi_sub_columns = np.array(
                original_ch4_file.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['INPUT_DATA'].variables[
                    'dry_air_subcolumns'])[0]

            for i in range(original_ch4_file.groups['PRODUCT'].dimensions['scanline'].size):
                for j in range(original_ch4_file.groups['PRODUCT'].dimensions['ground_pixel'].size):
                    if (ct.STUDY_REGION['Permian_Basin'][2] < pixel_centre_latitudes[i, j] <
                        ct.STUDY_REGION['Permian_Basin'][
                            3]) and \
                            (ct.STUDY_REGION['Permian_Basin'][0] < pixel_centre_longitudes[i, j] <
                             ct.STUDY_REGION['Permian_Basin'][1]):
                        if qa_values[i, j] >= 0.5:
                            tropomi_column = sum(tropomi_sub_columns[i, j, :])
                            era5_columns.append(era5_column[i, j])
                            tropomi_columns.append(tropomi_column)
                            residuals.append(era5_column[i, j] - tropomi_column)

        date_df = pd.DataFrame(list(zip([date]*len(residuals), era5_columns, tropomi_columns, residuals)),
               columns =['Date', 'ERA5_dry_air_column', 'TROPOMI_dry_air_column', 'Residuals'])

        dry_air_column_dfs.append(date_df)

    full_df = pd.concat(dry_air_column_dfs, ignore_index=True)

    full_df.to_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/dry_air_column_density_residuals.csv', index=False)

def calculate_pixel_area(latitudes, longitudes):
    '''A function for calculating the area of a pixel in square meters.

    :param latitudes: The four latitudes of the corners of the pixel.
    :type latitudes ndarray
    :param longitudes: The four longitudes of the corners of the pixel.
    :type longitudes: ndarray.

    :returns: The area of the pixel in square meters.'''

    # Add the first element to the end of each list to close the shape.
    latitudes.append(latitudes[0])
    longitudes.append(longitudes[0])

    # Specify the WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    # Create the polygon from the coordinates
    poly = geometry.Polygon([[x, y] for x, y in zip(longitudes, latitudes)])
    # Calculate the area
    area = abs(geod.geometry_area_perimeter(poly)[0]) #[m^2]

    return area

def calculate_final_results(fitted_results):
    '''This function is for writing a single large .csv file summarising all plotable quantities by date.

    :param fitted_results: The model run we want to summarise our plotable quantities for.
    :type fitted_results: FittedResults
    '''

    start_date, end_date, model_type = fitted_results.run_name.split('-')

    # Open the data summary csv file for this model run, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', index_col=0)

    # Create the pandas dataframe that we will eventually save as a csv file.
    final_results_df = pd.DataFrame(columns=('date',
                                             'day_id',
                                             'day_type',
                                             'N',
                                             'R',
                                             'flare_count',
                                             'noaa_background',
                                             'alpha_50',
                                             'alpha_16',
                                             'alpha_84',
                                             'beta_50',
                                             'beta_16',
                                             'beta_84',
                                             'gamma_50',
                                             'gamma_16',
                                             'gamma_84',
                                             'original_pixel_value_50',
                                             'augmented_pixel_value_50',
                                             'original_pixel_coverage',
                                             'augmented_pixel_coverage',
                                             'original_ch4_load',
                                             'original_ch4_load_precision',
                                             'fully_augmented_ch4_load',
                                             'fully_augmented_ch4_load_precision',
                                             'partially_augmented_ch4_load',
                                             'partially_augmented_ch4_load_precision',
                                             'original_no2_load',
                                             'original_no2_load_precision'))

    # Open the .csv containing the residuals of the dry air column densities, and calculate the RMS that we will use
    # to propogate the error from the dry air column density as we calculate the methane column density.
    dry_air_column_density_residual_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name +
                                                     '/dry_air_column_density_residuals.csv', header=0)
    residuals = np.array(dry_air_column_density_residual_df.Residuals)
    dry_air_column_density_error = np.std(residuals)

    # Open the .csv file containing the NOAA monthly CH4 background data
    noaa_ch4_df = pd.read_csv(ct.FILE_PREFIX + '/observations/NOAA/noaa_ch4_background.csv',
                              comment='#', delim_whitespace=True)
    # Create the interpolation function for determining the reference background level of methane.
    ch4_interpolation_function       = interp1d(noaa_ch4_df.decimal, noaa_ch4_df.average, kind='linear')
    ch4_error_interpolation_function = interp1d(noaa_ch4_df.decimal, noaa_ch4_df.average_unc, kind='linear')
    start_of_year              = datetime.datetime(year=2019, month=1, day=1)
    end_of_year                = datetime.datetime(year=2020, month=1, day=1)
    year_length                = end_of_year - start_of_year

    day_type = '-'.join(model_type.split('_'))

    for date in tqdm(summary_df.index, desc='Calculating results for ' + day_type + ' days'):

        # Set day_id, day_type, N and R
        day_id   = summary_df.day_id.loc[date]
        N        = summary_df.N.loc[date]
        R        = summary_df.R.loc[date]

        # Create string from the date datetime
        date_string = datetime.datetime.strptime(date, '%Y-%m-%d').strftime("%Y%m%d")

        # Determine the number of active flares spotted on this day.
        # Get the corrresponding VIIRS observation file, there should only be one.
        viirs_file = glob.glob(ct.FILE_PREFIX + '/observations/VIIRS/*' + date_string + '*.csv')[0]
        # Set number of active flares to zero and count up from here.
        flare_count = 0
        # Open the file.
        viirs_df = pd.read_csv(viirs_file)
        # Examine every active flare
        for index in viirs_df.index:
            # If latitudes of flare in study region:
            if (ct.STUDY_REGION['Permian_Basin'][2] < viirs_df.Lat_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][3]):
                # If longitudes of flare in study region:
                if (ct.STUDY_REGION['Permian_Basin'][0] < viirs_df.Lon_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][1]):
                    if viirs_df.Temp_BB[index] != 999999:
                        flare_count += 1

        # Determine the reference background level of methane.
        current_date    = datetime.datetime.strptime(date, '%Y-%m-%d')
        elapsed_time    = current_date - start_of_year
        fraction        = elapsed_time / year_length
        decimal_year    = 2019 + fraction
        noaa_background = float(ch4_interpolation_function(decimal_year))
        noaa_background_error = float(ch4_error_interpolation_function(decimal_year))

        # Determine the centile levels of alpha, beta and gamma
        alpha_50 = fitted_results.median_values['alpha.' + str(day_id)]
        beta_50  = fitted_results.median_values['beta.' + str(day_id)]
        gamma_50 = fitted_results.median_values['gamma.' + str(day_id)]

        alpha_16, alpha_84 = fitted_results.credible_intervals['alpha.' + str(day_id)]
        beta_16, beta_84   = fitted_results.credible_intervals['beta.' + str(day_id)]
        gamma_16, gamma_84 = fitted_results.credible_intervals['gamma.' + str(day_id)]

        # ------------------------------------------------------------------------------------------------
        # The following key is needed to understand some list names.
        # Type 1 = a quantity calculated purely from TROPOMI CH4 pixels with a qa value of 0.5 or greater
        # Type 2 = a quantity calculated from TROPOMI CH4 pixels with a qa value of 0.5 or greater and predicted
        #          values of CH4 from NO2 pixel values that had a qa value of 0.75 or greater.
        # Type 3 = a quantity calculated from TROPOMI CH4 pixels with a qa value of 0.5 or greater and predicted
        #          values of CH4 from NO2 pixel values that had a qa value of 0.75 or greater, with the additional
        #          requirement that the NO2 pixel value was at least two standard deviations away from zero.
        # ------------------------------------------------------------------------------------------------

        # The following two lists are used to hold pixel values contained within the study region
        type1_pixel_values = []
        type2_pixel_values = []

        # The following list is used to hold [value, uncertaintly] pairs of mols of nitrogen dioxide contained
        # in a pixel.
        mols_nitrogen_dioxide = []

        # The following lists are used to hold [value, uncertainty] pairs of mols of methane contained in a pixel.
        type1_mols_methane = []
        type2_mols_methane = []
        type3_mols_methane = []

        # The following variables are used to track how the total pixel coverage in the study region changes with
        # the type of observation used
        total_pixels      = 0
        type1_pixel_count = 0
        type2_pixel_count = 0

        # Create list of TROPOMI filenames that match the date. Sometimes there are two TROPOMI overpasses
        # that are a couple hours apart. Usually one overpass captures the whole study region.
        tropomi_overpasses = [file.split('/')[-1] for file in
                              glob.glob(
                                  ct.FILE_PREFIX + '/observations/NO2/' + date_string + '*.nc')]

        for filename in tropomi_overpasses:

            # Open the original CH4 file.
            original_ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')

            # Open the file of CH4 predictions.
            prediction_ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name + '/' + filename, 'r')

            # Open the original NO2 file.
            original_no2_file = nc4.Dataset(ct.FILE_PREFIX + '/observations/NO2/' + filename, 'r')

            # Generate the arrays of CH4 pixel centre latitudes and longitudes, same between both files.
            ch4_pixel_centre_latitudes  = np.array(original_ch4_file.groups['PRODUCT'].variables['latitude'])[0]
            ch4_pixel_centre_longitudes = np.array(original_ch4_file.groups['PRODUCT'].variables['longitude'])[0]

            # Get the arrays of pixel corner coordinates, needed to calculate pixel areas.
            ch4_pixel_corner_latitudes  = np.array(original_ch4_file.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['latitude_bounds'][0])
            ch4_pixel_corner_longitudes = np.array(original_ch4_file.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['longitude_bounds'][0])

            # Get the arrays of the original methane observational data.
            original_ch4_pixel_values     = np.array(original_ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio'])[0]
            original_ch4_pixel_precisions = np.array(original_ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio_precision'])[0]
            ch4_qa_values                 = np.array(original_ch4_file.groups['PRODUCT'].variables['qa_value'])[0]

            # Get the arrays of the predicted methane observational data.
            predicted_ch4_pixel_values       = np.array(prediction_ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio'])[0]
            predicted_ch4_pixel_precisions   = np.array(prediction_ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio_precision'])[0]
            predicted_ch4_qa_values          = np.array(prediction_ch4_file.groups['PRODUCT'].variables['prediction_pixel_qa_value'])[0]
            dry_air_column_densities         = np.array(prediction_ch4_file.groups['PRODUCT'].variables['dry_air_column_density'])[0]
            prediction_pixel_outside_of_zero = np.array(prediction_ch4_file.groups['PRODUCT'].variables['prediction_pixel_outside_of_zero'])[0]

            # Get the arrays needed for calculating total load of nitrogen dioxide in the study region.
            no2_pixel_centre_latitudes    = np.array(original_no2_file.groups['PRODUCT'].variables['latitude'])[0]
            no2_pixel_centre_longitudes   = np.array(original_no2_file.groups['PRODUCT'].variables['longitude'])[0]
            no2_pixel_corner_latitudes    = np.array(original_no2_file.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['latitude_bounds'][0])
            no2_pixel_corner_longitudes   = np.array(original_no2_file.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['longitude_bounds'][0])
            original_no2_pixel_values     = np.array(original_no2_file.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'])[0]
            original_no2_pixel_precisions = np.array(original_no2_file.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column_precision'])[0]
            no2_qa_values                 = np.array(original_no2_file.groups['PRODUCT'].variables['qa_value'])[0]

            for i in range(original_no2_file.groups['PRODUCT'].dimensions['scanline'].size):
                for j in range(original_no2_file.groups['PRODUCT'].dimensions['ground_pixel'].size):
                    if (ct.STUDY_REGION['Permian_Basin'][2] < no2_pixel_centre_latitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][3]) and \
                            (ct.STUDY_REGION['Permian_Basin'][0] < no2_pixel_centre_longitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][1]):
                        if no2_qa_values[i, j] >= 0.75:
                            pixel_area = calculate_pixel_area(list(no2_pixel_corner_latitudes[i, j, :]),
                                                              list(no2_pixel_corner_longitudes[i, j, :]))
                            pixel_value            = original_no2_pixel_values[i, j]
                            pixel_value_error      = original_no2_pixel_precisions[i, j]
                            no2_mols               = pixel_value * pixel_area
                            no2_mols_precision     = no2_mols * (pixel_value / pixel_value_error)
                            mols_nitrogen_dioxide.append([no2_mols, no2_mols_precision])

            for i in range(original_ch4_file.groups['PRODUCT'].dimensions['scanline'].size):
                for j in range(original_ch4_file.groups['PRODUCT'].dimensions['ground_pixel'].size):
                    if (ct.STUDY_REGION['Permian_Basin'][2] < ch4_pixel_centre_latitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][3]) and \
                            (ct.STUDY_REGION['Permian_Basin'][0] < ch4_pixel_centre_longitudes[i, j] < ct.STUDY_REGION['Permian_Basin'][1]):
                        total_pixels += 1
                        if ch4_qa_values[i, j] >= 0.5:
                            pixel_area = calculate_pixel_area(list(ch4_pixel_corner_latitudes[i, j, :]),
                                                              list(ch4_pixel_corner_longitudes[i, j, :]))
                            pixel_value            = original_ch4_pixel_values[i, j]
                            delta_pixel_value      = pixel_value - noaa_background
                            pixel_value_error      = np.sqrt(original_ch4_pixel_precisions[i, j]**2 + noaa_background_error**2)
                            dry_air_column_density = dry_air_column_densities[i, j]  # [mol m^-2]
                            methane_mols           = delta_pixel_value * 1e-9 * dry_air_column_density * pixel_area  # [mol], need the 1e-9 to convert out from ppbv
                            methane_mols_precision = methane_mols * np.sqrt((pixel_value_error / delta_pixel_value) ** 2
                                                                            + (dry_air_column_density_error / dry_air_column_density) ** 2)

                            # Append necessary values to lists for type 1 quantities
                            type1_pixel_count += 1
                            type1_pixel_values.append(pixel_value)
                            type1_mols_methane.append([methane_mols, methane_mols_precision])

                            # Append necessary values to lists for type 2 quantities
                            type2_pixel_count += 1
                            type2_pixel_values.append(pixel_value)
                            type2_mols_methane.append([methane_mols, methane_mols_precision])

                            # Append necessary values to lists for type 3 quantities (needed only for mols methane)
                            type3_mols_methane.append([methane_mols, methane_mols_precision])

                        elif (predicted_ch4_qa_values[i, j] >= 0.75) and (predicted_ch4_pixel_values[i, j] < 1e30):
                            pixel_area = calculate_pixel_area(list(ch4_pixel_corner_latitudes[i, j, :]),
                                                              list(ch4_pixel_corner_longitudes[i, j, :]))
                            pixel_value            = predicted_ch4_pixel_values[i, j]
                            delta_pixel_value      = pixel_value - noaa_background
                            pixel_value_error      = predicted_ch4_pixel_precisions[i, j]
                            dry_air_column_density = dry_air_column_densities[i, j]  # [mol m^-2]
                            methane_mols           = delta_pixel_value * 1e-9 * dry_air_column_density * pixel_area  # [mol], need the 1e-9 to convert out from ppbv
                            methane_mols_precision = methane_mols * np.sqrt((pixel_value_error / delta_pixel_value) ** 2
                                                                            + (dry_air_column_density_error / dry_air_column_density) ** 2)

                            # Append necessary values to lists for type 2 quantities
                            type2_pixel_count += 1
                            type2_pixel_values.append(pixel_value)
                            type2_mols_methane.append([methane_mols, methane_mols_precision])

                            if prediction_pixel_outside_of_zero[i, j] == 1: # If == 1, then the NO2 pixel was two standard deviations away from zero at least.
                                # Append necessary values to lists for type 3 quantities (needed only for mols methane)
                                type3_mols_methane.append([methane_mols, methane_mols_precision])

        type1_coverage = type1_pixel_count / total_pixels
        type2_coverage = type2_pixel_count / total_pixels

        median_type1_value = np.median(type1_pixel_values)
        median_type2_value = np.median(type2_pixel_values)

        no2_load           = sum(np.array(mols_nitrogen_dioxide)[:, 0])
        no2_load_precision = np.sqrt(
            sum([precision ** 2 for precision in np.array(mols_nitrogen_dioxide)[:, 1]]))

        type1_methane_load           = sum(np.array(type1_mols_methane)[:, 0])
        type1_methane_load_precision = np.sqrt(
            sum([precision ** 2 for precision in np.array(type1_mols_methane)[:, 1]]))

        type2_methane_load           = sum(np.array(type2_mols_methane)[:, 0])
        type2_methane_load_precision = np.sqrt(
            sum([precision ** 2 for precision in np.array(type2_mols_methane)[:, 1]]))

        type3_methane_load = sum(np.array(type3_mols_methane)[:, 0])
        type3_methane_load_precision = np.sqrt(
            sum([precision ** 2 for precision in np.array(type3_mols_methane)[:, 1]]))

        # Append all quantities for this day as a new row in the dataframe.
        final_results_df = final_results_df.append({'date': date,
                                                                'day_id': day_id,
                                                                'day_type': day_type,
                                                                'N': N,
                                                                'R': R,
                                                                'flare_count': flare_count,
                                                                'noaa_background': noaa_background,
                                                                'alpha_50': alpha_50,
                                                                'alpha_16': alpha_16,
                                                                'alpha_84': alpha_84,
                                                                'beta_50': beta_50,
                                                                'beta_16': beta_16,
                                                                'beta_84': beta_84,
                                                                'gamma_50': gamma_50,
                                                                'gamma_16': gamma_16,
                                                                'gamma_84': gamma_84,
                                                                'original_pixel_value_50': median_type1_value,
                                                                'augmented_pixel_value_50': median_type2_value,
                                                                'original_pixel_coverage': type1_coverage,
                                                                'augmented_pixel_coverage': type2_coverage,
                                                                'original_ch4_load': type1_methane_load,
                                                                'original_ch4_load_precision': type1_methane_load_precision,
                                                                'fully_augmented_ch4_load': type2_methane_load,
                                                                'fully_augmented_ch4_load_precision': type2_methane_load_precision,
                                                                'partially_augmented_ch4_load': type3_methane_load,
                                                                'partially_augmented_ch4_load_precision': type3_methane_load_precision,
                                                                'original_no2_load': no2_load,
                                                                'original_no2_load_precision': no2_load_precision},
                                                               ignore_index=True)

        final_results_df.to_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/final_results.csv',
                                      index=False)

def parse_tropomi_values(file_name):
    '''This function is for opening an original and augmented tropomi observation, determining if there is an
    original observation or augmented observation at that location and returning some lists.

    :param file_name: The filename.
    :type file_name: str
    '''

    augmented_ch4_dataset = nc4.Dataset(ct.AUGMENTED_PERMIAN_OBSERVATIONS + '/CH4/' + file_name, 'r')

    prediction_pixel_values     = np.array(augmented_ch4_dataset.groups['PRODUCT'].variables['methane_mixing_ratio'][0])
    prediction_precision_values = np.array(augmented_ch4_dataset.groups['PRODUCT'].variables['methane_mixing_ratio_precision'][0])
    prediction_pixel_qa_values  = np.array(augmented_ch4_dataset.groups['PRODUCT'].variables['prediction_pixel_qa_value'][0])

    original_ch4_dataset = nc4.Dataset(ct.PERMIAN_OBSERVATIONS + '/CH4/' + file_name, 'r')

    original_pixel_values     = np.array(original_ch4_dataset.groups['PRODUCT'].variables['methane_mixing_ratio'][0])
    original_precision_values = np.array(original_ch4_dataset.groups['PRODUCT'].variables['methane_mixing_ratio_precision'][0])
    original_pixel_qa_values  = np.array(original_ch4_dataset.groups['PRODUCT'].variables['qa_value'][0])
    scanline = original_ch4_dataset.groups["PRODUCT"].variables["scanline"].size
    ground_pixel = original_ch4_dataset.groups["PRODUCT"].variables["ground_pixel"].size
    latitudes = np.array(original_ch4_dataset.groups["PRODUCT"].variables["latitude"])[0]
    longitudes = np.array(original_ch4_dataset.groups["PRODUCT"].variables["longitude"])[0]

    reduced_original_latitudes = []
    reduced_original_longitudes = []
    reduced_original_ch4_values = []
    reduced_original_ch4_precisions = []
    reduced_prediction_latitudes = []
    reduced_prediction_longitudes = []
    reduced_prediction_ch4_values = []
    reduced_prediction_ch4_precisions = []

    for i in range(scanline):
        for j in range(ground_pixel):
            if ct.STUDY_REGION["Permian_Basin"][2] < latitudes[i, j] < ct.STUDY_REGION["Permian_Basin"][3]:
                if ct.STUDY_REGION["Permian_Basin"][0] < longitudes[i, j] < ct.STUDY_REGION["Permian_Basin"][1]:
                    if original_pixel_qa_values[i, j] >= 0.5:
                        reduced_original_latitudes.append(latitudes[i, j])
                        reduced_original_longitudes.append(longitudes[i, j])
                        reduced_original_ch4_values.append(original_pixel_values[i, j])
                        reduced_original_ch4_precisions.append(original_precision_values[i, j])

                    elif 1. >= prediction_pixel_qa_values[i, j] >= 0.75:
                        reduced_prediction_latitudes.append(latitudes[i, j])
                        reduced_prediction_longitudes.append(longitudes[i, j])
                        reduced_prediction_ch4_values.append(prediction_pixel_values[i, j])
                        reduced_prediction_ch4_precisions.append(prediction_precision_values[i, j])

    parsed_quantitues = {
        "reduced_original_latitudes": reduced_original_latitudes,
        "reduced_original_longitudes": reduced_original_longitudes,
        "reduced_original_ch4_values": reduced_original_ch4_values,
        "reduced_original_ch4_precisions": reduced_original_ch4_precisions,
        "reduced_prediction_latitudes": reduced_prediction_latitudes,
        "reduced_prediction_longitudes": reduced_prediction_longitudes,
        "reduced_prediction_ch4_values": reduced_prediction_ch4_values,
        "reduced_prediction_ch4_precisions": reduced_prediction_ch4_precisions
    }

    return parsed_quantitues

def compare_with_bremen(start_date, end_date):
    '''This function is for creating a few .csv files that are comparing the TROPOMI, Bremen and augmented TROPOMI
    observations with one another.

    :param start_date: The start date of the date range that we want to compare values for. Formatted as %Y-%m-%d
    :type start_date: str
    :param end_date: The end date of the date range that we want to compare values for. Formatted as %Y-%m-%d
    :type end_date: str
    '''

    # Create a list to hold pandas dataframes that you will eventually concatenate together?
    original_dfs = []
    prediction_dfs = []

    # Create the range of dates that we want to iterate over
    #date_range = pd.date_range(start=start_date, end=end_date)
    # Just use the data-rich days
    data_rich_df = pd.read_csv("data/20190101-20191231-data_rich/summary.csv")
    date_range = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in data_rich_df.date]

    for date in tqdm(date_range, desc="Iterating over days in range"):

        # Open the Bremen file and reduce it to values that are contained within the study region.
        # Screen according to qa value and mask values to 1e32 that are not "good".
        date_string = date.strftime("%Y%m%d")
        bremen_filename = "ESACCI-GHG-L2-CH4-CO-TROPOMI-WFMD-" + date_string + "-fv2.nc"

        # Open the Bremen file
        bremen_file = nc4.Dataset(ct.BREMEN_OBSERVATIONS + '/' + bremen_filename, 'r')

        latitudes = np.array(bremen_file.variables["latitude"])
        longitudes = np.array(bremen_file.variables["longitude"])
        ch4 = np.array(bremen_file.variables["xch4"])
        ch4_precisions = np.array(bremen_file.variables["xch4_uncertainty"])
        qa_value = np.array(bremen_file.variables["xch4_quality_flag"])

        # Reduce the values to what is in the study region on this day
        reduced_bremen_latitudes = []
        reduced_bremen_longitudes = []
        reduced_bremen_ch4 = []
        reduced_bremen_ch4_precisions = []

        for i in range(len(latitudes)):
            if ct.STUDY_REGION["Permian_Basin"][2] < latitudes[i] < ct.STUDY_REGION["Permian_Basin"][3]:
                if ct.STUDY_REGION["Permian_Basin"][0] < longitudes[i] < ct.STUDY_REGION["Permian_Basin"][1]:
                    reduced_bremen_latitudes.append(latitudes[i])
                    reduced_bremen_longitudes.append(longitudes[i])

                    # 0 is good, 1 is bad
                    if qa_value[i] == 0:
                        reduced_bremen_ch4.append(ch4[i])
                        reduced_bremen_ch4_precisions.append(ch4_precisions[i])
                    elif qa_value[i] == 1:
                        reduced_bremen_ch4.append(1e32)
                        reduced_bremen_ch4_precisions.append(1e32)

        # Check if reduced ch4 list is not empty
        if reduced_bremen_ch4:

            tropomi_filenames = [file.split("/")[-1] for file in
                                 glob.glob(ct.PERMIAN_OBSERVATIONS + '/CH4/' + date_string + '*.nc')]

            reduced_original_latitudes = []
            reduced_original_longitudes = []
            reduced_original_ch4_values = []
            reduced_original_ch4_precisions = []
            reduced_prediction_latitudes = []
            reduced_prediction_longitudes = []
            reduced_prediction_ch4_values = []
            reduced_prediction_ch4_precisions = []


            for filename in tropomi_filenames:
                try:
                    parsed_quantities = parse_tropomi_values(filename)
                except FileNotFoundError:
                    continue

                reduced_original_latitudes.extend(parsed_quantities.get("reduced_original_latitudes"))
                reduced_original_longitudes.extend(parsed_quantities.get("reduced_original_longitudes"))
                reduced_original_ch4_values.extend(parsed_quantities.get("reduced_original_ch4_values"))
                reduced_original_ch4_precisions.extend(parsed_quantities.get("reduced_original_ch4_precisions"))
                reduced_prediction_latitudes.extend(parsed_quantities.get("reduced_prediction_latitudes"))
                reduced_prediction_longitudes.extend(parsed_quantities.get("reduced_prediction_longitudes"))
                reduced_prediction_ch4_values.extend(parsed_quantities.get("reduced_prediction_ch4_values"))
                reduced_prediction_ch4_precisions.extend(parsed_quantities.get("reduced_prediction_ch4_precisions"))

            # Check if there were any original TROPOMI values
            if reduced_original_ch4_values:

                #TODO add in all this stuff but also for predictions

                bremen_values_colocated_with_original = []
                bremen_precisions_colocated_with_original = []
                original_values_colocated_with_bremen = []
                original_precisions_colocated_with_bremen = []

                bremen_values_colocated_with_predictions = []
                bremen_precisions_colocated_with_predictions = []
                prediction_values_colocated_with_bremen = []
                prediction_precisions_colocated_with_bremen = []


                # Interpolate the Bremen data to the original TROPOMI observations:
                bremen_values_interpolated_to_original = griddata((reduced_bremen_longitudes,
                                                                   reduced_bremen_latitudes),
                                                                  reduced_bremen_ch4,
                                                                  (reduced_original_longitudes,
                                                                   reduced_original_latitudes),
                                                                  method='linear')

                bremen_precisions_interpolated_to_original = griddata((reduced_bremen_longitudes,
                                                                       reduced_bremen_latitudes),
                                                                       reduced_bremen_ch4_precisions,
                                                                      (reduced_original_longitudes,
                                                                        reduced_original_latitudes),
                                                                      method='linear')
                # Interpolate the Bremen data to the predicted TROPOMI observations:
                bremen_values_interpolated_to_predictions = griddata((reduced_bremen_longitudes,
                                                                      reduced_bremen_latitudes),
                                                                     reduced_bremen_ch4,
                                                                     (reduced_prediction_longitudes,
                                                                      reduced_prediction_latitudes),
                                                                     method='linear')

                bremen_precisions_interpolated_to_predictions = griddata((reduced_bremen_longitudes,
                                                                          reduced_bremen_latitudes),
                                                                         reduced_bremen_ch4_precisions,
                                                                         (reduced_prediction_longitudes,
                                                                          reduced_prediction_latitudes),
                                                                         method='linear')


                for i in range(len(reduced_original_ch4_values)):

                    # Check if the Bremen data passed the qa threshold here, janky way of doing it
                    if bremen_values_interpolated_to_original[i] < 3000.:
                        bremen_values_colocated_with_original.append(bremen_values_interpolated_to_original[i])
                        bremen_precisions_colocated_with_original.append(bremen_precisions_interpolated_to_original[i])
                        original_values_colocated_with_bremen.append(reduced_original_ch4_values[i])
                        original_precisions_colocated_with_bremen.append(reduced_original_ch4_precisions[i])

                for i in range(len(reduced_prediction_ch4_values)):

                    # Check if the Bremen data passed the qa threshold here, janky way of doing it
                    if bremen_values_interpolated_to_predictions[i] < 3000.:
                        bremen_values_colocated_with_predictions.append(bremen_values_interpolated_to_predictions[i])
                        bremen_precisions_colocated_with_predictions.append(bremen_precisions_interpolated_to_predictions[i])
                        prediction_values_colocated_with_bremen.append(reduced_prediction_ch4_values[i])
                        prediction_precisions_colocated_with_bremen.append(reduced_prediction_ch4_values[i])

                original_df = pd.DataFrame(list(zip(original_values_colocated_with_bremen,
                                                    original_precisions_colocated_with_bremen,
                                                    bremen_values_colocated_with_original,
                                                    bremen_precisions_colocated_with_original)),
                                       columns=["Original_TROPOMI",
                                                "Original_TROPOMI_precision",
                                                "Bremen",
                                                "Bremen_precision"])

                prediction_df = pd.DataFrame(list(zip(prediction_values_colocated_with_bremen,
                                                      prediction_precisions_colocated_with_bremen,
                                                      bremen_values_colocated_with_predictions,
                                                      bremen_precisions_colocated_with_predictions)),
                                             columns=["Predicted_TROPOMI",
                                                      "Predicted_TROPOMI_precision",
                                                      "Bremen",
                                                      "Bremen_precision"])

                original_dfs.append(original_df)
                prediction_dfs.append(prediction_df)

    all_original_dfs = pd.concat(original_dfs, ignore_index=True)
    all_prediction_dfs = pd.concat(prediction_dfs, ignore_index=True)

    all_original_dfs.to_csv("Bremen_comparision_with_original.csv", index=False)
    all_prediction_dfs.to_csv("Bremen_comparison_with_predictions.csv", index=False)

