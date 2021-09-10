from src import constants as ct
from src import results as sr
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

def prepare_data_poor_dataset_for_cmdstanpy(run_name, date):
    #TODO make docstring

    df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', delimiter=',',
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
    start_date, end_date, model = run_name.split('-')
    data_rich_run_name          = start_date + '-' + end_date + '-data_rich'
    data_rich_results           = sr.FittedResults(data_rich_run_name)

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

    with open(ct.FILE_PREFIX + '/outputs/' + run_name + '/dummy/data.json', 'w') as outfile:
        json.dump(data, outfile)

def prepare_data_rich_dataset_for_cmdstanpy(run_name):
    '''This function takes the "dataset.cvs" file located at "data/run_name" and turns it into json
    that is suitable for usage by the cmdstanpy package (.csv files are unabled to be provided as data when we
    fit our models).

    :param run_name: Name of the model run.
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
    data['sigma_N']     = avg_sigma_N
    data['sigma_C']     = avg_sigma_C

    with open(ct.FILE_PREFIX + '/data/' + run_name + '/data.json', 'w') as outfile:
        json.dump(data, outfile)

def unpack_no2(filename):
    '''This function opens a "raw" TROPOMI observation file of :math:`\mathrm{NO}_2` and returns arrays of quantities
    of interest that we need for our analysis.

    :param filename: The name of the file, should be in the format YYYYMMDDTHHMMSS.nc .
    :type filename: string
    '''

    # Open the NO2 file.
    file = nc4.Dataset(ct.FILE_PREFIX + '/observations/NO2/' + filename,'r')

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
    ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')

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
    # TODO augment docstring.

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
    # TODO augment docstring.

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
    #TODO Make docstring, and eventually have arguments that control the date range to make the dataset for.
    #   For now no arguments as I build the function.

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

def create_dataset_data_poor_days(run_name):
    # TODO Make docstring

    start_date, end_date, model = run_name.split('-')

    start_datetime = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end_datetime   = datetime.datetime.strptime(end_date, "%Y%m%d").date()

    total_days         = 0
    data_poor_days     = 0
    total_observations = 0

    # It should be okay to have some Day IDs be shared between data-rich and data poor days.
    # FittedResults will always be segregated between data-rich and data-poor days, and
    # the "plotables" csv file will let you know if a day was data-rich or data-poor, so as long as we remember
    # to check that parameter, we should always be able to load up posterior parameter estimates for the correct day.
    day_id = 1  # Stan starts counting from 1!

    # Empty list to hold dataframes for each day's observations when checks are passed.
    daily_dfs = []

    # A summary dataframe for overall metrics of for this run's data.
    summary_df = pd.DataFrame(columns=(('date', 'day_id', 'N', 'R')))

    # Create the list of dates to iterate over.
    num_days  = (end_datetime - start_datetime).days + 1
    date_list = [start_datetime + datetime.timedelta(days=x) for x in range(num_days)]

    # Read in the summary of the data-rich days so that we can skip days that were included in the data-rich model run.
    data_rich_summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + start_date + '-' + end_date + '-data_rich/summary.csv')
    data_rich_date_list  = list(data_rich_summary_df.date)

    # For every date in range for this model run:
    for date in tqdm(date_list, desc='Processing TROPOMI observations for data-poor days'):
        if date.strftime('%Y-%m-%d') not in data_rich_date_list:

            total_days += 1

            # Create string from the date datetime
            date_string = date.strftime("%Y%m%d")

            # Create list of TROPOMI filenames that match this date. Sometimes there are two TROPOMI overpasses
            # that are a couple hours apart. Usually one overpass captures the whole study region.
            tropomi_overpasses = [file.split('/')[-1] for file in
                                  glob.glob(
                                      ct.FILE_PREFIX + '/observations/NO2/' + date_string + '*.nc')]

            total_obs_CH4   = []
            total_sigma_C   = []
            total_obs_NO2   = []
            total_sigma_N   = []
            total_latitude  = []
            total_longitude = []

            for overpass in tropomi_overpasses:
                obs_CH4, sigma_C, obs_NO2, sigma_N, latitude, longitude = get_colocated_measurements(overpass)

                total_obs_CH4.extend(obs_CH4)
                total_sigma_C.extend(sigma_C)
                total_obs_NO2.extend(obs_NO2)
                total_sigma_N.extend(sigma_N)
                total_latitude.extend(latitude)
                total_longitude.extend(longitude)

            # If there are more than 2 co-located measurements on this day ... (Note: this may change)
            # Currently 2 because that's the minimum needed to determine correlation coefficient R
            if len(total_obs_NO2) >= 2:

                r, p_value = stats.pearsonr(total_obs_NO2, total_obs_CH4)

                data_poor_days     += 1
                total_observations += len(total_obs_NO2)

                # Append summary of this day to the summary dataframe.
                summary_df = summary_df.append({'date': date,
                                                'day_id': day_id,
                                                'N': len(total_obs_NO2),
                                                'R': round(r, 2)},
                                               ignore_index=True)

                # Create a dataframe containing the observations for this day.
                day_df = pd.DataFrame(list(zip([day_id] * len(total_obs_NO2),
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
                daily_dfs.append(day_df)

                # Increment day_id
                day_id += 1

    # Sort the summary dataframe by date.
    summary_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/summary.csv', index=False)

    # Concatenate the daily dataframes together to make the dataset dataframe. Leave sorted by Day_ID.
    dataset_df = pd.concat(daily_dfs)
    dataset_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', index=False)

    f = open(ct.FILE_PREFIX + "/data/" + run_name + "/summary.txt", "a")
    f.write("Total number of days in range: " + str(total_days) + '\n')
    f.write("Total number of data-rich days in range: " + str(data_poor_days) + '\n')
    f.write("Total number of observations in range: " + str(total_observations) + '\n')
    f.close()

def create_dataset_data_rich_days(run_name):
    #TODO Make docstring

    start_date, end_date, model = run_name.split('-')

    start_datetime = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end_datetime   = datetime.datetime.strptime(end_date, "%Y%m%d").date()

    total_days         = 0
    data_rich_days     = 0
    total_observations = 0

    day_id = 1 # Stan starts counting from 1!

    # Empty list to hold dataframes for each day's observations when checks are passed.
    daily_dfs = []

    # A summary dataframe for overall metrics of for this run's data.
    summary_df = pd.DataFrame(columns=(('Date', 'Day_ID', 'M', 'R')))

    # Create the list of dates to iterate over.
    num_days  = (end_datetime - start_datetime).days + 1
    date_list = [start_datetime + datetime.timedelta(days=x) for x in range(num_days)]

    # For every date in range for this model run:
    for date in tqdm(date_list, desc='Processing TROPOMI observations for data-rich days'):

        total_days += 1

        # Create string from the date datetime
        date_string = date.strftime("%Y%m%d")

        # Create list of TROPOMI filenames that match this date. Sometimes there are two TROPOMI overpasses
        # that are a couple hours apart. Usually one overpass captures the whole study region.
        tropomi_overpasses = [file.split('/')[-1] for file in
                                   glob.glob(
                                       ct.FILE_PREFIX + '/observations/NO2/' + date_string + '*.nc')]

        total_obs_CH4   = []
        total_sigma_C   = []
        total_obs_NO2   = []
        total_sigma_N   = []
        total_latitude  = []
        total_longitude = []

        for overpass in tropomi_overpasses:

            obs_CH4, sigma_C, obs_NO2, sigma_N, latitude, longitude = get_colocated_measurements(overpass)

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

                data_rich_days += 1
                total_observations += len(total_obs_NO2)

                # Append summary of this day to the summary dataframe.
                summary_df = summary_df.append({'Date': date, 'Day_ID': day_id, 'M': len(total_obs_NO2),
                                                'R': round(r, 2)},
                                               ignore_index=True)

                # Create a dataframe containing the observations for this day.
                day_df = pd.DataFrame(list(zip([day_id] * len(total_obs_NO2),
                                               [date] * len(total_obs_NO2),
                                               total_obs_NO2,
                                               total_obs_CH4,
                                               total_sigma_N,
                                               total_sigma_C,
                                               total_latitude,
                                               total_longitude)),
                                      columns=('Day_ID', 'Date', 'obs_NO2', 'obs_CH4',
                                               'sigma_N', 'sigma_C', 'latitude', 'longitude'))

                # Append the dataframe to this day to the list of dataframes to later concatenate together.
                daily_dfs.append(day_df)

                # Increment day_id
                day_id += 1

    # Sort the summary dataframe by date.
    summary_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/summary.csv', index=False)

    # Concatenate the daily dataframes together to make the dataset dataframe. Leave sorted by Day_ID.
    dataset_df = pd.concat(daily_dfs)
    dataset_df.to_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', index=False)

    f = open(ct.FILE_PREFIX + "/data/" + run_name + "/summary.txt", "a")
    f.write("Total number of days in range: " + str(total_days) + '\n')
    f.write("Total number of data-rich days in range: " + str(data_rich_days) + '\n')
    f.write("Total number of observations in range: " + str(total_observations) + '\n')
    f.close()

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

    for date in tqdm(summary_df.index, desc='Calculating predictions for ' + '-'.join(model_type.split('_')) + ' days'):

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
                                      ct.FILE_PREFIX + '/observations/NO2/' + date_string + '*.nc')]

            for filename in tropomi_overpasses:

                # Unpack some of the data into arrays
                no2_pixel_values, no2_pixel_precisions, no2_pixel_centre_latitudes, no2_pixel_centre_longitudes, no2_qa_values \
                    = unpack_no2(filename)

                ch4_pixel_values, ch4_pixel_precisions, ch4_pixel_centre_latitudes, ch4_pixel_centre_longitudes, ch4_qa_values \
                    = unpack_ch4(filename)

                # Open the original CH4 file again to access some dimensions
                ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')

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
                augmented_methane_predictor_pixel_outside_of_zero =  np.full(ch4_pixel_values.shape, 1e32)

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

    # Read in the summary.csv file, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', header=0,
                             index_col=0)

    dry_air_column_dfs = []

    for date in tqdm(summary_df.index, desc='Calulating difference between ERA5 and TROPOMI dry air column densities'):

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
                                             + fitted_results.run_name + '/data_rich_days/' + filename, 'r')

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

    full_df.to_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/dry_air_column_densities.csv', index=False)

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

def write_plotable_quantities_csv_file(fitted_results):
    '''This function is for writing a single large .csv file summarising all plotable quantities by date.

    :param fitted_results: The model run we want to summarise our plotable quantities for.
    :type fitted_results: FittedResults
    '''

    start_date, end_date, model_type = fitted_results.run_name.split('-')

    # Open the data summary csv file for this model run, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', index_col=0)

    # Create the pandas dataframe that we will eventually save as a csv file.
    plotable_quantities_df = pd.DataFrame(columns=('date',
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
                                                     '/dry_air_column_densities.csv', header=0)
    residuals = np.array(dry_air_column_density_residual_df.Residuals)
    dry_air_column_density_error = np.std(residuals)

    # Open the .csv file containing the NOAA monthly CH4 background data
    noaa_ch4_df = pd.read_csv(ct.FILE_PREFIX + '/data/noaa_ch4_background.csv',
                              comment='#', delim_whitespace=True)
    # Create the interpolation function for determining the reference background level of methane.
    ch4_interpolation_function = interp1d(noaa_ch4_df.decimal, noaa_ch4_df.average, kind='linear')
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

        # Determine the centile levels of alpha, beta and gamma
        alpha_50 = fitted_results.median_values['alpha.' + str(day_id)]
        beta_50  = fitted_results.median_values['beta.' + str(day_id)]
        gamma_50 = fitted_results.median_values['gamma.' + str(day_id)]

        alpha_16, alpha_84 = fitted_results.credible_intervals['alpha.' + str(day_id)]
        beta_16, beta_84   = fitted_results.credible_intervals['beta.' + str(day_id)]
        gamma_16, gamma_84 = fitted_results.credible_intervals['gamma.' + str(day_id)]

        # ------------------------------------------------------------------------------------------------
        # The following key is needed to understand some list names.
        # Type 1 = a qauntity calculated purely from TROPOMI CH4 pixels with a qa value of 0.5 or greater
        # Type 2 = a quantity calculated from TROPOMI CH4 pixels with a qa value of 0.5 or greater and predicted
        #          values of CH4 from NO2 pixel values that had a qa value of 0.75 or greater.
        # Type 3 = a quantity calculated from TROPOMI CH4 pixels with a qa value of 0.5 or greater and predicted
        #          values of CH4 from NO2 pixel values that had a qa value of 0.75 or greater, with the additional
        #          requirement that the NO2 pixel value was at least two standard deivations away from zero.
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
                            pixel_value_error      = original_ch4_pixel_precisions[i, j]
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
        plotable_quantities_df = plotable_quantities_df.append({'date': date,
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

        plotable_quantities_df.to_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/plotable_quantities.csv',
                                      index=False)

def calculate_prediction_vs_heldout_pixel_value(fitted_results):
    '''This function is for creating a csv file where one column is the heldout CH4 pixel, and then the predicted pixel value
    from the dropout model for that location.

    :param fitted_results: The results of the dropout model run.
    :type fitted_results: FittedResults
    '''

    # Read in the dropout_dataset.csv file, index by date
    dropout_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/dropout_dataset.csv', header=0)

    pixels_df = pd.DataFrame(columns=('Date', 'heldout_pixel_values', 'colocated_prediction_values', 'residuals'))

    for i in tqdm(range(len(dropout_df.index)), desc='Comparing heldout pixels to predicted values'):
        prediction, precision = fitted_results.predict_ch4(dropout_df.obs_NO2.iloc[i],
                                                           dropout_df.sigma_N.iloc[i],
                                                           dropout_df.Day_ID.iloc[i])

        original_pixel = dropout_df.obs_CH4.iloc[i]

        residual = original_pixel - prediction

        pixels_df = pixels_df.append({'Date': dropout_df.Date.iloc[i],
                                      'heldout_pixel_values': original_pixel,
                                      'colocated_prediction_values': prediction,
                                      'residual': residual},
                                     ignore_index=True)

    pixels_df.to_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name.split('/')[0] + '/heldout_pixels_and_colocated_predictions.csv')

def calculate_prediction_vs_poor_pixel_value(fitted_results):
    '''This function is for creating a csv file where one column is the CH4 pixel value for pixels with qa < 0.5, and
    then the corresponding prediction at that location when possible from an NO2 pixel with qa >= 0.75.

    :param fitted_results: The results of this model run
    :type fitted_results: FittedResults
    '''

    # Read in the summary.csv file, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', header=0,
                             index_col=0)

    methane_pixel_dfs = []

    for date in tqdm(summary_df.index, desc='Comparing poor methane pixels to predicted values'):

        poor_pixel_values      = []
        colocated_predictions  = []
        residuals              = []

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
                                             + fitted_results.run_name + '/data_rich_days/' + filename, 'r')

            # Generate the arrays of CH4 pixel centre latitudes and longitudes, same between both files.
            pixel_centre_latitudes = np.array(original_ch4_file.groups['PRODUCT'].variables['latitude'])[0]
            pixel_centre_longitudes = np.array(original_ch4_file.groups['PRODUCT'].variables['longitude'])[0]

            # Get the arrays of the QA values
            qa_values                = np.array(original_ch4_file.groups['PRODUCT'].variables['qa_value'])[0]
            predicted_pixel_qa_value = np.array(augmented_ch4_file.groups['PRODUCT'].variables['prediction_pixel_qa_value'])[0]

            # Get the arrays of the pixel values
            original_pixel_values  = np.array(original_ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio'])[0]
            predicted_pixel_values = np.array(augmented_ch4_file.groups['PRODUCT'].variables['methane_mixing_ratio'])[0]


            for i in range(original_ch4_file.groups['PRODUCT'].dimensions['scanline'].size):
                for j in range(original_ch4_file.groups['PRODUCT'].dimensions['ground_pixel'].size):
                    if (ct.STUDY_REGION['Permian_Basin'][2] < pixel_centre_latitudes[i, j] <
                        ct.STUDY_REGION['Permian_Basin'][
                            3]) and \
                            (ct.STUDY_REGION['Permian_Basin'][0] < pixel_centre_longitudes[i, j] <
                             ct.STUDY_REGION['Permian_Basin'][1]):
                        if (qa_values[i, j] < 0.5) and \
                                (original_pixel_values[i, j] < 1e30) and \
                                (predicted_pixel_qa_value[i, j] >= 0.75) and \
                                (predicted_pixel_values[i, j] < 1e30):
                            poor_pixel_values.append(original_pixel_values[i, j])
                            colocated_predictions.append(predicted_pixel_values[i, j])
                            residuals.append(original_pixel_values[i, j] - predicted_pixel_values[i, j])

        date_df = pd.DataFrame(list(zip([date] * len(residuals), poor_pixel_values, colocated_predictions, residuals)),
                               columns=['date', 'poor_pixel_values', 'colocated_prediction_value', 'residuals'])

        methane_pixel_dfs.append(date_df)

    full_df = pd.concat(methane_pixel_dfs, ignore_index=True)

    full_df.to_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/poor_pixels_and_colocated_predictions.csv',
                   index=False)







