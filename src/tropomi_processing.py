from src import constants as ct
import netCDF4 as nc4
import numpy as np
from scipy.interpolate import griddata
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

def prepare_dataset_for_cmdstanpy(run_name):
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
    day_id  = list(df.Day_ID)
    D       = int(np.max(day_id))
    M       = len(obs_no2)

    group_sizes = []
    for i in range(D):
        day = i+1
        size = len(df[df.Day_ID == day])
        group_sizes.append(size)

    avg_sigma_N = []
    avg_sigma_C = []

    for i in range(D):
        day = i+1
        mean_sigma_N = round(np.mean(df[df.Day_ID == day].sigma_N),2)
        mean_sigma_C = round(np.mean(df[df.Day_ID == day].sigma_C),2)

        avg_sigma_N.append(mean_sigma_N)
        avg_sigma_C.append(mean_sigma_C)

    sigma_N = list(df.sigma_N)
    sigma_C = list(df.sigma_C)

    data = {}
    data['M']           = M
    data['D']           = D
    data['day_id']      = day_id
    data['group_sizes'] = group_sizes
    data['NO2_obs']     = obs_no2
    data['CH4_obs']     = obs_ch4
    if ('daily_mean_error' in run_name) or ('non_centered' in run_name):
        data['sigma_N']     = avg_sigma_N
        data['sigma_C']     = avg_sigma_C
    else:
        data['sigma_N'] = sigma_N
        data['sigma_C'] = sigma_C

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
        if (no2_pixel_values[j] < 1e30) and (interpolated_ch4_pixel_values[j] < 1e30):
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

def create_dataset(run_name):
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
    for date in tqdm(date_list, desc='Processing TROPOMI observations'):

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

def augment_data_rich_days(fitted_results):
    '''This function is for creating augmented .nc TROPOMI files that were part of the hierarchical model run
    using "data rich" days.

    :param fitted_results: The hierarchical model run.
    :type fitted_results: FittedResults
    '''

    # Make the directory for the run name
    try:
        os.makedirs(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name + '/data_rich')
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name + '/data_rich')
        os.makedirs(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name + '/data_rich')

    # Read the summary.csv file for this model run to get the dates of the days that were used in the hierarchical fit.
    # Index by day
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', index_col=0)

    for date in tqdm(summary_df.index, desc='Augmenting observations for data-rich days'):

        day_id = int(summary_df.loc[date].Day_ID)

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

            # Open the original NO2 file again to access some dimensions
            ch4_file = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')

            # Created the augmented netCDF4 dataset
            augmented_file = nc4.Dataset(ct.FILE_PREFIX + '/augmented_observations/' + fitted_results.run_name +
                                         '/data_rich/' + filename,
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
            augmented_methane_mixing_ratio           = np.full(ch4_pixel_values.shape, 1e32)
            augmented_methane_mixing_ratio_precision = np.full(ch4_pixel_precisions.shape, 1e32)

            # Iterate over the raw CH4 data, and calculate predictions and overwrite as necessary.
            for i in range(ch4_file.groups['PRODUCT'].dimensions['scanline'].size):
                for j in range(ch4_file.groups['PRODUCT'].dimensions['ground_pixel'].size):
                    if (ct.EXTENT['Permian_Basin'][2] < ch4_pixel_centre_latitudes[i, j] < ct.EXTENT['Permian_Basin'][3]) and \
                            (ct.EXTENT['Permian_Basin'][0] < ch4_pixel_centre_longitudes[i, j] < ct.EXTENT['Permian_Basin'][1]):
                        # Perform a prediction whenever there is an available 'good' NO2 pixel.
                        if interpolated_no2_qa_values[i, j] >= 0.75:
                            obs_no2               = interpolated_no2_pixel_values[i, j] * 1e3  # Convert to mmol / m^2
                            sigma_N               = interpolated_no2_pixel_value_precisions[i, j] * 1e3  # Convert to mmol / m^2
                            prediction, precision = fitted_results.predict_ch4(obs_no2, sigma_N, day_id)
                            augmented_methane_mixing_ratio[i, j] = prediction
                            augmented_methane_mixing_ratio_precision[i, j] = precision

            methane_mixing_ratio[0,:,:]           = augmented_methane_mixing_ratio
            methane_mixing_ratio_precision[0,:,:] = augmented_methane_mixing_ratio_precision
            augmented_file.close()

def create_methane_mass_time_series(fitted_results):
    '''This function is for creating a .csv file that describes a time series of calculated methane mass
    (with propogated errorbars), showing how the total mass detected over the study region changes with each variation
    of pixel usage.'''
    print(2)

def create_pixel_coverage_time_series(fitted_results):
    '''This function is for creating a .csv file that describes a time series of total pixel coverage in the Permian
    Basin, one time series for just "good" TROPOMI observations of methane and another for when we include augmented
    pixels.

    :param fitted_results: The model run we want to write a time series for.
    :type fitted_results: FittedResults
    '''

    # Open the summary csv file for this model run
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv')

    # Create the csv file that we will save our time series to
    time_series_df = pd.DataFrame(columns=('Date', 'Raw_coverage', 'Augmented_coverage'))

    for date in tqdm(summary_df.Date, desc='Creating pixel-coverage time series'):
        total_pixels     = 0
        qa_pixels        = 0
        all_pixels       = 0
        augmented_pixels = 0

        # Create string from the date datetime
        date_string = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

        # Create list of TROPOMI filenames that match this date. Sometimes there are two TROPOMI overpasses
        # that are a couple hours apart. Usually one overpass captures the whole study region.
        tropomi_overpasses = [file.split('/')[-1] for file in
                              glob.glob(
                                  ct.FILE_PREFIX + '/observations/NO2/' + date_string + '*.nc')]

        for filename in tropomi_overpasses:
            # Open the original CH4 file.
            original_ch4_file   = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')
            # Open the file of CH4 predictions.
            prediction_ch4_file =
            print(2)
            # Open both the original CH4 TROPOMI observation file and it's augmented counterpart.

    print(2)

def create_median_pixel_time_series(fitted_results):
    print(2)