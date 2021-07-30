from src import constants as ct
import netCDF4 as nc4
import numpy as np
from scipy.interpolate import griddata
from scipy import stats
import csv
import os
import pandas as pd
from tqdm import tqdm
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
    if 'daily_mean_error' in run_name:
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
            if (ct.STUDY_REGION["Permian_Basin"][2] < no2_pixel_centre_latitudes[i, j] < ct.STUDY_REGION["Permian_Basin"][3]) and \
                    (ct.STUDY_REGION["Permian_Basin"][0] < no2_pixel_centre_longitudes[i, j] < ct.STUDY_REGION["Permian_Basin"][1]):

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
            if (ct.STUDY_REGION["Permian_Basin"][2] < ch4_pixel_centre_latitudes[i, j] < ct.STUDY_REGION["Permian_Basin"][3]) and \
                    (ct.STUDY_REGION["Permian_Basin"][0] < ch4_pixel_centre_longitudes[i, j] < ct.STUDY_REGION["Permian_Basin"][1]):

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
            obs_NO2.append(no2_pixel_values[j] * 1e6)  # Convert to micro mol / m^2
            obs_CH4.append(interpolated_ch4_pixel_values[j])
            # Append the CH4 and NO2 precisions to the relevant lists.
            sigma_N.append(no2_pixel_precisions[j] * 1e6)  # Convert to micro mol / m^2
            sigma_C.append(interpolated_ch4_pixel_precisions[j])
            # Append the latitudes and longitudes to the relevant lists.
            latitude.append(no2_pixel_centre_latitudes[j])
            longitude.append(no2_pixel_centre_longitudes[j])

    return obs_CH4, sigma_C, obs_NO2, sigma_N, latitude, longitude

def create_dataset(run_name):
    #TODO Make docstring, and eventually have arguments that control the date range to make the dataset for.
    #For now no arguments as I build the function.

    start_date, end_date, model = run_name.split('-')

    total_days         = 0
    data_rich_days     = 0
    total_observations = 0

    day_id = 0

    with open(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', 'w') as csvfile, \
        open(ct.FILE_PREFIX + '/data/' + run_name + '/summary.csv', 'w') as summaryfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(('Day_ID','Date','obs_NO2', 'obs_CH4', 'sigma_N', 'sigma_C', 'latitude', 'longitude'))

        summarywriter = csv.writer(summaryfile, delimiter=',')
        summarywriter.writerow(('Day_ID', 'Date', 'M', 'R'))

        for file in tqdm(os.listdir(ct.FILE_PREFIX + '/observations/NO2')):
            date = file[:8]

            # If date in correct range for this run...
            if int(start_date) <= int(date) <= int(end_date):

                total_days += 1

                obs_CH4, sigma_C, obs_NO2, sigma_N, latitude, longitude = get_colocated_measurements(file)

                # If there are more than 100 co-located measurements on this day ...
                if len(obs_NO2) >= 100:

                    r, p_value = stats.pearsonr(obs_NO2, obs_CH4)

                    # if R >= 0.4 ...
                    if r >= 0.4:

                        # Write to the dataset.
                        day_id             += 1
                        data_rich_days     += 1
                        total_observations += len(obs_NO2)

                        summarywriter.writerow((day_id, date, len(obs_NO2), r))

                        for i in range(len(obs_NO2)):
                            csvwriter.writerow((day_id, date, round(obs_NO2[i], 2), round(obs_CH4[i], 2),
                                                round(sigma_N[i], 2), round(sigma_C[i], 2),
                                            round(latitude[i], 2), round(longitude[i], 2)))

    f = open(ct.FILE_PREFIX + "/data/" + run_name + "/summary.txt", "a")
    f.write("Total number of days in range: " + str(total_days) + '\n')
    f.write("Total number of data-rich days in range: " + str(data_rich_days) + '\n')
    f.write("Total number of observations in range: " + str(total_observations) + '\n')
    f.close()