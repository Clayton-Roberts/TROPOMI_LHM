import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from tqdm import tqdm
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import netCDF4 as nc4
import copy
import datetime
import glob
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import os
import shutil
import constants as ct
import results

class PlotHelper:
    #TODO make the docstring
    def __init__(self, filename, quantity,
                 qa_only=False,
                 include_predictions=False,
                 good_predictions_only=False,
                 show_precisions=False,
                 augmented_directory=None):

        self.figsize = (10.0, 6.0)
        self.extent  = (-106, -99, 29, 35)
        self.xticks  = range(-106, -99, 2)
        self.yticks  = range(29, 35, 2)

        date = filename.split("T")[0]

        if quantity == "NO2":
            # Open the TROPOMI observation file
            f = nc4.Dataset(ct.FILE_PREFIX + '/observations/NO2/' + filename, 'r')
            self.legend = "Column density [mmol m$^{-2}$]"
            self.title = 'Tropospheric column density NO$_2$' + \
                         '\n' + datetime.datetime.strptime(date, '%Y%m%d').strftime('%B %-d, %Y')
            self.vmax  = 0.1
            self.vmin  = 0.0

            # Access pixel values, convert to mmmol / m^2
            no2 = np.array(f.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'][0]) * 1e3


            if not qa_only:
                self.data = np.ma.masked_greater(no2, 1e30)
            # If you only want high quality pixels, need to do the below.
            else:
                filtered_no2 = np.empty(np.shape(no2))
                qa_values    = np.array(f.groups['PRODUCT'].variables['qa_value'][0])

                for i in range(np.shape(no2)[0]):
                    for j in range(np.shape(no2)[1]):
                        if qa_values[i, j] < 0.75:
                            filtered_no2[i, j] = 1e32
                        else:
                            filtered_no2[i, j] = no2[i, j]

                self.data = np.ma.masked_greater(filtered_no2, 1e30)

        elif quantity == 'CH4':
            # Open the TROPOMI observation file
            f = nc4.Dataset(ct.FILE_PREFIX + '/observations/CH4/' + filename, 'r')

            if not show_precisions:
                self.vmax  = 1900.0
                self.vmin  = 1820.0
                self.legend = "Mixing ratio [ppbv]"
                self.title = 'Column averaged mixing ratio CH$_4$' + \
                             '\n' + datetime.datetime.strptime(date, '%Y%m%d').strftime('%B %-d, %Y')

            else:
                self.vmax = 1.0
                self.vmin = 15.0
                self.legend = "Mixing ratio precision [ppbv]"
                self.title = 'Column averaged mixing ratio CH$_4$ precision' + \
                             '\n' + datetime.datetime.strptime(date, '%Y%m%d').strftime('%B %-d, %Y')

            # Access pixel values and precisions
            ch4         = np.array(f.groups['PRODUCT'].variables['methane_mixing_ratio'][0])
            precisions  = np.array(f.groups['PRODUCT'].variables['methane_mixing_ratio_precision'][0])

            if not qa_only:
                if not show_precisions:
                    data = ch4
                else:
                    data = precisions

            # If you only want high-quality pixels, have to do the below.
            else:
                filtered_ch4        = np.full(np.shape(ch4), 1e32)
                filtered_precisions = np.full(np.shape(ch4), 1e32)
                qa_values           = np.array(f.groups['PRODUCT'].variables['qa_value'][0])

                for i in range(np.shape(ch4)[0]):
                    for j in range(np.shape(ch4)[1]):
                        if qa_values[i, j] >= 0.5:
                            filtered_ch4[i, j]        = ch4[i, j]
                            filtered_precisions[i, j] = precisions[i, j]

                if not show_precisions:
                    data = filtered_ch4
                else:
                    data = filtered_precisions

            if include_predictions:
                #TODO change this hardcoded model run name
                g = nc4.Dataset(ct.FILE_PREFIX +
                                '/augmented_observations/' + augmented_directory + '/' +
                                filename,
                                'r')

                prediction_pixel_values     = np.array(g.groups['PRODUCT'].variables['methane_mixing_ratio'][0])
                prediction_precision_values = np.array(g.groups['PRODUCT'].variables['methane_mixing_ratio_precision'][0])
                prediction_pixel_qa_values  = np.array(g.groups['PRODUCT'].variables['prediction_pixel_qa_value'][0])

                for i in range(f.groups['PRODUCT'].dimensions['scanline'].size):
                    for j in range(f.groups['PRODUCT'].dimensions['ground_pixel'].size):
                        if (data[i, j] >= 1e32) and (prediction_pixel_values[i, j] < 1e32):
                            pixel_value = prediction_pixel_values[i, j]
                            precision   = prediction_precision_values[i, j]
                            if good_predictions_only:
                                if prediction_pixel_qa_values[i, j] >= 0.75:
                                    if not show_precisions:
                                        data[i, j] = pixel_value
                                    else:
                                        data[i, j] = precision
                            else:
                                if not show_precisions:
                                    data[i, j] = pixel_value
                                else:
                                    data[i, j] = precision

            self.data = np.ma.masked_greater(data, 1e30)

        # Acess pixel corner latitudes and longitudes
        lat_corners = np.array(
            f.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['latitude_bounds'][0])
        lon_corners = np.array(
            f.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['longitude_bounds'][0])

        rows    = np.shape(self.data)[0]
        columns = np.shape(self.data)[1]

        latitude_corners = np.empty((rows + 1, columns + 1))
        longitude_corners = np.empty((rows + 1, columns + 1))

        for i in range(rows):
            for j in range(columns):
                latitude_corners[i, j] = lat_corners[i][j][0]
                latitude_corners[i + 1, j] = lat_corners[i][j][1]
                latitude_corners[i + 1, j + 1] = lat_corners[i][j][2]
                latitude_corners[i, j + 1] = lat_corners[i][j][3]
                longitude_corners[i, j] = lon_corners[i][j][0]
                longitude_corners[i + 1, j] = lon_corners[i][j][1]
                longitude_corners[i + 1, j + 1] = lon_corners[i][j][2]
                longitude_corners[i, j + 1] = lon_corners[i][j][3]

        self.latitudes = latitude_corners
        self.longitudes = longitude_corners

# TODO this is used
def make_directory(date_range):
    '''This function makes a sub-directory in the figures directory that will hold the figures of this run.

    :param date_range: The date range of this analysis. Must be of the format "%Y%m%d-%Y%m%d"
    :type date_range: str
    '''
    try:
        os.makedirs(ct.FILE_PREFIX + '/figures/' + date_range)
    except FileExistsError:
        shutil.rmtree(ct.FILE_PREFIX + '/figures/' + date_range)
        os.makedirs(ct.FILE_PREFIX + '/figures/' + date_range)

# TODO this is to be saved
def tropomi_plot(date,
                 molecule,
                 plot_study_region=False,
                 qa_only=False,
                 show_flares=False,
                 include_predictions=False,
                 good_predictions_only=False,
                 show_precisions=False,
                 augmented_directory=None):
    '''This is a function for plotting TROPOMI observations of either :math:`\\mathrm{NO}_2` or :math:`\\mathrm{CH}_4`.

    :param date: Date you want to plot observations for, format as %Y-%m-%d
    :type date: str
    :param molecule: Molecule you want to plot observations of. Must be either "CH4" or "NO2".
    :type molecule: string
    :param plot_study_region: A flag to determine if you want to plot the study region or not.
    :type plot_study_region: bool
    :param qa_only: A flag to determine if you want to only use QA'd observations or not.
    :type qa_only: bool
    :param show_flares: A flag to determine if you want to plot flare stack locations that are "on" from VIIRS.
    :type show_flares: bool
    :param augment_ch4: A flag to determine if you want to show the possible augmented methane pixels.
    :type augment_ch4: bool
    '''

    file_date_prefix        = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    potential_tropomi_files = [file.split('/')[-1] for file in
                               glob.glob(ct.FILE_PREFIX + '/observations/CH4/' + file_date_prefix + '*.nc')]

    if len(potential_tropomi_files) > 1:
        print('Multiple files match the input date. Enter index of desired file:')
        for i in range(len(potential_tropomi_files)):
            print(f'[{i}]: {potential_tropomi_files[i]}')
        index = int(input('File index: '))
        file = potential_tropomi_files[index]
    else:
        file = potential_tropomi_files[0]

    # Set necessary details with the plot helper
    plot_helper = PlotHelper(file, molecule, qa_only, include_predictions, good_predictions_only, show_precisions,
                             augmented_directory=augmented_directory)

    # Get the outlines of counties
    reader   = shpreader.Reader(ct.FILE_PREFIX + '/misc/countyl010g_shp_nt00964/countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    # Set the figure size
    plt.figure(figsize=plot_helper.figsize)

    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set the ticks
    ax.set_xlabel('Longitude')
    ax.set_xticks(plot_helper.xticks)
    ax.set_ylabel('Latitude')
    ax.set_yticks(plot_helper.yticks)

    # Set up the limits to the plot
    ax.set_extent(plot_helper.extent, crs=ccrs.PlateCarree())

    if plot_study_region:
        # Create a Rectangle patch
        box       = ct.STUDY_REGION['Permian_Basin']
        rectangle = patches.Rectangle((box[0], box[2]),
                                    box[1] - box[0],
                                    box[3] - box[2],
                                    linewidth=2,
                                    edgecolor='r',
                                    facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rectangle)

    colors = copy.copy(cm.RdYlBu_r)
    colors.set_bad('grey', 1.0)

    # Define the coordinate system that the grid lons and grid lats are on
    plt.pcolormesh(plot_helper.longitudes,
                   plot_helper.latitudes,
                   plot_helper.data,
                   cmap=colors,
                   vmin=plot_helper.vmin,
                   vmax=plot_helper.vmax)

    if show_flares:
        # Find and open the relevant VIIRS observation file, there should only be one matching file in the list.
        viirs_file = glob.glob(ct.FILE_PREFIX + '/observations/VIIRS/*' + file_date_prefix + '*.csv')[0]
        viirs_df = pd.read_csv(viirs_file)

        flare_location_df = pd.DataFrame(columns=('Latitude', 'Longitude'))

        # Examine every active flare
        for index in viirs_df.index:

            # If latitudes of flare in study region:
            if (ct.STUDY_REGION['Permian_Basin'][2] < viirs_df.Lat_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][3]):

                # If longitudes of flare in study region:
                if (ct.STUDY_REGION['Permian_Basin'][0] < viirs_df.Lon_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][1]):

                    if viirs_df.Temp_BB[index] != 999999:
                        flare_location_df = flare_location_df.append({'Latitude': viirs_df.Lat_GMTCO[index],
                                                                      'Longitude': viirs_df.Lon_GMTCO[index]},
                                                                     ignore_index=True)

        ax.scatter(flare_location_df.Longitude, flare_location_df.Latitude,
                   marker="^",
                   s=20,
                   color='limegreen',
                   edgecolors='black')

    cbar = plt.colorbar()
    cbar.set_label(plot_helper.legend, rotation=270, labelpad=18)

    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='lightgray')
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor="lightgray")
    plt.title(plot_helper.title)
    plt.tight_layout()
    plt.show()

# TODO this is to be saved
def trace(fitted_model,
          parameter,
          date=None,
          show_warmup_draws=False):
    """Plot the trace and posterior distribution of a sampled scalar parameter.

    :param fitted_model: The object contained the results from a fitted model.
    :type fitted_model: FittedModel
    :param parameter: The name of the parameter. Must be one of 'alpha', 'beta' or 'gamma' with the appropriate date
        argument, or 'mu_alpha', 'mu_beta', 'sigma_alpha', 'sigma_beta' or 'rho'.
    :type parameter: string
    :param date: Date you want to see parameter of interest for, in format YYYYMMDD. Only needed for parameter =
        'alpha', 'beta' or 'gamma'.
    :type date: string
    :param compare_to_ground_truth: A boolean to include a comparison to the ground truth value for this parameter
        if this model run was a test run with fake data, which is why it defaults to False. Do not use with real data.
    :type compare_to_ground_truth: Boolean
    :param show_warmup_draws: A boolean to show the warmup draws for each chain.
    :type show_warmup_draws: Boolean
    """

    # Humans think in terms of dates, stan thinks in terms of day ids. Need to be able to access parameter.day_id
    # using the passed date. Use the summary.csv file and index by date
    summary_df   = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_model.run_name + '/summary.csv', header=0, index_col=0)

    # Daily parameters are saved in stan as parameter.day_id .
    if parameter == 'alpha' or parameter == 'beta' or parameter == 'gamma':
        model_key  = parameter + '.' + str(int(summary_df.loc[date].day_id))
        title_date = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%B %-d, %Y")
    else:
        model_key = parameter
        start_date, end_date, model = fitted_model.run_name.split('-')
        title_date = datetime.datetime.strptime(start_date, "%Y%m%d").strftime("%B %-d, %Y") + ' - ' + \
                     datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%B %-d, %Y")

    # Greek letters for labels
    parameter_symbol = {
        'beta': r"$\mathregular{\beta}$",
        'alpha': r"$\mathregular{\alpha}$",
        'gamma': r"$\mathregular{\gamma}$",
        'mu_alpha': r"$\mathregular{\mu_{\alpha}}$",
        'mu_beta': r"$\mathregular{\mu_{\beta}}$",
        'sigma_alpha': r"$\mathregular{\sigma_{\alpha}}$",
        'sigma_beta': r"$\mathregular{\sigma_{\beta}}$",
        'rho': r"$\mathregular{\rho}$",
        'lp__' : 'log posterior'
    }

    # Units
    parameter_units = {
        'beta': r"[ppbv / (mmol m$^{-2}$)]",
        'alpha': "[ppbv]",
        'gamma': "[ppbv]",
        'mu_alpha': "[ppbv]",
        'mu_beta': r"[ppbv / (mmol m$^{-2}$)]",
        'sigma_alpha': "[ppbv]",
        'sigma_beta': r"[ppbv / (mmol m$^{-2}$)]",
        'rho': '',
        'lp__': ''
    }


    title = 'Trace and Posterior Distribution for ' + parameter_symbol[parameter] + '\n' + title_date

    # Create the top panel showing how the chains have mixed.
    plt.subplot(2, 1, 1)
    if show_warmup_draws:
        plt.plot(pd.concat([fitted_model.draws['warmup']['chain_1'][model_key],
                            fitted_model.draws['sampled']['chain_1'][model_key]],
                           ignore_index=True))
        plt.plot(pd.concat([fitted_model.draws['warmup']['chain_2'][model_key],
                            fitted_model.draws['sampled']['chain_2'][model_key]],
                           ignore_index=True))
        plt.plot(pd.concat([fitted_model.draws['warmup']['chain_3'][model_key],
                            fitted_model.draws['sampled']['chain_3'][model_key]],
                           ignore_index=True))
        plt.plot(pd.concat([fitted_model.draws['warmup']['chain_4'][model_key],
                            fitted_model.draws['sampled']['chain_4'][model_key]],
                           ignore_index=True))

    else:
        plt.plot(fitted_model.draws['sampled']['chain_1'][model_key])
        plt.plot(fitted_model.draws['sampled']['chain_2'][model_key])
        plt.plot(fitted_model.draws['sampled']['chain_3'][model_key])
        plt.plot(fitted_model.draws['sampled']['chain_4'][model_key])
    plt.xlabel('Samples')
    plt.ylabel(parameter_symbol[parameter] + ' ' + parameter_units[parameter])
    plt.axhline(fitted_model.median_values[model_key], color='black', lw=2, linestyle='--')
    if show_warmup_draws:
        plt.axvline(500, linestyle='--', color='grey', linewidth=0.5, label='Sampling begins')
    plt.axhline(fitted_model.credible_intervals[model_key][0], linestyle=':', color='k', alpha=0.2)
    plt.axhline(fitted_model.credible_intervals[model_key][1], linestyle=':', color='k', alpha=0.2)
    if show_warmup_draws:
        plt.legend()
    plt.title(title)

    # Create the bottom panel showing the distribution of the parameter.
    plt.subplot(2, 1, 2)
    plt.hist(fitted_model.full_trace[model_key], 50, density=True)
    sns.kdeplot(fitted_model.full_trace[model_key], shade=True)
    plt.xlabel(parameter_symbol[parameter] + ' ' + parameter_units[parameter])
    plt.ylabel('Density')
    plt.axvline(fitted_model.median_values[model_key], color='black', lw=2, linestyle='--', label='Median value')
    plt.axvline(fitted_model.credible_intervals[model_key][0], linestyle=':', color='k', alpha=0.2, label=r'68% CI')
    plt.axvline(fitted_model.credible_intervals[model_key][1], linestyle=':', color='k', alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.show()

def observations_scatterplot(date, run_name):
    '''
    This function is for plotting a simple scatterplot of observations. It will always include errorbars on the observations,
    and if it is a test dataset than you can select to have the latent values shown as well.

    :param date: Date of observations that you want to plot, formatted YYYYMMDD
    :type date: string
    :param run_name: Name of the model run.
    :type run_name: string
    :return:
    '''

    sns.set()
    np.random.seed(101)

    dataset_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dataset.csv', header=0)
    date_df    = dataset_df[dataset_df.date == date]

    plt.errorbar(date_df.obs_NO2, date_df.obs_CH4, yerr=date_df.sigma_C, xerr=date_df.sigma_N,
                 ecolor="blue",
                 capsize=3,
                 fmt='D',
                 mfc='w',
                 color='red',
                 ms=4,
                 zorder=1,
                 label='Observed values')

    plt.xlabel(r'NO$_{2}^{\mathrm{obs}}$ [$\mathregular{\mu}$ mol m$^{-2}$]')
    plt.ylabel(r'CH$_{4}^{\mathrm{obs}}$ [ppbv]')

    plt.title(datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%B %-d, %Y"))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def dropout_scatterplot(date, run_name):
    '''
    This function is for plotting a scatterplot showing which observations were dropped out on a given day.
    :param date: Date of observations that you want to plot, formatted YYYYMMDD
    :type date: string
    :param run_name: Name of the model run.
    :type run_name: string
    '''

    sns.set()
    np.random.seed(101)

    full_dropout_df   = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/dropout_dataset.csv')
    full_remaining_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + run_name + '/dropout/remaining_dataset.csv')

    date_dropout_df   = full_dropout_df[full_dropout_df.Date == date]
    date_remaining_df = full_remaining_df[full_remaining_df.Date == date]

    plt.scatter(date_dropout_df.obs_NO2, date_dropout_df.obs_CH4, color='red', label='Dropped observations', marker='D', zorder=2)
    plt.scatter(date_remaining_df.obs_NO2, date_remaining_df.obs_CH4, color='black', label='Remaining observations', marker='D', zorder=1)

    plt.xlabel(r'NO$_{2}^{\mathrm{obs}}$ [$\mathregular{\mu}$ mol m$^{-2}$]')
    plt.ylabel(r'CH$_{4}^{\mathrm{obs}}$ [ppbv]')

    plt.title(datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%B %-d, %Y"))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def regression_scatterplot(date, fitted_model, compare_to_ground_truth=False):
    '''This function is for plotting a scatterplot of observed values of NO2 and CH4 on a given date.

    :param date: Date of observations that you want to plot, formatted YYYYMMDD
    :type date: string
    :param fitted_model: FittedModel object that contains the results for the day we're interested in
    :type fitted_model: FittedModel
    :param compare_to_ground_truth: A boolean to include a comparison to the ground truth regression if this
        model run was a test run with fake data, which is why it defaults to False. Do not use with real data.
    :type compare_to_ground_truth: Boolean
    '''

    sns.set()
    np.random.seed(101)

    dataset_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_model.run_name + '/dataset.csv', header=0)
    date_df    = dataset_df[dataset_df.date == date]

    # Needed to plot the regression lines
    x_min, x_max = np.min(date_df.obs_NO2), np.max(date_df.obs_NO2)
    x_domain = np.linspace(x_min, x_max, 100)

    # Humans think in terms of dates, stan thinks in terms of day ids. Need to be able to access parameter.day_id
    # using the passed date. Use the summary.csv file and index by date
    summary_df   = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_model.run_name + '/summary.csv', header=0, index_col=0)

    day_id = int(summary_df.loc[date].day_id)

    draws = np.arange(len(fitted_model.full_trace['alpha.' + str(day_id)]))
    np.random.shuffle(draws)
    alpha = fitted_model.full_trace['alpha.' + str(day_id)][draws]
    beta  = fitted_model.full_trace['beta.' + str(day_id)][draws]
    for i in range(500):
        plt.plot(x_domain, alpha[i] + beta[i] * x_domain, color='black',
                 alpha=0.05, zorder=2)

    if compare_to_ground_truth:
        alphas_betas_gammmas_df = pd.read_csv(ct.FILE_PREFIX + '/test_suite/ground_truths/' + fitted_model.run_name +
                                              '/alphas_betas_gammas.csv', header=0, index_col=0) # Index by date

        ground_truth_alpha = alphas_betas_gammmas_df.loc[date, 'alpha']
        ground_truth_beta  = alphas_betas_gammmas_df.loc[date, 'beta']

        plt.plot(x_domain, ground_truth_alpha + ground_truth_beta * x_domain, color='lime', ls='--',
                 label='True line', zorder=3)

    plt.errorbar(date_df.obs_NO2, date_df.obs_CH4, yerr=date_df.sigma_C, xerr=date_df.sigma_N,
                 ecolor="blue",
                 capsize=3,
                 fmt='D',
                 mfc='w',
                 color='red',
                 ms=4,
                 zorder=1,
                 label='Observed values')

    plt.xlabel(r'NO$_{2}^{\mathrm{obs}}$ [$\mathregular{\mu}$ mol m$^{-2}$]')
    plt.ylabel(r'CH$_{4}^{\mathrm{obs}}$ [ppbv]')

    plt.title(datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%B %-d, %Y"))
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def alpha_beta_scatterplot(fitted_model, compare_to_ground_truth=False):
    '''This function is for plotting estimated values of :math:`\\alpha_d` vs :math:`\\beta_d` for a range of days
    :math:`d`.

    :param fitted_model: The fitted model that contains the range of days you want :math:`\\alpha_d`
        vs :math:`\\beta_d` plotted for.
    :type fitted_model: FittedModel
    :param compare_to_ground_truth: A boolean to include a comparison to the ground truth values of :math:`\\alpha_d`
        and :math:`\\beta_d` if this model run was a test run with fake data, which is why it defaults to False.
        Do not use with real data.
    :type compare_to_ground_truth: Boolean
    '''

    # Create the figure with the specified size
    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)

    # Add the subplot axes
    ax0 = fig.add_subplot(spec[:, :])

    # Add ground truth values if you want to and this is a test dataset.
    if compare_to_ground_truth:

        # Get ground truth values for alpha and beta
        alphas_betas_gammas_df = pd.read_csv(ct.FILE_PREFIX + '/test_suite/ground_truths/' + fitted_model.run_name + '/alphas_betas_gammas.csv')

        ground_truth_alphas = alphas_betas_gammas_df.alpha
        ground_truth_betas  = alphas_betas_gammas_df.beta

        ax0.scatter(ground_truth_alphas,
                    ground_truth_betas,
                    color='lime',
                    marker='D',
                    zorder=-1)

        # Get ground truth values for the bivariate distribution.
        hyperparameters_df = pd.read_csv(ct.FILE_PREFIX + '/test_suite/ground_truths/' + fitted_model.run_name + '/hyperparameter_truths.csv',
                                         header=0)

        ellipse(hyperparameters_df.rho[0],
                hyperparameters_df.sigma_alpha[0],
                hyperparameters_df.sigma_beta[0],
                hyperparameters_df.mu_alpha[0],
                hyperparameters_df.mu_beta[0],
                ax0,
                n_std=2.0,
                edgecolor='blue',
                facecolor='blue',
                alpha=0.3)

    beta_values       = []
    beta_error_bounds = []

    alpha_values = []
    alpha_error_bounds = []

    for parameter in fitted_model.parameter_list:
        if 'beta.' in parameter:
            beta_values.append(fitted_model.median_values[parameter])
            beta_error_bounds.append([fitted_model.median_values[parameter] - fitted_model.credible_intervals[parameter][0],
                                      fitted_model.credible_intervals[parameter][1] - fitted_model.median_values[parameter]])
        elif 'alpha.' in parameter:
            alpha_values.append(fitted_model.median_values[parameter])
            alpha_error_bounds.append(
                [fitted_model.median_values[parameter] - fitted_model.credible_intervals[parameter][0],
                 fitted_model.credible_intervals[parameter][1] - fitted_model.median_values[parameter]])

    ax0.errorbar(alpha_values,
                 beta_values,
                 yerr=np.array(beta_error_bounds).T,
                 xerr=np.array(alpha_error_bounds).T,
                 ecolor="blue",
                 capsize=3,
                 fmt='D',
                 mfc='w',
                 color='red',
                 ms=4)

    pearson = fitted_model.median_values['rho']

    sigma_alpha = fitted_model.median_values['sigma_alpha']
    mu_alpha    = fitted_model.median_values['mu_alpha']

    sigma_beta = fitted_model.median_values['sigma_beta']
    mu_beta    = fitted_model.median_values['mu_beta']

    ellipse(pearson,
            sigma_alpha,
            sigma_beta,
            mu_alpha,
            mu_beta,
            ax0,
            n_std=2.0,
            edgecolor='red',
            facecolor='red',
            alpha=0.3)

    ax0.set_ylabel(r'$\mathregular{\beta}$ [ppbv / $\mathregular{\mu}$ mol m$^{-2}$]')
    ax0.set_xlabel(r'$\mathregular{\alpha}$ [ppbv]')

    # Check if this is a test dataset or not
    if 'days' in fitted_model.run_name:
        plt.title('Test dataset')
    else:
        start_date, end_date, model = fitted_model.run_name.split('-')
        plt.title(datetime.datetime.strptime(start_date, "%Y%m%d").strftime("%B %-d, %Y") + ' - '
                  + datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%B %-d, %Y"))
    plt.tight_layout()
    plt.show()

def ellipse(correlation_coefficient, sigma_alpha, sigma_beta, mu_alpha, mu_beta, ax, n_std=3.0, facecolor='none', **kwargs):
    '''A function to add an ellipse to scatterplots (used only in plots of :math:`\\alpha_d` vs :math:`\\beta_d`).

    :param correlation_coefficient: The Pearson correlation coefficient between :math:`\\alpha` and :math:`\\beta`,
        estimated in our model as :math:`\\rho`.
    :type correlation_coefficient: float
    :param sigma_alpha: Estimated model parameter :math:`\\sigma_\\alpha`.
    :type sigma_alpha: float
    :param sigma_beta: Estimated model parameter :math:`\\sigma_\\beta`.
    :type sigma_beta: float
    :param mu_alpha: Estimated model parameter :math:`\\mu_\\alpha`.
    :type mu_alpha: float
    :param mu_beta: Estimated model parameter :math:`\\mu_\\beta`.
    :type mu_beta: float
    :param ax: The plot to add the ellipse patch to.
    :type ax: Axes
    :param n_std: The number of standard deviations to scale the ellipse out to, defaults to 3.
    :type n_std: float
    :param facecolor: Color of the ellipse.
    :type facecolor: string
    '''

    ell_radius_x = np.sqrt(1 + correlation_coefficient)
    ell_radius_y = np.sqrt(1 - correlation_coefficient)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    scale_x = sigma_alpha * n_std
    mean_x  = mu_alpha

    scale_y = sigma_beta * n_std
    mean_y  = mu_beta

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def reduced_chi_squared(run_name):
    '''
    This function is for plotting a histogram of reduced :math:`\\chi^2` values for each day that was included in the model
    run when observations were dropped out.
    :param run_name: The name of the model run (must include "/dropout").
    :type run_name: string
    '''

    start_date, end_date, model_type = run_name.split('-')

    reduced_chi_square_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + run_name + '/dropout/reduced_chi_squared.csv')

    sns.displot(reduced_chi_square_df.Reduced_chi_squared, kde=True)
    plt.xlabel(r'$\mathregular{\chi^2_{\nu}}$')

    # Check if this is a test dataset or not
    if 'days' in run_name:
        title = '\n Test Dataset'
    else:
        start_date, end_date, model = run_name.split('-')
        title = datetime.datetime.strptime(start_date, "%Y%m%d").strftime("%B %-d, %Y") + ' - ' + \
                 datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%B %-d, %Y")
    plt.title(title)

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/paper/' + model_type + '_reduced_chi_squared.pdf',
                bbox_inches='tight',
                pad_inches=0.01)

    # Show the plot on-screen.
    plt.show()

def residuals(daily_mean_error_model_run, individual_error_model_run=None):
    '''A function for plotting the residuals between actual and predicted results for the held-out set of observations.
    :param daily_mean_error_model_run: The name of the daily mean error model run to plot the residuals for.
    :type daily_mean_error_model_run: string
    :param individual_error_model_run: The name of the individual error model run to plot the residuals for if you're comparing two models, not required.
    :type individual_error_model_run: string .
    '''

    if individual_error_model_run:
        residual_df_2 = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + individual_error_model_run + '/dropout/residuals.csv',
                                    header=0)
        plt.hist(residual_df_2.Residuals, alpha=0.5, label='Individual error model')

    residual_df_1 = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + daily_mean_error_model_run + '/dropout/residuals.csv')

    sns.displot(residual_df_1.Residuals, kde=True)

    title = 'Residual comparison'
    # Check if this is a test dataset
    if 'days' in daily_mean_error_model_run:
        title += '\n' + 'Test dataset'
    else:
        start_date, end_date, model = daily_mean_error_model_run.split('-')
        title += '\n' + datetime.datetime.strptime(start_date, "%Y%m%d").strftime("%B %-d, %Y") + ' - ' + \
                 datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%B %-d, %Y")

    plt.xlabel(r'$\mathregular{CH_4^{pred}} - \mathregular{CH_4^{obs}}$ [ppbv]')
    plt.title(title)
    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/paper/dropout_residuals.pdf',
                bbox_inches='tight',
                pad_inches=0.01)

    # Show the plot on-screen.
    plt.show()

def beta_flare_time_series(fitted_results):
    '''This function is for plotting two time series together of :math:`\\beta` values and flare stack counts for
    the specified run.

    :param fitted_results: The fitted results.
    :type fitted_results: FittedResults
    '''

    # Read in the flare stack count time series.
    flare_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/flare_counts.csv', header=0)

    # Read in the summary.csv file, index by date
    summary_df   = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', header=0, index_col=0)

    # Create the time series of mean inferred beta values and their credible intervals.
    beta_df = pd.DataFrame(columns=('Date', 'Beta', 'Lower_bound_95_CI', 'Upper_bound_95_CI'))

    for date in summary_df.index:
        parameter = 'beta.' + str(int(summary_df.loc[date].Day_ID))
        mode_beta = fitted_results.median_values[parameter]
        lower_bound, upper_bound = fitted_results.credible_intervals[parameter]
        beta_df = beta_df.append({'Date': date,
                                  'Beta': mode_beta,
                                  'Lower_bound_95_CI': lower_bound,
                                  'Upper_bound_95_CI': upper_bound},
                                 ignore_index=True)

    beta_datetimes  = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in beta_df.Date]
    errors          = [[beta_df.Beta[i] - beta_df.Lower_bound_95_CI[i], beta_df.Upper_bound_95_CI[i] - beta_df.Beta[i]]
                       for i in beta_df.index]

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.errorbar(beta_datetimes,
                beta_df.Beta,
                yerr=np.array(errors).T,
                linestyle="None",
                ecolor="blue",
                fmt='D',
                mfc='w',
                color='blue',
                capsize=3,
                ms=5)
    # set x-axis label
    ax.set_xlabel("Date", fontsize=14)

    # set y-axis label
    ax.set_ylabel(r'$\beta$', color="blue", fontsize=14)

    # set y-axis colors
    ax.tick_params(axis='y', colors='blue')

    # Separate flare data into dates shared with beta and dates that are not.
    # Indexes of flare dates that we were able to calculate beta on.
    shared_index = flare_df[flare_df.Date.isin(beta_df.Date)].index.to_list()
    # Indexes of flare dates that we were not able to calculate beta on.
    unshared_index = flare_df[~flare_df.Date.isin(beta_df.Date)].index.to_list()

    shared_flare_dates     = flare_df.Date[shared_index]
    shared_flare_counts    = flare_df.Flare_count[shared_index]
    shared_flare_datetimes = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in shared_flare_dates]

    unshared_flare_dates     = flare_df.Date[unshared_index]
    unshared_flare_counts    = flare_df.Flare_count[unshared_index]
    unshared_flare_datetimes = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in unshared_flare_dates]

    # twin object for two different y-axes on the same plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.scatter(shared_flare_datetimes, shared_flare_counts, color="red", marker="x", s=30)
    ax2.scatter(unshared_flare_datetimes, unshared_flare_counts, color="grey", marker="x", s=10)
    ax2.set_ylabel("Flare count", color="red", fontsize=14)
    ax2.tick_params(axis='y', colors='red')

    # Add in the ticks
    # This next line will only work for "real" datasets. Shouldn't need to make this plot for test datasets anyway.
    start_date, end_date, model  = fitted_results.run_name.split('-')
    # Find the first tick
    first_tick_date = datetime.datetime.strptime(start_date, "%Y%m%d").replace(day=1).strftime("%Y-%m-%d")
    # Find the last tick
    last_tick_date  = (datetime.datetime.strptime(end_date, "%Y%m%d").replace(day=1) + datetime.timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d")
    # Use pandas to create the date range, ticks on first of the month ('MS').
    tick_locations = pd.date_range(first_tick_date, last_tick_date, freq='MS')
    tick_labels    = tick_locations.strftime('%b-%y')
    # Add ticks to the plot
    plt.setp(ax2, xticks=tick_locations,
             xticklabels=tick_labels)
    # Add vlines
    for location in tick_locations:
        plt.axvline(location, linestyle='--', color='grey', linewidth=0.5)

    # Create the title
    plt.title(datetime.datetime.strptime(start_date, '%Y%m%d').strftime('%B %-d, %Y') + ' - ' +
              datetime.datetime.strptime(end_date, '%Y%m%d').strftime('%B %-d, %Y'))

    plt.tight_layout()
    plt.show()

def alpha_time_series(fitted_results):
    '''This function is for plotting a time series of alpha and some other quantities.

    :param fitted_results: The results of this model run.
    :type fitted_results: FittedResults
    '''

    # Open the plotables csv file.
    plotables_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/plotable_quantities.csv',
                               header=0,
                               index_col=0)

    # Read in the summary data file with the R correlations.
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv')

    # Create the datetime objects for all the dates we have beta inferences for ("data rich" days).
    alpha_datetimes = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in plotables_df.index]

    # Calculate the difference between alpha and the noaa background.
    alpha_noaa_diff = [plotables_df.alpha_50[i] - plotables_df.noaa_background[i] for i in plotables_df.index]

    # Create figure, use gridspec to divide it up into subplots. 3-column subplot where first plot spans two columns.
    plt.figure(figsize=(12.0, 4.45))
    G = gridspec.GridSpec(1, 3)
    # Need just a little bit of space between the subplots to have y-axis tick marks in between.
    G.update(wspace=0.19)

    #-------------------------------------------------------------------------------------------------
    # Right hand side plot (the time series). Needs to span the first two columns.
    ax_1 = plt.subplot(G[0, 1:])
    ax_1.set_xlabel("Date", fontsize=12)

    # Plot the time series of beta.
    ax_1.scatter(alpha_datetimes,
                 alpha_noaa_diff,
                 color='black')

    # Plot the text of panel B, right hand side
    ax_1.text(0.03, 0.95,
              'B',
              color='red',
              transform=ax_1.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center', )

    #-------------------------------------------------------------------------------------------------
    # Make a twin axes object that shares the x axis with the time series of beta.
    ax_twin = ax_1.twinx()

    # Plot the time series of flare count on this twin axes object.
    ax_twin.scatter(alpha_datetimes,
                    plotables_df.partially_augmented_ch4_load * 16.04 / 1e6 / 1e3, # Convert to grams, convert to tonnes, convert to kilotonnes
                    color="red",
                    marker="x",
                    s=30,
                    alpha=0.7)

    # Make the tick marks on the y axis of the flare count time series red, but don't include the numbers.
    ax_twin.tick_params(axis='y', colors='red', labelright=True)
    ax_twin.set_ylabel('Observed methane load above \n NOAA background [kilotonnes]',
                       color='red',
                       rotation=270,
                       labelpad=15,
                       fontsize=12)

    # Move the time series of beta on top of the time series of flare count.
    ax_1.set_zorder(2)
    ax_twin.set_zorder(1)
    ax_1.patch.set_visible(False)

    # Figure out the locations of the first and last tick mark in the date range of this run.
    start_date, end_date, model = fitted_results.run_name.split('-')
    first_tick_date = datetime.datetime.strptime(start_date, "%Y%m%d").replace(day=1).strftime("%Y-%m-%d")
    last_tick_date = (
            datetime.datetime.strptime(end_date, "%Y%m%d").replace(day=1) + datetime.timedelta(days=32)).replace(
        day=1).strftime("%Y-%m-%d")

    # Use pandas to create the date range, ticks on first of the month ('MS'), every two months ('2MS').
    tick_locations = pd.date_range(first_tick_date, last_tick_date, freq='2MS')
    # Change the tick labels to be abbreviated month and year.
    tick_labels = tick_locations.strftime('%b-%y')

    # Add date ticks to the x axis of the time series plot.
    plt.setp(ax_twin,
             xticks=tick_locations,
             xticklabels=tick_labels)

    # Add vertical dashed lines at the location of the date tick marks.
    for location in tick_locations:
        ax_twin.axvline(location, linestyle='--', color='grey', linewidth=0.5)

    # Add a horizontal line at zero.
    ax_1.axhline(0.0, linestyle='--', color='grey', linewidth=0.5)

    #-------------------------------------------------------------------------------------------------
    # Left hand side plot (the cross plot). Needs to span the third column and share y axis with the time series.
    ax_2 = plt.subplot(G[0, 0], sharey=ax_1)

    # Set the tick marks on y axis of the cross plot to be red, on the left hand side of the plot, include numbers.
    ax_2.tick_params(axis='y',
                     colors='black',
                     labelleft=False)
    ax_2.tick_params(axis='x',
                     colors='red')
    ax_2.yaxis.tick_right()

    # Define the colors that we will use for the plots.
    colors = copy.copy(cm.viridis)

    # Plot the cross plot of beta and flare count, including 95% CI on beta.
    im = ax_2.scatter(plotables_df.partially_augmented_ch4_load * 16.04 / 1e6 / 1e3, # Convert to grams, convert to tonnes, convert to kilotonnes
                      alpha_noaa_diff,
                      c=summary_df.R,
                      cmap=colors,
                      s=60,
                      alpha=0.7)
    # create an axes on the right side of ax_2. The width of cax will be 5%
    # of ax and the padding between cax and ax_2 will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax_2)
    cax     = divider.append_axes("left", size="10%", pad=0.05)
    plt.colorbar(im,
                 cax=cax)
    cax.text(-2.0, 0.5,
             'Pearson correlation coefficient',
             rotation=90,
             transform=cax.transAxes,
             horizontalalignment='center',
             verticalalignment='center'
             )
    cax.yaxis.tick_left()

    # Plot the text of panel A
    ax_2.text(0.05, 0.95,
              'A',
              color='red',
              transform=ax_2.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center')

    # Set the x-axis label on the cross plot.
    ax_2.set_xlabel('Observed methane load above \n NOAA background [kilotonnes]',
                    fontsize=12,
                    labelpad=0,
                    color='red')

    ax_2.tick_params(axis='y', colors='black', labelright=False)

    # Plot some red text to show that the red y-axis is the flare count. Plot in axes coordinates of the cross plot.
    ax_2.text(1.07, 1.05,
              r'$\alpha$ - NOAA background [ppbv]',
              color='black',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_2.transAxes,
              fontsize=12)

    ax_2.grid(which='both',
              linestyle='dashed',
              linewidth=0.5,
              color='grey')

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/autosaved/alpha_time_series.pdf',
                bbox_inches='tight',
                pad_inches=0.01)

    # Show the plot on-screen.
    plt.show()

def dry_air_column_density_cross_plot(fitted_results):
    '''This function is for plotting the ERA5-derived dry air column densities against the column densities derived
    from the TROPOMI CH4 data product.

    :param fitted_results: The model run in question.
    :type fitted_results: FittedResults'''

    # Read in the summary.csv file, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', header=0,
                             index_col=0)

    # Read in the dry air column density dataframe
    dry_air_column_density_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/dry_air_column_densities.csv', header=0)

    for date in tqdm(summary_df.index, desc='Creating dry air subcolumn cross plot'):

        # Want to plot by date to see if there's anything weird going on.
        sub_df = dry_air_column_density_df[dry_air_column_density_df.Date == date]

        plt.scatter(sub_df.ERA5_dry_air_column,
                    sub_df.TROPOMI_dry_air_column,
                    alpha=0.3)

    x = np.linspace(np.min(dry_air_column_density_df.ERA5_dry_air_column),
                    np.max(dry_air_column_density_df.ERA5_dry_air_column))
    y = x * 1.
    plt.xlabel(r'ERA5-derived dry air column density [mol m$^{-2}$]')
    plt.ylabel(r'TROPOMI-derived dry air column density [mol m$^{-2}$]')
    plt.plot(x, y, color='red')

    # Set the title on the plot using the start and end date of the model run.
    start_date, end_date, model = fitted_results.run_name.split('-')
    plt.title(datetime.datetime.strptime(start_date, '%Y%m%d').strftime('%B %-d, %Y') + ' - ' +
              datetime.datetime.strptime(end_date, '%Y%m%d').strftime('%B %-d, %Y'))

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/paper/dry_air_column_density_crossplot.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.01)

    plt.show()

def heldout_pixel_predicted_pixel_cross_plot(fitted_results):
    '''This function is for plotting heldout pixels of methane against predicted values at the same location from the
    dropout model.

    :param fitted_results: The model run we want to plot the pixels for.
    :type fitted_results: FittedResults
    '''

    # Read in the summary.csv file, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', header=0,
                             index_col=0)

    # Read in the dry air column density datafram
    pixels_df = pd.read_csv(
        ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/heldout_pixels_and_colocated_predictions.csv', header=0)

    for date in tqdm(summary_df.index, desc='Creating heldout pixel cross plot'):
        # Want to plot by date to see if there's anything weird going on.
        sub_df = pixels_df[pixels_df.Date == date]

        plt.scatter(sub_df.heldout_pixel_values,
                    sub_df.colocated_prediction_values,
                    alpha=0.3)

    x = np.linspace(np.min(pixels_df.heldout_pixel_values),
                    np.max(pixels_df.heldout_pixel_values))
    y = x * 1.
    plt.xlabel(r'Heldout pixel value [ppbv]')
    plt.ylabel(r'Predicted pixel value [ppbv]')
    plt.plot(x, y, color='red')

    # Set the title on the plot using the start and end date of the model run.
    start_date, end_date, model = fitted_results.run_name.split('-')
    plt.title(datetime.datetime.strptime(start_date, '%Y%m%d').strftime('%B %-d, %Y') + ' - ' +
              datetime.datetime.strptime(end_date, '%Y%m%d').strftime('%B %-d, %Y'))

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/paper/heldout_pixel_vs_predicted_pixel_crossplot.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.01)

    plt.show()

def poor_pixel_predicted_pixel_cross_plot(fitted_results):
    '''This function is for plotting poor pixels of methane (QA < 0.5) against predicted values at the same location.

    :param fitted_results: The model run we want to plot the pixels for.
    :type fitted_results: FittedResults
    '''

    # Read in the summary.csv file, index by date
    summary_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/summary.csv', header=0,
                             index_col=0)

    # Read in the csv file comparing poor pixels to predictions.
    pixels_df = pd.read_csv(
        ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/poor_pixels_and_colocated_predictions.csv', header=0)

    # Read in the grand plotables csv to get the range of NOAA background CH4 values.
    plotables_df = pd.read_csv(
        ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/plotable_quantities.csv', header=0)

    max_background = np.max(plotables_df.noaa_background)

    for date in tqdm(summary_df.index, desc='Creating poor pixel cross plot'):

        # Want to plot by date to see if there's anything weird going on.
        sub_df = pixels_df[pixels_df.Date == date]

        plt.scatter(sub_df.poor_pixel_values,
                    sub_df.colocated_prediction_value,
                    alpha=0.3)

    x_low  = np.min(pixels_df.poor_pixel_values)
    x_high = np.max(pixels_df.poor_pixel_values)
    x = np.linspace(x_low, x_high)
    y = x * 1.
    plt.xlabel(r'Poor pixel value [ppbv]')
    plt.ylabel(r'Predicted pixel value [ppbv]')
    plt.plot(x, y, color='red')

    # Plot a shaded grey region to show spread of NOAA background values
    plt.fill_between(np.arange(x_low - 15.0, max_background, 0.01), x_low - 15.0, max_background, alpha=0.3, color='grey')

    # Set the title on the plot using the start and end date of the model run.
    start_date, end_date, model = fitted_results.run_name.split('-')
    plt.title(datetime.datetime.strptime(start_date, '%Y%m%d').strftime('%B %-d, %Y') + ' - ' +
              datetime.datetime.strptime(end_date, '%Y%m%d').strftime('%B %-d, %Y'))

    plt.xlim([x_low-15.0, x_high+15.0])
    plt.ylim([x_low - 15.0, x_high + 15.0])

    plt.text(1690, 1675, 'Region below NOAA background')

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/autosaved/poor_pixel_vs_predicted_pixel_crossplot.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.01)

    plt.show()

def no2_ch4_flarestack_crossplot(fitted_results, molecule, show_augmented_load=False):
    '''This function is for plotting a cross plot of either CH4 loading or NO2 loading vs flare stack count in the study
    region.

    :param fitted_results: The model run that we want to plot results for.
    :type fitted_results: FittedResults
    :param molecule: The type of molecule that we want to plot loading for, must be either 'CH4' or 'NO2'
    :type molecule: string
    :param show_augmented_load: A flag to determine if you want to plot original or augmented methane load in the study
        region.
    :type show_augmented_load: bool
    '''

    # Open the plotables csv file.
    plotables_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/plotable_quantities.csv',
                               header=0,
                               index_col=0)

    if molecule == 'CH4':

        if show_augmented_load:
            mass   = plotables_df.augmented_ch4_load * 16.04 / 1e6  # Convert to grams, convert to tonnes
            errors = plotables_df.augmented_ch4_load_precision * 16.04 / 1e6 # Convert to grams, convert to tonnes

        else:
            mass = plotables_df.original_ch4_load * 16.04 / 1e6   # Convert to grams, convert to tonnes
            errors = plotables_df.original_ch4_load_precision * 16.04 / 1e6   # Convert to grams, convert to tonnes

        label    = r'CH$_{4}$'
        filename = 'ch4'

    elif molecule == 'NO2':
        mass   = plotables_df.original_no2_load * 46.0055 / 1e6 # Convert to grams, convert to tonnes
        errors = plotables_df.original_no2_load_precision * 46.0055 / 1e6 # Convert to grams, convert to tonnes

        label = r'NO$_{2}$'
        filename = 'no2'

    # Plot the cross plot.
    plt.errorbar(plotables_df.flare_count,
                 np.array(mass),
                 yerr=np.array(errors),
                 linestyle="None",
                 ecolor="black",
                 fmt='D',
                 mfc='w',
                 color='black',
                 capsize=3,
                 ms=4,
                 zorder=1,
                 elinewidth=0.7)

    plt.ylabel(label + ' [tonnes]')
    plt.xlabel('Flare count')
    plt.savefig(ct.FILE_PREFIX + '/figures/autosaved/' + filename + '_flarestack_crossplot.pdf',
                bbox_inches = 'tight',
                pad_inches = 0.01)

    plt.show()

# ---------------------------------------------------------------------------------
# The functions below are used to create the functions that went into the paper.
# ---------------------------------------------------------------------------------

def figure_1(date_range, date):
    '''This function is for creating and saving Figure 1 of the paper. Figure 1 will be a page-wide, two-panel figure
    of TROPOMI and VIIRS observations that establish the physical context of the paper.

    :param date_range: The date range of the analysis that this plot is being created for. Must be of the format
      "%Y%m%d-%Y%m%d".
    :type date_range: str
    :param date: The date of the observations to plot. Format must be "%Y-%m-%d"
    :type date: str
    '''

    # Get the relevant .nc4 files of the TROPOMI observations using the date.
    file_date_prefix = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    potential_tropomi_files = [file.split('/')[-1] for file in
                               glob.glob(
                                   ct.FILE_PREFIX + '/observations/NO2/' + file_date_prefix + '*.nc')]
    if len(potential_tropomi_files) > 1:
        print('Multiple files match the input date. Enter index of desired file:')
        for i in range(len(potential_tropomi_files)):
            print(f'[{i}]: {potential_tropomi_files[i]}')
        index = int(input('File index: '))
        file = potential_tropomi_files[index]
    else:
        file = potential_tropomi_files[0]

    # Get necessary details with the plot helper.
    ch4_plot_helper = PlotHelper(file, 'CH4', qa_only=True)
    no2_plot_helper = PlotHelper(file, 'NO2', qa_only=True)

    # Get the outlines of counties, these will be used in both plots.
    reader   = shpreader.Reader(ct.FILE_PREFIX + '/misc/countyl010g_shp_nt00964/countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    # Find and open the relevant VIIRS observation file, and get the location of active flares in the study region.
    viirs_file        = glob.glob(ct.FILE_PREFIX + '/observations/VIIRS/*' + file_date_prefix + '*.csv')[0]
    viirs_df          = pd.read_csv(viirs_file)
    flare_location_df = pd.DataFrame(columns=('Latitude', 'Longitude'))
    for index in viirs_df.index:
        if (ct.STUDY_REGION['Permian_Basin'][2] < viirs_df.Lat_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][3]):
            if (ct.STUDY_REGION['Permian_Basin'][0] < viirs_df.Lon_GMTCO[index] < ct.STUDY_REGION['Permian_Basin'][1]):
                if viirs_df.Temp_BB[index] != 999999:
                    flare_location_df = flare_location_df.append({'Latitude': viirs_df.Lat_GMTCO[index],
                                                                  'Longitude': viirs_df.Lon_GMTCO[index]},
                                                                 ignore_index=True)

    # Define the colors that we will use for the plots.
    colors = copy.copy(cm.RdYlBu_r)
    colors.set_bad('grey', 1.0)

    # Create figure, use gridspec to divide it up into subplots. 2-column subplot, each molecule gets a column.
    plt.figure(figsize=(10, 4))
    G = gridspec.GridSpec(1, 2, wspace=0.03)

    # ax_1 is the subplot for the CH4 observation, left hand side (first column), set projection here.
    ax_1 = plt.subplot(G[0, 0],
                       projection=ccrs.PlateCarree(),
                       extent=ch4_plot_helper.extent)

    # Set the ticks for the CH4 subplot.
    ax_1.set_xlabel('Longitude', labelpad=0)
    ax_1.set_xticks(ch4_plot_helper.xticks)
    ax_1.set_yticks(ch4_plot_helper.yticks)
    ax_1.yaxis.tick_right()

    # Add the latitude label, this will be for both subplots.
    ax_1.text(1.05, 0.85,
              'Latitude',
              horizontalalignment='center',
              verticalalignment='center',
              rotation=90,
              transform=ax_1.transAxes)

    # Plot the CH4 data.
    ch4_im = ax_1.pcolormesh(ch4_plot_helper.longitudes,
                             ch4_plot_helper.latitudes,
                             ch4_plot_helper.data,
                             cmap=colors,
                             vmin=ch4_plot_helper.vmin,
                             vmax=ch4_plot_helper.vmax,
                             zorder=0)

    # Add a colorbar to the methane plot, show it inside the plot towards the bottom.
    ch4_cbar_ax = ax_1.inset_axes([0.25, 0.02, 0.73, 0.05], #x0, y0, width, height
                                  transform=ax_1.transAxes)
    ch4_cbar = plt.colorbar(ch4_im,
                            cax=ch4_cbar_ax,
                            orientation='horizontal')
    ch4_cbar.set_ticks([])
    ch4_cbar_ax.text(1830, 1830, '1830', ha='center')
    ch4_cbar_ax.text(1860, 1830, '1860', ha='center')
    ch4_cbar_ax.text(1890, 1830, '1890', ha='center')

    # Add the borders to the CH4 subplot.
    ax_1.add_feature(COUNTIES, facecolor='none', edgecolor='lightgray', zorder=1)

    ax_1.set_title(r'CH$_4$ column-average mixing ratio [ppbv]', fontsize=10)

    # Plot the text of panel B
    ax_1.text(0.04, 0.94,
              'A',
              color='red',
              transform=ax_1.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center', )

    # ax_2 is the subplot for the NO2 observation, right hand side (second column), set projection here.
    ax_2 = plt.subplot(G[0, 1],
                       projection=ccrs.PlateCarree(),
                       sharey=ax_1,
                       extent=ch4_plot_helper.extent)

    # Set the ticks for the NO2 subplot, have ticks on left, but don't show the numbers.
    ax_2.set_xlabel('Longitude', labelpad=0)
    ax_2.set_xticks(no2_plot_helper.xticks)
    ax_2.set_yticks(no2_plot_helper.yticks)
    ax_2.tick_params(axis='y',
                     labelleft=False)

    # Plot the NO2 data.
    no2_im = ax_2.pcolormesh(no2_plot_helper.longitudes,
                             no2_plot_helper.latitudes,
                             no2_plot_helper.data,
                             cmap=colors,
                             vmin=no2_plot_helper.vmin,
                             vmax=no2_plot_helper.vmax,
                             zorder=0)
    
    # Plot the location of the VIIRS flares on top of the NO2 data.
    ax_2.scatter(flare_location_df.Longitude,
                 flare_location_df.Latitude,
                 marker="^",
                 s=20,
                 color='limegreen',
                 edgecolors='black',
                 zorder=3)

    # Add a colorbar to the NO2 plot, show it inside the plot.
    no2_cbar_ax = ax_2.inset_axes([0.25, 0.02, 0.73, 0.05],  # x0, y0, width, height
                                  transform=ax_2.transAxes)
    no2_cbar = plt.colorbar(no2_im,
                            cax=no2_cbar_ax,
                            orientation='horizontal')
    no2_cbar.set_ticks([])
    no2_cbar_ax.text(0.01, 0.02, '0.01', ha='center')
    no2_cbar_ax.text(0.05, 0.02, '0.05', ha='center')
    no2_cbar_ax.text(0.09, 0.02, '0.09', ha='center')

    # Add the borders to the NO2 subplot.
    ax_2.add_feature(COUNTIES, facecolor='none', edgecolor='lightgray', zorder=1)

    # Add the study region as a red box on the CH4 subplot.
    box = ct.STUDY_REGION['Permian_Basin']
    rectangle = patches.Rectangle((box[0], box[2]),
                                  box[1] - box[0],
                                  box[3] - box[2],
                                  linewidth=2,
                                  edgecolor='r',
                                  fill=False,
                                  zorder=2)
    ax_2.add_patch(rectangle)

    # Plot the title for the NO2 plot.
    ax_2.set_title(r'NO$_2$ column density [mmol m$^{-2}$]', fontsize=10)

    # Plot the text of panel B
    ax_2.text(0.04, 0.94,
              'B',
              color='red',
              transform=ax_2.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center', )

    # Save the figure as a png, too large otherwise, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/' + date_range + '/figure_1.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.01)

def figure_2(date_range):
    '''This figure is for creating a two-panel figure, left panel is a histogram of reduced chi-squared values,
    the right panel is a scatterplot of predictions vs observations. Will include both data-poor and data-rich days.

    :param date_range: The date range we want to plot this for.'''

    # Read the csv files we need.
    data_rich_chi_squared_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich/dropout/reduced_chi_squared.csv')
    data_poor_chi_squared_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor/dropout/reduced_chi_squared.csv')
    data_rich_residuals_df   = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich/dropout/residuals.csv')
    data_poor_residuals_df   = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich/dropout/residuals.csv')

    # Set up the figure. Page-wide, two-panel figure.
    plt.figure(figsize=(7.2, 3.45))
    G = gridspec.GridSpec(1, 2, wspace=0.05)

    # -------------------------------------------------------------------------------------------
    ax_1 = plt.subplot(G[0, 0])

    combined_residuals_df = pd.concat([data_poor_residuals_df, data_rich_residuals_df])

    ax_1.scatter(combined_residuals_df.actual_value,
                 combined_residuals_df.predicted_value,
                 alpha=0.2,
                 s=8,
                 zorder=1)

    x_values = np.arange(np.min(combined_residuals_df.actual_value), np.max(combined_residuals_df.actual_value))
    y_values = x_values

    ax_1.plot(x_values,
              y_values,
              color='red',
              zorder=2)

    ax_1.set_yticks([1840, 1880, 1920, 1960])
    ax_1.set_yticklabels(['1840', '1880', '1920', '1960'],
                         rotation=90,
                         va='center')
    ax_1.set_xlabel(r'$\mathrm{CH}_4^{\mathrm{obs}}$ [ppbv]')
    ax_1.set_ylabel(r'$\mathrm{CH}_4^{\mathrm{pred}}$ [ppbv]')
    ax_1.grid(color='grey',
              linestyle='dashed',
              zorder=0)

    # Plot the text of panel A
    ax_1.text(0.05, 0.94,
              'A',
              color='red',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_1.transAxes,
              fontsize=20)

    # -------------------------------------------------------------------------------------------

    ax_2 = plt.subplot(G[0, 1])

    combined_chi_squared_df = pd.concat([data_rich_chi_squared_df, data_poor_chi_squared_df])

    sns.distplot(combined_chi_squared_df.reduced_chi_squared,
                kde=True,
                 ax=ax_2)


    ax_2.set_yticks([0.5, 1, 1.5, 2])
    ax_2.set_yticklabels(['0.5', '1.0', '1.5', '2.0'],
                         rotation=270,
                         va='center')
    ax_2.yaxis.tick_right()
    ax_2.set_ylabel('Density',
                    rotation=270,
                    labelpad=15)
    ax_2.yaxis.set_label_position("right")
    ax_2.set_xlabel(r'$\chi^{2}_{\nu}$')

    # Plot the text of panel A
    ax_2.text(0.05, 0.94,
              'B',
              color='red',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_2.transAxes,
              fontsize=20)

    # -------------------------------------------------------------------------------------------
    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/' + date_range + '/figure_2.pdf',
                bbox_inches='tight',
                pad_inches=0.01)

def figure_3(fitted_results, date):
    '''This function is for creating and saving Figure 2 of the paper. Figure 2 will be a page-wide, two-panel figure.
    Left hand panel is a scatterplot of observations with errorbars and a subset of regression lines. Right hand panel
    is an alpha-beta cross plot with errorbars in both dimensions, with a 95% CI ellipse plotted underneath using the
    mode estimated value of mu and Sigma.

    :param fitted_results: The results of this model run.
    :type fitted_results: FittedResults
    :param date: The date of the observations to plot. Format must be "%Y-%m-%d"
    :type date: string
    '''

    start_date, end_date, model = fitted_results.run_name.split('-')

    # Set up the figure. Page-wide, two-panel figure.
    plt.figure(figsize=(7.2, 4.45))
    G = gridspec.GridSpec(1, 2, wspace=0.05)

    # ax_1 is the subplot for the scatterplot of observations, left hand side (first column).
    ax_1 = plt.subplot(G[0, 0])

    # Seed the random number generator.
    np.random.seed(101)

    # Open the plotables csv file.
    plotables_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/final_results.csv',
                               header=0,
                               index_col=0)

    # Fill the lists of mode values and the errors needed for the scatterplot.
    beta_values       = list(plotables_df.beta_50)
    beta_error_bounds = [[median - lower_bound, higher_bound - median]
                         for median, lower_bound, higher_bound
                         in zip(beta_values, list(plotables_df.beta_16), list(plotables_df.beta_84))]

    alpha_values       = list(plotables_df.alpha_50)
    alpha_error_bounds = [[median - lower_bound, higher_bound - median]
                        for median, lower_bound, higher_bound
                        in zip(alpha_values, list(plotables_df.alpha_16), list(plotables_df.alpha_84))]

    # Read in the observations the date in question.
    dataset_df = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_results.run_name + '/dataset.csv', header=0)
    date_df    = dataset_df[dataset_df.date == date]

    # Needed to plot the regression lines.
    x_min, x_max = np.min(date_df.obs_NO2), np.max(date_df.obs_NO2)
    x_domain = np.linspace(x_min, x_max, 100)

    # Get mode and upper and lower 68% CI bounds for alpha and beta on this day.
    median_beta  = plotables_df.loc[date]['beta_50']
    median_alpha = plotables_df.loc[date]['alpha_50']
    lower_beta_bound, upper_beta_bound   = plotables_df.loc[date]['beta_16'], plotables_df.loc[date]['beta_84']
    beta_diff                            = (upper_beta_bound - lower_beta_bound) / 2.
    lower_alpha_bound, upper_alpha_bound = plotables_df.loc[date]['alpha_16'], plotables_df.loc[date]['alpha_84']
    alpha_diff                           = (upper_alpha_bound - lower_alpha_bound) / 2.
    # Get the Day ID for this day
    day_id = plotables_df.loc[date]['day_id']


    # Extract a random subset of 500 alpha-beta draws that the sampler drew.
    draws = np.arange(len(fitted_results.full_trace['alpha.' + str(day_id)]))
    np.random.shuffle(draws)
    alpha = fitted_results.full_trace['alpha.' + str(day_id)][draws]
    beta  = fitted_results.full_trace['beta.' + str(day_id)][draws]
    for i in range(500):
        ax_1.plot(x_domain,
                 alpha[i] + beta[i] * x_domain,
                 color='lime',
                 alpha=0.01,
                 zorder=2)

    # Plot the observations with TROPOMI errors.
    ax_1.errorbar(date_df.obs_NO2, date_df.obs_CH4, yerr=date_df.sigma_C, xerr=date_df.sigma_N,
                 ecolor="blue",
                 capsize=3,
                 fmt='D',
                 mfc='w',
                 color='red',
                 ms=3.5,
                 zorder=1,
                 alpha=0.5)

    # Plot values of alpha and beta in axes coordinates.
    ax_1.text(0.65, 0.05,
              r'$\beta={}\pm{}$'.format(round(median_beta),
                                         '{' + str(round(beta_diff)) + '}') + '\n' +
              r'$\alpha={}\pm{}$'.format(round(median_alpha),
                                         '{' + str(round(alpha_diff)) + '}'),
              transform=ax_1.transAxes,
              fontsize=10)

    # Add the letter A to plot 1.
    ax_1.text(0.05, 0.95,
              'A',
              color='red',
              transform=ax_1.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center',)

    # Customise ticks for the left hand panel.
    ax_1.set_yticks([1840, 1880, 1920, 1960])
    plt.setp(ax_1.yaxis.get_majorticklabels(),
             rotation=90,
             va='center')

    ax_1.set_xlabel(r'NO$_{2}^{\mathrm{obs}}$ [mmol m$^{-2}$]')
    ax_1.set_ylabel(r'CH$_{4}^{\mathrm{obs}}$ [ppbv]')
    ax_1.title.set_text(datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%b %-d, %Y'))

    ax_1.grid(which='both',
              linestyle='dashed',
              color='grey',
              alpha=0.5)

    # ax_2 is the subplot for the cross plot of alpha and beta, right hand side (second column).
    ax_2 = plt.subplot(G[0, 1])

    # Plot the alpha and beta parameters.
    ax_2.errorbar(alpha_values,
                  beta_values,
                  yerr=np.array(beta_error_bounds).T,
                  xerr=np.array(alpha_error_bounds).T,
                  ecolor="blue",
                  capsize=3.5,
                  fmt='D',
                  mfc='w',
                  color='red',
                  ms=3)

    # Add in the 95% CI ellipse using mode estimated values of the hyperparameters.
    pearson     = fitted_results.median_values['rho']
    sigma_alpha = fitted_results.median_values['sigma_alpha']
    mu_alpha    = fitted_results.median_values['mu_alpha']
    sigma_beta  = fitted_results.median_values['sigma_beta']
    mu_beta     = fitted_results.median_values['mu_beta']

    ellipse(pearson,
            sigma_alpha,
            sigma_beta,
            mu_alpha,
            mu_beta,
            ax_2,
            n_std=2.0,
            edgecolor='red',
            facecolor='red',
            alpha=0.3)

    ax_2.set_ylabel(r'$\beta$ [ppbv / (mmol m$^{-2}$)]',
                    rotation=270,
                    labelpad=20)
    ax_2.set_xlabel(r'$\alpha$ [ppbv]')

    # Set the title of the right hand subplot.
    start_date, end_date, model = fitted_results.run_name.split('-')
    ax_2.title.set_text(datetime.datetime.strptime(start_date, '%Y%m%d').strftime('%b %-d') + ' - ' +
                    datetime.datetime.strptime(end_date, '%Y%m%d').strftime('%b %-d, %Y'))

    # Customise ticks for the right hand panel.
    ax_2.set_yticks([400, 800, 1200, 1600])
    ax_2.yaxis.tick_right()
    ax_2.yaxis.set_label_position("right")
    plt.setp(ax_2.yaxis.get_majorticklabels(),
             rotation=270,
             va='center')

    ax_2.grid(which='both',
              linestyle='dashed',
              color='grey',
              alpha=0.5)

    ax_2.text(0.05, 0.95,
              'B',
              color='red',
              transform=ax_2.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center', )

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/' + start_date + '-' + end_date + '/figure_3.pdf',
                bbox_inches='tight',
                pad_inches=0.01)

def figure_4(fitted_results):
    '''This function is for plotting Figure 4 of the paper. Figure 3 is a page-wide figure of two time series together
    of :math:`\\beta` values and flare stack counts for the specified run. There is also an extra panel on the right
    showing a cross plot of :math:`\\beta` and flare stack counts.

    :param fitted_results: The fitted results.
    :type fitted_results: FittedResults
    '''

    # Open the plotables csv file.
    plotables_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + fitted_results.run_name + '/final_results.csv',
                               header=0,
                               index_col=0)

    # Fill the lists of mode values and the errors needed for the scatterplot.
    beta_values = list(plotables_df.beta_50)
    errors      = [[median - lower_bound, higher_bound - median]
                   for median, lower_bound, higher_bound
                   in zip(beta_values, list(plotables_df.beta_16), list(plotables_df.beta_84))]

    # Create the datetime objects for all the dates we have beta inferences for ("data rich" days).
    beta_datetimes = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in plotables_df.index]
    # errors = [[beta_df.Beta[i] - beta_df.Lower_bound_95_CI[i], beta_df.Upper_bound_95_CI[i] - beta_df.Beta[i]]
    #           for i in beta_df.index]

    # Create figure, use gridspec to divide it up into subplots. 3-column subplot where first plot spans two columns.
    plt.figure(figsize=(12.0, 4.45))
    G    = gridspec.GridSpec(1, 3)
    # Need just a little bit of space between the subplots to have y-axis tick marks in between.
    G.update(wspace=0.19)

    # Left hand side plot (the time series). Needs to span the first two columns.
    ax_1 = plt.subplot(G[0, 1:])
    ax_1.set_xlabel("Date", fontsize=12)

    # Plot the time series of beta.
    ax_1.errorbar(beta_datetimes,
                  beta_values,
                  yerr=np.array(errors).T,
                  linestyle="None",
                  ecolor="black",
                  fmt='D',
                  mfc='w',
                  color='black',
                  capsize=3,
                  ms=4,
                  elinewidth=0.7)

    # Plot the text of panel A
    ax_1.text(0.03, 0.95,
              'B',
              color='red',
              transform=ax_1.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center', )

    # Make a twin axes object that shares the x axis with the time series of beta.
    ax_twin = ax_1.twinx()

    # Plot the time series of flare count on this twin axes object.
    ax_twin.scatter(beta_datetimes,
                    plotables_df.flare_count,
                    color="red",
                    marker="x",
                    s=30,
                    alpha=0.7)

    # Make the tick marks on the y axis of the flare count time series red, but don't include the numbers.
    ax_twin.tick_params(axis='y', colors='red', labelright=True)
    ax_twin.set_ylabel('Flare count',
                       color='red',
                       rotation=270,
                       labelpad=15,
                       fontsize=12)

    # Move the time series of beta on top of the time series of flare count.
    ax_1.set_zorder(2)
    ax_twin.set_zorder(1)
    ax_1.patch.set_visible(False)

    # Figure out the locations of the first and last tick mark in the date range of this run.
    start_date, end_date, model = fitted_results.run_name.split('-')
    first_tick_date = datetime.datetime.strptime(start_date, "%Y%m%d").replace(day=1).strftime("%Y-%m-%d")
    last_tick_date = (
                datetime.datetime.strptime(end_date, "%Y%m%d").replace(day=1) + datetime.timedelta(days=32)).replace(
        day=1).strftime("%Y-%m-%d")

    # Use pandas to create the date range, ticks on first of the month ('MS'), every two months ('2MS').
    tick_locations = pd.date_range(first_tick_date, last_tick_date, freq='2MS')
    # Change the tick labels to be abbreviated month and year.
    tick_labels    = tick_locations.strftime('%b-%y')

    # Add date ticks to the x axis of the time series plot.
    plt.setp(ax_twin,
             xticks=tick_locations,
             xticklabels=tick_labels)

    # Add vertical dashed lines at the location of the date tick marks.
    for location in tick_locations:
        plt.axvline(location, linestyle='--', color='grey', linewidth=0.5)

    # Right hand side plot (the cross plot). Needs to span the third column and share y axis with the time series.
    ax_2 = plt.subplot(G[0, 0], sharey=ax_1)

    # Set the tick marks on y axis of the cross plot to be red, on the left hand side of the plot, include numbers.
    ax_2.tick_params(axis='y',
                     colors='black',
                     labelleft=False)
    ax_2.tick_params(axis='x',
                     colors='red')
    ax_2.yaxis.tick_right()

    # Plot the cross plot of beta and flare count, including 95% CI on beta.
    ax_2.errorbar(plotables_df.flare_count,
                  beta_values,
                  yerr=np.array(errors).T,
                  linestyle="None",
                  ecolor="black",
                  fmt='D',
                  mfc='w',
                  color='black',
                  capsize=3,
                  ms=4,
                  elinewidth=0.7)

    # Plot the text of panel B
    ax_2.text(0.05, 0.95,
              'A',
              color='red',
              transform=ax_2.transAxes,
              fontsize=20,
              horizontalalignment='center',
              verticalalignment='center', )

    # Set the x-axis label on the cross plot.
    ax_2.set_xlabel('Flare count',
                    fontsize=12,
                    labelpad=0,
                    color='red')

    ax_2.tick_params(axis='y', colors='black', labelright=False)

    # Plot some red text to show that the red y-axis is the flare count. Plot in axes coordinates of the cross plot.
    ax_2.text(1.07, 1.05,
              r'$\beta$ [ppbv / (mmol m$^{-2}$)]',
              color='black',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_2.transAxes,
              fontsize=12)

    ax_2.grid(which='both',
              linestyle='dashed',
              linewidth=0.5,
              color='grey')

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/' + start_date + '-' + end_date + '/figure_4.pdf',
                bbox_inches='tight',
                pad_inches=0.01)

def figure_5(directory, date):
    #TODO make docstring

    # Get the relevant .nc4 files of the TROPOMI  CH4 observation using the date.
    file_date_prefix = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    # Sometimes there is more than one overpass per day.
    potential_tropomi_files = [file.split('/')[-1] for file in
                               glob.glob(
                                   ct.FILE_PREFIX + '/observations/NO2/' + file_date_prefix + '*.nc')]
    if len(potential_tropomi_files) > 1:
        print('Multiple files match the input date. Enter index of desired file:')
        for i in range(len(potential_tropomi_files)):
            print(f'[{i}]: {potential_tropomi_files[i]}')
        index = int(input('File index: '))
        file = potential_tropomi_files[index]
    else:
        file = potential_tropomi_files[0]

    # Get necessary details with the plot helper.
    good_pixels = PlotHelper(file, 'CH4', qa_only=True)
    good_pixels_and_good_predictions = PlotHelper(file,
                                                  'CH4',
                                                  qa_only=True,
                                                  include_predictions=True,
                                                  augmented_directory=directory)

    # Get the outlines of counties, these will be used in both plots.
    reader   = shpreader.Reader(ct.FILE_PREFIX + '/misc/countyl010g_shp_nt00964/countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    # Define the colors that we will use for the plots.
    colors = copy.copy(cm.RdYlBu_r)
    colors.set_bad('grey', 1.0)

    # Create figure, use gridspec to divide it up into subplots. 2-column subplot, each molecule gets a column.
    plt.figure(figsize=(10, 4.5))
    G = gridspec.GridSpec(1, 2, wspace=0.09, hspace=0.00)

    #------------------------------------------------------------------------------------------------------------
    # ax_1 is the subplot for the 'good' pixels TROPOMI observation, first row, first column, set projection here.
    ax_1 = plt.subplot(G[0, 0],
                       projection=ccrs.PlateCarree(),
                       extent=good_pixels.extent)

    # Plot the CH4 data.
    good_im = ax_1.pcolormesh(good_pixels.longitudes,
                              good_pixels.latitudes,
                              good_pixels.data,
                              cmap=colors,
                              vmin=good_pixels.vmin,
                              vmax=good_pixels.vmax,
                              zorder=0)

    # Set the ticks for the CH4 subplot.
    ax_1.set_xticks([-104, -102, -100])
    ax_1.set_yticks(good_pixels.yticks)
    ax_1.yaxis.tick_right()

    # Plot the text of panel A
    ax_1.text(0.04, 0.94,
              'A',
              color='red',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_1.transAxes,
              fontsize=20)

    # Add a colorbar to the methane plot, show it inside the plot towards the bottom.
    good_cbar_ax = ax_1.inset_axes([0.25, 0.02, 0.73, 0.05],  # x0, y0, width, height
                                   transform=ax_1.transAxes)
    good_cbar = plt.colorbar(good_im,
                            cax=good_cbar_ax,
                            orientation='horizontal')
    good_cbar.set_ticks([])
    good_cbar_ax.text(1830, 1830, '1830', ha='center')
    good_cbar_ax.text(1860, 1830, '1860', ha='center')
    good_cbar_ax.text(1890, 1830, '1890', ha='center')

    # Add the borders to the CH4 subplot.
    ax_1.add_feature(COUNTIES, facecolor='none', edgecolor='lightgray', zorder=1)

    # Add the latitude label, this will be for both subplots 1 and 2.
    ax_1.text(1.05, 0.85,
              'Latitude',
              horizontalalignment='center',
              verticalalignment='center',
              rotation=90,
              transform=ax_1.transAxes)

    # Add the longitude label, this will be for both subplots 1 and 3.
    ax_1.text(0.1, -0.05,
              'Longitude',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_1.transAxes)

    # Set title
    # ax_1.set_title(r'TROPOMI pixels with $\mathrm{QA}\geq 0.5$', pad=0)

    #------------------------------------------------------------------------------------------------------------
    # ax_2 is the subplot for all pixels TROPOMI observation, first row, second column, set projection here.
    ax_2 = plt.subplot(G[0, 1],
                       projection=ccrs.PlateCarree(),
                       extent=good_pixels_and_good_predictions.extent)

    # Plot the CH4 data.
    augmented_im = ax_2.pcolormesh(good_pixels_and_good_predictions.longitudes,
                                   good_pixels_and_good_predictions.latitudes,
                                   good_pixels_and_good_predictions.data,
                                   cmap=colors,
                                   vmin=good_pixels_and_good_predictions.vmin,
                                   vmax=good_pixels_and_good_predictions.vmax,
                                   zorder=0)

    # Set the ticks for the CH4 subplot.
    ax_2.set_xticks([-106, -104, -102])
    ax_2.set_yticks(good_pixels_and_good_predictions.yticks)
    ax_2.tick_params(axis='y',
                     labelleft=False)

    # Add a colorbar to the methane plot, show it inside the plot towards the bottom.
    all_cbar_ax = ax_2.inset_axes([0.25, 0.02, 0.73, 0.05],  # x0, y0, width, height
                                   transform=ax_2.transAxes)
    all_cbar = plt.colorbar(augmented_im,
                            cax=all_cbar_ax,
                            orientation='horizontal')
    all_cbar.set_ticks([])
    all_cbar_ax.text(1830, 1830, '1830', ha='center')
    all_cbar_ax.text(1860, 1830, '1860', ha='center')
    all_cbar_ax.text(1890, 1830, '1890', ha='center')

    # Add the borders to the CH4 subplot.
    ax_2.add_feature(COUNTIES, facecolor='none', edgecolor='lightgray', zorder=1)

    # Set title
    # ax_2.set_title('All TROPOMI pixels', pad=0)

    # Add the longitude label, this will be for both subplots 1 and 3.
    ax_2.text(0.9, -0.05,
              'Longitude',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_2.transAxes)

    # Add the text of panel B
    ax_2.text(0.04, 0.94,
              'B',
              color='red',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_2.transAxes,
              fontsize=20)

    start_date, end_date, model = directory.split('-')

    # Save the figure as a png, too large otherwise, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/' + start_date + '-' + end_date + '/figure_5.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.01)

def figure_6(date_range):
    '''This function is for creating and saving Figure 6 of the paper. Figure 6 will be a page-wide, 3-panel figure.
    Each panel is a page-wide row. Each panel shows a time series of a quantity calculated three times, once with just
    TROPOMI CH4 observations that pass the QA threshold, again with all TROPOMI CH4 observations, and then again including
    the predicted CH4 values from the model.

    :param fitted_results: The results of this model run.
    :type fitted_results: FittedResults
    '''

    # Open the plotables csv files.
    data_rich_plotables_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_rich/final_results.csv',
                               header=0,
                               index_col=0)

    data_poor_plotables_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + date_range + '-data_poor/final_results.csv',
                                         header=0,
                                         index_col=0)

    all_plotables_df = pd.concat([data_rich_plotables_df, data_poor_plotables_df])
    all_plotables_df = all_plotables_df.sort_values(by='date')
    # Create the datetime objects for all the dates we have calculated quantities for.
    all_datetimes = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in all_plotables_df.index]

    #TODO eventually get rid of this line.
    data_rich_datetimes = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in data_rich_plotables_df.index]
    data_poor_datetimes = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in data_poor_plotables_df.index]


    # Create figure, use gridspec to divide it up into subplots. 3-row subplot.
    plt.figure(figsize=(10.0, 12.0))
    G = gridspec.GridSpec(3, 3)
    # Need just a little bit of space between the subplots to have y-axis tick marks in between.
    G.update(hspace=0.05)

    # ------------------------------------------------------------------------------------------------------
    # Top row plot (the time series). Needs to span the first two columns.
    ax_1 = plt.subplot(G[0, :])

    # Plot the three quantities, all same color, but different markers and line styles.
    ax_1.plot(all_datetimes,
              all_plotables_df.original_pixel_coverage * 100,
              color='darkorange',
              linestyle='solid',
              zorder=0)
    ax_1.scatter(data_poor_datetimes,
                 data_poor_plotables_df.original_pixel_coverage * 100,
                 color='darkorange',
                 facecolor='white',
                 s=20,
                 zorder=1)
    ax_1.scatter(data_rich_datetimes,
                 data_rich_plotables_df.original_pixel_coverage * 100,
                 color='darkorange',
                 zorder=2)

    ax_1.plot(all_datetimes,
              all_plotables_df.augmented_pixel_coverage * 100,
              color='green',
              linestyle='dotted',
              zorder=0)
    ax_1.scatter(data_poor_datetimes,
                 data_poor_plotables_df.augmented_pixel_coverage * 100,
                 color='green',
                 facecolor='white',
                 s=20,
                 zorder=1)
    ax_1.scatter(data_rich_datetimes,
                 data_rich_plotables_df.augmented_pixel_coverage * 100,
                 color='green',
                 zorder=2)

    # Plot the text of panel A
    ax_1.text(0.02, 0.94,
              'A',
              color='red',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_1.transAxes,
              fontsize=20)

    # Set the ylabel
    ax_1.set_ylabel('% Pixel coverage of study region')

    ax_1.tick_params(axis='x',          # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,     # labels along the top edge are off
                     labelbottom=False) # labels along the bottom edge are off

    ax_1.grid(which='both',
              linestyle='dashed',
              color='grey',
              alpha=0.5)

    # ------------------------------------------------------------------------------------------------------
    ax_6 = ax_1.inset_axes([0.8, 0.0, 0.2, 1.0], sharey=ax_1)
    sns.kdeplot(all_plotables_df.augmented_pixel_coverage * 100,
                vertical=True,
                ax=ax_6,
                color='green',
                fill=True)
    ax_6.invert_xaxis()
    ax_6.patch.set_alpha(0.0)
    ax_6.set(xticklabels=[])
    ax_6.set(xticks=[])
    ax_6.set_xlabel('')
    ax_6.yaxis.tick_right()
    ax_6.spines['left'].set_visible(False)
    ax_6.set_ylabel('')

    # ------------------------------------------------------------------------------------------------------
    ax_7 = ax_1.inset_axes([0.8, 0.0, 0.2, 1.0], sharey=ax_1)
    sns.kdeplot(all_plotables_df.original_pixel_coverage * 100,
                vertical=True,
                ax=ax_7,
                color='darkorange',
                fill=True)
    ax_7.invert_xaxis()
    ax_7.patch.set_alpha(0.0)
    ax_7.set(xticklabels=[])
    ax_7.set(xticks=[])
    ax_7.set_xlabel('')
    ax_7.yaxis.tick_right()
    ax_7.spines['left'].set_visible(False)
    ax_7.set_ylabel('')

    # ------------------------------------------------------------------------------------------------------
    ax_2 = plt.subplot(G[1, :], sharex=ax_1)

    # Plot the three quantities, all same color, but different markers and line styles.
    ax_2.plot(all_datetimes,
              all_plotables_df.original_pixel_value_50,
              color='darkorange',
              linestyle='solid',
              zorder=0)
    ax_2.scatter(data_poor_datetimes,
                 data_poor_plotables_df.original_pixel_value_50,
                 color='darkorange',
                 marker='o',
                 facecolor='white',
                 label='Original, data-poor',
                 s=20,
                 zorder=1)
    ax_2.scatter(data_rich_datetimes,
                 data_rich_plotables_df.original_pixel_value_50,
                 color='darkorange',
                 marker='o',
                 label='Original, data-rich',
                 zorder=2)


    ax_2.plot(all_datetimes,
              all_plotables_df.augmented_pixel_value_50,
              color='green',
              linestyle='dotted',
              zorder=0)
    ax_2.scatter(data_poor_datetimes,
                 data_poor_plotables_df.augmented_pixel_value_50,
                 color='green',
                 facecolor='white',
                 label='With predictions, data-poor',
                 s=20,
                 zorder=1)
    ax_2.scatter(data_rich_datetimes,
                 data_rich_plotables_df.augmented_pixel_value_50,
                 color='green',
                 label='With predictions, data-rich',
                 zorder=2)

    # Set the ylabel
    ax_2.set_ylabel(r'Median CH$_4$ concentration [ppbv]')

    ax_2.tick_params(axis='x',          # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # labels along the top edge are off
                     labelbottom=False) # labels along the bottom edge are off

    ax_2.legend(loc='lower right')

    ax_2.grid(which='both',
              linestyle='dashed',
              color='grey',
              alpha=0.5)

    # Plot the text of panel B
    ax_2.text(0.02, 0.94,
              'B',
              color='red',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_2.transAxes,
              fontsize=20)

    # ------------------------------------------------------------------------------------------------------
    ax_3 = plt.subplot(G[2, :], sharex=ax_1)

    # We have methane load saved in the .csv file in mols of CH4, we will plot in kilotonnes
    ax_3.plot(all_datetimes,
              all_plotables_df.original_ch4_load * 16.04 / 1e6 / 1e3, # Convert to grams, convert to tonnes, convert to kilotonnes
              color='darkorange',
              linestyle="solid",
              zorder=0)
    ax_3.scatter(data_poor_datetimes,
                 data_poor_plotables_df.original_ch4_load * 16.04 / 1e6 / 1e3,
                 color='darkorange',
                 facecolor='white',
                 s=20,
                 zorder=1)
    ax_3.scatter(data_rich_datetimes,
                 data_rich_plotables_df.original_ch4_load * 16.04 / 1e6 / 1e3,
                 color='darkorange',
                 zorder=2)

    ax_3.plot(all_datetimes,
              all_plotables_df.partially_augmented_ch4_load * 16.04 / 1e6 / 1e3,
              color='green',
              linestyle='dotted',
              zorder=0)
    ax_3.scatter(data_poor_datetimes,
                 data_poor_plotables_df.partially_augmented_ch4_load * 16.04 / 1e6 / 1e3,
                 color='green',
                 facecolor='white',
                 s=20,
                 zorder=1)
    ax_3.scatter(data_rich_datetimes,
                 data_rich_plotables_df.partially_augmented_ch4_load * 16.04 / 1e6 / 1e3,
                 color='green',
                 zorder=2)

    # ------------------------------------------------------------------------------------------------------
    ax_4 = ax_3.inset_axes([0.8, 0.0, 0.2, 1.0], sharey=ax_3)
    sns.kdeplot(all_plotables_df.partially_augmented_ch4_load * 16.04 / 1e6 / 1e3,
                vertical=True,
                ax=ax_4,
                color='green',
                fill=True)
    ax_4.invert_xaxis()
    ax_4.patch.set_alpha(0.0)
    ax_4.set(xticklabels=[])
    ax_4.set(xticks=[])
    ax_4.set_xlabel('')
    ax_4.yaxis.tick_right()
    ax_4.spines['left'].set_visible(False)
    ax_4.set_ylabel('')

    # ------------------------------------------------------------------------------------------------------
    ax_5 = ax_3.inset_axes([0.8, 0.0, 0.2, 1.0], sharey=ax_3)
    sns.kdeplot(all_plotables_df.original_ch4_load * 16.04 / 1e6 / 1e3,
                vertical=True,
                ax=ax_5,
                color='darkorange',
                fill=True)
    ax_5.invert_xaxis()
    ax_5.patch.set_alpha(0.0)
    ax_5.set(xticklabels=[])
    ax_5.set(xticks=[])
    ax_5.set_xlabel('')
    ax_5.yaxis.tick_right()
    ax_5.spines['left'].set_visible(False)
    ax_5.set_ylabel('')

    # ------------------------------------------------------------------------------------------------------

    # Set the ylabel
    ax_3.set_ylabel('Observed methane load above' + '\n' + 'NOAA background [kilotonnes]')

    # Set the y limits.
    ax_3.set_ylim([-10, 20])

    # Set the x axis ticks for the plot
    first_tick_date = datetime.datetime.strptime('20190101', "%Y%m%d").strftime("%Y-%m-%d")
    last_tick_date  = datetime.datetime.strptime('20200101', "%Y%m%d").strftime("%Y-%m-%d")

    # Use pandas to create the date range, ticks on first of the month ('MS'), every one months ('1MS').
    tick_locations = pd.date_range(first_tick_date, last_tick_date, freq='1MS')
    # Change the tick labels to be abbreviated month and year.
    tick_labels = tick_locations.strftime('%b-%y')

    # Add date ticks to the x axis of the time series plots.
    plt.setp(ax_3,
             xticks=tick_locations,
             xticklabels=tick_labels)

    ax_3.grid(which='both',
              linestyle='dashed',
              color='grey',
              alpha=0.5)

    # Plot the text of panel C
    ax_3.text(0.02, 0.94,
              'C',
              color='red',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax_3.transAxes,
              fontsize=20)

    # Save the figure as a pdf, no need to set dpi, trim the whitespace.
    plt.savefig(ct.FILE_PREFIX + '/figures/' + date_range + '/figure_6.pdf',
                bbox_inches='tight',
                pad_inches=0.01)

# ---------------------------------------------------------------------------------
# Testing these functions.
# ---------------------------------------------------------------------------------

fitted_results = results.FittedResults('20190101-20191231-data_rich')

trace(fitted_results,
      'mu_alpha',
      show_warmup_draws=True)