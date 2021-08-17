import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patches as patches
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
from src import constants as ct

# Some global parameters for all plots.
#mpl.rcParams['font.family']    = 'Avenir'

class PlotHelper:
    #TODO make the docstring
    def __init__(self, filename, molecule, qa_only=False):
        self.figsize = (10.0, 6.0)
        self.extent  = (-106, -99, 29, 35)
        self.xticks  = range(-106, -99, 2)
        self.yticks  = range(29, 35, 2)

        # Open the TROPOMI observation file
        f = nc4.Dataset(ct.FILE_PREFIX + '/observations/' + molecule + '/' + filename, 'r')

        date = filename.split("T")[0]

        if molecule == "NO2":
            self.legend = "Column density [$\mu$mol m$^{-2}$]"
            self.title = 'Tropospheric column density NO$_2$' + \
                         '\n' + datetime.datetime.strptime(date, '%Y%m%d').strftime('%B %-d, %Y')
            self.vmax  = 100.0
            self.vmin  = 0.0

            # Access pixel values, convert to micromols / m^2
            no2 = np.array(f.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'][0]) * 1e6


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

        elif molecule == "CH4":
            self.legend = "Mixing ratio [ppbv]"
            self.title = 'Column averaged mixing ratio CH$_4$' + \
                         '\n' + datetime.datetime.strptime(date, '%Y%m%d').strftime('%B %-d, %Y')
            self.vmax  = 1900.0
            self.vmin  = 1820.0

            # Access pixel values
            ch4 = np.array(f.groups['PRODUCT'].variables['methane_mixing_ratio'][0])

            if not qa_only:
                self.data = np.ma.masked_greater(ch4, 1e30)
            # If you only want high-quality pixels, have to do the below.
            else:
                filtered_ch4 = np.empty(np.shape(ch4))
                qa_values    = np.array(f.groups['PRODUCT'].variables['qa_value'][0])

                for i in range(np.shape(ch4)[0]):
                    for j in range(np.shape(ch4)[1]):
                        if qa_values[i, j] < 0.5:
                            filtered_ch4[i, j] = 1e32
                        else:
                            filtered_ch4[i, j] = ch4[i, j]

                self.data = np.ma.masked_greater(filtered_ch4, 1e30)

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

def tropomi_plot(date, molecule, plot_study_region=False, qa_only=False, show_flares=False):
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
    '''

    file_date_prefix        = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    potential_tropomi_files = [file.split('/')[-1] for file in
                               glob.glob(ct.FILE_PREFIX + '/observations/' + molecule + '/' + file_date_prefix + '*.nc')]

    if len(potential_tropomi_files) > 1:
        print('Multiple files match the input date. Enter index of desired file:')
        for i in range(len(potential_tropomi_files)):
            print(f'[{i}]: {potential_tropomi_files[i]}')
        index = int(input('File index: '))
        file = potential_tropomi_files[index]
    else:
        file = potential_tropomi_files[0]

    # Set necessary details with the plot helper
    plot_helper = PlotHelper(file, molecule, qa_only)

    # Get the oulines of counties
    reader = shpreader.Reader(ct.FILE_PREFIX + '/misc/countyl010g_shp_nt00964/countyl010g.shp')
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
    plt.savefig('figures/autosaved/tropomi_observation.png', dpi=300, bbox_inches='tight')
    plt.show()

def trace(fitted_model, parameter, date=None, compare_to_ground_truth=False, show_warmup_draws=False):
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

    # Need to get the ground truth value for comparison if we're plotting a test dataset.
    if compare_to_ground_truth:
        daily_parameters_df = pd.read_csv(ct.FILE_PREFIX + '/test_suite/ground_truths/' + fitted_model.run_name +
                                          '/alphas_betas_gammas.csv', delimiter=",", header=0, index_col=0) # Index by date
        hyperparameter_truths_df = pd.read_csv(ct.FILE_PREFIX + '/test_suite/ground_truths/' + fitted_model.run_name +
                                               '/hyperparameter_truths.csv', delimiter=',', header=0)

        if parameter == 'alpha' or parameter == 'beta' or parameter == 'gamma':
            ground_truth = daily_parameters_df.loc[date, parameter]

        else:
            ground_truth = hyperparameter_truths_df[parameter][0]

    # Humans think in terms of dates, stan thinks in terms of day ids. Need to be able to access parameter.day_id
    # using the passed date. Use the summary.csv file and index by date
    summary_df   = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_model.run_name + '/summary.csv', header=0, index_col=0)

    # Daily parameters are saved in stan as parameter.day_id .
    if parameter == 'alpha' or parameter == 'beta' or parameter == 'gamma':
        model_key  = parameter + '.' + str(int(summary_df.loc[date].Day_ID))
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
        'beta': r"[ppbv / $\mathregular{\mu}$mol m$^{-2}$]",
        'alpha': "[ppbv]",
        'gamma': "[ppbv]",
        'mu_alpha': "[ppbv]",
        'mu_beta': r"[ppbv / $\mathregular{\mu}$mol m$^{-2}$]",
        'sigma_alpha': "[ppbv]",
        'sigma_beta': r"[ppbv / $\mathregular{\mu}$mol m$^{-2}$]",
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
    plt.axhline(fitted_model.mean_values[model_key], color='black', lw=2, linestyle='--')
    if show_warmup_draws:
        plt.axvline(500, linestyle='--', color='grey', linewidth=0.5, label='Sampling begins')
    if compare_to_ground_truth:
        plt.axhline(ground_truth, color='red', lw=2, linestyle='--',)
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
    plt.axvline(fitted_model.mean_values[model_key], color='black', lw=2, linestyle='--', label='Mean value')
    if compare_to_ground_truth:
        plt.axvline(ground_truth, color='red', lw=2, linestyle='--', label='True value')
    plt.axvline(fitted_model.credible_intervals[model_key][0], linestyle=':', color='k', alpha=0.2, label=r'95% CI')
    plt.axvline(fitted_model.credible_intervals[model_key][1], linestyle=':', color='k', alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/autosaved/trace.png', dpi=300, bbox_inches='tight')
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
    date_df    = dataset_df[dataset_df.Date == date]

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
    date_df    = dataset_df[dataset_df.Date == date]

    # Needed to plot the regression lines
    x_min, x_max = np.min(date_df.obs_NO2), np.max(date_df.obs_NO2)
    x_domain = np.linspace(x_min, x_max, 100)

    # Humans think in terms of dates, stan thinks in terms of day ids. Need to be able to access parameter.day_id
    # using the passed date. Use the summary.csv file and index by date
    summary_df   = pd.read_csv(ct.FILE_PREFIX + '/data/' + fitted_model.run_name + '/summary.csv', header=0, index_col=0)

    day_id = int(summary_df.loc[date].Day_ID)

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

    for parameter in fitted_model.mean_values.keys():
        if 'beta.' in parameter:
            beta_values.append(fitted_model.mean_values[parameter])
            beta_error_bounds.append([fitted_model.mean_values[parameter] - fitted_model.credible_intervals[parameter][0],
                                      fitted_model.credible_intervals[parameter][1] - fitted_model.mean_values[parameter]])
        elif 'alpha.' in parameter:
            alpha_values.append(fitted_model.mean_values[parameter])
            alpha_error_bounds.append(
                [fitted_model.mean_values[parameter] - fitted_model.credible_intervals[parameter][0],
                 fitted_model.credible_intervals[parameter][1] - fitted_model.mean_values[parameter]])

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

    pearson = fitted_model.mean_values['rho']

    sigma_alpha = fitted_model.mean_values['sigma_alpha']
    mu_alpha    = fitted_model.mean_values['mu_alpha']

    sigma_beta = fitted_model.mean_values['sigma_beta']
    mu_beta    = fitted_model.mean_values['mu_beta']

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

def reduced_chi_squared(model_run):
    '''
    This function is for plotting a histogram of reduced :math:`\\chi^2` values for each day that was included in the model
    run when observations were dropped out.
    :param model_run: The name of the model run (must include "/dropout").
    :type model_run: string
    '''

    reduced_chi_square_df = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + model_run + '/dropout/reduced_chi_squared.csv')

    title = 'Reduced chi-squared values by day'
    sns.displot(reduced_chi_square_df.Reduced_chi_squared, kde=True)
    plt.xlabel(r'$\mathregular{\chi^2_{\nu}}$')

    # Check if this is a test dataset or not
    if 'days' in model_run:
        title += '\n Test Dataset'
    else:
        start_date, end_date, model = model_run.split('-')
        title += '\n' + datetime.datetime.strptime(start_date, "%Y%m%d").strftime("%B %-d, %Y") + ' - ' + \
                 datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%B %-d, %Y")
    plt.title(title)
    plt.tight_layout()
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

    residual_df_1 = pd.read_csv(ct.FILE_PREFIX + '/outputs/' + daily_mean_error_model_run + '/dropout/residuals.csv',
                                header=0)

    plt.hist(residual_df_1.Residuals, alpha=0.5, label='Average error model')

    title = 'Residual comparison'
    # Check if this is a test dataset
    if 'days' in daily_mean_error_model_run:
        title += '\n' + 'Test dataset'
    else:
        start_date, end_date, model = daily_mean_error_model_run.split('-')
        title += '\n' + datetime.datetime.strptime(start_date, "%Y%m%d").strftime("%B %-d, %Y") + ' - ' + \
                 datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%B %-d, %Y")

    plt.title(title)
    plt.legend()
    plt.xlabel(r'$\mathregular{CH_4^{pred}} - \mathregular{CH_4^{obs}}$ [ppbv]')
    plt.tight_layout()
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
        mean_beta = fitted_results.mean_values[parameter]
        lower_bound, upper_bound = fitted_results.credible_intervals[parameter]
        beta_df = beta_df.append({'Date': date,
                                  'Beta': mean_beta,
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