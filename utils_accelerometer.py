"""
Copyright 2023
Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl

The main author of the code:
- Kamil Książek (ITAI PAS, ORCID ID: 0000-0002-0201-6220).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

---
Polar HRV Data Analysis Library (PDAL) v 1.0
---

A source code to the paper:

The analysis of heart rate variability and accelerometer mobility data
in the assessment of symptom severity in psychosis disorder patients
using a wearable Polar H10 sensor

Authors:
- Kamil Książek (ITAI PAS, ORCID ID: 0000-0002-0201-6220),
- Wilhelm Masarczyk (FMS MUS, ORCID ID: 0000-0001-9516-0709),
- Przemysław Głomb (ITAI PAS, ORCID ID: 0000-0002-0215-4674),
- Michał Romaszewski (ITAI PAS, ORCID ID: 0000-0002-8227-929X),
- Iga Stokłosa (FMS UMS, ORCID ID: 0000-0002-7283-5491),
- Piotr Ścisło (PDMH, ORCID ID: 0000-0003-1213-2935),
- Paweł Dębski (FMS UMS, ORCID ID: 0000-0001-5904-6407),
- Robert Pudlo (FMS UMS, ORCID ID: 0000-0002-5748-0063),
- Piotr Gorczyca (FMS UMS, ORCID ID: 0000-0002-9419-7988),
- Magdalena Piegza (FMS UMS, ORCID ID: 0000-0002-8009-7118).

*ITAI PAS* - Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences, Gliwice, Poland;
*FMS UMS* - Faculty of Medical Sciences in Zabrze,
Medical University of Silesia, Tarnowskie Góry, Poland;
*PDMH* - Psychiatric Department of the Multidisciplinary Hospital,
Tarnowskie Góry, Poland.
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter

from utils_loading import (
    load_data_for_single_person,
    load_and_preprocess_data_for_single_person,
)
from HRV_calculation import (
    calculate_HRV_in_windows,
    prepare_windows_any_frequency_any_step
)
from utils_others import append_row_to_file
from utils_postprocessing import save_parameters
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from scipy.stats import pearsonr


def load_raw_results_of_rest_states(path: str,
                                    group: str,
                                    number: int | str,
                                    method: str) -> pd.DataFrame:
    """
    Load raw results of the method for the selection
    of rest states and create a dataframe.

    Arguments:
    ----------
      *path* (string): path to the folder with results
      *group* (string): 'control' or 'treatment'
      *number* (int or string): number of the selected person
      *method* (string): suffix of the method for the selection
                         of rest states

    Returns:
    --------
      *data* (Pandas DataFrame) contains columns: 'start_timestamp'
             and 'end_timestamp' of pd.Timestamp type with the loaded
             results of the rest state selection method.
    """
    data = pd.read_csv(f'{path}{group}_{number}_{method}.csv',
                       delimiter=',',
                       names=['start_timestamp', 'end_timestamp'],
                       header=None)
    for column in ['start_timestamp', 'end_timestamp']:
        data[column] = pd.to_datetime(data[column])
    return data


def preprocess_acc_data(acc, s_Earth=1001):
    """
    Estimates and removes the momentary gravity component of the accelerometer
    measurement. It is estimated using a low pass filter.

    Arguments:
    ----------
       *acc* (Pandas DataFrame) contains accelerometer data
       *s_Earth* (int) value of the standard deviation of the Gaussian filter

    Returns:
    --------
       Modified accelerometer data
    """
    x, y, z = [acc[l].values for l in ['X [mg]', 'Y [mg]', 'Z [mg]']]
    xe, ye, ze = [gaussian_filter(a, sigma=s_Earth) for a in [x, y, z]]
    acc['Earth [mg, abs]'] = np.sqrt(xe**2 + ye**2 + ze**2)
    acc['Acc [mg, abs]'] = np.sqrt((x - xe)**2 + (y - ye)**2 + (z - ze)**2)
    return acc


def find_nearest_value(array, value):
    """
    Finds the timestamp from the array for which the distance
    to the value is the least

    Arguments:
    ----------
       *array* (Numpy array) contains timestamps of numpy.datetime64 format
       *value* (Numpy datetime64) timestamps for which the corresponding
               timestamp from *array* will be sought

    Returns:
    --------
       A timestamp which meets the above conditions.
    """
    return array[np.abs(array - value).argmin()]


def load_and_filter_data(parameters,
                         group,
                         number):
    """
    Load accelerometer and RR intervals data and prepare initial
    filtering ensuring the same time range for both dataframes

    Arguments:
    ----------
      *parameters* - (dictionary) contains hyperparameters of the experiment
      *group* - (str) 'treatment' or 'control'
      *number* - (int) defines number of a given person
                  from the selected group

    Returns:
    --------
      *data_ACC* - (Pandas DataFrame) contains accelerometer data
      *data_RR* - (Pandas DataFrame) contains RR interval data
      *min_timestamp* - (Pandas Timestamp) corresponds to the lower
                        time boundary
    """
    # Load accelerometer data
    data_ACC = load_data_for_single_person(parameters['accelerometer_folder'],
                                           group,
                                           number,
                                           'ACC')
    data_ACC.set_index('Phone timestamp', inplace=True)
    data_ACC = data_ACC.squeeze().copy()
    min_ACC_index, max_ACC_index = data_ACC.index[0], data_ACC.index[-1]

    # Load RR intervals data
    data_RR = load_and_preprocess_data_for_single_person(
        parameters,
        group,
        number
    )
    data_RR.set_index('Phone timestamp', inplace=True)
    min_RR_index, max_RR_index = data_RR.index[0], data_RR.index[-1]
    min_timestamp = pd.Series([min_ACC_index, min_RR_index]).max()
    max_timestamp = pd.Series([max_ACC_index, max_RR_index]).min()

    # Filtering both dataframes
    data_ACC = data_ACC.loc[
        (data_ACC.index >= min_timestamp) &
        (data_ACC.index <= max_timestamp)].copy()
    data_RR = data_RR.loc[
        (data_RR.index >= min_timestamp) &
        (data_RR.index <= max_timestamp)].copy()
    return data_ACC, data_RR, min_timestamp


def process_accelerometer_data(subseries_ACC_data):
    """
    Calculate mean values from the accelerometer data
    within selected windows.

    Arguments:
    ----------
      *subseries_ACC_data* - (list) contains Pandas DataFrames with
                             partial data

    Returns:
    --------
      *timestamps_ACC_numpy* - (Numpy array) contains values of timestamps
                               of Numpy.datetime64 format
      *results_for_ACC* - (list) contains float values corresponding to
                          mean mobility values
    """
    results_for_ACC, timestamps_ACC = [], []
    for i in range(len(subseries_ACC_data)):
        if len(subseries_ACC_data[i]) > 0:
            part_of_ACC_data = preprocess_acc_data(subseries_ACC_data[i].copy())
            timestamps_ACC.append(pd.Timestamp(mdates.num2date(
                np.median(mdates.date2num(subseries_ACC_data[i].index)))))
            results_for_ACC.append(part_of_ACC_data['Acc [mg, abs]'].mean())
    timestamps_ACC_numpy = pd.DatetimeIndex(timestamps_ACC).values
    return timestamps_ACC_numpy, results_for_ACC


def process_RR_data_corresponding_to_ACC(data_RR, timestamps_ACC):
    """
    Calculate HRV values based on RR intervals and change
    timestamps to the ones that will correspond to timestamps
    from the accelerometer data

    Arguments:
    ----------
      *data_RR* - (Pandas DataFrame) contains raw RR interval data
      *timestamps_ACC* - (Numpy array) contains values of numpy.datetime64

    Returns:
    --------
      *HRV_dataframe* - (Pandas DataFrame) contains values of HRV with
                        timestamps
    """
    data_RR = data_RR.reset_index()
    HRV_windows_values, timestamps_RR = calculate_HRV_in_windows(
        data_RR,
        parameters['step_frequency'],
        parameters['window_size'],
        'RMSSD')

    # Replace timestamps related to RR data by timestamps from ACC data
    # which are nearest to selected RR measurements
    for i in range(timestamps_RR.shape[0]):
        timestamps_RR[i] = pd.Timestamp(
            find_nearest_value(
                timestamps_ACC,
                timestamps_RR[i]
            )
        )

    HRV_dataframe = pd.DataFrame(
        HRV_windows_values,
        index=timestamps_RR,
        columns=['HRV']
    )
    HRV_dataframe = HRV_dataframe[
        ~HRV_dataframe.index.duplicated(keep='first')]
    return HRV_dataframe


def clean_accelerometer_data_and_fill_according_to_HRV(ACC_dataframe,
                                                       HRV_dataframe,
                                                       boundary_timestamp,
                                                       window_size):
    """
    Prepare postprocessing of dataframes containing accelerometer
    and HRV values. Remove duplicates and interpolate HRV data in the places
    where accelerometer data are available. Also, remove NaN data and prepare
    dataframes that they should have the same time range.

    Arguments:
    ----------
      *ACC_dataframe* -  (Pandas DataFrame) contains accelerometer data with
                        timestamps
      *HRV_dataframe* - (Pandas DataFrame) contains values of HRV with
                        timestamps
      *boundary_timestamp* - (Pandas Timestamp) corresponds to the lower
                             time boundary
      *window_size* - (Pandas Timedelta) defines length of the window
                      that should be prepared

    Returns:
    --------
      *ACC_dataframe* - (Pandas DataFrame) contains postprocessed accelerometer
                        data
      *HRV_resampled_dataframe* - (Pandas DataFrame) contains postprocessed
                                  HRV values
    """
    # Sometimes data may be duplicated due to lacks in data, for instance:
    # 11:13:36.970, 11:13:40.719, ..., 11:14:36.958, 11:14:40.707. With
    # 1-minute windows and 1-second time step there will be some rows
    # with exactly the same measurements. Therefore, it is necessary
    # to remove duplicates.
    ACC_dataframe = ACC_dataframe[
        ~ACC_dataframe.index.duplicated(keep='first')]

    positions_of_missing_indices = ~ACC_dataframe.index.isin(
        HRV_dataframe.index)
    missing_indices = ACC_dataframe[positions_of_missing_indices].copy()
    missing_indices.loc[:] = np.nan
    missing_indices = missing_indices.rename(columns={'mg': 'HRV'})
    HRV_resampled_dataframe = pd.concat([HRV_dataframe,
                                         missing_indices])
    assert len(ACC_dataframe) == len(HRV_dataframe) + len(missing_indices)
    HRV_resampled_dataframe = HRV_resampled_dataframe.sort_index()
    HRV_resampled_dataframe = HRV_resampled_dataframe.interpolate(
        method='linear')
    # SANITY CHECK! NaNs should be at most 'window_size' after
    # the beginning of ACC_dataframe
    nan_indices = HRV_resampled_dataframe['HRV'].index[
        HRV_resampled_dataframe['HRV'].apply(np.isnan)]
    if len(nan_indices) > 0:
        last_nan_index = nan_indices[-1]
        assert last_nan_index <= (boundary_timestamp + window_size)
        HRV_resampled_dataframe = HRV_resampled_dataframe.dropna()
        ACC_dataframe = ACC_dataframe.loc[
            ACC_dataframe.index > last_nan_index]
    return ACC_dataframe, HRV_resampled_dataframe


def plot_accelerometer_vs_HRV_data(HRV_dataframe,
                                   ACC_dataframe,
                                   parameters,
                                   group,
                                   number):
    """
    Prepare a plot comparing HRV with accelerometer data for a selected
    person and save results to the text file

    Arguments:
    ----------
      *HRV_dataframe* - (Pandas DataFrame) contains values of HRV with
                        timestamps
      *ACC_dataframe* - (Pandas DataFrame) contains accelerometer values
                        with timestamps
      *parameters* - (dictionary) contains following keys: 'plot_saving_folder'
                     and 'file_for_saving_results'
      *group* - (str) 'treatment' or 'control'
      *number* - (int) defines number of a given person
                  from the selected group
    """
    saving_folder = parameters['plot_saving_folder']
    file_for_saving_results = parameters['file_for_saving_results']
    mean_HRV = HRV_dataframe['HRV'].mean()
    # Plot of two curves
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    ax_2 = ax.twinx()
    ax.plot(HRV_dataframe.index.values,
            HRV_dataframe.values,
            color='red',
            label='HRV')
    ax.set_ylabel('HRV', color='red')
    ax.tick_params(axis='y', colors='red')
    ax.grid(color='red', alpha=0.2)

    ax_2.plot(ACC_dataframe.index.values,
              ACC_dataframe.values,
              color='blue',
              label='accelerometer')
    ax_2.set_ylabel('mobility [mg]', color='blue')
    ax_2.tick_params(axis='y', colors='blue')
    ax_2.grid(color='blue', alpha=0.2)
    ax.set_xlabel('Timestamp')
    myFmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)
    ax.tick_params(axis='x', labelrotation=90)
    statistics, p_value = pearsonr(
        HRV_dataframe.values.flatten(),
        ACC_dataframe.values.flatten()
    )

    plt.title('HRV vs mobility: '
              f'Pearson r: {statistics:.2f}, p-value: {p_value:.2f}; '
              f'mean HRV: {mean_HRV:.2f}')
    plt.tight_layout()
    plt.savefig(f'{saving_folder}{group}_{number}.pdf', dpi=400)
    plt.close()

    append_row_to_file(
        f'{saving_folder}{file_for_saving_results}',
        (f'{group};{number};{mean_HRV};{statistics};{p_value}')
    )


def plot_correlation_HRV_and_mobility_vs_HRV(saving_folder):
    """
    Plot a dependency between mean HRV and Pearson's r between
    mean HRV and mobility (calculated based on accelerometer
    data).

    Argument:
    ---------
      *saving_folder* (string) path to the folder with results
    """
    correlation_data = pd.read_csv(
        f'{saving_folder}results.csv', delimiter=';')
    palette = ['red', 'blue']
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.scatterplot(
        data=correlation_data,
        x="Pearson_r",
        y="mean_HRV",
        palette=sns.color_palette(palette, 2),
        s=40,
        alpha=0.75,
        hue="group"
    )
    plt.xlabel("Pearson's r between mean HRV and mobility")
    plt.ylabel('mean HRV')
    plt.tight_layout()
    plt.savefig(f'{saving_folder}correlation_mobility_vs_HRV.pdf', dpi=400)
    plt.close()


def main_accelerometer_processing(parameters,
                                  group,
                                  number):
    """
    Main procedure of the calculation of the accelerometer data
    for a single person.

    Arguments:
    ----------
       *parameters* - (dictionary) contains the following keys:
         -main_folder-, -step_frequency-, -window_size-,
         -cut_time_from_start-, -cut_time_before_finish-,
         -threshold_for_hole_duration-, -time_after_hole_for_removing',
         -interpolation-, -adjacent_beats_for_removing'
       *group* - (str) 'treatment' or 'control'
       *number* - (int) defines number of a given person
                  from the selected group
    """
    data_ACC, data_RR, min_timestamp = load_and_filter_data(
        parameters,
        group,
        number
    )
    subseries_ACC = prepare_windows_any_frequency_any_step(
        data_ACC,
        parameters['step_frequency'],
        parameters['window_size']
    )
    timestamps_ACC, results_for_ACC = process_accelerometer_data(
        subseries_ACC)
    HRV_dataframe = process_RR_data_corresponding_to_ACC(data_RR,
                                                         timestamps_ACC)
    ACC_dataframe = pd.DataFrame(
        results_for_ACC,
        index=timestamps_ACC,
        columns=['mg']
    )
    ACC_dataframe, HRV_dataframe = \
        clean_accelerometer_data_and_fill_according_to_HRV(
            ACC_dataframe,
            HRV_dataframe,
            min_timestamp,
            window_size
        )
    # Save calculated data
    HRV_dataframe.to_pickle(
        f'{saving_folder}{group}_{number}_HRV.pkl'
    )
    ACC_dataframe.to_pickle(
        f'{saving_folder}{group}_{number}_accelerometer.pkl'
    )
    plot_accelerometer_vs_HRV_data(HRV_dataframe,
                                   ACC_dataframe,
                                   parameters,
                                   group,
                                   number)


if __name__ == "__main__":
    saving_folder = '../Correlations_corrected/'
    os.makedirs(saving_folder, exist_ok=True)

    accelerometer_folder = (
        '/data/anonimized_accelerometer_data/'
    )
    RR_folder = (
        '/data/anonimized_raw_data/'
    )

    file_for_saving_results = 'results.csv'
    adjacents_beats_for_removing = '5 seconds'
    threshold_hole_duration = '30 seconds'
    time_after_hole_for_removing = '15 seconds'
    time_threshold_from_start = '45 seconds'
    time_threshold_before_finish = '45 seconds'
    step_frequency = pd.Timedelta('1s')
    window_size = pd.Timedelta('5 min')

    # Path to the RR-interval data should be in the 'main_folder'
    # key while path to the accelerometer data is located at
    # 'accelerometer_data' key
    parameters = {
        'main_folder': RR_folder,
        'accelerometer_folder': accelerometer_folder,
        'plot_saving_folder': saving_folder,
        'file_for_saving_results': file_for_saving_results,
        'cut_time_from_start': time_threshold_from_start,
        'cut_time_before_finish': time_threshold_before_finish,
        'threshold_for_hole_duration': threshold_hole_duration,
        'time_after_hole_for_removing': time_after_hole_for_removing,
        'adjacent_beats_for_removing': adjacents_beats_for_removing,
        'step_frequency': step_frequency,
        'window_size': window_size,
        'interpolation': False,
    }

    save_parameters(parameters)
    append_row_to_file(
        f'{saving_folder}{file_for_saving_results}',
        ('group;number;mean_HRV;Pearson_r;p_value')
    )

    persons = {
        'treatment': [1, 2, 3, 4, 7, 8, 9, 13, 15, 16,
                      17, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                      29, 31, 32, 33, 36, 37, 38, 40, 41, 42],
        'control': [2, 16, 18, 19, 20, 21, 22, 24, 25, 26,
                    28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                    38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    }

    for group in list(persons.keys()):
        for number in persons[group]:
            print(f'group: {group}, number: {number}')
            main_accelerometer_processing(
                parameters,
                group,
                number
            )
    plot_correlation_HRV_and_mobility_vs_HRV(saving_folder)
