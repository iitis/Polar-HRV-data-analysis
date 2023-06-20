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

from itertools import product
import os
import pandas as pd
import numpy as np

from HRV_calculation import (
    calculate_HRV_in_windows,
    RMSSD_HRV_calculation
)
from utils_loading import (
    create_dataframe_from_HRV_results_different_methods,
    store_HRV_results_different_methods
)
from utils_postprocessing import (
    save_results,
    save_parameters
)
from utils_basic_plots import (
    boxplot_HRV,
    plot_1D_signal,
    plot_distribution_PANSS_subcategories,
    regression_PANSS
)
from utils_loading import (
    load_data_for_single_person,
    load_and_preprocess_data_for_single_person
)
from utils_others import (
    compare_means_and_variances_in_groups,
    filter_patients_with_quetiapine
)


def pipeline_load_data_and_calculate_HRV(
    cur_person_group,
    cur_person_number,
    parameters
):
    """
    Pipeline for HRV calculation for a selected person.
    1) Load data and convert timestamps to pandas.datetime type.
    2) Detect and filter out anomalous observations using Discrete
    Wavelet Transform (DWT).
    3) Plot signal before and after the application of DWT.
    4) Calculate HRV value(s).

    Arguments:
    ----------
        *cur_person_group* (str) - the name of the person's group (e.g.
                                   'control' or 'treatment')
        *cur_person_number* (int) - number of the person in 'cur_person_group'
        *parameters* - dictionary with the following keys:
            -sequence_range- 'windows', 'full' or 'without_HRV_calculation'
            -method- 'RMSSD' or others
            -step_frequency- (only if sequence_range is 'windows') pd.Timedelta
            -window_size- (only if sequence_range is 'windows') pd.Timedelta
            -main_folder- path with data
            -name- a summary of experiment's parameters
            -plot_saving_folder- folder for saving plots
            -preprocessing- Boolean value: True if has to be done, False
                            in the opposite case
            -interpolation- Boolean value: True if has to be done, False
                            in the opposite case
            -plot- Boolean value: True if plot after the automatic anomaly
                   detection has to prepared, False in the opposite case

    Returns:
    --------
    Option 1): (Pandas Dataframe, float, None) RR-interval data, mean HRV
    Option 2): (Pandas Dataframe, Numpy array, Numpy array): RR-interval
               data, HRV values from consecutive time windows and corresponding
               median timestamps of windows
    Option 3): (Pandas Dataframe, None, None) RR-interval data
    """
    data_type = 'rr_intervals'
    column_name = 'RR-interval [ms]'
    abbrv = 'RR'
    main_folder = parameters["main_folder"]

    if "plot_saving_folder" in parameters:
        saving_folder = parameters["plot_saving_folder"]
    else:
        saving_folder = None
    print(f'group: {cur_person_group}, person: {cur_person_number}')

    # Load data for a selected person. Optionally, perform preprocessing
    if parameters['preprocessing']:
        data = load_and_preprocess_data_for_single_person(
            parameters,
            cur_person_group,
            cur_person_number,
            plot=parameters['plot']
        )
    else:
        data = load_data_for_single_person(
            main_folder,
            cur_person_group,
            cur_person_number,
            abbrv)

    plot_1D_signal(
        data,
        data_type,
        column_name=[column_name],
        saving_folder=saving_folder,
        name=f'filtered_{data_type}_{cur_person_group}_{cur_person_number}')

    if parameters['sequence_range'] == 'windows':
        HRV_windows_values, median_timestamps = calculate_HRV_in_windows(
            data,
            step_frequency=parameters['step_frequency'],
            window_size=parameters['window_size'],
            method=parameters['method'])
        return data, HRV_windows_values, median_timestamps
    elif parameters['sequence_range'] == 'full':
        if parameters['method'] == 'RMSSD':
            RR_intervals_series = pd.Series(
                data[column_name].values,
                index=data['Phone timestamp'].values,
                dtype=np.int64
            )
            return (
                data,
                RMSSD_HRV_calculation(RR_intervals_series),
                None
            )
        else:
            raise ValueError('Wrong method of HRV calculation!')
    elif parameters['sequence_range'] == 'without_HRV_calculation':
        return data, None, None
    else:
        raise ValueError(
            'Wrong mode of "sequence_range" in "parameters" dict.')


def experiment_1_calculate_HRV(parameters):
    """
    Load data from the initial series of experiments,
    based on data collected between April and September 2022.

    Returns a Pandas dataframe with mean HRV for both
    'control' and 'treatment' group.
    """
    results = []
    for group in ['control', 'treatment']:
        for person in range(1, 49):
            if (group == 'treatment' and (
                person in [5, 6, 10, 11, 12, 14, 18, 28, 30, 34, 35, 39] or
                person > 42)) or \
               (group == 'control' and (
                person > 48 or
                person in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 17, 23, 27, 48])):
                continue
            _, HRV_results, timestamps = pipeline_load_data_and_calculate_HRV(
                group,
                person,
                parameters
            )
            results.append(
                store_HRV_results_different_methods(
                    HRV_results,
                    timestamps,
                    group,
                    person
                )
            )
    dataframe = create_dataframe_from_HRV_results_different_methods(
        results,
        method=parameters['method']
    )
    return dataframe


if __name__ == "__main__":
    main_folder = (
        '/data/anonimized_raw_data/'
    )
    accelerometer_folder = (
        '/data/anonimized_accelerometer_data/'
    )
    PANSS_localization = f'{main_folder}../'
    HRV_method = 'RMSSD'
    exclude_quetiapine = False
    sensitivity_analysis = True
    # -sequence_range- 'windows' or 'full'
    if sensitivity_analysis:
        step_frequencies = [
            '15 seconds', '30 seconds', '45 seconds',
            '1 min', '2 min', '5 min', '10 min'
        ]
        window_sizes = [
            '1 min', '2 min', '3 min', '5 min',
            '10 min', '15 min', '20 min'
        ]
        interpolation_options = [True, False]
        result_saving_folder = '../article_results/sensitivity_analysis/'
    else:
        step_frequencies = ['1 min']
        window_sizes = ['15 min']
        interpolation_options = [False]
        result_saving_folder = '../article_results/detailed_analysis/'
    os.makedirs(result_saving_folder, exist_ok=True)
    # step frequency cannot be greater than window size
    for step_frequency, window_size, interpolation in product(
            step_frequencies,
            window_sizes,
            interpolation_options):
        if pd.Timedelta(step_frequency) > pd.Timedelta(window_size):
            continue
        else:
            print(f'step: {step_frequency}, window: {window_size}')
            # additional parameters for preprocessing
            adjacents_beats_for_removing = '5 seconds'
            threshold_hole_duration = '30 seconds'
            time_after_hole_for_removing = '15 seconds'
            time_threshold_from_start = '45 seconds'
            time_threshold_before_finish = '45 seconds'
            parameters = {
                'sequence_range': 'windows',
                'method': HRV_method,
                'step_frequency': pd.Timedelta(step_frequency),
                'window_size': pd.Timedelta(window_size),
                'adjacent_beats_for_removing': adjacents_beats_for_removing,
                'threshold_for_hole_duration': threshold_hole_duration,
                'time_after_hole_for_removing': time_after_hole_for_removing,
                'cut_time_from_start': time_threshold_from_start,
                'cut_time_before_finish': time_threshold_before_finish,
                'main_folder': main_folder,
                'accelerometer_folder': accelerometer_folder,
                'preprocessing': True,
                'interpolation': interpolation,
                'exclude_quetiapine': exclude_quetiapine,
                'plot': True,
                'PANSS_loading_folder': PANSS_localization
            }
            parameters['name'] = (
                f'HRV_{parameters["method"]}_'
                f'mode_{parameters["sequence_range"]}_'
                f'step_{step_frequency}_'
                f'window_{window_size}_'
                f'interpolation_{parameters["interpolation"]}'
            )
            parameters['result_saving_folder'] = (
                f'{result_saving_folder}'
                f'interpolation_{parameters["interpolation"]}/'
            )
            if parameters['exclude_quetiapine']:
                parameters['plot_saving_folder'] = (
                    '../article_results/without_quetiapine/'
                    f'{parameters["name"]}/'
                )
            else:
                parameters['plot_saving_folder'] = (
                    f'{parameters["result_saving_folder"]}'
                    f'{parameters["name"]}/'
                )
            full_results = experiment_1_calculate_HRV(parameters)
            # Load PANSS results
            PANSS = pd.read_csv(
                f'{parameters["PANSS_loading_folder"]}/PANSS.csv',
                delimiter=';'
            )
            PANSS.insert(0, "group", "treatment")
            merged_results = full_results.merge(PANSS, how='outer')

            save_parameters(parameters)
            processed_data, treatment_results = save_results(
                merged_results, parameters)

            quetiapine_patients_results = None
            if parameters['exclude_quetiapine']:
                no_of_quetiapine_patients = [2, 4, 7, 15, 20, 29, 31]
                treatment_results, quetiapine_patients_results = \
                    filter_patients_with_quetiapine(
                        no_of_quetiapine_patients,
                        treatment_results
                    )

            regression_PANSS(treatment_results,
                             f'HRV_{parameters["method"]}',
                             parameters,
                             quetiapine_patients=quetiapine_patients_results)
    if not sensitivity_analysis:
        statistical_tests_results = compare_means_and_variances_in_groups(
            processed_data,
            HRV_method,
            parameters["result_saving_folder"]
        )
        boxplot_HRV(processed_data,
                    parameters['plot_saving_folder'],
                    x_axis_variable=f'HRV_{parameters["method"]}',
                    y_axis_variable='group')
        plot_distribution_PANSS_subcategories(
            parameters["PANSS_loading_folder"],
            '../Plots/'
        )
