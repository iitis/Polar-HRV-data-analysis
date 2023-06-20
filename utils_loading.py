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

import pickle
import pandas as pd
import numpy as np
from typing import List
from retry import retry

from utils_preprocessing import (
    convert_absolute_time_to_timestamps_from_given_timestamp,
    interpolate_data_with_splines,
    remove_adjacent_beats,
    remove_consecutive_beats_after_holes,
    remove_first_and_last_indices,
    remove_manually_anomalies,
    remove_negative_timestamps,
    select_indices_to_filtering,
)
from utils_basic_plots import (
    plot_1D_signal,
    plot_accelerometer_data
)


def load_data_for_single_person(main_folder,
                                cur_person_group,
                                cur_person_number,
                                datatype):
    """
    Load measurements for a selected person.

    Arguments:
    ----------
      *main_folder*: (string) folder with experiment's files
      *cur_person_group*: (string) 'treatment' or 'control'
      *cur_person_number*: (int) number of the selected person
      *datatype*: (string) 'RR' (RR-interval) or 'ACC' (accelerometer)

    Returns:
      *data*: (Pandas dataframe) contains loaded data indicated
              by the function arguments
    """
    data = load_dataframe(
        main_folder, cur_person_group, cur_person_number, datatype)
    data["Phone timestamp"] = pd.to_datetime(data["Phone timestamp"])
    initial_timestamp = data.iloc[0]["Phone timestamp"]
    data = convert_absolute_time_to_timestamps_from_given_timestamp(
        data, initial_timestamp
    )
    return data


def load_and_preprocess_data_for_single_person(parameters,
                                               cur_person_group,
                                               cur_person_number,
                                               plot=False):
    """
    Prepare loading and full preprocesing of the data,
    i.e. removing of negative timestamps due to the device failure,
    removing of a few of first and last indices of the measurement,
    removing of a few heart beats after longer holes (e.g. due to
    device connection problems), manual anomaly detection + anomaly
    detection using Discrete Wavelet Transform, removing of a few
    heart beats near the anomalous ones. Possibly also apply
    data interpolation method.

    Arguments:
    ----------
      *parameters*: (dictionary) contains parameters, including
                    the number of seconds for which the indices
                    will be removed
      *cur_person_group*: (string) 'treatment' or 'control'
      *cur_person_number*: (int) number of the selected person
      *plot*: (Boolean) optional argument defining whether a plot
              after performing of Discrete Wavelet Transform
              should be prepared

    Returns:
    --------
      *data*: (Pandas Dataframe) loaded and preprocessed data
              with timestamps and corresponding RR intervals
    """
    data_type = 'rr_intervals'
    column_name = 'RR-interval [ms]'
    abbrv = 'RR'
    main_folder = parameters["main_folder"]

    # Load raw data for the selected person
    data = load_data_for_single_person(
        main_folder,
        cur_person_group,
        cur_person_number,
        abbrv)

    # Remove negative timedeltas. In some cases particular
    # measurements are obtained with delay
    data = remove_negative_timestamps(data)

    # Remove first and last few measurements as a typical source
    # of anomalies
    data = remove_first_and_last_indices(
        data,
        parameters['cut_time_from_start'],
        parameters['cut_time_before_finish']
    )

    # Remove some measurements after longer holes in the dataset
    data = remove_consecutive_beats_after_holes(
        data,
        parameters['threshold_for_hole_duration'],
        parameters['time_after_hole_for_removing']
    )

    data = data.reset_index(drop=True)
    # Prepare Discrete Wavelet Transform
    DWT_coefficients, filtered_indices = select_indices_to_filtering(
        data, column_name
    )
    if plot:
        if "plot_saving_folder" in parameters:
            saving_folder = parameters["plot_saving_folder"]
        else:
            saving_folder = None
        plot_1D_signal(
            data,
            data_type,
            column_name=[column_name],
            anomalies=filtered_indices,
            saving_folder=saving_folder,
            name=f'{data_type}_{cur_person_group}_{cur_person_number}'
        )

    if parameters['interpolation']:
        data_before_DWT = data.copy()
    # Remove neighbouring heart beats to the selected ones
    data = remove_adjacent_beats(
        data,
        filtered_indices,
        parameters['adjacent_beats_for_removing']
    )

    # Remove anomalies which have been detected manually
    data = remove_manually_anomalies(
        data,
        cur_person_group,
        cur_person_number
    )

    # Prepare data interpolation, if desired
    if parameters['interpolation']:
        data, predictions, predicted_timestamps = interpolate_data_with_splines(
            original_data=data_before_DWT,
            current_data=data,
            column_name=column_name
        )
    return data


@retry((FileNotFoundError, IOError))
def load_dataframe(folder, group, number, datatype):
    """
    Load Pandas dataframe according to the selected group
    and the number of the selected person in a given group.

    Arguments:
    ----------
       *folder*: (string) folder with experiment's files
       *group*: (string) a kind of people's group: 'control'
                or 'treatment'
       *number*: (int) the number of a given person in group
       *datatype*" (string) available options: 'ACC' or 'RR'

    Returns:
    --------
       *data*: Pandas dataframe with loaded data
    """
    if datatype not in ['RR', 'ACC']:
        return ValueError(
            'Wrong type of data. Possible options: "ACC" or "RR".')

    data = pd.read_csv(
        f'{folder}{group}_{number}.csv',
        delimiter=';'
    )
    return data


def store_HRV_results_different_methods(HRV_results: np.ndarray | float,
                                        timestamps: np.ndarray | None,
                                        group: str,
                                        person: int) -> List[float | list]:
    """
    Prepare a list summarizing results for a current person.

    Arguments:
    ----------
      *HRV_results*: (Numpy array | float) a single number or a table of
                     numbers representing consecutive HRV values
      *timestamps*: (Numpy array | None) represents a table of timestamps
                    to corresponding HRV values (in the case of Numpy array
                    in *HRV_results*) or None (in the case of a float number
                    in *HRV_results*)
      *group*: (string) the name of the tested group
      *person* (int) the number of the currently tested person
    """
    if timestamps is None:
        result = [group, person, HRV_results]
    else:
        result = [group, person, list(HRV_results), list(timestamps)]
    return result


def create_dataframe_from_HRV_results_different_methods(
        results: List[float | list],
        method: str) -> pd.DataFrame:
    """
    Prepares a Pandas Dataframe with previously prepared results.

    Arguments:
    ----------
       *results*: a list of floats or a list of lists having the names
                  of groups, the number of persons, the values of the HRV,
                  and potentially also timestamps
       *method*: (string) the name of the HRV calculation method

    Returns:
    --------
       Pandas Dataframe containing prepared results
    """
    if len(results[0]) == 3:
        dataframe = pd.DataFrame(
            results,
            columns=['group', 'no_of_person', f'HRV_{method}']
        )
    elif len(results[0]) == 4:
        dataframe = pd.DataFrame(
            results,
            columns=['group', 'no_of_person', f'HRV_{method}', 'timestamps']
        )
    else:
        raise ValueError('Wrong shape of the table with results!')
    return dataframe


def load_results_file(fname):
    """
    Load a pickle file.

    Argument:
    ---------
      *fname* (string) path to the file

    Returns an loaded object.
    """
    with open(fname, "rb") as fobj:
        return pickle.load(fobj)


if __name__ == "__main__":
    main_folder = (
        '/mnt/samba/Actual/Medical_project/Measurements_Exp_1/'
        'Exp_1_HRV_calculations_anonimized_accelerometer_data/'
    )

    # Plot accelerometer data
    folder_for_ACC_plots = '../Plots/raw_accelerometer_data/'
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
            else:
                data = load_data_for_single_person(
                    main_folder,
                    group,
                    person,
                    'ACC'
                )
                plot_accelerometer_data(
                    data,
                    folder_for_ACC_plots,
                    name=f'{group}_{person}'
                )
