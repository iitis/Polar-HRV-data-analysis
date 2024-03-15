"""
Copyright 2023-2024
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
Polar HRV Data Analysis Library (PDAL) v 1.1
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

from typing import Iterable, Tuple
import pickle
import pandas as pd
import numpy as np
import matplotlib.dates as mdates


def RMSSD_HRV_calculation(data):
    """
    Calculate the root mean square of successive differences
    between heartbeats.

    Arguments:
    ----------
      *data*: (Pandas series) data including RR-intervals' values

    Returns:
    --------
      (float) HRV value
    """
    # In the case of empty Series or Series with one elements
    if len(data) in [0, 1]:
        return 0
    else:
        RR_intervals_differences = data.diff()[1:]
        RR_intervals_squared = RR_intervals_differences ** 2
        # Remove elements just after at least 2-seconds holes
        time_differences = RR_intervals_squared.index.to_series().diff()
        outliers = np.argwhere(
            time_differences.values > np.timedelta64(2000000000, 'ns')
        ).flatten()
        indices_to_remove = time_differences.index[outliers]
        RR_intervals_squared = RR_intervals_squared.drop(
            labels=indices_to_remove)
        # Calculate HRV values
        HRV = np.sqrt(np.mean(RR_intervals_squared.values))
    return HRV


def SDNN_HRV_calculation(data):
    """
    Calculate standard deviation of RR intervals without outliers.

    Arguments:
    ----------
       *data*: (Pandas series) data including RR-intervals' values

    Returns:
    --------
      (float) HRV value
    """
    # In the case of empty Series or Series with one elements
    if len(data) in [0, 1]:
        return 0
    else:
        # Remove time intervals having values larger than 2 seconds
        outliers = data.loc[lambda x: x > 2000]
        RR_intervals = data.drop(outliers.index)
        HRV = RR_intervals.std(ddof=0)
        return HRV


def pNN50_HRV_calculation(data):
    """
    Calculate pNN50, i.e. number of pairs of adjacent RR intervals
    for which the difference between them is larger than 50 ms and
    divide this value by the total number of RR intervals.

    Arguments:
    ----------
       *data*: (Pandas series) data including RR-intervals' values

    Returns:
    --------
      (float) HRV value
    """
    # In the case of empty Series or Series with one elements
    if len(data) in [0, 1]:
        return 0.
    else:
        # Remove time intervals having values larger than 2 seconds
        outliers = data.loc[lambda x: x > 2000]
        RR_intervals = data.drop(outliers.index)
        RR_intervals_differences = RR_intervals.diff()[1:]
        NN50 = len(RR_intervals_differences.loc[lambda x: abs(x) > 50])
        if NN50 == 0:
            # There is a possibility that none of the differences
            # is larger than 50 miliseconds
            return 0.
        else:
            pNN50 = float(NN50 / len(RR_intervals_differences))
            return pNN50


def calculate_mean_HRV_based_on_windows(row, method):
    """
    Modify each row of Pandas dataframe by the calculation
    of mean HRV based on partial HRV results and replacing
    the list with all values.

    Arguments:
    ----------
      *row*: (Pandas series) contains results for a single person;
             one of the columns is called f'HRV_{method}'
      *method*: (string) the name of the method of HRV calculation

    Returns:
    --------
      *row*: (Pandas series) modified *row*
    """
    elements = np.array(row[f'HRV_{method}'])
    timestamps = np.array(row['timestamps'])
    # Zeros are not wrong elements for 'pNN50' method
    if method == 'pNN50':
        indices_of_wrong_elements = np.empty(
            shape=(0, 1), dtype=int)
    else:
        indices_of_wrong_elements = np.where(
            elements < 1e-6)
    elements = np.delete(elements, indices_of_wrong_elements)
    row[f'HRV_{method}'] = np.mean(elements)
    row['timestamps'] = list(np.delete(timestamps, indices_of_wrong_elements))
    return row


def get_indices_from_slides(element):
    # Prepare a conversion
    element = element.index.to_series()
    # Calculate differences between consecutive timestamps
    element = element.diff().dropna()
    # Calculate cumulative sums ensuring relative times
    # from the initial moment of the current sliding window
    element = element.dt.total_seconds().cumsum()
    return element


def sliding_data(data, interval_time):
    """
    Split Pandas dataframe, according to the selected interval.

    Arguments:
    ----------
      *data*: Pandas dataframe with index of DatetimeIndex type
              and selected values in a column
      *interval_time*: string or DateOffset or Timedelta representing
                       the length of the interval between consecutive
                       splits

    Returns:
    --------
      *sliding_window_data*: Pandas dataframe with index of DatetimeIndex
                             type and values from *data* stored in lists
                             in consecutive rows.
      *relative_times*: Pandas dataframe with index of DatetimeIndex
                        type and floats corresponding to the relative time
                        from the beginning of the current interval, stored
                        in lists.
      *original_times*: Pandas dataframe with index of DatetimeIndex
                        type and Pandas Timestamp corresponding to the
                        absolute time for observations within the current
                        interval, stored in lists.
    """
    sliding_window_data = data.resample(interval_time).apply(list)
    relative_times = data.resample(interval_time).apply(
        lambda x: list(get_indices_from_slides(x))
    )
    original_times = data.resample(interval_time).apply(
        lambda x: list(x.index.to_series())
    )

    assert sliding_window_data.shape[0] == relative_times.shape[0]
    assert sliding_window_data.shape[0] == original_times.shape[0]
    return sliding_window_data, relative_times, original_times


def generate_slide_over_series(series: pd.Series,
                               step_frequency: pd.Timedelta,
                               win_size: pd.Timedelta) -> Iterable[pd.Series]:
    """
    Returns (also irregularily sampled) a generator
    of windows of time length 'win_size' with any
    time step 'step_frequency' (it can be different
    than 'win_size').

    Arguments:
    ----------
      *series*: (Pandas Series) contains all data that should be split;
      *win_size*: (Pandas Timedelta) defines a time period during
                  which the data is collected, i.e. '2 min'
                  means that data between 12:00 and 12:02 will be
                  stored (if 12:00 is a starting point);
      *step_frequency*: (Pandas Timedelta) defines a time interval between
                        consecutive time windows, i.e. '3 min' means that
                        data will be stored between 12:00 and 12:03, 12:03
                        and 12:06, etc. Time windows can partially overlap.
                        *step_frequency* could not be greater than *win_size*.
    Returns:
    --------
      A generator yielding consecutive time windows with the collected data.
    """
    steps = pd.date_range(series.index[0],
                          series.index[-1],
                          freq=step_frequency)
    for step in steps:
        end = step + win_size
        yield series[step:end]


def prepare_windows_any_frequency_any_step(series: pd.Series,
                                           step_frequency: pd.Timedelta,
                                           win_size: pd.Timedelta) -> list[pd.Series]:
    """
    Returns (also irregularily sampled) a list of
    Pandas Series containing multiple parts of the
    *series*. Consecutive parts have time length
    'win_size' while a time interval between parts
    is denoted as 'step_frequency' (it can be different
    than 'win_size').

    Arguments:
    ----------
      *series*: (Pandas Series) contains all data that should be split;
      *win_size*: (Pandas Timedelta) defines a time period during
                  which the data is collected, i.e. '2 min'
                  means that data between 12:00 and 12:02 will be
                  stored (if 12:00 is a starting point);
      *step_frequency*: (Pandas Timedelta) defines a time interval between
                        consecutive time windows, i.e. '3 min' means that
                        data will be stored between 12:00 and 12:03, 12:03
                        and 12:06, etc. Time windows can partially overlap.
                        *step_frequency* could not be greater than *win_size*.
    Returns:
    --------
      A generator yielding consecutive time windows with the collected data.
    """
    # In the following case some data may be omitted!
    assert step_frequency <= win_size

    return list(generate_slide_over_series(
        series, step_frequency, win_size
    ))


def find_and_filter_missing_data(HRV_results: list | np.ndarray,
                                 timestamps: list | np.ndarray,
                                 method: str) \
                                    -> Tuple[np.ndarray, np.ndarray]:
    """
    In some cases, due to the lack of data, HRV values from
    particular windows may be equal to 0 and corresponding timestamps
    are related to the Unix epoch. They should be removed.
    0 is not an error when a tested method is 'pNN50'

    Arguments:
    ----------
      *HRV_results* - (list or Numpy array) contains HRV values
                      from consecutive timestamps
      *timestamps* - (list or Numpy array) contains timestamps
                     related to the HRV values from *HRV_results*
      *method* - (str) defines which HRV calculation method was used

    Returns:
    --------
      Potentially modified *HRV_results* and *timestamps*, both are
      Numpy arrays.
    """
    # For some methods like 'pNN50' all HRV values can be equal to 0
    # In such cases, it does not result from a calculation bug
    if method == 'pNN50':
        indices_HRV = np.empty(shape=(0, 1), dtype=int)
    else:
        indices_HRV = np.argwhere(np.array(HRV_results) < 1e-8)
    indices_time = np.argwhere(
        np.array(timestamps) == np.datetime64('1970-01-01T00:00:00'))
    all_indices_to_remove = np.union1d(indices_HRV, indices_time)
    HRV_results = np.delete(HRV_results, all_indices_to_remove)
    timestamps = np.delete(timestamps, all_indices_to_remove)
    return HRV_results, timestamps


def calculate_HRV_in_windows(data: pd.DataFrame,
                             step_frequency: str | pd.Timedelta,
                             window_size: str | pd.Timedelta,
                             method: str,
                             save: bool = False,
                             path_with_filename: str = "") -> Tuple[
                                 np.ndarray, np.ndarray]:
    """
    Calculate the values of HRV for a given person with a division
    of the sequence into multiple subsequences (windows), according
    to the selected method of HRV calculation.

    Arguments:
    ----------
       *data*: (Pandas Dataframe) contains with a columns: 'Phone timestamp'
               and 'RR-interval [ms]';
       *step_frequency*: (Pandas Timedelta) defines a time interval between
                        consecutive time windows, i.e. '3 min' means that
                        data will be stored between 12:00 and 12:03, 12:03
                        and 12:06, etc. Time windows can partially overlap.
                        *step_frequency* could not be greater than *win_size*.
        *win_size*: (Pandas Timedelta) defines a time period during
                    which the data is collected, i.e. '2 min'
                    means that data between 12:00 and 12:02 will be
                    stored (if 12:00 is a starting point);
        *method*: (str) method of HRV calculation;
                  possible options:
                  - RMSSD - root mean square of successive differences
                  - SDNN - standard deviation of RR intervals without
                           anomalies
                  - pNN50 - number of RR intervals differing by more than
                            50ms divided by the total number of RR intervals
        *save*: (optional Boolean) defines whether a list of Pandas series
                with filtered R-R intervals should be stored
        *path_with_filename*: (optional string) defines path and filename if
                              filtered R-R intervals have to be saved; if not,
                              leave empty. If *save* is True, but path is not
                              given, R-R intervals will be saved in the current
                              path.

    Returns:
    --------
        *HRV_divided_series*: (Numpy array) contains HRV values for consecutive
                              subsequences;
        *median_timestamps*: (Numpy array) contains median timestamps for
                             subsequences selected previously.
    """
    step_frequency = pd.Timedelta(step_frequency)
    window_size = pd.Timedelta(window_size)
    full_series = data.copy()
    full_series.set_index('Phone timestamp', inplace=True)
    full_series = full_series.squeeze()
    # Divide a given Series into multiple windows
    divided_series = prepare_windows_any_frequency_any_step(
        full_series, step_frequency, window_size)
    # Prepare filtering of the above windows
    divided_series = filter_windows_with_chunked_dataframe(
        divided_series)
    if save:
        if not path_with_filename:
            path_with_filename = './RR_filtered_intervals_with_time.pkl'
        with open(path_with_filename, 'wb') as f:
            pickle.dump(divided_series, f)
    # Calculate HRV values according to the selected method
    HRV_divided_series = np.zeros(len(divided_series))
    median_timestamps = np.zeros(len(divided_series), dtype='datetime64[ns]')
    for i in range(HRV_divided_series.shape[0]):
        if len(divided_series[i]) > 1:
            if method == 'RMSSD':
                HRV_divided_series[i] = RMSSD_HRV_calculation(
                    divided_series[i]
                )
            elif method == 'SDNN':
                HRV_divided_series[i] = SDNN_HRV_calculation(
                    divided_series[i]
                )
            elif method == 'pNN50':
                HRV_divided_series[i] = pNN50_HRV_calculation(
                    divided_series[i]
                )
            else:
                raise NotImplementedError
            median_timestamps[i] = mdates.num2date(
                np.median(mdates.date2num(divided_series[i].index)))
    HRV_divided_series, median_timestamps = find_and_filter_missing_data(
        HRV_divided_series, median_timestamps, method
    )
    return (HRV_divided_series, median_timestamps)


def filter_windows_with_chunked_dataframe(divided_series: list[pd.Series]
                                          ) -> list[pd.Series]:
    """
    Filter out selected subsets of values for further calculations
    to make them more reliable. To remove redundant repetitions,
    Pandas Series starting with the same timestamps and having
    the same number of elements will be filtered.
    Only the last Series will be left.

    Arguments:
    ----------
       *divided_series*: (list) contains Pandas Series of subsets of elements
                         divided in windows
    Returns:
    --------
       A filtered list of Pandas Series.
    """
    # If more than Pandas Series has the same starting point AND the same
    # number of elements we have to remove all series except the last one
    divided_series = np.array(divided_series, dtype='object')
    list_filter = np.zeros(len(divided_series), dtype=bool)

    for i in range(1, len(divided_series)):
        if len(divided_series[i] > 0) and len(divided_series[i - 1] > 0):
            if (divided_series[i].index[0] == divided_series[i - 1].index[0]) \
               and (len(divided_series[i]) == len(divided_series[i - 1])):
                list_filter[i - 1] = True

    filtered_series = list(np.delete(divided_series, np.argwhere(list_filter)))
    return filtered_series


if __name__ == "__main__":
    pass
