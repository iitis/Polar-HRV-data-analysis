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

import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from numpy.testing import assert_allclose
import matplotlib.dates as mdates

from HRV_calculation import (
    calculate_HRV_in_windows,
    calculate_mean_HRV_based_on_windows,
    filter_windows_with_chunked_dataframe,
    prepare_windows_any_frequency_any_step,
    RMSSD_HRV_calculation
)


class Test(unittest.TestCase):
    def test_RMSSD_HRV_calculation(self):
        time = pd.to_datetime([
            "2021-12-01 11:00:12",
            "2021-12-01 11:00:13",
            "2021-12-01 11:00:14",
            "2021-12-01 11:00:15.3"]
        )
        rr_intervals_1 = pd.Series(
            [650, 700, 675, 730],
            index=time)
        result_1 = RMSSD_HRV_calculation(rr_intervals_1)
        gt_1 = 45.276925691
        self.assertAlmostEqual(result_1, gt_1)

        time = pd.to_datetime([
            "2021-12-01 11:00:12",
            "2021-12-01 11:00:13",
            "2021-12-01 11:00:14",
            "2021-12-01 11:00:15",
            "2021-12-01 11:00:16"]
        )
        rr_intervals_2 = pd.Series(
            [600, 675, 525, 750, 800],
            index=time)
        result_2 = RMSSD_HRV_calculation(rr_intervals_2)
        gt_2 = 142.521928137
        self.assertAlmostEqual(result_2, gt_2)

        time = pd.to_datetime([
            "2021-12-01 11:00:12.00",
            "2021-12-01 11:00:13.00",
            "2021-12-01 11:00:14.00",
            "2021-12-01 11:00:15.30",
            "2021-12-01 11:00:17.40",
            "2021-12-01 11:00:18.00",
            "2021-12-01 11:00:19.20",
            "2021-12-01 11:00:19.85",
            "2021-12-01 11:00:21.00",
            "2021-12-01 11:00:23.50",
            "2021-12-01 11:00:25.00",
            "2021-12-01 11:00:28.00",
            "2021-12-01 11:00:29.00",
            "2021-12-01 11:00:30.50"]
        )
        values = [600, 675, 525, 750, 800,
                  610, 680, 540, 545, 740,
                  630, 690, 575, 550]
        # elements: 0, 1, 2, 3, 5, 6, 7, 8, 10, 12, 13
        rr_intervals_3 = pd.Series(
            values,
            index=time
        )
        differences = [75, -150, 225, -190, 70,
                       -140, 5, -110, -115, -25]
        result_3 = RMSSD_HRV_calculation(rr_intervals_3)
        gt_3 = 128.578769631693
        self.assertAlmostEqual(result_3, gt_3)

    def test_prepare_windows_any_frequency_any_step(self):
        time_index = pd.to_datetime(
            ["2021-12-01 11:00:00",
             "2021-12-01 11:00:12",
             "2021-12-01 11:01:30",
             "2021-12-01 11:02:00",
             "2021-12-01 11:02:35",
             "2021-12-01 11:03:01",
             "2021-12-01 11:03:48",
             "2021-12-01 11:04:30",
             "2021-12-01 11:07:21",
             "2021-12-01 11:10:33",
             "2021-12-01 11:10:44",
             "2021-12-01 11:13:27",
             "2021-12-01 11:16:00",
             "2021-12-01 11:16:08",
             "2021-12-01 11:17:03"]
        )
        values = [530, 550, 520, 780, 800,
                  610, 678, 540, 542, 748,
                  632, 658, 678, 720, 770]
        series = pd.Series(values,
                           index=time_index)
        # ##### UNITTEST 1) #####
        freq_1 = pd.Timedelta('1 min')
        win_size_1 = pd.Timedelta('2 min')
        result_1 = prepare_windows_any_frequency_any_step(
            series,
            freq_1,
            win_size_1
        )
        gt_result_1 = [
            pd.Series(
                [530, 550, 520, 780],
                index=pd.to_datetime(
                    ["2021-12-01 11:00:00",
                     "2021-12-01 11:00:12",
                     "2021-12-01 11:01:30",
                     "2021-12-01 11:02:00"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [520, 780, 800],
                index=pd.to_datetime(
                    ["2021-12-01 11:01:30",
                     "2021-12-01 11:02:00",
                     "2021-12-01 11:02:35"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [780, 800, 610, 678],
                index=pd.to_datetime(
                    ["2021-12-01 11:02:00",
                     "2021-12-01 11:02:35",
                     "2021-12-01 11:03:01",
                     "2021-12-01 11:03:48"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [610, 678, 540],
                index=pd.to_datetime(
                    ["2021-12-01 11:03:01",
                     "2021-12-01 11:03:48",
                     "2021-12-01 11:04:30"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [540],
                index=pd.to_datetime(
                    ["2021-12-01 11:04:30"]
                ),
                dtype=np.int64
            ),
            pd.Series([], dtype=np.int64),
            pd.Series(
                [542],
                index=pd.to_datetime(
                    ["2021-12-01 11:07:21"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [542],
                index=pd.to_datetime(
                    ["2021-12-01 11:07:21"]
                ),
                dtype=np.int64
            ),
            pd.Series([], dtype=np.int64),
            pd.Series(
                [748, 632],
                index=pd.to_datetime(
                    ["2021-12-01 11:10:33",
                     "2021-12-01 11:10:44"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [748, 632],
                index=pd.to_datetime(
                    ["2021-12-01 11:10:33",
                     "2021-12-01 11:10:44"]
                ),
                dtype=np.int64
            ),
            pd.Series([], dtype=np.int64),
            pd.Series(
                [658],
                index=pd.to_datetime(
                    ["2021-12-01 11:13:27"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [658],
                index=pd.to_datetime(
                    ["2021-12-01 11:13:27"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [678],
                index=pd.to_datetime(
                    ["2021-12-01 11:16:00"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [678, 720],
                index=pd.to_datetime(
                    ["2021-12-01 11:16:00",
                     "2021-12-01 11:16:08"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [678, 720, 770],
                index=pd.to_datetime(
                    ["2021-12-01 11:16:00",
                     "2021-12-01 11:16:08",
                     "2021-12-01 11:17:03"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [770],
                index=pd.to_datetime(
                     ["2021-12-01 11:17:03"]
                ),
                dtype=np.int64
            )
        ]
        for i in range(len(gt_result_1)):
            assert_series_equal(
                result_1[i],
                gt_result_1[i],
                check_index_type=False,
                check_freq=False
            )

        # ##### UNITTEST 2) #####
        freq_2 = pd.Timedelta('2 min')
        win_size_2 = pd.Timedelta('2 min')
        result_2 = prepare_windows_any_frequency_any_step(
            series,
            freq_2,
            win_size_2
        )
        gt_result_2 = [
            pd.Series(
                [530, 550, 520, 780],
                index=pd.to_datetime(
                    ["2021-12-01 11:00:00",
                     "2021-12-01 11:00:12",
                     "2021-12-01 11:01:30",
                     "2021-12-01 11:02:00"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [780, 800, 610, 678],
                index=pd.to_datetime(
                    ["2021-12-01 11:02:00",
                     "2021-12-01 11:02:35",
                     "2021-12-01 11:03:01",
                     "2021-12-01 11:03:48"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [540],
                index=pd.to_datetime(
                    ["2021-12-01 11:04:30"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [542],
                index=pd.to_datetime(
                    ["2021-12-01 11:07:21"]
                ),
                dtype=np.int64
            ),
            pd.Series([], dtype=np.int64),
            pd.Series(
                [748, 632],
                index=pd.to_datetime(
                    ["2021-12-01 11:10:33",
                     "2021-12-01 11:10:44"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [658], 
                index=pd.to_datetime(
                    ["2021-12-01 11:13:27"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [678], 
                index=pd.to_datetime(
                    ["2021-12-01 11:16:00"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [678, 720, 770],
                index=pd.to_datetime(
                    ["2021-12-01 11:16:00",
                     "2021-12-01 11:16:08",
                     "2021-12-01 11:17:03"]
                ),
                dtype=np.int64
            )
        ]
        for i in range(len(gt_result_2)):
            assert_series_equal(
                result_2[i],
                gt_result_2[i],
                check_index_type=False,
                check_freq=False
            )

        # ##### UNITTEST 3 #####
        freq_3 = pd.Timedelta('3 min')
        win_size_3 = pd.Timedelta('2 min')
        with self.assertRaises(AssertionError):
            prepare_windows_any_frequency_any_step(
                series,
                freq_3,
                win_size_3
            )

    def test_calculate_mean_HRV_on_windows(self):
        test_series_1 = pd.Series(
            ['treatment', 1,
             [2.20, 1.15, 0.0, 0, 2, 3, 7, 0.0],
             [np.datetime64('2022-04-21T10:00:00'),
              np.datetime64('2022-04-21T10:12:00'),
              np.datetime64('2022-04-21T10:24:00'),
              np.datetime64('2022-04-21T10:31:00'),
              np.datetime64('2022-04-21T10:38:00'),
              np.datetime64('2022-04-21T10:55:00'),
              np.datetime64('2022-04-21T11:02:00'),
              np.datetime64('2022-04-21T11:08:00')]],
            index=['group', 'no_of_person', 'HRV_RMSSD', 'timestamps']
        )
        result_series_1 = calculate_mean_HRV_based_on_windows(
            test_series_1, 'RMSSD'
        )
        gt_series_1 = pd.Series(
            ['treatment', 1,
             3.07, [
                np.datetime64('2022-04-21T10:00:00'),
                np.datetime64('2022-04-21T10:12:00'),
                np.datetime64('2022-04-21T10:38:00'),
                np.datetime64('2022-04-21T10:55:00'),
                np.datetime64('2022-04-21T11:02:00'),
             ]],
            index=['group', 'no_of_person', 'HRV_RMSSD', 'timestamps']
        )
        assert_series_equal(result_series_1, gt_series_1)

        test_series_2 = pd.Series(
            ['control', 4,
             [3, 5, 8, 4, 5],
             [np.datetime64('2022-04-21T11:00:00'),
              np.datetime64('2022-04-21T11:12:00'),
              np.datetime64('2022-04-21T11:24:00'),
              np.datetime64('2022-04-21T11:31:00'),
              np.datetime64('2022-04-21T11:38:00')]],
            index=['group', 'no_of_person', 'HRV_RMSSD', 'timestamps']
        )
        result_series_2 = calculate_mean_HRV_based_on_windows(
            test_series_2, 'RMSSD'
        )
        gt_series_2 = pd.Series(
            ['control', 4,
             5, [
                np.datetime64('2022-04-21T11:00:00'),
                np.datetime64('2022-04-21T11:12:00'),
                np.datetime64('2022-04-21T11:24:00'),
                np.datetime64('2022-04-21T11:31:00'),
                np.datetime64('2022-04-21T11:38:00')]],
            index=['group', 'no_of_person', 'HRV_RMSSD', 'timestamps']
        )
        assert_series_equal(result_series_2, gt_series_2)

    def test_calculate_HRV_in_windows(self):
        def calculate_median_timestamp_based_on_pydatetimes(times):
            return pd.Timestamp(mdates.num2date(
                np.median(mdates.date2num(times))))

        ts = [
            "2021-12-01 11:00:00.00",
            "2021-12-01 11:00:01.90",
            "2021-12-01 11:00:03.80",
            "2021-12-01 11:00:05.78",
            "2021-12-01 11:00:06.30",
            "2021-12-01 11:00:06.98",
            "2021-12-01 11:00:08.80",
            "2021-12-01 11:00:09.20",
            "2021-12-01 11:00:10.90",
            "2021-12-01 11:00:12.10",
            "2021-12-01 11:00:12.90",
            "2021-12-01 11:00:14.80",
            "2021-12-01 11:00:15.20",
            "2021-12-01 11:00:17.01",
            "2021-12-01 11:00:18.80"]

        values = [530, 550, 520, 780, 800,
                  610, 678, 540, 542, 748,
                  632, 658, 678, 720, 770]
        dataframe = pd.DataFrame(
            {'Phone timestamp': pd.to_datetime(ts),
             'RR-interval [ms]': values}
        )
        step_frequency = '2 seconds'
        window_size = '5 seconds'

        # UNITTEST 1)
        # Consecutive time windows:
        # 0-5 seconds, 2-7 sec., 4-9 sec.,
        # 6-11 sec., 8-13 sec., 10-15 sec.,
        # 12-17 sec., 14-19 sec., 16-21 sec.;
        # 18-23 sec should have only one element, therefore it is not counted
        gt_HRV_divided_series = np.zeros(9)
        gt_median_timestamps = np.zeros(9, dtype='datetime64[ns]')
        beginning_index = [0, 2, 3, 4, 6, 8, 9, 11, 13]
        end_index = [3, 6, 7, 9, 11, 12, 13, 15, 17]
        for index, (start, end) in enumerate(zip(beginning_index, end_index)):
            gt_median_timestamps[index] = calculate_median_timestamp_based_on_pydatetimes(
                ts[start:end])
            series = dataframe.iloc[start:end]['RR-interval [ms]']
            series = series.set_axis(pd.to_datetime(ts[start:end]))
            gt_HRV_divided_series[index] = RMSSD_HRV_calculation(
                series)
        test_HRV_divided_series, test_median_timestamps = calculate_HRV_in_windows(
            dataframe, step_frequency, window_size, 'RMSSD'
        )
        assert_allclose(test_HRV_divided_series,
                        gt_HRV_divided_series)
        pd.to_datetime(test_median_timestamps).equals(
            pd.to_datetime(gt_median_timestamps))

    def test_filter_windows_with_chunked_dataframe(self):
        input_test_1 = [
            pd.Series(
                [450, 510, 550, 700],
                index=pd.to_datetime(
                    ["2021-12-02 09:59:01",
                     "2021-12-02 09:59:22",
                     "2021-12-02 09:59:30",
                     "2021-12-02 09:59:37"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [530, 550, 520, 780],
                index=pd.to_datetime(
                    ["2021-12-02 10:00:00",
                     "2021-12-02 10:00:12",
                     "2021-12-02 10:01:30",
                     "2021-12-02 10:02:00"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [780, 800, 610, 678, 550],
                index=pd.to_datetime(
                    ["2021-12-02 10:00:00",
                     "2021-12-02 10:00:12",
                     "2021-12-02 10:01:30",
                     "2021-12-02 10:02:00",
                     "2021-12-02 10:03:15"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [780, 800, 610, 678, 550],
                index=pd.to_datetime(
                    ["2021-12-02 10:01:57",
                     "2021-12-02 10:02:12",
                     "2021-12-02 10:02:30",
                     "2021-12-02 10:03:00",
                     "2021-12-02 10:04:15"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [790, 800, 610, 678, 550],
                index=pd.to_datetime(
                    ["2021-12-02 10:01:57",
                     "2021-12-02 10:02:12",
                     "2021-12-02 10:02:30",
                     "2021-12-02 10:03:00",
                     "2021-12-02 10:04:15"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [790, 800, 610, 678],
                index=pd.to_datetime(
                    ["2021-12-02 10:01:57",
                     "2021-12-02 10:02:12",
                     "2021-12-02 10:02:30",
                     "2021-12-02 10:03:00"]
                ),
                dtype=np.int64
            )
        ]
        result_input_1 = filter_windows_with_chunked_dataframe(
            divided_series=input_test_1)
        gt_result_1 = [
            pd.Series(
                [450, 510, 550, 700],
                index=pd.to_datetime(
                    ["2021-12-02 09:59:01",
                     "2021-12-02 09:59:22",
                     "2021-12-02 09:59:30",
                     "2021-12-02 09:59:37"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [530, 550, 520, 780],
                index=pd.to_datetime(
                    ["2021-12-02 10:00:00",
                     "2021-12-02 10:00:12",
                     "2021-12-02 10:01:30",
                     "2021-12-02 10:02:00"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [780, 800, 610, 678, 550],
                index=pd.to_datetime(
                    ["2021-12-02 10:00:00",
                     "2021-12-02 10:00:12",
                     "2021-12-02 10:01:30",
                     "2021-12-02 10:02:00",
                     "2021-12-02 10:03:15"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [790, 800, 610, 678, 550],
                index=pd.to_datetime(
                    ["2021-12-02 10:01:57",
                     "2021-12-02 10:02:12",
                     "2021-12-02 10:02:30",
                     "2021-12-02 10:03:00",
                     "2021-12-02 10:04:15"]
                ),
                dtype=np.int64
            ),
            pd.Series(
                [790, 800, 610, 678],
                index=pd.to_datetime(
                    ["2021-12-02 10:01:57",
                     "2021-12-02 10:02:12",
                     "2021-12-02 10:02:30",
                     "2021-12-02 10:03:00"]
                ),
                dtype=np.int64
            )
        ]
        for i in range(len(gt_result_1)):
            assert_series_equal(
                result_input_1[i],
                gt_result_1[i],
            )


if __name__ == "__main__":
    print('Run tests from the external path.')
