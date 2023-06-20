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
import matplotlib.pyplot as plt

from pandas.testing import assert_frame_equal
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal
)
from utils_preprocessing import (
    convert_absolute_time_to_timestamps_from_given_timestamp,
    interpolate_data_with_splines,
    remove_preceding_and_following_beat,
    remove_consecutive_beats_after_holes,
    remove_adjacent_beats,
    remove_first_and_last_indices,
    remove_negative_timestamps,
    remove_selected_time_ranges,
    return_hour_from_datetime
)


class Test(unittest.TestCase):
    def test_convert_absolute_time_to_timestamps_from_given_timestamp(self):
        """
        Unittest for function
        convert_absolute_time_to_timestamps_from_given_timestamp().
        """
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:40:00'),
                    pd.Timestamp('2022-02-22 08:40:40'),
                    pd.Timestamp('2022-02-22 09:00:10'),
                    pd.Timestamp('2022-02-22 09:05:30')
                ],
            'RR intervals': [
                750, 800, 730, 550, 1000
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        initial_timestamp = pd.Timestamp('2022-02-21 15:00:00')
        output_dataframe = convert_absolute_time_to_timestamps_from_given_timestamp(
            input_dataframe,
            initial_timestamp=initial_timestamp
        )
        output_dict = {
         'Phone timestamp': [
                 pd.Timestamp('2022-02-21 15:00:00'),
                 pd.Timestamp('2022-02-21 15:10:00'),
                 pd.Timestamp('2022-02-21 15:10:40'),
                 pd.Timestamp('2022-02-21 15:30:10'),
                 pd.Timestamp('2022-02-21 15:35:30')
            ],
         'RR intervals': [
                750, 800, 730, 550, 1000
            ]
        }
        gt_dataframe = pd.DataFrame.from_dict(output_dict)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

    def test_remove_preceding_and_following_beat(self):
        # Unittest 1)
        filtered_indices = [0, 3, 5, 27, 28, 99]
        length = 100
        gt_with_precedings_and_followings = np.array([
            0, 1, 2, 3, 4, 5, 6, 26, 27, 28, 29, 98, 99
        ])
        returned_with_precs_and_follows = remove_preceding_and_following_beat(
            filtered_indices, length
        )
        assert_array_equal(gt_with_precedings_and_followings,
                           returned_with_precs_and_follows)
        # Unittest 2)
        filtered_indices = [1, 2, 3, 35, 36, 38, 40, 42, 48]
        length = 50
        gt_with_precedings_and_followings = np.array([
            0, 1, 2, 3, 4,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            47, 48, 49
        ])
        returned_with_precs_and_follows = remove_preceding_and_following_beat(
            filtered_indices, length
        )
        assert_array_equal(gt_with_precedings_and_followings,
                           returned_with_precs_and_follows)

    def test_remove_adjacent_beats(self):
        # Unittest 1)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:07'),
                    pd.Timestamp('2022-02-22 08:30:08'),
                    pd.Timestamp('2022-02-22 08:30:11'),
                    pd.Timestamp('2022-02-22 08:30:15'),
                    pd.Timestamp('2022-02-22 08:30:27'),
                    pd.Timestamp('2022-02-22 08:30:28'),
                    pd.Timestamp('2022-02-22 08:30:38'),
                    pd.Timestamp('2022-02-22 08:30:42'),
                    pd.Timestamp('2022-02-22 08:30:51'),
                ],
            'RR intervals': [
                750, 800, 600, 780, 820, 810,
                740, 710, 610, 680, 775
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        indices = np.array([2, 3, 9])
        output_dataframe = remove_adjacent_beats(
            input_dataframe, indices, time='5 seconds'
        )
        gt_data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:15'),
                    pd.Timestamp('2022-02-22 08:30:27'),
                    pd.Timestamp('2022-02-22 08:30:28'),
                    pd.Timestamp('2022-02-22 08:30:51'),
                ],
            'RR intervals': [
                750, 810, 740, 710, 775
            ],
        }
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index([[0, 5, 6, 7, 10]], inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

        # Unittest 2)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:29:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:07'),
                    pd.Timestamp('2022-02-22 08:30:08'),
                    pd.Timestamp('2022-02-22 08:30:14'),
                    pd.Timestamp('2022-02-22 08:30:15'),
                    pd.Timestamp('2022-02-22 08:30:27'),
                    pd.Timestamp('2022-02-22 08:30:28'),
                    pd.Timestamp('2022-02-22 08:30:38'),
                    pd.Timestamp('2022-02-22 08:30:42'),
                    pd.Timestamp('2022-02-22 08:30:51'),
                ],
            'RR intervals': [
                750, 800, 600, 780, 820, 810,
                740, 710, 610, 680, 775
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        input_dataframe.set_index([[3, 4, 5, 6, 21,
                                    22, 24, 27, 29, 31, 40]],
                                    inplace=True)
        indices = np.array([4, 21, 39])
        output_dataframe = remove_adjacent_beats(
            input_dataframe, indices, time='5 seconds'
        )
        gt_data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:29:00'),
                    pd.Timestamp('2022-02-22 08:30:08'),
                    pd.Timestamp('2022-02-22 08:30:27'),
                    pd.Timestamp('2022-02-22 08:30:28'),
                    pd.Timestamp('2022-02-22 08:30:38'),
                    pd.Timestamp('2022-02-22 08:30:42'),
                    pd.Timestamp('2022-02-22 08:30:51'),
                ],
            'RR intervals': [
                750, 780, 740, 710,
                610, 680, 775
            ],
        }
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index([[3, 6, 24, 27, 29, 31, 40]], inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )


    def test_remove_consecutive_beats_after_holes(self):
        # Unittest 1)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:33'),
                    pd.Timestamp('2022-02-22 08:30:35'),
                    pd.Timestamp('2022-02-22 08:30:47.9'),
                    pd.Timestamp('2022-02-22 08:31:00'),
                    pd.Timestamp('2022-02-22 08:31:28'),
                    pd.Timestamp('2022-02-22 08:32:00'),
                    pd.Timestamp('2022-02-22 08:32:31'),
                    pd.Timestamp('2022-02-22 08:32:34'),
                    pd.Timestamp('2022-02-22 08:32:35'),
                ],
            'RR intervals': [
                750, 800, 600, 780, 820, 810,
                740, 710, 610, 680, 775
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        output_dataframe = remove_consecutive_beats_after_holes(
            input_dataframe,
            hole_time='30 seconds',
            window_time='15 seconds'
        )
        gt_data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:31:00'),
                    pd.Timestamp('2022-02-22 08:31:28'),
                ],
            'RR intervals': [
                750, 800, 810, 740
            ],
        }
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index([[0, 1, 5, 6]], inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

        # Unittest 2)
        input_dataframe = pd.DataFrame.from_dict(data)
        input_dataframe.set_index([[2, 3, 5, 6, 8, 10,
                                    12, 14, 18, 22, 29]],
                                  inplace=True)
        output_dataframe = remove_consecutive_beats_after_holes(
            input_dataframe,
            hole_time='30 seconds',
            window_time='15 seconds'
        )
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index([[2, 3, 10, 12]], inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

    def test_remove_first_and_last_indices(self):
        # Unittest 1)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:44.9'),
                    pd.Timestamp('2022-02-22 08:30:50'),
                    pd.Timestamp('2022-02-22 08:30:55'),
                    pd.Timestamp('2022-02-22 08:32:00'),
                    pd.Timestamp('2022-02-22 08:33:15'),
                    pd.Timestamp('2022-02-22 08:32:28'),
                ],
            'RR intervals': [
                750, 800, 600, 780,
                820, 810, 740, 710,
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        output_dataframe = remove_first_and_last_indices(
            input_dataframe,
            initial_cut_window='45 seconds',
            end_cut_window='30 seconds'
        )
        gt_data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:50'),
                    pd.Timestamp('2022-02-22 08:30:55'),
                ],
            'RR intervals': [
                780, 820
            ],
        }
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index([[3, 4]], inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

        # Unittest 2)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:44.9'),
                    pd.Timestamp('2022-02-22 08:30:50'),
                    pd.Timestamp('2022-02-22 08:30:55'),
                    pd.Timestamp('2022-02-22 08:32:00'),
                    pd.Timestamp('2022-02-22 08:33:15'),
                    pd.Timestamp('2022-02-22 08:33:28'),
                ],
            'RR intervals': [
                750, 800, 600, 780,
                820, 810, 740, 710,
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        input_dataframe.set_index([[3, 4, 5, 8,
                                    9, 10, 11, 12]],
                                    inplace=True)
        output_dataframe = remove_first_and_last_indices(
            input_dataframe,
            initial_cut_window='40 seconds',
            end_cut_window='10 seconds'
        )
        gt_data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:44.9'),
                    pd.Timestamp('2022-02-22 08:30:50'),
                    pd.Timestamp('2022-02-22 08:30:55'),
                    pd.Timestamp('2022-02-22 08:32:00'),
                    pd.Timestamp('2022-02-22 08:33:15'),
                ],
            'RR intervals': [
                600, 780, 820, 810, 740,
            ]
        }
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index([[5, 8, 9, 10, 11]], inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

    def test_remove_negative_timestamps(self):
        # Unittest 1)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:50'),
                    pd.Timestamp('2022-02-22 08:30:35'),
                    pd.Timestamp('2022-02-22 08:30:47'),
                    pd.Timestamp('2022-02-22 08:30:38'),
                    pd.Timestamp('2022-02-22 08:30:39'),
                    pd.Timestamp('2022-02-22 08:30:41'),
                    pd.Timestamp('2022-02-22 08:30:49'),
                    pd.Timestamp('2022-02-22 08:31:00'),
                    pd.Timestamp('2022-02-22 08:31:10'),
                    pd.Timestamp('2022-02-22 08:31:15'),
                    pd.Timestamp('2022-02-22 08:31:12'),
                    pd.Timestamp('2022-02-22 08:31:14'),
                    pd.Timestamp('2022-02-22 08:31:20'),
                    pd.Timestamp('2022-02-22 08:31:22'),
                    pd.Timestamp('2022-02-22 08:31:25'),
                ],
            'RR intervals': [
                750, 800, 600, 780, 800,
                820, 810, 740, 710, 750,
                840, 765, 780, 795, 810,
                820, 724
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        input_dataframe.set_index([[3, 4, 5, 8, 9,
                                    10, 11, 12, 13, 14,
                                    16, 17, 19, 20, 21,
                                    22, 24]],
                                   inplace=True)
        output_dataframe = remove_negative_timestamps(
            input_dataframe
        )
        gt_data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:31:10'),
                    pd.Timestamp('2022-02-22 08:31:22'),
                    pd.Timestamp('2022-02-22 08:31:25'),
                ],
            'RR intervals': [
                750, 800, 840,
                820, 724
            ],
        }
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index(
            [[3, 4, 16, 22, 24]],
            inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )
        # Unittest 2)
        input_dataframe = input_dataframe.reset_index(drop=True)
        output_dataframe = remove_negative_timestamps(
            input_dataframe
        )
        gt_dataframe.set_index(
            [[0, 1, 10, 15, 16]],
            inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

        # Unittest 3)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:50'),
                    pd.Timestamp('2022-02-22 08:30:55'),
                    pd.Timestamp('2022-02-22 08:30:58'),
                    pd.Timestamp('2022-02-22 08:31:15'),
                    pd.Timestamp('2022-02-22 08:31:17'),
                    pd.Timestamp('2022-02-22 08:32:14'),
                ],
            'RR intervals': [
                750, 800, 600, 780,
                840, 765, 780, 795,
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        output_dataframe = remove_negative_timestamps(
            input_dataframe
        )
        gt_dataframe = input_dataframe.copy()
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

    def test_remove_selected_time_ranges(self):
        # Unittest 1)
        data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:02'),
                    pd.Timestamp('2022-02-22 08:30:50.1'),
                    pd.Timestamp('2022-02-22 08:30:55'),
                    pd.Timestamp('2022-02-22 08:30:58'),
                    pd.Timestamp('2022-02-22 08:31:15'),
                    pd.Timestamp('2022-02-22 08:31:17'),
                    pd.Timestamp('2022-02-22 08:32:14'),
                    pd.Timestamp('2022-02-22 08:32:24'),
                    pd.Timestamp('2022-02-22 08:32:34'),
                ],
            'RR intervals': [
                750, 800, 600, 780, 800,
                840, 765, 780, 795, 750
            ]
        }
        input_dataframe = pd.DataFrame.from_dict(data)
        timeranges_to_remove = [
            [pd.Timestamp('2022-02-22 08:30:02'),
             pd.Timestamp('2022-02-22 08:30:50')],
            [pd.Timestamp('2022-02-22 08:30:58'),
             pd.Timestamp('2022-02-22 08:31:15')],
            [pd.Timestamp('2022-02-22 08:32:20'),
             pd.Timestamp('2022-02-22 08:33:02')]
        ]
        output_dataframe = remove_selected_time_ranges(
            input_dataframe, timeranges_to_remove
        )
        gt_data = {
            'Phone timestamp': [
                    pd.Timestamp('2022-02-22 08:30:00'),
                    pd.Timestamp('2022-02-22 08:30:50.1'),
                    pd.Timestamp('2022-02-22 08:30:55'),
                    pd.Timestamp('2022-02-22 08:31:17'),
                    pd.Timestamp('2022-02-22 08:32:14')
                ],
            'RR intervals': [
                750, 600, 780, 765, 780
            ],
        }
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index(
            [[0, 2, 3, 6, 7]],
            inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

        # Unittest 2)
        input_dataframe = pd.DataFrame.from_dict(data)
        input_dataframe.set_index([[3, 4, 5, 8, 9,
                                    16, 18, 21, 22, 28]],
                                   inplace=True)
        output_dataframe = remove_selected_time_ranges(
            input_dataframe, timeranges_to_remove
        )
        gt_dataframe = pd.DataFrame.from_dict(gt_data)
        gt_dataframe.set_index(
            [[3, 5, 8, 18, 21]],
            inplace=True)
        self.assertIsNone(
            assert_frame_equal(gt_dataframe, output_dataframe)
        )

    def test_return_hour_from_datetime(self):
        datetime_1 = pd.Timestamp('2022-11-05 05:30:51')
        output_datetime_1 = return_hour_from_datetime(datetime_1)
        gt_datetime_1 = '05:30:51.000000'
        self.assertEqual(output_datetime_1, gt_datetime_1)
        datetime_2 = np.datetime64('2022-11-05T05:30:51.238')
        output_datetime_2 = return_hour_from_datetime(datetime_2)
        gt_datetime_2 = '05:30:51.238'
        self.assertEqual(output_datetime_2, gt_datetime_2)

    def test_interpolate_data_with_splines(self):
        def spline_function_for_testing(x):
            # Source: https://people.clas.ufl.edu/kees/files/CubicSplines.pdf
            if 0 <= x <= 1:
                return -12 / 11 * x + 23 / 11 * x**3
            if 1 < x < 2:
                return 1 + 57 / 11 * (x - 1) + 69 / 11 * (x - 1)**2 - \
                       49 / 11 * (x - 1)**3
            if 2 <= x <= 2.5:
                return 8 + 48 / 11 * (x - 2) - 78 / 11 * (x - 2)**2 + \
                       52 / 11 * (x - 2)**3
            else:
                raise ValueError('Wrong argument!')
        column_name = 'values'
        # Unittest 1)
        x_values = np.arange(0, 10.01, 0.25)
        y_values = list(np.cos(-x_values ** 2 / 9.0))
        data = {
            'Phone timestamp':
                list(x_values),
            'values':
                y_values
        }
        original_dataframe = pd.DataFrame.from_dict(data)
        data = {
            'Phone timestamp':
                list(np.arange(0, 11)),
            'values': [
                y_values[4 * i] for i in range(0, 11)
                ]
        }
        current_dataframe = pd.DataFrame.from_dict(data)

        gt_timestamps = np.array([
            0.,  0.75,  1.,  1.25,  1.5,  1.75,  2.,
            2.25,  2.5,  2.75,  3.,  3.25,  3.5,  3.75,  4., 
            4.25, 4.5,  4.75,  5.,  5.75,  6., 
            6.25,  6.5, 6.75,  7.,  7.25,  7.5,  7.75,  8., 
            8.25,  8.5,  8.75, 9.,  9.25, 9.5, 9.75, 10.])
        gt_predicted_timestamps = np.array([
            0.25,  0.5,  0.75,  1.25,  1.5,  1.75,
            2.25,  2.5,  2.75,  3.25,  3.5,  3.75,
            4.25,  4.5,  4.75,  5.25,  5.5,  5.75,
            6.25,  6.5,  6.75,  7.25,  7.5,  7.75,
            8.25,  8.5,  8.75,  9.25,  9.5,  9.75
        ])

        modified_dataframe, predictions, extreme_values = interpolate_data_with_splines(
            original_dataframe,
            current_dataframe,
            column_name
        )
        assert_array_almost_equal(gt_timestamps,
                                  modified_dataframe['Phone timestamp'].values)
        assert_array_almost_equal(gt_predicted_timestamps,
                                  extreme_values)

        # Unittest 2)
        data = {
            'Phone timestamp': [
                1, 1.5, 2, 3, 4
            ],
            'values': [
                1.0, 0.0, 5.0, 11.0, 8.0
            ]
        }
        original_dataframe = pd.DataFrame.from_dict(data)
        data = {
            'Phone timestamp': [
                1, 2, 3, 4
            ],
            'values': [
               1.0, 2.0, 5.0, 11.0
            ]
        }
        current_dataframe = pd.DataFrame.from_dict(data)
        modified_dataframe, predictions, extreme_values = interpolate_data_with_splines(
            original_dataframe,
            current_dataframe,
            column_name
        )
        gt_timestamps = np.array([
            1.0, 1.5, 2.0, 3.0, 4.0
        ])
        gt_values = np.array([
            1.0, 1.375, 2.0, 5.0, 11.0
        ])
        gt_predicted_timestamps = np.array([1.5])
        assert_array_almost_equal(gt_timestamps,
                                  modified_dataframe['Phone timestamp'].values)
        assert_array_almost_equal(gt_predicted_timestamps,
                                  extreme_values)
        assert_array_almost_equal(gt_values,
                                  modified_dataframe['values'].values)

        # Unittest 3)
        gt_x = np.arange(0, 2.5, 0.01)
        gt_y = np.zeros_like(gt_x)
        for i in range(gt_x.shape[0]):
            gt_y[i] = spline_function_for_testing(gt_x[i])

        # (0, 0), (1, 1), (2, 8), (5/2, 9)
        x_values = np.arange(0, 2.51, 0.01)
        y_values = list(np.zeros_like(x_values))
        data = {
            'Phone timestamp':
                list(x_values),
            'values':
                y_values
        }
        original_dataframe = pd.DataFrame.from_dict(data)
        data = {
            'Phone timestamp':
                [0, 1, 2, 2.5],
            'values': [
                0., 1., 8., 9.
                ]
        }
        current_dataframe = pd.DataFrame.from_dict(data)
        modified_dataframe, predictions, extreme_values = interpolate_data_with_splines(
            original_dataframe,
            current_dataframe,
            column_name
        )
        assert_array_almost_equal(gt_y[73:],
                                  modified_dataframe['values'].values[1:-1])
        # Previous values were removed due to the lower values than
        # the minimum in the current dataframe
        plt.plot(gt_x[73:], gt_y[73:], color='red', label='GT')
        plt.plot(modified_dataframe['Phone timestamp'].values[1:-1],
                 modified_dataframe['values'].values[1:-1],
                 label='interpolation',
                 alpha=0.5)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    print('Run tests from the external path.')
