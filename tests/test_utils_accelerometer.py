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

from utils_accelerometer import (
    clean_accelerometer_data_and_fill_according_to_HRV,
)
from pandas.testing import assert_frame_equal


class Test(unittest.TestCase):
    def test_clean_accelerometer_data_and_fill_according_to_HRV(self):
        index_ACC = [
            pd.Timestamp('2022-02-22 08:30:00'),
            pd.Timestamp('2022-02-22 08:30:28'),
            pd.Timestamp('2022-02-22 08:30:31'),
            pd.Timestamp('2022-02-22 08:30:35'),
            pd.Timestamp('2022-02-22 08:30:35'),
            pd.Timestamp('2022-02-22 08:30:42'),
            pd.Timestamp('2022-02-22 08:30:58'),
            pd.Timestamp('2022-02-22 08:31:02'),
            pd.Timestamp('2022-02-22 08:31:08'),
            pd.Timestamp('2022-02-22 08:31:21'),
            pd.Timestamp('2022-02-22 08:31:21'),
            pd.Timestamp('2022-02-22 08:31:35'),
            pd.Timestamp('2022-02-22 08:31:57'),
            pd.Timestamp('2022-02-22 08:32:00')
        ]
        data_ACC = [
            151.8, 178, 250, 350.5, 400,
            274.3, 475, 600, 217, 312.5,
            300, 280, 260.5, 190
        ]
        ACC_input_dataframe = pd.DataFrame(data_ACC,
                                           index=index_ACC,
                                           columns=['mg'])
        index_HRV = [
            pd.Timestamp('2022-02-22 08:30:31'),
            pd.Timestamp('2022-02-22 08:30:42'),
            pd.Timestamp('2022-02-22 08:30:58'),
            pd.Timestamp('2022-02-22 08:31:08'),
            pd.Timestamp('2022-02-22 08:31:21'),
            pd.Timestamp('2022-02-22 08:31:35'),
            pd.Timestamp('2022-02-22 08:31:57'),
        ]
        data_HRV = [
            10.5, 10, 9.5, 9.75, 11,
            8.8, 9.2
        ]
        HRV_input_dataframe = pd.DataFrame(data_HRV,
                                           index=index_HRV,
                                           columns=['HRV'])
        gt_index = [
            pd.Timestamp('2022-02-22 08:30:31'),
            pd.Timestamp('2022-02-22 08:30:35'),
            pd.Timestamp('2022-02-22 08:30:42'),
            pd.Timestamp('2022-02-22 08:30:58'),
            pd.Timestamp('2022-02-22 08:31:02'),
            pd.Timestamp('2022-02-22 08:31:08'),
            pd.Timestamp('2022-02-22 08:31:21'),
            pd.Timestamp('2022-02-22 08:31:35'),
            pd.Timestamp('2022-02-22 08:31:57'),
            pd.Timestamp('2022-02-22 08:32:00')
        ]
        gt_HRV = [
            10.5, np.nan, 10, 9.5, np.nan,
            9.75, 11, 8.8, 9.2, np.nan
        ]
        initial_GT_HRV = pd.DataFrame(gt_HRV,
                                      index=gt_index,
                                      columns=['HRV'])
        dataframe_GT_HRV = initial_GT_HRV.interpolate(method='linear')
        GT_index_ACC = [
            pd.Timestamp('2022-02-22 08:30:31'),
            pd.Timestamp('2022-02-22 08:30:35'),
            pd.Timestamp('2022-02-22 08:30:42'),
            pd.Timestamp('2022-02-22 08:30:58'),
            pd.Timestamp('2022-02-22 08:31:02'),
            pd.Timestamp('2022-02-22 08:31:08'),
            pd.Timestamp('2022-02-22 08:31:21'),
            pd.Timestamp('2022-02-22 08:31:35'),
            pd.Timestamp('2022-02-22 08:31:57'),
            pd.Timestamp('2022-02-22 08:32:00')
        ]
        GT_data_ACC = [
            250, 350.5, 274.3, 475, 600,
            217, 312.5, 280, 260.5, 190
        ]
        dataframe_GT_ACC = pd.DataFrame(GT_data_ACC,
                                        index=GT_index_ACC,
                                        columns=['mg'])
        boundary_timestamp = pd.Timestamp('2022-02-22 08:30:15')
        window_size = pd.Timedelta('15 sec')
        output_ACC, output_HRV = clean_accelerometer_data_and_fill_according_to_HRV(
            ACC_input_dataframe,
            HRV_input_dataframe,
            boundary_timestamp,
            window_size
        )
        assert_frame_equal(dataframe_GT_ACC, output_ACC)
        assert_frame_equal(dataframe_GT_HRV, output_HRV)


if __name__ == "__main__":
    print('Run tests from the external path.')
