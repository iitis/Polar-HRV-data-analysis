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

import pandas as pd
import numpy as np
import pywt
from scipy.interpolate import CubicSpline


def remove_manually_anomalies(data, group, number):
    """
    Apply a manual anomaly detection according to the observations
    of ECG and RR intervals. It is an additional step of preprocessing.
    Remove also a preceding and following beats

    Arguments:
    ----------
       *data*: (Pandas Dataframe) contains data; timestamps are inserted
               in the 'Phone timestamp' column
       *group*: (string) 'treatment' or 'control'
       *number*: (int) number of the person from the treatment or control
                 group

    Returns:
    --------
        Pandas Dataframe containing data without anomalous values
    """
    if group == 'treatment':
        if number == 1:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '12:43:35') |
                    ((data['Phone timestamp'] >= '13:37:42') &
                     (data['Phone timestamp'] <= '13:37:52')) |
                    ((data['Phone timestamp'] >= '13:38:20') &
                     (data['Phone timestamp'] <= '13:38:28')) |
                    ((data['Phone timestamp'] >= '13:48:15') &
                     (data['Phone timestamp'] <= '13:48:30')) |
                    ((data['Phone timestamp'] >= '13:54:40') &
                     (data['Phone timestamp'] <= '13:54:45')).values
                )
            ]

        elif number == 2:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '09:04:46') &
                     (data['Phone timestamp'] <= '09:04:48')) |
                    ((data['Phone timestamp'] >= '09:12:00') &
                     (data['Phone timestamp'] <= '09:12:05')) |
                    ((data['Phone timestamp'] >= '09:25:14') &
                     (data['Phone timestamp'] <= '09:25:18')) |
                    ((data['Phone timestamp'] >= '09:31:10') &
                     (data['Phone timestamp'] <= '09:31:13')) |
                    ((data['Phone timestamp'] >= '09:32:50') &
                     (data['Phone timestamp'] <= '09:32:55')) |
                    ((data['Phone timestamp'] >= '09:33:22') &
                     (data['Phone timestamp'] <= '09:33:25')) |
                    ((data['Phone timestamp'] >= '09:34:54') &
                     (data['Phone timestamp'] <= '09:34:56')) |
                    ((data['Phone timestamp'] >= '09:56:10') &
                     (data['Phone timestamp'] <= '09:56:45')).values
                )
            ]

        elif number == 3:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '08:20:13') &
                     (data['Phone timestamp'] <= '08:20:30')) |
                    ((data['Phone timestamp'] >= '08:22:45') &
                     (data['Phone timestamp'] <= '08:22:57')) |
                    ((data['Phone timestamp'] >= '08:24:13') &
                     (data['Phone timestamp'] <= '08:24:18')) |
                    ((data['Phone timestamp'] >= '08:24:43') &
                     (data['Phone timestamp'] <= '08:24:48')) |
                    ((data['Phone timestamp'] >= '08:26:09') &
                     (data['Phone timestamp'] <= '08:26:11')) |
                    ((data['Phone timestamp'] >= '08:58:29') &
                     (data['Phone timestamp'] <= '08:58:30')) |
                    ((data['Phone timestamp'] >= '09:11:35') &
                     (data['Phone timestamp'] <= '09:11:40')) |
                    ((data['Phone timestamp'] >= '09:38:19') &
                     (data['Phone timestamp'] <= '09:38:23')) |
                    ((data['Phone timestamp'] >= '09:43:28') &
                     (data['Phone timestamp'] <= '09:43:31')) |
                    ((data['Phone timestamp'] >= '09:49:35') &
                     (data['Phone timestamp'] <= '09:49:40')) |
                    ((data['Phone timestamp'] >= '09:50:58') &
                     (data['Phone timestamp'] <= '09:51:05')).values
                )
            ]

        elif number == 7:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '12:36:28') &
                     (data['Phone timestamp'] <= '12:36:31')) |
                    ((data['Phone timestamp'] >= '12:39:38') &
                     (data['Phone timestamp'] <= '12:39:40')) |
                    ((data['Phone timestamp'] >= '14:00:40') &
                     (data['Phone timestamp'] <= '14:00:45')).values
                )
            ]

        elif number == 8:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '12:40:17') &
                     (data['Phone timestamp'] <= '12:40:23')).values
                )
            ]

        elif number == 9:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '11:53:02') &
                     (data['Phone timestamp'] <= '11:53:04')) |
                    ((data['Phone timestamp'] >= '12:52:30') &
                     (data['Phone timestamp'] <= '12:52:40')) |
                    ((data['Phone timestamp'] >= '12:58:31') &
                     (data['Phone timestamp'] <= '12:58:35')) |
                    ((data['Phone timestamp'] >= '13:02:05') &
                     (data['Phone timestamp'] <= '13:02:10')) |
                    ((data['Phone timestamp'] >= '13:02:20') &
                     (data['Phone timestamp'] <= '13:02:26')).values
                )
            ]

        elif number == 13:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '12:22:38') &
                     (data['Phone timestamp'] <= '12:22:40')) |
                    ((data['Phone timestamp'] >= '12:24:50') &
                     (data['Phone timestamp'] <= '12:24:52')) |
                    ((data['Phone timestamp'] >= '12:28:55') &
                     (data['Phone timestamp'] <= '12:28:58')) |
                    ((data['Phone timestamp'] >= '12:29:17') &
                     (data['Phone timestamp'] <= '12:29:20')) |
                    ((data['Phone timestamp'] >= '12:40:55') &
                     (data['Phone timestamp'] <= '12:41:10')) |
                    ((data['Phone timestamp'] >= '12:53:15') &
                     (data['Phone timestamp'] <= '12:53:20')) |
                    ((data['Phone timestamp'] >= '13:29:40') &
                     (data['Phone timestamp'] <= '13:29:50')).values
                )
            ]

        elif number == 15:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '12:10:28') &
                     (data['Phone timestamp'] <= '12:11:00')) |
                    ((data['Phone timestamp'] >= '12:13:00') &
                     (data['Phone timestamp'] <= '12:13:15')) |
                    ((data['Phone timestamp'] >= '12:16:35') &
                     (data['Phone timestamp'] <= '12:16:55')) |
                    ((data['Phone timestamp'] >= '12:21:05') &
                     (data['Phone timestamp'] <= '12:21:55')) |
                    ((data['Phone timestamp'] >= '12:30:34') &
                     (data['Phone timestamp'] <= '12:31:30')) |
                    ((data['Phone timestamp'] >= '12:49:18') &
                     (data['Phone timestamp'] <= '12:50:50')) |
                    ((data['Phone timestamp'] >= '13:03:25') &
                     (data['Phone timestamp'] <= '13:03:41')) |
                    ((data['Phone timestamp'] >= '13:37:00') &
                     (data['Phone timestamp'] <= '13:37:10')) |
                    ((data['Phone timestamp'] >= '13:37:45') &
                     (data['Phone timestamp'] <= '13:38:00')) |
                    ((data['Phone timestamp'] >= '13:39:38') &
                     (data['Phone timestamp'] <= '13:39:42')) |
                    ((data['Phone timestamp'] >= '13:51:17') &
                     (data['Phone timestamp'] <= '13:51:20')) |
                    ((data['Phone timestamp'] >= '13:51:24') &
                     (data['Phone timestamp'] <= '13:51:28')) |
                    ((data['Phone timestamp'] >= '13:51:35') &
                     (data['Phone timestamp'] <= '13:51:40')) |
                    ((data['Phone timestamp'] >= '13:53:06') &
                     (data['Phone timestamp'] <= '13:53:08')) |
                    ((data['Phone timestamp'] >= '13:57:17') &
                     (data['Phone timestamp'] <= '13:57:19')).values
                )
            ]

        elif number == 16:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '12:04:37') &
                     (data['Phone timestamp'] <= '12:05:00')) |
                    ((data['Phone timestamp'] >= '12:06:52') &
                     (data['Phone timestamp'] <= '12:07:00')) |
                    ((data['Phone timestamp'] >= '12:19:15') &
                     (data['Phone timestamp'] <= '12:19:20')) |
                    ((data['Phone timestamp'] >= '12:29:23') &
                     (data['Phone timestamp'] <= '12:29:25')) |
                    (data['Phone timestamp'] >= '13:32:00').values
                )
            ]

        elif number == 17:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '12:01:38') &
                     (data['Phone timestamp'] <= '12:01:43')) |
                    ((data['Phone timestamp'] >= '12:01:47') &
                     (data['Phone timestamp'] <= '12:01:57')) |
                    ((data['Phone timestamp'] >= '12:02:10') &
                     (data['Phone timestamp'] <= '12:02:15')) |
                    ((data['Phone timestamp'] >= '12:07:15') &
                     (data['Phone timestamp'] <= '12:07:42')) |
                    ((data['Phone timestamp'] >= '12:10:09') &
                     (data['Phone timestamp'] <= '12:10:11')) |
                    ((data['Phone timestamp'] >= '12:22:30') &
                     (data['Phone timestamp'] <= '12:22:37')) |
                    ((data['Phone timestamp'] >= '12:22:55') &
                     (data['Phone timestamp'] <= '12:23:05')) |
                    ((data['Phone timestamp'] >= '12:23:15') &
                     (data['Phone timestamp'] <= '12:23:27')) |
                    ((data['Phone timestamp'] >= '12:26:23') &
                     (data['Phone timestamp'] <= '12:27:00')) |
                    ((data['Phone timestamp'] >= '12:44:35') &
                     (data['Phone timestamp'] <= '12:44:38')) |
                    ((data['Phone timestamp'] >= '12:46:19') &
                     (data['Phone timestamp'] <= '12:46:21')) |
                    ((data['Phone timestamp'] >= '12:46:30') &
                     (data['Phone timestamp'] <= '12:46:40')) |
                    ((data['Phone timestamp'] >= '12:48:05') &
                     (data['Phone timestamp'] <= '12:48:15')) |
                    ((data['Phone timestamp'] >= '12:49:20') &
                     (data['Phone timestamp'] <= '12:49:30')) |
                    ((data['Phone timestamp'] >= '12:58:23') &
                     (data['Phone timestamp'] <= '12:58:30')) |
                    ((data['Phone timestamp'] >= '12:58:47') &
                     (data['Phone timestamp'] <= '12:58:53')) |
                    ((data['Phone timestamp'] >= '12:59:23') &
                     (data['Phone timestamp'] <= '12:59:28')) |
                    ((data['Phone timestamp'] >= '12:59:37') &
                     (data['Phone timestamp'] <= '12:59:41')) |
                    ((data['Phone timestamp'] >= '13:06:10') &
                     (data['Phone timestamp'] <= '13:07:00')) |
                    ((data['Phone timestamp'] >= '13:14:32') &
                     (data['Phone timestamp'] <= '13:14:36')) |
                    ((data['Phone timestamp'] >= '13:14:53') &
                     (data['Phone timestamp'] <= '13:14:58')) |
                    ((data['Phone timestamp'] >= '13:15:13') &
                     (data['Phone timestamp'] <= '13:15:18')) |
                    (data['Phone timestamp'] >= '13:31:40').values
                )
            ]

        elif number == 19:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '15:01:16') &
                     (data['Phone timestamp'] <= '15:01:20'))
                )
            ]

        elif number == 20:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '13:46:20') &
                     (data['Phone timestamp'] <= '13:46:37')) |
                    ((data['Phone timestamp'] >= '13:47:17') &
                     (data['Phone timestamp'] <= '13:47:22')) |
                    ((data['Phone timestamp'] >= '13:49:17') &
                     (data['Phone timestamp'] <= '13:49:20')) |
                    ((data['Phone timestamp'] >= '14:02:49') &
                     (data['Phone timestamp'] <= '14:02:51')) |
                    ((data['Phone timestamp'] >= '14:19:52') &
                     (data['Phone timestamp'] <= '14:19:55')) |
                    ((data['Phone timestamp'] >= '14:20:40') &
                     (data['Phone timestamp'] <= '14:20:45')) |
                    ((data['Phone timestamp'] >= '14:37:18') &
                     (data['Phone timestamp'] <= '14:37:20')) |
                    ((data['Phone timestamp'] >= '14:48:35') &
                     (data['Phone timestamp'] <= '14:48:42')) |
                    ((data['Phone timestamp'] >= '14:59:17') &
                     (data['Phone timestamp'] <= '14:59:20')) |
                    ((data['Phone timestamp'] >= '15:01:42') &
                     (data['Phone timestamp'] <= '15:01:44')) |
                    ((data['Phone timestamp'] >= '15:03:00') &
                     (data['Phone timestamp'] <= '15:03:03')) |
                    ((data['Phone timestamp'] >= '15:04:47') &
                     (data['Phone timestamp'] <= '15:04:50')) |
                    ((data['Phone timestamp'] >= '15:05:38') &
                     (data['Phone timestamp'] <= '15:05:41')) |
                    ((data['Phone timestamp'] >= '15:05:54') &
                     (data['Phone timestamp'] <= '15:05:57')).values
                )
            ]

        elif number == 21:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '11:46:08') &
                     (data['Phone timestamp'] <= '11:46:11')) |
                    ((data['Phone timestamp'] >= '11:46:55') &
                     (data['Phone timestamp'] <= '11:47:00')) |
                    ((data['Phone timestamp'] >= '12:53:19') &
                     (data['Phone timestamp'] <= '12:53:22'))
                )
            ]

        elif number == 22:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '12:24:25') |
                    ((data['Phone timestamp'] >= '12:25:40') &
                     (data['Phone timestamp'] <= '12:25:58')) |
                    ((data['Phone timestamp'] >= '12:26:28') &
                     (data['Phone timestamp'] <= '12:26:33')) |
                    ((data['Phone timestamp'] >= '12:28:02') &
                     (data['Phone timestamp'] <= '12:28:05')) |
                    ((data['Phone timestamp'] >= '12:28:30') &
                     (data['Phone timestamp'] <= '12:28:39')) |
                    ((data['Phone timestamp'] >= '12:35:24') &
                     (data['Phone timestamp'] <= '12:35:28')) |
                    ((data['Phone timestamp'] >= '12:53:20') &
                     (data['Phone timestamp'] <= '12:53:23')) |
                    ((data['Phone timestamp'] >= '12:53:58') &
                     (data['Phone timestamp'] <= '12:54:14')) |
                    ((data['Phone timestamp'] >= '12:59:50') &
                     (data['Phone timestamp'] <= '13:00:22')) |
                    ((data['Phone timestamp'] >= '13:00:50') &
                     (data['Phone timestamp'] <= '13:00:53')) |
                    ((data['Phone timestamp'] >= '13:03:13') &
                     (data['Phone timestamp'] <= '13:03:17')) |
                    ((data['Phone timestamp'] >= '13:03:43') &
                     (data['Phone timestamp'] <= '13:03:47')) |
                    ((data['Phone timestamp'] >= '13:10:30') &
                     (data['Phone timestamp'] <= '13:11:00')) |
                    ((data['Phone timestamp'] >= '13:12:59') &
                     (data['Phone timestamp'] <= '13:13:02')) |
                    ((data['Phone timestamp'] >= '13:16:00') &
                     (data['Phone timestamp'] <= '13:16:55')) |
                    ((data['Phone timestamp'] >= '13:17:56') &
                     (data['Phone timestamp'] <= '13:17:59')) |
                    ((data['Phone timestamp'] >= '13:21:12') &
                     (data['Phone timestamp'] <= '13:21:14')) |
                    ((data['Phone timestamp'] >= '13:23:28') &
                     (data['Phone timestamp'] <= '13:23:50')) |
                    ((data['Phone timestamp'] >= '13:24:15') &
                     (data['Phone timestamp'] <= '13:24:20')) |
                    ((data['Phone timestamp'] >= '13:28:53') &
                     (data['Phone timestamp'] <= '13:29:00')) |
                    ((data['Phone timestamp'] >= '13:29:10') &
                     (data['Phone timestamp'] <= '13:29:15')) |
                    ((data['Phone timestamp'] >= '13:29:20') &
                     (data['Phone timestamp'] <= '13:29:22')) |
                    ((data['Phone timestamp'] >= '13:29:30') &
                     (data['Phone timestamp'] <= '13:29:40')) |
                    ((data['Phone timestamp'] >= '13:30:00') &
                     (data['Phone timestamp'] <= '13:30:05')) |
                    ((data['Phone timestamp'] >= '13:32:15') &
                     (data['Phone timestamp'] <= '13:32:22')) |
                    ((data['Phone timestamp'] >= '13:33:25') &
                     (data['Phone timestamp'] <= '13:33:30')) |
                    ((data['Phone timestamp'] >= '13:48:19') &
                     (data['Phone timestamp'] <= '13:48:22')) |
                    ((data['Phone timestamp'] >= '13:48:27') &
                     (data['Phone timestamp'] <= '13:48:30')) |
                    ((data['Phone timestamp'] >= '13:48:50') &
                     (data['Phone timestamp'] <= '13:48:53')) |
                    ((data['Phone timestamp'] >= '13:50:27') &
                     (data['Phone timestamp'] <= '13:50:30')).values
                )
            ]

        elif number == 23:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '12:40:45') |
                    ((data['Phone timestamp'] >= '12:41:05') &
                     (data['Phone timestamp'] <= '12:41:10')) |
                    ((data['Phone timestamp'] >= '12:41:33') &
                     (data['Phone timestamp'] <= '12:41:58')) |
                    ((data['Phone timestamp'] >= '12:42:01') &
                     (data['Phone timestamp'] <= '12:42:04')) |
                    ((data['Phone timestamp'] >= '12:42:12') &
                     (data['Phone timestamp'] <= '12:42:15')) |
                    ((data['Phone timestamp'] >= '12:43:03') &
                     (data['Phone timestamp'] <= '12:43:07')) |
                    ((data['Phone timestamp'] >= '12:46:58') &
                     (data['Phone timestamp'] <= '12:47:01')) |
                    ((data['Phone timestamp'] >= '12:50:15') &
                     (data['Phone timestamp'] <= '12:50:17')).values
                )
            ]

        elif number == 24:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '11:19:16') &
                     (data['Phone timestamp'] <= '11:19:21')) |
                    ((data['Phone timestamp'] >= '11:31:17') &
                     (data['Phone timestamp'] <= '11:31:20')) |
                    ((data['Phone timestamp'] >= '11:31:37') &
                     (data['Phone timestamp'] <= '11:31:40')) |
                    ((data['Phone timestamp'] >= '11:32:26') &
                     (data['Phone timestamp'] <= '11:32:29')) |
                    ((data['Phone timestamp'] >= '12:01:15') &
                     (data['Phone timestamp'] <= '12:02:20')) |
                    ((data['Phone timestamp'] >= '12:17:23') &
                     (data['Phone timestamp'] <= '12:17:27')) |
                    ((data['Phone timestamp'] >= '12:21:35') &
                     (data['Phone timestamp'] <= '12:21:38')) |
                    ((data['Phone timestamp'] >= '12:23:02') &
                     (data['Phone timestamp'] <= '12:23:05')) |
                    ((data['Phone timestamp'] >= '12:28:00') &
                     (data['Phone timestamp'] <= '12:28:15')) |
                    ((data['Phone timestamp'] >= '12:30:10') &
                     (data['Phone timestamp'] <= '12:30:15')) |
                    ((data['Phone timestamp'] >= '12:30:35') &
                     (data['Phone timestamp'] <= '12:30:40')) |
                    ((data['Phone timestamp'] >= '12:48:05') &
                     (data['Phone timestamp'] <= '12:48:10')) |
                    ((data['Phone timestamp'] >= '12:57:52') &
                     (data['Phone timestamp'] <= '12:57:58')).values
                )
            ]

        elif number == 25:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:42:22') |
                    ((data['Phone timestamp'] >= '10:42:50') &
                     (data['Phone timestamp'] <= '10:42:58')) |
                    ((data['Phone timestamp'] >= '10:43:23') &
                     (data['Phone timestamp'] <= '10:43:28')) |
                    ((data['Phone timestamp'] >= '10:43:40') &
                     (data['Phone timestamp'] <= '10:44:07')) |
                    ((data['Phone timestamp'] >= '10:45:05') &
                     (data['Phone timestamp'] <= '10:45:25')) |
                    ((data['Phone timestamp'] >= '10:45:40') &
                     (data['Phone timestamp'] <= '10:45:55')) |
                    ((data['Phone timestamp'] >= '10:46:20') &
                     (data['Phone timestamp'] <= '10:46:25')) |
                    ((data['Phone timestamp'] >= '10:46:50') &
                     (data['Phone timestamp'] <= '10:47:30')) |
                    ((data['Phone timestamp'] >= '10:48:10') &
                     (data['Phone timestamp'] <= '10:48:40')) |
                    ((data['Phone timestamp'] >= '10:57:08') &
                     (data['Phone timestamp'] <= '10:57:13')) |
                    ((data['Phone timestamp'] >= '10:57:30') &
                     (data['Phone timestamp'] <= '10:57:35')) |
                    ((data['Phone timestamp'] >= '11:17:20') &
                     (data['Phone timestamp'] <= '11:17:55')) |
                    ((data['Phone timestamp'] >= '11:26:30') &
                     (data['Phone timestamp'] <= '11:26:36')) |
                    ((data['Phone timestamp'] >= '11:31:43') &
                     (data['Phone timestamp'] <= '11:31:48')) |
                    ((data['Phone timestamp'] >= '12:01:10') &
                     (data['Phone timestamp'] <= '12:01:15')) |
                    ((data['Phone timestamp'] >= '12:02:11') &
                     (data['Phone timestamp'] <= '12:02:14')).values
                )
            ]

        elif number == 26:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:14:30') |
                    ((data['Phone timestamp'] >= '11:23:25') &
                     (data['Phone timestamp'] <= '11:24:10')) |
                    ((data['Phone timestamp'] >= '11:26:45') &
                     (data['Phone timestamp'] <= '11:27:10')) |
                    ((data['Phone timestamp'] >= '11:27:30') &
                     (data['Phone timestamp'] <= '11:27:45')) |
                    ((data['Phone timestamp'] >= '11:29:30') &
                     (data['Phone timestamp'] <= '11:32:10')) |
                    ((data['Phone timestamp'] >= '11:32:51') &
                     (data['Phone timestamp'] <= '11:32:55')) |
                    ((data['Phone timestamp'] >= '11:36:40') &
                     (data['Phone timestamp'] <= '11:36:55')) |
                    ((data['Phone timestamp'] >= '11:37:08') &
                     (data['Phone timestamp'] <= '11:37:45')) |
                    ((data['Phone timestamp'] >= '11:38:15') &
                     (data['Phone timestamp'] <= '11:38:40')) |
                    ((data['Phone timestamp'] >= '11:43:40') &
                     (data['Phone timestamp'] <= '11:43:57')) |
                    ((data['Phone timestamp'] >= '11:45:18') &
                     (data['Phone timestamp'] <= '11:45:24')) |
                    ((data['Phone timestamp'] >= '11:52:00') &
                     (data['Phone timestamp'] <= '11:52:30')) |
                    ((data['Phone timestamp'] >= '11:52:52') &
                     (data['Phone timestamp'] <= '11:53:00')) |
                    ((data['Phone timestamp'] >= '11:53:30') &
                     (data['Phone timestamp'] <= '11:53:32')) |
                    ((data['Phone timestamp'] >= '11:53:47') &
                     (data['Phone timestamp'] <= '11:53:50')) |
                    ((data['Phone timestamp'] >= '12:11:25') &
                     (data['Phone timestamp'] <= '12:11:35')) |
                    ((data['Phone timestamp'] >= '12:12:00') &
                     (data['Phone timestamp'] <= '12:12:08')) |
                    ((data['Phone timestamp'] >= '12:13:00') &
                     (data['Phone timestamp'] <= '12:13:20')) |
                    ((data['Phone timestamp'] >= '12:15:45') &
                     (data['Phone timestamp'] <= '12:16:00')) |
                    ((data['Phone timestamp'] >= '12:24:15') &
                     (data['Phone timestamp'] <= '12:24:27')) |
                    ((data['Phone timestamp'] >= '12:36:25') &
                     (data['Phone timestamp'] <= '12:36:40')) |
                    ((data['Phone timestamp'] >= '12:37:20') &
                     (data['Phone timestamp'] <= '12:38:00')) |
                    ((data['Phone timestamp'] >= '12:43:27') &
                     (data['Phone timestamp'] <= '12:43:31')) |
                    ((data['Phone timestamp'] >= '12:44:16') &
                     (data['Phone timestamp'] <= '12:44:18')) |
                    ((data['Phone timestamp'] >= '12:44:23') &
                     (data['Phone timestamp'] <= '12:44:27')) |
                    ((data['Phone timestamp'] >= '12:45:50') &
                     (data['Phone timestamp'] <= '12:45:53')) |
                    ((data['Phone timestamp'] >= '12:46:20') &
                     (data['Phone timestamp'] <= '12:46:28')) |
                    ((data['Phone timestamp'] >= '12:53:00') &
                     (data['Phone timestamp'] <= '12:53:28')) |
                    ((data['Phone timestamp'] >= '12:53:55') &
                     (data['Phone timestamp'] <= '12:55:10')) |
                    ((data['Phone timestamp'] >= '12:56:00') &
                     (data['Phone timestamp'] <= '12:56:05')) |
                    ((data['Phone timestamp'] >= '12:57:05') &
                     (data['Phone timestamp'] <= '12:57:45')) |
                    ((data['Phone timestamp'] >= '13:06:05') &
                     (data['Phone timestamp'] <= '13:06:20')) |
                    ((data['Phone timestamp'] >= '13:07:00') &
                     (data['Phone timestamp'] <= '13:07:25')).values
                )
            ]

        elif number == 27:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:30:27') |
                    ((data['Phone timestamp'] >= '11:31:45') &
                     (data['Phone timestamp'] <= '11:31:50')) |
                    ((data['Phone timestamp'] >= '12:00:00') &
                     (data['Phone timestamp'] <= '12:00:30')) |
                    ((data['Phone timestamp'] >= '12:00:50') &
                     (data['Phone timestamp'] <= '12:01:05')) |
                    ((data['Phone timestamp'] >= '12:01:27') &
                     (data['Phone timestamp'] <= '12:01:29')) |
                    ((data['Phone timestamp'] >= '12:21:22') &
                     (data['Phone timestamp'] <= '12:21:27')) |
                    ((data['Phone timestamp'] >= '12:40:58') &
                     (data['Phone timestamp'] <= '12:41:02')) |
                    ((data['Phone timestamp'] >= '12:41:37') &
                     (data['Phone timestamp'] <= '12:41:40')) |
                    ((data['Phone timestamp'] >= '13:12:00') &
                     (data['Phone timestamp'] <= '13:12:22')) |
                    ((data['Phone timestamp'] >= '13:12:40') &
                     (data['Phone timestamp'] <= '13:12:45')) |
                    ((data['Phone timestamp'] >= '13:12:50') &
                     (data['Phone timestamp'] <= '13:12:57')).values
                )
            ]

        elif number == 29:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '12:12:25') |
                    ((data['Phone timestamp'] >= '12:12:30') &
                     (data['Phone timestamp'] <= '12:12:50')) |
                    ((data['Phone timestamp'] >= '12:13:17') &
                     (data['Phone timestamp'] <= '12:13:30')) |
                    ((data['Phone timestamp'] >= '12:16:30') &
                     (data['Phone timestamp'] <= '12:16:42')) |
                    ((data['Phone timestamp'] >= '13:05:30') &
                     (data['Phone timestamp'] <= '13:05:38')) |
                    ((data['Phone timestamp'] >= '13:33:30') &
                     (data['Phone timestamp'] <= '13:33:35')) |
                    ((data['Phone timestamp'] >= '13:43:30') &
                     (data['Phone timestamp'] <= '13:43:39')).values
                )
            ]

        elif number == 31:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '09:12:25') |
                    ((data['Phone timestamp'] >= '09:13:13') &
                     (data['Phone timestamp'] <= '09:13:43')) |
                    ((data['Phone timestamp'] >= '09:13:52') &
                     (data['Phone timestamp'] <= '09:14:15')) |
                    ((data['Phone timestamp'] >= '09:17:07') &
                     (data['Phone timestamp'] <= '09:17:13')) |
                    ((data['Phone timestamp'] >= '09:18:23') &
                     (data['Phone timestamp'] <= '09:18:28')) |
                    ((data['Phone timestamp'] >= '09:18:47') &
                     (data['Phone timestamp'] <= '09:18:50')) |
                    ((data['Phone timestamp'] >= '09:19:35') &
                     (data['Phone timestamp'] <= '09:19:42')) |
                    ((data['Phone timestamp'] >= '09:27:10') &
                     (data['Phone timestamp'] <= '09:27:55')) |
                    ((data['Phone timestamp'] >= '09:29:40') &
                     (data['Phone timestamp'] <= '09:29:45')) |
                    ((data['Phone timestamp'] >= '09:30:10') &
                     (data['Phone timestamp'] <= '09:30:15')) |
                    ((data['Phone timestamp'] >= '09:30:35') &
                     (data['Phone timestamp'] <= '09:30:40')) |
                    ((data['Phone timestamp'] >= '09:30:55') &
                     (data['Phone timestamp'] <= '09:31:15')) |
                    ((data['Phone timestamp'] >= '09:36:05') &
                     (data['Phone timestamp'] <= '09:36:10')) |
                    ((data['Phone timestamp'] >= '10:06:05') &
                     (data['Phone timestamp'] <= '10:06:08')) |
                    ((data['Phone timestamp'] >= '10:06:43') &
                     (data['Phone timestamp'] <= '10:06:47')) |
                    ((data['Phone timestamp'] >= '10:10:32') &
                     (data['Phone timestamp'] <= '10:10:37')) |
                    ((data['Phone timestamp'] >= '10:24:40') &
                     (data['Phone timestamp'] <= '10:24:45')) |
                    ((data['Phone timestamp'] >= '10:26:10') &
                     (data['Phone timestamp'] <= '10:26:12')).values
                )
            ]

        elif number == 32:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '09:24:45') |
                    ((data['Phone timestamp'] >= '09:45:57') &
                     (data['Phone timestamp'] <= '09:45:58')) |
                    ((data['Phone timestamp'] >= '10:00:12') &
                     (data['Phone timestamp'] <= '10:00:19')) |
                    ((data['Phone timestamp'] >= '10:03:29') &
                     (data['Phone timestamp'] <= '10:03:30')) |
                    ((data['Phone timestamp'] >= '10:35:17') &
                     (data['Phone timestamp'] <= '10:35:19')).values
                )
            ]

        elif number == 33:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:46:05') |
                    ((data['Phone timestamp'] >= '11:58:48') &
                     (data['Phone timestamp'] <= '11:59:12')).values
                )
            ]

        elif number == 36:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:31:28') |
                    ((data['Phone timestamp'] >= '10:37:50') &
                     (data['Phone timestamp'] <= '10:38:20')) |
                    ((data['Phone timestamp'] >= '10:54:35') &
                     (data['Phone timestamp'] <= '10:55:05')) |
                    ((data['Phone timestamp'] >= '10:58:11') &
                     (data['Phone timestamp'] <= '10:58:41')) |
                    ((data['Phone timestamp'] >= '11:00:00') &
                     (data['Phone timestamp'] <= '11:00:45')) |
                    ((data['Phone timestamp'] >= '11:02:07') &
                     (data['Phone timestamp'] <= '11:03:00')) |
                    ((data['Phone timestamp'] >= '11:03:22') &
                     (data['Phone timestamp'] <= '11:03:25')) |
                    ((data['Phone timestamp'] >= '11:06:40') &
                     (data['Phone timestamp'] <= '11:06:50')) |
                    ((data['Phone timestamp'] >= '11:07:04') &
                     (data['Phone timestamp'] <= '11:07:07')) |
                    ((data['Phone timestamp'] >= '11:29:30') &
                     (data['Phone timestamp'] <= '11:29:50')) |
                    ((data['Phone timestamp'] >= '11:46:04') &
                     (data['Phone timestamp'] <= '11:46:07')) |
                    ((data['Phone timestamp'] >= '11:46:50') &
                     (data['Phone timestamp'] <= '11:46:53')) |
                    ((data['Phone timestamp'] >= '11:47:10') &
                     (data['Phone timestamp'] <= '11:47:15')) |
                    ((data['Phone timestamp'] >= '11:47:32') &
                     (data['Phone timestamp'] <= '11:47:35')) |
                    ((data['Phone timestamp'] >= '11:47:42') &
                     (data['Phone timestamp'] <= '11:47:54')) |
                    ((data['Phone timestamp'] >= '11:49:23') &
                     (data['Phone timestamp'] <= '11:49:27')) |
                    ((data['Phone timestamp'] >= '11:49:33') &
                     (data['Phone timestamp'] <= '11:49:36')) |
                    ((data['Phone timestamp'] >= '11:49:52') &
                     (data['Phone timestamp'] <= '11:49:58')).values
                )
            ]

        elif number == 37:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:37:45') |
                    ((data['Phone timestamp'] >= '10:38:13') &
                     (data['Phone timestamp'] <= '10:38:17')) |
                    ((data['Phone timestamp'] >= '10:38:28') &
                     (data['Phone timestamp'] <= '10:38:32')) |
                    ((data['Phone timestamp'] >= '10:38:45') &
                     (data['Phone timestamp'] <= '10:39:50')) |
                    ((data['Phone timestamp'] >= '10:40:00') &
                     (data['Phone timestamp'] <= '10:40:05')) |
                    ((data['Phone timestamp'] >= '10:41:03') &
                     (data['Phone timestamp'] <= '10:41:45')) |
                    ((data['Phone timestamp'] >= '10:41:58') &
                     (data['Phone timestamp'] <= '10:42:03')) |
                    ((data['Phone timestamp'] >= '10:42:28') &
                     (data['Phone timestamp'] <= '10:48:45')) |
                    ((data['Phone timestamp'] >= '10:51:20') &
                     (data['Phone timestamp'] <= '10:51:40')) |
                    ((data['Phone timestamp'] >= '10:57:20') &
                     (data['Phone timestamp'] <= '10:57:40')) |
                    ((data['Phone timestamp'] >= '11:10:19') &
                     (data['Phone timestamp'] <= '11:10:21')) |
                    ((data['Phone timestamp'] >= '11:13:16') &
                     (data['Phone timestamp'] <= '11:13:20')) |
                    ((data['Phone timestamp'] >= '11:18:58') &
                     (data['Phone timestamp'] <= '11:18:59')) |
                    ((data['Phone timestamp'] >= '11:19:37') &
                     (data['Phone timestamp'] <= '11:19:39')) |
                    ((data['Phone timestamp'] >= '11:20:28') &
                     (data['Phone timestamp'] <= '11:20:35')) |
                    ((data['Phone timestamp'] >= '11:39:16') &
                     (data['Phone timestamp'] <= '11:39:17')).values
                )
            ]

        elif number == 38:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:50:40') |
                    ((data['Phone timestamp'] >= '10:51:40') &
                     (data['Phone timestamp'] <= '10:51:50')) |
                    ((data['Phone timestamp'] >= '10:54:15') &
                     (data['Phone timestamp'] <= '10:54:22')) |
                    ((data['Phone timestamp'] >= '10:55:18') &
                     (data['Phone timestamp'] <= '10:55:25')) |
                    ((data['Phone timestamp'] >= '10:55:28') &
                     (data['Phone timestamp'] <= '10:55:40')) |
                    ((data['Phone timestamp'] >= '10:56:17') &
                     (data['Phone timestamp'] <= '10:56:28')) |
                    ((data['Phone timestamp'] >= '10:56:47') &
                     (data['Phone timestamp'] <= '10:57:24')) |
                    ((data['Phone timestamp'] >= '11:00:12') &
                     (data['Phone timestamp'] <= '11:00:30')) |
                    ((data['Phone timestamp'] >= '11:01:15') &
                     (data['Phone timestamp'] <= '11:01:20')) |
                    ((data['Phone timestamp'] >= '11:04:00') &
                     (data['Phone timestamp'] <= '11:04:05')) |
                    ((data['Phone timestamp'] >= '11:04:20') &
                     (data['Phone timestamp'] <= '11:04:27')) |
                    ((data['Phone timestamp'] >= '11:04:32') &
                     (data['Phone timestamp'] <= '11:04:38')) |
                    ((data['Phone timestamp'] >= '11:07:00') &
                     (data['Phone timestamp'] <= '11:07:20')) |
                    ((data['Phone timestamp'] >= '11:09:09') &
                     (data['Phone timestamp'] <= '11:09:10')) |
                    ((data['Phone timestamp'] >= '11:09:40') &
                     (data['Phone timestamp'] <= '11:10:00')) |
                    ((data['Phone timestamp'] >= '11:10:20') &
                     (data['Phone timestamp'] <= '11:10:45')) |
                    ((data['Phone timestamp'] >= '11:12:40') &
                     (data['Phone timestamp'] <= '11:13:00')) |
                    ((data['Phone timestamp'] >= '11:15:00') &
                     (data['Phone timestamp'] <= '11:15:07')) |
                    ((data['Phone timestamp'] >= '11:16:45') &
                     (data['Phone timestamp'] <= '11:16:58')) |
                    ((data['Phone timestamp'] >= '11:17:10') &
                     (data['Phone timestamp'] <= '11:17:45')) |
                    ((data['Phone timestamp'] >= '11:19:05') &
                     (data['Phone timestamp'] <= '11:19:17')) |
                    ((data['Phone timestamp'] >= '11:20:16') &
                     (data['Phone timestamp'] <= '11:20:18')) |
                    ((data['Phone timestamp'] >= '11:21:30') &
                     (data['Phone timestamp'] <= '11:21:35')) |
                    ((data['Phone timestamp'] >= '11:21:40') &
                     (data['Phone timestamp'] <= '11:21:45')) |
                    ((data['Phone timestamp'] >= '11:22:20') &
                     (data['Phone timestamp'] <= '11:22:25')) |
                    ((data['Phone timestamp'] >= '11:22:50') &
                     (data['Phone timestamp'] <= '11:22:55')) |
                    ((data['Phone timestamp'] >= '11:33:26') &
                     (data['Phone timestamp'] <= '11:33:27')) |
                    ((data['Phone timestamp'] >= '12:12:41') &
                     (data['Phone timestamp'] <= '12:12:47')).values
                )
            ]

        elif number == 39:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '09:50:00') |
                    ((data['Phone timestamp'] >= '09:51:10') &
                     (data['Phone timestamp'] <= '09:51:20')) |
                    ((data['Phone timestamp'] >= '09:51:45') &
                     (data['Phone timestamp'] <= '09:52:00')) |
                    ((data['Phone timestamp'] >= '09:53:30') &
                     (data['Phone timestamp'] <= '09:53:35')) |
                    ((data['Phone timestamp'] >= '09:54:18') &
                     (data['Phone timestamp'] <= '09:54:23')) |
                    ((data['Phone timestamp'] >= '10:01:46') &
                     (data['Phone timestamp'] <= '10:01:58')) |
                    ((data['Phone timestamp'] >= '10:24:32') &
                     (data['Phone timestamp'] <= '10:24:42')) |
                    ((data['Phone timestamp'] >= '10:26:54') &
                     (data['Phone timestamp'] <= '10:26:57')) |
                    ((data['Phone timestamp'] >= '10:42:40') &
                     (data['Phone timestamp'] <= '10:42:42')) |
                    ((data['Phone timestamp'] >= '11:18:32') &
                     (data['Phone timestamp'] <= '11:18:34')) |
                    ((data['Phone timestamp'] >= '11:20:00') &
                     (data['Phone timestamp'] <= '11:20:05')).values
                )
            ]

        elif number == 40:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:01:07') |
                    ((data['Phone timestamp'] >= '10:01:40') &
                     (data['Phone timestamp'] <= '10:01:50')) |
                    ((data['Phone timestamp'] >= '10:02:10') &
                     (data['Phone timestamp'] <= '10:02:30')) |
                    ((data['Phone timestamp'] >= '10:06:05') &
                     (data['Phone timestamp'] <= '10:06:10')) |
                    ((data['Phone timestamp'] >= '10:06:50') &
                     (data['Phone timestamp'] <= '10:06:53')) |
                    ((data['Phone timestamp'] >= '10:12:20') &
                     (data['Phone timestamp'] <= '10:12:30')) |
                    ((data['Phone timestamp'] >= '10:12:39') &
                     (data['Phone timestamp'] <= '10:12:54')) |
                    ((data['Phone timestamp'] >= '10:21:14') &
                     (data['Phone timestamp'] <= '10:21:17')) |
                    ((data['Phone timestamp'] >= '10:24:52') &
                     (data['Phone timestamp'] <= '10:24:54')) |
                    ((data['Phone timestamp'] >= '10:28:35') &
                     (data['Phone timestamp'] <= '10:28:36')) |
                    ((data['Phone timestamp'] >= '10:30:30') &
                     (data['Phone timestamp'] <= '10:30:47')) |
                    ((data['Phone timestamp'] >= '10:34:15') &
                     (data['Phone timestamp'] <= '10:34:18')) |
                    ((data['Phone timestamp'] >= '10:37:20') &
                     (data['Phone timestamp'] <= '10:37:23')) |
                    ((data['Phone timestamp'] >= '10:38:54') &
                     (data['Phone timestamp'] <= '10:38:55')) |
                    ((data['Phone timestamp'] >= '10:55:28') &
                     (data['Phone timestamp'] <= '10:55:34')) |
                    ((data['Phone timestamp'] >= '10:56:34') &
                     (data['Phone timestamp'] <= '10:56:36')) |
                    ((data['Phone timestamp'] >= '10:57:05') &
                     (data['Phone timestamp'] <= '10:57:08')) |
                    ((data['Phone timestamp'] >= '11:07:22') &
                     (data['Phone timestamp'] <= '11:07:25')) |
                    ((data['Phone timestamp'] >= '11:12:31') &
                     (data['Phone timestamp'] <= '11:12:33')) |
                    ((data['Phone timestamp'] >= '11:24:16') &
                     (data['Phone timestamp'] <= '11:24:19')).values
                )
            ]

        elif number == 41:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:06:12') |
                    ((data['Phone timestamp'] >= '10:06:45') &
                     (data['Phone timestamp'] <= '10:06:55')) |
                    ((data['Phone timestamp'] >= '10:07:35') &
                     (data['Phone timestamp'] <= '10:07:45')) |
                    ((data['Phone timestamp'] >= '10:08:02') &
                     (data['Phone timestamp'] <= '10:08:10')) |
                    ((data['Phone timestamp'] >= '10:08:47') &
                     (data['Phone timestamp'] <= '10:11:07')) |
                    ((data['Phone timestamp'] >= '10:12:00') &
                     (data['Phone timestamp'] <= '10:12:10')) |
                    ((data['Phone timestamp'] >= '10:12:40') &
                     (data['Phone timestamp'] <= '10:12:47')) |
                    ((data['Phone timestamp'] >= '10:13:05') &
                     (data['Phone timestamp'] <= '10:13:12')) |
                    ((data['Phone timestamp'] >= '10:14:07') &
                     (data['Phone timestamp'] <= '10:14:12')) |
                    ((data['Phone timestamp'] >= '10:23:55') &
                     (data['Phone timestamp'] <= '10:24:20')) |
                    ((data['Phone timestamp'] >= '10:24:47') &
                     (data['Phone timestamp'] <= '10:25:00')) |
                    ((data['Phone timestamp'] >= '10:25:17') &
                     (data['Phone timestamp'] <= '10:25:27')) |
                    ((data['Phone timestamp'] >= '10:25:53') &
                     (data['Phone timestamp'] <= '10:26:05')) |
                    ((data['Phone timestamp'] >= '10:26:17') &
                     (data['Phone timestamp'] <= '10:26:25')) |
                    ((data['Phone timestamp'] >= '10:26:50') &
                     (data['Phone timestamp'] <= '10:27:00')) |
                    ((data['Phone timestamp'] >= '10:30:05') &
                     (data['Phone timestamp'] <= '10:30:25')) |
                    ((data['Phone timestamp'] >= '10:35:50') &
                     (data['Phone timestamp'] <= '10:36:15')) |
                    ((data['Phone timestamp'] >= '10:38:15') &
                     (data['Phone timestamp'] <= '10:38:30')) |
                    ((data['Phone timestamp'] >= '10:39:36') &
                     (data['Phone timestamp'] <= '10:39:40')) |
                    ((data['Phone timestamp'] >= '10:44:00') &
                     (data['Phone timestamp'] <= '10:48:00')) |
                    ((data['Phone timestamp'] >= '10:50:00') &
                     (data['Phone timestamp'] <= '10:50:20')).values
                )
            ]

        elif number == 42:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '10:02:55') &
                     (data['Phone timestamp'] <= '10:03:02')) |
                    ((data['Phone timestamp'] >= '10:03:36') &
                     (data['Phone timestamp'] <= '10:03:40')) |
                    ((data['Phone timestamp'] >= '10:04:10') &
                     (data['Phone timestamp'] <= '10:04:17')) |
                    ((data['Phone timestamp'] >= '10:11:45') &
                     (data['Phone timestamp'] <= '10:11:58')) |
                    ((data['Phone timestamp'] >= '10:21:06') &
                     (data['Phone timestamp'] <= '10:21:08')) |
                    ((data['Phone timestamp'] >= '10:46:33') &
                     (data['Phone timestamp'] <= '10:46:35')) |
                    ((data['Phone timestamp'] >= '10:58:14') &
                     (data['Phone timestamp'] <= '10:58:18')) |
                    ((data['Phone timestamp'] >= '10:58:38') &
                     (data['Phone timestamp'] <= '10:58:40')) |
                    ((data['Phone timestamp'] >= '11:07:24') &
                     (data['Phone timestamp'] <= '11:07:28')) |
                    ((data['Phone timestamp'] >= '11:24:42') &
                     (data['Phone timestamp'] <= '11:24:47')).values
                )
            ]

        else:
            return data

    elif group == 'control':
        if number == 1:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:01:35') |
                    ((data['Phone timestamp'] >= '10:04:50') &
                     (data['Phone timestamp'] <= '10:05:00')) |
                    ((data['Phone timestamp'] >= '10:32:24') &
                     (data['Phone timestamp'] <= '10:32:28')) |
                    ((data['Phone timestamp'] >= '10:37:30') &
                     (data['Phone timestamp'] <= '10:37:33')) |
                    ((data['Phone timestamp'] >= '10:41:12') &
                     (data['Phone timestamp'] <= '10:41:14')) |
                    ((data['Phone timestamp'] >= '10:49:22') &
                     (data['Phone timestamp'] <= '10:49:30')) |
                    ((data['Phone timestamp'] >= '10:50:53') &
                     (data['Phone timestamp'] <= '10:51:00')) |
                    ((data['Phone timestamp'] >= '10:56:22') &
                     (data['Phone timestamp'] <= '10:56:29')) |
                    ((data['Phone timestamp'] >= '11:15:00') &
                     (data['Phone timestamp'] <= '11:15:03')) |
                    ((data['Phone timestamp'] >= '11:21:08') &
                     (data['Phone timestamp'] <= '11:21:12')) |
                    ((data['Phone timestamp'] >= '11:27:05') &
                     (data['Phone timestamp'] <= '11:27:08')) |
                    ((data['Phone timestamp'] >= '11:32:26') &
                     (data['Phone timestamp'] <= '11:32:30')) |
                    ((data['Phone timestamp'] >= '11:42:36') &
                     (data['Phone timestamp'] <= '11:42:40')) |
                    ((data['Phone timestamp'] >= '11:57:33') &
                     (data['Phone timestamp'] <= '11:57:36')).values
                )
            ]

        elif number == 2:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:11:03') |
                    ((data['Phone timestamp'] >= '10:15:00') &
                     (data['Phone timestamp'] <= '10:21:00')) |
                    ((data['Phone timestamp'] >= '10:35:43') &
                     (data['Phone timestamp'] <= '10:35:50')) |
                    ((data['Phone timestamp'] >= '10:50:17') &
                     (data['Phone timestamp'] <= '10:50:20')) |
                    ((data['Phone timestamp'] >= '10:50:37') &
                     (data['Phone timestamp'] <= '10:50:44')) |
                    ((data['Phone timestamp'] >= '10:53:25') &
                     (data['Phone timestamp'] <= '10:53:32')) |
                    ((data['Phone timestamp'] >= '11:06:50') &
                     (data['Phone timestamp'] <= '11:06:53')) |
                    ((data['Phone timestamp'] >= '11:23:06') &
                     (data['Phone timestamp'] <= '11:23:09')) |
                    ((data['Phone timestamp'] >= '11:37:04') &
                     (data['Phone timestamp'] <= '11:37:07')) |
                    ((data['Phone timestamp'] >= '11:41:19') &
                     (data['Phone timestamp'] <= '11:41:22')) |
                    ((data['Phone timestamp'] >= '11:53:07') &
                     (data['Phone timestamp'] <= '11:53:09')) |
                    ((data['Phone timestamp'] >= '12:06:39') &
                     (data['Phone timestamp'] <= '12:06:41')).values
                )
            ]

        elif number == 5:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:08:46') |
                    ((data['Phone timestamp'] >= '10:09:33') &
                     (data['Phone timestamp'] <= '10:09:38')) |
                    ((data['Phone timestamp'] >= '10:16:07') &
                     (data['Phone timestamp'] <= '10:16:10')) |
                    ((data['Phone timestamp'] >= '10:20:11') &
                     (data['Phone timestamp'] <= '10:20:13')) |
                    ((data['Phone timestamp'] >= '10:20:22') &
                     (data['Phone timestamp'] <= '10:20:25')) |
                    ((data['Phone timestamp'] >= '10:24:30') &
                     (data['Phone timestamp'] <= '10:24:33')) |
                    ((data['Phone timestamp'] >= '10:47:30') &
                     (data['Phone timestamp'] <= '10:47:33')) |
                    ((data['Phone timestamp'] >= '10:47:47') &
                     (data['Phone timestamp'] <= '10:47:50')) |
                    ((data['Phone timestamp'] >= '12:10:22') &
                     (data['Phone timestamp'] <= '12:10:28')).values
                )
            ]

        elif number == 16:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '13:51:40') |
                    ((data['Phone timestamp'] >= '13:52:25') &
                     (data['Phone timestamp'] <= '13:52:32')) |
                    ((data['Phone timestamp'] >= '14:17:10') &
                     (data['Phone timestamp'] <= '14:17:40')) |
                    ((data['Phone timestamp'] >= '14:18:01') &
                     (data['Phone timestamp'] <= '14:18:30')) |
                    ((data['Phone timestamp'] >= '14:18:47') &
                     (data['Phone timestamp'] <= '14:19:22')) |
                    ((data['Phone timestamp'] >= '14:53:47') &
                     (data['Phone timestamp'] <= '14:53:55')) |
                    ((data['Phone timestamp'] >= '14:54:05') &
                     (data['Phone timestamp'] <= '14:54:11')).values
                )
            ]

        elif number == 18:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:44:32') |
                    ((data['Phone timestamp'] >= '12:09:06') &
                     (data['Phone timestamp'] <= '12:09:08')) |
                    ((data['Phone timestamp'] >= '12:12:18') &
                     (data['Phone timestamp'] <= '12:12:22')) |
                    ((data['Phone timestamp'] >= '12:33:03') &
                     (data['Phone timestamp'] <= '12:33:07')) |
                    ((data['Phone timestamp'] >= '12:59:13') &
                     (data['Phone timestamp'] <= '12:59:19')) |
                    ((data['Phone timestamp'] >= '12:59:27') &
                     (data['Phone timestamp'] <= '12:59:34')) |
                    ((data['Phone timestamp'] >= '13:04:41') &
                     (data['Phone timestamp'] <= '13:04:45')) |
                    ((data['Phone timestamp'] >= '13:18:03') &
                     (data['Phone timestamp'] <= '13:18:05')) |
                    ((data['Phone timestamp'] >= '13:21:37') &
                     (data['Phone timestamp'] <= '13:21:39')).values
                )
            ]

        elif number == 19:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '14:05:34') |
                    ((data['Phone timestamp'] >= '14:06:58') &
                     (data['Phone timestamp'] <= '14:07:00')) |
                    ((data['Phone timestamp'] >= '14:07:09') &
                     (data['Phone timestamp'] <= '14:07:11')) |
                    ((data['Phone timestamp'] >= '14:07:18') &
                     (data['Phone timestamp'] <= '14:07:20')) |
                    ((data['Phone timestamp'] >= '14:11:12') &
                     (data['Phone timestamp'] <= '14:11:13')) |
                    ((data['Phone timestamp'] >= '14:11:43') &
                     (data['Phone timestamp'] <= '14:11:45')) |
                    ((data['Phone timestamp'] >= '14:12:00') &
                     (data['Phone timestamp'] <= '14:12:02')) |
                    ((data['Phone timestamp'] >= '14:12:27') &
                     (data['Phone timestamp'] <= '14:12:29')) |
                    ((data['Phone timestamp'] >= '14:13:04') &
                     (data['Phone timestamp'] <= '14:13:05')) |
                    ((data['Phone timestamp'] >= '14:14:05') &
                     (data['Phone timestamp'] <= '14:14:07')) |
                    ((data['Phone timestamp'] >= '14:14:27') &
                     (data['Phone timestamp'] <= '14:14:28')) |
                    ((data['Phone timestamp'] >= '14:14:38') &
                     (data['Phone timestamp'] <= '14:14:39')) |
                    ((data['Phone timestamp'] >= '14:14:48') &
                     (data['Phone timestamp'] <= '14:14:49')) |
                    ((data['Phone timestamp'] >= '14:15:38') &
                     (data['Phone timestamp'] <= '14:15:42')) |
                    ((data['Phone timestamp'] >= '14:15:47') &
                     (data['Phone timestamp'] <= '14:15:49')) |
                    ((data['Phone timestamp'] >= '14:23:03') &
                     (data['Phone timestamp'] <= '14:23:05')) |
                    ((data['Phone timestamp'] >= '14:23:35') &
                     (data['Phone timestamp'] <= '14:23:36')) |
                    ((data['Phone timestamp'] >= '14:23:48') &
                     (data['Phone timestamp'] <= '14:23:50')) |
                    ((data['Phone timestamp'] >= '14:23:57') &
                     (data['Phone timestamp'] <= '14:23:59')) |
                    ((data['Phone timestamp'] >= '14:24:02') &
                     (data['Phone timestamp'] <= '14:24:03')) |
                    ((data['Phone timestamp'] >= '14:24:19') &
                     (data['Phone timestamp'] <= '14:24:20')) |
                    ((data['Phone timestamp'] >= '14:24:26') &
                     (data['Phone timestamp'] <= '14:24:28')) |
                    ((data['Phone timestamp'] >= '14:25:13') &
                     (data['Phone timestamp'] <= '14:25:14')) |
                    ((data['Phone timestamp'] >= '14:25:17') &
                     (data['Phone timestamp'] <= '14:25:19')) |
                    ((data['Phone timestamp'] >= '14:30:25') &
                     (data['Phone timestamp'] <= '14:30:26')) |
                    ((data['Phone timestamp'] >= '14:30:43') &
                     (data['Phone timestamp'] <= '14:30:45')) |
                    ((data['Phone timestamp'] >= '14:31:09') &
                     (data['Phone timestamp'] <= '14:31:11')) |
                    ((data['Phone timestamp'] >= '14:31:41') &
                     (data['Phone timestamp'] <= '14:31:42')) |
                    ((data['Phone timestamp'] >= '14:32:26') &
                     (data['Phone timestamp'] <= '14:32:28')) |
                    ((data['Phone timestamp'] >= '14:32:38') &
                     (data['Phone timestamp'] <= '14:32:39')) |
                    ((data['Phone timestamp'] >= '14:37:41') &
                     (data['Phone timestamp'] <= '14:37:42')) |
                    ((data['Phone timestamp'] >= '14:38:14') &
                     (data['Phone timestamp'] <= '14:38:15')) |
                    ((data['Phone timestamp'] >= '14:38:36') &
                     (data['Phone timestamp'] <= '14:38:38')) |
                    ((data['Phone timestamp'] >= '14:38:57') &
                     (data['Phone timestamp'] <= '14:38:59')) |
                    ((data['Phone timestamp'] >= '14:39:40') &
                     (data['Phone timestamp'] <= '14:39:42')) |
                    ((data['Phone timestamp'] >= '14:39:46') &
                     (data['Phone timestamp'] <= '14:39:48')) |
                    ((data['Phone timestamp'] >= '14:42:02') &
                     (data['Phone timestamp'] <= '14:42:04')) |
                    ((data['Phone timestamp'] >= '14:43:27') &
                     (data['Phone timestamp'] <= '14:43:29')) |
                    ((data['Phone timestamp'] >= '14:44:09') &
                     (data['Phone timestamp'] <= '14:44:11')) |
                    ((data['Phone timestamp'] >= '14:44:42') &
                     (data['Phone timestamp'] <= '14:44:43')) |
                    ((data['Phone timestamp'] >= '14:48:27') &
                     (data['Phone timestamp'] <= '14:48:29')) |
                    ((data['Phone timestamp'] >= '14:49:18') &
                     (data['Phone timestamp'] <= '14:49:19')) |
                    ((data['Phone timestamp'] >= '14:52:15') &
                     (data['Phone timestamp'] <= '14:52:17')) |
                    ((data['Phone timestamp'] >= '14:52:48') &
                     (data['Phone timestamp'] <= '14:52:50')) |
                    ((data['Phone timestamp'] >= '14:53:24') &
                     (data['Phone timestamp'] <= '14:53:26')) |
                    ((data['Phone timestamp'] >= '14:53:57') &
                     (data['Phone timestamp'] <= '14:53:59')) |
                    ((data['Phone timestamp'] >= '14:56:35') &
                     (data['Phone timestamp'] <= '14:56:36')) |
                    ((data['Phone timestamp'] >= '14:57:27') &
                     (data['Phone timestamp'] <= '14:57:29')) |
                    ((data['Phone timestamp'] >= '14:57:39') &
                     (data['Phone timestamp'] <= '14:57:40')) |
                    ((data['Phone timestamp'] >= '14:58:03') &
                     (data['Phone timestamp'] <= '14:58:07')) |
                    ((data['Phone timestamp'] >= '14:58:42') &
                     (data['Phone timestamp'] <= '14:58:44')) |
                    ((data['Phone timestamp'] >= '15:09:25') &
                     (data['Phone timestamp'] <= '15:09:27')) |
                    ((data['Phone timestamp'] >= '15:09:29') &
                     (data['Phone timestamp'] <= '15:09:30')) |
                    ((data['Phone timestamp'] >= '15:09:55') &
                     (data['Phone timestamp'] <= '15:09:57')) |
                    ((data['Phone timestamp'] >= '15:10:11') &
                     (data['Phone timestamp'] <= '15:10:12')) |
                    ((data['Phone timestamp'] >= '15:11:02') &
                     (data['Phone timestamp'] <= '15:11:03')) |
                    ((data['Phone timestamp'] >= '15:11:14') &
                     (data['Phone timestamp'] <= '15:11:15')) |
                    ((data['Phone timestamp'] >= '15:11:41') &
                     (data['Phone timestamp'] <= '15:11:43')) |
                    ((data['Phone timestamp'] >= '15:12:02') &
                     (data['Phone timestamp'] <= '15:12:03')) |
                    ((data['Phone timestamp'] >= '15:12:08') &
                     (data['Phone timestamp'] <= '15:12:09')) |
                    ((data['Phone timestamp'] >= '15:12:50') &
                     (data['Phone timestamp'] <= '15:12:51')) |
                    ((data['Phone timestamp'] >= '15:13:30') &
                     (data['Phone timestamp'] <= '15:13:31')) |
                    ((data['Phone timestamp'] >= '15:13:59') &
                     (data['Phone timestamp'] <= '15:14:00')) |
                    ((data['Phone timestamp'] >= '15:14:43') &
                     (data['Phone timestamp'] <= '15:14:44')) |
                    ((data['Phone timestamp'] >= '15:15:14') &
                     (data['Phone timestamp'] <= '15:15:16')) |
                    ((data['Phone timestamp'] >= '15:15:53') &
                     (data['Phone timestamp'] <= '15:15:54')) |
                    ((data['Phone timestamp'] >= '15:17:23') &
                     (data['Phone timestamp'] <= '15:17:25')) |
                    ((data['Phone timestamp'] >= '15:17:32') &
                     (data['Phone timestamp'] <= '15:17:33')) |
                    ((data['Phone timestamp'] >= '15:18:41') &
                     (data['Phone timestamp'] <= '15:18:42')) |
                    ((data['Phone timestamp'] >= '15:20:02') &
                     (data['Phone timestamp'] <= '15:20:03')) |
                    ((data['Phone timestamp'] >= '15:20:54') &
                     (data['Phone timestamp'] <= '15:20:55')) |
                    ((data['Phone timestamp'] >= '15:21:17') &
                     (data['Phone timestamp'] <= '15:21:19')) |
                    ((data['Phone timestamp'] >= '15:21:52') &
                     (data['Phone timestamp'] <= '15:21:53')) |
                    ((data['Phone timestamp'] >= '15:22:13') &
                     (data['Phone timestamp'] <= '15:22:14')) |
                    ((data['Phone timestamp'] >= '15:23:35') &
                     (data['Phone timestamp'] <= '15:23:37')) |
                    ((data['Phone timestamp'] >= '15:24:42') &
                     (data['Phone timestamp'] <= '15:24:43')) |
                    ((data['Phone timestamp'] >= '15:25:23') &
                     (data['Phone timestamp'] <= '15:25:24')) |
                    ((data['Phone timestamp'] >= '15:26:00') &
                     (data['Phone timestamp'] <= '15:26:01')) |
                    ((data['Phone timestamp'] >= '15:29:28') &
                     (data['Phone timestamp'] <= '15:29:29')) |
                    ((data['Phone timestamp'] >= '15:33:23') &
                     (data['Phone timestamp'] <= '15:33:24')) |
                    ((data['Phone timestamp'] >= '15:41:04') &
                     (data['Phone timestamp'] <= '15:41:05')) |
                    ((data['Phone timestamp'] >= '15:41:35') &
                     (data['Phone timestamp'] <= '15:41:36')) |
                    ((data['Phone timestamp'] >= '15:42:22') &
                     (data['Phone timestamp'] <= '15:42:23')) |
                    ((data['Phone timestamp'] >= '15:42:50') &
                     (data['Phone timestamp'] <= '15:42:52')) |
                    ((data['Phone timestamp'] >= '15:43:08') &
                     (data['Phone timestamp'] <= '15:43:09')) |
                    ((data['Phone timestamp'] >= '15:43:41') &
                     (data['Phone timestamp'] <= '15:43:42')) |
                    ((data['Phone timestamp'] >= '15:43:51') &
                     (data['Phone timestamp'] <= '15:43:53')) |
                    ((data['Phone timestamp'] >= '15:44:05') &
                     (data['Phone timestamp'] <= '15:44:06')) |
                    ((data['Phone timestamp'] >= '15:44:42') &
                     (data['Phone timestamp'] <= '15:44:44')) |
                    ((data['Phone timestamp'] >= '15:45:29') &
                     (data['Phone timestamp'] <= '15:45:30')) |
                    ((data['Phone timestamp'] >= '15:45:59') &
                     (data['Phone timestamp'] <= '15:46:00')) |
                    ((data['Phone timestamp'] >= '15:46:25') &
                     (data['Phone timestamp'] <= '15:46:26')) |
                    ((data['Phone timestamp'] >= '15:46:44') &
                     (data['Phone timestamp'] <= '15:46:45')) |
                    ((data['Phone timestamp'] >= '15:47:04') &
                     (data['Phone timestamp'] <= '15:47:05')) |
                    ((data['Phone timestamp'] >= '15:47:08') &
                     (data['Phone timestamp'] <= '15:47:10')) |
                    ((data['Phone timestamp'] >= '15:47:26') &
                     (data['Phone timestamp'] <= '15:47:28')) |
                    ((data['Phone timestamp'] >= '15:47:36') &
                     (data['Phone timestamp'] <= '15:47:37')) |
                    ((data['Phone timestamp'] >= '15:47:53') &
                     (data['Phone timestamp'] <= '15:47:55')) |
                    ((data['Phone timestamp'] >= '15:48:20') &
                     (data['Phone timestamp'] <= '15:48:22')) |
                    ((data['Phone timestamp'] >= '15:48:25') &
                     (data['Phone timestamp'] <= '15:48:26')) |
                    ((data['Phone timestamp'] >= '15:48:49') &
                     (data['Phone timestamp'] <= '15:48:51')) |
                    ((data['Phone timestamp'] >= '15:49:45') &
                     (data['Phone timestamp'] <= '15:49:46')).values
                )
            ]

        elif number == 20:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '12:54:30').values
                )
            ]

        elif number == 21:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '13:00:20') |
                    ((data['Phone timestamp'] >= '14:04:21') &
                     (data['Phone timestamp'] <= '14:04:44')) |
                    ((data['Phone timestamp'] >= '14:17:33') &
                     (data['Phone timestamp'] <= '14:17:57')).values
                )
            ]

        elif number == 22:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '15:59:50').values
                )
            ]

        elif number == 24:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '09:44:25') |
                    ((data['Phone timestamp'] >= '09:45:32') &
                     (data['Phone timestamp'] <= '09:45:37')) |
                    ((data['Phone timestamp'] >= '09:46:08') &
                     (data['Phone timestamp'] <= '09:46:12')) |
                    ((data['Phone timestamp'] >= '09:59:38') &
                     (data['Phone timestamp'] <= '09:59:40')) |
                    ((data['Phone timestamp'] >= '10:05:15') &
                     (data['Phone timestamp'] <= '10:05:30')) |
                    ((data['Phone timestamp'] >= '10:06:57') &
                     (data['Phone timestamp'] <= '10:07:02')) |
                    ((data['Phone timestamp'] >= '10:24:00') &
                     (data['Phone timestamp'] <= '10:24:15')) |
                    ((data['Phone timestamp'] >= '10:27:31') &
                     (data['Phone timestamp'] <= '10:27:34')) |
                    ((data['Phone timestamp'] >= '10:27:38') &
                     (data['Phone timestamp'] <= '10:27:41')) |
                    ((data['Phone timestamp'] >= '10:30:20') &
                     (data['Phone timestamp'] <= '10:30:32')) |
                    ((data['Phone timestamp'] >= '10:44:43') &
                     (data['Phone timestamp'] <= '10:44:44')) |
                    ((data['Phone timestamp'] >= '10:58:00') &
                     (data['Phone timestamp'] <= '10:58:08')) |
                    ((data['Phone timestamp'] >= '11:01:21') &
                     (data['Phone timestamp'] <= '11:01:26')).values
                )
            ]

        elif number == 25:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:00:55') |
                    ((data['Phone timestamp'] >= '10:02:30') &
                     (data['Phone timestamp'] <= '10:02:52')) |
                    ((data['Phone timestamp'] >= '10:15:25') &
                     (data['Phone timestamp'] <= '10:15:30')) |
                    ((data['Phone timestamp'] >= '10:35:54') &
                     (data['Phone timestamp'] <= '10:35:58')) |
                    ((data['Phone timestamp'] >= '10:57:28') &
                     (data['Phone timestamp'] <= '10:57:35')) |
                    ((data['Phone timestamp'] >= '11:15:20') &
                     (data['Phone timestamp'] <= '11:15:25')) |
                    ((data['Phone timestamp'] >= '11:29:53') &
                     (data['Phone timestamp'] <= '11:30:00')).values
                )
            ]

        elif number == 26:
            mask = [
                np.flatnonzero(
                    ((data['Phone timestamp'] >= '11:18:00') &
                     (data['Phone timestamp'] <= '11:18:10')) |
                    ((data['Phone timestamp'] >= '11:25:15') &
                     (data['Phone timestamp'] <= '11:25:30')) |
                    ((data['Phone timestamp'] >= '11:25:36') &
                     (data['Phone timestamp'] <= '11:25:42')) |
                    ((data['Phone timestamp'] >= '11:39:54') &
                     (data['Phone timestamp'] <= '11:39:58')) |
                    ((data['Phone timestamp'] >= '11:56:56') &
                     (data['Phone timestamp'] <= '11:57:00')) |
                    ((data['Phone timestamp'] >= '12:03:26') &
                     (data['Phone timestamp'] <= '12:03:32')) |
                    ((data['Phone timestamp'] >= '12:41:50') &
                     (data['Phone timestamp'] <= '12:41:53')).values
                )
            ]

        elif number == 27:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '16:08:08') |
                    ((data['Phone timestamp'] >= '16:10:10') &
                     (data['Phone timestamp'] <= '16:10:15')) |
                    ((data['Phone timestamp'] >= '16:11:23') &
                     (data['Phone timestamp'] <= '16:11:30')) |
                    ((data['Phone timestamp'] >= '16:20:15') &
                     (data['Phone timestamp'] <= '16:20:23')) |
                    ((data['Phone timestamp'] >= '16:23:20') &
                     (data['Phone timestamp'] <= '16:23:25')) |
                    ((data['Phone timestamp'] >= '16:53:40') &
                     (data['Phone timestamp'] <= '16:53:50')) |
                    ((data['Phone timestamp'] >= '16:56:04') &
                     (data['Phone timestamp'] <= '16:56:08')) |
                    ((data['Phone timestamp'] >= '16:56:40') &
                     (data['Phone timestamp'] <= '16:56:43')) |
                    ((data['Phone timestamp'] >= '16:58:34') &
                     (data['Phone timestamp'] <= '16:58:35')) |
                    ((data['Phone timestamp'] >= '17:01:23') &
                     (data['Phone timestamp'] <= '17:01:28')) |
                    ((data['Phone timestamp'] >= '17:05:14') &
                     (data['Phone timestamp'] <= '17:05:20')) |
                    ((data['Phone timestamp'] >= '17:10:33') &
                     (data['Phone timestamp'] <= '17:11:28')) |
                    ((data['Phone timestamp'] >= '17:14:11') &
                     (data['Phone timestamp'] <= '17:14:18')) |
                    ((data['Phone timestamp'] >= '17:25:44') &
                     (data['Phone timestamp'] <= '17:25:50')) |
                    ((data['Phone timestamp'] >= '17:29:00') &
                     (data['Phone timestamp'] <= '17:29:05')).values
                )
            ]
        
        elif number == 28:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:07:50') |
                    ((data['Phone timestamp'] >= '11:12:32') &
                     (data['Phone timestamp'] <= '11:12:35')) |
                    ((data['Phone timestamp'] >= '11:13:00') &
                     (data['Phone timestamp'] <= '11:13:05')) |
                    ((data['Phone timestamp'] >= '11:13:22') &
                     (data['Phone timestamp'] <= '11:13:27')) |
                    ((data['Phone timestamp'] >= '11:18:20') &
                     (data['Phone timestamp'] <= '11:18:25')) |
                    ((data['Phone timestamp'] >= '11:18:46') &
                     (data['Phone timestamp'] <= '11:18:50')) |
                    ((data['Phone timestamp'] >= '11:19:35') &
                     (data['Phone timestamp'] <= '11:19:40')) |
                    ((data['Phone timestamp'] >= '11:30:48') &
                     (data['Phone timestamp'] <= '11:30:55')) |
                    ((data['Phone timestamp'] >= '11:42:41') &
                     (data['Phone timestamp'] <= '11:42:45')) |
                    ((data['Phone timestamp'] >= '11:43:35') &
                     (data['Phone timestamp'] <= '11:43:42')) |
                    ((data['Phone timestamp'] >= '11:46:52') &
                     (data['Phone timestamp'] <= '11:47:15')) |
                    ((data['Phone timestamp'] >= '12:19:13') &
                     (data['Phone timestamp'] <= '12:19:18')) |
                    ((data['Phone timestamp'] >= '12:22:52') &
                     (data['Phone timestamp'] <= '12:22:57')) |
                    ((data['Phone timestamp'] >= '12:23:55') &
                     (data['Phone timestamp'] <= '12:24:00')) |
                    ((data['Phone timestamp'] >= '12:24:25') &
                     (data['Phone timestamp'] <= '12:24:30')) |
                    ((data['Phone timestamp'] >= '12:24:42') &
                     (data['Phone timestamp'] <= '12:24:47')) |
                    ((data['Phone timestamp'] >= '12:25:35') &
                     (data['Phone timestamp'] <= '12:25:40')) |
                    ((data['Phone timestamp'] >= '12:29:52') &
                     (data['Phone timestamp'] <= '12:29:58')) |
                    ((data['Phone timestamp'] >= '12:30:20') &
                     (data['Phone timestamp'] <= '12:30:40')) |
                    ((data['Phone timestamp'] >= '12:32:20') &
                     (data['Phone timestamp'] <= '12:32:25')) |
                    ((data['Phone timestamp'] >= '12:32:52') &
                     (data['Phone timestamp'] <= '12:33:10')) |
                    ((data['Phone timestamp'] >= '12:38:04') &
                     (data['Phone timestamp'] <= '12:38:10')).values
                )
            ]

        elif number == 29:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '13:59:22') |
                    ((data['Phone timestamp'] >= '13:59:38') &
                     (data['Phone timestamp'] <= '13:59:44')) |
                    ((data['Phone timestamp'] >= '13:59:50') &
                     (data['Phone timestamp'] <= '14:00:00')) |
                    ((data['Phone timestamp'] >= '14:00:13') &
                     (data['Phone timestamp'] <= '14:00:38')) |
                    ((data['Phone timestamp'] >= '14:01:10') &
                     (data['Phone timestamp'] <= '14:01:20')) |
                    ((data['Phone timestamp'] >= '14:22:57') &
                     (data['Phone timestamp'] <= '14:23:05')) |
                    ((data['Phone timestamp'] >= '14:34:06') &
                     (data['Phone timestamp'] <= '14:34:09')) |
                    ((data['Phone timestamp'] >= '14:44:37') &
                     (data['Phone timestamp'] <= '14:44:40')) |
                    ((data['Phone timestamp'] >= '14:54:20') &
                     (data['Phone timestamp'] <= '14:54:43')) |
                    ((data['Phone timestamp'] >= '14:56:32') &
                     (data['Phone timestamp'] <= '14:56:40')) |
                    ((data['Phone timestamp'] >= '15:03:11') &
                     (data['Phone timestamp'] <= '15:03:13')) |
                    ((data['Phone timestamp'] >= '15:16:28') &
                     (data['Phone timestamp'] <= '15:16:32')) |
                    ((data['Phone timestamp'] >= '15:24:52') &
                     (data['Phone timestamp'] <= '15:24:55')).values
                )
            ]

        elif number == 30:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '14:11:45') |
                    ((data['Phone timestamp'] >= '14:11:55') &
                     (data['Phone timestamp'] <= '14:12:01')) |
                    ((data['Phone timestamp'] >= '14:12:28') &
                     (data['Phone timestamp'] <= '14:12:30')) |
                    ((data['Phone timestamp'] >= '14:14:33') &
                     (data['Phone timestamp'] <= '14:14:35')) |
                    ((data['Phone timestamp'] >= '14:32:03') &
                     (data['Phone timestamp'] <= '14:32:10')) |
                    ((data['Phone timestamp'] >= '14:44:04') &
                     (data['Phone timestamp'] <= '14:44:12')) |
                    ((data['Phone timestamp'] >= '14:58:34') &
                     (data['Phone timestamp'] <= '14:58:39')) |
                    ((data['Phone timestamp'] >= '15:18:50') &
                     (data['Phone timestamp'] <= '15:19:08')) |
                    ((data['Phone timestamp'] >= '15:30:25') &
                     (data['Phone timestamp'] <= '15:30:30')).values
                )
            ]

        elif number == 31:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:13:46') |
                    ((data['Phone timestamp'] >= '11:23:00') &
                     (data['Phone timestamp'] <= '11:26:41')) |
                    ((data['Phone timestamp'] >= '11:27:05') &
                     (data['Phone timestamp'] <= '11:27:18')) |
                    (data['Phone timestamp'] >= '12:27:47').values
                )
            ]

        elif number == 32:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:06:25') |
                    ((data['Phone timestamp'] >= '11:07:55') &
                     (data['Phone timestamp'] <= '11:08:00')) |
                    ((data['Phone timestamp'] >= '11:11:03') &
                     (data['Phone timestamp'] <= '11:11:16')) |
                    ((data['Phone timestamp'] >= '11:14:33') &
                     (data['Phone timestamp'] <= '11:14:49')) |
                    ((data['Phone timestamp'] >= '11:17:10') &
                     (data['Phone timestamp'] <= '11:17:20')) |
                    ((data['Phone timestamp'] >= '11:17:30') &
                     (data['Phone timestamp'] <= '11:17:42')) |
                    ((data['Phone timestamp'] >= '11:17:52') &
                     (data['Phone timestamp'] <= '11:18:10')) |
                    ((data['Phone timestamp'] >= '11:19:10') &
                     (data['Phone timestamp'] <= '11:19:45')) |
                    ((data['Phone timestamp'] >= '11:20:05') &
                     (data['Phone timestamp'] <= '11:20:15')) |
                    ((data['Phone timestamp'] >= '11:20:52') &
                     (data['Phone timestamp'] <= '11:20:57')) |
                    ((data['Phone timestamp'] >= '11:21:00') &
                     (data['Phone timestamp'] <= '11:21:08')) |
                    ((data['Phone timestamp'] >= '11:21:13') &
                     (data['Phone timestamp'] <= '11:21:18')) |
                    ((data['Phone timestamp'] >= '11:21:22') &
                     (data['Phone timestamp'] <= '11:21:25')) |
                    ((data['Phone timestamp'] >= '11:21:32') &
                     (data['Phone timestamp'] <= '11:21:35')) |
                    ((data['Phone timestamp'] >= '11:22:00') &
                     (data['Phone timestamp'] <= '11:22:04')) |
                    ((data['Phone timestamp'] >= '11:22:13') &
                     (data['Phone timestamp'] <= '11:22:15')) |
                    ((data['Phone timestamp'] >= '11:25:38') &
                     (data['Phone timestamp'] <= '11:25:45')) |
                    ((data['Phone timestamp'] >= '11:25:52') &
                     (data['Phone timestamp'] <= '11:25:57')) |
                    ((data['Phone timestamp'] >= '11:26:15') &
                     (data['Phone timestamp'] <= '11:26:22')) |
                    ((data['Phone timestamp'] >= '11:26:57') &
                     (data['Phone timestamp'] <= '11:27:00')) |
                    ((data['Phone timestamp'] >= '11:42:28') &
                     (data['Phone timestamp'] <= '11:42:32')) |
                    ((data['Phone timestamp'] >= '11:54:31') &
                     (data['Phone timestamp'] <= '11:54:35')) |
                    ((data['Phone timestamp'] >= '12:14:30') &
                     (data['Phone timestamp'] <= '12:14:40')) |
                    ((data['Phone timestamp'] >= '12:19:38') &
                     (data['Phone timestamp'] <= '12:19:42')) |
                    (data['Phone timestamp'] >= '12:25:38').values
                )
            ]

        elif number == 33:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:11:20') |
                    ((data['Phone timestamp'] >= '11:11:36') &
                     (data['Phone timestamp'] <= '11:11:38')) |
                    ((data['Phone timestamp'] >= '11:11:46') &
                     (data['Phone timestamp'] <= '11:11:49')) |
                    ((data['Phone timestamp'] >= '11:11:57') &
                     (data['Phone timestamp'] <= '11:12:00')) |
                    ((data['Phone timestamp'] >= '11:13:17') &
                     (data['Phone timestamp'] <= '11:13:24')) |
                    ((data['Phone timestamp'] >= '11:13:38') &
                     (data['Phone timestamp'] <= '11:13:42')) |
                    ((data['Phone timestamp'] >= '11:13:45') &
                     (data['Phone timestamp'] <= '11:14:00')) |
                    ((data['Phone timestamp'] >= '11:14:30') &
                     (data['Phone timestamp'] <= '11:14:50')) |
                    ((data['Phone timestamp'] >= '11:15:08') &
                     (data['Phone timestamp'] <= '11:15:12')) |
                    ((data['Phone timestamp'] >= '11:15:15') &
                     (data['Phone timestamp'] <= '11:15:22')) |
                    ((data['Phone timestamp'] >= '11:15:50') &
                     (data['Phone timestamp'] <= '11:16:00')) |
                    ((data['Phone timestamp'] >= '11:16:09') &
                     (data['Phone timestamp'] <= '11:16:35')) |
                    ((data['Phone timestamp'] >= '11:18:35') &
                     (data['Phone timestamp'] <= '11:18:45')) |
                    ((data['Phone timestamp'] >= '11:18:50') &
                     (data['Phone timestamp'] <= '11:18:54')) |
                    ((data['Phone timestamp'] >= '11:19:20') &
                     (data['Phone timestamp'] <= '11:19:30')) |
                    ((data['Phone timestamp'] >= '11:19:45') &
                     (data['Phone timestamp'] <= '11:19:52')) |
                    ((data['Phone timestamp'] >= '11:21:05') &
                     (data['Phone timestamp'] <= '11:21:25')) |
                    ((data['Phone timestamp'] >= '11:21:42') &
                     (data['Phone timestamp'] <= '11:22:07')) |
                    ((data['Phone timestamp'] >= '11:22:50') &
                     (data['Phone timestamp'] <= '11:22:55')) |
                    ((data['Phone timestamp'] >= '11:26:25') &
                     (data['Phone timestamp'] <= '11:26:35')) |
                    ((data['Phone timestamp'] >= '11:27:35') &
                     (data['Phone timestamp'] <= '11:27:41')) |
                    ((data['Phone timestamp'] >= '11:27:52') &
                     (data['Phone timestamp'] <= '11:27:57')) |
                    ((data['Phone timestamp'] >= '11:33:00') &
                     (data['Phone timestamp'] <= '11:33:22')) |
                    ((data['Phone timestamp'] >= '11:35:28') &
                     (data['Phone timestamp'] <= '11:35:52')) |
                    ((data['Phone timestamp'] >= '11:40:52') &
                     (data['Phone timestamp'] <= '11:40:55')) |
                    ((data['Phone timestamp'] >= '11:47:00') &
                     (data['Phone timestamp'] <= '11:47:12')) |
                    ((data['Phone timestamp'] >= '11:49:20') &
                     (data['Phone timestamp'] <= '11:49:30')) |
                    ((data['Phone timestamp'] >= '11:50:10') &
                     (data['Phone timestamp'] <= '11:50:40')) |
                    ((data['Phone timestamp'] >= '11:53:10') &
                     (data['Phone timestamp'] <= '11:53:17')) |
                    ((data['Phone timestamp'] >= '11:53:50') &
                     (data['Phone timestamp'] <= '11:54:00')) |
                    ((data['Phone timestamp'] >= '12:00:40') &
                     (data['Phone timestamp'] <= '12:00:50')) |
                    ((data['Phone timestamp'] >= '12:11:44') &
                     (data['Phone timestamp'] <= '12:11:49')) |
                    ((data['Phone timestamp'] >= '12:19:37') &
                     (data['Phone timestamp'] <= '12:19:50')).values
                )
            ]

        elif number == 34:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:17:15') |
                    ((data['Phone timestamp'] >= '10:17:38') &
                     (data['Phone timestamp'] <= '10:17:42')) |
                    ((data['Phone timestamp'] >= '10:18:22') &
                     (data['Phone timestamp'] <= '10:18:30')) |
                    ((data['Phone timestamp'] >= '10:18:40') &
                     (data['Phone timestamp'] <= '10:18:43')) |
                    ((data['Phone timestamp'] >= '10:19:38') &
                     (data['Phone timestamp'] <= '10:19:41')) |
                    ((data['Phone timestamp'] >= '10:20:00') &
                     (data['Phone timestamp'] <= '10:20:28')) |
                    ((data['Phone timestamp'] >= '10:32:25') &
                     (data['Phone timestamp'] <= '10:32:28')) |
                    ((data['Phone timestamp'] >= '10:44:24') &
                     (data['Phone timestamp'] <= '10:44:28')) |
                    ((data['Phone timestamp'] >= '11:26:47') &
                     (data['Phone timestamp'] <= '11:26:50')) |
                    (data['Phone timestamp'] >= '11:40:28').values
                )
            ]

        elif number == 35:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:24:12') |
                    ((data['Phone timestamp'] >= '11:24:54') &
                     (data['Phone timestamp'] <= '11:25:05')) |
                    ((data['Phone timestamp'] >= '11:25:17') &
                     (data['Phone timestamp'] <= '11:25:25')) |
                    ((data['Phone timestamp'] >= '11:27:46') &
                     (data['Phone timestamp'] <= '11:27:49')) |
                    ((data['Phone timestamp'] >= '11:28:22') &
                     (data['Phone timestamp'] <= '11:28:25')) |
                    ((data['Phone timestamp'] >= '11:33:54') &
                     (data['Phone timestamp'] <= '11:34:00')) |
                    ((data['Phone timestamp'] >= '11:36:30') &
                     (data['Phone timestamp'] <= '11:36:37')) |
                    ((data['Phone timestamp'] >= '12:36:40') &
                     (data['Phone timestamp'] <= '12:36:42')) |
                    (data['Phone timestamp'] >= '12:48:48').values
                )
            ]

        elif number == 36:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '11:22:45') |
                    ((data['Phone timestamp'] >= '11:24:13') &
                     (data['Phone timestamp'] <= '11:24:16')) |
                    ((data['Phone timestamp'] >= '11:38:27') &
                     (data['Phone timestamp'] <= '11:38:30')) |
                    ((data['Phone timestamp'] >= '12:20:50') &
                     (data['Phone timestamp'] <= '12:20:55')) |
                    ((data['Phone timestamp'] >= '12:09:47') &
                     (data['Phone timestamp'] <= '12:09:50')) |
                    ((data['Phone timestamp'] >= '12:35:12') &
                     (data['Phone timestamp'] <= '12:35:16')).values
                )
            ]

        elif number == 37:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '12:23:33') |
                    ((data['Phone timestamp'] >= '12:27:19') &
                     (data['Phone timestamp'] <= '12:27:28')) |
                    ((data['Phone timestamp'] >= '13:16:47') &
                     (data['Phone timestamp'] <= '13:16:50')) |
                    ((data['Phone timestamp'] >= '13:28:42') &
                     (data['Phone timestamp'] <= '13:28:47')) |
                    ((data['Phone timestamp'] >= '13:32:57') &
                     (data['Phone timestamp'] <= '13:33:01')) |
                    ((data['Phone timestamp'] >= '13:39:23') &
                     (data['Phone timestamp'] <= '13:39:27')) |
                    ((data['Phone timestamp'] >= '13:42:13') &
                     (data['Phone timestamp'] <= '13:42:16')) |
                    ((data['Phone timestamp'] >= '13:53:12') &
                     (data['Phone timestamp'] <= '13:53:22')) |
                    ((data['Phone timestamp'] >= '13:54:00') &
                     (data['Phone timestamp'] <= '13:54:10')) |
                    ((data['Phone timestamp'] >= '13:54:30') &
                     (data['Phone timestamp'] <= '13:54:35')).values
                )
            ]

        elif number == 38:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '09:58:00') |
                    ((data['Phone timestamp'] >= '10:01:24') &
                     (data['Phone timestamp'] <= '10:01:28')) |
                    ((data['Phone timestamp'] >= '10:12:48') &
                     (data['Phone timestamp'] <= '10:12:55')) |
                    ((data['Phone timestamp'] >= '10:16:45') &
                     (data['Phone timestamp'] <= '10:16:48')) |
                    ((data['Phone timestamp'] >= '10:21:17') &
                     (data['Phone timestamp'] <= '10:21:20')) |
                    ((data['Phone timestamp'] >= '10:32:34') &
                     (data['Phone timestamp'] <= '10:32:39')) |
                    ((data['Phone timestamp'] >= '10:32:41') &
                     (data['Phone timestamp'] <= '10:32:44')) |
                    ((data['Phone timestamp'] >= '10:34:40') &
                     (data['Phone timestamp'] <= '10:34:43')) |
                    ((data['Phone timestamp'] >= '10:39:15') &
                     (data['Phone timestamp'] <= '10:39:24')) |
                    ((data['Phone timestamp'] >= '11:32:06') &
                     (data['Phone timestamp'] <= '11:32:09')) |
                    ((data['Phone timestamp'] >= '11:43:37') &
                     (data['Phone timestamp'] <= '11:43:39')) |
                    (data['Phone timestamp'] >= '12:08:24').values
                )
            ]

        elif number == 39:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:04:00') |
                    ((data['Phone timestamp'] >= '10:08:55') &
                     (data['Phone timestamp'] <= '10:09:10')) |
                    ((data['Phone timestamp'] >= '10:14:10') &
                     (data['Phone timestamp'] <= '10:14:20')) |
                    ((data['Phone timestamp'] >= '10:17:45') &
                     (data['Phone timestamp'] <= '10:18:00')) |
                    ((data['Phone timestamp'] >= '10:22:00') &
                     (data['Phone timestamp'] <= '10:22:15')) |
                    ((data['Phone timestamp'] >= '10:22:30') &
                     (data['Phone timestamp'] <= '10:23:00')) |
                    ((data['Phone timestamp'] >= '10:25:50') &
                     (data['Phone timestamp'] <= '10:26:15')) |
                    ((data['Phone timestamp'] >= '10:34:24') &
                     (data['Phone timestamp'] <= '10:35:15')) |
                    ((data['Phone timestamp'] >= '10:40:00') &
                     (data['Phone timestamp'] <= '10:40:20')) |
                    ((data['Phone timestamp'] >= '10:51:55') &
                     (data['Phone timestamp'] <= '10:52:15')) |
                    ((data['Phone timestamp'] >= '10:52:30') &
                     (data['Phone timestamp'] <= '10:53:30')) |
                    ((data['Phone timestamp'] >= '10:58:40') &
                     (data['Phone timestamp'] <= '10:58:43')) |
                    ((data['Phone timestamp'] >= '11:01:00') &
                     (data['Phone timestamp'] <= '11:01:08')) |
                    ((data['Phone timestamp'] >= '11:01:23') &
                     (data['Phone timestamp'] <= '11:01:33')) |
                    ((data['Phone timestamp'] >= '11:02:03') &
                     (data['Phone timestamp'] <= '11:02:08')) |
                    ((data['Phone timestamp'] >= '11:06:06') &
                     (data['Phone timestamp'] <= '11:06:09')) |
                    ((data['Phone timestamp'] >= '11:09:50') &
                     (data['Phone timestamp'] <= '11:09:54')) |
                    ((data['Phone timestamp'] >= '11:21:50') &
                     (data['Phone timestamp'] <= '11:22:30')) |
                    ((data['Phone timestamp'] >= '11:23:45') &
                     (data['Phone timestamp'] <= '11:24:20')) |
                    ((data['Phone timestamp'] >= '11:24:45') &
                     (data['Phone timestamp'] <= '11:24:55')) |
                    ((data['Phone timestamp'] >= '11:25:50') &
                     (data['Phone timestamp'] <= '11:26:00')) |
                    ((data['Phone timestamp'] >= '11:27:35') &
                     (data['Phone timestamp'] <= '11:27:45')) |
                    ((data['Phone timestamp'] >= '11:34:30') &
                     (data['Phone timestamp'] <= '11:34:55')) |
                    ((data['Phone timestamp'] >= '11:38:35') &
                     (data['Phone timestamp'] <= '11:38:50')) |
                    ((data['Phone timestamp'] >= '11:55:00') &
                     (data['Phone timestamp'] <= '11:55:40')) |
                    ((data['Phone timestamp'] >= '11:58:00') &
                     (data['Phone timestamp'] <= '11:58:15')).values
                )
            ]

        elif number == 40:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '09:20:50') |
                    ((data['Phone timestamp'] >= '09:32:20') &
                     (data['Phone timestamp'] <= '09:32:30')) |
                    ((data['Phone timestamp'] >= '09:40:20') &
                     (data['Phone timestamp'] <= '09:40:40')) |
                    ((data['Phone timestamp'] >= '09:51:15') &
                     (data['Phone timestamp'] <= '09:51:40')) |
                    ((data['Phone timestamp'] >= '09:54:07') &
                     (data['Phone timestamp'] <= '09:54:10')) |
                    ((data['Phone timestamp'] >= '09:57:17') &
                     (data['Phone timestamp'] <= '09:57:19')) |
                    ((data['Phone timestamp'] >= '09:57:54') &
                     (data['Phone timestamp'] <= '09:57:55')) |
                    ((data['Phone timestamp'] >= '09:59:55') &
                     (data['Phone timestamp'] <= '10:00:15')) |
                    ((data['Phone timestamp'] >= '10:15:33') &
                     (data['Phone timestamp'] <= '10:15:42')) |
                    ((data['Phone timestamp'] >= '10:36:15') &
                     (data['Phone timestamp'] <= '10:36:28')) |
                    ((data['Phone timestamp'] >= '10:44:42') &
                     (data['Phone timestamp'] <= '10:44:43')) |
                    ((data['Phone timestamp'] >= '10:47:18') &
                     (data['Phone timestamp'] <= '10:47:21')) |
                    ((data['Phone timestamp'] >= '10:49:48') &
                     (data['Phone timestamp'] <= '10:49:49')) |
                    ((data['Phone timestamp'] >= '10:54:51') &
                     (data['Phone timestamp'] <= '10:54:53')) |
                    ((data['Phone timestamp'] >= '10:56:13') &
                     (data['Phone timestamp'] <= '10:56:14')) |
                    ((data['Phone timestamp'] >= '10:57:50') &
                     (data['Phone timestamp'] <= '10:57:57')) |
                    ((data['Phone timestamp'] >= '10:58:07') &
                     (data['Phone timestamp'] <= '10:58:13')) |
                    ((data['Phone timestamp'] >= '11:17:58') &
                     (data['Phone timestamp'] <= '11:18:06')) |
                    ((data['Phone timestamp'] >= '11:34:54') &
                     (data['Phone timestamp'] <= '11:34:59')) |
                    ((data['Phone timestamp'] >= '11:44:25') &
                     (data['Phone timestamp'] <= '11:44:30')) |
                    ((data['Phone timestamp'] >= '11:44:46') &
                     (data['Phone timestamp'] <= '11:44:51')) |
                    ((data['Phone timestamp'] >= '11:46:46') &
                     (data['Phone timestamp'] <= '11:46:48')).values
                )
            ]

        elif number == 41:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '10:19:08') |
                    ((data['Phone timestamp'] >= '10:43:01') &
                     (data['Phone timestamp'] <= '10:43:04')) |
                    ((data['Phone timestamp'] >= '10:43:11') &
                     (data['Phone timestamp'] <= '10:43:13')) |
                    ((data['Phone timestamp'] >= '10:50:10') &
                     (data['Phone timestamp'] <= '10:50:13')) |
                    ((data['Phone timestamp'] >= '10:54:59') &
                     (data['Phone timestamp'] <= '10:55:01')) |
                    ((data['Phone timestamp'] >= '10:57:00') &
                     (data['Phone timestamp'] <= '10:57:07')) |
                    ((data['Phone timestamp'] >= '10:57:14') &
                     (data['Phone timestamp'] <= '10:57:16')) |
                    ((data['Phone timestamp'] >= '11:21:21') &
                     (data['Phone timestamp'] <= '11:21:24')) |
                    ((data['Phone timestamp'] >= '11:27:30') &
                     (data['Phone timestamp'] <= '11:27:33')) |
                    ((data['Phone timestamp'] >= '11:29:11') &
                     (data['Phone timestamp'] <= '11:29:16')) |
                    ((data['Phone timestamp'] >= '11:42:25') &
                     (data['Phone timestamp'] <= '11:42:28')) |
                    ((data['Phone timestamp'] >= '11:54:07') &
                     (data['Phone timestamp'] <= '11:54:08')) |
                    ((data['Phone timestamp'] >= '11:54:26') &
                     (data['Phone timestamp'] <= '11:54:28')) |
                    ((data['Phone timestamp'] >= '11:58:43') &
                     (data['Phone timestamp'] <= '11:58:46')) |
                    ((data['Phone timestamp'] >= '12:04:42') &
                     (data['Phone timestamp'] <= '12:04:45')) |
                    ((data['Phone timestamp'] >= '12:07:55') &
                     (data['Phone timestamp'] <= '12:07:59')) |
                    ((data['Phone timestamp'] >= '12:08:33') &
                     (data['Phone timestamp'] <= '12:08:36')).values
                )
            ]

        elif number == 42:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '09:03:15') |
                    ((data['Phone timestamp'] >= '09:07:03') &
                     (data['Phone timestamp'] <= '09:07:35')) |
                    ((data['Phone timestamp'] >= '09:11:22') &
                     (data['Phone timestamp'] <= '09:11:29')) |
                    ((data['Phone timestamp'] >= '09:18:33') &
                     (data['Phone timestamp'] <= '09:18:39')) |
                    ((data['Phone timestamp'] >= '09:20:27') &
                     (data['Phone timestamp'] <= '09:20:30')) |
                    ((data['Phone timestamp'] >= '09:23:54') &
                     (data['Phone timestamp'] <= '09:24:00')) |
                    ((data['Phone timestamp'] >= '09:36:44') &
                     (data['Phone timestamp'] <= '09:36:50')) |
                    ((data['Phone timestamp'] >= '09:39:29') &
                     (data['Phone timestamp'] <= '09:39:33')) |
                    ((data['Phone timestamp'] >= '09:47:00') &
                     (data['Phone timestamp'] <= '09:47:50')) |
                    ((data['Phone timestamp'] >= '09:49:10') &
                     (data['Phone timestamp'] <= '09:49:30')) |
                    ((data['Phone timestamp'] >= '09:50:08') &
                     (data['Phone timestamp'] <= '09:50:15')) |
                    ((data['Phone timestamp'] >= '09:50:30') &
                     (data['Phone timestamp'] <= '09:50:35')) |
                    ((data['Phone timestamp'] >= '09:50:45') &
                     (data['Phone timestamp'] <= '09:50:53')) |
                    ((data['Phone timestamp'] >= '10:03:35') &
                     (data['Phone timestamp'] <= '10:03:46')) |
                    ((data['Phone timestamp'] >= '10:03:58') &
                     (data['Phone timestamp'] <= '10:04:02')) |
                    ((data['Phone timestamp'] >= '10:08:35') &
                     (data['Phone timestamp'] <= '10:08:40')) |
                    ((data['Phone timestamp'] >= '10:09:33') &
                     (data['Phone timestamp'] <= '10:09:39')) |
                    ((data['Phone timestamp'] >= '10:13:05') &
                     (data['Phone timestamp'] <= '10:13:10')) |
                    ((data['Phone timestamp'] >= '10:13:40') &
                     (data['Phone timestamp'] <= '10:13:45')) |
                    ((data['Phone timestamp'] >= '10:13:55') &
                     (data['Phone timestamp'] <= '10:14:00')) |
                    ((data['Phone timestamp'] >= '10:18:25') &
                     (data['Phone timestamp'] <= '10:18:30')) |
                    ((data['Phone timestamp'] >= '10:21:45') &
                     (data['Phone timestamp'] <= '10:21:52')) |
                    ((data['Phone timestamp'] >= '10:29:45') &
                     (data['Phone timestamp'] <= '10:29:46')) |
                    ((data['Phone timestamp'] >= '10:30:20') &
                     (data['Phone timestamp'] <= '10:30:32')) |
                    ((data['Phone timestamp'] >= '10:39:03') &
                     (data['Phone timestamp'] <= '10:39:07')) |
                    ((data['Phone timestamp'] >= '10:39:38') &
                     (data['Phone timestamp'] <= '10:39:42')) |
                    ((data['Phone timestamp'] >= '10:41:11') &
                     (data['Phone timestamp'] <= '10:41:19')) |
                    ((data['Phone timestamp'] >= '10:48:15') &
                     (data['Phone timestamp'] <= '10:48:20')) |
                    ((data['Phone timestamp'] >= '10:50:10') &
                     (data['Phone timestamp'] <= '10:50:20')) |
                    ((data['Phone timestamp'] >= '10:50:40') &
                     (data['Phone timestamp'] <= '10:50:52')) |
                    ((data['Phone timestamp'] >= '10:51:35') &
                     (data['Phone timestamp'] <= '10:51:40')) |
                    ((data['Phone timestamp'] >= '10:51:52') &
                     (data['Phone timestamp'] <= '10:52:00')) |
                    (data['Phone timestamp'] >= '11:05:00').values
                )
            ]

        elif number == 43:
            mask = [
                np.flatnonzero(
                    (data['Phone timestamp'] <= '15:12:00') |
                    ((data['Phone timestamp'] >= '16:32:04') &
                     (data['Phone timestamp'] <= '16:32:08')) |
                    ((data['Phone timestamp'] >= '17:00:03') &
                     (data['Phone timestamp'] <= '17:07:08')) |
                    (data['Phone timestamp'] >= '17:12:00').values
                )
            ]

        elif number == 44:
            mask = [
                np.flatnonzero(
                   (data['Phone timestamp'] <= '15:07:00') |
                   ((data['Phone timestamp'] >= '15:09:27') &
                    (data['Phone timestamp'] <= '15:09:32')) |
                   ((data['Phone timestamp'] >= '15:09:37') &
                    (data['Phone timestamp'] <= '15:09:40')) |
                   ((data['Phone timestamp'] >= '15:22:17') &
                    (data['Phone timestamp'] <= '15:22:20')) |
                   ((data['Phone timestamp'] >= '15:31:40') &
                    (data['Phone timestamp'] <= '15:32:00')) |
                   ((data['Phone timestamp'] >= '15:37:10') &
                    (data['Phone timestamp'] <= '15:37:28')) |
                   ((data['Phone timestamp'] >= '15:38:00') &
                    (data['Phone timestamp'] <= '15:38:08')) |
                   ((data['Phone timestamp'] >= '15:39:52') &
                    (data['Phone timestamp'] <= '15:39:58')) |
                   ((data['Phone timestamp'] >= '15:44:10') &
                    (data['Phone timestamp'] <= '15:44:28')) |
                   ((data['Phone timestamp'] >= '16:00:03') &
                    (data['Phone timestamp'] <= '16:00:08')) |
                   ((data['Phone timestamp'] >= '16:00:35') &
                    (data['Phone timestamp'] <= '16:00:38')) |
                   ((data['Phone timestamp'] >= '16:06:45') &
                    (data['Phone timestamp'] <= '16:06:50')) |
                   ((data['Phone timestamp'] >= '16:07:14') &
                    (data['Phone timestamp'] <= '16:07:17')) |
                   ((data['Phone timestamp'] >= '16:07:30') &
                    (data['Phone timestamp'] <= '16:07:33')) |
                   ((data['Phone timestamp'] >= '16:13:22') &
                    (data['Phone timestamp'] <= '16:13:25')) |
                   ((data['Phone timestamp'] >= '16:15:27') &
                    (data['Phone timestamp'] <= '16:15:30')) |
                   ((data['Phone timestamp'] >= '16:15:49') &
                    (data['Phone timestamp'] <= '16:15:52')) |
                   ((data['Phone timestamp'] >= '16:26:32') &
                    (data['Phone timestamp'] <= '16:26:35')) |
                   ((data['Phone timestamp'] >= '16:26:44') &
                    (data['Phone timestamp'] <= '16:26:48')) |
                   ((data['Phone timestamp'] >= '16:31:25') &
                    (data['Phone timestamp'] <= '16:31:32')) |
                   ((data['Phone timestamp'] >= '16:32:55') &
                    (data['Phone timestamp'] <= '16:33:00')) |
                   ((data['Phone timestamp'] >= '16:33:17') &
                    (data['Phone timestamp'] <= '16:33:34')) |
                   ((data['Phone timestamp'] >= '16:39:04') &
                    (data['Phone timestamp'] <= '16:39:08')) |
                   ((data['Phone timestamp'] >= '16:43:04') &
                    (data['Phone timestamp'] <= '16:43:06')) |
                   ((data['Phone timestamp'] >= '16:46:23') &
                    (data['Phone timestamp'] <= '16:46:55')) |
                   ((data['Phone timestamp'] >= '16:53:00') &
                    (data['Phone timestamp'] <= '16:53:05')) |
                   ((data['Phone timestamp'] >= '16:55:51') &
                    (data['Phone timestamp'] <= '16:55:53')) |
                   ((data['Phone timestamp'] >= '16:59:55') &
                    (data['Phone timestamp'] <= '17:00:20')) |
                   ((data['Phone timestamp'] >= '17:04:10') &
                    (data['Phone timestamp'] <= '17:04:45')) |
                   ((data['Phone timestamp'] >= '17:05:13') &
                    (data['Phone timestamp'] <= '17:05:22')) |
                   ((data['Phone timestamp'] >= '17:06:38') &
                    (data['Phone timestamp'] <= '17:06:42')) |
                   (data['Phone timestamp'] >= '17:12:00').values
                )
            ]

        elif number == 45:
            mask = [
                np.flatnonzero(
                   (data['Phone timestamp'] <= '11:12:56') |
                   ((data['Phone timestamp'] >= '11:25:40') &
                    (data['Phone timestamp'] <= '11:25:50')) |
                   ((data['Phone timestamp'] >= '12:11:11') &
                    (data['Phone timestamp'] <= '12:11:14')) |
                   ((data['Phone timestamp'] >= '12:11:31') &
                    (data['Phone timestamp'] <= '12:11:36')) |
                   ((data['Phone timestamp'] >= '12:11:55') &
                    (data['Phone timestamp'] <= '12:11:58')) |
                   ((data['Phone timestamp'] >= '12:12:03') &
                    (data['Phone timestamp'] <= '12:12:08')) |
                   ((data['Phone timestamp'] >= '12:13:52') &
                    (data['Phone timestamp'] <= '12:14:03')) |
                   ((data['Phone timestamp'] >= '12:17:40') &
                    (data['Phone timestamp'] <= '12:18:06')).values
                )
            ]

        elif number == 46:
            mask = [
                np.flatnonzero(
                   (data['Phone timestamp'] <= '11:08:40') |
                   ((data['Phone timestamp'] >= '11:09:15') &
                    (data['Phone timestamp'] <= '11:09:50')) |
                   ((data['Phone timestamp'] >= '11:14:12') &
                    (data['Phone timestamp'] <= '11:14:16')) |
                   ((data['Phone timestamp'] >= '11:18:37') &
                    (data['Phone timestamp'] <= '11:18:53')) |
                   ((data['Phone timestamp'] >= '11:22:29') &
                    (data['Phone timestamp'] <= '11:22:32')) |
                   ((data['Phone timestamp'] >= '11:48:52') &
                    (data['Phone timestamp'] <= '11:49:00')) |
                   ((data['Phone timestamp'] >= '11:50:02') &
                    (data['Phone timestamp'] <= '11:50:12')) |
                   ((data['Phone timestamp'] >= '11:52:20') &
                    (data['Phone timestamp'] <= '11:52:30')) |
                   ((data['Phone timestamp'] >= '12:13:15') &
                    (data['Phone timestamp'] <= '12:13:20')) |
                   ((data['Phone timestamp'] >= '12:35:51') &
                    (data['Phone timestamp'] <= '12:36:00')) |
                   ((data['Phone timestamp'] >= '12:36:10') &
                    (data['Phone timestamp'] <= '12:36:15')) |
                   ((data['Phone timestamp'] >= '12:37:14') &
                    (data['Phone timestamp'] <= '12:37:18')) |
                   ((data['Phone timestamp'] >= '12:43:00') &
                    (data['Phone timestamp'] <= '12:43:10')) |
                   ((data['Phone timestamp'] >= '12:43:50') &
                    (data['Phone timestamp'] <= '12:43:55')) |
                   ((data['Phone timestamp'] >= '12:46:32') &
                    (data['Phone timestamp'] <= '12:46:36')).values
                )
            ]

        elif number == 47:
            mask = [
                np.flatnonzero(
                   (data['Phone timestamp'] <= '12:35:30') |
                   ((data['Phone timestamp'] >= '12:35:52') &
                    (data['Phone timestamp'] <= '12:35:55')) |
                   ((data['Phone timestamp'] >= '12:36:46') &
                    (data['Phone timestamp'] <= '12:36:50')) |
                   ((data['Phone timestamp'] >= '12:38:37') &
                    (data['Phone timestamp'] <= '12:38:43')) |
                   ((data['Phone timestamp'] >= '12:38:55') &
                    (data['Phone timestamp'] <= '12:39:06')) |
                   ((data['Phone timestamp'] >= '12:46:02') &
                    (data['Phone timestamp'] <= '12:46:06')) |
                   ((data['Phone timestamp'] >= '12:47:35') &
                    (data['Phone timestamp'] <= '12:47:40')) |
                   ((data['Phone timestamp'] >= '12:48:36') &
                    (data['Phone timestamp'] <= '12:48:42')) |
                   ((data['Phone timestamp'] >= '12:51:22') &
                    (data['Phone timestamp'] <= '12:51:26')) |
                   ((data['Phone timestamp'] >= '12:51:45') &
                    (data['Phone timestamp'] <= '12:51:48')) |
                   ((data['Phone timestamp'] >= '12:51:57') &
                    (data['Phone timestamp'] <= '12:52:00')) |
                   ((data['Phone timestamp'] >= '12:52:07') &
                    (data['Phone timestamp'] <= '12:52:10')) |
                   ((data['Phone timestamp'] >= '12:53:33') &
                    (data['Phone timestamp'] <= '12:53:37')) |
                   ((data['Phone timestamp'] >= '12:54:30') &
                    (data['Phone timestamp'] <= '12:54:37')) |
                   ((data['Phone timestamp'] >= '12:54:51') &
                    (data['Phone timestamp'] <= '12:54:55')) |
                   ((data['Phone timestamp'] >= '13:04:08') &
                    (data['Phone timestamp'] <= '13:04:14')) |
                   ((data['Phone timestamp'] >= '13:04:38') &
                    (data['Phone timestamp'] <= '13:04:40')) |
                   ((data['Phone timestamp'] >= '13:08:00') &
                    (data['Phone timestamp'] <= '13:08:22')) |
                   ((data['Phone timestamp'] >= '13:09:42') &
                    (data['Phone timestamp'] <= '13:10:19')) |
                   ((data['Phone timestamp'] >= '13:20:50.8') &
                    (data['Phone timestamp'] <= '13:20:54')) |
                   ((data['Phone timestamp'] >= '13:22:47') &
                    (data['Phone timestamp'] <= '13:22:50')) |
                   ((data['Phone timestamp'] >= '13:26:42') &
                    (data['Phone timestamp'] <= '13:27:10')) |
                   ((data['Phone timestamp'] >= '13:38:18') &
                    (data['Phone timestamp'] <= '13:38:25')) |
                   ((data['Phone timestamp'] >= '13:40:23') &
                    (data['Phone timestamp'] <= '13:40:26')) |
                   ((data['Phone timestamp'] >= '13:41:35') &
                    (data['Phone timestamp'] <= '13:41:38')) |
                   ((data['Phone timestamp'] >= '13:42:27') &
                    (data['Phone timestamp'] <= '13:42:42')) |
                   ((data['Phone timestamp'] >= '13:42:55') &
                    (data['Phone timestamp'] <= '13:43:00')) |
                   ((data['Phone timestamp'] >= '13:44:00') &
                    (data['Phone timestamp'] <= '13:44:40')) |
                   ((data['Phone timestamp'] >= '13:45:20') &
                    (data['Phone timestamp'] <= '13:45:25')) |
                   ((data['Phone timestamp'] >= '13:45:28') &
                    (data['Phone timestamp'] <= '13:45:50')) |
                   ((data['Phone timestamp'] >= '13:57:58') &
                    (data['Phone timestamp'] <= '13:58:01')) |
                   ((data['Phone timestamp'] >= '14:06:39') &
                    (data['Phone timestamp'] <= '14:06:41')) |
                   ((data['Phone timestamp'] >= '14:37:01') &
                    (data['Phone timestamp'] <= '14:37:04')).values
                )
            ]

        else:
            return data

    else:
        raise ValueError('Wrong name of group!')

    rows_to_remove = np.concatenate(mask)
    if len(rows_to_remove) > 0:
        # Based on the values found it is possible to remove
        # these beats as well as the preceding and the following ones
        rows_to_remove = remove_preceding_and_following_beat(
            rows_to_remove, len(data)
        )
        modified_data = data.drop(data.index[rows_to_remove])
        return modified_data
    else:
        return data


def remove_preceding_and_following_beat(to_remove, length):
    """
    Get the list with indices and for each element
    add a preceding and a following index if it is not
    outside the length of the dataframe.

    Arguments:
    ----------
      *to_remove*: (list) contains indices for removing
                   from a dataframe
      *length*: (int) corresponds to the length of the
                dataframe

    Returns:
      *new_indices_to_remove*: (list) contains indices
           for removing after adding some indices
    """
    to_remove = np.array(to_remove)
    # Check whether indices are not outside the index range.
    args_outside_range = np.argwhere(to_remove >= length)
    assert len(args_outside_range) == 0

    new_indices_to_remove = []
    for index in to_remove:
        for adj_index in [index - 1, index + 1]:
            if adj_index >= 0 and adj_index < length and \
               adj_index not in to_remove and \
               adj_index not in new_indices_to_remove:
                new_indices_to_remove.append(adj_index)
    new_indices_to_remove = np.sort(np.concatenate(
        [to_remove, new_indices_to_remove]
    ))
    # Check for duplicates
    assert len(new_indices_to_remove) == len(
        np.sort(new_indices_to_remove))
    return new_indices_to_remove


def remove_adjacent_beats(data, filtered_indices, time):
    """
    Remove all RR-intervals within the neighborhood
    [index - time, index + time], where index
    is a value from filtered_indices selected to remove
    from measurements.

    Arguments:
    ----------
      *data* - (Pandas DataFrame) contains all measurements
      *filtered_indices* - (list / Numpy.ndarray) contains indices
                           to remove from Pandas DataFrame
      *time* - (object) everything which can be a proper
                argument of the pandas.Timedelta object:
                that defines the neighborhood for removing

    Returns:
    --------
      *data* - (Pandas DataFrame) thinned DataFrame, without a subset
               of measurements.
    """
    data = data.copy()
    current_indices = data.index.values
    existing_and_selected_indices = np.intersect1d(
        current_indices,
        filtered_indices
    )

    timestamps_of_selected_indices = data[data.index.isin(
        existing_and_selected_indices
    )]['Phone timestamp'].values

    timeranges_to_remove = []
    for time_index in timestamps_of_selected_indices:
        timeranges_to_remove.append(
            [time_index - pd.Timedelta(time),
             time_index + pd.Timedelta(time)]
        )
    data = remove_selected_time_ranges(data, timeranges_to_remove)
    return data


def remove_selected_time_ranges(data, ranges_to_remove):
    """
    Remove measurements from selected time ranges.

    Arguments:
    ----------
      *data* - (Pandas DataFrame) contains all measurements
      *ranges_to_remove* - (list of lists / Numpy array with two-element
                            tuples): contains time ranges to remove
                            from *data*

    Returns:
    --------
      *data* - (Pandas DataFrame) contains measurements
               after removing selected time ranges.
    """
    data = data.copy()
    for current_timestamp in ranges_to_remove:
        start_timestamp = current_timestamp[0]
        end_timestamp = current_timestamp[1]
        indices_of_dataframe_to_remove = data[
            (data['Phone timestamp'] >= start_timestamp) &
            (data['Phone timestamp'] <= end_timestamp)].index
        data.drop(indices_of_dataframe_to_remove, inplace=True)
    return data


def remove_consecutive_beats_after_holes(data, hole_time, window_time):
    """
    Find holes (greater than hole_time) in the dataset and remove
    all beats coming within the *window_time* window after occurring
    of holes.

    Arguments:
    ----------
      *data* - (Pandas DataFrame) contains all measurements
      *hole_time* - (object) everything which can be a proper
                    argument of the pandas.Timedelta object:
                    defines the size of the hole which should
                    be suspected.
      *window_time* - (object) everything which can be a proper
                      argument of the pandas.Timedelta object:
                      defines the size of the window within
                      all measurements coming after a given hole
                      will be removed

    Returns:
    --------
       *data* - (Pandas DataFrame) contains measurements
                without a subset of them, according to the defined
                rules.
    """
    data = data.copy()
    hole_size = pd.to_timedelta(hole_time)
    removing_further_beats = pd.to_timedelta(window_time)
    # Returns timestamps of the first occurrences after holes in data
    holes = data['Phone timestamp'].sort_values().diff() > hole_size
    timestamps_after_holes = data.loc[holes]['Phone timestamp'].values
    timeranges_to_remove = []
    for time_index in timestamps_after_holes:
        timeranges_to_remove.append(
            [time_index,
             time_index + removing_further_beats]
        )
    data = remove_selected_time_ranges(data, timeranges_to_remove)
    return data


# initial_cut_window = '45 seconds'
# end_cut_window = '30 seconds'

def remove_first_and_last_indices(data, initial_cut_window, end_cut_window):
    """
    Remove a first few and last few elements from the dataframe.

    Arguments:
    ----------
      *data* - (Pandas DataFrame) contains all measurements
      *initial_cut_window* - (object) everything which can be a proper
                             argument of the pandas.Timedelta object:
                             defines the upper bound of cutting from
                             the beginning of the dataframe
      *end_cut_window* - (object) everything which can be a proper
                         argument of the pandas.Timedelta object:
                         defines the lower bound of cutting
                         among the last elements of the dataframe

    Returns:
    --------
      *data* - (Pandas DataFrame) contains measurements
                without a subset of them, according to the defined
                rules.
    """
    data = data.copy()
    first_timestamp = data.iloc[0]['Phone timestamp']
    last_timestamp = data.iloc[-1]['Phone timestamp']
    first_cut_timestamp = first_timestamp + pd.Timedelta(initial_cut_window)
    last_cut_timestamp = last_timestamp - pd.Timedelta(end_cut_window)

    initial_indices_to_remove = data[
        (data['Phone timestamp'] <= first_cut_timestamp)].index
    last_indices_to_remove = data[
        (data['Phone timestamp'] >= last_cut_timestamp)].index
    indices_to_remove = initial_indices_to_remove.union(last_indices_to_remove)
    data.drop(indices_to_remove, inplace=True)
    return data


def remove_negative_timestamps(data):
    """
    Remove anomalies in datasets related to negative Timedeltas
    between consecutive measurements. Remove also neighboring
    measurements.

    Argument:
    ---------
      *data* - (Pandas DataFrame) contains all measurements

    Returns:
    --------
      *data* - (Pandas DataFrame) thinned dataframe, without
               anomalous measurements
    """
    data, deltas = data.copy(), data.copy()
    indices = data.index.values
    current_max = data.loc[data.index == indices[0], 'Phone timestamp'].values[0]
    deltas.loc[deltas.index == indices[0], 'difference'] = pd.Timedelta(0)
    for i in range(1, indices.shape[0]):
        current_index = indices[i]
        deltas.loc[deltas.index == current_index, 'difference'] = \
            data.loc[data.index == current_index, 'Phone timestamp'].values[0] - \
            current_max
        if data.loc[data.index == current_index, 'Phone timestamp'].values[0] > current_max:
            current_max = data.loc[
                data.index == current_index, 'Phone timestamp'].values[0]
    deltas = deltas.loc[deltas['difference'] < pd.Timedelta(0)]
    deltas = deltas.index.values
    indices_to_remove = []
    # Take the current element and previous as well as next element 
    # in the dataset if only it is possible
    for i in range(deltas.shape[0]):
        current_position = np.argwhere(indices == deltas[i]).flatten()[0]
        if current_position == 0:
            values_to_remove = [deltas[i], indices[current_position + 1]]
        elif current_position == (indices.shape[0] - 1):
            values_to_remove = [indices[current_position - 1], deltas[i]]
        else:
            values_to_remove = [indices[current_position - 1],
                                deltas[i],
                                indices[current_position + 1]]
        for value in values_to_remove:
            if (value not in indices_to_remove) and (value in indices):
                indices_to_remove.append(value)
    data.drop(indices_to_remove, inplace=True)
    return data


def convert_absolute_time_to_timestamps_from_given_timestamp(
    data, initial_timestamp=pd.Timestamp("2022-01-01 00:00:00")
):
    """
    Start time will be rescheduled to Timestamp given as argument
    'initial_timestamp' and consecutive timestamps will correspond
    to time from the beginning of measurements.

    Arguments:
    ----------
     *data*: Pandas dataframe which contains 'Phone timestamp' column
             (datetime64[ns] type) and a column with a measurement
             corresponding to a given timestamp.
     *initial_timestamp*: (optional) Pandas Timestamp which contains
                          a timestamp corresponding to the beginning
                          of measurements.

    Returns:
    --------
     *dataframe*: Pandas dataframe which contains 'Phone timestamp' column
                  (datetime64[ns] type) with values rescheduled to the
                  starting timestamp given as 'initial_timestamp'.
    """
    dataframe = data.copy()
    dataframe["Phone timestamp"] = dataframe["Phone timestamp"].apply(
        lambda ts: ts.replace(
            year=initial_timestamp.year,
            month=initial_timestamp.month,
            day=initial_timestamp.day,
            hour=initial_timestamp.hour,
            minute=initial_timestamp.minute,
            second=initial_timestamp.second,
            microsecond=initial_timestamp.microsecond,
        )
    )
    differences = data["Phone timestamp"].diff()
    differences.iloc[0] = pd.Timedelta("0 days 00:00:00.000000")
    differences = differences.cumsum()
    dataframe["Phone timestamp"] = dataframe["Phone timestamp"] + differences
    return dataframe


def select_indices_to_filtering(data,
                                column_name):
    """
    Use Discrete Wavelet Transform (DWT) to filter out the indices
    which have to be replaced due to non-sinus beats.

    Arguments:
    ----------

    Returns:
    --------
    """
    # Discrete wavelet transform. The following function returns
    # the approximation of coefficients and their detail values.
    (coeff_approx, coeff_detail) = pywt.dwt(
        data[column_name],
        'db5',
        mode='smooth'
        # 'haar',
        # mode='sym'
    )
    # Estimate noise from original data samples and calculate
    # a threshold for filtering
    noise_estimation = np.mean(np.power(coeff_detail, 2))
    no_of_samples = data[column_name].shape[0]
    threshold = np.sqrt(noise_estimation * np.log(
        no_of_samples))
    # Select anomaly indices
    indicated_indices = 2 * np.ravel(
        np.argwhere(np.abs(coeff_detail) > threshold))
    return coeff_detail, indicated_indices


def interpolate_data_with_splines(original_data,
                                  current_data,
                                  column_name):
    """
    Prepare cubic interpolation for a selected subset of data.

    Arguments:
    ----------
      *original_data*: Pandas dataframe with 'Phone timestamp' and at least
                       one column with data before the filtering of the
                       dataframe
      *current_data*: Pandas dataframe with 'Phone timestamp' and at least
                      one column with data after removing the anomalies
      *column_name*: string containing name of column in a dataframe
                     for data interpolation, e.g. 'RR-interval [ms]'

    Returns:
    --------
      *modified_dataframe*: Pandas dataframe with modified values
                            of the selected column
      *predictions*: Pandas series with predictions for missing data
                     (present in *current_data* and absent in *original_data*)
      *removed_timestamps*: Numpy array with timestamps for which predictions
                            have been made
    """
    original_data, current_data = original_data.copy(), current_data.copy()
    original_timestamps = original_data["Phone timestamp"].values
    current_timestamps = current_data["Phone timestamp"].values

    # Timestamps which were removed in the meantime
    removed_timestamps = np.setdiff1d(original_timestamps,
                                      current_timestamps)
    # Assumption that first row contains the oldest timestamp
    # while the last row contains the newest one
    min_timestamp = current_data.iloc[0]['Phone timestamp']
    max_timestamp = current_data.iloc[-1]['Phone timestamp']
    if type(min_timestamp) == pd.Timestamp:
        min_timestamp = min_timestamp.to_numpy()
    if type(max_timestamp) == pd.Timestamp:
        max_timestamp = max_timestamp.to_numpy()
    # Remove timestamps from the boundaries because DWT cannot be performed
    # for the first and the last values
    filtering_extreme_timestamps = np.delete(
        removed_timestamps,
        np.argwhere(
            (removed_timestamps < min_timestamp) |
            (removed_timestamps > max_timestamp)
        )
    )
    # Prepare data interpolation using splines
    f = CubicSpline(current_data["Phone timestamp"].values.astype(float),
                    current_data[column_name].values,
                    bc_type='natural')
    # Calculate predictions for missing data
    predictions = f(filtering_extreme_timestamps.astype(float))
    # WARNING! IF DATA IN CURRENT_DATA ARE INTEGERS, THEN ALL PREDICTIONS
    # WILL BE ROUNDED!!! It is due to the fact that RR-intervals are
    # integers (derived in miliseconds).
    if current_data[column_name].dtypes == np.int64:
        predictions = np.round(predictions)
    predictions = pd.Series(data=predictions,
                            index=filtering_extreme_timestamps,
                            dtype=current_data[column_name].dtype)

    # Get original data from indices which have been removed and replace them
    # with predictions
    modified_rows = original_data[
        original_data["Phone timestamp"].isin(filtering_extreme_timestamps)].copy()
    modified_rows.loc[:, column_name] = predictions.values

    # Evaluation of predictions - remove values which are lower than
    # the previous minimum or higher than the previous maximum
    original_range = current_data[column_name].values
    values_to_remove = modified_rows.loc[
        (modified_rows[column_name] < np.min(original_range)) |
        (modified_rows[column_name] > np.max(original_range))
    ]
    modified_rows.drop(values_to_remove.index, inplace=True)

    # For timestamps for which predictions have been calculated
    # there should be no information inside tha 'current_data' dataframe.
    assert np.intersect1d(
        modified_rows['Phone timestamp'].values,
        current_data['Phone timestamp'].values
    ).shape[0] == 0
    # Create a single dataframe with the stored dataframe
    # and newly calculated predictions.
    modified_dataframe = pd.concat([modified_rows, current_data])
    modified_dataframe = modified_dataframe.sort_values(
        by=["Phone timestamp"], ascending=True
    ).reset_index(drop=True)
    return modified_dataframe, predictions, filtering_extreme_timestamps


def return_hour_from_datetime(datetime: np.datetime64 | pd.Timestamp) -> str:
    """
    Change format of the given timestamp: remove day, month
    and year, leave only time.

    Argument:
    ---------
      *datetime*: (Numpy datetime64 or Pandas Timestamp) contains
                  day, month, year and time, e.g. '2022-12-31T23:59:59'.

    Returns:
    --------
      String containing only time
    """
    if type(datetime) == np.datetime64:
        return datetime.astype(str).split('T')[1]
    elif type(datetime) == pd.Timestamp:
        return datetime.strftime('%H:%M:%S.%f')
    else:
        raise NotImplementedError


if __name__ == "__main__":
    pass
