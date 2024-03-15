## Polar HRV Data Analysis Library (PDAL) v 1.1
Library for HRV and accelerometer data analysis based on measurements from Polar H10 wearable devices.

It is a source code related to the paper:

> The analysis of heart rate variability and accelerometer mobility data
in the assessment of symptom severity in psychosis disorder patients
using a wearable Polar H10 sensor

Authors:
- Kamil Książek (ITAI PAS, ORCID ID: [0000-0002-0201-6220](https://orcid.org/0000-0002-0201-6220)),
- Wilhelm Masarczyk (FMS MUS, ORCID ID: [0000-0001-9516-0709](https://orcid.org/0000-0001-9516-0709)),
- Przemysław Głomb (ITAI PAS, ORCID ID: [0000-0002-0215-4674](https://orcid.org/0000-0002-0215-4674)),
- Michał Romaszewski (ITAI PAS, ORCID ID: [0000-0002-8227-929X](https://orcid.org/0000-0002-8227-929X)),
- Iga Stokłosa (FMS UMS, ORCID ID: [0000-0002-7283-5491](https://orcid.org/0000-0002-7283-5491)),
- Piotr Ścisło (PDMH, ORCID ID: [0000-0003-1213-2935](https://orcid.org/0000-0003-1213-2935)),
- Paweł Dębski (FMS UMS, ORCID ID: [0000-0001-5904-6407](https://orcid.org/0000-0001-5904-6407)),
- Robert Pudlo (FMS UMS, ORCID ID: [0000-0002-5748-0063](https://orcid.org/0000-0002-5748-0063)),
- Piotr Gorczyca (FMS UMS, ORCID ID: [0000-0002-9419-7988](https://orcid.org/0000-0002-9419-7988)),
- Magdalena Piegza (FMS UMS, ORCID ID: [0000-0002-8009-7118](https://orcid.org/0000-0002-8009-7118)).

*ITAI PAS* - Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences, Gliwice, Poland;  
*FMS UMS* - Faculty of Medical Sciences in Zabrze,
Medical University of Silesia, Tarnowskie Góry, Poland;  
*PDMH* - Psychiatric Department of the Multidisciplinary Hospital,
Tarnowskie Góry, Poland.  

## LICENSE:
Copyright 2023-2024
Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl

The main author of the code:
- Kamil Książek (ITAI PAS, ORCID ID: [0000-0002-0201-6220](https://orcid.org/0000-0002-0201-6220)).

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

## FUNCTIONALITY:
- Loading RR intervals and accelerometer data from Polar H10 wearable devices collected through [Polar Sensor Logger](https://play.google.com/store/apps/details?id=com.j_ware.polarsensorlogger&hl=pl&gl=US) application.
- Preparation of data preprocessing: identification and removal of anomalous measurements, data interpolation.
- Calculation of HRV values using the RMSSD (Root Mean Square of the Successive Differences), SDNN (Standard Deviation of N-N intervals) or pNN50 (number of pairs of adjacent R-R intervals with a difference greater than 50 ms) approaches in sliding windows.
- Calculation of mobility coefficient based on accelerometer data.
- Data postprocessing: plots of results, distributions, the calculation of correlation coefficients, etc.

This library prepares a full analysis of the dataset considered in the related publication to ensure the reproducibility of the described results (including all plots).

## DATASET:

Download the dataset from the following data repository:
The recommended path for data samples is `data` folder.

## RULES AND USAGE:

- Main HRV calculations are performed in the `main.py` file. Data is loaded, preprocessed, and HRV metrics are calculated in this file. Furthermore, the summary plots and calculation coefficients / statistical tests are performed. It is possible to choose one of the available HRV metrics, i.e. RMSSD, SDNN or pNN50, by setting `HRV_method` to `RMSSD`, `SDNN` or `pNN50`.
To reproduce detailed results for the window size of 15 minutes and the time interval between consecutive windows set as 1 minute, set `exclude_quetiapine = False` and `sensitivity_analysis = False` and run the file.
To reproduce sensitivity analysis for different window sizes and values of the time interval between consecutive time windows, set `exclude_quetiapine = False` and `sensitivity_analysis = True` and run the file. Then, to prepare the heatmaps of correlation, run the `utils_advanced_plots.py` file with the proper parameters according to the selected mode.
To reproduce results without the patients taking quetiapine, set `exclude_quetiapine = True` and `sensitivity_analysis = False`.

- Main accelerometer calculations are performed in the `utils_accelerometer.py` file. Both RR interval and accelerometer data are loaded in this file. Then, accelerometer data is downsampled to achieve the same sample frequency in both data types. In the next step, the mobility coefficient for each person is calculated. Finally, a correlation coefficient between HRV and mobility value is computed. Furthermore, the dependency is presented in separate files per person and in the collective picture for all the tested persons.
To reproduce experiments comparing mobility and HRV data, just run the `utils_accelerometer.py` file.

- To reproduce the histogram of age distribution in the two compared groups, just run the `utils_basic_plots.py` file.

- To plot the collected accelerometer data, run the `utils_loading.py` file.

- To perform unit tests of the code, run the `run_tests.py` file.

**WARNING! Please ensure the proper paths to the dataset are set in the `main.py`, `utils_accelerometer.py` and `utils_loading.py` files.**

## FILES:

- `HRV_calculation.py`: contains functions for the calculation of mean HRV values according to the RMSSD approach as well as for the creation of sliding windows.
- `main.py`: contains a mechanism for the loading and preprocessing data, calculation of HRV values and result analysis.
- `run_tests.py`: runs tests in the `/tests/` catalogue.
- `utils_accelerometer.py`: contains a mechanism for the loading and calculating mobility coefficient based on accelerometer data.
- `utils_advanced_plots.py`: contains a function for heatmap plotting used in the sensitivity analysis.
- `utils_basic_plots.py`: contains functions for preparing 1D plots of signal, scatterplots comparing HRV with the PANSS test values, box plots and plots of PANSS and age distributions.
- `utils_loading.py`: contains functions loaded data, scores and data frames with intermediate results.
- `utils_others.py`: contains auxiliary functions (i.e., appending rows to files, filtering patients and preparing the statistical test comparing HRV between the tested groups).
- `utils_postprocessing.py`: functions for result saving.
- `utils_preprocessing.py`: functions for data preprocessing, including manually selected anomalous values for removal.
