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

import csv
import pandas as pd
from typing import Tuple

from HRV_calculation import calculate_mean_HRV_based_on_windows


def save_results(results: pd.DataFrame,
                 parameters: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save results to .csv file. If HRV was calculated in subsequences,
    at the beginning, mean HRV should be calculated. Furthermore,
    timestamps should be removed.

    Arguments:
    ----------
      *results* (Pandas Dataframe) contains results from all patients
      *parameters* (dictionary) stores information about experiment
                   and parameters required for saving

    Returns:
    --------
      *results* (Pandas Dataframe) contains results after processing
      *treatment_results* (Pandas Dataframe) only contains results
                          for treatment group (without control group)
    """
    results = results.copy()
    results.to_pickle(
        f'{parameters["plot_saving_folder"]}/'
        f'results_{parameters["name"]}.pkl')
    if parameters['sequence_range'] == 'full':
        results.to_csv(
            f'{parameters["plot_saving_folder"]}/'
            f'results_{parameters["name"]}.csv')
    elif parameters['sequence_range'] == 'windows':
        results = results.apply(
            lambda x: calculate_mean_HRV_based_on_windows(
                x, method=parameters['method']), axis=1
        )
        results = results.drop('timestamps', axis=1)
        results.to_csv(
            f'{parameters["plot_saving_folder"]}/'
            f'mean_results_{parameters["name"]}.csv')
    treatment_results = results.loc[results['group'] == 'treatment']
    return (results, treatment_results)


def save_parameters(parameters: dict,
                    name: str = 'parameters') -> None:
    """
    Save current parameters of the dictionary.

    Arguments:
    ----------
      *parameters*: (dict) contains all parameters defining the current
                    experiment, including 'plot_saving_folder' key
      *name* (string) optional argument defining name of the file
             for saving parameters
    """
    with open(f"{parameters['plot_saving_folder']}/"
              f"{name}.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=parameters.keys())
        writer.writeheader()
        writer.writerow(parameters)


if __name__ == "__main__":
    pass
