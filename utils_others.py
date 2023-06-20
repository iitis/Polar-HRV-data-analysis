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
from scipy.stats import (
    levene,
    mannwhitneyu
)


def append_row_to_file(filename,
                       elements):
    '''
    Append a single row to the given file.

    Parameters
    ----------
    filename: folder and name of file
    elements: elements to saving in filename
    '''
    if not filename.endswith('.csv'):
        filename += '.csv'
    with open(filename, "a+") as stream:
        np.savetxt(stream,
                   np.array(elements)[np.newaxis],
                   delimiter=';',
                   fmt='%s')


def filter_patients_with_quetiapine(list_of_quetiapine_patients,
                                    treatment_results):
    """
    Filter out patients taking quetiapine from the main
    dataframe with all patients.

    Arguments:
    ----------
      *list_of_quetiapine_patients* - (list) contains integers corresponding
                                      to the number of patients taking quetiapine
      *treatment_results* - Pandas Dataframe with results for all patients: 
                            PANSS_P, PANSS_N, PANSS_G and PANSS_total containing
                            PANSS scores in a positive, negative and general
                            scale, total results and HRV scores

    Returns:
    --------
      *treatment_results* - Pandas Dataframe without patients taking quetiapine
      *quetiapine_patients_results* - Pandas Dataframe containing patients taking
                                      quetiapine
    """
    quetiapine_patients_results = treatment_results.loc[
        treatment_results['no_of_person'].isin(list_of_quetiapine_patients)
    ].copy()
    treatment_results = treatment_results.drop(
        quetiapine_patients_results.index.values)
    return treatment_results, quetiapine_patients_results


def compare_means_and_variances_in_groups(
        processed_data,
        HRV_method,
        saving_folder):
    """
    Prepare a non-parametric version of the statistical test
    comparing mean HRV values between the treatment and the control
    group as well as the variance equality test between two groups.

    Parameters:
    ----------
      *processed_data*: (Pandas Dataframe) contains (at least) columns:
                        group to distinguish patients from the control group
                        and 'HRV_{name of the method used}' with a value
                        for the corresponding person
      *HRV_method*: (string) represents the name of the method used
                    for calculation of the HRV value
      *saving_folder*: (string) contains path to the folder where results
                       should be saved

    Returns:
    --------
    A dictionary containing the following keys:
      *u_test_statistic*: (float) statistic of the Mann-Whitney U-test
      *u_test_p_value*: (float) p-value for the corresponding U-test statistic
      *levene_statistic*: (float) statistic of the Levene's test
      *levene_p_value*: (float) p-value for the corresponding Levene's test statistic
    """
    processed_data_HRV = processed_data[['group', f'HRV_{HRV_method}']]
    control_values = processed_data_HRV[
        processed_data_HRV['group'] == 'control'][f'HRV_{HRV_method}'].values
    treatment_values = processed_data_HRV[
        processed_data_HRV['group'] == 'treatment'][f'HRV_{HRV_method}'].values
    # We test the hypothesis that HRV within the treatment group
    # is statistically significantly lower than within the control group
    u_test_statistic, u_test_p_value = mannwhitneyu(
        treatment_values,
        control_values,
        alternative='less')
    # We test the hypothesis about variance equality between two groups
    levene_statistic, levene_p_value = levene(
        treatment_values,
        control_values
    )
    print(f'Result of the Mann-Whitney U-test: \n statistic: {u_test_statistic} '
          f'with p-value: {u_test_p_value} \n')
    print(f"Result of the Levene\'s test: \n statistic: {levene_statistic}"
          f'with p-value: {levene_p_value}')
    path = f'{saving_folder}/statistical_tests_results.csv'
    append_row_to_file(path,
                       ('Mann-Whitney U-test statistic;p-value;'
                        "Levene\'s test statistic;p-value;"
                        'median_treatment;std_treatment;'
                        'median_control;std_control'))
    append_row_to_file(path,
                       (f'{u_test_statistic};{u_test_p_value};'
                        f'{levene_statistic};{levene_p_value};'
                        f'{np.median(treatment_values)};{np.std(treatment_values)};'
                        f'{np.median(control_values)};{np.std(control_values)}'))
    return {
        'u_test_statistic': u_test_statistic,
        'u_test_p_value': u_test_p_value,
        'levene_statistic': levene_statistic,
        'levene_p_value': levene_p_value
    }


if __name__ == "__main__":
    pass
