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
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product


def plot_heatmap_correlation(mode: str,
                             interpolation: bool,
                             category: str,
                             path: str,
                             file: str,
                             critical_value: float) -> None:
    """
    Plot heatmaps of correlation plots for different values
    of window size and window steps (both in minutes). One
    of the heatmap is related to Pearson's r values between HRV
    and PANSS while the second one represents p-values for
    corresponding correlation coefficients.
    ATTENTION!
    The setup below defines the limits of the plots based on values
    considered in the dataset analyzed in the paper. In other cases,
    one has to adjust the setting.

    Arguments:
    ----------
        *mode* (string): 'correlation' or 'p-value'; defines which plot
                          will be prepared
        *interpolation* (bool): reponsibles for the plot title; in the
                                positive case adds 'yes'; in the negative
                                case adds 'no' to the title; also has
                                an impact on the filename
        *category* (str): 'PANSS_G', 'PANSS_P', 'PANSS_N', 'PANSS_T';
                          defines which category of the PANSS test will
                          be considered
        *path* (str): path for loading file with results and for plot saving
        *file* (str): .csv file with results for loading
        *critical_value* (float): defines the threshold for statistical
                                  significance
    """
    dict_with_full_categories = {
        'PANSS_G': 'PANSS general',
        'PANSS_P': 'PANSS positive',
        'PANSS_N': 'PANSS negative',
        'PANSS_T': 'PANSS_total'
    }
    data = pd.read_csv(f'{path}{file}', delimiter=';')
    data = data.loc[data["category"] == category]
    data = data.astype({'window_size': 'int32'})
    data.rename(columns={'window_size': 'window size [min]',
                         'step': 'step [min]'},
                inplace=True)
    summary = pd.pivot_table(data=data,
                             index='step [min]',
                             columns='window size [min]',
                             values='correlation')
    mask = pd.pivot_table(data=data,
                          index='step [min]',
                          columns='window size [min]',
                          values='pvalue')
    # minimum = np.nanmin(np.array(summary).ravel())
    limits = {
        'correlation': {'minimum': -0.515, 'maximum': -0.4},
        'p-value': {'minimum': 0.01, 'maximum': 0.05}
    }
    palletes = {
        'correlation': sns.light_palette(
            "seagreen", as_cmap=True, reverse=True),
        'p-value': sns.color_palette(
            "YlOrBr", as_cmap=True)
    }
    cbar_label = {
        'correlation': "Pearson\'s r",
        'p-value': 'p-value'
    }
    titles = {
        'correlation': "Pearson\'s r",
        'p-value': "Pearson\'s r (p-value)"
    }
    results_table_refs = {
        'correlation': summary,
        'p-value': mask
    }
    font_sizes = {
        'correlation': 11, 'p-value': 10
    }

    fig, ax = plt.subplots(figsize=(5.75, 4.5))
    sns.heatmap(results_table_refs[mode],
                mask=mask >= critical_value,
                annot_kws={"fontstyle": "oblique", "weight": "bold"},
                linewidth=0.5,
                vmin=limits[mode]['minimum'],
                vmax=limits[mode]['maximum'],
                cmap=palletes[mode],
                annot=True,
                cbar=False,
                fmt='.3f')
    sns.heatmap(results_table_refs[mode],
                mask=mask < critical_value,
                linewidth=0.5,
                vmin=limits[mode]['minimum'],
                vmax=limits[mode]['maximum'],
                cmap=palletes[mode],
                annot=True,
                cbar_kws={'label': cbar_label[mode]},
                fmt='.3f')
    title = (
        f"{titles[mode]}; HRV vs "
        f'{dict_with_full_categories[category]}; '
        'interpolation: '
    )
    title += "yes" if interpolation else "no"
    plt.title(title, x=0.45, fontsize=font_sizes[mode])
    plt.tight_layout()
    plt.savefig(
        f'{path}heatmap_{mode}_{category}_interpolation_{interpolation}.pdf',
        dpi=400)
    plt.close()


if __name__ == "__main__":
    interpolations = [True, False]
    modes = ('correlation', 'p-value')
    category = 'PANSS_G'  # 'PANSS_G', 'PANSS_P', 'PANSS_N', 'PANSS_T'
    file = 'results.csv'
    # Based on the Bonferroni correction; basic threshold: 0.05,
    # four statistical tests
    critical_value = 0.0125
    for interpolation, mode in product(interpolations, modes):
        path = (
            '../article_results/sensitivity_analysis/'
            f'interpolation_{interpolation}/'
        )
        plot_heatmap_correlation(mode,
                                 interpolation,
                                 category,
                                 path,
                                 file,
                                 critical_value)
