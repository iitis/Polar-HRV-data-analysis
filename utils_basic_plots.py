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

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter, MinuteLocator

from utils_others import append_row_to_file


def prepare_labels(name):
    """
    Prepare labels for further plots according to the data type.

    Argument:
    ---------
       *name* (string) 'rr_intervals' or 'accelerometer'

    Returns:
    --------
       A dictionary with axis labels and corresponding data in dataframes
       (X, Y for RR intervals and X, Y, Z for accelerometer data) and title
       for plots.
    """
    if name == 'rr_intervals':
        return {
            'x_data': 'Phone timestamp',
            'y_data': 'RR-interval [ms]',
            'x_label': 'Timestamp',
            'y_label': 'RR interval [ms]',
            'title': 'Plot of RR intervals (milliseconds) depending on time'
        }

    elif name == 'accelerometer':
        return {
            'x_data': 'X [mg]',
            'y_data': 'Y [mg]',
            'z_data': 'Z [mg]',
            'x_label': 'Timestamp',
            'time': 'Phone timestamp',
            'title': 'Values of accelerometer depending on time'
        }
    else:
        raise ValueError('Wrong type of the analyzed data!')


def change_plot_range(ranges):
    """
    Change range of the current plot, according to the values
    in dictionary 'ranges'.

    Argument:
    ---------
     *ranges* a dictionary with some or all following keys:
      -bottom- defines the lowest value to plot on y-axis
     *top* defines the largest value to plot on y-axis
      -left- defines the lowest value to plot on x-axis
      -right- defines the largest value to plot on x-axis
    """
    if 'bottom' in ranges and ranges['bottom'] is not None:
        plt.ylim(bottom=ranges['bottom'])
    if 'top' in ranges and ranges['top'] is not None:
        plt.ylim(top=ranges['top'])
    if 'left' in ranges and ranges['left'] is not None:
        plt.xlim(left=ranges['left'])
    if 'right' in ranges and ranges['right'] is not None:
        plt.xlim(right=ranges['right'])


def plot_anomalies(ax,
                   data,
                   anomalies):
    """
    Plot anomalies using vertical lines.

    Arguments:
    ----------
       *ax*: Axes object for plotting
       *data*: Pandas dataframe containing timestamps
       *anomalies*: list or Numpy array with anomalies for plotting
    """
    anomalies = np.array(anomalies)
    for index in range(0, anomalies.shape[0]):
        ax.axvline(data["Phone timestamp"].iloc[anomalies[index]],
                   color='skyblue')


def plot_1D_signal(data,
                   data_type,
                   column_name=None,
                   ranges={
                     'bottom': None,
                     'top': None,
                     'left': None,
                     'right': None
                   },
                   anomalies=None,
                   saving_folder=None,
                   name=None):
    """
    Plot one-dimensional signal, e.g. RR-intervals.

    Arguments:
    ----------
      *data*: (Pandas Dataframe) contains data which should be plotted
      *data_type*: (string) type of the plotted data, e.g. 'rr_intervals'
      *column_name*: (list of strings) columns of *data*, e.g. ['rr-intervals']
      *ranges* (dict) optional, may change the range of the plot
      *anomalies* (list or Numpy array) optional anomalies for plot
      *saving_folder* (string) optional, custom folder for saving
      *name* (string) optional, custom filename for saving
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    labels = prepare_labels(data_type)
    if anomalies is not None:
        plot_anomalies(ax, data, anomalies)
    if len(column_name) == 2:
        modified_data = pd.melt(
            data,
            id_vars="Phone timestamp",
            value_vars=column_name,
            var_name='processing_type',
            value_name='value'
        )
        plot = sns.lineplot(
            data=modified_data,
            x='Phone timestamp',
            y='value',
            hue='processing_type',
            lw=1,
            palette=['red', 'blue'],
            dashes=False)
        plot.legend().set_title(None)
    elif len(column_name) == 1:
        sns.lineplot(data=data,
                     x='Phone timestamp',
                     y=column_name[0],
                     lw=1,
                     color='red')
    else:
        raise NotImplementedError
    plt.xlabel(labels['x_label'])
    plt.ylabel(labels['y_label'])
    plt.title(labels['title'])

    # If it is desired, change plot ranges.
    change_plot_range(ranges)
    myFmt = DateFormatter("%H:%M:%S")
    myLct = MinuteLocator(interval=5)
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(myLct)
    if labels['x_label'] == 'Timestamp':
        plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    if name is None:
        timestamp = data.iloc[0]['Phone timestamp'].strftime('%Y-%m-%d_%H%M%S')
        name = f'{data_type}_plot_{timestamp}.png'
    if saving_folder is not None:
        os.makedirs(saving_folder, exist_ok=True)
        name = f'{saving_folder}/{name}'
    plt.savefig(name, dpi=400)
    plt.close()


def plot_column_of_values_for_given_person(dataframe,
                                           column,
                                           group,
                                           number,
                                           interval,
                                           saving_folder='.'):
    """
    Plot values from a selected column against Phone timestamps.
    It can be used for instance for plotting HRV in time.

    Arguments:
    ----------
      *dataframe* - Pandas dataframe with Phone timestamp
                    and at least one other column
      *column* - the name of column for plotting in y-axis
      *group* - (string) for the description purposes; defines
                a group of measurements
      *number* - (int / string) for the description purposes;
                 defines a person number in the group
      *interval* - (string) for the description purposes;
                   defines the interval between consecutive
                   windows
      *saving_folder* - (string), optional: defines a folder
                        for saving the plot
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    sns.lineplot(data=dataframe,
                 x="Phone timestamp",
                 y=column)
    myFmt = DateFormatter("%H:%M:%S")
    ax.xaxis.set_major_formatter(myFmt)
    plt.title(f'{group}: {number}, interval: {interval}')
    plt.xticks(rotation=90)
    column = column.replace(' ', '_').replace('/', '_')
    name = f'{column}_{group}_{number}_{interval}'
    plt.tight_layout()
    plt.savefig(f'{saving_folder}/{name}.pdf', dpi=300)
    plt.close()


def regression_PANSS(dataframe,
                     HRV_columnname,
                     parameters,
                     quetiapine_patients=None):
    """
    Create regression plot for PANSS scores.

    Arguments:
    ----------
      *dataframe*: Pandas dataframe with columns: PANSS_P,
                   PANSS_N, PANSS_G and PANSS_total containing
                   PANSS scores in a positive, negative and general
                   scale, total results and HRV scores
      *HRV_columname*: (string) column containing HRV scores
      *parameters*: (dictionary) contains following keys:
        -step_frequency- (pd.Timedelta) step between consecutive windows
        -window_size- (pd.Timedelta) range of time windows
        -result_saving_folder- (string) folder for saving correlation results
        -plot_saving_folder- (string) folder for saving correlation plots
      *quetiapine_patients*: (optional) Pandas Dataframe containing data from
                             people taking quetiapine; they are not taken into
                             account during the calculation of correlation
                             coefficient but they are additionally displayed
                             in the final plot
    """
    columns = ['PANSS_P', 'PANSS_N', 'PANSS_G', 'PANSS_total']
    labels = ['PANSS positive scale',
              'PANSS negative scale',
              'PANSS general scale',
              'PANSS total result']
    for column, label in zip(columns, labels):
        correlation_result = pearsonr(
            x=dataframe[HRV_columnname],
            y=dataframe[column]
        )
        statistic, pvalue = correlation_result[0], correlation_result[1]
        confidence_interval = correlation_result.confidence_interval()
        # Save results to the file
        path = f"{parameters['result_saving_folder']}/results.csv"
        if not os.path.exists(path):
            append_row_to_file(
                path,
                "step;window_size;category;correlation;pvalue;CI_start;CI_end"
            )
        append_row_to_file(
            path,
            (f"{parameters['step_frequency'].total_seconds() / 60};"
             f"{parameters['window_size'].total_seconds() / 60};"
             f"{column};{statistic};{pvalue};"
             f"{confidence_interval[0]};{confidence_interval[1]}")
        )
        sns.regplot(data=dataframe,
                    x=HRV_columnname,
                    y=column)
        if quetiapine_patients is not None:
            sns.scatterplot(
                data=quetiapine_patients,
                x=HRV_columnname,
                y=column,
                color='red',
                s=50
            )
            plot_objects = plt.gca().get_children()
            plt.legend([plot_objects[0], plot_objects[3]],
                       ['w/o quetiapine', 'with quetiapine'])
        plt.ylabel(label)
        plt.title(f'A relationship between HRV and {label}\n'
                  f'Pearson r: {statistic:.3f}, p-value: {pvalue:.4f}, 95% CI: '
                  f'[{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]')
        plt.savefig(f"{parameters['plot_saving_folder']}/HRV_{column}.pdf",
                    dpi=300)
        plt.close()


def boxplot_HRV(dataframe,
                saving_folder,
                x_axis_variable='HRV_RMSSD',
                y_axis_variable='group'):
    """
    Create boxplot which compares distributions from
    different categories.

    Arguments:
    ----------
      *dataframe*: Pandas dataframe with full results
      *saving_folder*: (string) name of the folder for plots
      *x_axis_variable*: variable from 'dataframe' for x-axis
      *y_axis_variable*: variable from 'dataframe' for y-axis
    """
    colors = ['cornflowerblue', 'indianred']
    sns.set_palette(sns.color_palette(colors))
    fig, ax = plt.subplots(figsize=(6, 2))
    sns.boxplot(data=dataframe,
                x=x_axis_variable,
                y=y_axis_variable)
    plt.title('Box and whisker plot displaying data distributions '
              'in two groups',
              fontsize=11)
    plt.xlabel(x_axis_variable, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{saving_folder}/boxplot_HRV.pdf', dpi=300)
    plt.close()


def plot_distribution_PANSS_subcategories(load_folder,
                                          save_folder=None):
    """
    Prepare a box plot with distributions of PANSS subcategories
    for all patients chosen for experiments.

    Arguments:
    ----------
      *load_folder*: (string) path for the folder with PANSS results
                     located in 'PANSS.csv' file
      *save_folder*: (string) optional argument defining path to the
                     prepared plot with the distributions of PANSS
    """
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette())
    fig, ax = plt.subplots(figsize=(7, 2.5))
    PANSS_summary = pd.read_csv(f'{load_folder}PANSS.csv', delimiter=';')
    PANSS_summary = PANSS_summary.drop(
        PANSS_summary[PANSS_summary.no_of_person.isin(
         [5, 6, 10, 11, 12, 14, 18, 28, 30, 34, 35, 39])].index
    )
    PANSS_summary.rename(columns={
        'PANSS_P': 'PANSS positive',
        'PANSS_N': 'PANSS negative',
        'PANSS_G': 'PANSS general',
        'PANSS_total': 'PANSS total'
    }, inplace=True)
    reordered_PANSS = pd.melt(
        PANSS_summary,
        id_vars=['no_of_person'],
        value_vars=['PANSS positive', 'PANSS negative',
                    'PANSS general', 'PANSS total'])
    sns.boxplot(x=reordered_PANSS["value"], y=reordered_PANSS["variable"],
                linewidth=1.)
    if save_folder is not None:
        save_path = f'{save_folder}PANSS_distribution.pdf'
    else:
        save_path = 'PANSS_distribution.pdf'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_accelerometer_data(data,
                            saving_folder=None,
                            name=''):
    """
    Plot accelerometer data (three plots, each image represents
    one of three dimensions).

    Arguments:
    ----------
     *data* - Pandas dataframe with columns 'Phone timestamp'
              with Pandas timestamp and measurements (in three
              dimensions) corresponding to given timestamps
     *saving_folder* - (optional str) defines a folder when a plot
                       will be saved, by default it is a current folder
     *name* - (optional str) defines an additional string located
              in the plot filename
    """
    sns.set_style("whitegrid")
    fig, (ax_x, ax_y, ax_z) = plt.subplots(nrows=3,
                                           sharex=True)
    axes = {
        'x': ax_x,
        'y': ax_y,
        'z': ax_z
    }
    labels = prepare_labels('accelerometer')
    for axis in list(axes.keys()):
        sns.lineplot(data=data,
                     x=labels['time'],
                     y=labels[f'{axis}_data'],
                     ax=axes[axis],
                     lw=1,
                     ci=None,
                     color='red')
    myFmt = DateFormatter("%H:%M:%S")
    axes[axis].xaxis.set_major_formatter(myFmt)
    # One legend above all plots
    fig.suptitle(labels['title'])

    plt.xticks(rotation=90)
    plt.tight_layout()
    if len(name) == 0:
        name = data.iloc[0]['Phone timestamp'].strftime('%Y-%m-%d_%H%M%S')
    fullname = f'ACC_plot_{name}.png'
    if saving_folder is not None:
        os.makedirs(saving_folder, exist_ok=True)
        fullname = f'{saving_folder}{fullname}'
    plt.savefig(fullname, dpi=600)
    plt.close()


def age_histograms(age_patients,
                   age_control,
                   folder='../Plots/'):
    """
    Compare age distribution in patients and healthy control.
    Prepare kernel density estimation plots.

    Arguments:
    ----------
      *age_patients* - (list) contains age of patients
      *age_control* - (list) contains age of healthy people
      *folder* - (str) optional, defines a folder for saving
    """
    ax = sns.displot(
        {'control': age_control,
         'treatment': age_patients},
        binwidth=5,
        kde=True)
    sns.move_legend(ax, 'upper right', bbox_to_anchor=(0.95, 0.9))
    ax.add_legend()
    ax.set(xlabel='Age',
           ylabel='Count',
           title='Distributions of age with kernel density estimation plots')
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}hist_age_kde.pdf', dpi=300)
    plt.close()


if __name__ == "__main__":
    age_patients = [
        41, 39, 27, 21, 66, 29, 48, 62, 44,
        38, 55, 36, 46, 30, 28, 20, 40, 36,
        44, 29, 53, 35, 51, 69, 33, 61, 55,
        36, 28, 43
    ]
    age_control = [
        50, 32, 51, 41, 42, 48, 44, 35, 43,
        36, 27, 27, 53, 28, 34, 29, 28, 30,
        29, 24, 63, 63, 62, 69, 53, 66, 62,
        56, 45, 26
    ]
    age_histograms(
        age_patients,
        age_control
    )
