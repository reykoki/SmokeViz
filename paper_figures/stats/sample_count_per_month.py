import os
from collections import defaultdict
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import monthrange
import calendar
from glob import glob
import pickle
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable

fp  = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/dataset_pointers/thresh/round3/thresh_1.pkl'

with open(fp, 'rb') as handle:
    data_dict = pickle.load(handle)

file_list = data_dict['train']['truth']+ data_dict['val']['truth'] + data_dict['test']['truth']


all_yr = ['2018','2019','2020','2021','2022','2023','2024']

def get_density_count_per_year():
    num_yr = len(all_yr)
    sample_counts = pd.DataFrame({
        'Light' : np.zeros(num_yr),
        'Medium': np.zeros(num_yr),
        'Heavy' : np.zeros(num_yr)
    }, index=all_yr)

    for path in file_list:
        parts = path.split('/')
        year = parts[-4]
        density = parts[-3]
        if year in sample_counts.index and density in sample_counts.columns:
            sample_counts.loc[year, density] += 1

    print(sample_counts)

    ax = sample_counts.plot(
        figsize=(5, 4), kind='bar', stacked=True, color=['orange', 'indianred', 'darkred'], alpha=0.8
    )
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Sample Counts', fontsize=12)

    totals = sample_counts.sum(axis=1)  # Total per year
    for i, (year, total) in enumerate(zip(all_yr, totals)):
        y_offset = 0  
        for category in sample_counts.columns:
            value = sample_counts.at[year, category]
            percentage = (value / total) * 100
            ax.text(
                i, 
                y_offset + value / 2,  
                f"{percentage:.1f}", 
                ha="center", 
                va="center", 
                fontsize=7.5, 
                color="black" 
            )
            y_offset += value 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.savefig('stat_figures/sample_count_per_yr_percentages.png', bbox_inches='tight', dpi=300)

def get_month_count_w_month_iou_per_year():
    month_counts = pd.DataFrame({'2018' : [0]*12, '2019' : [0]*12, '2020' : [0]*12, '2021' : [0]*12, '2022' : [0]*12, '2023' : [0]*12, '2024' : [0]*12},
                                 index = list(range(1,13)))

    for path in file_list:
        match = re.search(r's(\d{4})(\d{3})', path)
        if match:
            year = match.group(1)
            doy = int(match.group(2))
            if year in month_counts.columns:
                # convert day-of-year → month
                month = pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=doy-1)
                month_idx = month.month
                month_counts.at[month_idx, year] += 1

    print(month_counts)
    month_IoU = [0.3809, 0.4226, 0.4268, 0.5739, 0.7169, 0.6735, 0.7534, 0.6246, 0.6721, 0.5478, 0.3309, 0.5068]
    month_IoU = [0.4655850827693939, 0.5816534161567688, 0.5747901201248169, 0.6792684197425842, 0.7474293112754822, 0.7166976928710938, 0.7686308026313782, 0.6717162132263184, 0.7151005864143372, 0.6135199666023254, 0.5182576179504395, 0.5945199728012085]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    cmap = plt.colormaps.get_cmap('Dark2').resampled(len(month_counts.columns))
    ax1.grid(alpha=0.5)
    month_counts.plot(
        kind='bar',
        stacked=True,
        colormap=plt.cm.colors.ListedColormap(cmap.colors),
        ax=ax1, legend=False, zorder=3
    )
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Sample Count', fontsize=12)
    ax1.set_xticks(range(len(month_counts)))
    ax1.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)], rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(range(len(month_IoU)), month_IoU, 'o-', color='blue')
    ax2.set_ylabel('IoU', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 0.8)

    ax1.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        title='Year'
    )
    plt.savefig('stat_figures/count_per_month_per_year_w_iou.png', bbox_inches='tight', dpi=300)

#get_density_count_per_year()
get_month_count_w_month_iou_per_year()
