import os
from matplotlib.colors import Normalize
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

#fp  = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/dataset_pointers/thresh/round3/thresh_1.pkl'

#with open(fp, 'rb') as handle:
#    data_dict = pickle.load(handle)

#file_list = data_dict['train']['truth']+ data_dict['val']['truth'] + data_dict['test']['truth']


all_yr = ['2018','2019','2020','2021','2022','2023','2024']

quad_iou = {"NW": 0.6765313744544983, "NE": 0.7336801886558533, "SW": 0.6445415616035461, "SE": 0.647982656955719}
quad_recall = {"NW": 0.7946525812149048, "NE": 0.848886251449585, "SW": 0.8253467082977295, "SE": 0.7578535079956055}
quad_precision = {"NW": 0.8198626637458801, "NE": 0.843897819519043, "SW": 0.7463364005088806, "SE": 0.8171705007553101}


#records = []
#for path in file_list:
#    m = re.search(r'_([-]?\d+\.\d+)_([-]?\d+\.\d+)_', path)
#    if m:
#        lat = float(m.group(1))
#        lon = float(m.group(2))
#        records.append({'Latitude': lat, 'Longitude': lon})

#lat_lon_df = pd.DataFrame(records)
#lat_lon_df.to_pickle("lat_lon_df.pkl")
lat_lon_df = pd.read_pickle('lat_lon_df.pkl')

states = geopandas.read_file('./shape_files/states.shp')
countries = geopandas.read_file('./shape_files/NA_countries.shp')

geometry= geopandas.points_from_xy(x=lat_lon_df['Longitude'],y=lat_lon_df['Latitude'], crs=states.crs)
gdf=geopandas.GeoDataFrame(lat_lon_df, geometry=geometry)

countries = countries[:2]

states_dict = {}
for idx, row in states.iterrows():
    state_dict = {row['shapeISO']: 0}
    states_dict.update(state_dict)

countries_dict = {}
for idx, row in countries.iterrows():
    if not (row.shapeName == 'United States'):
        country_dict = {row['shapeISO']: 0}
        countries_dict.update(country_dict)


for idx, row in states.iterrows():
    states_dict[row['shapeISO']] = len(gdf.geometry.clip(row['geometry']))
for idx, row in countries.iterrows():
    countries_dict[row['shapeISO']] = len(gdf.geometry.clip(row['geometry']))

#print(countries_dict)

#print(state_dict)

def assign_state_smoke_count(states):
    count = states_dict[states['shapeISO']]
    states['smoke_count'] = count
    return states
states = states.apply(assign_state_smoke_count, axis=1)




def assign_country_smoke_count(countries):
    count = countries_dict[countries['shapeISO']]
    countries['smoke_count'] = count
    return countries
countries = countries.apply(assign_country_smoke_count, axis=1)
#print(states)

total = float(
    np.nansum(states['smoke_count'].to_numpy())
    + np.nansum(countries['smoke_count'].to_numpy())
)
states = states.copy()
countries = countries.copy()
states['smoke_percent'] = 100.0 * states['smoke_count'] / total
countries['smoke_percent'] = 100.0 * countries['smoke_count'] / total


color = 'YlGnBu'
vmin = 0.0
vmax = max(states['smoke_percent'].max(), countries['smoke_percent'].max())

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

states.plot(column='smoke_percent', cmap=color, edgecolor='black', linewidth=0.5, ax=ax, legend=False, vmin=vmin, vmax=vmax)
countries.plot(column='smoke_percent', cmap=color, edgecolor='black', linewidth=0.5, ax=ax, legend=False, vmin=vmin, vmax=vmax)

ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)

ax.plot([-183, -45], [40, 40], 'k--', lw=1, alpha=0.8)
ax.plot([-105, -105], [-90, 90], 'k--', lw=1, alpha=0.8)

region_text = {
    'NW': (quad_iou['NW'], -183, 75),
    'SW': (quad_iou['SW'], -183, 16),
    'NE': (quad_iou['NE'], -67, 75),
    'SE': (quad_iou['SE'], -67, 16)
}
for region, (iou, lon, lat) in region_text.items():
    ax.text(lon, lat, f'{region} IoU:\n {iou:.4f}', fontsize=10, ha='left', color='black')

ax.set_xlim(-185, -47)
ax.set_ylim(10, 85)

cmap = plt.colormaps.get_cmap(color)
norm = Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

pos = ax.get_position()
cbar_width = pos.width * 0.8     # 80% of the map width
cbar_height = 0.025              # thin horizontal bar
cbar_left = pos.x0 + (pos.width - cbar_width) / 2  # center it horizontally
cbar_bottom = pos.y0 - 0.01      # move below the map (tweak as needed)

cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('percent of total samples (%)')

#oplt.tight_layout(rect=[0, 0.05, 1, 1])  # keep 5% margin at bottom
plt.subplots_adjust(bottom=0.15)  # increase bottom margin fraction
#plt.show()

plt.savefig('../stat_figures/sample_percent_per_state_MX_CA_w_quad_iou.png', bbox_inches='tight', dpi=300)

