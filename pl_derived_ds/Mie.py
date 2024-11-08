from pyorbital import astronomy
import numpy as np
import pyproj
from datetime import datetime
from datetime import timedelta
import pytz
import cartopy.crs as ccrs

def get_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                     central_latitude=38.5,
                                     standard_parallels=(38.5, 38.5),
                                     globe=ccrs.Globe(semimajor_axis=6371229,
                                                      semiminor_axis=6371229))

    return lcc_proj


def get_closest_to(dt, min_diff):
    approx = round(dt.minute/min_diff) * min_diff
    dt = dt.replace(minute=0)
    dt += timedelta(seconds=approx * 60)
    return dt

# get time list from time window
def get_time_list(start_dt, end_dt):
    M3_to_M6 = pytz.utc.localize(datetime(2019, 4, 1, 0, 0)) # April 2019 switch from Mode 3 to Mode 6 (every 15 to 10 mins)
    if end_dt < M3_to_M6:
        min_diff = 15 #M6
    else:
        min_diff = 10 #M3
    t = get_closest_to(start_dt, min_diff)
    time_list = [t]
    while t < end_dt:
        t += timedelta(minutes=min_diff)
        time_list.append(t)
    return time_list

# get western and eastern most points lat/lon
def west_east_lat_lon(lat, lon, res, img_size): # img_size - number of pixels
    lcc_proj = pyproj.Proj(get_proj())
    x, y = lcc_proj(lon,lat)
    dist = int(img_size/2*res)
    lon_w, lat_w = lcc_proj(x-dist, y-dist, inverse=True) # lower left
    lon_e, lat_e = lcc_proj(x+dist, y+dist, inverse=True) # upper right
    return (lat_w, lon_w), (lat_e, lon_e)

def get_best_sat_from_szas(w_coords, e_coords, img_times):
    mid_time = img_times[int(len(img_times)/2)]
    sza_west = astronomy.sun_zenith_angle(mid_time, w_coords[1], w_coords[0])
    sza_east = astronomy.sun_zenith_angle(mid_time, e_coords[1], e_coords[0])
    # closer to sunrise
    if sza_west >= sza_east:
        lat, lon = w_coords[0], w_coords[1]
        sat = '17'
    else:
        lat, lon = e_coords[0], e_coords[1]
        sat = '16'
    return sat, lat, lon

def get_best_time_from_szas(lat, lon, img_times):
    szas = np.zeros(len(img_times))
    for i, t in enumerate(img_times):
        szas[i] = astronomy.sun_zenith_angle(t, lon, lat)
    thresh = 88
    szas[szas>thresh]=1e5 #arbitrarily large number
    best_time_idx = (np.abs(szas - thresh)).argmin()
    best_time = img_times[best_time_idx]
    return best_time

def sza_best_time(lat, lon, start_dt, end_dt, res=1000, img_size=256):
    img_times = get_time_list(start_dt, end_dt)
    #print(img_times)
    w_coords, e_coords = west_east_lat_lon(lat, lon, res, img_size)
    sat, lat, lon = get_best_sat_from_szas(w_coords, e_coords, img_times)
    best_time = get_best_time_from_szas(lat, lon, img_times)
    return sat, best_time

# valid times are when the furthest lat/lon away from the sat has a solar zenith angle <90
def valid_times_from_szas(lat, lon, img_times):
    szas = np.zeros(len(img_times))
    for i, t in enumerate(img_times):
        szas[i] = astronomy.sun_zenith_angle(t, lon, lat)
    thresh = 90
    valid_indices = np.where(szas<thresh)[0]
    valid_times = []
    for idx in valid_indices:
        valid_times.append(img_times[idx])
    return valid_times

def sza_sat_valid_times(lat, lon, start_dt, end_dt, res=1000, img_size=256):
    img_times = get_time_list(start_dt, end_dt)
    w_coords, e_coords = west_east_lat_lon(lat, lon, res, img_size)
    sat, lat, lon = get_best_sat_from_szas(w_coords, e_coords, img_times)
    valid_times = valid_times_from_szas(lat,lon, img_times)
    return sat, valid_times


