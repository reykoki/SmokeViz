from pyorbital import astronomy
import pyproj
from datetime import timedelta

# get time list from time window
def get_time_list(start_dt, end_dt):
    t = start_dt
    time_list = [t]
    while t < end_dt:
        t += timedelta(minutes=10)
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

def get_best_time_from_szas(dt, w_coords, e_coords, img_times):
    mid_time = img_times[int(len(img_times)/2)]
    sza_west = astronomy.sun_zenith_angle(mid_time, w_coords[0], w_coords[1])
    sza_east = astronomy.sun_zenith_angle(mid_time, e_coords[0], e_coords[1])
    # closer to sunrise
    if sza_west >= sza_east:
        lat, lon = w_coords[0], w_coords[1]
        sat = '17'
    else:
        lat, lon = e_coords[0], e_coords[1]
        sat = '16'
    szas = np.zeros(len(img_times))
    for i, t in enumerate(img_times):
        szas[i] = astronomy.sun_zenith_angle(t, lat, lon)
    thresh = 90
    szas[szas>thresh]=1e5 #arbitrarily large number
    best_time_idx = (np.abs(szas - thresh)).argmin()
    best_time = img_times[best_time_idx]
    return sat, best_time

def sza_best_time(lat, lon, start_dt, end_dt, res=1000, img_size=256):
    img_times = get_time_list(start_dt, end_dt)
    #print(img_times)
    w_coords, e_coords = west_east_lat_lon(lat, lon, res, img_size)
    sat, best_time = get_best_time_from_szas(dt, w_coords, e_coords, img_times)
    return sat, best_time

