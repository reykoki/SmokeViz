import glob
import random
import geopandas
import os
import shutil
import wget
from datetime import datetime
import pytz
from datetime import timedelta
import pandas as pd
import cartopy.crs as ccrs
import pyproj
import random
from suntime import Sun, SunTimeException
pd.options.mode.copy_on_write = True


def use_existing_goes(yr, dn, dt):
    goes_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/GOES/'
    yr = dt.year
    dn = dt.strftime('%j')
    goes_loc = '{}{}/{}/'.format(goes_dir, yr, dn)
    existing_goes_fns = glob.glob("{}*C01_G*_s{}{}*.nc".format(goes_loc, yr, dn))
    goes_fn = random.choice(existing_goes_fns)
    HHMM = goes_fn.split('_s{}{}'.format(yr, dn))[-1][0:4]
    start_end = '{}{} {}'.format(yr, dn, HHMM)
    return start_end

def get_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                     central_latitude=38.5,
                                     standard_parallels=(38.5, 38.5),
                                     globe=ccrs.Globe(semimajor_axis=6371229,
                                                      semiminor_axis=6371229))
    return lcc_proj

def get_smoke_fn_url(dt):
    tt = dt.timetuple()
    month = str(tt.tm_mon).zfill(2)
    day = str(tt.tm_mday).zfill(2)
    yr = str(tt.tm_year)
    fn = 'hms_smoke{}{}{}.zip'.format(yr, month, day)
    url = 'https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/{}/{}/{}'.format(yr, month, fn)
    return fn, url

def get_smoke(dt, smoke_dir):
    fn, url = get_smoke_fn_url(dt)
    smoke_shape_fn = smoke_dir + fn
    print(smoke_shape_fn)
    if os.path.exists(smoke_shape_fn):
        print("{} already exists".format(fn))
    else:
        print('DOWNLOADING SMOKE: {}'.format(fn))
        filename = wget.download(url, out=smoke_dir)
        shutil.unpack_archive(filename, smoke_dir)
    smoke = geopandas.read_file(smoke_shape_fn)
    return smoke

def three_day_smoke(dt):
    smoke_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/smoke/'
    prev_day = dt - timedelta(days=1)
    next_day = dt + timedelta(days=1)
    smoke_prev = get_smoke(prev_day, smoke_dir)
    smoke_o = get_smoke(dt, smoke_dir)
    smoke_next = get_smoke(next_day, smoke_dir)
    smoke = pd.concat([smoke_prev, smoke_o, smoke_next])
    smoke['Start'] = prev_day.strftime('%Y%j %H%M')
    smoke['End'] = next_day.strftime('%Y%j %H%M')
    return smoke, smoke_o

def rand_t_sr_ss(sr, ss):
    delta = ss - sr
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    if int_delta>0:
        random_second = random.randrange(int_delta)
    else:
        random_second = 0
    return sr + timedelta(seconds=random_second)

def get_time(smoke_null,idx, dt):
    lat = smoke_null.loc[idx]['geometry'].y
    lon = smoke_null.loc[idx]['geometry'].x
    try:
        sun = Sun(lat, lon)
        sr = sun.get_sunrise_time(dt)
        ss = sun.get_sunset_time(dt)
        if ss < sr:
            ss = sun.get_sunset_time(dt +timedelta(hours=24))
        print(sr, ss)
        rand_t = rand_t_sr_ss(sr, ss)
    except:
        start = datetime.strptime(smoke_null.loc[idx]['Start'], "%Y%j %H%M")
        end = datetime.strptime(smoke_null.loc[idx]['End'], "%Y%j %H%M")
        rand_t = rand_t_sr_ss(start, end)
    rand_t = rand_t.strftime('%Y%j %H%M')
    return rand_t

def get_rand_pts(num_null_samples, no_smoke_area, smoke_null):
    pts = []
    for n in range(num_null_samples):
        pts.append(no_smoke_area.sample_points(1)[0])
    smoke_null = smoke_null.set_geometry(pts)
    return smoke_null

def get_null_smoke(yr, dn, dt, res=1000, size=256):
    smoke, smoke_o = three_day_smoke(dt)
    smoke = smoke.to_crs(get_proj())
    aoi = geopandas.read_file('/scratch1/RDARCH/rda-ghpcs/Rey.Koki/shapefiles/Can_US_Mex.shp')
    aoi = aoi.to_crs(get_proj())
    smoke['geometry']= smoke.buffer(res*size) # 256e3 meters
    no_smoke_area = aoi.overlay(smoke, how='difference')
    no_smoke_area = no_smoke_area.to_crs("EPSG:4326")
    #no_smoke_area.plot()
    num_null_samples = 20
    smoke_null = geopandas.GeoDataFrame({"Start": ["0"]*num_null_samples,
                            "End": ["0"]*num_null_samples,
                            "Density": ["Null"]*num_null_samples,
                            "geometry": no_smoke_area.sample_points(num_null_samples).explode(ignore_index=True)})
    smoke_null.index=["null_" + str(i) for i in range(num_null_samples)]
    for idx in smoke_null.index:
        #start_end = get_time(smoke_null,idx, dt)
        start_end = use_existing_goes(yr, dn, dt)
        smoke_null.loc[idx,'Start'] = start_end
        smoke_null.loc[idx,'End'] = start_end

    return smoke_null

def iter_smoke_null(date):
    dn = 1
    yr = 2021
    dt = pytz.utc.localize(datetime.strptime("{}{}".format(yr,dn), '%Y%j'))
    null_smoke = get_null_smoke(yr, dn, dt)

iter_smoke_null(None)
