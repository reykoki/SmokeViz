from datetime import datetime
from glob import glob
import geopandas
import pytz

data_dir = './data/'

def coords_from_fn(fn, res=1000, img_size=512): # img_size - number of pixels
    fn_split = fn.split('.tif')[0].split('_')
    lat = fn_split[-2]
    lon = fn_split[-1]
    lcc_proj = pyproj.Proj(get_proj())
    x, y = lcc_proj(lon,lat)
    dist = int(img_size/2*res)
    lon_0, lat_0 = lcc_proj(x-dist, y-dist, inverse=True) # lower left
    lon_1, lat_1 = lcc_proj(x+dist, y+dist, inverse=True) # upper right
    lats = np.linspace(lat_1, lat_0, 5)
    lons = np.linspace(lon_0, lon_1, 5)
    return lats, lons

def get_dt_from_fn(fn):
    start = fn.split('_')[1][1:-1]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    dt = pytz.utc.localize(start_dt)
    return dt

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

def get_dt(input_dt):
    fmt = '%Y/%m/%d %H:%M'
    dt = datetime.strptime(input_dt, fmt)
    dt = pytz.utc.localize(dt)
    return dt

def get_fns_from_dt(dt):
    hr, dn, yr = get_dt_str(dt)
    goes_dir = data_dir + 'goes/'
    fns = glob(goes_dir + '*C0[123]*_s{}{}{}*'.format(yr,dn,hr))
    print(fns)
    return fns

# get state shape object
def get_states(proj):
    state_shape = './data/shape_files/cb_2018_us_state_500k.shp'
    states = geopandas.read_file(state_shape)
    states = states.to_crs(proj)
    return states
