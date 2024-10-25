from datetime import datetime
from glob import glob
import geopandas
import pytz

data_dir = './data/'

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
