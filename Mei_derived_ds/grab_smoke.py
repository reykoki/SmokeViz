import geopandas
import os
import shutil
import wget

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
