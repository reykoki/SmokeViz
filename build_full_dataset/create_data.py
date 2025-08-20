import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyresample import create_area_def
from satpy import Scene
from satpy.writers import get_enhanced_image
from PIL import Image, ImageOps
import os
import skimage
import numpy as np
import pytz
from datetime import timedelta

def pick_temporal_smoke(smoke_shape, t_0, t_f):
    use_idx = []
    bounds = smoke_shape.bounds
    for idx, row in smoke_shape.iterrows():
        end = row['End']
        start = row['Start']
        # the ranges overlap if:
        if t_0-timedelta(minutes=10)<= end and start-timedelta(minutes=10) <= t_f:
            use_idx.append(idx)
    rel_smoke = smoke_shape.loc[use_idx]
    return rel_smoke

def save_data(RGB, fn_data, full_data_dir):
    total = np.sum(np.sum(RGB))
    if total > 100 and total < 1.5e5:
        skimage.io.imsave(fn_data, RGB)
        return True
    else:
        print("TOTAL SUM: ", total)
        fn = fn_data.split('/')[-1]
        bad_fn = "{}bad_img/{}".format(full_data_dir, fn)
        with open(bad_fn, 'w') as fp:
            pass
        return False

def plot_truth(x, y, lcc_proj, smoke, png_fn, img_shape):
    fig = plt.figure(figsize=(img_shape[2]/100, img_shape[1]/100), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)
    smoke.plot(ax=ax, facecolor='black')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(png_fn, dpi=100)
    plt.close(fig)
    img = Image.open(png_fn)
    bw = img.convert('1')
    bw = ImageOps.invert(bw)

    truth = np.asarray(bw).astype('i')
    os.remove(png_fn)
    return truth

def get_truth(x, y, lcc_proj, smoke, png_fn, tif_fn, img_shape, full_data_dir):

    low_smoke = smoke.loc[smoke['Density'] == 'Light']
    med_smoke = smoke.loc[smoke['Density'] == 'Medium']
    high_smoke = smoke.loc[smoke['Density'] == 'Heavy']

    # high = [1,1,1], med = [0, 1, 1], low = [0, 0, 1]
    low_truth = plot_truth(x, y, lcc_proj, low_smoke, png_fn, img_shape)
    med_truth = plot_truth(x, y, lcc_proj, med_smoke, png_fn, img_shape)
    high_truth = plot_truth(x, y, lcc_proj, high_smoke, png_fn, img_shape)
    low_truth += med_truth + high_truth
    low_truth = np.clip(low_truth, 0, 1)
    med_truth += high_truth
    med_truth = np.clip(med_truth, 0, 1)

    truth_layers = np.dstack([high_truth, med_truth, low_truth])
    if np.sum(truth_layers) > 0:
        skimage.io.imsave(tif_fn, truth_layers)
        return True
    else:
        fn = tif_fn.split('/')[-1]
        bad_fn = "{}bad_img/{}".format(full_data_dir, fn)
        with open(bad_fn, 'w') as fp:
            pass
        return False

def get_extent(center, rand_xy):
    cent_x = center.x+(rand_xy[0]*1e3) # multipy by 2km resolution
    cent_y = center.y+(rand_xy[1]*1e3)
    x0 = cent_x - 1.28e5
    y0 = cent_y - 1.28e5
    x1 = cent_x + 1.28e5
    y1 = cent_y + 1.28e5
    return [x0, y0, x1, y1]

def get_lcc_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                    central_latitude=38.5,
                                    standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                    semiminor_axis=6371229))
    return lcc_proj

def get_scn(fns, composite, extent):
    scn = Scene(reader='abi_l1b', filenames=fns)

    scn.load([composite], generate=False)
    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection=get_lcc_proj(),
                              resolution=1000,
                              area_extent=extent)
    new_scn = scn.resample(my_area)
    return new_scn

def create_data_truth(sat_fns, smoke, idx0, yr, dn, density, rand_xy, fn_head, full_data_dir):
    print('idx: ', idx0)
    lcc_proj = get_lcc_proj()
    smoke_lcc = smoke.to_crs(lcc_proj)
    centers = smoke_lcc.centroid
    center = centers.loc[idx0]
    composite = 'cimss_true_color_sunz_rayleigh'

    try:
        extent = get_extent(center, rand_xy)
    except:
        return fn_head
    try:
        scn = get_scn(sat_fns, composite, extent)
    except Exception as e:
        print(e)
        print('{} did not download, moving on'.format(sat_fns))
        for sat_fn in sat_fns:
            if os.path.exists(sat_fn):
                os.remove(sat_fn)
        return fn_head

    scan_start = pytz.utc.localize(scn[composite].attrs['start_time'])
    scan_end = pytz.utc.localize(scn[composite].attrs['end_time'])
    rel_smoke = pick_temporal_smoke(smoke_lcc, scan_start, scan_end)

    # make sure the smoke shape is within the bounds of the
    x = scn[composite].coords['x']
    y = scn[composite].coords['y']
    lon, lat = scn[composite].attrs['area'].get_lonlats()
    mid_pt = int(lon.shape[0]/2)
    lon_cent = np.round(lon[mid_pt, mid_pt], 2)
    lat_cent = np.round(lat[mid_pt, mid_pt], 2)
    fn_head = '{}_{}_{}_{}'.format(fn_head, lat_cent, lon_cent, idx0)

    corr_data = get_enhanced_image(scn[composite]).data.compute().data
    RGB = np.einsum('ijk->jki', corr_data)
    RGB[np.isnan(RGB)] = 0

    img_shape = scn[composite].shape

    png_fn_truth = full_data_dir + 'temp_png/truth_' + fn_head + '_{}'.format(idx0) + '.png'
    tif_fn_truth = full_data_dir + 'truth/{}/{}/{}/{}.tif'.format(yr, density, dn, fn_head)
    tif_fn_data = full_data_dir + 'data/{}/{}/{}/{}.tif'.format(yr, density, dn, fn_head)
    data_saved = save_data(RGB, tif_fn_data, full_data_dir)
    if data_saved:
        truth_saved  = get_truth(x, y, lcc_proj, rel_smoke, png_fn_truth, tif_fn_truth, img_shape, full_data_dir)
    del scn
    del RGB
    return

