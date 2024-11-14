from pyorbital import astronomy
import pyproj
import cartopy.crs as ccrs

def get_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                     central_latitude=38.5,
                                     standard_parallels=(38.5, 38.5),
                                     globe=ccrs.Globe(semimajor_axis=6371229,
                                                      semiminor_axis=6371229))

    return lcc_proj

def west_east_lat_lon(lat, lon, res, img_size): # img_size - number of pixels
    lcc_proj = pyproj.Proj(get_proj())
    x, y = lcc_proj(lon,lat)
    dist = int(img_size/2*res)
    lon_w, lat_w = lcc_proj(x-dist, y-dist, inverse=True) # lower left
    lon_e, lat_e = lcc_proj(x+dist, y+dist, inverse=True) # upper right
    return (lat_w, lon_w), (lat_e, lon_e)

def get_sza(lat, lon, img_time, res=1000, img_size=256):
    w_coords, e_coords = west_east_lat_lon(lat, lon, res, img_size)
    sza_mid = astronomy.sun_zenith_angle(img_time, lon, lat)
    sza_west = astronomy.sun_zenith_angle(img_time, w_coords[1], w_coords[0])
    sza_east = astronomy.sun_zenith_angle(img_time, e_coords[1], e_coords[0])
    print('solar zenith West:   ', sza_west)
    print('solar zenith middle: ', sza_mid)
    print('solar zenith East:   ', sza_east)
    return

