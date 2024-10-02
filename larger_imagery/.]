from suntime import Sun
from datetime import timedelta

def closest_to_sunrise(st,et,actual_sunrise,bounds):
    west_lon = bounds['maxx']
    #print("WEST_LON: ", west_lon)
    if west_lon > -85:
        sat = '16'
        delay_time = 30
    else:
        delay_time = 1.1 * west_lon + 209
        sat = '17'
    sunrise = actual_sunrise + timedelta(minutes=delay_time)
    if et == st:
        return sat, st
    elif st >= sunrise:
        return sat, st
    elif st < sunrise and et >= sunrise:
        return sat, sunrise
    elif et <= sunrise and et >= actual_sunrise:
        return sat, et
    else:
        print('THERE IS AN ERROR FOR SUNRISE')
        print('et: ', et)
        print('st: ', st)
        print('sunrise: ', sunrise)
        return None, None

def closest_to_sunset(st, et, sunset):
    sunset = sunset - timedelta(minutes=25)
    if et == st:
        return '16', et
    elif et <= sunset:
        return '16', et - timedelta(minutes=5)
    elif et > sunset and st <= sunset:
        return '16', sunset
    else:
        print('THERE IS AN ERROR FOR SUNSET')
        print('et: ', et)
        print('st: ', st)
        print('sunset: ', sunset)
        return None, None

def closer_east_west(bounds, st, et):
    # if closer to west coast:
    if bounds['minx'] < -98:
        sat = '16'
        best_time = st
    else:
        sat = '17'
        best_time = et
    return sat, best_time

def get_ss(bounds, st, et):
    try:
        east = Sun(bounds['maxy'], bounds['maxx'])
        sr_dt_st = {-1: abs(st - east.get_sunset_time(st+timedelta(days=-1))),
                     0: abs(st - east.get_sunset_time(st+timedelta(days=0))),
                     1: abs(st - east.get_sunset_time(st+timedelta(days=1)))}
        sr_dt_et = {-1: abs(et - east.get_sunset_time(et+timedelta(days=-1))),
                     0: abs(et - east.get_sunset_time(et+timedelta(days=0))),
                     1: abs(et - east.get_sunset_time(et+timedelta(days=1)))}
    except Exception as e:
        print(e)
        try:
            # actually west
            east = Sun(bounds['maxy'], bounds['minx'])
            sr_dt_st = {-1: abs(st - east.get_sunset_time(st+timedelta(days=-1))),
                         0: abs(st - east.get_sunset_time(st+timedelta(days=0))),
                         1: abs(st - east.get_sunset_time(st+timedelta(days=1)))}
            sr_dt_et = {-1: abs(et - east.get_sunset_time(et+timedelta(days=-1))),
                         0: abs(et - east.get_sunset_time(et+timedelta(days=0))),
                         1: abs(et - east.get_sunset_time(et+timedelta(days=1)))}
        except Exception as e:
            print(e)
            return None, None
    st_dt = min(sr_dt_st, key=sr_dt_st.get)
    et_dt = min(sr_dt_et, key=sr_dt_et.get)
    if sr_dt_st[st_dt] > sr_dt_et[et_dt]:
        return east.get_sunset_time(et+timedelta(days=et_dt)), sr_dt_et[et_dt]
    return east.get_sunset_time(et+timedelta(days=st_dt)), sr_dt_st[st_dt]

def get_sr(bounds, st, et):
    try:
        west = Sun(bounds['maxy'], bounds['minx'])
        sr_dt_st = {-1: abs(st - west.get_sunrise_time(st+timedelta(days=-1))),
                     0: abs(st - west.get_sunrise_time(st+timedelta(days=0))),
                     1: abs(st - west.get_sunrise_time(st+timedelta(days=1)))}
        sr_dt_et = {-1: abs(et - west.get_sunrise_time(et+timedelta(days=-1))),
                     0: abs(et - west.get_sunrise_time(et+timedelta(days=0))),
                     1: abs(et - west.get_sunrise_time(et+timedelta(days=1)))}
    except Exception as e:
        print(e)
        try:
            #actually east
            west = Sun(bounds['maxy'], bounds['maxx'])
            sr_dt_st = {-1: abs(st - west.get_sunrise_time(st+timedelta(days=-1))),
                         0: abs(st - west.get_sunrise_time(st+timedelta(days=0))),
                         1: abs(st - west.get_sunrise_time(st+timedelta(days=1)))}
            sr_dt_et = {-1: abs(et - west.get_sunrise_time(et+timedelta(days=-1))),
                         0: abs(et - west.get_sunrise_time(et+timedelta(days=0))),
                         1: abs(et - west.get_sunrise_time(et+timedelta(days=1)))}
        except Exception as e:
            print(e)
            return None, None

    st_dt = min(sr_dt_st, key=sr_dt_st.get)
    et_dt = min(sr_dt_et, key=sr_dt_et.get)
    if sr_dt_st[st_dt] > sr_dt_et[et_dt]:
        return west.get_sunrise_time(et+timedelta(days=et_dt)), sr_dt_et[et_dt]
    return west.get_sunrise_time(st+timedelta(days=st_dt)), sr_dt_st[st_dt]

def get_best_time(st, et, bounds):
    sunrise, sr_dt = get_sr(bounds, st, et)
    sunset, ss_dt  = get_ss(bounds, st, et)
    # no sunrise or sunset (assuming sun isnt setting)
    if sr_dt == None or ss_dt == None:
        sat, best_time = closer_east_west(bounds, st, et)
    # times are closer to sunset
    elif sr_dt >= ss_dt:
        sat, best_time = closest_to_sunset(st,et,sunset)
    else:
        sat, best_time = closest_to_sunrise(st,et,sunrise,bounds)
    return sat, best_time

