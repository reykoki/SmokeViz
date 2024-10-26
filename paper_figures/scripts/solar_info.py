from glob import glob
import numpy as np
import time
import pytz
from suntime import Sun
from datetime import datetime
from datetime import timedelta
from datetime import date

def get_sunrise_sunset(dt, lat, lon):
    loc = Sun(lat, lon)
    sunrise = loc.get_sunrise_time(dt)
    sunset = loc.get_sunset_time(dt)
    #print('-------')
    #print('sunrise:', sunrise)
    #print('sunset:  ', sunset)
    #print('-------')
    sunrise_prev = loc.get_sunrise_time(dt-timedelta(days=1))
    sunset_prev = loc.get_sunset_time(dt-timedelta(days=1))
    sunrise_next = loc.get_sunrise_time(dt+timedelta(days=1))
    sunset_next = loc.get_sunset_time(dt+timedelta(days=1))

    if sunset > sunrise:
        mid_prev_ss_sr = sunset_prev + (sunrise - sunset_prev)/2
        mid_ss_next_sr = sunset + (sunrise_next - sunset)/2
        if dt < mid_prev_ss_sr:
            sunset = sunset_prev
            sunrise = sunrise_prev
        if dt > mid_ss_next_sr:
            sunrise = sunrise_next
            sunset = sunset_next
    elif sunset < sunrise:
        # find middle of night between sunset and next sunrise
        mid_ss_sr = sunset + (sunrise - sunset)/2
        if dt < mid_ss_sr:
            sunrise = sunrise_prev
        elif dt > mid_ss_sr:
            sunset = sunset_next

    print('sunrise diff: ', abs(dt - sunrise))
    print('sunset diff:  ', abs(dt - sunset))
    return sunrise, sunset

def get_sunrise_sunset_orig(dt, lat, lon):
    loc = Sun(lat, lon)
    sunset = loc.get_sunset_time(dt)
    sunrise = loc.get_sunrise_time(dt)

    if sunrise > dt:
        sunrise = loc.get_sunrise_time(dt - timedelta(days=1))
    if sunset < dt:
        sunset = loc.get_sunset_time(dt + timedelta(days=1))

    #print('sample time: ', dt)
    #print("sunrise:     ", sunrise)
    #print("sunset:      ", sunset)
    print('sunrise diff: ', abs(dt - sunrise))
    print('sunset diff:  ', abs(dt - sunset))
    return sunrise, sunset
