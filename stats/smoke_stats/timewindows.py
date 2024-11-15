import pickle
import geopandas
from datetime import datetime
import pytz

global smoke_dir
smoke_dir = "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/smoke/"

def get_smoke(dt):
    tt = dt.timetuple()
    month = str(tt.tm_mon).zfill(2)
    day = str(tt.tm_mday).zfill(2)
    yr = str(tt.tm_year)
    fn = 'hms_smoke{}{}{}.zip'.format(yr, month, day)
    smoke_shape_fn = smoke_dir + fn
    smoke = geopandas.read_file(smoke_shape_fn)
    return smoke

def smoke_utc(time_str):
    fmt = '%Y%j %H%M'
    return pytz.utc.localize(datetime.strptime(time_str, fmt))

def main():
    years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']
    tw_dict = {'2018':{'total': 0, 'average': 0, 'num_samples':0},
               '2019':{'total': 0, 'average': 0, 'num_samples':0},
               '2020':{'total': 0, 'average': 0, 'num_samples':0},
               '2021':{'total': 0, 'average': 0, 'num_samples':0},
               '2022':{'total': 0, 'average': 0, 'num_samples':0},
               '2023':{'total': 0, 'average': 0, 'num_samples':0},
               '2024':{'total': 0, 'average': 0, 'num_samples':0},
               'all':{'total': 0, 'average': 0, 'num_samples':0}}

    max_window = 21600
    windows = []
    for yr in years:
        dn = 1
        num_days = 365
        dns = list(range(dn, dn+num_days+1))
        for dn in dns:
            s = '{}/{}'.format(yr, dn)
            fmt = '%Y/%j'
            dt = pytz.utc.localize(datetime.strptime(s, fmt))
            month = dt.strftime('%m')
            day = dt.strftime('%d')
            try:
                smoke = get_smoke(dt)
                smoke['Start'] = smoke['Start'].apply(smoke_utc)
                smoke['End'] = smoke['End'].apply(smoke_utc)
                for idx, row in smoke.iterrows():
                    diff = smoke.loc[idx]['End'] - smoke.loc[idx]['Start']
                    if diff.seconds > max_window:
                        windows.append(diff.seconds)
                        max_window = diff.seconds
                        print(diff.seconds)
                    tw_dict[yr]['total']+=diff.seconds
                    tw_dict[yr]['num_samples']+=1
                    #if idx == 10:
                    #   break
            except:
                pass

        tw_dict['all']['total'] += tw_dict[yr]['total']
        tw_dict['all']['num_samples'] += tw_dict[yr]['num_samples']

        tw_dict[yr]['average'] = tw_dict[yr]['total']/tw_dict[yr]['num_samples']


        print('{}: {}'.format(yr, tw_dict[yr]['average']))
    tw_dict['all']['average'] = tw_dict['all']['total']/tw_dict['all']['num_samples']
    print(windows)
    print(len(windows))
    print(max_window)

    print('all: {}'.format(tw_dict['all']['average']))
    with open('timewindow_dict.pkl', 'wb') as handle:
        pickle.dump(tw_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    main()

