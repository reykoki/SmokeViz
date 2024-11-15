
import pickle
with open('timewindow_dict.pkl', 'rb') as handle:
    tw_dict = pickle.load(handle)

print(tw_dict)
total_num_imgs = tw_dict['all']['total']/60/10

print('total num imgs')
print(total_num_imgs)

av = tw_dict['all']['average']
print('average minutes')
print(av/60)
print('average hour mins')

print(av%60)

