import os
import glob

# truth
#   2018
#     Light
#     Medium
#     Heavy

root_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/more_data/'
root_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz2/'
root_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL_SmokeViz/'
data_type = ['truth/', 'data/']
yrs = [ '2018/', '2019/', '2020/', '2021/', '2022/', '2023/', '2024/']
densities = ['Light/', 'Medium/', 'Heavy/']

for dt in data_type:
    for yr in yrs:
        for den in densities:
            den_path = root_dir + dt + yr + den
            if not os.path.exists(den_path):
                os.makedirs(den_path)

#other = [root_dir+'temp_png/', root_dir+'goes_temp/', root_dir+'smoke/', root_dir+'temp_data/', root_dir+'low_iou/']
other = []
other = [root_dir+'bad_img/', root_dir+'temp_data/', root_dir+'low_iou/']
for directory in other:
    if not os.path.exists(directory):
        os.makedirs(directory)
def list_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
#list_files(root_dir)

