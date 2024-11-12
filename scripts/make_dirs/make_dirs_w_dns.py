import os
import glob

# truth
#   2018
#     Light
#     Medium
#     Heavy

root_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/more_data/'
root_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz2/'
root_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/'
root_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL2/'
data_type = ['truth/', 'data/']
yrs = [ '2018/', '2019/', '2020/', '2021/', '2022/', '2023/', '2024/']
densities = ['Light/', 'Medium/', 'Heavy/']

dns = [str(x).zfill(3) for x in range(1,366)]


for dt in data_type:
    for yr in yrs:
        for den in densities:
            for dn in dns:
                dn_path = root_dir + dt + yr + den + dn
                if not os.path.exists(dn_path):
                    os.makedirs(dn_path)


#other = [root_dir+'temp_png/', root_dir+'goes_temp/', root_dir+'smoke/', root_dir+'temp_data/', root_dir+'low_iou/']
#other = [root_dir+'bad_img/', root_dir+'temp_data/', root_dir+'low_iou/']
other = []
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

