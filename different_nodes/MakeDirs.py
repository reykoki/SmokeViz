import os
import glob

# truth
#   2018
#     Light
#     Medium
#     Heavy

class MakeDirs():
    def __init__(self, root_dir, yr):
        data_type = ['coords/', 'truth/', 'data/']
        densities = ['/Light/', '/Medium/', '/Heavy/']
        for dt in data_type:
            for den in densities:
                den_path = root_dir + dt + yr + den
                if not os.path.exists(den_path):
                    os.makedirs(den_path)
        other = [root_dir+'temp_png/', root_dir+'goes_temp/', root_dir+'smoke/']
        for directory in other:
            if not os.path.exists(directory):
                os.makedirs(directory)
        for root, dirs, files in os.walk(root_dir):
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
