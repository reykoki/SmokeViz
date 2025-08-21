yr = 2024
yrs = [2018, 2019, 2020, 2021, 2022, 2023]
dn = 1
for yr in yrs:
    while dn < 365:
        start = dn
        end = dn + 28
        print("sbatch --export=START={},END={},YEAR={} head.script;".format(start, end, yr))
        dn = end

