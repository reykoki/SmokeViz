dn = 1
yr = 2020 #while dn < 365:
#    start = dn
#    end = dn + 28
#    print("sbatch --export=START={},END={},YEAR={} run.script;".format(start, end, yr))
#    dn = end

yrs = [2018, 2019, 2020, 2021, 2022, 2023]
yrs = [2019]
days = []
while dn < 365:
    start = dn
    dn = dn + 15
    if dn > 365:
        dn = 365
    days.append((start,dn))
print(days)
for yr in yrs:
    for d in days:
        print("sbatch --export=START={},END={},YEAR={} run.script;".format(d[0], d[1], yr))

