
dn = 1
#yrs = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
yrs = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
yrs = [2023]
days = []
interval = 60
while dn < 365:
    start = dn
    dn = dn + interval
    if dn > 365:
        dn = 365
    days.append((start,dn))
    dn = dn + 1
print(days)
for yr in yrs:
    for d in days:
        #print("sbatch --export=START_DN={},END_DN={},YEAR={} head.script;".format(d[0], d[1], yr))
        print("sbatch --export=START_DN={},END_DN={},YEAR={} compute.script;".format(d[0], d[1], yr))



