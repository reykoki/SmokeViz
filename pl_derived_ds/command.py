dn = 1

yrs = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
yrs = [2020]
days = []
#while dn < 300:
while dn < 365:
    start = dn
    dn = dn + 9
    if dn > 365:
        dn = 365
    days.append((start,dn))
    dn = dn + 1
print(days)
for yr in yrs:
    for d in days:
        print("bash two_nodes.script --start={} --end={} --year={}".format(d[0], d[1], yr))

