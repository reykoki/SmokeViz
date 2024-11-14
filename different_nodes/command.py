
yrs = [2018, 2019, 2020, 2021, 2022, 2023]
yrs = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
yrs = [2022]
for yr in yrs:
    dn = 2
    while dn < 10:
        dn = dn + 1
        print("bash --export=DN={},YEAR={} two_nodes.script".format(dn, yr))

