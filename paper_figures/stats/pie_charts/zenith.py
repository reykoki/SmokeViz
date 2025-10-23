import matplotlib.pyplot as plt


mie_counts = [85171, 114818]
pldr_counts = [96026, 67453]



def my_fmt_1(x):
    return '{:.2f}%\n({:.0f})'.format(x, (mie_counts[0]+mie_counts[1])*x/100)
def my_fmt_2(x):
    return '{:.2f}%\n({:.0f})'.format(x, (pldr_counts[0]+pldr_counts[1])*x/100)

labels = ['GOES-WEST\nIoU: 0.7270', 'GOES-EAST\nIoU: 0.6186']
colors = ['gold', 'goldenrod']
labels_full = ['<66° SZA', '≥66° SZA']

# Figure with 2 pies
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

axes[0].pie(mie_counts, labels=labels, colors=colors, autopct=my_fmt_1)
axes[0].set_title('Mie-Derived Dataset')

axes[1].pie(pldr_counts, labels=labels_full, colors=colors, autopct=my_fmt_2)
axes[1].set_title('SmokeViz Dataset')

plt.savefig('../stat_figures/pie_midday.png', bbox_inches='tight', dpi=300)
#plt.tight_layout()
#plt.show()
