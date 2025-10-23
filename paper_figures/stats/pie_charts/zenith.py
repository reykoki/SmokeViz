import matplotlib.pyplot as plt


mie_counts = [99382, 100607]
pldr_counts = [110595, 52884]


def my_fmt_1(x):
    return '{:.2f}%\n({:.0f})'.format(x, (mie_counts[0]+mie_counts[1])*x/100)
def my_fmt_2(x):
    return '{:.2f}%\n({:.0f})'.format(x, (pldr_counts[0]+pldr_counts[1])*x/100)

labels_Mie = ['<70° SZA\nIoU: 0.4793', '≥70° SZA\nIoU: 0.4471']
colors = ['gold', 'goldenrod']
labels_pldr = ['<70° SZA\nIoU: 0.6661', '≥70° SZA\nIoU: 0.6639']

# Figure with 2 pies
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

axes[0].pie(mie_counts, labels=labels_Mie, colors=colors, autopct=my_fmt_1, startangle=90)
axes[0].set_title('Mie-Derived Dataset')

axes[1].pie(pldr_counts, labels=labels_pldr, colors=colors, autopct=my_fmt_2, startangle=90)
axes[1].set_title('SmokeViz Dataset')

plt.savefig('../stat_figures/pie_midday.png', bbox_inches='tight', dpi=300)
#plt.tight_layout()
#plt.show()
