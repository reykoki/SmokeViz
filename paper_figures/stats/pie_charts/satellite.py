import matplotlib.pyplot as plt

test_counts = [5818, 12671]
full_counts = [35208, 128271]

def my_fmt_1(x):
    return '{:.2f}%\n({:.0f})'.format(x, (test_counts[0]+test_counts[1])*x/100)
def my_fmt_2(x):
    return '{:.2f}%\n({:.0f})'.format(x, (full_counts[0]+full_counts[1])*x/100)

labels = ['GOES-WEST\nIoU: 0.7270', 'GOES-EAST\nIoU: 0.6186']
colors = ['lightskyblue', 'lightsalmon']
labels_full = ['GOES-WEST', 'GOES-EAST']

# Figure with 2 pies
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

axes[0].pie(test_counts, labels=labels, colors=colors, autopct=my_fmt_1)
axes[0].set_title('Test Set Distribution')

axes[1].pie(full_counts, labels=labels_full, colors=colors, autopct=my_fmt_2)
axes[1].set_title('Full Dataset Distribution')
plt.savefig('../stat_figures/pie_satellite_performance.png', bbox_inches='tight', dpi=300)
#plt.tight_layout()
#plt.show()
