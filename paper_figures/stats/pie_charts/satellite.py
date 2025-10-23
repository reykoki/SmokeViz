import matplotlib.pyplot as plt

G16 = {'test_count': 12671,
       'test_percent': 12671/(5818+12671),
       'iou': 0.6185824275016785,
       'full_count': 128271,
       'full_percent': 128271/(35208+128271),
       }
G17 = {'test_count': 5818,
       'test_percent': 5818/(5818+12671),
       'iou': 0.7270441651344299,
       'full_count': 35208,
       'full_percent': 128271/(35208+128271),
       }

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), dpi=300)
colors = ['#a7d4df', '#ffa37d']  # light blue (West), salmon (East)

# Test set pie
axes[0].pie(
    [G17['test_count'], G16['test_count']],
    labels=['GOES-WEST', 'GOES-EAST'],
    colors=colors,
    autopct=lambda p: f"{p:.1f}%\n({int(p/100*(G17['test_count']+G16['test_count'])):,})",
    startangle=0
)
axes[0].set_title("Test Set Distribution", fontsize=11)
axes[0].text(-1.4, 0.6, f"GOES-WEST\nIoU: {G17['iou']:.4f}", fontsize=8)
axes[0].text(1.3, -0.3, f"GOES-EAST\nIoU: {G16['iou']:.4f}", fontsize=8)

# Full dataset pie
axes[1].pie(
    [G17['full_count'], G16['full_count']],
    labels=['GOES-WEST', 'GOES-EAST'],
    colors=colors,
    autopct=lambda p: f"{p:.1f}%\n({int(p/100*(G17['full_count']+G16['full_count'])):,})",
    startangle=0
)
axes[1].set_title("Full Dataset Distribution", fontsize=11)
axes[1].text(-1.3, 0.6, f"GOES-WEST\nIoU: {G17['iou']:.4f}", fontsize=8)
axes[1].text(1.3, -0.3, f"GOES-EAST\nIoU: {G16['iou']:.4f}", fontsize=8)

plt.tight_layout()
#plt.show()
plt.savefig('../stat_figures/pie_satellite_performance.png', bbox_inches='tight', dpi=300)
