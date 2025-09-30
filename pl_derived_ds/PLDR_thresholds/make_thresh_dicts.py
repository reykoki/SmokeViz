import pickle
import glob
import numpy as np
import os

# Directory with PLDR pickle files
pkl_dir = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/PLDR_IoUs/round1"
# Directory to save thresholded .npy files
save_dir = "./PLDR_thresh_lists/"
os.makedirs(save_dir, exist_ok=True)

# IoU thresholds to filter by
thresholds = [.005, .01, .05, .1, .15, .2, .25, .3, .35, .4]

# Initialize dictionary per threshold
threshold_dict = {thr: {} for thr in thresholds}

# Find all PLDR pickle files
pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "PLDR_*.pkl")))

for pkl_file in pkl_files:
    # Extract year from filename, e.g., PLDR_2018_123.pkl -> 2018
    base_name = os.path.basename(pkl_file)
    year_str = base_name.split("_")[1]
    year = int(year_str)

    # Load the per-day PLDR pickle
    with open(pkl_file, "rb") as f:
        pl_dict = pickle.load(f)

    # Iterate over smoke annotations
    for smoke_idx, smoke_data in pl_dict.items():
        best_iou = smoke_data.get('best_IoU', 0)
        best_fn = smoke_data.get('best_IoU_fn', None)
        if best_fn is None:
            continue  # skip if no valid file

        # Append filename to every threshold it meets
        for thr in thresholds:
            if best_iou >= thr:
                if year not in threshold_dict[thr]:
                    threshold_dict[thr][year] = []
                threshold_dict[thr][year].append(best_fn)

# Remove duplicates and save each threshold dictionary
for thr, year_dict in threshold_dict.items():
    num_fns = 0
    for yr in year_dict:
        year_dict[yr] = list(set(year_dict[yr]))  # remove duplicates
        num_fns += len(year_dict[yr])

    save_name = os.path.join(save_dir, f"PLDR_{str(thr).replace('.', '')[1:]}.pkl")
    with open(save_name, 'wb') as f:
        pickle.dump(year_dict, f)
    print(f"Saved threshold {thr} dict with {num_fns} samples to {save_name}")
