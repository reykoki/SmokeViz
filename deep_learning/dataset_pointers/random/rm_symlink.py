import pickle

old_prefix = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/Random/"
new_prefix = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/full_dataset/"

with open("random.pkl", "rb") as f:
    data_dict = pickle.load(f)

# walk through train/val/test and replace in-place for both truth and data
for split in ["train", "val", "test"]:
    for key in ["truth", "data"]:
        data_dict[split][key] = [
            fn.replace(old_prefix, new_prefix) for fn in data_dict[split][key]
        ]

print(data_dict[split][key][0])

with open("random_2.pkl", "wb") as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
