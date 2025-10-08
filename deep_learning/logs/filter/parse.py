import numpy as np
import pickle
import re

# thresholds you want on axes
train_thresholds = np.array([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
test_thresholds  = np.array([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

# initialize empty matrices
shape = (len(train_thresholds), len(test_thresholds))
matrices = {
    "high_iou": np.zeros(shape),
    "med_iou": np.zeros(shape),
    "low_iou": np.zeros(shape),
    "overall_iou": np.zeros(shape),
    "high_recall": np.zeros(shape),
    "med_recall": np.zeros(shape),
    "low_recall": np.zeros(shape),
    "overall_recall": np.zeros(shape),
    "high_precision": np.zeros(shape),
    "med_precision": np.zeros(shape),
    "low_precision": np.zeros(shape),
    "overall_precision": np.zeros(shape),
}

def parse_logs(log_text: str):
    # regexes
    thresh_re = re.compile(r"training threshold:\s*([\d.]+).*testing threshold:\s*([\d.]+)", re.S)
    value_re = re.compile(r"(high|med|low|Overall)\s+(IoU|recall|precision):\s*([0-9.eE+-]+)")

    # split blocks by "training threshold"
    blocks = log_text.split("training threshold:")
    for block in blocks[1:]:
        # prepend "training threshold" back
        block = "training threshold:" + block

        # extract thresholds
        m = thresh_re.search(block)
        if not m:
            continue
        train_val = float(m.group(1))
        test_val = float(m.group(2))

        i = np.where(train_thresholds == train_val)[0][0]
        j = np.where(test_thresholds == test_val)[0][0]

        # extract values
        for match in value_re.finditer(block):
            level, metric, val = match.groups()
            key = f"{level.lower()}_{metric.lower()}"
            if key in matrices:
                matrices[key][i, j] = float(val)

    return matrices

# Example usage
if __name__ == "__main__":
    round_nums = [1, 2, 3]
    for round_num in round_nums:
        with open("round{}.log".format(round_num), "r") as f:
            log_text = f.read()
        matrices = parse_logs(log_text)
        with open("round{}_matrices.pkl".format(round_num), "wb") as f:
            pickle.dump(matrices, f)
        # Access e.g. high_iou_matrix
        #high_iou_matrix = matrices["high_iou"]
        #print("High IoU matrix:\n", high_iou_matrix)
