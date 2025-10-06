import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

# Define thresholds
train_threshes = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
test_threshes  = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

with open("round2_matrices.pkl", "rb") as f:
        matrices = pickle.load(f)

def plot_density(matrices, density):
    density_iou_matrix = matrices['{}_iou'.format(density)]
    density_recall_matrix = matrices['{}_recall'.format(density)]
    density_precision_matrix = matrices['{}_precision'.format(density)]

    fig, axes = plt.subplots(3, 1, figsize=(3, 12))
    sns.heatmap(density_iou_matrix, annot=True, fmt=".3f", xticklabels=test_threshes, yticklabels=train_threshes, cmap='Blues', ax=axes[0])
    axes[0].set_title("IoU")
    axes[0].set_xlabel("Test Threshold")
    axes[0].set_ylabel("Training Threshold")

    sns.heatmap(density_recall_matrix, annot=True, fmt=".3f", xticklabels=test_threshes, yticklabels=train_threshes, cmap='Greens', ax=axes[1])
    axes[1].set_title("Recall")
    axes[1].set_xlabel("Test Threshold")
    axes[1].set_ylabel("Training Threshold")

    sns.heatmap(density_precision_matrix, annot=True, fmt=".3f", xticklabels=test_threshes, yticklabels=train_threshes, cmap='Reds', ax=axes[2])
    axes[2].set_title("Precision")
    axes[2].set_xlabel("Test Threshold")
    axes[2].set_ylabel("Training Threshold")

    plt.tight_layout()
    plt.show()

def plot_all_densities(matrices, metric):
    high_metric_matrix = matrices['high_{}'.format(metric)]
    med_metric_matrix = matrices['med_{}'.format(metric)]
    low_metric_matrix = matrices['low_{}'.format(metric)]
    overall_metric_matrix = matrices['overall_{}'.format(metric)]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.heatmap(low_metric_matrix, annot=True, fmt=".3f", xticklabels=test_threshes, yticklabels=train_threshes, cmap='Reds', ax=axes[0,0])
    axes[0,0].set_title("Low {}".format(metric))
    axes[0,0].set_xlabel("Test Threshold")
    axes[0,0].set_ylabel("Training Threshold")

    sns.heatmap(med_metric_matrix, annot=True, fmt=".3f", xticklabels=test_threshes, yticklabels=train_threshes, cmap='Oranges', ax=axes[0,1])
    axes[0,1].set_title("Medium {}".format(metric))
    axes[0,1].set_xlabel("Test Threshold")
    axes[0,1].set_ylabel("Training Threshold")

    sns.heatmap(high_metric_matrix, annot=True, fmt=".3f", xticklabels=test_threshes, yticklabels=train_threshes, cmap='Greens', ax=axes[1,0])
    axes[1,0].set_title("High {}".format(metric))
    axes[1,0].set_xlabel("Test Threshold")
    axes[1,0].set_ylabel("Training Threshold")

    sns.heatmap(overall_metric_matrix, annot=True, fmt=".3f", xticklabels=test_threshes, yticklabels=train_threshes, cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title("Overall {}".format(metric))
    axes[1,1].set_xlabel("Test Threshold")
    axes[1,1].set_ylabel("Training Threshold")

    plt.tight_layout()
    plt.show()

plot_density(matrices, 'high')
plot_all_densities(matrices, 'recall')
