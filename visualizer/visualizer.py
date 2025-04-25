import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metrics(metrics_dict, save_path="metrics_plot.png"):
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_values = [metrics_dict[name] for name in metric_names]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=metric_names, y=metric_values, palette='viridis')
    plt.title('Evaluation Metrics')
    plt.ylim(0, 1)
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    plt.tight_layout()

    # Save and Show
    plt.savefig(save_path, format=os.path.splitext(save_path)[-1][1:])
    plt.show()

def plot_confusion_matrix(cm, class_labels=None, save_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()

    # Save and Show
    plt.savefig(save_path, format=os.path.splitext(save_path)[-1][1:])
    plt.show()
