"""
utils.py
--------
Helper functions for evaluation reporting and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def print_evaluation(model, X_test, y_test, model_name="Model"):
    """Print classification report and confusion matrix for a fitted model."""
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    """Plot a styled confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(results_df):
    """Bar chart of model accuracy comparison."""
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", len(results_df))
    bars = plt.barh(results_df["Model"], results_df["Accuracy (%)"], color=colors)
    plt.xlabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xlim(60, 100)
    for bar, val in zip(bars, results_df["Accuracy (%)"]):
        plt.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot top-N feature importances for tree-based models
    (DecisionTree, RandomForest, GradientBoosting).
    """
    if not hasattr(model, "feature_importances_"):
        print("This model does not expose feature_importances_.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
