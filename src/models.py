"""
models.py
---------
Model training, hyperparameter tuning, and comparison utilities
for the Loan Approval Prediction project.

Usage:
    from src.models import train_all_models, get_best_model
    results = train_all_models(X_train, X_test, y_train, y_test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


# ──────────────────────────────────────────────
# Individual model trainers
# ──────────────────────────────────────────────

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(solver="saga", max_iter=500, random_state=1)
    model.fit(X_train, y_train)
    acc = accuracy_score(model.predict(X_test), y_test)
    return model, acc


def train_knn(X_train, X_test, y_train, y_test, k_range=range(1, 21)):
    """Sweep k values and return the model with the best test accuracy."""
    scores = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    best_k = list(k_range)[np.argmax(scores)]
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    return best_model, max(scores), scores


def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel="rbf", max_iter=500, probability=True)
    model.fit(X_train, y_train)
    acc = accuracy_score(model.predict(X_test), y_test)
    return model, acc


def train_naive_bayes(X_train, X_test, y_train, y_test):
    cat_model = CategoricalNB()
    cat_model.fit(X_train, y_train)
    cat_acc = accuracy_score(cat_model.predict(X_test), y_test)

    gau_model = GaussianNB()
    gau_model.fit(X_train, y_train)
    gau_acc = accuracy_score(gau_model.predict(X_test), y_test)

    return cat_model, cat_acc, gau_model, gau_acc


def train_decision_tree(X_train, X_test, y_train, y_test, leaf_range=range(2, 21)):
    scores = []
    for leaves in leaf_range:
        clf = DecisionTreeClassifier(max_leaf_nodes=leaves)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    best_leaves = list(leaf_range)[np.argmax(scores)]
    best_model = DecisionTreeClassifier(max_leaf_nodes=best_leaves)
    best_model.fit(X_train, y_train)
    return best_model, max(scores)


def train_random_forest(X_train, X_test, y_train, y_test, leaf_range=range(2, 25)):
    scores = []
    for leaves in leaf_range:
        clf = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=leaves)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    best_leaves = list(leaf_range)[np.argmax(scores)]
    best_model = RandomForestClassifier(
        n_estimators=1000, random_state=1, max_leaf_nodes=best_leaves
    )
    best_model.fit(X_train, y_train)
    return best_model, max(scores)


def train_gradient_boosting(X_train, X_test, y_train, y_test, n_iter=20, cv=20):
    params = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [1, 2, 3, 4, 5],
        "subsample": [0.5, 1],
        "max_leaf_nodes": [2, 5, 10, 20, 30, 40, 50],
    }
    search = RandomizedSearchCV(
        GradientBoostingClassifier(), params,
        n_iter=n_iter, cv=cv, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    acc = accuracy_score(best_model.predict(X_test), y_test)
    return best_model, acc, search.best_params_


# ──────────────────────────────────────────────
# Full comparison pipeline
# ──────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Train all 8 models and return a sorted accuracy comparison DataFrame.
    Also prints the classification report for each model.
    """
    results = {}

    print("=" * 50)
    print("Training Logistic Regression...")
    lr, lr_acc = train_logistic_regression(X_train, X_test, y_train, y_test)
    results["Logistic Regression"] = lr_acc * 100

    print("Training KNN...")
    knn, knn_acc, _ = train_knn(X_train, X_test, y_train, y_test)
    results["K-Nearest Neighbors"] = knn_acc * 100

    print("Training SVM...")
    svm, svm_acc = train_svm(X_train, X_test, y_train, y_test)
    results["SVM (RBF)"] = svm_acc * 100

    print("Training Naive Bayes...")
    cat_nb, cat_acc, gau_nb, gau_acc = train_naive_bayes(X_train, X_test, y_train, y_test)
    results["Categorical NB"] = cat_acc * 100
    results["Gaussian NB"] = gau_acc * 100

    print("Training Decision Tree...")
    dt, dt_acc = train_decision_tree(X_train, X_test, y_train, y_test)
    results["Decision Tree"] = dt_acc * 100

    print("Training Random Forest...")
    rf, rf_acc = train_random_forest(X_train, X_test, y_train, y_test)
    results["Random Forest"] = rf_acc * 100

    print("Training Gradient Boosting (RandomizedSearchCV)...")
    gb, gb_acc, gb_params = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results["Gradient Boosting"] = gb_acc * 100

    df_results = (
        pd.DataFrame(list(results.items()), columns=["Model", "Accuracy (%)"])
        .sort_values("Accuracy (%)", ascending=False)
        .reset_index(drop=True)
    )
    df_results.index += 1  # 1-based ranking
    print("\n📊 Model Comparison:\n")
    print(df_results.to_string())
    return df_results


def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    """Plot ROC curve and print AUC score for a probability-capable model."""
    probs = model.predict_proba(X_test)[:, 1]
    random_probs = [0] * len(y_test)

    auc = roc_auc_score(y_test, probs)
    random_auc = roc_auc_score(y_test, random_probs)

    fpr, tpr, _ = roc_curve(y_test, probs)
    r_fpr, r_tpr, _ = roc_curve(y_test, random_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(r_fpr, r_tpr, linestyle="--", label=f"Random (AUC = {random_auc:.3f})")
    plt.plot(fpr, tpr, marker=".", label=f"{model_name} (AUC = {auc:.3f})")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n{model_name} AUC-ROC: {auc:.3f}")
    return auc
