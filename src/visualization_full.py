import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

sns.set(style="whitegrid")

# 1. LOSS CURVE
def plot_loss(losses):
    plt.figure(figsize=(6,4))
    plt.plot(losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


# 2. ROC CURVE
def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label="Model")
    plt.plot([0,1],[0,1],'--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 3. PRECISION-RECALL CURVE
def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(5,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.show()


# 4. CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# 5. CALIBRATION PLOT
from sklearn.calibration import calibration_curve

def plot_calibration_curve(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(5,4))
    plt.plot(prob_pred, prob_true, marker='o', label="Model")
    plt.plot([0,1],[0,1],'--', label="Perfect")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 6. PROBABILITY DISTRIBUTION
def plot_prediction_distribution(y_prob):
    plt.figure(figsize=(6,4))
    sns.histplot(y_prob, bins=30, kde=True)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.tight_layout()
    plt.show()


# 7. CLASS IMBALANCE
def plot_class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(4,4))
    plt.bar(unique.astype(str), counts)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# 8. GENDER DISTRIBUTION
def plot_gender_distribution(sexes):
    unique, counts = np.unique(sexes, return_counts=True)

    plt.figure(figsize=(4,4))
    plt.bar(unique, counts)
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# 9. FAIRNESS COMPARISON
def plot_fairness_bar(report):
    labels = ["Male", "Female"]
    accs = [report["male_acc"], report["female_acc"]]

    plt.figure(figsize=(5,4))
    plt.bar(labels, accs)
    plt.title("Accuracy by Gender")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


# 10. ROC BY GROUP
def plot_group_roc(y_true, y_prob, sexes):

    male = sexes == "Male"
    female = sexes == "Female"

    plt.figure(figsize=(5,4))

    for group, label in [(male, "Male"), (female, "Female")]:
        if len(np.unique(y_true[group])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true[group], y_prob[group])
        plt.plot(fpr, tpr, label=label)

    plt.plot([0,1],[0,1],'--')
    plt.title("ROC by Gender")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 11. ERROR ANALYSIS
def plot_error_cases(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)

    errors = (y_pred != y_true)

    plt.figure(figsize=(6,4))
    sns.histplot(y_prob[errors], bins=20)
    plt.title("Prediction Errors Distribution")
    plt.xlabel("Predicted Probability")
    plt.tight_layout()
    plt.show()