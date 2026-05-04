import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

def plot_loss(losses):
    plt.plot(losses)
    plt.title("Training Loss")
    plt.show()


def plot_roc(y, p, title):
    fpr, tpr, _ = roc_curve(y, p)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],"--")
    plt.title(title)
    plt.show()


def plot_confusion(y, p):
    cm = confusion_matrix(y, p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.show()