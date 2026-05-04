import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_calibration(y_true, y_prob):

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1],"--")
    plt.title("Calibration Curve")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.grid()
    plt.show()