import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def evaluate(model, loader, device):

    model.eval()

    preds, labels, sexes = [], [], []

    with torch.no_grad():
        for x, y, s in loader:

            x = x.to(device)
            out = torch.sigmoid(model(x)).cpu().numpy()

            preds.extend(out)
            labels.extend(y.numpy())
            sexes.extend(s)

    return np.array(preds).flatten(), np.array(labels), np.array(sexes)


def metrics(preds, labels):

    binary = (preds > 0.5).astype(int)

    return {
        "acc": accuracy_score(labels, binary),
        "auc": roc_auc_score(labels, preds),
        "f1": f1_score(labels, binary)
    }