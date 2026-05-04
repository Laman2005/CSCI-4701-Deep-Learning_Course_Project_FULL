from sklearn.metrics import accuracy_score, roc_auc_score

def fairness_report(preds, labels, sexes):

    male = sexes == "Male"
    female = sexes == "Female"

    binary = (preds > 0.5).astype(int)

    def safe_auc(y, p):
        return roc_auc_score(y, p) if len(set(y)) > 1 else 0

    return {
        "male_acc": accuracy_score(labels[male], binary[male]),
        "female_acc": accuracy_score(labels[female], binary[female]),
        "male_auc": safe_auc(labels[male], preds[male]),
        "female_auc": safe_auc(labels[female], preds[female]),
        "gap_acc": abs(
            accuracy_score(labels[male], binary[male]) -
            accuracy_score(labels[female], binary[female])
        )
    }