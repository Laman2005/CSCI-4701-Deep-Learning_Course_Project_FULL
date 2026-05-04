import torch
from src.train import train_model
from src.evaluate import evaluate, metrics
from src.fairness import fairness_report


# RUN SINGLE EXPERIMENT
def run_experiment(
    model,
    train_loader,
    valid_loader,
    device,
    name="baseline",
    pos_weight=None,
    epochs=5
):

    print(f"\n=== Running Experiment: {name} ===")

    model, losses = train_model(
        model,
        train_loader,
        device,
        epochs=epochs,
        pos_weight=pos_weight
    )

    preds, labels, sexes = evaluate(model, valid_loader, device)

    perf = metrics(preds, labels)
    fair = fairness_report(preds, labels, sexes)

    results = {
        "name": name,
        "performance": perf,
        "fairness": fair,
        "losses": losses
    }

    return results, preds, labels, sexes


# COMPARE MULTIPLE EXPERIMENTS
def compare_experiments(results_list):

    print("\n=== EXPERIMENT COMPARISON ===\n")

    for res in results_list:

        print(f"Experiment: {res['name']}")

        print("  Accuracy:", round(res["performance"]["acc"], 3))
        print("  AUC:", round(res["performance"]["auc"], 3))
        print("  F1:", round(res["performance"]["f1"], 3))

        print("  Male Acc:", round(res["fairness"]["male_acc"], 3))
        print("  Female Acc:", round(res["fairness"]["female_acc"], 3))
        print("  Gap:", round(res["fairness"]["gap_acc"], 3))

        print("-" * 40)