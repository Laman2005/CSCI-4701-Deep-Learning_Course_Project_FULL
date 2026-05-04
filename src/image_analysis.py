import torch
import matplotlib.pyplot as plt
import numpy as np


# GET PREDICTIONS WITH METADATA
def collect_predictions(model, dataset, device, max_samples=300):

    model.eval()

    results = []

    for i in range(min(len(dataset), max_samples)):

        img, label, sex = dataset[i]

        x = img.unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        pred = int(prob > 0.5)

        results.append({
            "img": img,
            "label": int(label.item()),
            "pred": pred,
            "prob": prob,
            "sex": sex
        })

    return results


# SELECT 15 MEANINGFUL CASES
def select_cases(results):

    correct = [r for r in results if r["pred"] == r["label"]]
    wrong = [r for r in results if r["pred"] != r["label"]]

    correct_sorted = sorted(correct, key=lambda x: -abs(x["prob"] - 0.5))
    wrong_sorted = sorted(wrong, key=lambda x: -abs(x["prob"] - 0.5))

    # split by gender
    male = [r for r in results if r["sex"] == "Male"]
    female = [r for r in results if r["sex"] == "Female"]

    selected = []

    # 5 confident correct
    selected += correct_sorted[:5]

    # 5 confident wrong
    selected += wrong_sorted[:5]

    # 5 fairness samples
    selected += male[:2] + female[:3]

    return selected[:15]


# VISUALIZE CASES
def plot_cases(cases):

    plt.figure(figsize=(15,10))

    for i, case in enumerate(cases):

        plt.subplot(3,5,i+1)

        img = case["img"].permute(1,2,0)

        plt.imshow(img)
        plt.axis("off")

        plt.title(
            f"P:{case['prob']:.2f}\n"
            f"T:{case['label']} | Pred:{case['pred']}\n"
            f"{case['sex']}",
            fontsize=8
        )

    plt.suptitle("Model Predictions Analysis (15 Cases)", fontsize=16)
    plt.tight_layout()
    plt.show()


# FULL PIPELINE FUNCTION
def run_image_analysis(model, dataset, device):

    results = collect_predictions(model, dataset, device)
    cases = select_cases(results)
    plot_cases(cases)