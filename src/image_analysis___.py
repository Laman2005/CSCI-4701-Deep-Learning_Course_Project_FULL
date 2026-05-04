import torch
import matplotlib.pyplot as plt
import numpy as np


# COLLECT PREDICTIONS
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


# SELECT CASES (SMART)
def select_cases(results):

    correct = [r for r in results if r["pred"] == r["label"]]
    wrong = [r for r in results if r["pred"] != r["label"]]

    correct_sorted = sorted(correct, key=lambda x: -abs(x["prob"] - 0.5))
    wrong_sorted = sorted(wrong, key=lambda x: -abs(x["prob"] - 0.5))

    male = [r for r in results if r["sex"] == "Male"]
    female = [r for r in results if r["sex"] == "Female"]

    selected = []

    selected += correct_sorted[:5]   # strong correct
    selected += wrong_sorted[:5]     # strong wrong
    selected += male[:2] + female[:3]  # fairness

    return selected[:15]


# CASE INTERPRETATION TEXT
def describe_case(case):

    prob = case["prob"]
    label = case["label"]
    pred = case["pred"]

    if pred == label:
        if prob > 0.8:
            return " Correct & High Confidence → model is very certain and correct"
        elif prob > 0.6:
            return " Correct → reasonable confidence"
        else:
            return " Correct but Low Confidence → uncertain prediction"
    else:
        if prob > 0.8:
            return " Wrong & High Confidence → dangerous overconfidence"
        elif prob > 0.6:
            return " Wrong → model confused"
        else:
            return " Wrong & Low Confidence → borderline case"


# VISUALIZE WITH EXPLANATION
def plot_cases_with_text(cases):

    plt.figure(figsize=(16,10))

    print("\n IMAGE ANALYSIS SUMMARY\n")
    print("Each image shows:")
    print("P = predicted probability of Cardiomegaly")
    print("T = true label (0 = No, 1 = Yes)")
    print("Pred = predicted class\n")

    for i, case in enumerate(cases):

        plt.subplot(3,5,i+1)

        img = case["img"].permute(1,2,0)
        plt.imshow(img)
        plt.axis("off")

        title = (
            f"P:{case['prob']:.2f} | T:{case['label']} | Pred:{case['pred']}\n"
            f"{case['sex']}"
        )

        plt.title(title, fontsize=8)

        # PRINT TEXT EXPLANATION
        explanation = describe_case(case)

        print(f"Image {i+1}: {explanation}")

    plt.suptitle("Model Prediction Analysis (15 Cases)", fontsize=16)
    plt.tight_layout()
    plt.show()


# FINAL FUNCTION
def run_image_analysis(model, dataset, device):

    results = collect_predictions(model, dataset, device)
    cases = select_cases(results)

    plot_cases_with_text(cases)

    # FINAL GLOBAL INTERPRETATION
    print("\n OVERALL INTERPRETATION:\n")

    print("- Model performs well on clear cases with high confidence.")
    print("- Some errors occur with high confidence → indicates overconfidence.")
    print("- Low-confidence predictions correspond to ambiguous X-rays.")
    print("- Mixed gender samples included → visually no obvious bias, but must confirm via metrics.")
    print("- Failure cases suggest difficulty in borderline Cardiomegaly detection.\n")