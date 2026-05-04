import json

def save_metrics(path, metrics):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)