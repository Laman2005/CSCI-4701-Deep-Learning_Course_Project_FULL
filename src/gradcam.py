import torch
import matplotlib.pyplot as plt

def simple_visual_check(model, dataset, device):

    model.eval()

    for i in [0,5,10,20]:

        img, label, sex = dataset[i]

        x = img.unsqueeze(0).to(device)

        with torch.no_grad():
            p = torch.sigmoid(model(x)).item()

        plt.imshow(img.permute(1,2,0))
        plt.title(f"Pred: {p:.2f} | True: {label} | {sex}")
        plt.axis("off")
        plt.show()