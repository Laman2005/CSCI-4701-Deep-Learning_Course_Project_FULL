import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train_model(model, loader, device, epochs=5, pos_weight=None):

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    scaler = GradScaler()

    losses = []

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for x, y, _ in tqdm(loader):

            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            optimizer.zero_grad()

            with autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        losses.append(avg)

        print(f"Epoch {epoch+1}: {avg:.4f}")

    return model, losses