import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from brats_detection_dataset import BraTSDetectionDataset
from detection_model import DetectionModel

# Chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# ====================== 70 PATIENTS ======================
with open(os.path.join(project_root, "train_ids.txt"), 'r') as f:
    train_ids = f.read().splitlines()[:70]   # ← CHANGÉ : 70 patients

train_dataset = BraTSDetectionDataset(train_ids)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectionModel().to(device)
optimizer = Adam(model.parameters(), lr=5e-5)

def detection_loss(batch):
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    bboxes = batch['bbox'].to(device)
    has_tumor = torch.as_tensor(batch['has_tumor'], dtype=torch.bool).to(device)
    
    presence_pred, bbox_pred = model(images)
    bce = torch.nn.BCELoss()(presence_pred, labels)
    if has_tumor.sum() > 0:
        smooth_l1 = torch.nn.SmoothL1Loss()(bbox_pred[has_tumor], bboxes[has_tumor])
        return bce + 0.05 * smooth_l1
    return bce

# ====================== ENTRAÎNEMENT ======================
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in progress:
        loss = detection_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress.set_postfix({"Loss": f"{train_loss / (progress.n + 1):.4f}"})

    print(f"✅ Epoch {epoch+1} terminée | Loss moyenne = {train_loss / len(train_loader):.4f}\n")

torch.save(model.state_dict(), 'detection_model.pth')
print(" Entraînement terminé avec 70 patients !")
 