import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from brats_detection_dataset import BraTSDetectionDataset
from detection_model import DetectionModel

# ====================== CHEMINS ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "images", "BraTS2021_Training_Data")

# ====================== 70 PATIENTS ======================
with open(os.path.join(project_root, "test_ids.txt"), 'r') as f:
    test_ids = f.read().splitlines()[:70]   # ← 50 patients

test_dataset = BraTSDetectionDataset(test_ids)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

# ====================== MODÈLE ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectionModel().to(device)
model.load_state_dict(torch.load('detection_model.pth', map_location=device))
model.eval()

print(f" Modèle chargé – {len(test_dataset)} slices (70 patients)")

os.makedirs("visualizations_detection", exist_ok=True)

with torch.no_grad():
    count = 0
    for batch_idx, batch in enumerate(test_loader):
        images = batch['image'].to(device)
        presence_pred, bbox_pred = model(images)
        presence_pred = presence_pred.cpu().numpy()
        bbox_pred = bbox_pred.cpu().numpy()

        for j in range(len(batch['label'])):
            real_has = bool(batch['has_tumor'][j])
            if not real_has:
                continue  # on saute les slices sans tumeur

            # ====================== VISUALISATION (avec tumeur) ======================
            img = batch['image'][j][0].cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Patient {batch['patient_id'][j]} - slice {batch['slice_idx'][j]} (TUMEUR)")

            # Boîte réelle (VERT)
            x, y, w, h = batch['bbox'][j].cpu().numpy()
            ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=8, edgecolor='lime', facecolor='none', label='Vraie boîte'))

            # Boîte prédite (ROUGE)
            x, y, w, h = bbox_pred[j]
            ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=8, edgecolor='red', facecolor='none', label='Boîte prédite'))

            ax.legend(fontsize=14, loc='upper right')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"visualizations_detection/exemple_avec_tumeur_{count:03d}.png", dpi=300)
            plt.close()

            count += 1
            if count >= 20:  # on s'arrête après 20 bonnes images
                break
        if count >= 20:
            break

print("\n 20 images avec tumeur générées dans 'visualizations_detection/'")