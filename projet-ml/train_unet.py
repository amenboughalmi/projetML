import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from brats_dataset import BraTSDataset
from unet_model import UNet

# Setup data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'images', 'BraTS2021_Training_Data')

# Dice Loss function
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()  # Loss = 1 - Dice

# Chargez splits
train_ids_path = os.path.join(script_dir, 'train_ids.txt')
val_ids_path = os.path.join(script_dir, 'val_ids.txt')

with open(train_ids_path, 'r') as f:
    train_ids = f.read().splitlines()[:3]  # Reduce to 3 patients for CPU speed
with open(val_ids_path, 'r') as f:
    val_ids = f.read().splitlines()[:2]   # Reduce to 2 patients for validation

print(f"Loading {len(train_ids)} train patients and {len(val_ids)} val patients...")

train_dataset = BraTSDataset(train_ids, data_dir)  # Sans transforms pour simplicité
val_dataset = BraTSDataset(val_ids, data_dir)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)  # Reduce batch to 2
val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)

# Modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

model = UNet(in_channels=4, out_channels=1, features=[32, 64, 128, 256]).to(device)  # Smaller model for CPU
optimizer = Adam(model.parameters(), lr=1e-3)

print("Starting training...\n")

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss = 0
    batch_count = 0
    
    for batch_idx, (inputs, masks) in enumerate(train_loader):
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = dice_loss(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_count += 1
        print(f"  Epoch {epoch+1} - Batch {batch_idx+1}/{len(train_loader)}: Loss {loss.item():.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            preds = model(inputs)
            val_loss += dice_loss(preds, masks).item()
    
    avg_train_loss = train_loss / batch_count
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Complete: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}\n")

# Sauvegardez modèle
model_save_path = os.path.join(script_dir, 'unet_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path} ✅")