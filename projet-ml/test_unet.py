import torch
import os
from unet_model import UNet
from brats_dataset import BraTSDataset
from torch.utils.data import DataLoader

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'images', 'BraTS2021_Training_Data')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create model
model = UNet(in_channels=4, out_channels=1, features=[64, 128, 256, 512])
model = model.to(device)
print(f"\nU-Net model created successfully")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Load dataset
print("\nLoading dataset...")
train_ids_path = os.path.join(script_dir, 'train_ids.txt')
with open(train_ids_path, 'r') as f:
    train_ids = f.read().splitlines()[:5]  # Use 5 patients for quick test

train_dataset = BraTSDataset(train_ids, data_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)

# Forward pass test
print(f"\nLoading first batch...")
for inputs, masks in train_loader:
    inputs = inputs.to(device)
    masks = masks.to(device)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Mask shape: {masks.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"\nModel test PASSED! ✅")
    break
