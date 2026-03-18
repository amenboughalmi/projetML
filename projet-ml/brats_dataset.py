import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

# Setup data directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'images', 'BraTS2021_Training_Data')

# Fonctions utilitaires (du code précédent)
def load_volume(file_path):
    return nib.load(file_path).get_fdata(dtype='float32')

def normalize_volume(volume):
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    volume_normalized = (volume - mean) / std
    volume_normalized[~mask] = 0
    return volume_normalized

def get_non_empty_slices(seg_volume, min_tumor_voxels=5):
    slices = []
    for z in range(seg_volume.shape[2]):
        if np.sum(seg_volume[:, :, z] > 0) > min_tumor_voxels:
            slices.append(z)
    return slices

class BraTSDataset(Dataset):
    def __init__(self, patient_ids, data_dir):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        # Pré-lister les slices pour tous patients (pour __len__)
        self.slice_list = []
        for pid in self.patient_ids:
            patient_dir = os.path.join(data_dir, pid)
            seg_path = os.path.join(patient_dir, f'{pid}_seg.nii.gz')
            seg = load_volume(seg_path)
            non_empty = get_non_empty_slices(seg)
            for z in non_empty:
                self.slice_list.append((pid, z))

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, idx):
        pid, z = self.slice_list[idx]
        patient_dir = os.path.join(self.data_dir, pid)
        
        # Chargez et normalisez modalités
        flair = normalize_volume(load_volume(os.path.join(patient_dir, f'{pid}_flair.nii.gz')))
        t1 = normalize_volume(load_volume(os.path.join(patient_dir, f'{pid}_t1.nii.gz')))
        t1ce = normalize_volume(load_volume(os.path.join(patient_dir, f'{pid}_t1ce.nii.gz')))
        t2 = normalize_volume(load_volume(os.path.join(patient_dir, f'{pid}_t2.nii.gz')))
        seg = load_volume(os.path.join(patient_dir, f'{pid}_seg.nii.gz'))
        
        # Extrayez slice
        input_slice = np.stack([flair[:, :, z], t1[:, :, z], t1ce[:, :, z], t2[:, :, z]], axis=0)  # Shape: (4, H, W)
        mask_slice = (seg[:, :, z] > 0).astype(np.float32)  # Binaire WT: (H, W)
        
        # Rot90 pour orientation standard (optionnel)
        input_slice = np.rot90(input_slice, k=1, axes=(1, 2)).copy()
        mask_slice = np.rot90(mask_slice, k=1).copy()
        
        # To tensors
        input_tensor = torch.from_numpy(input_slice).float()
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)  # (1, H, W) pour U-Net
        
        return input_tensor, mask_tensor

# Exemple d'utilisation
if __name__ == '__main__':
    # Chargez vos splits (du premier script)
    train_ids_path = os.path.join(script_dir, 'train_ids.txt')
    with open(train_ids_path, 'r') as f:
        train_ids = f.read().splitlines()[:20]  # Limite à 20 pour test
    
    print(f"Loading {len(train_ids)} patients...")
    
    # Créez le dataset sans transforms pour l'instant
    train_dataset = BraTSDataset(train_ids, data_dir)
    print(f"Total samples in dataset: {len(train_dataset)}")
    
    # Créez le DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Test : Chargez un batch
    print("\nLoading first batch...")
    for inputs, masks in train_loader:
        print(f"Input shape: {inputs.shape}")  # Attendu: (batch, 4, 240, 240)
        print(f"Mask shape: {masks.shape}")    # Attendu: (batch, 1, 240, 240)
        print(f"Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"Mask unique values: {torch.unique(masks)}")
        break