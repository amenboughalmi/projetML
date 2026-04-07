import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

# Chemins automatiques
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "images", "BraTS2021_Training_Data")

def load_volume(file_path):
    return nib.load(file_path).get_fdata(dtype=np.float32)

def normalize_volume(volume):
    brain_mask = volume > 0
    if not brain_mask.any():
        return volume.copy()
    mean = volume[brain_mask].mean()
    std = volume[brain_mask].std() + 1e-8
    normalized = (volume - mean) / std
    normalized[~brain_mask] = 0
    return normalized

def get_bbox_from_mask(mask_slice):
    coords = np.where(mask_slice > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return np.array([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1], dtype=np.float32)

class BraTSDetectionDataset(Dataset):
    def __init__(self, patient_ids):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.cache = {}          # ← CACHE : on garde les volumes en mémoire
        self.samples = []

        for pid in patient_ids:
            pdir = os.path.join(self.data_dir, pid)
            # Chargement UNE SEULE FOIS par patient
            self.cache[pid] = {
                'flair': normalize_volume(load_volume(os.path.join(pdir, f"{pid}_flair.nii.gz"))),
                't1ce':  normalize_volume(load_volume(os.path.join(pdir, f"{pid}_t1ce.nii.gz"))),
                't2':    normalize_volume(load_volume(os.path.join(pdir, f"{pid}_t2.nii.gz"))),
                'seg':   load_volume(os.path.join(pdir, f"{pid}_seg.nii.gz"))
            }

            seg = self.cache[pid]['seg']
            for z in range(seg.shape[2]):
                bbox = get_bbox_from_mask(seg[:, :, z])
                has_tumor = bbox is not None
                self.samples.append((pid, z, bbox, has_tumor))

        print(f" Dataset créé – {len(self.samples)} slices (dont {sum(1 for s in self.samples if s[3])} avec tumeur)")

    def __getitem__(self, idx):
        pid, z, bbox, has_tumor = self.samples[idx]
        volumes = self.cache[pid]

        # On prend seulement les 3 modalités les plus utiles
        input_3ch = np.stack([
            volumes['flair'][:, :, z],
            volumes['t1ce'][:, :, z],
            volumes['t2'][:, :, z]
        ], axis=0)

        input_tensor = torch.from_numpy(input_3ch).float()
        input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min() + 1e-8)

        label = torch.tensor(1.0 if has_tumor else 0.0, dtype=torch.float32)
        bbox_tensor = torch.from_numpy(bbox) if has_tumor else torch.zeros(4, dtype=torch.float32)

        return {
            "image": input_tensor,
            "label": label,
            "bbox": bbox_tensor,
            "has_tumor": has_tumor,
            "patient_id": pid,
            "slice_idx": z
        }

    def __len__(self):
        return len(self.samples)
    
# ====================== TEST RAPIDE ======================
if __name__ == "__main__":
    print("=== Test du dataset avec cache ===")
    
    # Charge 5 patients seulement pour tester vite
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    with open(os.path.join(project_root, "train_ids.txt"), 'r') as f:
        test_ids = f.read().splitlines()[:5]
    
    dataset = BraTSDetectionDataset(test_ids)
    print(f"✅ Dataset prêt ! {len(dataset)} slices chargées avec succès.")
    print("Tu peux maintenant lancer l'entraînement.")