import os
import nibabel as nib
import numpy as np

# Setup data directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'images', 'BraTS2021_Training_Data')

def load_volume(file_path):
    return nib.load(file_path).get_fdata(dtype='float32')  # En float pour précision

def normalize_volume(volume):
    mask = volume > 0  # Voxels cerveau (non-fond)
    if mask.sum() == 0:
        return volume  # Volume vide
    mean = volume[mask].mean()
    std = volume[mask].std()
    volume_normalized = (volume - mean) / std
    volume_normalized[~mask] = 0  # Gardez fond à 0
    return volume_normalized

# Exemple pour un patient
patient_id = 'BraTS2021_00000'  # Choisissez-en un
patient_dir = os.path.join(data_dir, patient_id)

# Charger toutes les modalités
modalities = ['flair', 't1', 't1ce', 't2']
volumes = {}

for modality in modalities:
    file_path = os.path.join(patient_dir, f'{patient_id}_{modality}.nii.gz')
    volumes[modality] = load_volume(file_path)
    volumes[modality] = normalize_volume(volumes[modality])
    print(f"Loaded {modality}: shape {volumes[modality].shape}")

# Charger la segmentation (cible)
seg_path = os.path.join(patient_dir, f'{patient_id}_seg.nii.gz')
seg = load_volume(seg_path)
print(f"Loaded segmentation: shape {seg.shape}")

# Combiner les 4 modalités dans un volume 4D (240, 240, 155, 4)
volume_4d = np.stack([volumes[mod] for mod in modalities], axis=-1)
print(f"Combined 4D volume: {volume_4d.shape}")