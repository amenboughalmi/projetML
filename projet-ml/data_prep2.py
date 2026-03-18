import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Setup data directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'images', 'BraTS2021_Training_Data')
output_dir = os.path.join(script_dir, 'visualizations')
os.makedirs(output_dir, exist_ok=True)

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

def get_non_empty_slices(volume, seg_volume, min_tumor_voxels=5):
    slices = []
    for z in range(volume.shape[2]):
        seg_slice = seg_volume[:, :, z]
        if np.sum(seg_slice > 0) > min_tumor_voxels:  # Slice avec tumeur
            slices.append(z)
    return slices

# Charger un patient exemple
patient_id = 'BraTS2021_00000'
patient_dir = os.path.join(data_dir, patient_id)

# Charger les données du patient
flair = load_volume(os.path.join(patient_dir, f'{patient_id}_flair.nii.gz'))
flair_norm = normalize_volume(flair)
seg = load_volume(os.path.join(patient_dir, f'{patient_id}_seg.nii.gz'))

# Trouver les slices avec tumeur
non_empty = get_non_empty_slices(flair_norm, seg)
print(f"Found {len(non_empty)} slices with tumor")

# Visualisez une slice
if non_empty:
    z = non_empty[len(non_empty) // 2]  # Milieu
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.rot90(flair_norm[:, :, z]), cmap='gray')  # rot90 pour orientation médicale standard
    ax[0].set_title(f'FLAIR Normalisé (slice {z})')
    ax[1].imshow(np.rot90(flair_norm[:, :, z]), cmap='gray')
    ax[1].imshow(np.rot90(seg[:, :, z]), cmap='jet', alpha=0.5)  # Masque overlay
    ax[1].set_title('Masque Tumeur')
    
    # Sauvegardez comme PNG
    output_path = os.path.join(output_dir, f'{patient_id}_slice_{z}.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()