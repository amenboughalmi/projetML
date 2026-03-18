import os
from sklearn.model_selection import train_test_split

# Chemin vers les données - utilise le répertoire du script pour la compatibilité
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'images', 'BraTS2021_Training_Data')

# Lister les dossiers patients (seulement ceux commençant par 'BraTS2021_')
patient_ids = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder)) and folder.startswith('BraTS2021_')]

print(f"Nombre de patients : {len(patient_ids)}")  # Devrait être ~1251

# Split : train 80%, val 10%, test 10% (stratifié aléatoirement)
train_ids, temp_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# Optionnel : Sauvegarder les lists en TXT pour reproductibilité
with open(os.path.join(script_dir, 'train_ids.txt'), 'w') as f:
    f.write('\n'.join(train_ids))
with open(os.path.join(script_dir, 'val_ids.txt'), 'w') as f:
    f.write('\n'.join(val_ids))
with open(os.path.join(script_dir, 'test_ids.txt'), 'w') as f:
    f.write('\n'.join(test_ids))
