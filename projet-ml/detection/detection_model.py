from pyexpat import features

import torch
import torch.nn as nn
import torchvision.models as models

#Ce fichier définit le modèle de détection.
class DetectionModel(nn.Module): #classe PyTorch classique (hérite de nn.Module).
    def __init__(self):
        super().__init__()  #(construction du modèle)
        # Backbone = ResNet50 pré-entraîné sur ImageNet
        # Il sert à extraire des "features" (caractéristiques) à partir de l'image
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        # ResNet50 a été entraîné pour classer 1000 classes d'objets (chat, chien, voiture, etc.)
        # il passe par plusieurs couches de convolution pour extraire des features, puis une couche finale (fc) pour faire la classification.
        #  On va réutiliser les premières couches (extraction de features)
        # mais pas la dernière (classification) qui est spécifique à ImageNet.
        self.backbone.fc = nn.Identity()  # on enlève la dernière couche
        
        #features = vecteur de 2048 nombres résumant l'image (après les convolutions)
        # Tête de classification :
        # prend les 2048 features et prédit la présence d'une tumeur (0 ou 1)
        self.classifier = nn.Linear(2048, 1)   # entrée : 2048 nombres (features) et sortie : 1 nombre

        # Tête de régression :
        # prédit les coordonnées de la bounding box [x, y, w, h]
        self.bbox_head = nn.Linear(2048, 4)    # x,y,w,h
        
    def forward(self, x):  #(ce qui se passe quand on passe une image)
        # Passage de l'image dans le backbone pour extraire les features
        features = self.backbone(x)

        # Prédiction de la présence de tumeur (sigmoid → probabilité entre 0 et 1)
        # features → classifier → nombre → sigmoid → probabilité
        presence = torch.sigmoid(self.classifier(features)).squeeze() #transforme le nombre en : valeur entre 0 et 1 


        # Prédiction de la bounding box
        bbox = self.bbox_head(features)
        return presence.squeeze(), bbox