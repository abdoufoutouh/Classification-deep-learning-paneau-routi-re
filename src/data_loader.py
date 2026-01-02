import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR = "data/processed"
IMG_SIZE = 224

# Classes ordonnées et fixes
CLASS_NAMES = [
    'children',
    'no_entry', 
    'pedestrian',
    'road_work',
    'speed_30',
    'speed_50',
    'speed_70',
    'speed_80',
    'stop',
    'turn_left',
    'turn_right',
    'yield'
]

class_to_label = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def load_and_preprocess_data():
    """Charge et prétraite les données avec preprocessing MobileNetV2"""
    images = []
    labels = []
    
    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_path):
            print(f"Attention: dossier {class_path} introuvable")
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Redimensionner en 224x224
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Convertir BGR en RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocessing MobileNetV2
            img = preprocess_input(img)
            
            images.append(img)
            labels.append(class_to_label[class_name])
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Images chargées : {X.shape}")
    print(f"Labels : {y.shape}")
    print(f"Classes : {class_to_label}")
    
    return X, y, CLASS_NAMES, class_to_label

# Charger les données
X, y, CLASS_NAMES, class_to_label = load_and_preprocess_data()
