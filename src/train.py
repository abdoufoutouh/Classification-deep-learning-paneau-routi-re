import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import X, y, CLASS_NAMES
from model import build_model

# Vérifier que les données sont chargées
if X.shape[0] == 0:
    print("ERREUR: Aucune donnée chargée. Vérifiez le dossier data/processed")
    exit(1)

print(f"Données chargées: {X.shape[0]} images, {len(CLASS_NAMES)} classes")

# Split train / validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} images")
print(f"Validation: {X_val.shape[0]} images")

# Build modèle MobileNetV2
model = build_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES))

# Callbacks
checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
    mode="max"
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    verbose=1,
    mode="max",
    restore_best_weights=True
)

# Créer le dossier models si nécessaire
os.makedirs("models", exist_ok=True)

# Entraînement optimisé (3-5 epochs)
print("\nDébut de l'entraînement...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

print("\nEntraînement terminé!")
print(f"Meilleure accuracy de validation: {max(history.history['val_accuracy']):.4f}")