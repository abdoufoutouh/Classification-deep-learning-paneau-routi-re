import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3), num_classes=12):
    # Base MobileNetV2 pré-entraîné sur ImageNet
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    
    # Geler la base du modèle
    base_model.trainable = False
    
    # Tête de classification personnalisée
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    
    # Modèle complet
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compilation avec optimizer Adam et learning rate réduit
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
