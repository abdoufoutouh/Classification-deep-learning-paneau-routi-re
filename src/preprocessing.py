import cv2
import os

RAW_DIR = "data/raw/selected_classes"
PROCESSED_DIR = "data/processed"
IMG_SIZE = 64

os.makedirs(PROCESSED_DIR, exist_ok=True)

for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_dir = os.path.join(PROCESSED_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        output_path = os.path.join(
            output_class_dir,
            img_name.replace(".ppm", ".jpg")
        )

        cv2.imwrite(output_path, (img * 255).astype("uint8"))

print("Prétraitement terminé avec succès.")
