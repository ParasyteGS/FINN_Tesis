import os
import shutil
import random

# Configuración
DATASET_DIR = "dataset"  # tu dataset actual
OUTPUT_DIR = "dataset_split"  # dataset dividido
SPLIT_RATIO = 0.8  # 80% train, 20% val

# Crear carpetas destino
for split in ["train", "val"]:
    for cls in ["birds", "no_birds"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Dividir imágenes
for cls in ["birds", "no_birds"]:
    img_dir = os.path.join(DATASET_DIR, cls)
    images = os.listdir(img_dir)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    for f in train_files:
        shutil.copy(os.path.join(img_dir, f), os.path.join(OUTPUT_DIR, "train", cls, f))
    for f in val_files:
        shutil.copy(os.path.join(img_dir, f), os.path.join(OUTPUT_DIR, "val", cls, f))

print("✅ Dataset dividido en train/ y val/")
