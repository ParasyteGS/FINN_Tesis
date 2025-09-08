import os
import shutil

# Carpetas temporales (donde descargaste con openimages)
tmp_dirs = {
    "car": "./dataset_no_birds/car/images",
    "house": "./dataset_no_birds/house/images",
    "person": "./dataset_no_birds/person/images",
}

# Carpetas finales
final_dirs = {"no_bird": "./dataset/no_bird"}

# Crear carpetas finales si no existen
for d in final_dirs.values():
    os.makedirs(d, exist_ok=True)


def move_with_prefix(src, dst, prefix):
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(".jpg"):
                new_name = f"{prefix}_{f}"
                shutil.copy(os.path.join(root, f), os.path.join(dst, new_name))


# Mover no_bird (Car + House)
move_with_prefix(tmp_dirs["car"], final_dirs["no_bird"], "Car")
move_with_prefix(tmp_dirs["house"], final_dirs["no_bird"], "House")
move_with_prefix(tmp_dirs["person"], final_dirs["no_bird"], "Person")

print("✅ Dataset organizado en ./dataset listo para YOLOv8-cls")
