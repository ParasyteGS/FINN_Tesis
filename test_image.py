from ultralytics import YOLO
import os

# Carpeta con imágenes
IMAGE_FOLDER = "Images_test"
model = YOLO("runs/classify/train/weights/best.pt")

# Obtener todas las imágenes de la carpeta (jpg, png)
imagenes = [
    f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for img_name in imagenes:
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    print("🔍 Probando en imagen:", img_name)

    results = model.predict(source=img_path)

    for r in results:
        # Obtener la clase y la confianza top1
        clase = r.names[r.probs.top1]
        conf = r.probs.top1conf.item()

        if clase == "bird":
            print(f"✅ Hay un ave con confianza {conf:.2f}")
        else:
            print(f"❌ No hay ave (confianza {conf:.2f})")

    print("-" * 40)  # Separador entre imágenes
