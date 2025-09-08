from ultralytics import YOLO

# ========= CONFIGURACIÓN =========
DATASET_PATH = "dataset_split"  # aquí tienes /train/bird, /train/no_bird...
MODEL = "yolov8n-cls.pt"  # modelo base pequeño y rápido
EPOCHS = 50
IMGSZ = 224
TEST_IMAGE = "65a9d7da0d6bb119203b1c13.jpg"
# =================================

# 1. Cargar modelo
model = YOLO(MODEL)

# 2. Entrenar
print("🚀 Entrenando YOLOv8-cls...")
model.train(data=DATASET_PATH, epochs=EPOCHS, imgsz=IMGSZ, augment=True)

# 3. Validar
print("📊 Validando modelo...")
metrics = model.val()
print("Métricas:", metrics)


# 4. Probar en una imagen
print("🔍 Probando en imagen:", TEST_IMAGE)
results = model.predict(source=TEST_IMAGE)
for r in results:
    clase = r.names[r.probs.top1]
    conf = r.probs.top1conf.item()
    if clase == "bird":
        print(f"✅ Hay un ave con confianza {conf:.2f}")
    else:
        print(f"❌ No hay ave (confianza {conf:.2f})")
