import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('model\yolo11n.pt')

# Realizar a predição na imagem
results = model("image/img03.png",show=True)
results[0].show()