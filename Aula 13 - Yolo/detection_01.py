import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('model/yolo11n.pt')

# Ler a imagem de entrada
image = cv2.imread('image/Time 5.jpg')

# Realizar a predição na imagem
results = model.predict(image, verbose=False, save=True)

# Mostrar a imagem com as detecções
for objeto in results:
    objeto.show()