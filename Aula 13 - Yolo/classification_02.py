import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('model/yolo11m-cls.pt')

# Ler a imagem de entrada
image = cv2.imread('image/img03.png')

# Realizar a predição na imagem
result = model.predict(image, verbose=False, save=True)

# Mostrar a imagem com as detecções
for obj in result:
    #obj.show()
    #print(obj.probs)

    names = obj.names
    top5 = obj.probs.top5
    for item in top5:
        print(names[item])
    obj.show()