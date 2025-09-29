import cv2
import time
import psutil
import pandas as pd
from ultralytics import YOLO

model = YOLO("model/best1.pt")

video = cv2.VideoCapture('video/estacionamento.mp4')  # <<-- troque pelo seu vídeo

if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter(
    'video/saved_predictions.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (frame_width, frame_height)
)

metricas = []
status_vagas = {}


# parâmetros configuráveis
CONF_THRES = 0.35      # confiança mínima para testar (ajuste)
MODEL_IOU = 0.45       # IoU usado internamente no NMS do model call
IOU_MATCH = 0.3        # IoU mínimo para considerar que é a mesma vaga
CENTER_DIST = 50       # fallback: distância máxima do centro (pixels)
FRAME_CONFIRM = 2      # quantos frames consecutivos para confirmar estado
MAX_MISSING_FRAMES = 50  # remove track se não visto por X frames
DEBUG = False          # True para prints de debug

def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    return (interArea / union) if union > 0 else 0

def center_distance(boxA, boxB):
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    return ((cxA - cxB)**2 + (cyA - cyB)**2) ** 0.5

frame_idx = 0
# status_vagas structure:
# id -> {'bbox':[x1,y1,x2,y2], 'ocupado':int, 'livre':int, 'estado':str, 'last_seen':int}
status_vagas = {}

while True:
    check, img = video.read()
    if not check:
        print("Não foi possível ler o frame. Finalizando...")
        break

    inicio = time.time()
    # predição com thresholds já aplicados
    results = model(img, conf=CONF_THRES, iou=MODEL_IOU, verbose=False)[0]
    nomes = results.names

    # cria lista de detecções deste frame para processamento (x1,y1,x2,y2,classe,conf)
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls.item())
        nomeClasse = nomes[cls]
        conf = float(box.conf.item())
        detections.append((x1, y1, x2, y2, nomeClasse, conf))

    if DEBUG:
        print(f"[Frame {frame_idx}] Detecções:", [(d[4], round(d[5],2)) for d in detections])

    matched_ids = set()
    # para cada detecção, acha o melhor match entre tracks existentes (maior IoU)
    for det in detections:
        x1, y1, x2, y2, nomeClasse, conf = det
        det_box = [x1, y1, x2, y2]

        best_id = None
        best_iou = 0.0
        # busca o track com maior IoU que ainda não foi matched
        for tid, track in status_vagas.items():
            if tid in matched_ids:
                continue
            track_box = track['bbox']
            val_iou = iou(det_box, track_box)
            if val_iou > best_iou:
                best_iou = val_iou
                best_id = tid

         # se melhor IoU é baixo, tenta fallback por centro
        if best_iou < IOU_MATCH:
            # tenta achar por menor distância do centro
            best_id_center = None
            best_dist = float('inf')
            for tid, track in status_vagas.items():
                if tid in matched_ids:
                    continue
                dist = center_distance(det_box, track['bbox'])
                if dist < best_dist:
                    best_dist = dist
                    best_id_center = tid
            if best_dist <= CENTER_DIST:
                best_id = best_id_center
                best_iou = 0.0  # sinal que veio por centro

        # se não achou match válido, cria novo track
        if best_id is None:
            new_id = len(status_vagas) + 1
            status_vagas[new_id] = {
                'bbox': det_box,
                'ocupado': 0,
                'livre': 0,
                'estado': 'livre',
                'last_seen': frame_idx
            }
            best_id = new_id

        # atualiza track
        track = status_vagas[best_id]
        track['bbox'] = det_box  # atualiza bbox para o novo (poderia smooth)
        track['last_seen'] = frame_idx
        matched_ids.add(best_id)

        # atualização de contadores conforme classe detectada
        if nomeClasse.lower() == 'occupied':
            track['ocupado'] += 1
            track['livre'] = 0
        elif nomeClasse.lower() == 'empty':
            track['livre'] += 1
            track['ocupado'] = 0
        else:
            # fallback: se aparecer outra classe inesperada, trate como livre
            track['livre'] += 1
            track['ocupado'] = 0

        # só confirma estado após FRAME_CONFIRM frames
        if track['ocupado'] >= FRAME_CONFIRM:
            track['estado'] = 'ocupado'
        elif track['livre'] >= FRAME_CONFIRM:
            track['estado'] = 'livre'

    # opcional: tracks que não foram matched nesse frame permanecem, mas podemos
    # opcionalmente decrementar contadores / ou só atualizar last_seen e remover
    # tracks muito antigas:
    to_delete = []
    for tid, track in status_vagas.items():
        if frame_idx - track['last_seen'] > MAX_MISSING_FRAMES:
            to_delete.append(tid)
    for tid in to_delete:
        del status_vagas[tid]

    # desenha todas as tracks atuais (estado estável)
    for tid, track in status_vagas.items():
        x1, y1, x2, y2 = track['bbox']
        estado_final = track['estado']
        cor = (0, 0, 255) if estado_final == "ocupado" else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 1)
        # mostra confiança média simples (opcional) - aqui usamos 0.00 se não tiver
        text = f"{estado_final}"
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

    # métricas e gravação (mantém seu bloco)
    tempo_inferencia = time.time() - inicio
    fps_atual = 1 / tempo_inferencia if tempo_inferencia > 0 else 0
    uso_cpu = psutil.cpu_percent()
    uso_memoria = psutil.virtual_memory().percent

    ocupadas = sum(1 for v in status_vagas.values() if v["estado"] == "ocupado")
    livres = sum(1 for v in status_vagas.values() if v["estado"] == "livre")

    metricas.append({
        "Tempo_inferencia (s)": round(tempo_inferencia, 4),
        "FPS": round(fps_atual, 2),
        "Uso_CPU (%)": uso_cpu,
        "Uso_Memória (%)": uso_memoria,
        "Vagas_ocupadas": ocupadas,
        "Vagas_livres": livres
    })

    # Exibe e grava frame
    cv2.imshow('IMG', img)
    output_video.write(img)

    if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
        break

    frame_idx += 1

video.release()
output_video.release()
cv2.destroyAllWindows()

df = pd.DataFrame(metricas)
df.to_excel("video/metricas_yolo.xlsx", index=False)
print("Métricas salvas em 'video/metricas_yolo.xlsx'")