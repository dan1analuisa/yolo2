from ultralytics import YOLO
import cv2
import math

# Cargamos el modelo YOLO pre-entrenado
model = YOLO(r'C:\Users\Usuario\Desktop\SEXTO SEMESTRE\IA\Lab 08\yolo11n.pt')

# Lista de clases que nos interesan
allowed_classes = ["person", "bicycle", "car", "motorbike", "bus", "train", "truck", "traffic light", 
                   "fire hydrant", "stop sign", "parking meter", "bench"]

# Configuramos la captura de video
captura = cv2.VideoCapture(0)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Iniciamos un bucle para procesar los fotogramas de la cámara
while True:
    success, img = captura.read()

    # Realizamos la detección de objetos
    results = model(img, stream=True)

    # Procesamos los resultados de la detección
    for r in results:
        boxes = r.boxes

        # Iteramos sobre las cajas delimitadoras detectadas
        for box in boxes:
            # Obtenemos las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Obtenemos la confianza de la detección
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Obtenemos el nombre de la clase detectada
            cls = int(box.cls[0])
            class_name = model.names[cls]  # Obtén el nombre de la clase directamente del modelo

            # Verificamos si la clase detectada está en la lista de clases permitidas
            if class_name in allowed_classes:
                # Dibujamos la caja delimitadora
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

                # Mostramos la clase detectada
                cv2.putText(img, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    # Mostramos la imagen con las detecciones
    cv2.imshow('Webcam', img)

    # Salimos del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos todas las ventanas
captura.release()
cv2.destroyAllWindows()