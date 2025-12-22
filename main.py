import cv2
from ultralytics import YOLO

# Загружаем ваш файл (он должен лежать рядом)
model = YOLO("best.pt")

# Открываем веб-камеру
cap = cv2.VideoCapture(0)

print("Камера включена! Нажмите 'q' (англ) на клавиатуре, чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ищем объекты
    results = model(frame, conf=0.5)
    
    # Рисуем рамки
    frame_with_boxes = results[0].plot()

    # Показываем окно
    cv2.imshow("My AI Vision", frame_with_boxes)

    # Выход по кнопке 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()