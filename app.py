import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Заголовок с актуальными эмодзи
st.title("Smart Checkout AI 🍎🍋🥚")
st.write("Загрузите фото продуктов, и нейросеть определит их.")

# Загружаем модель
model = YOLO('best.pt') 

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # ОБНОВЛЕННЫЙ ПАРАМЕТР (вместо use_container_width)
    st.image(image, caption='Загруженное фото', width='stretch')
    
    if st.button('Распознать продукты'):
        results = model(image)
        res_plotted = results[0].plot()
        
        # ОБНОВЛЕННЫЙ ПАРАМЕТР ТУТ ТОЖЕ
        st.image(res_plotted, caption='Результат распознавания', width='stretch')
        
        st.write("### Список покупок:")
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            st.write(f"- {label}")