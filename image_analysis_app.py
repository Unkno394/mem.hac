import os
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

# Настроим модель (например, для классификации изображений)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Функция для предсказания содержимого изображения
def predict_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # изменим размер изображения
    img_array = np.array(img) / 255.0  # нормализация изображения
    img_array = np.expand_dims(img_array, axis=0)  # добавляем размер для батча

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

    return decoded_predictions[0][1], decoded_predictions[0][2]  # Название и вероятность

# Страница для загрузки и анализа изображения
@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        # Проверим, был ли загружен файл
        file = request.files.get('image')
        if file:
            # Сохраняем изображение на сервере
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Прогнозируем содержимое изображения
            label, confidence = predict_image(file_path)

            return render_template('index.html', label=label, confidence=confidence, image_url=file_path)
        else:
            return "Нет изображения для анализа", 400

    return render_template('analyse.html', label=label, confidence=confidence, image_url=file_path)


    

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
