import os
import requests
import praw
import random
import numpy as np
import pytesseract
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from textblob import TextBlob
from PIL import Image

# Инициализация Flask
app = Flask(__name__)

# Загрузка модели ResNet50 для анализа изображений
model_resnet = ResNet50(weights='imagenet')

# Функция для прогнозирования объектов на изображении
def predict_image(image_path):
    # Загружаем изображение
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Прогнозируем с использованием модели ResNet50
    predictions = model_resnet.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Извлекаем метку и уверенность прогноза
    label = decoded_predictions[0][1]  # Название объекта
    confidence = decoded_predictions[0][2]  # Уверенность

    # Извлекаем текст с изображения с помощью pytesseract
    extracted_text = pytesseract.image_to_string(img)
    mood = analyze_mood_from_text(extracted_text)

    return label, confidence, extracted_text, mood

# Функция для анализа настроений из извлечённого текста
def analyze_mood_from_text(text):
    # Простой анализ настроений с использованием TextBlob
    if text.strip() == "":
        return "Не найдено текста на изображении"
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Сентиментальный показатель (-1 до 1)

    if sentiment > 0:
        return "Позитивное настроение"
    elif sentiment < 0:
        return "Негативное настроение"
    else:
        return "Нейтральное настроение"

# Страница для загрузки изображений и анализа
@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        # Проверяем, был ли загружен файл
        file = request.files.get('image')
        if file:
            # Путь для сохранения изображения
            upload_folder = 'uploads'  # Директория для загрузки
            # Проверяем, существует ли директория, если нет — создаём её
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            # Полный путь для сохранения изображения
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Прогнозируем содержимое изображения
            label, confidence, extracted_text, mood = predict_image(file_path)

            return render_template('index.html', 
                                   label=label, 
                                   confidence=confidence, 
                                   extracted_text=extracted_text, 
                                   mood=mood, 
                                   image_url=file_path)
        else:
            return "Нет изображения для анализа", 400

    return render_template('index.html', label=None, confidence=None, extracted_text=None, mood=None)

# Настроим Reddit API
reddit = praw.Reddit(
    client_id="itM_bk3WBOwUytE5jUieYA",
    client_secret="BPS-mPLJ67_TJmavdRsM4DXCp26R3g",
    user_agent="MemeFetcherBot/1.0 (by u/Same_Day8260)"
)

# Функция для получения русских мемов
def get_russian_memes():
    subreddits = ["ru_memes", "RussianMemes", "ANIME_MOMENTS_RU"]
    memes = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            print(f"Ищем мемы в {sub}...")
            for post in subreddit.hot(limit=10):
                if "i.redd.it" in post.url:
                    response = requests.head(post.url)
                    if response.status_code == 200:
                        memes.append(post.url)
                        print(f"Добавлен мем: {post.url}")
        except Exception as e:
            print(f"Ошибка при запросе к {sub}: {e}")

    unique_memes = list(dict.fromkeys(memes))
    return unique_memes[:4] if len(unique_memes) >= 4 else None

# Функция для получения популярных мемов
def get_popular_memes():
    subreddits = ["memes", "dankmemes"]
    memes = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            print(f"Ищем мемы в {sub}...")
            for post in subreddit.hot(limit=10):
                if "i.redd.it" in post.url:
                    response = requests.head(post.url)
                    if response.status_code == 200:
                        memes.append(post.url)
                        print(f"Добавлен мем: {post.url}")
        except Exception as e:
            print(f"Ошибка при запросе к {sub}: {e}")

    unique_memes = list(dict.fromkeys(memes))
    return unique_memes[:6] if len(unique_memes) >= 6 else None

# Функция для получения мемов о животных
def get_animal_memes():
    subreddits = ["AnimalsBeingBros", "aww"]
    memes = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            print(f"Ищем мемы в {sub}...")
            for post in subreddit.hot(limit=10):
                if "i.redd.it" in post.url:
                    response = requests.head(post.url)
                    if response.status_code == 200:
                        memes.append(post.url)
                        print(f"Добавлен мем: {post.url}")
        except Exception as e:
            print(f"Ошибка при запросе к {sub}: {e}")

    return list(dict.fromkeys(memes))

# Функция для получения технических мемов
def get_tech_memes():
    subreddits = ["programmingmemes", "techmemes"]
    memes = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            print(f"Ищем мемы в {sub}...")
            for post in subreddit.hot(limit=10):
                if "i.redd.it" in post.url:
                    response = requests.head(post.url)
                    if response.status_code == 200:
                        memes.append(post.url)
                        print(f"Добавлен мем: {post.url}")
        except Exception as e:
            print(f"Ошибка при запросе к {sub}: {e}")

    return list(dict.fromkeys(memes))

# Функция для получения случайных мемов
def get_random_memes():
    # Собираем все доступные мемы из разных категорий
    russian_memes = get_russian_memes()
    popular_memes = get_popular_memes()
    animal_memes = get_animal_memes()
    tech_memes = get_tech_memes()
    
    all_memes = []
    
    # Добавляем мемы из разных категорий
    if russian_memes:
        all_memes.extend(russian_memes)
    if popular_memes:
        all_memes.extend(popular_memes)
    if animal_memes:
        all_memes.extend(animal_memes)
    if tech_memes:
        all_memes.extend(tech_memes)

    # Случайный выбор мемов, если их достаточно
    if len(all_memes) >= 4:
        return random.sample(all_memes, 4)
    return None

# Страница случайных мемов
@app.route('/случайные-мемы')
def random_memes():
    random_memes = get_random_memes()
    if random_memes:
        return render_template('случайный.html', memes=random_memes)
    return "Не удалось загрузить случайные мемы", 500

# Страница случайных мемов (API)
@app.route('/get_random_memes', methods=['GET'])
def fetch_random_memes():
    random_memes = get_random_memes()  # Получаем случайные мемы
    if random_memes:
        return jsonify({
            "meme1_url": random_memes[0],
            "meme2_url": random_memes[1],
            "meme3_url": random_memes[2],
            "meme4_url": random_memes[3]
        })
    return jsonify({"error": "Не удалось найти случайные мемы"}), 500

@app.route('/get_memes', methods=['GET'])
def fetch_memes():
    meme_urls = get_russian_memes()
    if meme_urls:
        return jsonify({
            "meme1_url": meme_urls[0],
            "meme2_url": meme_urls[1],
            "meme3_url": meme_urls[2],
            "meme4_url": meme_urls[3]
        })
    return jsonify({"error": "Не удалось найти русские мемы"}), 500

# API для получения популярных мемов
@app.route('/get_popular_memes', methods=['GET'])
def fetch_popular_memes():
    meme_urls = get_popular_memes()
    if meme_urls:
        return jsonify({
            "meme1_url": meme_urls[0],
            "meme2_url": meme_urls[1],
            "meme3_url": meme_urls[2],
            "meme4_url": meme_urls[3],
            "meme5_url": meme_urls[4],
            "meme6_url": meme_urls[5]
        })
    return jsonify({"error": "Не удалось найти популярные мемы"}), 500

# Главная страница (с русскими мемами)
@app.route('/')
def home():
    russian_memes = get_russian_memes()
    if russian_memes:
        return render_template('Главная.html', memes=russian_memes)
    return "Не удалось загрузить русские мемы", 500

# Страница популярных мемов
@app.route('/популярные-мемы')
def popular_memes():
    popular_memes = get_popular_memes()
    if popular_memes:
        return render_template('популярныемемы.html', memes=popular_memes)
    return "Не удалось загрузить популярные мемы", 500

# Страница входа
@app.route('/вход')
def login():
    return render_template('вход.html')

    # Функция для получения мемов с котами и собаками
def get_cat_and_dog_memes():
    subreddits = ["cats", "dogs", "AnimalsBeingBros", "aww"]
    memes = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            print(f"Ищем мемы в {sub}...")
            for post in subreddit.hot(limit=10):
                if "i.redd.it" in post.url:
                    response = requests.head(post.url)
                    if response.status_code == 200:
                        memes.append(post.url)
                        print(f"Добавлен мем: {post.url}")
        except Exception as e:
            print(f"Ошибка при запросе к {sub}: {e}")

    print(f"Найдено {len(memes)} мемов с котами и собаками.")
    return list(dict.fromkeys(memes))



# Страница с мемами о котах и собаках (JSON)
@app.route('/get_cat_and_dog_memes', methods=['GET'])
def fetch_cat_and_dog_memes_json():
    cat_and_dog_memes = get_cat_and_dog_memes()
    print(cat_and_dog_memes)  # Логируем мемы перед отправкой ответа
    if cat_and_dog_memes:
        return jsonify({
            "meme1_url": cat_and_dog_memes[0],
            "meme2_url": cat_and_dog_memes[1],
            "meme3_url": cat_and_dog_memes[2],
            "meme4_url": cat_and_dog_memes[3]
        })
    return jsonify({"error": "Не удалось найти мемы с котами и собаками"}), 500


# Страница с мемами о котах и собаках (HTML-шаблон)
@app.route('/категория')
def category():
    cat_and_dog_memes = get_cat_and_dog_memes()  # Используем функцию для получения мемов
    if cat_and_dog_memes:
        return render_template('категории.html', memes=cat_and_dog_memes)
    return "Не удалось загрузить мемы с котами и собаками", 500

# Функция для получения уникальных популярных мемов
def get_unique_popular_memes():
    subreddits = ["memes", "dankmemes"]
    memes = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            print(f"Ищем мемы в {sub}...")
            for post in subreddit.hot(limit=30):  # Получаем больше мемов, чтобы избежать повторений
                if "i.redd.it" in post.url:
                    response = requests.head(post.url)
                    if response.status_code == 200:
                        memes.append(post.url)
                        print(f"Добавлен мем: {post.url}")
        except Exception as e:
            print(f"Ошибка при запросе к {sub}: {e}")

    # Убираем дубликаты и возвращаем первые 10 уникальных мемов
    unique_memes = list(dict.fromkeys(memes))
    return unique_memes[:10] if len(unique_memes) >= 10 else None

# Страница популярных мемов (для /get_popular_memes)
@app.route('/get_unique_popular_memes', methods=['GET'])
def fetch_unique_popular_memes():
    meme_urls = get_unique_popular_memes()  # Используем функцию для получения уникальных популярных мемов
    if meme_urls:
        return jsonify({
            "meme1_url": meme_urls[0],
            "meme2_url": meme_urls[1],
            "meme3_url": meme_urls[2],
            "meme4_url": meme_urls[3],
            "meme5_url": meme_urls[4],
            "meme6_url": meme_urls[5],
            "meme7_url": meme_urls[6],
            "meme8_url": meme_urls[7],
            "meme9_url": meme_urls[8],
            "meme10_url": meme_urls[9]
        })
    return jsonify({"error": "Не удалось найти уникальные популярные мемы"}), 500

# Страница популярных мемов на странице top.html (не повторяются с get_popular_memes)
@app.route('/топ')
def top_memes():
    # Получаем уникальные мемы для страницы "top"
    unique_popular_memes = get_unique_popular_memes()
    if unique_popular_memes:
        return render_template('top.html', memes=unique_popular_memes)
    return "Не удалось загрузить уникальные популярные мемы", 500

@app.route('/feg')
def feg_page():
    memes = get_random_memes()  # Функция для получения случайных мемов
    if memes:
        return render_template('feg.html', memes=memes)
    return "Не удалось загрузить", 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
