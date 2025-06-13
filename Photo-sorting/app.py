import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time

# Класи CIFAR-10
class_names = [
    'Літак', 'Автомобіль', 'Птах', 'Кіт', 'Олень',
    'Собака', 'Жаба', 'Кінь', 'Корабель', 'Вантажівка'
]

# API ключ Unsplash
UNSPLASH_ACCESS_KEY = "rAw_zm1HInsy79usUHCbk6IZu0lXE6sgagoPmE2gZKQ"

# Завантаження моделі
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cifar10_model.h5')

model = load_model()

# Стилізація додатку
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.8s ease-out;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        color: #2c3e50;
    }
    
    .stButton>button {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    
    .stFileUploader>div>div>div>div {
        border: 2px dashed #4facfe;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.7);
    }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .photo-card {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.4s ease;
        margin-bottom: 25px;
        background: white;
    }
    
    .photo-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .header-text {
        background: linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Автоматичне сортування фото(за темою)")
st.markdown('<p class="header-text">Завантажте зображення для аналізу та отримання рекомендацій</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Оберіть фото...", type=["jpg", "jpeg", "png"])

def search_similar_photos(query, per_page=6, page=1):
    """Пошук схожих фото на Unsplash за категорією"""
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": "landscape"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "url": photo["urls"]["regular"],
                "small": photo["urls"]["small"],
                "user": photo["user"]["name"],
                "description": photo.get("alt_description", "Без опису") or "Без опису"
            } 
            for photo in data["results"]
        ]
    except Exception as e:
        st.error(f"Помилка пошуку схожих фото: {str(e)}")
        return []

if uploaded_file is not None:
    # Скидаємо стан схожих фото при новому завантаженні
    if 'prev_uploaded_file' not in st.session_state or st.session_state.prev_uploaded_file != uploaded_file.name:
        st.session_state.similar_photos = []
        st.session_state.page = 1
        st.session_state.prev_uploaded_file = uploaded_file.name
    
    # Відображення зображення
    st.subheader("🖼 Ваше зображення:")
    image = Image.open(uploaded_file)
    st.image(image, caption='Завантажене зображення', width=400, use_container_width='auto')
    
    # Підготовка зображення
    with st.spinner('🔍 Обробка зображення...'):
        img = image.resize((32, 32))
        img_array = np.array(img)
        
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        img_array = img_array.astype('float32') / 255
        img_array = np.expand_dims(img_array, axis=0)
    
    # Передбачення
    with st.spinner('🤖 Аналіз зображення...'):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        predictions = model.predict(img_array)
    
    # Визначення основного класу
    main_class_index = np.argmax(predictions)
    main_class = class_names[main_class_index]
    
    # Відображення результатів
    st.subheader("📊 Результати аналізу:")
    st.markdown(f'<div class="animated"><div class="metric-card">'
                f'<h3>Основна категорія</h3><h2 style="color:#4facfe;">{main_class}</h2>'
                f'</div></div>', unsafe_allow_html=True)
    
    # Топ-3 передбачення
    top_k = 3
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    st.write("**Ймовірності категорій:**")
    for i in top_indices:
        prob = predictions[0][i]
        progress_value = int(prob * 100)
        color = "#00c853" if i == main_class_index else "#4facfe"
        st.markdown(f'<div class="animated"><div class="metric-card">'
                    f'<h4>{class_names[i]}</h4>'
                    f'<div style="display:flex; align-items:center;">'
                    f'<div style="width:100%; background:#f0f0f0; border-radius:10px; margin-right:10px;">'
                    f'<div style="width:{progress_value}%; background:{color}; height:10px; border-radius:10px;"></div>'
                    f'</div>'
                    f'<span style="font-weight:bold; color:{color};">{prob:.2%}</span>'
                    f'</div></div></div>', unsafe_allow_html=True)
    
    # Пошук схожих фото
    category_translation = {
        'Літак': 'airplane',
        'Автомобіль': 'car',
        'Птах': 'bird',
        'Кіт': 'cat',
        'Олень': 'deer',
        'Собака': 'dog',
        'Жаба': 'frog',
        'Кінь': 'horse',
        'Корабель': 'ship',
        'Вантажівка': 'truck'
    }
    
    # Ініціалізація стану для пагінації
    if 'page' not in st.session_state:
        st.session_state.page = 1
    if 'similar_photos' not in st.session_state:
        st.session_state.similar_photos = []
    
    # Перший пошук
    if not st.session_state.similar_photos:
        with st.spinner('🌐 Шукаємо схожі зображення...'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            st.session_state.similar_photos = search_similar_photos(
                category_translation[main_class],
                per_page=6,
                page=st.session_state.page
            )
    
    # Відображення результатів
    if st.session_state.similar_photos:
        st.subheader(f"🎨 Схожі зображення ({main_class}):")
        
        # Відображення фото з чіткими відступами
        photos_per_row = 3
        for i in range(0, len(st.session_state.similar_photos), photos_per_row):
            cols = st.columns(photos_per_row)
            row_photos = st.session_state.similar_photos[i:i+photos_per_row]
            
            for col, photo in zip(cols, row_photos):
                with col:
                    # Контейнер для кожного фото
                    st.markdown(f'<div class="animated"><div class="photo-card">', unsafe_allow_html=True)
                    st.image(
                        photo["small"],
                        use_container_width=True
                    )
                    st.markdown(f'<div style="padding:15px;">'
                                f'<p style="font-weight:bold; margin-bottom:5px;">Автор: {photo["user"]}</p>'
                                f'<p style="font-size:0.9em; color:#555; margin-bottom:10px;">{photo["description"]}</p>'
                                f'<a href="{photo["url"]}" target="_blank" style="display:inline-block; padding:8px 15px; background:#4facfe; color:white; border-radius:20px; text-decoration:none; font-size:0.9em;">Відкрити оригінал</a>'
                                f'</div></div></div>', unsafe_allow_html=True)
        
        # Кнопка "Завантажити ще"
        if st.button("🔄 Завантажити ще фото", key="load_more"):
            st.session_state.page += 1
            with st.spinner('⏳ Завантажуємо додаткові фото...'):
                new_photos = search_similar_photos(
                    category_translation[main_class],
                    per_page=6,
                    page=st.session_state.page
                )
                if new_photos:
                    st.session_state.similar_photos.extend(new_photos)
                    st.rerun()
                else:
                    st.warning("Більше фото не знайдено")
    else:
        st.warning("Не вдалося знайти схожі зображення")

# Додамо декоративні елементи у бічну панель
st.sidebar.markdown("## 🚀 Про додаток")
st.sidebar.info("Цей додаток використовує нейронну мережу для класифікації зображень за 10 категоріями та знаходить схожі фото з Unsplash.")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Точність моделі")
st.sidebar.image("training_history.png", caption="Графік навчання моделі", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🛠 Технології")
st.sidebar.markdown("- TensorFlow для моделі машинного навчання")
st.sidebar.markdown("- Streamlit для інтерфейсу")
st.sidebar.markdown("- Unsplash API для пошуку зображень")
st.sidebar.markdown("---")
st.sidebar.markdown("#### 👤 Виконали проект")
st.sidebar.markdown("- Черняхівський Володимир")
st.sidebar.markdown("- Сопільняк Богдан")
st.sidebar.markdown("- Погорілий Борис")