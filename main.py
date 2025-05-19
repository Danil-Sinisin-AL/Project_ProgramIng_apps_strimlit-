import streamlit as st
import torch

from PIL import Image
from torchvision import transforms

# Настройки страницы
st.set_page_config(
    page_title="Классификатор животных",
    layout="wide"
)

# Заголовок приложения
st.title("Классификатор животных по изображению")
st.write("Загрузите изображение животного, и модель определит его класс")

# Список классов (замените на ваши реальные классы)
CLASS_NAMES = [
    "Alces", "Bison", "Canis lupus", "Capreolus", "Cnipon",
    "Lepus", "Lutra", "Lynx", "Martes", "Meles",
    "Neovison", "Nyctereutes", "OTHER ANIMAL", "Putorius", "Sus"
]


# Функция для загрузки модели (замените на путь к вашей модели)
@st.cache_resource
def load_model():
    model = torch.load("resnet18_or.pt",weights_only=False, map_location=torch.device('cpu'))
    return model


# Функция для предсказания класса
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor  = transform(image)
    img_tensor  = img_tensor.unsqueeze(0)
    #Преобразование изображения для модели

    # img = image.resize((224, 224))  # Размер должен соответствовать ожиданиям модели
    # img_array = np.array(img)
    # img_array = np.expand_dims(img_array, axis=0)

    outputs = model(img_tensor)
    _, pred_idx = torch.max(outputs,1)
    probs = max(torch.nn.functional.softmax(outputs, dim=1)[0])

    # Здесь должно быть предсказание модели (замените на ваш код)
    # prediction = model.predict(img_array)
    # predicted_class = CLASS_NAMES[np.argmax(prediction)]
    # confidence = np.max(prediction)

    # Заглушка для примера (удалите в реальном приложении)
    predicted_class = CLASS_NAMES[pred_idx]

    return predicted_class, probs

if __name__ ==  '__main__':
    # Загрузка модели
    model = load_model()

    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение животного...",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file is not None:
        # Отображение изображения
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        # Кнопка для классификации
        if st.button("Классифицировать"):
            with st.spinner("Анализ изображения..."):
                # Получение предсказания
                predicted_class, confidence = predict(image, model)

                # Отображение результата
                st.success(f"Результат классификации: {predicted_class}")
                st.metric("Уверенность модели", f"{confidence:.2%}")

                # Дополнительная информация (опционально)
                st.subheader("Дополнительная информация")
                st.write(f"Модель уверена, что на изображении {predicted_class.lower()} с вероятностью {confidence:.2%}")

    # Боковая панель с информацией
    st.sidebar.header("О приложении")
    st.sidebar.write("""
    Это веб-приложение использует модель глубокого обучения 
    для классификации изображений животных на 15 классов.
    """)
    st.sidebar.write("**Доступные классы:**")
    st.sidebar.write(", ".join(CLASS_NAMES))
