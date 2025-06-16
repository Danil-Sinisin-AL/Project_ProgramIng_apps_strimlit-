import pytest
from PIL import Image
import numpy as np
import io
from main import predict, CLASS_NAMES, load_model
import torch

@pytest.fixture
def sample_image():
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img

def test_predict(sample_image, mocker):
    # 1. Подготовка тестовых данных
    num_classes = len(CLASS_NAMES)
    test_output = torch.zeros(1, num_classes)
    target_class = 3  # Выбранный класс для теста
    
    # Устанавливаем значения, которые после softmax дадут ~1.0 для target_class
    test_output[0, target_class] = 20.0  # Большое значение для выбранного класса
    test_output[0, :] -= 10.0  # Делаем другие классы менее вероятными
    
    # 2. Настройка мок-модели
    mock_model = mocker.MagicMock()
    mock_model.return_value = test_output
    
    # 3. Вызов тестируемой функции
    predicted_class, confidence = predict(sample_image, mock_model)
    
    # 4. Проверки
    assert predicted_class == CLASS_NAMES[target_class]
    
    # Обработка confidence (тензор или float)
    confidence_value = confidence.item() if isinstance(confidence, torch.Tensor) else confidence
    
    assert isinstance(confidence_value, float)
    # Менее строгая проверка, учитывающая softmax
    assert confidence_value > 0.9  # Должно быть близко к 1.0

def test_model_loading():
    try:
        model = load_model()
        assert model is not None
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")