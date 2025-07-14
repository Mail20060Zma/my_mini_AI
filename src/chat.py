import torch
import numpy as np
from pathlib import Path
from model import CharLSTM
from utils import safe_load_model

# Конфигурация
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "char_lstm_model.pt"

def load_model_for_chat(device):
    """Загрузка модели и словарей для чата"""
    if MODEL_PATH.exists():
        model, checkpoint = safe_load_model(MODEL_PATH, device, CharLSTM)
        if model and checkpoint:
            char_to_idx = checkpoint['char_to_idx']
            idx_to_char = checkpoint['idx_to_char']
            vocab_size = checkpoint['vocab_size']
            hidden_size = checkpoint['hidden_size']
            
            # Передаем параметры в модель
            model.vocab_size = vocab_size
            model.char_to_idx = char_to_idx
            model.idx_to_char = idx_to_char
            
            print(f"Модель загружена: {MODEL_PATH}")
            print(f"Размер словаря: {vocab_size} символов")
            return model, char_to_idx, idx_to_char
    print("Ошибка: Модель не найдена или повреждена")
    return None, None, None

def generate_response(model, prompt, max_length=500, temperature=0.7):
    """Генерация ответа на основе промпта"""
    device = next(model.parameters()).device
    model.eval()
    
    # Преобразование промпта в индексы
    input_seq = []
    for char in prompt:
        if char in model.char_to_idx:
            input_seq.append(model.char_to_idx[char])
        else:
            # Обработка неизвестных символов
            input_seq.append(model.char_to_idx.get(' ', 0))
    
    # Инициализация скрытого состояния
    hidden = model.init_hidden(1, device)
    
    # "Прогрев" модели с промптом
    with torch.no_grad():
        for char in input_seq[:-1]:
            x = torch.tensor([[char]], dtype=torch.long).to(device)
            _, hidden = model(x, hidden)
    
    # Генерация ответа
    response = prompt
    next_char = input_seq[-1]
    
    for _ in range(max_length):
        x = torch.tensor([[next_char]], dtype=torch.long).to(device)
        with torch.no_grad():
            output, hidden = model(x, hidden)
        
        # Применяем температуру
        probs = torch.softmax(output[0, -1] / temperature, dim=-1).cpu().numpy()
        next_char = np.random.choice(len(probs), p=probs)
        
        # Преобразуем индекс обратно в символ
        char = model.idx_to_char.get(next_char, '�')
        response += char
        
        # Остановка при достижении конца предложения
        if char in ".!?\n":
            break
    
    return response

def chat_interface():
    """Интерфейс чата с моделью"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Загрузка модели
    model, char_to_idx, idx_to_char = load_model_for_chat(device)
    if not model:
        return
    
    print("\n" + "=" * 60)
    print("Чат с русскоязычной языковой моделью")
    print("Введите ваш вопрос или сообщение (для выхода введите 'выход')")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nВы: ")
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("Завершение сеанса...")
                break
                
            if not user_input.strip():
                print("Пожалуйста, введите сообщение")
                continue
                
            # Генерация ответа
            response = generate_response(
                model, 
                user_input + "\nМодель: ",
                max_length=500,
                temperature=0.7
            )
            
            # Извлекаем только ответ модели (после метки)
            if "Модель: " in response:
                model_response = response.split("Модель: ", 1)[1]
            else:
                model_response = response
                
            # Убираем возможные артефакты генерации
            model_response = model_response.replace("�", "").strip()
            print(f"Модель: {model_response}")
            
        except KeyboardInterrupt:
            print("\nЗавершение сеанса...")
            break
        except Exception as e:
            print(f"Ошибка генерации: {str(e)}")

if __name__ == "__main__":
    chat_interface()