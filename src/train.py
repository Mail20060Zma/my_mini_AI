import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from model import CharLSTM
from parser import RussianTextParser
from data_loader import prepare_data, create_dataset, create_data_loader
from utils import generate_text, save_model, load_model, safe_load_model, log_print, get_device

# Конфигурация
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
MODEL_DIR = PROJECT_ROOT / "models"

# Создание директорий
for dir_path in [DATA_DIR, RAW_DIR, PARSED_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

try:
    from colab_setup import setup_colab_environment, sync_with_drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Параметры
MODEL_PATH = MODEL_DIR / "char_lstm_model.pt"
SEQ_LENGTH = 100
HIDDEN_SIZE = 512
BATCH_SIZE = 64
LEARNING_RATE = 0.005
NUM_EPOCHS = 1000000  # Очень большое число для бесконечного обучения
PRINT_EVERY = 1
MIN_DATA_LENGTH = 50000
DATA_CHECK_INTERVAL = 100  # Проверять данные каждые 100 эпох
MIN_FILES_COUNT = 3  # Минимальное количество файлов для начала обучения

def train_model():
    # Определение устройства
    if IN_COLAB:
        colab_path, drive_path = setup_colab_environment()
        PROJECT_ROOT = colab_path
    else:
        PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    device, device_name = get_device()
    log_print(f"Используемое устройство: {device_name.upper()}")
    
    # Инициализация парсера
    parser = RussianTextParser(RAW_DIR, PARSED_DIR)
    
    # Основной цикл обучения
    while True:
        # Проверка и сбор данных
        raw_files = list(RAW_DIR.glob('*.txt'))
        parsed_files = list(PARSED_DIR.glob('*.txt'))
        total_files = len(raw_files) + len(parsed_files)
        
        if total_files < MIN_FILES_COUNT:
            log_print(f"Недостаточно данных ({total_files} файлов), запускаем парсинг...")
            parser.run_parsing()
            continue
            
        # Подготовка данных
        text, char_to_idx, idx_to_char, vocab_size = prepare_data(RAW_DIR, PARSED_DIR)
        
        # Проверка объема данных
        if len(text) < MIN_DATA_LENGTH:
            log_print(f"Недостаточно текста ({len(text)} символов), запускаем парсинг...")
            parser.run_parsing()
            continue
            
        # Создание датасета
        data = create_dataset(text, char_to_idx, SEQ_LENGTH)
        loader = create_data_loader(data, BATCH_SIZE)
        
        # Инициализация модели
        model = CharLSTM(vocab_size, HIDDEN_SIZE, vocab_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        epoch_counter = 0
        
        # Загрузка существующей модели
        if MODEL_PATH.exists():
            log_print(f"Загружаем существующую модель: {MODEL_PATH}")
            model, checkpoint = safe_load_model(MODEL_PATH, device, CharLSTM)
            
            if model and checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_epoch = checkpoint['epoch'] + 1
                char_to_idx = checkpoint['char_to_idx']
                idx_to_char = checkpoint['idx_to_char']
                vocab_size = checkpoint['vocab_size']
                log_print(f"Продолжаем обучение с эпохи {start_epoch}")
            else:
                log_print("Ошибка загрузки, создаем новую модель")
        else:
            log_print("Создаем новую модель")
        
        # Цикл обучения
        try:
            for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
                total_loss = 0
                
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Инициализация скрытого состояния
                    hidden = model.init_hidden(inputs.size(0), device)
                    
                    # Прямой проход и оптимизация
                    optimizer.zero_grad()
                    output, hidden = model(inputs, hidden)
                    loss = criterion(output.view(-1, vocab_size), targets.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                epoch_counter += 1
                
                # Проверка данных каждые DATA_CHECK_INTERVAL эпох
                if epoch_counter % DATA_CHECK_INTERVAL == 0:
                    log_print("\nПроверка состояния данных...")
                    break  # Выйти из цикла обучения для проверки данных
                
                # Логирование прогресса
                if (epoch_counter + 1) % PRINT_EVERY == 0:
                    epoch_loss = total_loss / len(loader)
                    log_print(f"Эпоха {epoch_counter+1}/{NUM_EPOCHS}, Потеря: {epoch_loss:.4f}")
                    
                    # Генерация примера текста
                    sample = generate_text(
                        model, "Наука", 200, 
                        char_to_idx, idx_to_char, device
                    )
                    log_print(f"Пример генерации:\n{sample}\n")
                    
                    # Сохранение модели
                    save_model(
                        model, optimizer, epoch_counter,
                        char_to_idx, idx_to_char,
                        HIDDEN_SIZE, vocab_size,
                        MODEL_PATH
                    )
        
        except KeyboardInterrupt:
            log_print("\nОбучение прервано пользователем")
            save_model(
                model, optimizer, epoch_counter,
                char_to_idx, idx_to_char,
                HIDDEN_SIZE, vocab_size,
                MODEL_PATH
            )
            generated_text = generate_text(
                model, "Завершение", 100, 
                char_to_idx, idx_to_char, device
            )
            log_print(f"Финальная генерация:\n{generated_text}")
            return

if __name__ == "__main__":
    # Подсчет файлов перед выводом
    raw_count = len(list(RAW_DIR.glob('*.txt')))
    parsed_count = len(list(PARSED_DIR.glob('*.txt')))
    total_files = raw_count + parsed_count
    
    log_print("=" * 60)
    log_print(f"Проект: Создание русскоязычной языковой модели")
    log_print(f"Корневая директория: {PROJECT_ROOT}")
    log_print(f"Данные: {DATA_DIR} ({total_files} файлов)")
    log_print(f"Модели: {MODEL_DIR}")
    log_print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 60)
    
    try:
        train_model()
    except Exception as e:
        log_print(f"Критическая ошибка: {str(e)}")
        log_print("Программа завершена")