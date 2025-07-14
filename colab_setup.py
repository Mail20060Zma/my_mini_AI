# colab_setup.py
import os
import shutil
from pathlib import Path
from google.colab import drive

def setup_colab_environment():
    """Настройка среды в Google Colab"""
    # Монтирование Google Drive
    drive.mount('/content/drive')
    
    # Пути для проекта
    colab_project_path = Path("/content/lstm_project")
    drive_project_path = Path("/content/drive/MyDrive/lstm_project")
    
    # Создание директорий
    colab_project_path.mkdir(exist_ok=True)
    drive_project_path.mkdir(exist_ok=True)
    
    # Структура папок
    dirs = [
        "data/raw",
        "data/parsed",
        "models",
        "src"
    ]
    
    for d in dirs:
        (colab_project_path / d).mkdir(parents=True, exist_ok=True)
        (drive_project_path / d).mkdir(parents=True, exist_ok=True)
    
    # Копирование файлов из Drive (если они есть)
    for file_type in ['*.py', '*.txt']:
        for src in drive_project_path.glob(f"src/{file_type}"):
            dst = colab_project_path / "src" / src.name
            if not dst.exists():
                shutil.copy(src, dst)
    
    # Установка зависимостей
    os.system("pip install torch numpy beautifulsoup4 requests wikipedia-api feedparser tqdm")
    
    return colab_project_path, drive_project_path

def sync_with_drive(src_path, dst_path):
    """Синхронизация результатов с Google Drive"""
    # Синхронизация данных
    for data_type in ['raw', 'parsed']:
        src_data = src_path / "data" / data_type
        dst_data = dst_path / "data" / data_type
        os.system(f"rsync -a {src_data}/ {dst_data}/")
    
    # Синхронизация моделей
    src_models = src_path / "models"
    dst_models = dst_path / "models"
    os.system(f"rsync -a {src_models}/ {dst_models}/")
    
    # Синхронизация исходного кода
    src_src = src_path / "src"
    dst_src = dst_path / "src"
    os.system(f"rsync -a {src_src}/ {dst_src}/")