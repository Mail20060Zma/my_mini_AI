from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_all_text(raw_dir, parsed_dir):
    text = ""
    print("\nЗагрузка текстовых данных...")
    
    for dir_path in [raw_dir, parsed_dir]:
        for filepath in dir_path.glob("*.txt"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    text += file_text + "\n"
                    print(f"Загружен файл: {filepath.name} ({len(file_text)} символов)")
            except Exception as e:
                print(f"Ошибка чтения {filepath}: {str(e)}")
    
    print(f"Общий размер данных: {len(text):,} символов")
    return text

def prepare_data(raw_dir, parsed_dir):
    text = load_all_text(raw_dir, parsed_dir)
    
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return text, char_to_idx, idx_to_char, len(chars)

def create_dataset(text, char_to_idx, seq_length):
    encoded = [char_to_idx[ch] for ch in text]
    data = []
    
    for i in range(0, len(encoded) - seq_length - 1, seq_length):
        input_seq = encoded[i:i+seq_length]
        target_seq = encoded[i+1:i+seq_length+1]
        data.append((input_seq, target_seq))
    
    return data

def create_data_loader(data, batch_size):
    train_data = torch.tensor([item[0] for item in data], dtype=torch.long)
    train_labels = torch.tensor([item[1] for item in data], dtype=torch.long)
    dataset = TensorDataset(train_data, train_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)