import torch
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_text(model, start_str, length, char_to_idx, idx_to_char, device, temperature=0.8):
    model.eval()
    chars = [char_to_idx[c] for c in start_str]
    hidden = model.init_hidden(1, device)
    
    for char in chars[:-1]:
        x = torch.tensor([[char]], dtype=torch.long).to(device)
        _, hidden = model(x, hidden)
    
    next_char = chars[-1]
    for _ in range(length):
        x = torch.tensor([[next_char]], dtype=torch.long).to(device)
        output, hidden = model(x, hidden)
        probs = torch.softmax(output[0, -1] / temperature, dim=-1).detach().cpu().numpy()
        next_char = np.random.choice(len(probs), p=probs)
        chars.append(next_char)
    
    return ''.join([idx_to_char[i] for i in chars])

def save_model(model, optimizer, epoch, char_to_idx, idx_to_char, hidden_size, vocab_size, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size
    }, path)
    print(f"Модель сохранена в {path}")

def save_model(model, optimizer, epoch, char_to_idx, idx_to_char, hidden_size, vocab_size, path):
    """Сохранение модели с логированием"""
    try:
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'hidden_size': hidden_size,
            'vocab_size': vocab_size
        }, path)
        log_print(f"Модель сохранена в {path}")
    except Exception as e:
        log_print(f"Ошибка сохранения модели: {str(e)}")

def safe_load_model(path, device, model_class):
    """Безопасная загрузка модели с обработкой ошибок и логированием"""
    try:
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=device)
            model = model_class(
                checkpoint['vocab_size'],
                checkpoint['hidden_size'],
                checkpoint['vocab_size']
            ).to(device)
            model.load_state_dict(checkpoint['model_state'])
            return model, checkpoint
        log_print(f"Файл модели не найден: {path}")
        return None, None
    except Exception as e:
        log_print(f"Ошибка загрузки модели: {str(e)}")
        return None, None
    
def log_print(*args, **kwargs):
    """Печатает сообщение с временной меткой"""
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    # Преобразуем все аргументы в строки
    message = ' '.join(str(arg) for arg in args)
    print(timestamp, message, **kwargs)

def get_device():
    """Определение устройства с учетом Colab"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Оптимизация для GPU
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        return device, "gpu"
    elif torch.backends.mps.is_available():
        return torch.device('mps'), "mps"
    else:
        return torch.device('cpu'), "cpu"
    
def train_epoch(model, loader, optimizer, criterion, device):
    """Обучение на одной эпохе с прогресс-баром"""
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc="Обучение", leave=False)
    
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Инициализация скрытого состояния
        hidden = model.init_hidden(inputs.size(0), device)
        
        # Прямой проход и оптимизация
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)