import os
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
import requests
from io import BytesIO

def load_image(image_source) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert('RGB')
    elif isinstance(image_source, str):
        if image_source.startswith('http://') or image_source.startswith('https://'):
            response = requests.get(image_source, timeout=10)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_source)
        return image.convert('RGB')
    else:
        raise ValueError(f'Unsupported image source type: {type(image_source)}')

class VLMDataset(Dataset):

    def __init__(self, data: List[Dict], processor: AutoProcessor, max_length: int=2048, image_size: int=336, split: str='train'):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        try:
            image = load_image(item.get('image', item.get('image_path', '')))
        except Exception as e:
            print(f'Error loading image for item {idx}: {e}')
            image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
        question = item.get('question', item.get('instruction', 'Опиши изображение.'))
        answer = item.get('answer', item.get('response', ''))
        conversation = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}, {'role': 'assistant', 'content': answer}]
        try:
            inputs = self.processor(images=image, text=self.processor.apply_chat_template(conversation, tokenize=False), return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        except Exception:
            prompt = f'<image>\nПользователь: {question}\nАссистент: {answer}'
            inputs = self.processor(images=image, text=prompt, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        return {k: v.squeeze(0) for (k, v) in inputs.items()}

def load_deepvk_dataset(dataset_name: str, split: str='train', max_samples: Optional[int]=None) -> List[Dict]:
    from datasets import load_dataset
    print(f'Loading dataset: {dataset_name} ({split})')
    try:
        dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f'Loaded {len(dataset)} samples from {dataset_name}')
        return list(dataset)
    except Exception as e:
        print(f'Error loading {dataset_name}: {e}')
        print('Generating synthetic data for testing...')
        return generate_synthetic_data(max_samples or 100)

def generate_synthetic_data(n_samples: int=100) -> List[Dict]:
    import random
    questions = ['Что изображено на этой картинке?', 'Какого цвета основной объект на изображении?', 'Сколько объектов находится на изображении?', 'Опиши сцену на изображении.', 'Что происходит на фотографии?']
    answers = ['На изображении показан предмет повседневного использования.', 'Основной объект красного цвета.', 'На изображении находится несколько объектов.', 'На сцене изображена типичная бытовая обстановка.', 'На фотографии запечатлён момент повседневной жизни.']
    data = []
    for i in range(n_samples):
        img = Image.new('RGB', (336, 336), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        data.append({'image': img, 'question': random.choice(questions), 'answer': random.choice(answers), 'id': f'synthetic_{i}'})
    return data

def split_dataset(data: List[Dict], train_ratio: float=0.9) -> Tuple[List[Dict], List[Dict]]:
    n_train = int(len(data) * train_ratio)
    return (data[:n_train], data[n_train:])