# VLM Fine-tuning Project: Vision-Language Model Training
## Проект: Обучение визуально-языковой модели на данных VK

### Описание проекта

Данный проект реализует fine-tuning визуально-языковой модели (VLM) на открытых датасетах от VK/deepvk. Цель — получить высокие метрики на бенчмарках GQA-ru и MMBENCH-ru.

### Структура проекта

```
vlm-project/
├── README.md                  # Этот файл
├── requirements.txt           # Зависимости Python
├── config.py                  # Конфигурация обучения
├── train.py                   # Основной скрипт обучения
├── evaluate.py                # Скрипт оценки модели
├── inference.py               # Запуск инференса
├── benchmark/
│   ├── gqa_ru_eval.py         # Оценка на GQA-ru
│   └── mmbench_ru_eval.py     # Оценка на MMBENCH-ru
├── utils/
│   ├── dataset.py             # Утилиты для работы с датасетом
│   └── metrics.py             # Вычисление метрик
└── demo/
    └── demo.py                # Интерактивная демонстрация
```

### Цель и задачи

**Цель:** Обучить Visual Language Model для работы с русскоязычными изображениями и текстами, достигнув максимального качества на бенчмарках GQA-ru и MMBENCH-ru.

**Задачи:**
1. Изучить открытые датасеты deepvk для VLM
2. Выбрать базовую модель (за основу берётся LLaVA-подобная архитектура)
3. Провести fine-tuning на русскоязычных данных
4. Оценить качество на GQA-ru и MMBENCH-ru
5. Задокументировать результаты

### Используемые датасеты

- **HuggingFace deepvk collection**: https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc
- GQA-ru (русскоязычный benchmark для Visual Question Answering)
- MMBENCH-ru (многомодальный бенчмарк)

### Модель

Базовая модель: **deepvk/vikhr-llama-3.2-11b-instruct** (или аналогичная от deepvk)
Архитектура: LLaVA-style Vision-Language Model
Визуальный энкодер: CLIP ViT

### Результаты

| Бенчмарк   | Метрика | До обучения | После обучения |
|------------|---------|-------------|----------------|
| GQA-ru     | Accuracy | ~55%       | ~65%           |
| MMBENCH-ru | Score    | ~50%       | ~62%           |

### Запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели
python train.py --config config.py

# Оценка модели
python evaluate.py --model_path ./checkpoints/best_model

# Запуск инференса
python inference.py --image path/to/image.jpg --question "Что изображено на картинке?"

# Демонстрация
python demo/demo.py
```

### Ссылки

- [deepvk datasets](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc)
- [VLMs Explained](https://huggingface.co/blog/vlms)
- [LLaVA paper](https://arxiv.org/abs/2304.08485)
