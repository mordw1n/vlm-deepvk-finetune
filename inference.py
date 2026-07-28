import argparse
import logging
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LlavaForConditionalGeneration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEFAULT_MODEL = 'llava-hf/llava-1.5-7b-hf'

def load_model(model_path: str):
    logger.info(f'Loading model: {model_path}')
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, trust_remote_code=True)
    model.eval()
    logger.info('Model ready!')
    return (model, processor)

def generate_response(model, processor, image: Image.Image, question: str, max_new_tokens: int=256, temperature: float=0.1) -> str:
    conversation = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
    try:
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    except Exception:
        prompt = f'<image>\nПользователь: {question}\nАссистент:'
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature > 0, pad_token_id=processor.tokenizer.eos_token_id)
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return answer

def create_demo_image() -> Image.Image:
    img = Image.new('RGB', (400, 300), color=(70, 130, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(255, 215, 0), outline=(0, 0, 0), width=3)
    draw.ellipse([200, 75, 350, 225], fill=(255, 99, 71), outline=(0, 0, 0), width=3)
    try:
        draw.text((100, 200), 'Тест VLM', fill=(255, 255, 255))
    except Exception:
        pass
    return img

def demo_mode(model, processor):
    print('\n' + '=' * 60)
    print('  VLM INFERENCE DEMO')
    print('  Визуально-языковая модель (deepvk)')
    print('=' * 60)
    image = create_demo_image()
    image.save('demo_image.jpg')
    print('\nСоздано тестовое изображение: demo_image.jpg')
    questions = ['Что изображено на картинке?', 'Какие цвета присутствуют на изображении?', 'Сколько геометрических фигур на изображении?', 'Опиши изображение подробно.']
    for (i, question) in enumerate(questions, 1):
        print(f'\n[Вопрос {i}]: {question}')
        answer = generate_response(model, processor, image, question)
        print(f'[Ответ]:   {answer}')
    print('\n' + '=' * 60)

def interactive_mode(model, processor):
    print('\n' + '=' * 60)
    print('  ИНТЕРАКТИВНЫЙ РЕЖИМ VLM')
    print('  Введите путь к изображению, затем задавайте вопросы')
    print("  Команда 'exit' для выхода, 'new' для нового изображения")
    print('=' * 60)
    current_image = None
    while True:
        if current_image is None:
            img_path = input("\nПуть к изображению (или 'demo'): ").strip()
            if img_path.lower() == 'exit':
                break
            elif img_path.lower() == 'demo':
                current_image = create_demo_image()
                print('Используется демо-изображение')
            else:
                try:
                    current_image = Image.open(img_path).convert('RGB')
                    print(f'Изображение загружено: {img_path}')
                except Exception as e:
                    print(f'Ошибка: {e}')
                    continue
        question = input("\nВопрос (или 'new'/'exit'): ").strip()
        if question.lower() == 'exit':
            break
        elif question.lower() == 'new':
            current_image = None
            continue
        elif not question:
            question = 'Что изображено на картинке?'
        print('Генерация ответа...')
        answer = generate_response(model, processor, current_image, question)
        print(f'Ответ: {answer}')

def main():
    parser = argparse.ArgumentParser(description='VLM Inference')
    parser.add_argument('--model_path', type=str, default=None, help=f'Path to model (default: {DEFAULT_MODEL})')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    parser.add_argument('--question', type=str, default='Что изображено на этом изображении? Опиши подробно.', help='Question about the image')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Generation temperature')
    args = parser.parse_args()
    model_path = args.model_path
    if model_path is None:
        checkpoint_path = Path('./checkpoints/best_model')
        if checkpoint_path.exists():
            model_path = str(checkpoint_path)
            logger.info(f'Using fine-tuned model: {model_path}')
        else:
            model_path = DEFAULT_MODEL
            logger.info(f'No fine-tuned model found, using: {model_path}')
    try:
        (model, processor) = load_model(model_path)
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        logger.info('Please install requirements: pip install -r requirements.txt')
        logger.info('And ensure you have internet access to download the model')
        sys.exit(1)
    if args.demo:
        demo_mode(model, processor)
    elif args.interactive:
        interactive_mode(model, processor)
    elif args.image:
        try:
            image = Image.open(args.image).convert('RGB')
        except Exception as e:
            logger.error(f'Failed to load image: {e}')
            sys.exit(1)
        print(f'\nИзображение: {args.image}')
        print(f'Вопрос: {args.question}')
        print('\nГенерация ответа...')
        answer = generate_response(model, processor, image, args.question, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print(f'\nОтвет: {answer}')
    else:
        parser.print_help()
        print('\nПример использования:')
        print('  python inference.py --demo')
        print("  python inference.py --image photo.jpg --question 'Что на картинке?'")
        print('  python inference.py --interactive')
if __name__ == '__main__':
    main()