import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from config import benchmark_config
from utils.metrics import compute_gqa_accuracy, compute_mmbench_score, compute_metrics, print_metrics_table
from utils.dataset import load_deepvk_dataset, load_image
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMEvaluator:

    def __init__(self, model_path: str, device: str='auto'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        logger.info(f'Loading model from: {self.model_path}')
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, device_map=self.device, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, trust_remote_code=True)
        self.model.eval()
        logger.info('Model loaded successfully!')

    def generate_answer(self, image: Image.Image, question: str) -> str:
        conversation = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
        try:
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception:
            prompt = f'<image>\nПользователь: {question}\nАссистент:'
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=benchmark_config.max_new_tokens, temperature=benchmark_config.temperature, do_sample=benchmark_config.temperature > 0, pad_token_id=self.processor.tokenizer.eos_token_id)
        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return answer

    def evaluate_gqa_ru(self, max_samples: Optional[int]=None) -> Dict:
        logger.info('Evaluating on GQA-ru benchmark...')
        max_samples = max_samples or benchmark_config.gqa_ru_max_samples
        try:
            data = load_deepvk_dataset(benchmark_config.gqa_ru_dataset, split=benchmark_config.gqa_ru_split, max_samples=max_samples)
        except Exception as e:
            logger.error(f'Failed to load GQA-ru: {e}')
            return {'error': str(e)}
        predictions = []
        references = []
        for item in tqdm(data, desc='GQA-ru'):
            try:
                image = load_image(item.get('image', ''))
                question = item.get('question', '')
                reference = item.get('answer', '')
                prediction = self.generate_answer(image, question)
                predictions.append(prediction)
                references.append(reference)
            except Exception as e:
                logger.warning(f'Error processing item: {e}')
                predictions.append('')
                references.append(item.get('answer', ''))
        results = compute_gqa_accuracy(predictions, references)
        results.update(compute_metrics(predictions, references))
        return results

    def evaluate_mmbench_ru(self, max_samples: Optional[int]=None) -> Dict:
        logger.info('Evaluating on MMBENCH-ru benchmark...')
        max_samples = max_samples or benchmark_config.mmbench_ru_max_samples
        try:
            data = load_deepvk_dataset(benchmark_config.mmbench_ru_dataset, split=benchmark_config.mmbench_ru_split, max_samples=max_samples)
        except Exception as e:
            logger.error(f'Failed to load MMBENCH-ru: {e}')
            return {'error': str(e)}
        predictions = []
        references = []
        categories = []
        for item in tqdm(data, desc='MMBENCH-ru'):
            try:
                image = load_image(item.get('image', ''))
                question = item.get('question', '')
                reference = item.get('answer', item.get('correct_answer', ''))
                category = item.get('category', 'unknown')
                prediction = self.generate_answer(image, question)
                predictions.append(prediction)
                references.append(reference)
                categories.append(category)
            except Exception as e:
                logger.warning(f'Error processing item: {e}')
                predictions.append('')
                references.append(item.get('answer', ''))
                categories.append('unknown')
        results = compute_mmbench_score(predictions, references, categories)
        return results

def run_demo_evaluation():
    import random
    logger.info('Running DEMO evaluation (mock results)...')
    n_gqa = 1000
    gqa_predictions = [random.choice(['да', 'нет', 'красный', 'синий', 'большой', 'маленький']) for _ in range(n_gqa)]
    gqa_references = [random.choice(['да', 'нет', 'красный', 'синий', 'большой', 'маленький']) for _ in range(n_gqa)]
    gqa_results = compute_gqa_accuracy(gqa_predictions, gqa_references)
    gqa_results['benchmark'] = 'GQA-ru'
    n_mmb = 500
    choices = ['A', 'B', 'C', 'D']
    mmb_predictions = [random.choice(choices) for _ in range(n_mmb)]
    mmb_references = [random.choice(choices) for _ in range(n_mmb)]
    mmb_categories = [random.choice(['reasoning', 'perception', 'knowledge', 'ocr']) for _ in range(n_mmb)]
    mmb_results = compute_mmbench_score(mmb_predictions, mmb_references, mmb_categories)
    mmb_results['benchmark'] = 'MMBENCH-ru'
    print_metrics_table(gqa_results, title='GQA-ru Results (DEMO)')
    print_metrics_table(mmb_results, title='MMBENCH-ru Results (DEMO)')
    all_results = {'gqa_ru': gqa_results, 'mmbench_ru': mmb_results}
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        clean_results = {}
        for (k, v) in all_results.items():
            clean_results[k] = {kk: vv for (kk, vv) in v.items() if isinstance(vv, (int, float, str, dict))}
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    logger.info('Demo results saved to evaluation_results.json')
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM on GQA-ru and MMBENCH-ru')
    parser.add_argument('--model_path', type=str, default=None, help='Path to fine-tuned model (None = demo mode)')
    parser.add_argument('--benchmark', type=str, default='all', choices=['gqa', 'mmbench', 'all'], help='Which benchmark to evaluate on')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json', help='Output file for results')
    parser.add_argument('--demo', action='store_true', help='Run demo evaluation with mock results')
    args = parser.parse_args()
    if args.demo or args.model_path is None:
        run_demo_evaluation()
        return
    if not Path(args.model_path).exists():
        logger.error(f'Model path does not exist: {args.model_path}')
        sys.exit(1)
    evaluator = VLMEvaluator(model_path=args.model_path)
    evaluator.load()
    all_results = {}
    if args.benchmark in ['gqa', 'all']:
        gqa_results = evaluator.evaluate_gqa_ru(max_samples=args.max_samples)
        all_results['gqa_ru'] = gqa_results
        print_metrics_table(gqa_results, title='GQA-ru Results')
    if args.benchmark in ['mmbench', 'all']:
        mmbench_results = evaluator.evaluate_mmbench_ru(max_samples=args.max_samples)
        all_results['mmbench_ru'] = mmbench_results
        print_metrics_table(mmbench_results, title='MMBENCH-ru Results')
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f'Results saved to {args.output_file}')
if __name__ == '__main__':
    main()