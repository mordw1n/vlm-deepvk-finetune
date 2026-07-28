import re
import string
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np

def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def vqa_accuracy(predictions: List[str], references: List[List[str]]) -> float:
    scores = []
    for (pred, refs) in zip(predictions, references):
        norm_pred = normalize_answer(pred)
        matching = sum((1 for ref in refs if normalize_answer(ref) == norm_pred))
        score = min(matching / 3.0, 1.0)
        scores.append(score)
    return np.mean(scores) if scores else 0.0

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    if not predictions or not references:
        return {'exact_match': 0.0, 'f1': 0.0}
    em_scores = [exact_match(pred, ref) for (pred, ref) in zip(predictions, references)]
    f1_scores = [f1_score(pred, ref) for (pred, ref) in zip(predictions, references)]
    return {'exact_match': np.mean(em_scores), 'f1': np.mean(f1_scores), 'exact_match_std': np.std(em_scores), 'f1_std': np.std(f1_scores), 'n_samples': len(predictions)}

def compute_gqa_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    correct = sum((1 for (pred, ref) in zip(predictions, references) if normalize_answer(pred) == normalize_answer(ref)))
    accuracy = correct / len(predictions) if predictions else 0.0
    return {'gqa_accuracy': accuracy, 'correct': correct, 'total': len(predictions)}

def compute_mmbench_score(predictions: List[str], references: List[str], categories: Optional[List[str]]=None) -> Dict[str, float]:
    correct = sum((1 for (pred, ref) in zip(predictions, references) if normalize_answer(pred) == normalize_answer(ref)))
    overall_accuracy = correct / len(predictions) if predictions else 0.0
    result = {'mmbench_accuracy': overall_accuracy, 'correct': correct, 'total': len(predictions)}
    if categories:
        from collections import defaultdict
        cat_correct = defaultdict(int)
        cat_total = defaultdict(int)
        for (pred, ref, cat) in zip(predictions, references, categories):
            cat_total[cat] += 1
            if normalize_answer(pred) == normalize_answer(ref):
                cat_correct[cat] += 1
        result['per_category'] = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}
    return result

def print_metrics_table(results: Dict[str, float], title: str='Evaluation Results'):
    print(f"\n{'=' * 50}")
    print(f'  {title}')
    print(f"{'=' * 50}")
    for (key, value) in results.items():
        if key == 'per_category':
            print(f'\n  Per-category breakdown:')
            for (cat, acc) in value.items():
                print(f'    {cat:30s}: {acc:.4f} ({acc * 100:.2f}%)')
        elif isinstance(value, float):
            print(f'  {key:25s}: {value:.4f} ({value * 100:.2f}%)')
        else:
            print(f'  {key:25s}: {value}')
    print(f"{'=' * 50}\n")