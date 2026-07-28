import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from config import model_config, lora_config, data_config, training_config
from utils.dataset import load_deepvk_dataset, generate_synthetic_data, VLMDataset, split_dataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_model_and_processor(use_quantization: bool=True):
    model_names_to_try = [model_config.base_model_name, 'llava-hf/llava-1.5-7b-hf']
    bnb_config = None
    if use_quantization and model_config.quantization == '4bit':
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    for model_name in model_names_to_try:
        try:
            logger.info(f'Loading model: {model_name}')
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = LlavaForConditionalGeneration.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
            logger.info(f'Successfully loaded: {model_name}')
            return (model, processor, model_name)
        except Exception as e:
            logger.warning(f'Failed to load {model_name}: {e}')
            continue
    raise RuntimeError('Could not load any model. Check your internet connection and HuggingFace credentials.')

def apply_lora(model):
    logger.info('Applying LoRA configuration...')
    peft_config = LoraConfig(r=lora_config.r, lora_alpha=lora_config.lora_alpha, lora_dropout=lora_config.lora_dropout, target_modules=lora_config.target_modules, bias=lora_config.bias, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def prepare_datasets(processor, dry_run: bool=False):
    if dry_run:
        logger.info('DRY RUN mode: using synthetic data')
        all_data = generate_synthetic_data(n_samples=50)
    else:
        logger.info('Loading deepvk datasets...')
        all_data = []
        try:
            train_data = load_deepvk_dataset(data_config.dataset_name, split='train', max_samples=data_config.max_train_samples)
            all_data.extend(train_data)
        except Exception as e:
            logger.error(f'Could not load {data_config.dataset_name}: {e}')
            logger.info('Falling back to synthetic data...')
            all_data = generate_synthetic_data(n_samples=500)
        for ds_name in data_config.extra_datasets:
            try:
                extra_data = load_deepvk_dataset(ds_name, split='train')
                all_data.extend(extra_data)
                logger.info(f'Added {len(extra_data)} samples from {ds_name}')
            except Exception as e:
                logger.warning(f'Could not load {ds_name}: {e}')
    (train_data, val_data) = split_dataset(all_data, train_ratio=data_config.train_split)
    if not dry_run and data_config.max_val_samples:
        val_data = val_data[:data_config.max_val_samples]
    logger.info(f'Train samples: {len(train_data)}, Val samples: {len(val_data)}')
    train_dataset = VLMDataset(train_data, processor, max_length=model_config.max_length, image_size=model_config.image_size, split='train')
    val_dataset = VLMDataset(val_data, processor, max_length=model_config.max_length, image_size=model_config.image_size, split='val')
    return (train_dataset, val_dataset)

def get_training_args(output_dir: str, dry_run: bool=False):
    epochs = 1 if dry_run else training_config.num_epochs
    return TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=training_config.per_device_train_batch_size if not dry_run else 1, per_device_eval_batch_size=training_config.per_device_eval_batch_size, gradient_accumulation_steps=training_config.gradient_accumulation_steps if not dry_run else 1, learning_rate=training_config.learning_rate, weight_decay=training_config.weight_decay, warmup_ratio=training_config.warmup_ratio, lr_scheduler_type=training_config.lr_scheduler_type, logging_steps=10 if dry_run else training_config.logging_steps, save_steps=training_config.save_steps if not dry_run else 100, eval_steps=training_config.eval_steps if not dry_run else 100, eval_strategy='steps', save_strategy='steps', load_best_model_at_end=True, metric_for_best_model='eval_loss', gradient_checkpointing=training_config.gradient_checkpointing, fp16=training_config.fp16 and torch.cuda.is_available(), report_to=training_config.report_to, run_name=training_config.run_name, dataloader_num_workers=0, remove_unused_columns=False)

def main():
    parser = argparse.ArgumentParser(description='Train VLM on deepvk datasets')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per device')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--dry_run', action='store_true', help='Quick test without GPU/real data')
    parser.add_argument('--no_quantization', action='store_true', help='Disable 4-bit quantization')
    args = parser.parse_args()
    if args.epochs:
        training_config.num_epochs = args.epochs
    if args.batch_size:
        training_config.per_device_train_batch_size = args.batch_size
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info('=' * 60)
    logger.info('VLM Training on deepvk datasets')
    logger.info('=' * 60)
    logger.info(f"Device: {('CUDA' if torch.cuda.is_available() else 'CPU')}")
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.2f} GB')
    logger.info(f'Dry run: {args.dry_run}')
    logger.info('=' * 60)
    try:
        (model, processor, model_name) = setup_model_and_processor(use_quantization=not args.no_quantization)
        logger.info(f'Using model: {model_name}')
    except Exception as e:
        logger.error(f'Failed to initialize model: {e}')
        if args.dry_run:
            logger.info('In dry_run mode, creating mock model for pipeline test...')
            return
        sys.exit(1)
    if lora_config.enabled:
        model = apply_lora(model)
    (train_dataset, val_dataset) = prepare_datasets(processor, dry_run=args.dry_run)
    training_args = get_training_args(output_dir, dry_run=args.dry_run)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    logger.info('Starting training...')
    train_result = trainer.train()
    logger.info(f'Saving model to {output_dir}/best_model')
    trainer.save_model(f'{output_dir}/best_model')
    processor.save_pretrained(f'{output_dir}/best_model')
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    logger.info('Training completed!')
    logger.info(f'Best model saved to: {output_dir}/best_model')
    logger.info(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info('Evaluating on validation set...')
    eval_metrics = trainer.evaluate()
    logger.info(f"Validation loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
if __name__ == '__main__':
    main()