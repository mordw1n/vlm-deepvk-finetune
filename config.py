from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    base_model_name: str = 'deepvk/vikhr-llama-3.2-11b-instruct'
    vision_encoder: str = 'openai/clip-vit-large-patch14-336'
    quantization: Optional[str] = '4bit'
    image_size: int = 336
    max_length: int = 2048

@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda : ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    bias: str = 'none'
    task_type: str = 'CAUSAL_LM'

@dataclass
class DataConfig:
    dataset_name: str = 'deepvk/COCO-ru'
    extra_datasets: List[str] = field(default_factory=lambda : ['deepvk/GQA-ru'])
    train_split: float = 0.9
    val_split: float = 0.1
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = 1000

@dataclass
class TrainingConfig:
    output_dir: str = './checkpoints'
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = 'cosine'
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    gradient_checkpointing: bool = True
    fp16: bool = True
    optim: str = 'adamw_torch_fused'
    report_to: str = 'none'
    run_name: str = 'vlm-deepvk-finetune'

@dataclass
class BenchmarkConfig:
    gqa_ru_dataset: str = 'deepvk/GQA-ru'
    gqa_ru_split: str = 'validation'
    gqa_ru_max_samples: int = 1000
    mmbench_ru_dataset: str = 'deepvk/MMBench-ru'
    mmbench_ru_split: str = 'test'
    mmbench_ru_max_samples: int = 500
    temperature: float = 0.1
    max_new_tokens: int = 128
model_config = ModelConfig()
lora_config = LoRAConfig()
data_config = DataConfig()
training_config = TrainingConfig()
benchmark_config = BenchmarkConfig()