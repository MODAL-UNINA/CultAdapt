# ============================================================
# MODEL IDENTIFIERS
# ============================================================
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
JUDGE_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q8_K_XL"

# ============================================================
# DIRECTORY PATHS
# ============================================================
DIR_BASE = "data_splits"
DIR_TRAIN = f"{DIR_BASE}/train"
DIR_VAL = f"{DIR_BASE}/val"
DIR_IMAGES_GENERIC = f"{DIR_BASE}/images_generic"

ADAPTERS_OUT_DIR = "trained_adapters"
EVAL_RESULTS_DIR = f"{ADAPTERS_OUT_DIR}/EVAL_RESULTS"

ROUTER_DIR = "router"
ROUTER_FEATURES_DIR = f"{ROUTER_DIR}/extracted_features_9classes"
ROUTER_WEIGHTS_PATH = f"{ROUTER_DIR}/gating_network.pt"

IW_MAPPING_PATH = "data/iw_mapping.json"

# ============================================================
# CULTURAL CLUSTER TAXONOMY
# 8 Inglehart-Welzel clusters + 1 domain-agnostic generic class
# ============================================================
ADAPTER_NAMES = [
    "protestant_europe",
    "catholic_europe",
    "orthodox_europe",
    "english_speaking",
    "latin_america",
    "west_south_asia",
    "confucian",
    "african_islamic",
    "generic",
]

# Router labels retain their original order for checkpoint compatibility.
# Generic is a routing class, not a trainable LoRA adapter.
CULTURAL_ADAPTERS = [name for name in ADAPTER_NAMES if name != "generic"]
NUM_ADAPTERS = len(ADAPTER_NAMES)
NUM_CULTURAL_ADAPTERS = len(CULTURAL_ADAPTERS)
DATA_SEED = 42
VALIDATION_IMAGE_FRACTION = 0.1

# ============================================================
# QUANTIZATION CONFIGURATION (4-bit NF4)
# ============================================================
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
}

# ============================================================
# LoRA ADAPTER CONFIGURATION
# ============================================================
LORA_CONFIG = {
    "r": 16,                                # Default rank
    "lora_alpha": 32,                       # Scaling factor (alpha / r = 2)
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1,                    # Regularization for low-resource clusters
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "exclude_modules": "vision_tower",
}

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
TRAINING_ARGS = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,       # Global effective batch = 8 * 4 * world_size
    "per_device_eval_batch_size": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "num_train_epochs": 3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "eval_strategy": "steps",
    "eval_steps": 15,
    "save_strategy": "steps",
    "save_steps": 15,
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
    "greater_is_better": False,
    "save_total_limit": 1,
    "bf16": True,
    "optim": "paged_adamw_8bit",
    "dataloader_num_workers": 4,
}

# ============================================================
# DATA PIPELINE CONFIGURATION
# ============================================================
# Target sample counts per cluster per split
DATASET_TARGETS = {
    "train": {"OE": 3600, "MCQ": 450, "TF": 450},
    "val":   {"OE": 400,  "MCQ": 50,  "TF": 50},
}

# ScienceQA generic (non-cultural) partition sizes
SCIENCEQA_TARGETS = {"train": 4500, "val": 500}

# ============================================================
# ROUTER CONFIGURATION
# ============================================================
ROUTER_CONFIG = {
    "input_dim": 1024,       # CLIP ViT-L/14 [CLS] embedding dimension
    "num_classes": 9,        # 8 cultural + 1 generic
    "learning_rate": 1e-3,
    "weight_decay": 5e-3,
    "batch_size": 128,
    "max_epochs": 500,
    "patience": 15,          # Early stopping patience
    "label_smoothing": 0.1,  # Prevents overconfidence / calibration drift
}

# ============================================================
# INFERENCE CONFIGURATION
# ============================================================
INFERENCE_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,      # Greedy decoding for deterministic outputs
}

TRAINING_SEED = 42
ADAPTER_EARLY_STOPPING_PATIENCE = 2
ROUTER_CLASS_WEIGHTING = "inverse_frequency"
