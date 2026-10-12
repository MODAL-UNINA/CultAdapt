from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.config import MODEL_ID, LORA_CONFIG, BNB_CONFIG, CULTURAL_ADAPTERS, ROUTER_CONFIG
from utils.runtime import resolve_adapter_path


class GatingNetwork(nn.Module):
    def __init__(self, input_dim=ROUTER_CONFIG['input_dim'], num_classes=ROUTER_CONFIG['num_classes']):
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def build_bnb_config():
    config = dict(BNB_CONFIG)
    config['bnb_4bit_compute_dtype'] = getattr(torch, config['bnb_4bit_compute_dtype'])
    return BitsAndBytesConfig(**config)


def build_lora_config(rank=None, alpha=None):
    config = dict(LORA_CONFIG)
    if rank is not None:
        config['r'] = rank
    if alpha is not None:
        config['lora_alpha'] = alpha
    return LoraConfig(**config)

def load_processor(model_id=MODEL_ID, revision=None):
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, use_fast=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def configure_processor(processor, model):
    config = model.config
    processor.patch_size = config.vision_config.patch_size
    processor.vision_feature_select_strategy = config.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1  # CLIP CLS token


def load_base_model(model_id=MODEL_ID, device_map=None, use_quantization=True, revision=None):
    if device_map is None:
        device_map = {'': 0}
    if use_quantization and not torch.cuda.is_available():
        raise RuntimeError('4-bit execution requires the configured CUDA/bitsandbytes environment')
    kwargs = dict(low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                  device_map=device_map, revision=revision)
    if use_quantization:
        kwargs['quantization_config'] = build_bnb_config()
    return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def build_multi_adapter_model(base_model, adapter_names=CULTURAL_ADAPTERS, rank=None, alpha=None):
    if not adapter_names or len(set(adapter_names)) != len(adapter_names):
        raise ValueError('Expected distinct cultural adapters')
    if 'generic' in adapter_names:
        raise ValueError('generic is a router-only class')
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    cfg = build_lora_config(rank, alpha)
    model = get_peft_model(base_model, cfg, adapter_name=adapter_names[0])
    for name in adapter_names[1:]:
        model.add_adapter(name, cfg)
    activate_adapter(model, adapter_names[0])
    return model


def activate_adapter(model, cluster_name):
    if cluster_name == 'generic':
        raise ValueError('generic cannot be trained')
    for name, param in model.named_parameters():
        if 'vision_tower' in name or 'multi_modal_projector' in name:
            param.requires_grad = False
    model.set_adapter(cluster_name)
    count = 0
    for name, param in model.named_parameters():
        active = ('lora_' in name and f'.{cluster_name}.' in name and 'language_model' in name)
        param.requires_grad = active
        if active:
            count += param.numel()
    if not count:
        raise RuntimeError('No trainable language-model LoRA parameters were found')


def load_inference_model(lora_base_path, adapter_names=CULTURAL_ADAPTERS,
                         device_map=None, model_id=MODEL_ID, revision=None):
    names = [name for name in adapter_names if name != 'generic']
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected distinct adapter names')
    paths = {name: resolve_adapter_path(lora_base_path, name) for name in names}
    base = load_base_model(model_id, device_map, revision=revision)
    model = PeftModel.from_pretrained(base, paths[names[0]], adapter_name=names[0], is_trainable=False)
    for name in names[1:]:
        model.load_adapter(paths[name], adapter_name=name, is_trainable=False)
    model.eval()
    return model


def get_vision_tower(model):
    current = model.get_base_model() if isinstance(model, PeftModel) else model
    if hasattr(current, 'vision_tower'):
        return current.vision_tower
    if hasattr(current, 'model') and hasattr(current.model, 'vision_tower'):
        return current.model.vision_tower
    raise AttributeError('Cannot locate the LLaVA vision tower')


def extract_clip_embedding(model, pixel_values):
    if pixel_values.ndim == 5:
        if pixel_values.shape[0] != 1:
            raise ValueError('Router embeddings require one image per call')
        pixel_values = pixel_values[0]
    if pixel_values.ndim != 4:
        raise ValueError('Expected (1, crops, C, H, W) or (crops, C, H, W)')
    tower = get_vision_tower(model)
    param = next(tower.parameters())
    with torch.no_grad():
        output = tower(pixel_values.to(device=param.device, dtype=param.dtype), output_hidden_states=True)
        embedding = output.hidden_states[-1][:, 0, :].float().mean(dim=0)
    return F.normalize(embedding, dim=-1)
