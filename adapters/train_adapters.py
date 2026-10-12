import argparse
import gc
import json
import os
from pathlib import Path
import torch
from PIL import Image
from utils.config import (MODEL_ID, DIR_TRAIN, DIR_VAL, TRAINING_SEED,
                          ADAPTERS_OUT_DIR, TRAINING_ARGS,
                          ADAPTER_EARLY_STOPPING_PATIENCE, LORA_CONFIG)
from utils.runtime import write_json
import torch.distributed as dist
from datasets import Dataset
from transformers import set_seed, AutoProcessor, Trainer, TrainingArguments, EarlyStoppingCallback
from utils.model_utils import load_base_model, build_multi_adapter_model, activate_adapter

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class VQACollator:
    def __init__(self, processor, is_main_process=True):
        self.processor = processor
        self.is_main_process = is_main_process

    def __call__(self, batch):
        processor = self.processor
        is_main_process = self.is_main_process
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        image_sizes_list = []
        for sample in batch:
            try:
                with Image.open(sample['image']) as _img:
                    if _img.mode in ('RGB', 'L', 'CMYK'):
                        image = _img.convert('RGB')
                    elif _img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', _img.size, (255, 255, 255))
                        source_img = _img.convert('RGBA') if _img.mode == 'P' else _img
                        background.paste(source_img, mask=source_img.split()[-1])
                        image = background
                    else:
                        image = _img.convert('RGB')
            except Exception as e:
                if is_main_process:
                    print(f"⚠️ Corrupted image or complex transparency in {sample['image']}: {e}. Falling back to a white image.")
                image = Image.new('RGB', (336, 336), color='white')
            question = sample['conversations'][0]['value'].replace('<image>\n', '')
            answer = sample['conversations'][1]['value'] + processor.tokenizer.eos_token
            prompt_messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
            full_messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}]
            prompt_text = processor.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
            full_text = processor.apply_chat_template(full_messages, tokenize=False)
            prompt_inputs = processor(text=prompt_text, images=image, return_tensors='pt')
            full_inputs = processor(text=full_text, images=image, return_tensors='pt')
            prompt_len = prompt_inputs['input_ids'].shape[1]
            labels = full_inputs['input_ids'].clone()
            labels[:, :prompt_len] = -100
            input_ids_list.append(full_inputs['input_ids'][0])
            labels_list.append(labels[0])
            pixel_values_list.append(full_inputs['pixel_values'][0])
            image_sizes_list.append(full_inputs['image_sizes'][0])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(processor.tokenizer.pad_token_id).long()
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        max_patches = max((p.shape[0] for p in pixel_values_list))
        padded_pixel_values = []
        for p in pixel_values_list:
            num_patches, c, h, w = p.shape
            if num_patches < max_patches:
                padding = torch.zeros((max_patches - num_patches, c, h, w), dtype=p.dtype)
                p = torch.cat([p, padding], dim=0)
            padded_pixel_values.append(p)
        pixel_values_tensor = torch.stack(padded_pixel_values)
        image_sizes = torch.stack(image_sizes_list)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'pixel_values': pixel_values_tensor, 'image_sizes': image_sizes}


def discover_adapters(train_dir):
    # Preserve sorted file-based ordering; generic remains router-only.
    return sorted(path.name[len('dataset_'):-len('.json')]
                  for path in Path(train_dir).glob('dataset_*.json')
                  if path.name != 'dataset_generic.json')

def main():
    set_seed(TRAINING_SEED)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', default=DIR_TRAIN)
    parser.add_argument('--val-dir', default=DIR_VAL)
    parser.add_argument('--output-dir', default=ADAPTERS_OUT_DIR)
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--revision')
    parser.add_argument('--adapters', nargs='+', help='Optional subset of groups, trained in sorted order')
    parser.add_argument('--rank', type=int, default=LORA_CONFIG['r'])
    parser.add_argument('--alpha', type=int, default=LORA_CONFIG['lora_alpha'])
    parser.add_argument('--early-stopping-patience', type=int, default=ADAPTER_EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()
    if args.rank <= 0 or args.alpha <= 0 or args.early_stopping_patience < 1:
        parser.error('Rank, alpha and patience must be positive')
    names = sorted(args.adapters) if args.adapters else discover_adapters(args.train_dir)
    if not names or 'generic' in names or len(names) != len(set(names)):
        parser.error('Expected distinct non-generic adapter groups')
    for name in names:
        for directory in (args.train_dir, args.val_dir):
            path = Path(directory) / f'dataset_{name}.json'
            if not path.is_file():
                raise FileNotFoundError(path)
        output = Path(args.output_dir) / name
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Choose a fresh output directory: {output}')

    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    initialized_here = world_size > 1 and not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend='nccl')
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision, use_fast=True)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        base = load_base_model(args.model_id, {'': local_rank}, revision=args.revision)
        model = build_multi_adapter_model(base, names, args.rank, args.alpha)
        config = dict(TRAINING_ARGS)
        config.update(logging_strategy='steps', logging_steps=1, gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={'use_reentrant': False},
                      remove_unused_columns=False, report_to='none', label_names=['labels'],
                      ddp_backend='nccl', ddp_find_unused_parameters=False)
        if is_main_process:
            lora = dict(LORA_CONFIG, r=args.rank, lora_alpha=args.alpha)
            write_json(Path(args.output_dir)/'config.json',
                       {'model_id':args.model_id, 'lora_config':lora, 'training_args':config,
                        'early_stopping_patience':args.early_stopping_patience,
                        'adapter_order':names})
        if dist.is_initialized():
            dist.barrier()
        for name in names:
            output = Path(args.output_dir)/name
            if is_main_process:
                output.mkdir(parents=True, exist_ok=True)
            if dist.is_initialized():
                dist.barrier()
            train = Dataset.from_json(str(Path(args.train_dir)/f'dataset_{name}.json'))
            val = Dataset.from_json(str(Path(args.val_dir)/f'dataset_{name}.json'))
            activate_adapter(model, name)
            training_args = TrainingArguments(output_dir=str(output), **config)
            model.config.use_cache = False
            trainer = Trainer(model=model, args=training_args,
                              train_dataset=train, eval_dataset=val,
                              data_collator=VQACollator(processor, is_main_process),
                              callbacks=[EarlyStoppingCallback(
                                  early_stopping_patience=args.early_stopping_patience)])
            trainer.train()
            if dist.is_initialized():
                dist.barrier()
            if is_main_process:
                model.save_pretrained(str(output), selected_adapters=[name])
                write_json(output/'trainer_history.json', trainer.state.log_history)
            if dist.is_initialized():
                dist.barrier()
    finally:
        if initialized_here:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
