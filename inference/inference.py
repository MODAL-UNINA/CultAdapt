import argparse
from utils.config import MODEL_ID, ADAPTERS_OUT_DIR, ROUTER_WEIGHTS_PATH, ADAPTER_NAMES, INFERENCE_CONFIG
from utils.runtime import (read_json, append_jsonl, read_jsonl, versions,
                           ensure_run_manifest, validate_adapter_path, normalize_country)
from utils.inference_utils import read_annotations, cvqa_record, culturalvqa_record, exact_mcq_score
from datasets import load_dataset
import ast
import torch
from transformers import set_seed
from peft import PeftModel
from utils.model_utils import (load_base_model, load_inference_model, load_processor,
                                configure_processor, GatingNetwork, extract_clip_embedding)
from utils.image_utils import composite_to_rgb
from contextlib import nullcontext

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['culturalvqa', 'cvqa-oe', 'cvqa-mcq'], required=True)
    p.add_argument('--mode', choices=['learned', 'oracle', 'backbone', 'single'], default='learned')
    p.add_argument('--output', required=True, help='Incrementally written JSONL')
    p.add_argument('--adapters-dir', default=ADAPTERS_OUT_DIR)
    p.add_argument('--router', default=ROUTER_WEIGHTS_PATH)
    p.add_argument('--single-adapter')
    p.add_argument('--mapping', default='data/iw_evaluation_mapping.json')
    p.add_argument('--countries-file', help='JSON list of evaluation countries (identical across comparisons)')
    p.add_argument('--annotations', help='CVQA annotations JSON keyed by ID')
    p.add_argument('--model-id', default=MODEL_ID)
    p.add_argument('--revision')
    p.add_argument('--dataset-revision')
    p.add_argument('--max-new-tokens', type=int, default=INFERENCE_CONFIG['max_new_tokens'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.max_new_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        p.error('Token limit and sample limit must be positive')
    if args.mode == 'single' and not args.single_adapter:
        p.error('--single-adapter is required in single mode')
    
    mapping = read_json(args.mapping) if args.mode == 'oracle' else None
    countries = set(map(normalize_country, read_json(args.countries_file))) if args.countries_file else None
    annotations = read_annotations(args.annotations)
    dataset_id = 'mair-lab/CulturalVQA' if args.benchmark == 'culturalvqa' else 'afaji/cvqa'
    dataset = load_dataset(dataset_id, split='test', revision=args.dataset_revision)
    # Validate schema and GT before GPU loading. Image objects remain loaded lazily by HF.
    records = []
    metadata = dataset.remove_columns('image')
    for index, item in enumerate(metadata):
        if countries is not None:
            
            subset = item.get('Subset')
            if isinstance(subset, str):
                subset = ast.literal_eval(subset)
            country = item['country'] if args.benchmark == 'culturalvqa' else subset[1]
            if normalize_country(country) not in countries:
                continue
        item['image'] = None
        row = culturalvqa_record(item) if args.benchmark == 'culturalvqa' else cvqa_record(item, annotations, args.benchmark)
        if countries is not None and row['country'] not in countries:
            continue
        if mapping is not None and row['country'] not in mapping:
            raise ValueError(f'No oracle assignment for {row["country"]}; '
                             'use an explicit countries file to restrict BOTH configurations')
        row['_dataset_index'] = index
        records.append(row)
        if args.limit and len(records) >= args.limit:
            break
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Empty evaluation or duplicate sample IDs')
    
    if not torch.cuda.is_available():
        raise RuntimeError('Quantized generation requires CUDA')
    set_seed(args.seed)
    device = torch.device('cuda:0')
    classes = ADAPTER_NAMES
    router = None
    if args.mode == 'learned':
        checkpoint = torch.load(args.router, map_location='cpu', weights_only=True)
        if 'state_dict' in checkpoint:
            classes = checkpoint['class_names']
            if classes != ADAPTER_NAMES:
                raise ValueError('Router class order differs from the configured checkpoint taxonomy')
            manifest_model = checkpoint.get('feature_manifest', {}).get('model_id')
            if manifest_model and manifest_model != args.model_id:
                raise ValueError('Router embeddings use a different backbone')
            state = checkpoint['state_dict']
        else:
            state = checkpoint
        router = GatingNetwork(num_classes=len(classes)).to(device)
        router.load_state_dict(state)
        router.eval()
    names = sorted(set(mapping.values())) if mapping else [n for n in classes if n != 'generic']
    if args.mode in ('learned', 'oracle'):
        model = load_inference_model(args.adapters_dir, names, {'': 0}, args.model_id, args.revision)
    else:
        model = load_base_model(args.model_id, {'': 0}, revision=args.revision)
        if args.mode == 'single':
            model = PeftModel.from_pretrained(model, validate_adapter_path(args.single_adapter), is_trainable=False)
    processor = load_processor(args.model_id, args.revision)
    configure_processor(processor, model)
    processor.tokenizer.padding_side = 'left'
    model.eval()
    model.config.use_cache = True
    manifest_args = {k:v for k,v in vars(args).items() if k != 'resume'}
    manifest = {'arguments': manifest_args, 'sample_ids': [r['id'] for r in records],
                'versions': versions()}
    ensure_run_manifest(args.output, manifest, args.resume)
    previous = read_jsonl(args.output)
    done = {row['id'] for row in previous}
    if len(done) != len(previous) or not done.issubset({r['id'] for r in records}):
        raise ValueError('Invalid existing output IDs')
    for row in records:
        if row['id'] in done:
            continue
        image = composite_to_rgb(dataset[row['_dataset_index']]['image'])
        messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)
        chosen, confidence = None, None
        if router is not None:
            with torch.inference_mode():
                embedding = extract_clip_embedding(model, inputs['pixel_values'])
                probs = router(embedding.unsqueeze(0)).softmax(-1)[0]
                index = probs.argmax().item()
                chosen, confidence = classes[index], probs[index].item()
        elif mapping is not None:
            chosen = mapping[row['country']]
        if chosen and chosen != 'generic':
            model.set_adapter(chosen)
            model.eval()
        
        ctx = model.disable_adapter() if chosen == 'generic' else nullcontext()
        with ctx, torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, num_beams=1, use_cache=True)
        answer = processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        record = {k:v for k,v in row.items() if k not in ('image', '_dataset_index')}
        record.update(mode=args.mode, predicted_adapter=chosen, router_confidence=confidence,
                      model_response=answer)
        if args.benchmark == 'cvqa-mcq':
            record['correct'] = exact_mcq_score(answer, row['answer_label'])
        append_jsonl(args.output, record)
    print(f'Saved {len(records)} responses to {args.output}')


if __name__ == '__main__':
    main()
