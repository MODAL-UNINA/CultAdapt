import argparse
from pathlib import Path
from utils.config import ADAPTER_NAMES, DIR_BASE, MODEL_ID, ROUTER_FEATURES_DIR
from utils.runtime import read_json, write_json, versions
import torch
from utils.image_utils import safe_open_image
from utils.model_utils import load_base_model, load_processor, configure_processor, extract_clip_embedding
    

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=DIR_BASE)
    p.add_argument('--output-dir', default=ROUTER_FEATURES_DIR)
    p.add_argument('--model-id', default=MODEL_ID)
    p.add_argument('--revision')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('The supplied embedding pipeline requires CUDA/bfloat16')
    # Validate all files and image isolation before model initialization.
    source, images = {}, {'train': set(), 'val': set()}
    for split in images:
        for name in ADAPTER_NAMES:
            path = Path(args.data_dir) / split / f'dataset_{name}.json'
            rows = read_json(path)
            if not rows:
                raise ValueError(f'Empty source: {path}')
            source[(split, name)] = (path, rows)
            images[split].update(str(Path(row['image']).resolve()) for row in rows)
    if images['train'] & images['val']:
        raise ValueError('Router training and validation share image paths')
    model = load_base_model(args.model_id, {'': 0}, use_quantization=False, revision=args.revision)
    model.eval()
    processor = load_processor(args.model_id, args.revision)
    configure_processor(processor, model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (split, name), (path, rows) in source.items():
        extracted, cache = [], {}
        for index, row in enumerate(rows):
            key = str(Path(row['image']).resolve())
            if key not in cache:
                image = safe_open_image(row['image'])
                # Direct image processing avoids mismatched text/image tokens.
                inputs = processor.image_processor(images=image, return_tensors='pt')
                cache[key] = extract_clip_embedding(model, inputs['pixel_values']).cpu()

            extracted.append({'image': cache[key], 'label': ADAPTER_NAMES.index(name),
                              'sample_id': row.get('id', f'{name}_{index}'), 'image_path': key})
        torch.save(extracted, out / f'{split}_{name}.pt')
        print(f'{split}/{name}: {len(extracted)} records, {len(cache)} unique images')
    write_json(out / 'manifest.json', {'model_id': args.model_id, 'revision': args.revision,
               'class_names': ADAPTER_NAMES, 'feature': 'last_hidden_state CLS; mean crops; float32 L2',
               'versions': versions()})


if __name__ == '__main__':
    main()
