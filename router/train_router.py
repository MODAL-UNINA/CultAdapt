import argparse
from pathlib import Path
from utils.config import ADAPTER_NAMES, ROUTER_CONFIG, ROUTER_FEATURES_DIR, ROUTER_WEIGHTS_PATH, TRAINING_SEED
from utils.runtime import read_json, write_json, versions
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed
from utils.model_utils import GatingNetwork

def main():
    set_seed(TRAINING_SEED)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir', default=ROUTER_FEATURES_DIR)
    p.add_argument('--output', default=ROUTER_WEIGHTS_PATH)
    p.add_argument('--label-smoothing', type=float, default=ROUTER_CONFIG['label_smoothing'])
    p.add_argument('--class-weighting', choices=['inverse_frequency', 'none'], default='inverse_frequency')
    args = p.parse_args()
    if not 0 <= args.label_smoothing < 1:
        p.error('label smoothing must be in [0,1)')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = Path(args.features_dir)
    manifest = read_json(root / 'manifest.json')
    if manifest['class_names'] != ADAPTER_NAMES:
        raise ValueError('Feature class order differs from router class order')
    splits, paths, counts = {}, {}, []
    for split in ('train', 'val'):
        features, labels = [], []
        paths[split] = set()
        for label, name in enumerate(ADAPTER_NAMES):
            data = torch.load(root / f'{split}_{name}.pt', map_location='cpu', weights_only=True)
            if not data:
                raise ValueError(f'Missing samples for {split}/{name}')
            if split == 'train':
                counts.append(len(data))
            for row in data:
                if row['label'] != label or row['image'].shape != (ROUTER_CONFIG['input_dim'],):
                    raise ValueError(f'Feature/label mismatch in {split}/{name}')
                if not torch.isfinite(row['image']).all():
                    raise ValueError('Nonfinite router feature')
                features.append(row['image'].float())
                labels.append(label)
                paths[split].add(row['image_path'])
        splits[split] = TensorDataset(torch.stack(features), torch.tensor(labels, dtype=torch.long))
    if paths['train'] & paths['val']:
        raise ValueError('Router splits share images')
    loaders = {s: DataLoader(ds, batch_size=ROUTER_CONFIG['batch_size'], shuffle=(s == 'train'))
               for s, ds in splits.items()}
    model = GatingNetwork().to(device)
    weights = None
    if args.class_weighting == 'inverse_frequency':
        weights = 1 / torch.tensor(counts, dtype=torch.float32, device=device)
        weights /= weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=ROUTER_CONFIG['learning_rate'],
                                 weight_decay=ROUTER_CONFIG['weight_decay'])
    best, patience, history = float('inf'), 0, []
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a fresh router output path: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(ROUTER_CONFIG['max_epochs']):
        record = {'epoch': epoch + 1}
        for split, loader in loaders.items():
            training = split == 'train'
            model.train(training)
            total_loss, denominator, correct, n = 0., 0., 0, 0
            with torch.set_grad_enabled(training):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss_sum = criterion(logits, y).sum()
                    denom = weights[y].sum() if weights is not None else y.new_tensor(len(y), dtype=torch.float32)
                    loss = loss_sum / denom
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total_loss += loss_sum.detach().item()
                    denominator += denom.item()
                    correct += (logits.argmax(-1) == y).sum().item()
                    n += len(y)
            record[split + '_loss'] = total_loss / denominator
            record[split + '_accuracy'] = correct / n
        history.append(record)
        print(record)
        if record['val_loss'] < best:
            best, patience = record['val_loss'], 0
            torch.save({'state_dict': model.state_dict(), 'class_names': ADAPTER_NAMES,
                        'input_dim': ROUTER_CONFIG['input_dim'], 'feature_manifest': manifest,
                        'config': {**ROUTER_CONFIG, **vars(args)}, 'best_epoch': epoch + 1,
                        'best_val_loss': best}, output)
        else:
            patience += 1
            if patience >= ROUTER_CONFIG['patience']:
                break
    write_json(str(output) + '.history.json', {'history': history, 'versions': versions()})


if __name__ == '__main__':
    main()
