import hashlib
import importlib.metadata
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()


def versions():
    result = {}
    for name in ('torch', 'transformers', 'peft', 'datasets', 'accelerate', 'bitsandbytes', 'ollama'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def resolve_adapter_path(root, name):
    """Accept canonical exports and the previous nested-name export layout."""
    root = Path(root)
    candidates = [root / name, root / name / name]
    for path in candidates:
        if (path / 'adapter_config.json').is_file() and any(
            (path / filename).is_file() for filename in ('adapter_model.safetensors', 'adapter_model.bin')
        ):
            return str(path)
    raise FileNotFoundError(f'No complete checkpoint for {name!r} under {root}')


def validate_adapter_path(path):
    path = Path(path)
    if not (path / 'adapter_config.json').is_file() or not any(
        (path / filename).is_file() for filename in ('adapter_model.safetensors', 'adapter_model.bin')
    ):
        raise FileNotFoundError(f'Incomplete adapter checkpoint: {path}')
    return str(path)


def ensure_run_manifest(output, manifest, resume=False):
    path = Path(str(output) + '.run.json')
    if Path(output).exists():
        if not resume:
            raise FileExistsError(f'{output} exists; choose a new output or use --resume')
        if not path.exists() or read_json(path) != manifest:
            raise ValueError('Resume manifest differs from the current configuration/input')
    else:
        write_json(path, manifest)


def normalize_country(value):
    return str(value).strip().lower().replace(' ', '_').replace('-', '_')
