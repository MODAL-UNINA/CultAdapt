import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from datasets import load_dataset
from utils.config import (
    CULTURAL_ADAPTERS, DATA_SEED, DATASET_TARGETS, DIR_BASE,
    IW_MAPPING_PATH, SCIENCEQA_TARGETS, VALIDATION_IMAGE_FRACTION,
)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rng(seed, *keys):
    return random.Random(int(_digest([seed, *keys]), 16))


def canonical_image_key(path):
    """Collapse relative/absolute path aliases and symlinks when resolvable."""
    return str(Path(path).expanduser().resolve())


def _normalize_image_path(path):
    path = str(path).replace("\\", "/")
    if path.startswith("Wikidata_images_v3/"):
        path = "CultureGroundImages/" + path[len("Wikidata_images_v3/"):]
    return path


def _extract_country(path, mapping):
    parts = set(path.lower().replace("\\", "/").split("/"))
    matches = sorted(parts.intersection(mapping))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous country in image path: {path}")
    return matches[0] if matches else None


def normalize_samples(samples, mapping):
    """Normalize records and remove identical image/question/answer duplicates.

    Exported IDs are deterministic content hashes; source IDs are retained in
    source_id. They distinguish multiple questions sharing a source image ID.
    """
    result, seen, owners = [], set(), {}
    stats = Counter()
    for item in samples:
        if item.get("language") != "en":
            stats["excluded_non_english"] += 1
            continue
        image = _normalize_image_path(item["image"])
        country = _extract_country(image, mapping)
        if country is None:
            stats["excluded_unmapped_country"] += 1
            continue
        conversations = item.get("conversations", [])
        if (len(conversations) < 2 or
                any(not isinstance(c.get("value"), str) or not c["value"].strip()
                    for c in conversations[:2])):
            raise ValueError(f"Missing question/answer for source ID {item.get('id')!r}")
        task = str(item.get("question_type", "oe")).upper()
        if task not in {"OE", "MCQ", "TF"}:
            raise ValueError(f"Unsupported question_type {task!r}")
        key = canonical_image_key(image)
        if key in owners and owners[key] != country:
            raise ValueError(f"Image belongs to multiple countries: {image}")
        owners[key] = country
        label = None
        if task == "TF":
            label_aliases = {
                "true": "true", "false": "false",
                "yes": "true", "no": "false",
            }
            label = str(item.get("correct_answer", "")).strip().lower()
            if label not in label_aliases:
                match = re.match(
                    r"^\s*(true|false|yes|no)\b",
                    conversations[1]["value"],
                    re.I,
                )
                if match is None:
                    raise ValueError(
                        f"Unrecognized TF answer for source ID {item.get('id')!r}"
                    )
                label = match.group(1).lower()
            label = label_aliases[label]
        # Use the portable normalized path for the ID, canonical path for overlap.
        identity = [image, task, conversations, item.get("options"),
                    label if task == "TF" else item.get("correct_answer")]
        dedup_key = _digest([key, *identity[1:]])
        if dedup_key in seen:
            stats["duplicate_samples_removed"] += 1
            continue
        seen.add(dedup_key)
        row = {
            "id": "sample_" + _digest(identity),
            "source_id": item.get("id"), "image": image,
            "conversations": conversations, "cluster": mapping[country],
            "country": country, "task_type": task,
        }
        if task == "TF":
            row["correct_answer"] = label
        elif "correct_answer" in item:
            row["correct_answer"] = item["correct_answer"]
        if "options" in item:
            row["options"] = item["options"]
        result.append(row)
    stats["eligible_unique_samples"] = len(result)
    return result, dict(stats)


def assign_image_splits(rows, seed, validation_fraction):
    """Seeded image-level split within each country, independent of cluster IDs."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    images_by_country = defaultdict(dict)
    owners = {}
    for row in rows:
        key = canonical_image_key(row["image"])
        owner = row["country"]
        if key in owners and owners[key] != owner:
            raise ValueError(f"Conflicting country assignments for {key}")
        owners[key] = owner
        previous = images_by_country[owner].get(key, row["image"])
        images_by_country[owner][key] = min(previous, row["image"])
    assignments = {}
    for country in sorted(images_by_country):
        images = sorted(images_by_country[country], key=images_by_country[country].get)
        if len(images) < 2:
            raise ValueError(f"Country {country} needs at least two distinct images")
        _rng(seed, "image_split", country).shuffle(images)
        n_val = max(1, min(len(images) - 1, round(len(images) * validation_fraction)))
        for index, image in enumerate(images):
            assignments[image] = "val" if index < n_val else "train"
    return assignments


def _balanced_sample(candidates, target, countries, rng, context):
    """Sample without replacement with country quotas as even as availability permits."""
    if target < 0:
        raise ValueError("Targets must be nonnegative")
    pools = {country: [] for country in sorted(countries)}
    for row in candidates:
        pools[row["country"]].append(row)
    available = sum(map(len, pools.values()))
    if available < target:
        counts = {country: len(pool) for country, pool in pools.items()}
        raise ValueError(
            f"Insufficient unique samples for {context}: requested {target}, "
            f"available {available}; country counts={counts}. "
            "No oversampling or transfer between image splits is performed. "
            "Check the source data or explicitly revise the sampling targets."
        )
    for pool in pools.values():
        pool.sort(key=lambda row: row["id"])
        rng.shuffle(pool)
    selected = []
    active = [country for country in sorted(pools) if pools[country]]
    while len(selected) < target:
        rng.shuffle(active)
        for country in active:
            if len(selected) == target:
                break
            selected.append(pools[country].pop())
        active = [country for country in active if pools[country]]
    return selected


def _get_stats(rows):
    counts = Counter()
    for row in rows:
        task = row["task_type"]
        counts["TF_" + row["correct_answer"] if task == "TF" else task] += 1
    return dict(sorted(counts.items()))


def validate_partitions(partitions, targets):
    """Fail before exporting if any count, ID, or image isolation constraint fails."""
    split_images, split_ids = {}, {}
    for split, groups in partitions.items():
        images, ids = set(), set()
        for cluster, rows in groups.items():
            row_ids = [row["id"] for row in rows]
            if len(set(row_ids)) != len(row_ids) or ids.intersection(row_ids):
                raise ValueError(f"Repeated sample IDs in {split}/{cluster}")
            ids.update(row_ids)
            images.update(canonical_image_key(row["image"]) for row in rows)
            counts = _get_stats(rows)
            expected = targets[split]
            if any(counts.get(task, 0) != expected[task] for task in ("OE", "MCQ")):
                raise ValueError(f"OE/MCQ quota mismatch in {split}/{cluster}")
            if any(counts.get("TF_" + label, 0) != expected["TF"] // 2
                   for label in ("true", "false")):
                raise ValueError(f"TF quota mismatch in {split}/{cluster}")
            if len(rows) != sum(expected.values()):
                raise ValueError(f"Total quota mismatch in {split}/{cluster}")
        split_images[split], split_ids[split] = images, ids
    if split_images["train"] & split_images["val"]:
        raise ValueError("Image overlap between training and validation")
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Sample ID overlap between training and validation")


def build_cultural_partitions(samples, mapping, targets=None, seed=DATA_SEED,
                              validation_fraction=VALIDATION_IMAGE_FRACTION):
    """Build both splits in memory and validate them before any files are written."""
    targets = DATASET_TARGETS if targets is None else targets
    for split in ("train", "val"):
        if targets[split]["TF"] % 2:
            raise ValueError(f"TF target must be even in {split}")
    rows, filtering_stats = normalize_samples(samples, mapping)
    assignments = assign_image_splits(rows, seed, validation_fraction)
    pools = defaultdict(list)
    for row in rows:
        split = assignments[canonical_image_key(row["image"])]
        bucket = "TF_" + row["correct_answer"] if row["task_type"] == "TF" else row["task_type"]
        pools[(split, row["cluster"], bucket)].append(row)
    clusters = sorted(set(mapping.values()))
    partitions = {split: {cluster: [] for cluster in clusters} for split in ("train", "val")}
    for split in partitions:
        counts = targets[split]
        quotas = {"OE": counts["OE"], "MCQ": counts["MCQ"],
                  "TF_true": counts["TF"] // 2, "TF_false": counts["TF"] // 2}
        for cluster in clusters:
            countries = sorted(country for country, group in mapping.items() if group == cluster)
            selected = partitions[split][cluster]
            for bucket, target in quotas.items():
                selected.extend(_balanced_sample(
                    pools[(split, cluster, bucket)], target, countries,
                    _rng(seed, split, cluster, bucket), f"{split}/{cluster}/{bucket}",
                ))
            _rng(seed, "final_order", split, cluster).shuffle(selected)
    validate_partitions(partitions, targets)
    report = {
        "seed": seed, "candidate_validation_image_fraction": validation_fraction,
        "image_identity": "canonical file path; no pixel-level duplicate detection",
        "targets_per_cluster": targets, "filtering": filtering_stats,
        "sampling": "without replacement; balanced by country within each task/TF label",
        "train_val_image_overlap": 0, "train_val_sample_overlap": 0, "splits": {},
    }
    for split, groups in partitions.items():
        report["splits"][split] = {
            cluster: {"samples": len(selected),
                      "unique_images": len({canonical_image_key(r["image"]) for r in selected}),
                      "task_counts": _get_stats(selected),
                      "country_counts": dict(sorted(Counter(r["country"] for r in selected).items()))}
            for cluster, selected in groups.items()
        }
    return partitions, report


def partition_cultural_data(samples, mapping, output_dir=DIR_BASE, **kwargs):
    partitions, report = build_cultural_partitions(samples, mapping, **kwargs)
    root = Path(output_dir)
    manifest = []
    for split, groups in partitions.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for cluster, rows in groups.items():
            (directory / f"dataset_{cluster}.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest.extend({"split": split, **{key: row[key] for key in
                            ("id", "source_id", "image", "country", "cluster", "task_type")}}
                            for row in rows)
            print(f"{split}/{cluster}: {len(rows)} samples; {_get_stats(rows)}")
    (root / "cultural_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "cultural_sampling_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return partitions, report


def process_science_qa(output_dir=DIR_BASE):
    """Preserve the original natural-science filter and official ScienceQA splits.

    These records are for router training only, not LoRA training.
    """
    

    ds = load_dataset("derek-thomas/ScienceQA")
    root = Path(output_dir)
    image_dir = root / "images_generic"
    image_dir.mkdir(parents=True, exist_ok=True)
    for source_split, split in (("train", "train"), ("validation", "val")):
        rows = []
        for index, item in enumerate(ds[source_split]):
            if len(rows) >= SCIENCEQA_TARGETS[split]:
                break
            if item.get("image") is None or item.get("subject") != "natural science":
                continue
            path = image_dir / f"generic_{split}_{index}.jpg"
            item["image"].convert("RGB").save(path, "JPEG")
            rows.append({"id": f"scienceqa_{source_split}_{index}", "image": str(path),
                         "question": item.get("question", ""), "answer": item.get("answer", ""),
                         "cluster": "generic"})
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "dataset_generic.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{split}/generic (router only): {len(rows)}/{SCIENCEQA_TARGETS[split]} samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oe-json", default="CulturalGround-OE-Filtered-14M.json")
    parser.add_argument("--mcq-json", default="CulturalGround-MCQs-Filtered-7M.json")
    parser.add_argument("--mapping", default=IW_MAPPING_PATH)
    parser.add_argument("--output-dir", default=DIR_BASE)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    parser.add_argument("--skip-scienceqa", action="store_true")
    args = parser.parse_args()
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    if set(mapping.values()) != set(CULTURAL_ADAPTERS):
        raise ValueError("This main-experiment pipeline expects the eight IW groups")
    samples = []
    for filename in (args.oe_json, args.mcq_json):
        with open(filename, encoding="utf-8") as stream:
            samples.extend(json.load(stream))
    partition_cultural_data(samples, mapping, output_dir=args.output_dir, seed=args.seed)
    if not args.skip_scienceqa:
        process_science_qa(args.output_dir)


if __name__ == "__main__":
    main()
