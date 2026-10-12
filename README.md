# CultAdapt: Culturally structured low-rank Adaptation for visual question answering

## Abstract
Vision-language models (VLMs) have demonstrated strong capabilities across a broad range of multimodal tasks, but their performance can degrade when visual inputs contain culture-specific objects, practices, symbols, or traditions. In such settings, models may produce overly generic, culturally misattributed, or insufficiently grounded responses. We introduce CultAdapt, a modular parameter-efficient framework for culture-aware adaptation of VLMs. CultAdapt partitions culturally grounded multimodal training data into macro-cultural clusters using the Inglehart–Welzel cultural map and trains an independent LoRA expert for each cluster while keeping the pretrained backbone frozen. At inference time, a lightweight router operating on frozen visual representations selects a cultural adapter, while a dedicated generic class allows the system to fall back to the unadapted model for non-cultural inputs. This design separates cultural specialization from general multimodal processing and adds only limited inference overhead, since at most one adapter is activated for each input. Experiments on culturally grounded visual question answering benchmarks show that CultAdapt improves answer accuracy over the corresponding unadapted backbones, a single LoRA adapter trained on the same pooled training data, and the evaluated general-purpose hallucination mitigation methods. These results highlight the potential of structured, modular adaptation for improving culturally specific visual question answering.

![Overview of the CultAdapt framework](framework.png)

## Installation

Create and activate the Conda environment:

```bash
conda env create -n cultadapt_env -f environment.yml
conda activate cultadapt_env
```
## Data preparation

Download the CulturalGround annotation files and images, ensuring that image paths in the annotations resolve correctly from the repository root. Then run:

```bash
python -m data.prepare_dataset \
  --oe-json CulturalGround-OE-Filtered-14M.json \
  --mcq-json CulturalGround-MCQs-Filtered-7M.json \
  --output-dir data_splits
```

The script prepares image-disjoint training and validation partitions for the eight cultural groups. It also downloads ScienceQA and selects image-containing examples labeled `natural science` for the router's `generic` class. This class is not used for LoRA training.

## Training

Train the cultural adapters:

```bash
python -m adapters.train_adapters \
  --train-dir data_splits/train \
  --val-dir data_splits/val \
  --output-dir trained_adapters
```

For two-GPU training, replace `python -m` with `torchrun --standalone --nproc_per_node=2 --module` in the command above.

Extract visual features and train the router:

```bash
python -m router.extract_embeddings \
  --data-dir data_splits \
  --output-dir router/extracted_features_9classes

python -m router.train_router \
  --features-dir router/extracted_features_9classes \
  --output router/gating_network.pt
```

Feature extraction produces the embeddings and metadata required by router training. Use fresh output directories for adapters and a fresh router checkpoint path.

## Inference

Run CultAdapt on CulturalVQA:

```bash
python -m inference.inference \
  --benchmark culturalvqa --mode learned \
  --adapters-dir trained_adapters \
  --router router/gating_network.pt \
  --output results/culturalvqa.jsonl
```

Supported benchmarks are `culturalvqa`, `cvqa-oe`, and `cvqa-mcq`. Use `--mode backbone` for the unadapted baseline.


For open-ended evaluation, start an Ollama service with the judge model specified in `utils/config.py`, then run:

```bash
python -m inference.eval_judge \
  --input results/culturalvqa.jsonl \
  --output results/culturalvqa_judged.jsonl

python -m inference.metrics \
  --input results/culturalvqa_judged.jsonl \
  --output results/culturalvqa_metrics.json
```

CVQA-MCQ uses exact option-letter matching: pass its prediction file directly to `inference.metrics`, without running the judge.
