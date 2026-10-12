import ast
import json
from pathlib import Path
from utils.runtime import normalize_country

SYSTEM_PROMPT = """You are an expert cultural anthropologist.
Your task is to evaluate a candidate answer for a visual question answering task by comparing it against an expert reference answer.

### Rating Scale:
- **Rating 2 (Correct):** The candidate answer matches the specific concepts or cultural facts of the reference answer.
- **Rating 1 (Incorrect/Inadequate):** The answer is too generic with respect to the reference answer or contradicts the specific concepts or cultural facts of the reference answer.

Follow these strict rules:

1. **Entity Match (Rating 2):** If the candidate identifies the same core entity as the reference, it is Correct.
2. **Specificity (Rating 1):** The answer is incorrect if it is more generic than the reference.
3. **Core Concept:** Rating 2 requires the specific name or concept from the reference to be present.
"""
USER_TEMPLATE = """### Data to Evaluate:
- **Question:** {question}
- **Reference answer:** {gt}
- **Candidate answer:** {ai_response}

Return the evaluation in JSON format with 'reasoning' and 'rating' keys."""


def references(value):
    if isinstance(value, str):
        if value.lstrip().startswith('['):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        if isinstance(value, str):
            value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('Missing reference answers')
    if any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError('References must be nonempty strings')
    return list(dict.fromkeys(v.strip() for v in value))


def read_annotations(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or any(not isinstance(v, dict) for v in data.values()):
        raise ValueError('Annotations must map sample ID to {label: 0..3, answers: [...]}')
    return data


def cvqa_record(item, annotations, mode):
    sample_id = str(item['ID'])
    subset = item['Subset']
    if isinstance(subset, str):
        subset = ast.literal_eval(subset)
    if not isinstance(subset, (list, tuple)) or len(subset) != 2:
        raise ValueError(f'Invalid CVQA subset: {subset!r}')
    options = item['Translated Options']
    question = item['Translated Question'].strip()
    ann = annotations.get(sample_id, {})
    label = ann.get('label', item.get('Label'))
    valid_label = type(label) is int and 0 <= label < len(options)
    gt = ann.get('answers')
    if gt is None and valid_label:
        gt = [options[label]]
    if gt is None or (mode == 'cvqa-mcq' and not valid_label):
        raise ValueError(f'CVQA {sample_id}: missing valid ground truth; supply --annotations. '
                         'Public Label=-1 is not an answer index.')
    answers = references(gt)
    if mode == 'cvqa-mcq':
        question += '\n' + '\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(options))
        question += '\nAnswer with only the option letter.'
    return {'id': sample_id, 'country': normalize_country(subset[1]), 'question': question,
            'reference_answers': answers, 'options': options,
            'answer_label': chr(65+label) if valid_label else None, 'image': item['image']}


def culturalvqa_record(item):
    return {'id': str(item['u_id']), 'country': normalize_country(item['country']),
            'question': item['question'], 'reference_answers': references(item['all_answers']),
            'answer_label': None, 'image': item['image']}


def exact_mcq_score(response, answer_label):
    if answer_label not in ('A', 'B', 'C', 'D'):
        raise ValueError('Missing MCQ reference label')
    return int(response.strip() == answer_label)

def validate_judgment(value):
    if not isinstance(value, dict) or type(value.get('rating')) is not int or value['rating'] not in (1, 2):
        raise ValueError('Judge rating must be integer 1 or 2')
    if not isinstance(value.get('reasoning'), str):
        raise ValueError('Judge reasoning must be a string')
    return {"rating": value["rating"], "reasoning": value["reasoning"]}


def judge_references(question, answers, response, invoke, mode):
    if mode == 'joint':
        return validate_judgment(invoke(question, str(answers), response))
    if mode != 'best-of':
        raise ValueError('Reference mode must be joint or best-of')
    results = [validate_judgment(invoke(question, answer, response)) for answer in answers]
    if not results:
        raise ValueError('No references')
    best = max(results, key=lambda r:r['rating'])
    return {**best, 'per_reference': results}
