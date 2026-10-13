import ast
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

    if any(v is not None and not isinstance(v, str) for v in value):
        raise ValueError('Reference answers must be strings or None')

    cleaned = list(dict.fromkeys(
        v.strip() for v in value
        if v is not None and v.strip()
    ))

    return cleaned


def cvqa_record(item, mode):
    sample_id = str(item['ID'])
    subset = item['Subset']
    if isinstance(subset, str):
        subset = ast.literal_eval(subset)
    if not isinstance(subset, (list, tuple)) or len(subset) != 2:
        raise ValueError(f'Invalid CVQA subset: {subset!r}')
    options = item['Translated Options']
    question = item['Translated Question'].strip()
    label = item.get('Label')
    valid_label = type(label) is int and 0 <= label < len(options)
    if not valid_label:
        raise ValueError(f'CVQA {sample_id}: expected a valid dataset Label between '
                         f'0 and {len(options) - 1}, got {label!r}.')
    answers = references([options[label]])
    if mode == 'cvqa-mcq':
        question += '\n' + '\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(options))
        question += '\nAnswer with only the option letter.'
    return {'id': sample_id, 'country': normalize_country(subset[1]), 'question': question,
            'reference_answers': answers, 'options': options,
            'answer_label': chr(65+label), 'image': item['image']}


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
