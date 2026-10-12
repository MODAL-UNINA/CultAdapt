import argparse
import json
from utils.config import JUDGE_MODEL
from utils.runtime import (read_jsonl, append_jsonl, ensure_run_manifest, versions)
from utils.inference_utils import SYSTEM_PROMPT, USER_TEMPLATE, validate_judgment, judge_references, references
from ollama import Client

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference-mode', choices=['joint','best-of'],
                   default='best-of', help='Joint: all references must be correct; best-of: any reference is sufficient')
    p.add_argument('--model', default=JUDGE_MODEL)
    p.add_argument('--host', default='http://localhost:11434')
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.retries < 1:
        p.error('--retries must be positive')
    rows = read_jsonl(args.input)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Input is empty or contains duplicate IDs')
    if any('correct' in row for row in rows):
        raise ValueError('MCQ uses exact matching; do not run the OE judge on MCQ outputs')
    for row in rows:
        references(row['reference_answers'])
        if not isinstance(row['model_response'], str):
            raise ValueError('Missing model response')
    manifest = {'arguments':{k:v for k,v in vars(args).items() if k!='resume'},
                'system_prompt':SYSTEM_PROMPT,
                'user_template':USER_TEMPLATE, 'versions':versions()}
    ensure_run_manifest(args.output, manifest, args.resume)
    # Latest record per ID: resume can replace an error with a successful judgment.
    previous = {row['id']:row for row in read_jsonl(args.output)}
    if not set(previous).issubset({r['id'] for r in rows}):
        raise ValueError('Output contains IDs absent from input')
    
    client = Client(host=args.host, timeout=300)
    def invoke(question, gt, response):
        last_error = None
        for _ in range(args.retries):
            try:
                result = client.chat(model=args.model, messages=[
                    {'role':'system','content':SYSTEM_PROMPT},
                    {'role':'user','content':USER_TEMPLATE.format(question=question,gt=gt,ai_response=response)}],
                    format='json', options={'temperature':0,'seed':args.seed})
                return validate_judgment(json.loads(result['message']['content']))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f'Judge failed after {args.retries} attempts: {last_error}')
    for row in rows:
        if previous.get(row['id'],{}).get('status') == 'ok':
            continue
        try:
            judgment = judge_references(row['question'], references(row['reference_answers']),
                                        row['model_response'], invoke, args.reference_mode)
            record = {**row, **judgment, 'status':'ok'}
        except Exception as exc:
            record = {**row, 'status':'error', 'rating':None, 'error':str(exc)}
        append_jsonl(args.output, record)
        previous[row['id']] = record
    failures = sum(row.get('status')!='ok' for row in previous.values())
    if failures:
        raise SystemExit(f'{failures} judgments unresolved; rerun with --resume. No final accuracy computed.')
    print(f'Accuracy: {100*sum(r["rating"]==2 for r in previous.values())/len(previous):.4f}%')


if __name__ == '__main__':
    main()
