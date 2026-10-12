import argparse
from collections import defaultdict
from utils.runtime import read_jsonl, read_json, write_json, normalize_country


def scored_rows(path, countries=None):
    # Judge retries append an updated record. Its latest state is authoritative.
    latest = {}
    for row in read_jsonl(path):
        key = str(row['id'])
        if key in latest and latest[key].get('status') != 'error':
            raise ValueError(f'Duplicate completed sample: {key}')
        latest[key] = row
    result = {}
    for key, row in latest.items():
        if countries is not None and normalize_country(row['country']) not in countries:
            continue
        if row.get('status') == 'error':
            raise ValueError(f'Unresolved judge error: {key}')
        if 'correct' in row:
            value = row['correct']
            if type(value) is not int or value not in (0, 1):
                raise ValueError(f'Invalid exact-match score: {key}')
        else:
            rating = row.get('rating')
            if row.get('status') != 'ok' or type(rating) is not int or rating not in (1, 2):
                raise ValueError(f'Missing/invalid judge rating: {key}')
            value = int(rating == 2)
        result[key] = {**row, 'correct': value}
    if not result:
        raise ValueError('No scored examples')
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows.values():
        groups[normalize_country(row['country'])].append(row['correct'])
    by_country = {c:{'count':len(v), 'accuracy':sum(v)/len(v)} for c,v in sorted(groups.items())}
    return {'count':len(rows), 'accuracy':sum(r['correct'] for r in rows.values())/len(rows),
            'country_macro_accuracy':sum(r['accuracy'] for r in by_country.values())/len(by_country),
            'by_country':by_country}


def compare(baseline, method):
    if set(baseline) != set(method):
        raise ValueError('Paired comparison requires identical sample IDs; use the same countries filter')
    for key in baseline:
        for field in ('country', 'question', 'reference_answers', 'answer_label'):
            if baseline[key].get(field) != method[key].get(field):
                raise ValueError(f'Comparison inputs differ: {key}/{field}')
    wins=sum(method[k]['correct'] > baseline[k]['correct'] for k in baseline)
    losses=sum(method[k]['correct'] < baseline[k]['correct'] for k in baseline)
    return {'wins':wins, 'losses':losses, 'ties':len(baseline)-wins-losses,
            'gain_percentage_points':100*(wins-losses)/len(baseline)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--baseline')
    p.add_argument('--countries-file')
    p.add_argument('--output', required=True)
    args=p.parse_args()
    countries=set(map(normalize_country,read_json(args.countries_file))) if args.countries_file else None
    rows=scored_rows(args.input,countries)
    report=summarize(rows)
    if args.baseline:
        baseline=scored_rows(args.baseline,countries)
        report['baseline']=summarize(baseline)
        report['paired_comparison']=compare(baseline,rows)
    write_json(args.output,report)
    print(f"Accuracy: {100*report['accuracy']:.4f}% on {report['count']} examples")


if __name__ == '__main__':
    main()
