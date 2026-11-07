import json
import re
import string
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
DATA = Path('squad_validation.json')
OUT = Path('dataset_validation_report.json')

def normalize(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())

def exact_match(prediction, reference):
    return int(normalize(prediction) == normalize(reference))

def token_f1(prediction, reference):
    a, b = normalize(prediction).split(), normalize(reference).split()
    if not a or not b:
        return float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)

def main():
    if not DATA.exists():
        print('Downloading SQuAD validation dataset...')
        urlretrieve(URL, DATA)
    raw = json.loads(DATA.read_text(encoding='utf-8'))
    rows = []
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                references = [a['text'] for a in qa['answers']]
                rows.append({'id': qa['id'], 'question': qa['question'],
                             'context': paragraph['context'], 'references': references})
    assert rows and all(row['references'] for row in rows)
    # A reference-as-prediction sanity check, NOT a Gemini evaluation.
    sample = rows[:100]
    em = sum(max(exact_match(r['references'][0], x) for x in r['references']) for r in sample) / len(sample)
    f1 = sum(max(token_f1(r['references'][0], x) for x in r['references']) for r in sample) / len(sample)
    report = {'dataset': 'SQuAD v1.1 validation', 'source': URL,
              'total_questions': len(rows), 'sample_checked': len(sample),
              'reference_sanity_exact_match': em, 'reference_sanity_f1': f1,
              'gemini_evaluated': False,
              'note': 'Reference sanity check only; no model responses were generated.'}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
