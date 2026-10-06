"""Validate frozen Golden labels and verbatim official manual references."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_eval_common import load_golden

if __name__ == '__main__':
    dataset, records, fingerprint = load_golden()
    positive = sum(c['answerable'] for c in dataset['cases'])
    print(f'cases={len(dataset["cases"])} answerable={positive} unanswerable={len(dataset["cases"])-positive}')
    print(f'golden_sha256={fingerprint}')
    print('GOLDEN_VALIDATION: PASS')
