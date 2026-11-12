"""GeneCIS object task protocol and reproducible subset selection."""
from pathlib import Path
import json
import gzip
import random

ROOT=Path(__file__).resolve().parent
ANNOTATIONS=ROOT/'data'/'annotations'
TASKS=('focus_object','change_object')

def load_cases(task, limit, seed=42):
    if task not in TASKS: raise ValueError('Tác vụ không hợp lệ')
    with gzip.open(ANNOTATIONS/f'{task}.json.gz', 'rt', encoding='utf-8') as source:
        cases=json.load(source)
    if not 1 <= limit <= len(cases): raise ValueError(f'Số truy vấn phải từ 1 đến {len(cases)}')
    order=list(range(len(cases))); random.Random(seed).shuffle(order)
    return [(i,cases[i]) for i in order[:limit]]

def get_id(item): return int(item['val_image_id'])

def case_ids(case):
    return get_id(case['reference']),get_id(case['target']),[get_id(x) for x in case['gallery']]

def required_ids(cases):
    output=set()
    for _,case in cases:
        ref,target,gallery=case_ids(case)
        output.update([ref,target,*gallery])
    return sorted(output)

def candidates(case, seed=42):
    _,target,gallery=case_ids(case)
    choices=[target,*gallery]
    random.Random(seed).shuffle(choices)
    return choices

def image_path(image_id): return ROOT/'data'/'images'/f'{image_id:012d}.jpg'
