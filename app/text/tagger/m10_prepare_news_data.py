import sys; sys.path.append("./src")
import os
import json
import jsonlines
from tqdm import tqdm
import hashlib
import random
from collections import defaultdict

from quick_utils.common.mp_utils import execute_parallel
from utils.text.normalizer.rule.normalizer_with_diffs import RuleBasedTextNormalizerWithDiffs
from utils.text.tagger.augment.base import Pattern, InvalidTagError


def sample_to_dict(sample: dict):
    return dict(
        content=sample["content"],
        patterns=[p.to_dict() for p in sample["patterns"]]
    )
    
    
def hash_sample(sample: dict):
    sample = sample_to_dict(sample)
    return hashlib.sha256(json.dumps(sample).encode()).hexdigest()


def load_tag_mapping():
    tag_mapping = {}
    with open("data/train/tag_mapping.md") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            old, _, new = line.split("=")
            tag_mapping[old.strip()] = new.strip()
    return tag_mapping


def worker(n: RuleBasedTextNormalizerWithDiffs, text: str, tag_mapping: dict=None):
    try:
        _, diffs = n.normalize(text)
    except Exception as e:
        # Add more detailed error info for debugging
        import traceback
        return dict(
            original=text,
            status="error",
            error=f"{type(e).__name__}: {str(e)}",
            traceback=traceback.format_exc(),
        )
    if tag_mapping is not None:
        assert isinstance(tag_mapping, dict)
        diffs = [d for d in diffs if d.tag in tag_mapping]
    if len(diffs) == 0:
        return dict(
            original=text,
            status="no_diff",
        )
    try:
        # Check if diffs have a to_dict method before calling it
        diff_dicts = []
        for d in diffs:
            if hasattr(d, 'to_dict'):
                diff_dicts.append(d.to_dict())
            else:
                # Fallback for Difference objects without to_dict method
                diff_dicts.append({
                    'start_index': d.start_index,
                    'end_index': d.end_index,
                    'original': d.original,
                    'modified': d.modified,
                    'tag': d.tag,
                    'root_start_index': d.root_start_index,
                    'root_end_index': d.root_end_index,
                    'root_original': d.root_original,
                })
        return dict(
            original=text,
            status="diff",
            diffs=diff_dicts,
        )
    except Exception as e:
        return dict(
            original=text,
            status="error",
            error=f"Error converting diffs: {type(e).__name__}: {str(e)}",
        )


def eda(diff_path: str):
    # with open("data/corpus-full.txt") as f_in:
    #     lines = []
    #     for l in tqdm(f_in):
    #         lines.append(l)
    #     print(len(lines), len(set(lines)))

    texts = set()
    with jsonlines.open(diff_path) as f:
        for item in tqdm(f):
            text = item["original"]
            texts.add(text)
    print(len(texts))


def scan(diff_path: str, B: int, S: int, E: int):
    n = RuleBasedTextNormalizerWithDiffs()
    n_diff = 0
    n_error = 0

    parts = os.path.splitext(diff_path)
    error_path = parts[0] + "_error" + parts[1]

    tag_mapping = load_tag_mapping()
    lines = []
    with (
        open("data/corpus-full.txt") as f_in,
        jsonlines.open(diff_path, "w") as f_out,
        jsonlines.open(error_path, "w") as f_err,
    ):
        pbar = tqdm(total=E)
        
        for index, line in enumerate(f_in):
            if index < S:
                continue
            
            if (index >= E or len(lines) >= B) and len(lines) > 0:
                for result in execute_parallel(worker, lines, max_workers=16, disable_tqdm=True, batch_size=B, use_process=True):
                    if result["status"] == "diff":
                        n_diff += 1
                        f_out.write(result)
                    elif result["status"] == "error":
                        n_error += 1
                        f_err.write(result)
                    pbar.set_postfix(diff=n_diff, error=n_error)
                    pbar.update()
                lines = []

            lines.append(dict(n=n, text=line.strip(), tag_mapping=tag_mapping))
                
            if index >= E:
                break
                

def show():
    with (
        jsonlines.open("data/difference.jsonl") as f_in,
        open("data/difference.md", "w") as f_out
    ):
        for item in f_in:
            text = item["original"]
            diffs = sorted(item["diffs"], key=lambda x: (x["root_start_index"], x["root_end_index"]), reverse=True)
            for diff in diffs:
                s = diff["root_start_index"]
                e = diff["root_end_index"]
                o = diff["original"]
                t = diff["tag"]
                assert text[s:e] == o
                text = text[:s] + "**" + text[s:e] + f"**[{t}]" + text[e:]
            print(text, file=f_out)


def extract(diff_path: str, M: int, N: int=8888, stat_only=False):
    tag_mapping = load_tag_mapping()
    
    tags_cnt = defaultdict(int)
    dataset = defaultdict(dict)
    num_errors = 0
    with jsonlines.open(diff_path) as f:
        for index, item in tqdm(enumerate(f)):
            if index >= M:
                break
            
            diffs = item["diffs"]
            tags = list(set([d["tag"] for d in diffs]))
            for t in tags:
                tags_cnt[t] += 1
            
            patterns: list[Pattern] = []
            for diff in diffs:
                if diff["tag"] in tag_mapping:
                    try:
                        patterns.append(Pattern(
                            content=diff["original"],
                            tag=tag_mapping[diff["tag"]],
                            start=diff["root_start_index"],
                            end=diff["root_end_index"],
                        ))
                    except InvalidTagError:
                        num_errors += 1
                    
            if len(patterns) > 0:
                new = dict(
                    content=item["original"],
                    patterns=patterns
                )
                
                k_new = hash_sample(new)
                
                for t in tags:
                    if t in tag_mapping:
                        dataset[tag_mapping[t]][k_new] = new
                
    for tag, v in sorted(tags_cnt.items()):
        print(tag, "=", v, "|", len(dataset.get(tag, [])))
    print("num_errors =", num_errors)

    if stat_only:
        return
    
    random.seed(N)
    keys = set()
    num_samples = 0
    dataset_lengths = sorted({k: len(v) for k, v in dataset.items()}.items(), key=lambda x: (x[1], x[0]))
    
    with jsonlines.open("data/train/dataset_news.jsonl", "w") as f:
        for tag, _ in dataset_lengths:
            values = list(dataset[tag].items())
            random.shuffle(values)
            
            cnt = 0
            for k, v in tqdm(values, desc=tag):
                if k not in keys:
                    keys.add(tag)
                    cnt += 1
                    f.write(sample_to_dict(v))
                    num_samples += 1
                if cnt >= N:
                    break
                
    print("num_samples =", num_samples)


if __name__ == "__main__":
    # eda("data/difference.jsonl")
    # scan(diff_path="data/difference.jsonl", B=10_000, S=0, E=10_000_000)
    # scan(diff_path="temp/difference.jsonl", B=10_000, M=10_000)
    # show()
    # extract(diff_path="data/difference.jsonl", M=1_000_000_000, N=8888)
    extract(diff_path="data/difference.jsonl", M=1_000_000_000, N=8888, stat_only=True)
    pass