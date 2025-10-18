import os
import sys

sys.path.append(".")
import jsonlines
import random
from tqdm import tqdm
from collections import Counter, defaultdict

from utils.text.tagger.prompts.elite import TAG_TO_PROMPT
from utils.text.tagger.augment.date import (
    Date_dm_Tagger,
    Date_dmy_Tagger,
    Date_my_Tagger,
    DateRange_dm_dmy_Tagger,
    DateRange_m_my_Tagger,
    DateRange_y_y_Tagger,
)
from utils.text.tagger.augment.number import (
    Integer_n_Tagger,
    Integer_big_Tagger,
    Float_n_Tagger,
    Float_big_Tagger,
    NumberRangeTagger,
    FractionTagger,
    MeasurementTagger,
    DimensionTagger,
    SportScoreTagger,
    MoneyTagger,
    MathExprTagger,
)
from utils.text.tagger.augment.time import TimeTagger, TimeRangeTagger
from utils.text.tagger.augment.base import calibrate_tags, BaseTagger, BaseRangeTagger, Pattern, RangePattern
from utils.text.tagger.augment.plate import PlateTagger
from utils.text.tagger.augment.phone import PhoneTagger


all_taggers = [
    Date_dm_Tagger(),
    Date_dmy_Tagger(),
    Date_my_Tagger(),
    DateRange_dm_dmy_Tagger(),
    DateRange_m_my_Tagger(),
    DateRange_y_y_Tagger(),
    Integer_n_Tagger(skip_extract_unit=True),
    Integer_big_Tagger(skip_extract_unit=True),
    Float_n_Tagger(skip_extract_unit=True),
    Float_big_Tagger(skip_extract_unit=True),
    NumberRangeTagger(skip_extract_unit=True),
    FractionTagger(skip_extract_unit=True),
    MeasurementTagger(strict_unit=True),
    DimensionTagger(),
    SportScoreTagger(),
    MoneyTagger(),
    MathExprTagger(),
    TimeTagger(),
    TimeRangeTagger(),
]
TAG_TO_TAGGER_for_augment: dict[str, BaseTagger | BaseRangeTagger] = {t.tag(): t for t in all_taggers}

TAG_TO_TAGGER_for_extract = {
    "ADDRESS": None,
    "ALPHANUM_ID": None,
    "DATE__FRACTION": {
        "DATE": Date_dm_Tagger(),
        "FRACTION": FractionTagger(skip_extract_unit=False),
    },
    "DATE_dm": Date_dm_Tagger(),
    "DATE_dmy": Date_dmy_Tagger(),
    "DATE_my": Date_my_Tagger(),
    "DATE_RANGE_y_y": DateRange_y_y_Tagger(),
    "DATE_RANGE_dm_dmy": DateRange_dm_dmy_Tagger(),
    "DATE_RANGE_m_my": DateRange_m_my_Tagger(),
    "DIMENSION": DimensionTagger(),
    "EMAIL": None,
    "FLOAT_n": Float_n_Tagger(skip_extract_unit=False),
    "FLOAT_big": Float_big_Tagger(skip_extract_unit=False),
    "FRACTION": FractionTagger(skip_extract_unit=False),
    "ID_NUMBER": None,
    "INTEGER__FLOAT": {
        "INTEGER": [Integer_big_Tagger(skip_extract_unit=False), Integer_n_Tagger(skip_extract_unit=False)],
        "FLOAT": Float_n_Tagger(skip_extract_unit=False),
    },
    "INTEGER_n": Integer_n_Tagger(skip_extract_unit=False),
    "INTEGER_big": Integer_big_Tagger(skip_extract_unit=False),
    "LEGAL_DOC_ID": None,
    "MATH_EXPR": MathExprTagger(),
    "MEASUREMENT": MeasurementTagger(strict_unit=False),
    "MONEY": MoneyTagger(),
    "NUMBER_RANGE__SCORE": {
        "NUMBER_RANGE": NumberRangeTagger(skip_extract_unit=True),
        "SCORE": SportScoreTagger(),
    },
    "NUMBER_RANGE": NumberRangeTagger(skip_extract_unit=False),
    "PHONE__INTEGER": {
        "PHONE": PhoneTagger(),
        "INTEGER": [Integer_n_Tagger(skip_extract_unit=True), Integer_big_Tagger(skip_extract_unit=True)],
    },
    "PHONE": PhoneTagger(),
    "PLATE": PlateTagger(),
    "ROMAN_NUMERAL__ALPHANUM_ID": {
        "ROMAN_NUMERAL": None,
        "ALPHANUM_ID": None,
    },
    "ROMAN_NUMERAL": None,
    "SCORE": SportScoreTagger(),
    "TIME_hm": TimeTagger(),
    "TIME_hms": TimeTagger(),
    "TIME_h": TimeTagger(),
    "TIME_RANGE": TimeRangeTagger(),
    "URL": None,
}
# Kiểm tra với TAG_TO_PROMPT để xem còn thiếu tag nào ko
oke = True
for k in TAG_TO_PROMPT:
    if k not in TAG_TO_PROMPT:
        print(k)
        oke = False
if oke is False:
    exit(1)


def extract(working_dir: str, verbose=False):
    outputs: list[tuple[str, str, list[Pattern | RangePattern]]] = []

    for fn in sorted(os.listdir(working_dir)):
        if fn.endswith(".jsonl") is False:
            continue

        with jsonlines.open(os.path.join(working_dir, fn)) as f:
            items = list(f)

        tag = fn[: -len(".jsonl")]
        if "__" not in tag:
            errors = []
            tagger = TAG_TO_TAGGER_for_extract[tag]

            if tagger is None:
                for item in items:
                    content, patterns = calibrate_tags(item["sentence"], default_tag=tag)
                    assert isinstance(patterns, list)
                    if len(patterns) == 0:
                        errors.append(1)
                    outputs.append((tag, content, patterns))

            else:
                for item in items:
                    try:
                        content, patterns = tagger.extract(item["sentence"])
                    except:
                        errors.append(1)
                        # if tag == "TIME_2":
                        #     print(item["sentence"])
                        #     raise
                        continue

                    outputs.append((tag, content, patterns))

            if verbose:
                print("[", tag, "]", "success =", len(items) - len(errors), "/", len(items), "|", "error =", Counter(errors))

        else:
            errors = []
            sub_errors = []

            for item in items:
                tagger_dict = TAG_TO_TAGGER_for_extract[tag]
                assert isinstance(tagger_dict, dict)

                try:
                    content, patterns = calibrate_tags(item["sentence"], default_tag=None)
                    assert isinstance(patterns, list)
                except:
                    # if tag == "ROMAN_NUMERAL__ALPHANUM_ID_0":
                    #     print(item["sentence"])
                    #     return
                    errors.append(2)
                    continue

                new_patterns = []
                for pattern in patterns:
                    if pattern.tag is None:
                        # if tag == "INTEGER__FLOAT_0":
                        # print(item["sentence"])
                        # print(content)
                        # print(patterns)
                        # return

                        sub_errors.append(1)
                        continue

                    if pattern.tag not in tagger_dict:
                        sub_errors.append(2)
                        continue

                    tagger = tagger_dict[pattern.tag]

                    if tagger is not None:
                        if isinstance(tagger, list) is False:
                            tagger = [tagger]

                        valid = False
                        for t in tagger:
                            if t.validate(pattern.content) == 0:
                                pattern.tag = t.tag()
                                valid = True

                        if valid is False:
                            # if tag == "INTEGER__FLOAT_0":
                            # if tag == "PHONE__INTEGER_0":
                            #     print(item["sentence"])
                            #     print(pattern)
                            #     return
                            sub_errors.append(3)
                            continue

                    new_patterns.append(pattern)

                if len(new_patterns) < len(patterns):
                    errors.append(1)

                if len(new_patterns) > 0:
                    outputs.append((tag, content, new_patterns))

            if verbose:
                print(
                    "[",
                    tag,
                    "]",
                    "success =",
                    len(items) - len(errors),
                    "/",
                    len(items),
                    "|",
                    "error =",
                    Counter(errors),
                    "|",
                    "sub_error =",
                    Counter(sub_errors),
                )

    print("outputs =", len(outputs))
    return outputs


def split(content: str, patterns: list[Pattern | RangePattern]) -> list[str]:
    if len(patterns) == 0:
        return [content]

    outsides = []
    for index, pattern in enumerate(patterns):
        if index == 0:
            outsides.append(content[: pattern.start])

        if index > 0:
            prev = patterns[index - 1]
            outsides.append(content[prev.end : pattern.start])

        if index == len(patterns) - 1:
            outsides.append(content[pattern.end :])
    assert len(outsides) == len(patterns) + 1, f"{len(outsides)} | {len(patterns)}"
    assert content == group(outsides, [p.content for p in patterns])
    return outsides


def group(outsides: list[str], insides: list[str]):
    assert len(outsides) == len(insides) + 1, f"{len(outsides)} | {len(insides)}"
    content = ""
    for i in range(len(insides)):
        content += outsides[i] + insides[i]
    content += outsides[-1]
    return content


def validate(content: str, patterns: list[Pattern | RangePattern]):
    for pattern in patterns:
        assert content[pattern.start : pattern.end] == pattern.content


def augment(content: str, patterns: list[Pattern | RangePattern], N: int, augment_range_pattern_ratio: float = 0.5):
    augmented = []

    linking_words = [
        " và ",
        " hoặc ",
        " với ",
        " cùng ",
        " hay ",
        " đến ",
        ", và ",
        ", hoặc ",
        ", với ",
        ", cùng ",
        ", hay ",
        ", đến ",
        " cùng với ",
        " hoặc là ",
        " hay là ",
        ", cùng với ",
        ", hoặc là ",
        ", hay là ",
        ", thậm chí ",
        " thậm chí ",
    ]

    for _ in range(N):
        new_parts: list[Pattern | RangePattern] = []
        outsides = split(content, patterns)
        new_content = ""
        is_aug = False

        for out, pattern in zip(outsides, patterns):
            new_content += out

            if pattern.tag not in TAG_TO_TAGGER_for_augment:
                new_parts.append(Pattern(content=pattern.content, tag=pattern.tag, start=len(new_content), end=None))
                new_content += pattern.content

            else:
                is_aug = True

                rand = random.random()
                if rand < 0.7:
                    dup = 1
                elif rand < 0.9:
                    dup = 2
                else:
                    dup = 3

                for dup_index in range(dup):
                    part = TAG_TO_TAGGER_for_augment[pattern.tag].augment(None)

                    if isinstance(part, RangePattern) and random.random() < augment_range_pattern_ratio:
                        _content, _parts = part.to_normalized_patterns()
                        for _part in _parts:
                            new_parts.append(_part.shift_index(len(new_content)))

                        new_content += _content

                        # if pattern.tag == "DATE_RANGE_y_y":
                        # if pattern.tag == "NUMBER_RANGE":
                        # if pattern.tag == "FRACTION":
                        #     print(_parts)
                        #     print(_content)
                        #     exit(0)

                    else:
                        new_parts.append(part.reset_start(len(new_content)))
                        new_content += part.content

                    if dup_index < dup - 1:
                        new_content += random.choice(linking_words)

                # if dup > 1:
                #     print(new_content)
                #     exit(0)

        new_content += outsides[-1]

        if is_aug:
            for part in new_parts:
                if new_content[part.start : part.end] != part.content:
                    print(new_content)
                    print(new_content[part.start : part.end])
                    print(part)

                    raise ValueError()

            augmented.append((new_content, new_parts))

    return augmented


def soften(p: Pattern | RangePattern):
    if isinstance(p, RangePattern):
        return Pattern(content=p.content, tag=p.tag, start=p.start, end=p.end)
    return p


def save(dataset: list, path: str):
    with jsonlines.open(path, "w") as f:
        for c, ps in dataset:
            f.write(dict(content=c, patterns=[soften(p).to_dict() for p in ps]))


def main():
    dataset: list[tuple[str, list[Pattern | RangePattern]]] = []
    dataset += extract("data/generate_6", verbose=False)
    dataset += extract("data/generate_7", verbose=False)

    if False:
        random.shuffle(dataset)
        for t, c, ps in dataset:
            oke = False
            for p in ps:
                # if p.tag in TAG_TO_TAGGER_for_augment:
                if p.tag == "FRACTION":
                    oke = True

            if oke:
                for k, v in augment(c, ps, 5):
                    print(k)
                    print(v)
                    print()
                break
        return

    split_by_tag = defaultdict(list)
    ratio = 0.1
    random.seed(8686)
    random.shuffle(dataset)

    for t, c, ps in dataset:
        split_by_tag[t].append((c, ps))

    train, test = [], []
    for vs in split_by_tag.values():
        N = int(len(vs) * ratio)
        test += vs[:N]
        train += vs[N:]

    augmented = []
    for c, ps in tqdm(train):
        augmented += augment(c, ps, 10)

    print("train =", len(train))
    print("test =", len(test))
    print("augmented =", len(augmented))

    train = train + augmented
    random.shuffle(train)
    print("train =", len(train))

    save(train, "data/train/train.jsonl")
    save(test, "data/train/test.jsonl")


def eda():
    dataset: list[tuple[str, list[Pattern | RangePattern]]] = []
    dataset += extract("data/generate_6", verbose=False)
    dataset += extract("data/generate_7", verbose=False)
    print(len(dataset))

    import jsonlines
    from collections import Counter, defaultdict

    lengths = defaultdict(list)
    contents = []

    for _, patterns in dataset:
        for pattern in patterns:
            lengths[pattern.tag].append(len(pattern.content))
            contents.append((pattern.tag, pattern.content))

    contents = sorted(contents)
    with jsonlines.open("data/train/patterns.jsonl", "w") as f:
        f.write_all(contents)

    for k, vs in lengths.items():
        print(k)
        print(Counter(vs))
        print()


if __name__ == "__main__":
    # eda()
    main()
