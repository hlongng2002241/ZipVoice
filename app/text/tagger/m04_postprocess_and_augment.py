import sys; sys.path.append(".") # fmt: skip
import os
import re
import jsonlines
import random
from typing import TypeAlias
from collections import Counter

from src.utils.text.tagger.prompts.elite import TAG_TO_PROMPT
from src.utils.text.tagger.augment.base import (
    calibrate_tags,
    validate_patterns,
    convert_to_raw_content,
    clean_content_with_patterns,
    BaseTagger,
    BaseRangeTagger,
    Pattern,
    RangePattern,
    InvalidTagError,
)
from src.utils.text.tagger.augment.date import (
    Date_dm_Tagger,
    Date_dmy_Tagger,
    Date_my_Tagger,
    DateRange_dm_dmy_Tagger,
    DateRange_m_my_Tagger,
    DateRange_y_y_Tagger,
)
from src.utils.text.tagger.augment.number import (
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
    MathExprTaggerV2,
)
from src.utils.text.tagger.augment.time import TimeTagger, TimeRangeTagger
from src.utils.text.tagger.augment.plate import PlateTagger
from src.utils.text.tagger.augment.phone import PhoneTagger
from src.utils.text.tagger.m13_evaluate import calibrate_rule


TaggerType: TypeAlias = BaseTagger | BaseRangeTagger


class BeautifulFormatAugment:
    ADDRESS = None
    ALPHANUM_ID = None
    DATE_dm = None
    DATE_dmy = None
    DATE_my = None
    DATE = None
    DATE_RANGE_y_y = None
    DATE_RANGE_dm_dmy = None
    DATE_RANGE_m_my = None


TAGGERS_FOR_BEAUTIFULLY_AUGMENT = {
    Date_dm_Tagger(): None,
    Date_dmy_Tagger(): None,
    Date_my_Tagger(): None,
    DateRange_dm_dmy_Tagger(): None,
    DateRange_m_my_Tagger(): None,
    DateRange_y_y_Tagger(): None,
    Integer_n_Tagger(skip_extract_unit=True): ["-10:10", "0:100", "0:100+", "1:5", "-5:0", "10:20+", "50:100", "-20:-10"],
    Integer_big_Tagger(skip_extract_unit=True, allow_space_splitter=True): [
        "_6_z3",
        "_6_z4",
        " 5_z4",
        " 9_z6",
        " 9_z5",
        "_7_z4",
        "_8_z6",
        " 4_z3",
        " 6_z4",
        " 7_z5",
    ],
    Float_n_Tagger(skip_extract_unit=True): ["-10:10f2", "0:10f1", "1:5f3", "-5:0f2", "10:20f4", "0:1f5", "0:0.1f6", "0.5:0.9f2"],
    Float_big_Tagger(skip_extract_unit=True): [
        "_6_f2zd3zf1",
        " 6_f2zd4",
        " 8_f2zd5zf2",
        "_9_f1zd5",
        "_7_f3zd5zf0",
        "_8_f2zd6zf1",
        " 7_f4zd6zf2",
        " 4_f5zd3zf3",
        " 6_f2zd4zf2",
        " 4_f1zd3zf0",
    ],
    NumberRangeTagger(skip_extract_unit=True): [
        "i_n=-10:10||i_n=-10:10",
        "i_n=0:100||f_n=0:10f2",
        "f_n=-5:5f1||f_big=_4_f2zd3",
        "i_big=_6_z4||f_big=_7_f2zd6zf0",
        "i_big= 8_z4||f_big= 8_f2zd6zf0",
        "i_n=1:10|_||i_n=10:20|_",
        "f_n=0:1f2|_||f_n=1:2f2|_",
        "i_big=_4_z3|_||i_big= 5_z4|_",
        "f_big=_6_f2zd4zf1|_||f_big=_7_f2zd5zf0|_",
    ],
    FractionTagger(skip_extract_unit=True): [
        "i_n=1:10/i_n=1:10",
        "f_n=0:1f2/i_n=1:10",
        "i_big=_4_/i_n=1:10/f_n=0:1f2",
        "i_n=1:5/i_n=6:10",
        "f_n=0.1:0.5f1/f_n=0.6:0.9f1",
        "i_big=_5_z4/i_big= 6_z4",
        "i_big= 6_z4/i_big=_7_z4",
        "f_big=_7_f2zd5zf0/f_big= 8_f2zd7zf0",
        "f_big= 7_f2zd5zf0/f_big=_9_f2zd7zf0",
        "i_n=1:10/i_n=1:10/i_n=1:10",
    ],
    MeasurementTagger(strict_unit=True): [
        "i_n=1:10|",
        "i_n=1:10|",
        "i_n=1:10|",
        "f_n=0:1f2|",
        "f_n=0:1f2|",
        "f_n=0:1f2|",
        "i_big=_4_|",
        "i_big=_4_|",
        "i_big=_4_|",
        "i_big= 5_|",
        "i_big= 5_|",
        "i_n=10:20|",
        "i_n=10:20|",
        "i_n=10:20|",
        "f_n=0.5:0.9f1|",
        "f_n=0.5:0.9f1|",
        "f_n=0.5:0.9f1|",
        "i_big=_5_z3|",
        "i_big= 6_z3|",
        "i_big= 6_z3|",
        "i_big= 6_z3|",
        "f_big=_6_f2zd0zf0|",
        "f_big=_6_f2zd0zf0|",
        "f_big=_6_f2zd0zf0|",
        "i_n=1:5|",
        "i_n=1:5|",
        "i_n=1:5|",
        "f_big= 7_f2zd4zf0|",
        "fr==i_n=1:10/i_n=1:10|",
        "fr==f_n=0:1f2/i_n=1:10|",
        "fr==f_big=_7_f2zd5zf0/f_big= 8_f2zd7zf0|",
        "fr==i_big= 6_z4/i_big=_7_z4|",
    ],
    DimensionTagger(): [
        "i_n=1:10||i_n=1:10",
        "f_n=0:1f2||i_n=1:10|_",
        "i_n=1:5||i_n=6:10||i_n=11:15|_",
        "f_n=0.1:0.5f1||f_n=0.6:0.9f1||f_n=1:1.5f1|_",
        "i_big=_4_z3||i_big=_5_z4",
        "f_big=_6_f2zd5zf0||f_big=_7_f2zd6zf0|_",
        "i_n=1:10||i_n=1:10||i_n=1:10",
        "i_n=1:10||i_n=1:10|_||i_n=1:10|_",
    ],
    SportScoreTagger(): [
        "i_n=0:5||i_n=0:5",
        "i_n=0:100||i_n=0:100",
        "i_n=1:10||i_n=1:10",
        "i_n=0:10||i_n=0:10",
        "i_n=0:3||i_n=0:3",
        "i_n=0:2||i_n=0:2",
        "i_n=0:1||i_n=0:1",
        "i_n=0:50||i_n=0:50",
    ],
    MoneyTagger(): [
        "i_n=1000:10000|vnd",
        "i_big=_6_|$",
        "f_n=1:10f2|€!",
        "i_n=100:500|usd",
        "i_big=_7_z6|đ",
        "f_n=0.5:0.9f1|£!",
        "i_big=_7_z6|vnđ",
        "i_big=_8_z6|đ",
        "i_big=_8_z6|k",
        "i_big=_8_z6|K",
        "i_big=_8_z6|tr",
        "i_big=_8_z6|tr",
        "i_big=_8_z6|đ",
    ],
    MathExprTaggerV2(): ["i_n", "f_n", "i_big", "f_big", "i_n=1:10", "f_n=0:1f2", "i_big=_4_z0", "f_big=_6_f2zd5zf0"],
    PhoneTagger(): None,
    PlateTagger(): None,
    TimeTagger(): [
        "i_n",
        "i_n#h|i_n#m|c=1",
        "i_n#h|i_n#m|i_n#s|c=1",
        "i_n=0:23#h|c=0",
        "i_n=0:59#m|c=0",
        "i_n=0:59#s|c=0",
        "i_n=0:23#h|i_n=0:59#m|c=0",
        "i_n=0:23#h|i_n=0:59#m|i_n=0:59#s|c=0",
        "i_n=0:23#h|i_n=0:59#m|i_n=0:59#s|c=1",
    ],
    TimeRangeTagger(): [
        "i_n||i_n",
        "i_n#h|i_n#m|c=1||i_n#h|i_n#m|c=1",
        "i_n=0:23#h|c=0||i_n=0:23#h|c=0",
        "i_n=0:23#h|i_n=0:59#m|c=0||i_n=0:23#h|i_n=0:59#m|c=0",
        "i_n=0:23#h|i_n=0:59#m|c=1||i_n=0:23#h|i_n=0:59#m|c=1",
        "i_n=0:23#h|i_n=0:59#m|i_n=0:59#s|c=0||i_n=0:23#h|i_n=0:59#m|i_n=0:59#s|c=0",
        "i_n=0:23#h|i_n=0:59#m|i_n=0:59#s|c=1||i_n=0:23#h|i_n=0:59#m|i_n=0:59#s|c=1",
        "i_n=1:10#s|c=0||i_n=11:20#s|c=0",
        "i_n=1:5#h|c=0||i_n=6:10#h|c=0",
        "i_n=1:5#h|i_n=1:5#m|c=1||i_n=6:10#h|i_n=6:10#m|c=1",
    ],
}

TAGGERS_FOR_BEAUTIFULLY_AUGMENT = {k.tag(): (k, v) for k, v in TAGGERS_FOR_BEAUTIFULLY_AUGMENT.items()}

TAG_TO_TAGGER_FOR_AUGMENT: dict[str, tuple | list[tuple]] = {
    "ADDRESS": None,
    "ALPHANUM_ID": None,
    "DATE_dm": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_dm"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_dmy"],
    ],
    "DATE_dmy": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_dm"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_dmy"],
    ],
    "DATE_my": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_my"],
    "DATE": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_dm"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_dmy"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_my"],
    ],
    "DATE_RANGE_y_y": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_RANGE_y_y"],
    "DATE_RANGE_dm_dmy": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_RANGE_dm_dmy"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_RANGE_m_my"],
    ],
    "DATE_RANGE_m_my": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_RANGE_dm_dmy"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_RANGE_m_my"],
    ],
    "DATE_RANGE": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_RANGE_dm_dmy"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DATE_RANGE_m_my"],
    ],
    "DIMENSION": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["DIMENSION"],
    "EMAIL": None,
    "FLOAT_n": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["FLOAT_n"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["FLOAT_big"],
    ],
    "FLOAT_big": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["FLOAT_n"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["FLOAT_big"],
    ],
    "FLOAT": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["FLOAT_n"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["FLOAT_big"],
    ],
    "FOREIGN_WORD": None,
    "FRACTION": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["FRACTION"],
    "INTEGER_n": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["INTEGER_n"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["INTEGER_big"],
    ],
    "INTEGER_big": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["INTEGER_n"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["INTEGER_big"],
    ],
    "INTEGER": [
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["INTEGER_n"],
        TAGGERS_FOR_BEAUTIFULLY_AUGMENT["INTEGER_big"],
    ],
    "LEGAL_DOC_ID": None,
    "MATH_EXPR": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["MATH_EXPR"],
    "MEASUREMENT": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["MEASUREMENT"],
    "MONEY": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["MONEY"],
    "NUMBER_RANGE": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["NUMBER_RANGE"],
    "PHONE": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["PHONE"],
    "PLATE": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["PLATE"],
    "ROMAN_NUMERAL": None,
    "SPORT_SCORE": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["SPORT_SCORE"],
    "TIME_hms": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["TIME_hms"],
    "TIME_hm": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["TIME_hms"],
    "TIME_h": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["TIME_hms"],
    "TIME": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["TIME_hms"],
    "TIME_RANGE": TAGGERS_FOR_BEAUTIFULLY_AUGMENT["TIME_RANGE"],
    "URL": None,
}

TAG_TO_TAGGER_FOR_EXTRACT = {
    "ADDRESS": None,
    "ALPHANUM_ID": None,
    "DATE__FRACTION": {
        "DATE": [Date_dm_Tagger()],
        "FRACTION": [FractionTagger(skip_extract_unit=False)],
    },
    "DATE_dm": Date_dm_Tagger(),
    "DATE_dmy": Date_dmy_Tagger(),
    "DATE_my": Date_my_Tagger(),
    "DATE": [Date_dm_Tagger(), Date_dmy_Tagger(), Date_my_Tagger()],
    "DATE_RANGE_y_y": DateRange_y_y_Tagger(),
    "DATE_RANGE_dm_dmy": DateRange_dm_dmy_Tagger(),
    "DATE_RANGE_m_my": DateRange_m_my_Tagger(),
    "DATE_RANGE": [DateRange_dm_dmy_Tagger(), DateRange_m_my_Tagger()],
    "DATE_RANGE__MATH_EXPR": {
        "DATE_RANGE": [DateRange_dm_dmy_Tagger(), DateRange_m_my_Tagger(), DateRange_y_y_Tagger()],
        "MATH_EXPR": [MathExprTaggerV2(strict_unit=False)],
    },
    "DIMENSION": DimensionTagger(allow_closed=True),
    "EMAIL": None,
    "FLOAT_n": Float_n_Tagger(skip_extract_unit=False),
    "FLOAT_big": Float_big_Tagger(skip_extract_unit=False),
    "FLOAT": [Float_n_Tagger(skip_extract_unit=False)],
    "FOREIGN_WORD": None,
    "FRACTION": FractionTagger(skip_extract_unit=False),
    "INTEGER__FLOAT": {
        "INTEGER": [Integer_big_Tagger(skip_extract_unit=False), Integer_n_Tagger(skip_extract_unit=False)],
        "FLOAT": [Float_n_Tagger(skip_extract_unit=False)],
    },
    "INTEGER_n": Integer_n_Tagger(skip_extract_unit=False),
    "INTEGER_big": Integer_big_Tagger(skip_extract_unit=False),
    "INTEGER": [Integer_big_Tagger(skip_extract_unit=False), Integer_n_Tagger(skip_extract_unit=False)],
    "LEGAL_DOC_ID": None,
    "MATH_EXPR": MathExprTaggerV2(strict_unit=False),
    "MATH_EXPR__NUMBER_RANGE": {
        "MATH_EXPR": [MathExprTaggerV2(strict_unit=False)],
        "NUMBER_RANGE": [NumberRangeTagger(skip_extract_unit=False)],
    },
    "MATH_EXPR__SPORT_SCORE": {
        "MATH_EXPR": [MathExprTaggerV2(strict_unit=False)],
        "SPORT_SCORE": [SportScoreTagger()],
    },
    # "MATH_EXPR__TIME_RANGE": {
    #     "MATH_EXPR": [MathExprTaggerV2(strict_unit=False)],
    #     "TIME_RANGE": [TimeRangeTagger()],
    # },
    "MEASUREMENT": MeasurementTagger(strict_unit=False),
    "MONEY": MoneyTagger(),
    "NUMBER_RANGE__SPORT_SCORE": {
        "NUMBER_RANGE": [NumberRangeTagger(skip_extract_unit=True)],
        "SPORT_SCORE": [SportScoreTagger()],
    },
    "NUMBER_RANGE": NumberRangeTagger(skip_extract_unit=False),
    "PHONE__INTEGER": {
        "PHONE": [PhoneTagger()],
        "INTEGER": [Integer_n_Tagger(skip_extract_unit=True), Integer_big_Tagger(skip_extract_unit=True)],
    },
    "PHONE": PhoneTagger(),
    "PLATE": PlateTagger(),
    "ROMAN_NUMERAL__ALPHANUM_ID": {
        "ROMAN_NUMERAL": None,
        "ALPHANUM_ID": None,
    },
    "ROMAN_NUMERAL": None,
    "SPORT_SCORE": SportScoreTagger(),
    "TIME_hm": TimeTagger(),
    "TIME_hms": TimeTagger(),
    "TIME_h": TimeTagger(),
    "TIME": TimeTagger(),
    "TIME_RANGE": TimeRangeTagger(),
    "URL": None,
}
# Kiểm tra với TAG_TO_PROMPT để xem còn thiếu tag nào ko
oke = True
for k in TAG_TO_PROMPT:
    if k not in TAG_TO_TAGGER_FOR_EXTRACT and k not in ["ID_NUMBER"]:
        print(k)
        oke = False
if oke is False:
    print("--- MISSING TAG_TO_TAGGER_for_extract ---")
    exit(1)

missing = []
for k in TAG_TO_PROMPT:
    if "__" not in k and k not in TAG_TO_TAGGER_FOR_AUGMENT:
        missing.append(k)
missing = sorted(missing)
if missing != ["ID_NUMBER"]:
    print("--- MISSING TAG_TO_TAGGER_for_augment ---")
    print(missing)
    exit(1)


def shift_patterns(patterns: list[Pattern], shift: int, start: int = 0, end: int = None, content: str = None):
    if end is None:
        end = len(patterns)

    for index in range(start, end):
        nxt = patterns[index]
        nxt = nxt.shift_index(shift)
        if content is not None:
            assert content[nxt.start : nxt.end] == nxt.content, f"{content[nxt.start : nxt.end]} != {nxt.content}"
        patterns[index] = nxt


def augment_with_retry(tagger: BaseTagger, fmt, num_retry: int):
    for _ in range(num_retry):
        try:
            return tagger.augment(fmt)
        except InvalidTagError:
            pass


def augment_beautifully(tag: str):
    aug_and_fmts = TAG_TO_TAGGER_FOR_AUGMENT[tag]
    if aug_and_fmts is None:
        return None

    if isinstance(aug_and_fmts, list):
        aug_and_fmts = random.choice(aug_and_fmts)
    aug, fmts = aug_and_fmts

    if fmts is not None:
        fmts = random.choice(fmts)

    # new = aug.augment(fmts)
    return augment_with_retry(aug, fmts, 100)


def resolve_by_augment(content: str, patterns: list[Pattern], index: int):
    pattern = patterns[index]
    new = augment_beautifully(pattern.tag)
    if new is None:
        return content, None
    new = new.reset_start(pattern.start)

    content = content[: pattern.start] + new.content + content[pattern.end :]
    shift_pos = len(new.content) - len(pattern.content)
    shift_patterns(patterns, shift_pos, index + 1)

    patterns[index] = new

    return content, new


def validate_and_resolve_error(sentence: str, default_tag=None, auto_resolve=True):
    content, patterns = calibrate_tags(sentence, default_tag=default_tag)
    errors = []

    for index in range(len(patterns)):
        pattern = patterns[index]

        if pattern.tag in TAG_TO_TAGGER_FOR_EXTRACT:
            taggers = TAG_TO_TAGGER_FOR_EXTRACT[pattern.tag]
            if taggers is None:
                continue

            if isinstance(taggers, list) is False:
                taggers = [taggers]

            tag = None
            for tagger in taggers:
                tagger: BaseTagger | BaseRangeTagger
                if tagger.validate(pattern.content) == 0:
                    tag = tagger.tag()
                    break

            if tag is not None:
                if pattern.tag in ["INTEGER", "FLOAT"]:
                    # print(tag, pattern.content)
                    patterns[index] = pattern.reset_tag(tag)
            else:
                # If the content is invalid, then augment it
                errors.append(pattern.reset_start(0))

                if auto_resolve:
                    content, new = resolve_by_augment(content, patterns, index)
                    assert new is not None

                    # new = augment_beautifully(pattern.tag)
                    # new = new.reset_start(pattern.start)

                    # content = content[: pattern.start] + new.content + content[pattern.end :]
                    # shift_pos = len(new.content) - len(pattern.content)
                    # shift_patterns(patterns, shift_pos, index + 1)

                    # patterns[index] = new

        else:
            raise NotImplementedError(pattern.tag)

    validate_patterns(content, patterns)
    return content, patterns, errors


def group(outsides: list[str], insides: list[str]):
    assert len(outsides) == len(insides) + 1, f"{len(outsides)} | {len(insides)}"
    content = ""
    for i in range(len(insides)):
        content += outsides[i] + insides[i]
    content += outsides[-1]
    return content


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


def augment(
    content: str,
    patterns: list[Pattern],
    beautiful_augment_prob: float = 0.5,
    split_range_pattern_prob: float = 0.5,
    linking_word_prob: float = 0.5,
):
    linking_words = [
        " và ",
        " hoặc ",
        " với ",
        " cùng ",
        " hay ",
        " đến ",
        " cùng với ",
        " hoặc là ",
        " hay là ",
        " thậm chí ",
        " , ",
        " , và ",
        " , hoặc ",
        " , với ",
        " , cùng ",
        " , hay ",
        " , đến ",
        " , cùng với ",
        " , hoặc là ",
        " , hay là ",
        " , thậm chí ",
    ]

    new_patterns = []
    outsides = split(content, patterns)
    new_content = ""
    is_aug = False

    for out, pattern in zip(outsides, patterns):
        new_content += out

        aug_and_fmts = TAG_TO_TAGGER_FOR_AUGMENT[pattern.tag]
        if aug_and_fmts is None:
            new_patterns.append(pattern.reset_start(len(new_content)))
            new_content += pattern.content
            continue

        if isinstance(aug_and_fmts, list) is False:
            aug_and_fmts = [aug_and_fmts]

        aug, fmts = random.choice(aug_and_fmts)
        aug: BaseTagger | BaseRangeTagger

        is_aug = True
        dup_prob = random.random()
        num_dup = 1 if dup_prob < 0.7 else (2 if dup_prob < 0.9 else 3)

        for dup_index in range(num_dup):
            if random.random() < beautiful_augment_prob and fmts is not None:
                # pattern = aug.augment()
                pattern = augment_with_retry(aug, random.choice(fmts), 100)
            else:
                # pattern = aug.augment(None)
                pattern = augment_with_retry(aug, None, 100)

            # Transform range pattern
            if (isinstance(pattern, RangePattern) and random.random() < split_range_pattern_prob) or isinstance(
                aug, (MeasurementTagger, MathExprTaggerV2)
            ):
                if isinstance(aug, (MeasurementTagger, MathExprTaggerV2)):
                    _content = pattern.content
                    _patterns = pattern.parts
                else:
                    _content, _patterns = pattern.to_normalized_patterns()

                if len(_patterns) == 0:
                    new_content += _content

                else:
                    _outsides = split(_content, _patterns)

                    # Restore sub patterns
                    new_content += _outsides[0]
                    for _idx, _pattern in enumerate(_patterns):
                        new_patterns.append(_pattern.reset_start(len(new_content)))
                        new_content += _pattern.content
                        new_content += _outsides[_idx + 1]

            else:
                new_patterns.append(pattern.reset_start(len(new_content)))
                new_content += pattern.content

            if dup_index < num_dup - 1:
                if random.random() < linking_word_prob:
                    new_content += random.choice(linking_words)
                else:
                    new_content += " "

        new_content += outsides[-1]

        validate_patterns(new_content, new_patterns)

    return is_aug, new_content, new_patterns


def augment_multiple(
    content: str, patterns: list[Pattern], N: int, beautiful_augment_prob: float = 0.5, split_range_pattern_prob: float = 0.5
):
    augments = []
    for _ in range(N):
        status, new_content, new_patterns = augment(
            content, patterns, beautiful_augment_prob=beautiful_augment_prob, split_range_pattern_prob=split_range_pattern_prob
        )
        if status:
            augments.append((new_content, new_patterns))
    return augments


def load_md(path: str, resolve: bool):
    inputs = []

    with open(path) as f:
        tag = None
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            if line[0] == "#":
                tag = line[1:].strip()
            elif line[0] == "-":
                assert tag is not None
                sentence = line[1:].strip()
                # sentence = " ".join(sentence.split())
                inputs.append((sentence, tag))
            else:
                raise NotImplementedError(line)

    inputs = list(set(inputs))
    results = []
    errors = []

    for sentence, default_tag in inputs:
        if "__" in default_tag:
            default_tag = None

        if resolve:
            content, patterns, es = validate_and_resolve_error(sentence, default_tag=default_tag)
            errors.extend(es)
        else:
            content, patterns = calibrate_tags(sentence, default_tag=default_tag)
        results.append((content, patterns))

    return results, errors


def load_v2_jsonl(path, resolve: bool):
    inputs: list[tuple[str, list[Pattern]]] = []
    errors = []
    with jsonlines.open(path) as f:
        for item in f:
            for e, sentence in item["sentences"]:
                if e is None:
                    if resolve:
                        content, patterns, es = validate_and_resolve_error(sentence)
                        errors.extend(es)
                    else:
                        content, patterns = calibrate_tags(sentence)
                    inputs.append((content, patterns))

    return inputs, errors


def load_v2_md(path: str, resolve: bool):
    inputs: list[tuple[str, list[Pattern]]] = []
    errors = []

    with open(path, encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            if line[0] == "-":
                line = line[1:].strip()
                if resolve:
                    content, patterns, es = validate_and_resolve_error(line)
                    errors.extend(es)
                else:
                    content, patterns = calibrate_tags(line)
                inputs.append((content, patterns))

    return inputs, errors


def load_v3_from_eval(path: str, resolve: bool):
    assert resolve is False  # BUG: Should think about resolving here
    inputs = []
    errors = []
    p = re.compile(r"\*\*(.*?)\*\*\[(.*?)\]")
    err_codes = []

    with jsonlines.open(path) as f:
        for item in f:
            raw_content = item["messages"][0]["content"]
            lines = [l.strip() for l in raw_content.split("\n") if l.strip() != ""]
            assert lines[-2].startswith("INPUT_TEXT")
            raw_content = lines[-1][1:].strip()

            # content, patterns = validate_and_resolve_error(content, auto_resolve=False)
            try:
                content, patterns = calibrate_tags(raw_content)
            except InvalidTagError:
                print(raw_content)
                err_codes.append(1)
                continue

            invalid_eval = False

            evaluations = []
            for ev in item["evaluations"]:
                ps = p.search(ev["entity"])
                if ps is None:
                    continue
                    invalid_eval = True
                    print(raw_content)
                    raise ValueError()
                    break

                assert "content" not in ev
                assert "tag" not in ev
                ev["content"] = ps.group(1)
                ev["tag"] = ps.group(2)

                result = ev["evaluation"].lower()
                assert result in ["đúng", "sai"]

                evaluations.append(ev)

            # if invalid_eval:
            #     err_codes.append(1)
            #     continue

            if len(patterns) != len(evaluations):
                err_codes.append(2)
                continue

            ec = set()
            for pattern, ev in zip(patterns, evaluations):
                if pattern.content != ev["content"]:
                    ec.add(3)
                if pattern.tag.lower() != ev["tag"].lower():
                    ec.add(4)

            if len(ec):
                err_codes.extend(list(ec))
                continue

            for index in range(len(patterns)):
                pattern = patterns[index]
                ev = evaluations[index]

                if ev["evaluation"].lower() == "đúng":
                    continue
                else:
                    new_tag = calibrate_rule(ev)
                    if new_tag is not None:
                        continue
                    else:
                        # assert resolve is True
                        # if resolve:
                        content, new = resolve_by_augment(content, patterns, index)
                        # print(new)
                        if new is None:
                            # delete this pattern
                            patterns[index] = None
                            shift_patterns(patterns, -len(pattern.content), start=index + 1)
                            content = content[: pattern.start] + content[pattern.end :]

            patterns = [p for p in patterns if p is not None]
            # print(content)
            validate_patterns(content, patterns)
            content, patterns = clean_content_with_patterns(content, patterns)

            # content, patterns, es = validate_and_resolve_error(convert_to_raw_content(content, patterns))
            # errors.extend(es)

            inputs.append((content, patterns))

    print("err_codes =", Counter(err_codes))

    return inputs, errors


def soften(p: Pattern | RangePattern):
    if isinstance(p, RangePattern):
        return Pattern(content=p.content, tag=p.tag, start=p.start, end=p.end)
    return p


def save(dataset: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, "w") as f:
        for c, ps in dataset:
            f.write(dict(content=c, patterns=[soften(p.refine()).to_dict() for p in ps]))


def extract_and_augment(input_path: str, output_path: str, version: int, resolve: bool, num_augment: int = 10):
    if version == 1:
        inputs, errors = load_md(input_path, resolve)
    elif version == 2:
        if input_path.endswith(".jsonl"):
            inputs, errors = load_v2_jsonl(input_path, resolve)
        else:
            inputs, errors = load_v2_md(input_path, resolve)
    elif version == 3:
        inputs, errors = load_v3_from_eval(input_path, resolve)
    else:
        raise NotImplementedError(version)

    print("Number of error patterns =", len(errors))
    for e in errors[:10]:
        print(e)
    print("Resolve =", resolve)
    print("Number of samples =", len(inputs))

    input("Start augmenting (press any key to continue) ")

    random.seed(8686)

    beautiful_augment_prob = 0.5
    split_range_pattern_prob = 0.5
    augmented = []
    for content, patterns in inputs:
        augmented.extend(
            augment_multiple(
                content,
                patterns,
                N=num_augment,
                beautiful_augment_prob=beautiful_augment_prob,
                split_range_pattern_prob=split_range_pattern_prob,
            )
        )

    print("Number of augmented samples =", len(augmented))
    print("\n\n")

    inputs += augmented
    random.seed(8686)
    random.shuffle(inputs)

    save(inputs, output_path)


def merge_files(input_paths: list[str], output_path: str):
    with jsonlines.open(output_path, "w") as f_out:
        for path in input_paths:
            with jsonlines.open(path) as f:
                for item in f:
                    f_out.write(item)


def main():
    output_dir = "data/train_10"

    extract_and_augment(
        input_path="data/generate_8/evaluations.jsonl",
        output_path=os.path.join(output_dir, "raw/train_from_multi_gen_evaluated.jsonl"),
        version=3,
        resolve=False,
    )
    extract_and_augment(
        input_path="data/results/training_sorted.md",
        output_path=os.path.join(output_dir, "raw/train_from_single_gen.jsonl"),
        version=1,
        resolve=True,
    )
    extract_and_augment(
        input_path="data/human_validated_sorted.md",
        output_path=os.path.join(output_dir, "raw/test_from_single_gen.jsonl"),
        version=1,
        resolve=True,
        num_augment=1,
    )
    extract_and_augment(
        input_path="data/generate_8/human_validated.md",
        output_path=os.path.join(output_dir, "raw/test_from_multi_gen.jsonl"),
        version=2,
        resolve=False,
        num_augment=1,
    )

    merge_files(
        [
            os.path.join(output_dir, "raw/train_from_single_gen.jsonl"),
            os.path.join(output_dir, "raw/train_from_multi_gen_evaluated.jsonl"),
        ],
        os.path.join(output_dir, "train.jsonl"),
    )
    merge_files(
        [
            os.path.join(output_dir, "raw/test_from_multi_gen.jsonl"),
            os.path.join(output_dir, "raw/test_from_single_gen.jsonl"),
        ],
        os.path.join(output_dir, "test.jsonl"),
    )


if __name__ == "__main__":
    main()
    exit()

    content, patterns, es = validate_and_resolve_error(
        "khoảng **18-4-2029**[DATE] đến **25-5/1846**[DATE] tại **19**[ADDRESS] Láng **Soft**[FOREIGN_WORD]. vào lúc **13h-14h50**[TIME_RANGE] cùng ngày với thể tích **10 km**[MEASUREMENT], và **2+3**[MATH_EXPR]"
    )
    print(es)
    print(convert_to_raw_content(content, patterns))

    status, content, patterns = augment(content, patterns, beautiful_augment_prob=0, split_range_pattern_prob=0)
    print(status)
    print(convert_to_raw_content(content, patterns))
