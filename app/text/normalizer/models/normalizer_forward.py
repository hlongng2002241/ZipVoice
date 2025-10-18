import re
import logging
import string
from typing import Callable, Literal
from dataclasses import dataclass

from roman import fromRoman, InvalidRomanNumeralError

from ...tagger.prompts.elite import TAG_TO_PROMPT
from ...tagger.augment.base import BaseTagger, convert_to_raw_content
from ...tagger.augment.number import (
    Integer_n_Tagger,
    Integer_big_Tagger,
    Float_n_Tagger,
    Float_big_Tagger,
)
from ..rule.cores import (
    time2words,
    num2words_float,
    num2words_integer,
    phone2words,
    alpha_num2words,
    break_word,
)
from ..rule.utils.units import (
    UNITS_DICT,
    CURRENCY_OVER_ANY_UNITS,
    PREFIX_CURRENCY_UNITS,
    SUFFIX_CURRENCY_UNITS,
)
from ..rule.utils.characters import (
    FULL_MATH_OPERATOR_DICT,
    BASIC_MATH_OPERATOR_DICT,
    BASIC_NUMBER_OPERATOR_DICT,
    NON_SILENT_SYMBOL_DICT,
    SILENT_SYMBOL_LIST,
    EMAIL_AND_URL_COMPONENT_DICT,
    SYMBOL_DICT,
)
from ..rule.normalizer import RuleBasedTextNormalizer
from .inference_v2 import TaggerInference, LabelMatching


logger = logging.getLogger(__name__)


def load_vocab(path: str) -> list:
    vocab = set()
    with open(path, encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            word = line.split()[0]
            vocab.add(word)
    return list(vocab)


ALPHABET_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz"
VI_CHARACTERS = (
    "à á ả ã ạ ă ằ ắ ẳ ẵ ặ â ầ ấ ẩ ẫ ậ đ è é ẻ ẽ ẹ ê ề ế ể ễ ệ ì í ỉ ĩ ị ò ó ỏ õ ọ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ ù ú ủ ũ ụ ư ừ ứ ử ữ ự ỳ ý ỷ ỹ ỵ"
    "À Á Ả Ã Ạ Ă Ằ Ắ Ẳ Ẵ Ặ Â Ầ Ấ Ẩ Ẫ Ậ Đ È É Ẻ Ẽ Ẹ Ê Ề Ế Ể Ễ Ệ Ì Í Ỉ Ĩ Ị Ò Ó Ỏ Õ Ọ Ô Ồ Ố Ổ Ỗ Ộ Ơ Ờ Ớ Ở Ỡ Ợ Ù Ú Ủ Ũ Ụ Ư Ừ Ứ Ử Ữ Ự Ỳ Ý Ỷ Ỹ Ỵ"
)
VI_CHARACTERS = ALPHABET_CHARACTERS + VI_CHARACTERS
VI_CHARACTERS = VI_CHARACTERS.replace(" ", "")

LINKING_WORDS = ["và", "hoặc", "với", "cùng", "hay", "đến"]
LINKING_WORDS.extend([l + "," for l in LINKING_WORDS])


def endswith(text: str, suffixes: list[str]):
    for suf in suffixes:
        if text.endswith(suf):
            return True
    return False


def get_last_words(text: str, n: int, ignore_linking_words=False, ignore_punctuation=False):
    assert n > 0
    ws = text.strip().split(" ")
    last_words = []
    while len(ws) > 0 and len(last_words) < n:
        w = ws.pop()
        if ignore_linking_words and w in LINKING_WORDS:
            continue
        if ignore_punctuation and has_no_alpha_or_num(w):
            continue
        last_words.append(w)
    return last_words[::-1]


def is_in_last_words(text: str, n: int, words: list[str], ignore_case=True, ignore_linking_words=False, ignore_punctuation=False):
    if ignore_case:
        words = [w.lower() for w in words]

    for w in get_last_words(text, n, ignore_linking_words=ignore_linking_words, ignore_punctuation=ignore_punctuation):
        if ignore_case:
            w = w.lower()
        if w in words:
            return True
    return False


def split_parts(text: str, conn_str: str, num_parts: int):
    parts = []
    for c in [f" {conn_str} ", f"{conn_str} ", f" {conn_str}", conn_str]:
        if c in text:
            parts = text.split(c)
            break
    if num_parts is not None and len(parts) != num_parts:
        raise NormalizeError()
    return parts[0].strip(), parts[1].strip()


def split_by_chars(text: str, chars: str):
    parts = [""]
    for ch in text:
        if ch not in chars:
            parts[-1] += ch
        else:
            parts.append(ch)
            parts.append("")
    return [p.strip() for p in parts if p.strip() != ""]


def has_only_number(text: str):
    for ch in text:
        if ch.isdigit() is False:
            return False
    return True


def has_number(text: str):
    for ch in text:
        if ch.isdigit():
            return True
    return False


def has_no_alpha_or_num(text: str):
    for ch in text:
        if ch.isalpha() or ch.isdigit():
            return False
    return True


def has_non_silent_char(text: str):
    for ch in text:
        if ch.strip() != "" and ch not in SILENT_SYMBOL_LIST:
            return True
    return False


def is_alpha(char: str):
    return char in VI_CHARACTERS


def startswith_unit(text: str, unit: str):
    if unit == "":
        return True
    if text.startswith(unit):
        if has_no_alpha_or_num(unit):
            return True

        text = text[len(unit) :]
        if (
            len(text) == 0
            or text[0] in string.punctuation
            or text[0].strip() == ""
            or text[0] in "0123456789"
            or is_alpha(unit[-1]) is False
        ):
            return True
    return False


def all_is_in(values: list, is_in_list: list):
    for v in values:
        if v not in is_in_list:
            return False
    return True


def any_is_in(values: list, is_in_list: list):
    for v in values:
        if v in is_in_list:
            return True
    return False


def all_none(values: list):
    for v in values:
        if v is not None:
            return False
    return True


@dataclass
class NumberPart:
    TAG_NUMBER = "number"
    TAG_OPERATOR = "operator"
    TAG_OTHER = "other"
    TAG_OPERATOR_UNIT = "operator_unit"

    TAGS = [TAG_NUMBER, TAG_OPERATOR, TAG_OTHER, TAG_OPERATOR_UNIT]

    def __init__(self, start: int, content: str, tag: str | None):
        self._validate_tag(tag)

        self.start = start
        self.content = content
        self.tag = tag

    def __str__(self):
        tag = self.tag if self.tag is None else f"'{self.tag}'"
        return self.__class__.__name__ + "(" + (f"start={self.start}, " f"content='{self.content}', " f"tag={tag}") + ")"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def validate_parts_content(text: str, parts: list["NumberPart"]):
        for part in parts:
            assert text[part.start : part.start + len(part.content)] == part.content, (
                text[part.start : part.start + len(part.content)] + " != " + part.content
            )
        return parts

    @staticmethod
    def validate_parts_order(parts: list["NumberPart"]):
        for index, part in enumerate(parts):
            if index > 0:
                prev = parts[index - 1]
                assert prev.start + len(prev.content) <= part.start, str(prev) + " > " + str(part)
        return parts

    @classmethod
    def _validate_tag(cls, tag):
        assert tag is None or tag in cls.TAGS

    @staticmethod
    def add_outside_parts(text: str, parts: list["NumberPart"], tag: str = None):
        all_parts: list["NumberPart"] = []
        index = 0
        for part in parts:
            if index < part.start:
                all_parts.append(NumberPart(index, text[index : part.start], tag))
            all_parts.append(part)
            index = part.start + len(part.content)
        if index < len(text):
            all_parts.append(NumberPart(index, text[index:], tag))
        return all_parts

    @staticmethod
    def strip_parts(parts: list["NumberPart"]):
        new_parts: list["NumberPart"] = []
        for part in parts:
            content = part.content
            if content.strip() != "":
                new_parts.append(NumberPart(part.start + len(content) - len(content.lstrip()), content.strip(), part.tag))
        return new_parts

    @staticmethod
    def merge_parts(parts: list["NumberPart"]):
        merge_stack = None
        final_parts: list[NumberPart] = []
        for part in parts:
            if part.tag == NumberPart.TAG_NUMBER:
                if merge_stack is None:
                    merge_stack = [part]
                else:
                    final_parts.extend(merge_stack)
                    merge_stack = [part]
            elif part.tag == NumberPart.TAG_OPERATOR:
                if merge_stack is not None:
                    merge_stack.append(part)
                else:
                    final_parts.append(NumberPart(part.start, part.content, NumberPart.TAG_OPERATOR_UNIT))
            elif part.tag == NumberPart.TAG_OTHER:
                if merge_stack is not None:
                    for m in merge_stack:
                        if m.tag == NumberPart.TAG_OPERATOR:
                            final_parts.append(NumberPart(m.start, m.content, NumberPart.TAG_OPERATOR_UNIT))
                        else:
                            final_parts.append(m)
                    merge_stack = None
                final_parts.append(part)
            else:
                raise NotImplementedError()

        if merge_stack is not None:
            for m in merge_stack:
                if m.tag == NumberPart.TAG_OPERATOR:
                    final_parts.append(NumberPart(m.start, m.content, NumberPart.TAG_OPERATOR_UNIT))
                else:
                    final_parts.append(m)

        assert len(parts) == len(final_parts)
        NumberPart.validate_parts_order(final_parts)

        tmp_parts = final_parts
        final_parts = []
        skip = False

        for index, part in enumerate(tmp_parts):
            if skip:
                skip = False
                continue
            if part.tag == NumberPart.TAG_OPERATOR_UNIT and part.content in ["/", "^"]:
                if index + 1 < len(tmp_parts):
                    next_part = tmp_parts[index + 1]
                    if next_part.tag != NumberPart.TAG_NUMBER or (part.content == "^" and len(next_part.content) == 1):
                        skip = True
                        part = NumberPart.concat_part(part, next_part, new_tag=NumberPart.TAG_OTHER)

                if len(final_parts) == 0:
                    final_parts.append(part)
                else:
                    if final_parts[-1].tag != NumberPart.TAG_NUMBER:
                        final_parts.append(NumberPart.concat_part(final_parts.pop(), part, new_tag=NumberPart.TAG_OTHER))
                    else:
                        final_parts.append(part)
            else:
                final_parts.append(part)

        return NumberPart.validate_parts_order(final_parts)

    @staticmethod
    def extract_parts(text: str, strip_content=True):
        # num_p = re.compile(r"(?<![a-zA-Z0-9])[+\-]?[0-9][\,\.0-9 ]*")
        # num_p = re.compile(r"(?<![a-zA-Z0-9])[+\-]?[0-9](?:[\,\. ]*[0-9])*(?![a-zA-Z])")
        num_p = re.compile(r"(?<![a-zA-Z0-9])[+\-]?[0-9](?:[\,\. ]*[0-9])*")

        num_parts = []
        pos = 0
        while True:
            number = num_p.search(text, pos)
            if not number:
                break
            num_parts.append(NumberPart(number.span()[0], number.group().strip(), NumberPart.TAG_NUMBER))
            pos = number.span()[1]

        op_p_text = "|".join(["\\" + p if p[0] != " " else p for p in BASIC_MATH_OPERATOR_DICT])
        op_p = re.compile(r"(" + op_p_text + r")")

        all_parts = []
        for part in NumberPart.add_outside_parts(text, num_parts):
            if part.tag is not None:
                all_parts.append(part)
                continue

            pos = 0
            while True:
                op = op_p.search(part.content, pos)
                if not op:
                    break
                all_parts.append(NumberPart(part.start + op.span()[0], op.group().strip(), NumberPart.TAG_OPERATOR))
                pos = op.span()[1]

        all_parts = NumberPart.add_outside_parts(text, all_parts, NumberPart.TAG_OTHER)
        all_parts = NumberPart.merge_parts(all_parts)
        if strip_content:
            all_parts = NumberPart.strip_parts(all_parts)

        return NumberPart.validate_parts_content(text, all_parts)

    @staticmethod
    def split_num_and_unit(num_and_unit: str, raise_error=True):
        """
        Two consequence tokens cannot be both not number
        """
        if num_and_unit.strip() == "":
            return "", num_and_unit

        if num_and_unit[0] == "/":
            return "", num_and_unit

        num = "0123456789"
        valid = num + "/.,"
        special_prefix = "+-"
        sep_index = None

        for idx, ch in enumerate(num_and_unit):
            if idx > 0 and ch not in num and num_and_unit[idx - 1] not in num:
                sep_index = idx - 1
                break
            elif ch in special_prefix:
                if idx != 0:
                    sep_index = idx
                    break
            elif ch in valid:
                pass
            else:
                sep_index = idx
                break
        if sep_index is None:
            sep_index = len(num_and_unit)

        while sep_index > 0 and num_and_unit[sep_index - 1] not in num + "/":
            sep_index -= 1
        while sep_index > 0 and num_and_unit[sep_index - 1] == "/":
            sep_index -= 1
        if sep_index < 0:
            if raise_error:
                raise NormalizeError("Invalid: " + num_and_unit)
            else:
                return None, None

        num = num_and_unit[:sep_index]
        unit = num_and_unit[sep_index:]

        if num.strip() == "" and unit[0] in special_prefix and unit[1] in special_prefix:
            if raise_error:
                raise NormalizeError("Invalid: " + num_and_unit)
            else:
                return None, None

        return num, unit

    @staticmethod
    def concat_part(left: "NumberPart", right: "NumberPart", new_tag):
        space = right.start - left.start - len(left.content)
        assert space >= 0
        return NumberPart(
            start=left.start,
            content=left.content + " " * space + right.content,
            tag=new_tag,
        )

    @staticmethod
    def replace_space_in_number(number: str, repl: str = ""):
        new = ""
        for i in range(len(number)):
            n = number[i]
            if n == " " and i - 1 >= 0 and i + 1 < len(number) and number[i - 1].isdigit() and number[i + 1].isdigit():
                new += repl
            else:
                new += n
        return new

    @staticmethod
    def replace_parts(text: str, parts: list[tuple["NumberPart", str]], sort=False):
        if sort:
            parts = sorted(parts, key=lambda x: x[0].start)

        for part, spoken in parts[::-1]:
            end = part.start + len(part.content)
            text = text[: part.start].strip() + " " + spoken + " " + text[end:].strip()
        return text


class NormalizeError(Exception):
    pass


class TagBasedTextNormalizer:
    """
    A text normalizer designed to work with pre-tagged text segments.
    Each method assumes the input text is of the correct entity type
    and directly applies the transformation rule without keyword matching.
    """

    SORTED_UNITS_DICT = {k: UNITS_DICT[k] for k in sorted(UNITS_DICT.keys(), key=lambda x: (len(x), x), reverse=True)}

    # Sort currency dictionaries by length (longest first) to avoid pattern overlap issues
    SORTED_CURRENCY_OVER_ANY_UNITS = {
        k: CURRENCY_OVER_ANY_UNITS[k] for k in sorted(CURRENCY_OVER_ANY_UNITS.keys(), key=lambda x: (len(x), x), reverse=True)
    }
    SORTED_PREFIX_CURRENCY_UNITS = {
        k: PREFIX_CURRENCY_UNITS[k] for k in sorted(PREFIX_CURRENCY_UNITS.keys(), key=lambda x: (len(x), x), reverse=True)
    }
    SORTED_SUFFIX_CURRENCY_UNITS = {
        k: SUFFIX_CURRENCY_UNITS[k] for k in sorted(SUFFIX_CURRENCY_UNITS.keys(), key=lambda x: (len(x), x), reverse=True)
    }

    def __init__(self, verbose=False, simple=False) -> None:
        self.verbose = verbose

        self._tag_to_function: dict[str, Callable[[str, str], tuple[str, bool]]] = {
            # Date & Time
            "DATE": self.normalize_date,
            "DATE_RANGE_y_y": self.normalize_date_range_y_y,
            "DATE_RANGE": self.normalize_date_range,
            "TIME": self.normalize_time,
            "TIME_RANGE": self.normalize_time,
            # Numbers & Quantities
            "INTEGER_n": self.normalize_integer_auto if not simple else self.normalize_integer_simple,
            "INTEGER_big": self.normalize_integer_auto if not simple else self.normalize_integer_simple,
            "FLOAT_n": self.normalize_float_auto if not simple else self.normalize_float_simple,
            "FLOAT_big": self.normalize_float_auto if not simple else self.normalize_float_simple,
            "FRACTION": self.normalize_fraction,
            "NUMBER_RANGE": self.normalize_number_range,
            "MONEY": self.normalize_money,
            "MEASUREMENT": self.normalize_measurement_auto,
            "DIMENSION": self.normalize_dimension,
            "SPORT_SCORE": self.normalize_sport_score,
            "ROMAN_NUMERAL": self.normalize_roman_numeral,
            "MATH_EXPR": self.normalize_math_expr,
            # Identifiers & Codes
            "PHONE": self.normalize_phone,
            "LEGAL_DOC_ID": self.normalize_legal_doc_id,
            "PLATE": self.normalize_plate,
            "ALPHANUM_ID": self.normalize_alphanum_id,
            # Other
            "ADDRESS": self.normalize_address,
            "EMAIL": self.normalize_email,
            "URL": self.normalize_url,
            "FOREIGN_WORD": self.normalize_foreign_word,
        }

        missing = []
        for tag in TAG_TO_PROMPT:
            if "__" not in tag and tag not in self._tag_to_function:
                missing.append(tag)
        if len(missing) > 0:
            # raise ValueError(f"Missing tags: {missing}") # BUG
            pass

        self.rule = RuleBasedTextNormalizer()

        taggers_true = [
            Integer_n_Tagger(False, strict_validate=False),
            Integer_big_Tagger(False, strict_validate=False, ignore_3_sep=True),
            Float_n_Tagger(False, strict_validate=False),
            Float_big_Tagger(False, strict_validate=False, ignore_3_sep=True),
        ]
        taggers_false = [
            Integer_n_Tagger(False, strict_validate=False),
            Integer_big_Tagger(False, strict_validate=False, ignore_3_sep=False),
            Float_n_Tagger(False, strict_validate=False),
            Float_big_Tagger(False, strict_validate=False, ignore_3_sep=False),
        ]
        self.number_taggers: dict[bool, dict[str, BaseTagger]] = {
            True: {t.tag(): t for t in taggers_true},
            False: {t.tag(): t for t in taggers_false},
        }

        self.vi_vocab_sorted = sorted(load_vocab("data/lexicon_vi.tsv"), key=lambda x: (len(x), x), reverse=True)

    def normalize(self, tag: str, text: str, prefix_text: str = ""):
        if tag not in self._tag_to_function:
            raise ValueError(f"Invalid tag `{tag}`")
        spoken, need_infer_again = self._tag_to_function[tag](text, prefix_text)
        return spoken.strip(), need_infer_again

    def _detect_number_type(self, num_str: str, ignore_3_sep: bool):
        return [nt.tag() for nt in self.number_taggers[ignore_3_sep].values() if nt.validate(num_str) == 0]

    def normalize_measurement_auto(
        self, text: str, prefix_text: str = "", num_type: Literal["auto", "integer", "float"] = "auto"
    ):
        """
        Call to:
        - normalize_number_auto
        - normalize_integer_simple
        - normalize_float_simple
        - normalize_unit
        """
        parts = NumberPart.extract_parts(text)
        spoken_texts = []
        need_infer_again = False
        cache_spoken_currency = None

        for index, part in enumerate(parts):
            if part.tag == part.TAG_NUMBER:
                if num_type == "auto":
                    spoken, nia = self.normalize_number_auto(
                        part.content.replace(" ", ""), prefix_text=prefix_text, fallback_measurement=False
                    )
                elif num_type == "integer":
                    spoken, nia = self.normalize_integer_simple(text, prefix_text=prefix_text)
                elif num_type == "float":
                    spoken, nia = self.normalize_float_simple(text, prefix_text=prefix_text)
                else:
                    raise NotImplementedError()

                spoken_texts.append(spoken)
                if nia is True:
                    need_infer_again = True
                if cache_spoken_currency is not None:
                    spoken_texts.append(cache_spoken_currency)
                    cache_spoken_currency = None

            elif part.tag == part.TAG_OPERATOR:
                for ch in part.content:
                    spoken_texts.append(BASIC_NUMBER_OPERATOR_DICT[ch])

            elif part.tag == part.TAG_OPERATOR_UNIT:
                for ch in part.content:
                    spoken_texts.append(BASIC_MATH_OPERATOR_DICT[ch])

            elif part.tag == part.TAG_OTHER:
                spoken_units, nia = self.normalize_unit(part.content)
                if nia is True:
                    need_infer_again = True

                assert cache_spoken_currency is None
                if index + 1 < len(parts) and len(part.content) != 0:
                    next_part = parts[index + 1]
                    if next_part.tag == next_part.TAG_NUMBER and part.start + len(part.content) == next_part.start:
                        for currency, spoken_currency in self.SORTED_PREFIX_CURRENCY_UNITS.items():
                            if part.content.endswith(currency):
                                cache_spoken_currency = spoken_currency.strip()
                                break

                if cache_spoken_currency is None:
                    spoken_texts.extend(spoken_units)
                else:
                    spoken_texts.extend(spoken_units[:-1])
                    spoken_unit = spoken_units[-1]
                    assert spoken_unit.endswith(cache_spoken_currency), f"'{spoken_unit}' not ends with '{cache_spoken_currency}'"
                    spoken_unit = spoken_unit[: -len(cache_spoken_currency)].strip()
                    if spoken_unit != "":
                        spoken_texts.append(spoken_unit)

            else:
                raise NotImplementedError()

        return " ".join(spoken_texts), need_infer_again

    def normalize_money(self, text: str, prefix_text: str) -> tuple[str, bool]:
        return self.normalize_measurement_auto(text, prefix_text)

    def normalize_unit(self, text: str) -> tuple[list[str], bool]:
        spoken_texts = []
        need_infer_again = False

        for unit in text.split():
            unit = unit.strip()

            while len(unit) > 0:
                self.log("normalize_unit: unit =", unit)
                found = False

                # Search in unit dict
                for u, u_spoken in self.SORTED_UNITS_DICT.items():
                    if len(unit) == 0:
                        break
                    if startswith_unit(unit, u) or (u[-1] == " " and u.startswith(unit)):
                        self.log(f"normalize_unit: convert {u} -> {u_spoken}")

                        spoken_texts.append(u_spoken.strip())
                        unit = unit[len(u) :].strip()
                        found = True

                # Check math operator
                if (
                    len(unit) > 0
                    and (unit[0] in FULL_MATH_OPERATOR_DICT or unit[0] in string.punctuation)
                    and unit[0] not in self.SORTED_UNITS_DICT
                ):
                    if unit[0] in BASIC_MATH_OPERATOR_DICT:
                        spoken_texts.append(BASIC_MATH_OPERATOR_DICT[unit[0]])
                    elif unit[0] in NON_SILENT_SYMBOL_DICT:
                        spoken_texts.append(NON_SILENT_SYMBOL_DICT[unit[0]])
                    else:
                        spoken_texts.append(unit[0])
                    unit = unit[1:].strip()
                    found = True

                # Search in vocab
                for word in self.vi_vocab_sorted:
                    if startswith_unit(unit, word):
                        spoken_texts.append(word)
                        unit = unit[len(word) :].strip()
                        found = True

                # Not found
                if found is False:
                    # spoken_texts.append(alpha_num2words(unit))
                    spoken_texts.append(unit)
                    need_infer_again = True
                    break

        return spoken_texts, need_infer_again

    def normalize_number_auto(self, text: str, prefix_text: str = "", fallback_measurement=True) -> tuple[str, bool]:
        """
        Call to:
        - normalize_measurement_auto
        """
        original = text

        text = NumberPart.replace_space_in_number(text)
        num_types = self._detect_number_type(text, False)
        self.log("normalize_number_auto: num_types =", num_types)

        if len(num_types) == 1:
            # convert to spoken text
            num_type = num_types[0]
            if num_type in [Integer_n_Tagger.tag(), Integer_big_Tagger.tag()]:
                return num2words_integer(text, remove_sep=True).strip(), False
            elif num_type in [Float_n_Tagger.tag(), Float_big_Tagger.tag()]:
                return num2words_float(text, prefer_int=False).strip(), False
            else:
                raise NormalizeError()  # guard

        elif len(num_types) == 0:
            # raise NormalizeError(num)
            try:
                return num2words_float(text, prefer_int=True).strip(), False
            except:
                pass

        if fallback_measurement:
            text, need_infer_again = self.normalize_measurement_auto(original, prefix_text)
            self.log("normalize_number_auto: fallback text =", text)
        else:
            need_infer_again = True

        return text, need_infer_again

    def _normalize_with_number_parts(
        self, handle_num_fn: Callable[[NumberPart], tuple[str, bool]], text: str, prefix_text: str = ""
    ) -> tuple[str, bool]:
        parts = NumberPart.extract_parts(text)
        spoken_parts = []
        need_infer_again = False

        for part in parts:
            if part.tag == NumberPart.TAG_OPERATOR:
                spoken_parts.append((part, BASIC_NUMBER_OPERATOR_DICT[part.content]))

            elif part.tag == NumberPart.TAG_OPERATOR_UNIT:
                spoken_parts.append((part, BASIC_MATH_OPERATOR_DICT[part.content]))

            elif part.tag == NumberPart.TAG_OTHER:
                if has_non_silent_char(part.content):
                    spoken, nia = self.normalize_unit(part.content)
                    if nia:
                        need_infer_again = True
                    spoken_parts.append((part, " ".join(spoken)))
                else:
                    # need_infer_again = True
                    spoken_parts.append((part, part.content))

            elif part.tag == NumberPart.TAG_NUMBER:
                spoken, nia = handle_num_fn(part)
                if nia:
                    need_infer_again = True
                spoken_parts.append((part, spoken))

            else:
                raise NotImplementedError()

        return NumberPart.replace_parts(text, spoken_parts), need_infer_again

    def normalize_integer_simple(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_unit

        BUG: should we ensure that the text is always number ?
        """

        def handle_num(part):
            return num2words_integer(part.content, remove_sep=True).strip(), False

        return self._normalize_with_number_parts(handle_num, text, prefix_text)

    def normalize_integer_auto(self, text: str, prefix_text: str = "", fallback_float=True) -> tuple[str, bool]:
        """
        Handle case like 1, 2, 100, 100.000, 200 000

        If got error or failed, fallback to normalize_measurement

        Call to:
        - normalize_unit
        - normalize_float_auto
        - normalize_measurement_auto
        - normalize_integer_simple
        """

        def handle_num(part) -> tuple[str, bool]:
            num_str = part.content.replace(" ", "")
            num_types = self._detect_number_type(num_str, True)
            self.log("normalize_integer_auto: num_types =", num_types)

            if len(num_types) > 0:
                if all_is_in(num_types, [Integer_n_Tagger.tag(), Integer_big_Tagger.tag()]):
                    return num2words_integer(num_str, remove_sep=True).strip(), False

                elif fallback_float and all_is_in(num_types, [Float_n_Tagger.tag(), Float_big_Tagger.tag()]):
                    # BUG: thinking carefully at this fallback, should move into the elif or stays like at present
                    return self.normalize_float_auto(num_str, prefix_text, fallback_integer=False)

                elif any_is_in(num_types, [Float_n_Tagger.tag(), Float_big_Tagger.tag()]):
                    return num2words_float(num_str, prefer_int=True, prefer_comma_as_int_sep=True).strip(), False

                else:
                    spoken, nia = self.normalize_measurement_auto(num_str, prefix_text)
                    self.log("normalize_integer_auto: fallback text =", num_str)

                    if nia is False:
                        return spoken, False
                    else:
                        try:
                            return self.normalize_integer_simple(num_str, prefix_text)
                        except:
                            return num_str, nia

            else:
                return part.content, True

        return self._normalize_with_number_parts(handle_num, text, prefix_text)

    def normalize_float_simple(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_unit
        """

        def handle_num(part):
            return num2words_float(part.content, prefer_int=False).strip(), False

        return self._normalize_with_number_parts(handle_num, text, prefix_text)

    def normalize_float_auto(self, text: str, prefix_text: str = "", fallback_integer=True) -> tuple[str, bool]:
        """
        Call to:
        - normalize_unit
        - normalize_integer_auto
        - normalize_measurement_auto
        - normalize_float_simple
        """

        def handle_num(part) -> tuple[str, bool]:
            num_str = part.content.replace(" ", "")
            num_types = self._detect_number_type(num_str, True)
            self.log("normalize_float_auto: num_types =", num_types)

            if len(num_types) > 0:
                if fallback_integer and all_is_in(num_types, [Integer_n_Tagger.tag(), Integer_big_Tagger.tag()]):
                    return self.normalize_integer_auto(num_str, prefix_text, fallback_float=False)

                elif any_is_in(num_types, [Float_n_Tagger.tag(), Float_big_Tagger.tag()]):
                    return num2words_float(num_str, prefer_int=False).strip(), False

                else:
                    spoken, nia = self.normalize_measurement_auto(num_str, prefix_text)
                    self.log("normalize_float_auto: fallback text =", num_str)

                    if nia is False:
                        return spoken, False
                    else:
                        try:
                            return self.normalize_float_simple(num_str, prefix_text)
                        except:
                            return num_str, nia
            else:
                return part.content, True

        return self._normalize_with_number_parts(handle_num, text, prefix_text)

    def normalize_fraction(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_measurement_auto
        """
        parts = text.strip().replace(":", "/").split("/")
        need_infer_again = False
        spoken_text = []
        for index, part in enumerate(parts):
            part_spoken, nia = self.normalize_measurement_auto(part, prefix_text=prefix_text if index == 0 else "")
            spoken_text.append(part_spoken.strip())
            if nia:
                need_infer_again = True

        return " phần ".join(spoken_text), need_infer_again

    def normalize_dimension(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_measurement_auto
        """
        encode = {}
        for unit in self.SORTED_UNITS_DICT.keys():
            if "x" in unit.lower():
                cnt = 1
                while True:
                    enc = f"ENC{len(encode)}Z{cnt}"
                    if enc not in text:
                        break
                    cnt += 1

                text = re.sub(unit, enc, text, flags=re.IGNORECASE)
                encode[enc] = unit

        for enc, unit in encode.items():
            text = text.replace(enc, " " + self.SORTED_UNITS_DICT[unit] + " ")

        parts = text.strip().split("x")
        need_infer_again = False
        spoken_text = []
        for index, part in enumerate(parts):
            part_spoken, nia = self.normalize_measurement_auto(part, prefix_text=prefix_text if index == 0 else "")
            spoken_text.append(part_spoken.strip())
            if nia:
                need_infer_again = True

        return " nhân ".join(spoken_text), need_infer_again

    def normalize_number_range(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_measurement_auto
        """
        parts = text.strip().split("-")
        spoken_text = []
        need_infer_again = False

        for index, part in enumerate(parts):
            part_spoken, nia = self.normalize_measurement_auto(part, prefix_text=prefix_text if index == 0 else "")
            spoken_text.append(part_spoken.strip())
            if nia:
                need_infer_again = True

        return " đến ".join(spoken_text), need_infer_again

    def normalize_math_expr(self, text: str, prefix_text: str) -> tuple[str, bool]:
        return self.normalize_measurement_auto(text, prefix_text)

    def normalize_sport_score(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_integer_auto
        """
        parts = text.split("-")
        spoken_parts = []
        need_infer_again = False

        for part in parts:
            spoken, nia = self.normalize_integer_auto(part.strip())
            spoken_parts.append(spoken.strip())
            if nia:
                need_infer_again = True

        return " ".join(spoken_parts), need_infer_again

    def _speak_date_part(self, part: str):
        if has_only_number(part):
            return num2words_integer(part).strip()
        return alpha_num2words(part).strip()

    def _get_day_prefix(self, prefix_text: str):
        add_day_prefix = True
        support_day_lt10 = True
        if is_in_last_words(prefix_text, 2, ["ngày", "hôm"]):
            add_day_prefix = False
        if is_in_last_words(prefix_text, 1, ["mồng", "mùng"]):
            add_day_prefix = False
            support_day_lt10 = False
        return dict(add_day_prefix=add_day_prefix, support_day_lt10=support_day_lt10)

    def _speak_dmy(self, d: str, m: str, y: str | None, prefix_text: str):
        kwargs = self._get_day_prefix(prefix_text)

        dmy_str = self._speak_date_part(d) + " tháng " + self._speak_date_part(m)
        if y is not None:
            dmy_str += " năm " + self._speak_date_part(y)
        if kwargs["support_day_lt10"] and has_only_number(d) and 0 < int(d) < 10:
            dmy_str = "mùng " + dmy_str
        if kwargs["add_day_prefix"]:
            dmy_str = "ngày " + dmy_str
        return dmy_str

    def _speak_qy(self, q: str, y: str | None, prefix_text: str):
        qy_str = self._speak_date_part(q)
        if y is not None:
            qy_str += " năm " + self._speak_date_part(y)

        if is_in_last_words(prefix_text, 1, ["quý"], ignore_linking_words=False, ignore_punctuation=False) is False:
            qy_str = "quý " + qy_str

        return qy_str

    def _speak_my(self, m: str, y: str, prefix_text: str):
        my_str = self._speak_date_part(m) + " năm " + self._speak_date_part(y)

        if is_in_last_words(prefix_text, 1, ["tháng"]) is False:
            my_str = "tháng " + my_str

        return my_str

    def normalize_date(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        connector : / - .

        DM: 6/7, 01/04, 1/04, x/y -> ngày x tháng y

        MY: 4/2024, Q3/2025, x/y -> tháng/quý x năm y

        DMY: dd/mm/yyyy

        Case ngày 1/2 năm 2025
        """

        p_dmy = re.compile(r"([A-z0-9]+)\s*[\/\-\.]\s*([A-z0-9]+)\s*[\/\-\.]\s*([A-z0-9]+)")
        p_dm_or_my = re.compile(r"([A-z0-9]+)\s*[\/\-\.]\s*([A-z0-9]+)")
        p_q = re.compile(r"Q([A-z0-9]+)")

        dates = []
        start = 0
        while True:
            dmy_pattern = p_dmy.search(text, start)
            dm_or_my_pattern = None
            q_pattern = None
            if not dmy_pattern:
                dm_or_my_pattern = p_dm_or_my.search(text, start)
                if not dm_or_my_pattern:
                    q_pattern = p_q.search(text, start)
                    if not q_pattern:
                        break

            if dmy_pattern is not None:
                start = dmy_pattern.span()[1]
                last_end = dates[-1][1][1] if len(dates) > 0 else None
                prefix_text += text[last_end : dmy_pattern.span()[0]]

                d, m, y = dmy_pattern.group(1), dmy_pattern.group(2), dmy_pattern.group(3)
                dates.append((self._speak_dmy(d, m, y, prefix_text), dmy_pattern.span()))

            if dm_or_my_pattern is not None:
                start = dm_or_my_pattern.span()[1]
                last_end = dates[-1][1][1] if len(dates) > 0 else None
                prefix_text += text[last_end : dm_or_my_pattern.span()[0]]

                self.log(f"normalize_date: prefix_text = `{prefix_text}`")

                d, m = dm_or_my_pattern.group(1), dm_or_my_pattern.group(2)
                if d[0].upper() == "Q":
                    dates.append((self._speak_qy(d[1:], m, prefix_text), dm_or_my_pattern.span()))

                elif len(m) >= 4:
                    if is_in_last_words(prefix_text, 1, ["quý"], ignore_linking_words=True, ignore_punctuation=True):
                        dates.append((self._speak_qy(d, m, prefix_text), dm_or_my_pattern.span()))
                    else:
                        dates.append((self._speak_my(d, m, prefix_text), dm_or_my_pattern.span()))
                else:
                    if (
                        is_in_last_words(prefix_text, 3, ["tháng"], ignore_linking_words=True, ignore_punctuation=True)
                        and is_in_last_words(prefix_text, 3, ["ngày"], ignore_linking_words=True, ignore_punctuation=True)
                        is False
                    ):
                        dates.append((self._speak_my(d, m, prefix_text), dm_or_my_pattern.span()))
                    else:
                        dates.append((self._speak_dmy(d, m, None, prefix_text), dm_or_my_pattern.span()))

            if q_pattern is not None:
                start = q_pattern.span()[1]
                last_end = dates[-1][1][1] if len(dates) > 0 else None
                prefix_text += text[last_end : q_pattern.span()[0]]

                dates.append((self._speak_qy(q_pattern.group(1), None, prefix_text), q_pattern.span()))

        for spoken, (s, e) in dates[::-1]:
            text = text[:s] + spoken + text[e:]

        need_infer_again = False
        for ch in text:
            if ch.isdigit() or ch in "/.-":
                need_infer_again = True
                break

        return text, need_infer_again

    def _split_date_range(self, text: str):
        C = "-"
        parts = None
        for c in [" " + C + " ", C + " ", " " + C, C]:
            if c in text:
                parts = text.split(c)
                break
        if parts is None:
            parts = [text]
        parts = [s.strip() for s in parts]
        return parts

    def normalize_date_range(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        parts = self._split_date_range(text)
        spokens = []
        need_infer_again = False

        for p in parts:
            spk, nia = self.normalize_date(p, prefix_text=prefix_text)
            spokens.append(spk)
            if nia:
                need_infer_again = True

        return " đến ".join(spokens), need_infer_again

    def normalize_date_range_y_y(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_integer_auto
        """
        parts = text.split("-")
        spoken_parts = []
        need_infer_again = False

        for part in parts:
            spoken, nia = self.normalize_integer_auto(part.strip())
            if nia:
                need_infer_again = True
            spoken_parts.append(spoken.strip())

        return " ".join(spoken_parts), need_infer_again

    def normalize_time(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to
        - normalize_fraction
        - normalize_number_auto
        """
        TIME_REGEX = (
            r"(\d+(?:[\.\,\/]\d+)?)(\:|[hg])(\d+(?:[\.\,\/]\d+)?)(\:|[mp])(\d+(?:[\.\,\/]\d+)?)(s?)|"
            r"(\d+(?:[\.\,\/]\d+)?)(\:|[hg])(\d+(?:[\.\,\/]\d+)?)([mps]?)|"
            r"(\d+(?:[\.\,\/]\d+)?)(\:|[mp])(\d+(?:[\.\,\/]\d+)?)([s]?)|"
            r"(\d+(?:[\.\,\/]\d+)?)[hgmps]"
        )
        p = re.compile(TIME_REGEX)

        need_infer_again = False

        times = []
        start = 0
        while start < len(text):
            time_pattern = p.search(text, start)
            if not time_pattern:
                break

            times.append((time_pattern.group(), time_pattern.span()))
            start = time_pattern.span()[1]

        unit_map = {
            "h": ("giờ", "p"),
            "g": ("giờ", "p"),
            "p": ("phút", "s"),
            "m": ("phút", "s"),
            "s": ("giây", None),
        }
        units = "hgmps"

        for time_str, (s, e) in times[::-1]:
            # BUG: Cannot handle case "1.000.000,3h"

            if "/" in time_str:
                parts = split_by_chars(time_str, units)
                spoken_text = ""
                next_unit = None

                for part in parts:
                    if "/" in part:
                        spoken, nia = self.normalize_fraction(part)
                        if nia:
                            need_infer_again = True

                    elif part in units:
                        spoken, next_unit = unit_map[part]

                    else:
                        spoken, nia = self.normalize_number_auto(part)
                        if nia:
                            need_infer_again = True

                    spoken_text += spoken + " "

                if parts[-1] not in units and next_unit is not None:
                    spoken_text += unit_map[next_unit][0]

                spoken_text = spoken_text.strip()

            else:
                spoken_text = time2words(time_str)

            text = text[:s] + spoken_text + text[e:]

        for ch in text:
            if ch.isdigit() or ch in "/.-":
                need_infer_again = True
                break

        return text, need_infer_again

    def normalize_time_range(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        """
        Call to:
        - normalize_integer_auto
        """
        parts = text.split("-")
        spoken_parts = []
        need_infer_again = False

        for part in parts:
            spoken, nia = self.normalize_time(part.strip())
            if nia:
                need_infer_again = True
            spoken_parts.append(spoken.strip())

        return " đến ".join(spoken_parts), need_infer_again

    def normalize_alphanum_id(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        original = text
        text = self.normalize_abbreviation(text)
        if text != original:
            return text, True
        parts = [alpha_num2words(p).strip() for p in text.split()]
        return " ".join(parts), False

    def normalize_foreign_word(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        return self.normalize_abbreviation(text), False

    def normalize_roman_numeral(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        p = re.compile(r"(" r"(X{0,3})(IX|IV|VI{0,3}|V|I{1,3})|" r"(x{0,3})(ix|iv|vi{0,3}|v|i{1,3})" r")")
        roman_list = []
        need_infer_again = False
        start = 0

        while True:
            roman_pattern = p.search(text, start)
            if not roman_pattern:
                break
            # print(roman_pattern.group())
            roman_list.append((roman_pattern.group(), roman_pattern.span()))
            start = roman_pattern.span()[1]

        for rom, (s, e) in roman_list[::-1]:
            try:
                rom_str = num2words_integer(fromRoman(rom)).strip()
            except InvalidRomanNumeralError:
                rom_str = rom
                need_infer_again = True
            text = text[:s].rstrip() + " " + rom_str + " " + text[e:].lstrip()

        return text, need_infer_again

    def normalize_phone(self, text: str, prefix_text: str = ""):
        phone = "".join([ch for ch in text if ch in "1234567890"])
        return phone2words(phone).strip(), False

    def normalize_plate(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        # Handle license plates like "29A-12345"
        text = text.replace("-", " ").replace(".", "").strip()
        text = re.sub(r"\s+", ",", text)
        return (
            " , ".join([alpha_num2words(p.strip(), num_mode="shortest").strip() for p in text.split(",") if p.strip() != ""]),
            False,
        )

    def normalize_legal_doc_id(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        # Example: 110/2013/NĐ-CP -> một trăm mười năm hai không mười ba nờ đê cê pê
        # The second number is year
        parts = text.split("/")
        parts = [p.replace("-", "").strip() for p in parts]
        parts = [p for p in parts if p != ""]
        spoken_parts = []
        has_year = False
        cnt_num = 0
        for p in parts:
            if has_only_number(p):
                cnt_num += 1
                is_year = False
                if cnt_num >= 2 and len(p) == 4 and int(p) >= 1945 and has_year is False:
                    is_year = True
                    has_year = True
                spoken = num2words_integer(p).strip()
                if is_year:
                    spoken = "năm " + spoken
                spoken_parts.append(spoken)

            else:
                spoken_parts.append(alpha_num2words(p).strip())

        return " , ".join(spoken_parts), False

    def normalize_address(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        # Handle common address abbreviations
        text = text.replace("TDP.", "tổ dân phố ")
        text = text.replace("tdp.", "tổ dân phố ")
        text = text.replace("Tdp.", "tổ dân phố ")

        text = text.replace("TP.", "thành phố ")
        text = text.replace("tp.", "thành phố ")
        text = text.replace("Tp.", "thành phố ")

        text = text.replace("TT.", "thị trấn ")
        text = text.replace("Tt.", "thị trấn ")
        text = text.replace("tt.", "thị trấn ")

        text = text.replace("P.", "phường ")
        text = text.replace("PH.", "phường ")
        text = text.replace("Ph.", "phường ")
        text = text.replace("F.", "phường ")

        text = text.replace("Q.", "quận ")
        text = text.replace("q.", "quận ")

        text = text.replace("X.", "xã ")
        text = text.replace("x.", "xã ")

        if has_number(text) or "/" in text:
            parts = text.split("/")
            spoken_parts = [alpha_num2words(p.strip(), num_mode="full").strip() for p in parts if p.strip() != ""]
            text = " trên ".join(spoken_parts)

        text = re.sub(r"\s+", " ", text)

        return text, False

    def normalize_email(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        parts = split_by_chars(text, "".join(list(SYMBOL_DICT.keys())))
        new_parts = []
        for part in parts:
            if part in SYMBOL_DICT:
                new_parts.append(SYMBOL_DICT[part])
            elif part in EMAIL_AND_URL_COMPONENT_DICT:
                new_parts.append(EMAIL_AND_URL_COMPONENT_DICT[part])
            else:
                words = break_word(part)
                for word in words:
                    if word in EMAIL_AND_URL_COMPONENT_DICT:
                        new_parts.append(EMAIL_AND_URL_COMPONENT_DICT[word])
                    else:
                        new_parts.append(word)
        return " ".join(new_parts), False

    def normalize_url(self, text: str, prefix_text: str = "") -> tuple[str, bool]:
        return self.normalize_email(text, prefix_text)

    def normalize_abbreviation(self, text: str):
        """
        Normalize abbreviations with standard dictionary
        """
        return RuleBasedTextNormalizer.normalize_abbreviation(text)

    def log(self, *values, sep=" ", end="\n"):
        if self.verbose:
            print(*values, sep=sep, end=end)


class AutoTextNormalizer:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "microsoft/Multilingual-MiniLM-L12-H384",
        label_matching: LabelMatching = LabelMatching.MATCH_ALL_TAGS,
        rule_based_fallback=True,
        spacing_puncs=True,
        max_batch_size=64,
        max_iterations=8,
        verbose: Literal["none", "logger", "stdout"] = "none",
        device="cuda",
    ):
        self.rule_based_fallback = rule_based_fallback
        self.verbose = verbose
        self.spacing_puncs = spacing_puncs
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations

        self.rule_based_norm = RuleBasedTextNormalizer()
        self.tag_based_norm = TagBasedTextNormalizer()
        self.tagger = TaggerInference(
            model_path, tokenizer_name, label_matching=label_matching, device=device, max_batch_size=max_batch_size
        )

    def normalize(self, texts: list[str], norm_puncs=False) -> list[str]:
        results = []
        for start in range(0, len(texts), self.max_batch_size):
            batch = texts[start : start + self.max_batch_size]
            results.extend(self._normalize(batch, norm_puncs=norm_puncs))
        return results

    def preprocess(self, text: str):
        text = text.strip().replace("\n", " , ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        text = self.rule_based_norm.remove_emoji(text)
        text = self.rule_based_norm.remove_special_characters(text)
        return text

    def _normalize(self, texts: list[str], norm_puncs=False):
        texts = [self.preprocess(text) for text in texts]

        normalized_texts = []
        for text in texts:
            if self.rule_based_fallback:
                normalized_texts.append(self.rule_based_norm.normalize(text, norm_puncs=norm_puncs, spacing_puncs=False))
            else:
                normalized_texts.append(None)

        need_infer_indexes = []
        for idx, (text, normed) in enumerate(zip(texts, normalized_texts)):
            if text != normed:
                need_infer_indexes.append(idx)

        # BUG: Must add max steps here
        if len(need_infer_indexes) > 0:
            normalized_texts = [t for t in texts]

            try:
                for step in range(self.max_iterations):
                    if len(need_infer_indexes) == 0:
                        break

                    input_texts = [normalized_texts[i] for i in need_infer_indexes]
                    outputs = self.tagger.inference(input_texts)

                    num_unchanged_texts = 0
                    new_need_infer_indexes = []

                    for ni_idx, output in zip(need_infer_indexes, outputs):
                        text = output.content

                        self._log(f"Current text: '{text}'", step=step)
                        self._log(f"Patterns: {output.patterns}", step=step)
                        self._log(f"{output.token_level_outputs}", step=step)
                        self._log(convert_to_raw_content(text, output.patterns))
                        self._log("")

                        need_infer_again = False
                        prefix_text = ""
                        replacements = []
                        for ip, pattern in enumerate(output.patterns):
                            # With case that the spoken text is depended on prefix text like date, for example
                            #   in case "quý 3/2024 , 4/2024 và 5/2025", we need the word "quý" at the first place
                            #   for the second and the third dates
                            if ip == 0:
                                prefix_text = text[: pattern.start]
                            else:
                                prev = output.patterns[ip - 1]
                                prefix_text += " " + text[prev.end : pattern.start]

                            spoken, nia = self.tag_based_norm.normalize(pattern.tag, pattern.content, prefix_text=prefix_text)
                            replacements.append((spoken, pattern))
                            if nia:
                                need_infer_again = True

                        new_text = text
                        for spoken, pattern in replacements[::-1]:
                            new_text = new_text[: pattern.start].rstrip() + " " + spoken + " " + new_text[pattern.end :].lstrip()
                        new_text = new_text.strip()

                        self._log(f"New text: '{new_text}'", step=step)
                        self._log(f"need_infer_again = {need_infer_again}", step=step)
                        self._log("--------")

                        if text == new_text:
                            num_unchanged_texts += 1

                        normalized_texts[ni_idx] = new_text
                        if need_infer_again:
                            new_need_infer_indexes.append(ni_idx)

                    self._log("num_unchanged_texts =", num_unchanged_texts, step=step)

                    if num_unchanged_texts == len(need_infer_indexes) and need_infer_again:
                        break

                    need_infer_indexes = new_need_infer_indexes

                    self._log("need_infer_indexes =", need_infer_indexes, step=step)
                    self._log("----***----")

            except:
                logger.exception(f"[AutoTextNormalizer]: ERROR at tagger inference. texts = {texts}")

        if self.rule_based_fallback is False:
            normalized_texts = [RuleBasedTextNormalizer.normalize_abbreviation(text) for text in normalized_texts]
            return normalized_texts

        return [
            self.rule_based_norm.normalize(text, norm_puncs=norm_puncs, spacing_puncs=self.spacing_puncs)
            for text in normalized_texts
        ]

    def _log(self, *values, sep=" ", end="\n", step: int = None):
        prefix = "[AutoTextNormalizer] -"
        if step is not None:
            prefix += f" [step={step}] -"
        if self.verbose == "stdout":
            print(prefix, *values, sep=sep, end=end)
        elif self.verbose == "logger":
            logger.debug(prefix + " " + sep.join([str(v) for v in values]))
