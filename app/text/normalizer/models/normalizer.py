import re
import logging
from typing import Callable
from roman import fromRoman

from utils.text.tagger.prompts.elite import TAG_TO_PROMPT
from utils.text.normalizer.rule.cores import (
    date_dmy2words,
    date_dm2words,
    date_my2words,
    time2words,
    num2words_float,
    num2words_integer,
    phone2words,
    alpha_num2words,
    break_word,
)
from utils.text.normalizer.rule.utils.units import (
    UNITS_DICT, 
    CURRENCY_OVER_ANY_UNITS,
    PREFIX_CURRENCY_UNITS,
    SUFFIX_CURRENCY_UNITS,
)
from utils.text.normalizer.rule.utils.characters import (
    FULL_MATH_OPERATOR_DICT,
    EMAIL_AND_URL_COMPONENT_DICT,
    SYMBOL_DICT,
)
from utils.text.normalizer.rule.normalizer import RuleBasedTextNormalizer
from utils.text.normalizer.models.inference import TaggerInference


logger = logging.getLogger(__name__)


def endswith(text: str, suffixes: list[str]):
    for suf in suffixes:
        if text.endswith(suf):
            return True
    return False


def get_last_words(text: str, n: int):
    assert n > 0
    ws = text.split(" ")
    return ws[-n:]


def check_last_words(text: str, n: int, words: list[str]):
    for w in get_last_words(text, n):
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


def split(text: str, seps: str):
    parts = [""]
    for ch in text:
        if ch in seps:
            parts.append(ch)
            parts.append("")
        else:
            parts[-1] += ch
    while len(parts) > 0 and parts[0] == "":
        parts = parts[1:]
    while len(parts) > 0 and parts[-1] == "":
        parts = parts[:-1]
    return parts


def strip_single(text: str, sub: str):
    for ch in sub:
        if text.startswith(ch):
            text = text.lstrip(ch)
            break
    for ch in sub:
        if text.endswith(ch):
            text = text.rstrip(ch)
            break
    return text


def has_number(text: str):
    for ch in text:
        if ch in "0123456789":
            return True
    return False


class NormalizeError(Exception):
    pass


class TagBasedTextNormalizer():
    """
    A text normalizer designed to work with pre-tagged text segments.
    Each method assumes the input text is of the correct entity type
    and directly applies the transformation rule without keyword matching.
    """
    
    SORTED_UNITS_DICT = {k: UNITS_DICT[k] for k in sorted(UNITS_DICT.keys(), key=lambda x: (len(x), x), reverse=True)}
    
    # Sort currency dictionaries by length (longest first) to avoid pattern overlap issues
    SORTED_CURRENCY_OVER_ANY_UNITS = {k: CURRENCY_OVER_ANY_UNITS[k] for k in sorted(CURRENCY_OVER_ANY_UNITS.keys(), key=lambda x: (len(x), x), reverse=True)}
    SORTED_PREFIX_CURRENCY_UNITS = {k: PREFIX_CURRENCY_UNITS[k] for k in sorted(PREFIX_CURRENCY_UNITS.keys(), key=lambda x: (len(x), x), reverse=True)}
    SORTED_SUFFIX_CURRENCY_UNITS = {k: SUFFIX_CURRENCY_UNITS[k] for k in sorted(SUFFIX_CURRENCY_UNITS.keys(), key=lambda x: (len(x), x), reverse=True)}
    

    def __init__(self) -> None:
        self._tag_to_function: dict[str, Callable[[str, str], tuple[str, bool]]] = {
            # Date & Time
            "DATE_dm": self.normalize_date_dm,
            "DATE_dmy": self.normalize_date_dmy,
            "DATE_my": self.normalize_date_my,
            
            "DATE_RANGE_y_y": self.normalize_date_range_y_y,
            "DATE_RANGE_dm_dmy": self.normalize_date_range_dm_dmy,
            "DATE_RANGE_m_my": self.normalize_date_range_m_my,
            
            "TIME_hm": self.normalize_time,
            "TIME_hms": self.normalize_time,
            "TIME_h": self.normalize_time,
            "TIME_RANGE": self.normalize_time,

            # Numbers & Quantities
            # "INTEGER_n": self.normalize_integer,
            # "INTEGER_big": self.normalize_integer,
            "INTEGER_n": self.normalize_measurement,
            "INTEGER_big": self.normalize_measurement,
            
            # "FLOAT_n": self.normalize_float,
            # "FLOAT_big": self.normalize_float,
            "FLOAT_n": self.normalize_measurement,
            "FLOAT_big": self.normalize_measurement,

            "FRACTION": self.normalize_fraction,
            "NUMBER_RANGE": self.normalize_number_range,
            "MONEY": self.normalize_money,
            "MEASUREMENT": self.normalize_measurement,
            "DIMENSION": self.normalize_dimension,
            "SCORE": self.normalize_score,
            "ROMAN_NUMERAL": self.normalize_roman_numeral,
            "MATH_EXPR": self.normalize_math_expr,

            # Identifiers & Codes
            "PHONE": self.normalize_phone,
            "ID_NUMBER": self.normalize_id_number,
            "LEGAL_DOC_ID": self.normalize_legal_doc_id,
            "PLATE": self.normalize_plate,
            "ALPHANUM_ID": self.normalize_alphanum_id,

            # Other
            "ADDRESS": self.normalize_address,
            "EMAIL": self.normalize_email,
            "URL": self.normalize_url,
        }
    
        missing = []
        for tag in TAG_TO_PROMPT:
            if "__" not in tag and tag not in self._tag_to_function:
                missing.append(tag)
        if len(missing) > 0:
            raise ValueError(f"Missing tags: {missing}")
        
        self.rule = RuleBasedTextNormalizer()
        
    def normalize(self, tag: str, text: str, prefix_text: str = ""):
        if tag not in self._tag_to_function:
            raise ValueError(f"Invalid tag `{tag}`")
        
        text = strip_single(text.strip(), ",.?'\"`")
        if tag != "MATH_EXPR":
            # ! có thể là giai thừa
            text = strip_single(text, "[](){}!")
        
        return self._tag_to_function[tag](text, prefix_text)
        
    def _replace_space_in_number(self, number: str, repl: str=""):
        new = ""
        for i in range(len(number)):
            n = number[i]
            if (
                n == " " and i - 1 >= 0 and i + 1 < len(number) 
                and number[i - 1].isdigit() and number[i + 1].isdigit()
            ):
                new += repl
            else:
                new += n
        return new
        
    def normalize_integer(self, text: str, prefix_text: str = ""):
        text = self._replace_space_in_number(text)
        text = text.replace(".", "").replace(",", "").replace("_", "")
        return num2words_integer(text), False
    
    def normalize_float(self, text: str, prefix_text: str = ""):
        text = self._replace_space_in_number(text)
        text = text.replace("_", "")
        return num2words_float(text, prefer_int=False), False
    
    def _get_date_kwargs(self, prefix_text: str):
        add_day_prefix = True
        support_day_lt10 = True
        if check_last_words(prefix_text, 2, ["ngày", "hôm"]):
            add_day_prefix = False
        if check_last_words(prefix_text, 1, ["mồng", "mùng"]):
            add_day_prefix = False
            support_day_lt10 = False
        return dict(add_day_prefix=add_day_prefix, support_day_lt10=support_day_lt10)

    def normalize_date_dm(self, text: str, prefix_text: str = ""):
        return date_dm2words(text, **self._get_date_kwargs(prefix_text)).strip(), False

    def normalize_date_dmy(self, text: str, prefix_text: str = ""):
        return date_dmy2words(text, **self._get_date_kwargs(prefix_text)).strip(), False

    def normalize_date_my(self, text: str, prefix_text: str = ""):
        date_str = date_my2words(text, add_month_prefix=False)
        # if endswith(prefix_text, 1, ["tháng", "quý", ])
        return date_str.strip(), False
    
    def _find_date_format(self, text: str):
        text = text.replace("-", "/").replace(".", "/")
        cnt = text.count("/")
        if cnt == 0:
            return "INTEGER_n"
        if cnt == 2:
            return "DATE_dmy"
        if cnt == 1:
            _, m = text.split("/")
            try:
                m = int(m)
            except:
                raise NormalizeError()
            if m >= 100:
                return "DATE_my"
            else:
                return "DATE_dm"
        raise NormalizeError()

    def normalize_date_range_y_y(self, text: str, prefix_text: str = ""):
        left, right = split_parts(text, "-", 2)
        return num2words_integer(left) + " " + num2words_integer(right), True
    
    def normalize_date_range_dm_dmy(self, text: str, prefix_text: str = ""):
        # Handle date range like "15/3 - 20/4/2024"
        start_date, end_date = split_parts(text, "-", 2)
        start_fmt = self._find_date_format(start_date)
        end_fmt = self._find_date_format(end_date)
        
        start_date = self._tag_to_function[start_fmt](start_date, prefix_text)[0]
        if start_fmt in ["DATE_my", "INTEGER_n"]:
            if check_last_words(prefix_text, 1, ["ngày", "tháng", "quý", "quí"]):
                date_range = start_date
            else:
                date_range = "tháng " + start_date
        else:
            if check_last_words(prefix_text, 1, ["ngày", "tháng", "quý", "quí"]):
                add_prefix = False
            date_range = start_date
            
        date_range += " đến "
        
        end_date = self._tag_to_function[end_fmt](end_date, "")[0]
        if end_fmt in ["DATE_my", "INTEGER_n"]:
            if check_last_words(prefix_text, 1, ["ngày", "tháng", "quý", "quí"]):
                date_range += get_last_words(prefix_text, 1)[0] + " " + end_date
            else:
                date_range += "tháng " + end_date
        else:
            date_range += end_date
        
        add_prefix = True
        if prefix_text.strip() == "":
            add_prefix = True
        elif check_last_words(prefix_text, 1, ["ngày", "tháng", "quý", "quí"]):
            add_prefix = False

        if add_prefix and check_last_words(prefix_text, 3, ["từ"]) is False:
            date_range = "từ " + date_range

        return date_range.strip(), True

    def normalize_date_range_m_my(self, text: str, prefix_text: str = ""):
        return self.normalize_date_range_dm_dmy(text, prefix_text)

    def normalize_time(self, text: str, prefix_text: str = ""):
        return time2words(text).strip(), False

    def normalize_fraction(self, text: str, prefix_text: str = ""):
        parts = text.replace(":", "/").split('/')
        return " phần ".join(parts), True

    def normalize_number_range(self, text: str, prefix_text: str = ""):
        start_num, end_num = split_parts(text, "-", num_parts=2)
        range_str = start_num + " đến " + end_num
        if check_last_words(prefix_text, 2, ["từ"]) is False:
            range_str = "từ " + range_str
        return range_str, True
    
    def normalize_dimension(self, text: str, prefix_text: str = ""):
        x_unit = ["px", "pixel", "lux", "lx"]
        for u in x_unit:
            text = text.replace(u, u.replace("x", "__"))
        parts = split_parts(text, "x", num_parts=None)
        text = " nhân ".join(parts)
        for u in x_unit:
            text = text.replace(u.replace("x", "__"), u)
        return text, True
    
    def normalize_score(self, text: str, prefix_text: str = ""):
        text = text.replace(":", "-")
        parts = split_parts(text, "-", 2)
        return num2words_float(parts[0]) + " " + num2words_float(parts[1]), False

    def normalize_money(self, text: str, prefix_text: str = ""):
        # Handle various money formats like $100, 100đ, $100/CP, etc.
        original_text = text
        
        # Handle complex patterns like $100/CP - check full patterns first (longest patterns first)
        for pattern, replacement in self.SORTED_CURRENCY_OVER_ANY_UNITS.items():
            if text == pattern:
                return replacement.strip(), False  # Simple pattern replacement, no retry needed
        
        # Handle prefix currency patterns like $100, $100/CP (longest prefixes first)
        for prefix, currency_word in self.SORTED_PREFIX_CURRENCY_UNITS.items():
            if text.startswith(prefix):
                remaining_text = text[len(prefix):]
                
                # Check if there's a unit pattern like /CP in the remaining text
                if "/" in remaining_text:
                    parts = remaining_text.split("/", 1)
                    number_part = parts[0]
                    unit_part = "/" + parts[1]
                    
                    # Look for currency + unit pattern (e.g. $/CP matches $100/CP)
                    target_pattern = prefix + unit_part  # e.g. "$/CP"
                    if target_pattern in self.SORTED_CURRENCY_OVER_ANY_UNITS:
                        unit_replacement = self.SORTED_CURRENCY_OVER_ANY_UNITS[target_pattern]
                        if number_part:
                            try:
                                # Convert number to words
                                if "." in number_part or "," in number_part:
                                    normalized_number = self.normalize_float(number_part)[0]
                                else:
                                    normalized_number = self.normalize_integer(number_part)[0]
                                # For $100/CP -> "một trăm đô la trên cổ phiếu"
                                result = (normalized_number + unit_replacement).strip()
                                result = re.sub(r'\s+', ' ', result)  # Remove redundant whitespace
                                return result, False
                            except:
                                pass
                
                # If no unit pattern, just handle as simple prefix currency
                number_match = ""
                for char in remaining_text:
                    if char.isdigit() or char in ".,":
                        number_match += char
                    else:
                        break
                
                if number_match:
                    try:
                        # Convert number to words
                        if "." in number_match or "," in number_match:
                            normalized_number = self.normalize_float(number_match)[0]
                        else:
                            normalized_number = self.normalize_integer(number_match)[0]
                        result = (normalized_number + currency_word).strip()
                        result = re.sub(r'\s+', ' ', result)  # Remove redundant whitespace
                        return result, False
                    except:
                        pass
        
        # Handle simple currency over unit patterns (like đ/CP without prefix) - longest patterns first
        # This handles cases like "1/4$/CP" where we replace $/CP but leave 1/4 for reprocessing
        for pattern, replacement in self.SORTED_CURRENCY_OVER_ANY_UNITS.items():
            if pattern in text:
                text = text.replace(pattern, replacement)
                # Clean up whitespace
                text = text.strip()
                text = re.sub(r'\s+', ' ', text)
                # Return True because the result may contain other entities (like fractions) that need processing
                return text, True
        
        # Handle suffix currencies (like 100đ, 100USD, 100k, 100tr) - longest suffixes first
        for suffix, currency_word in self.SORTED_SUFFIX_CURRENCY_UNITS.items():
            # Handle both with and without space for "đ "
            if text.endswith(suffix) or (suffix.strip() and text.endswith(suffix.strip())):
                if text.endswith(suffix):
                    number_part = text[:-len(suffix)]
                else:
                    number_part = text[:-len(suffix.strip())]
                    
                if number_part:
                    try:
                        # Convert number to words
                        if "." in number_part or "," in number_part:
                            normalized_number = self.normalize_float(number_part)[0]
                        else:
                            normalized_number = self.normalize_integer(number_part)[0]
                        result = (normalized_number + currency_word).strip()
                        result = re.sub(r'\s+', ' ', result)  # Remove redundant whitespace
                        return result, False
                    except:
                        pass
        
        # If no specific pattern matched, return original text
        return self.rule.normalize_money_number(original_text), False
        return original_text, False

    def normalize_measurement(self, text: str, prefix_text: str = ""):
        # Handle measurements like 100kg, 5m, 2.5km, etc.
        original_text = text
        
        # Try to find matching units (longest first to avoid overlap)
        for unit, unit_word in self.SORTED_UNITS_DICT.items():
            # Handle both with and without space (like "l " vs "l")
            if text.endswith(unit) or (unit.strip() and text.endswith(unit.strip())):
                # Determine which version matched
                if text.endswith(unit):
                    number_part = text[:-len(unit)]
                else:
                    number_part = text[:-len(unit.strip())]
                
                if number_part:
                    try:
                        # Convert number to words
                        if "." in number_part or "," in number_part:
                            normalized_number = num2words_float(number_part)
                        else:
                            normalized_number = num2words_integer(number_part)
                        
                        # Combine number + unit
                        result = (normalized_number + unit_word).strip()
                        result = re.sub(r'\s+', ' ', result)  # Clean up whitespace
                        return result, False
                    except:
                        pass
        
        # If no unit pattern matched, return original text
        return original_text, False

    def normalize_roman_numeral(self, text: str, prefix_text: str = ""):
        # Convert roman numerals to Vietnamese words
        try:
            number = fromRoman(text.upper())
            return num2words_integer(str(number)), False
        except:
            return text, False

    def normalize_phone(self, text: str, prefix_text: str = ""):
        phone = "".join([ch for ch in text if ch in "1234567890"])
        return phone2words(phone).strip(), False

    def normalize_id_number(self, text: str, prefix_text: str = ""):
        return alpha_num2words(text), False
    
    def normalize_alphanum_id(self, text: str, prefix_text: str = ""):
        return alpha_num2words(text), False

    def normalize_legal_doc_id(self, text: str, prefix_text: str = ""):
        # Example: 110/2013/NĐ-CP -> một trăm mười năm hai không mười ba nờ đê cê pê
        parts = text.split("/")
        parts = [p.replace("-", "") for p in parts]
        new_parts = []
        for p in parts[:-2]:
            if len(p) > 3:
                new_parts.append(alpha_num2words(p))
            else:
                new_parts.append(alpha_num2words(p, num_mode="longest"))
        new_parts.append("năm " + num2words_integer(parts[-2]))
        new_parts.append(alpha_num2words(parts[-1]))
        return " ".join(new_parts), False

    def normalize_plate(self, text: str, prefix_text: str = ""):
        # Handle license plates like "29A-12345"
        text = text.replace("-", "").replace(".", "").replace(" ", "")
        return alpha_num2words(text, num_mode="shortest"), False

    def normalize_address(self, text: str, prefix_text: str = ""):
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
        
        retry = False
        if "/" in text:
            text = text.replace("/", " trên ")
            retry = True

        text = re.sub(r"\s+", " ", text)

        return text, retry

    def normalize_math_expr(self, text: str, prefix_text: str = ""):
        for k, v in FULL_MATH_OPERATOR_DICT.items():
            text = text.replace(k, " " + v + " ")
        text = re.sub(r"\s+", " ", text)
        return text, True
    
    def normalize_email(self, text: str, prefix_text: str = ""):
        parts = split(text, "".join(list(SYMBOL_DICT.keys())))
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
    
    def normalize_url(self, text: str, prefix_text: str = ""):
        return self.normalize_email(text, prefix_text)


class AutoTextNormalizer:
    def __init__(
        self, 
        model_path: str, 
        tokenizer_name: str="microsoft/mdeberta-v3-base", 
        label_consistency_level=2,
        rule_based_fallback=True, 
        spacing_puncs=True,
        verbose=False,
        device="cuda",
    ):
        self.rule_based_fallback = rule_based_fallback
        self.verbose = verbose
        self.spacing_puncs = spacing_puncs
        self.rule_based_norm = RuleBasedTextNormalizer()
        self.tag_based_norm = TagBasedTextNormalizer()
        self.tagger = TaggerInference(model_path, tokenizer_name, label_consistency_level=label_consistency_level, device=device)

    def normalize(self, texts: list[str], norm_puncs=False):
        texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
        
        normalized_texts = []
        for text in texts:
            if self.rule_based_fallback:
                normalized_texts.append(self.rule_based_norm.normalize(text, norm_puncs=norm_puncs, spacing_puncs=False))
            else:
                normalized_texts.append(None)
        
        inference_indexes = []
        for idx, (text, normed) in enumerate(zip(texts, normalized_texts)):
            if text != normed:
                inference_indexes.append(idx)

        if len(inference_indexes) > 0:
            normalized_texts = [t for t in texts]

            try:
                n_step = 0
                while len(inference_indexes) > 0:
                    input_texts = [normalized_texts[i] for i in inference_indexes]
                    outputs = self.tagger.inference(input_texts)
                    
                    new_inference_indexes = []
                    for idx, output in zip(inference_indexes, outputs):
                        text = normalized_texts[idx]

                        self._log(f"Step: {n_step}")
                        self._log(f"Current text: {text}")
                        self._log(f"Patterns: {output.patterns}")
                        self._log(f"{output.token_level_outputs}")
                        
                        need_infer = False
                        for pattern in output.patterns[::-1]:
                            normed, must_cont = self.tag_based_norm.normalize(
                                pattern.tag, 
                                pattern.content, 
                                text[: pattern.start]
                            )
                            text = text[: pattern.start] + normed + text[pattern.end :]
                            if must_cont is True:
                                need_infer = True

                        normalized_texts[idx] = text
                        if need_infer:
                            new_inference_indexes.append(idx)

                        self._log("New text:", text)
                    
                    inference_indexes = new_inference_indexes

                    n_step += 1
                
                if self.rule_based_fallback is False:
                    return normalized_texts

            except:
                print(texts)
                logger.exception("[AutoTextNormalizer]: ERROR at tagger inference")
                exit(0)

        return [
            self.rule_based_norm.normalize(text, norm_puncs=norm_puncs, spacing_puncs=self.spacing_puncs)
                for text in normalized_texts
        ]
        
    def _log(self, *values, sep=" ", end="\n"):
        if self.verbose:
            print("[AutoTextNormalizer] -", *values, sep=sep, end=end)
        else:
            logger.debug("[AutoTextNormalizer] - " + sep.join(values))
