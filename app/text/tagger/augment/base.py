import re
import abc
import random
import warnings
from dataclasses import dataclass, asdict, field, replace

import nltk
from nltk import word_tokenize

# from ..prompts.elite import TAG_TO_PROMPT

# try:
#     word_tokenize("test")
# except:
#     nltk.download("punkt")
#     nltk.download("punkt_tab")


class Connector:
    CONNECTOR_CONTENT = "đến"
    CONNECTOR_CONTENTS = ["đến", "tới"]

    @classmethod
    def tag(cls):
        return "CONNECTOR"

    @classmethod
    def random_connector_with_spaces(cls, connector: str, space: str = None):
        default_spaces = ["", " "]
        if space is None:
            return "".join([random.choice(default_spaces), connector, random.choice(default_spaces)])
        return "".join([space, connector, space])


DETAILED_TAG_GROUPS = [
    ["INTEGER_n"],
    ["INTEGER_big"],
    ["FLOAT_n"],
    ["FLOAT_big"],
    [
        "MEASUREMENT",
        "MEASUREMENT (không được dùng %)",
        "MEASUREMENT (diện tích, ví dụ như m2, km2, ...)",
        "MEASUREMENT (thể tích, ví dụ như m3, L(lít), ...",
        "MEASUREMENT (khối lượng, ví dụ như g, kg, tấn, ...)",
        "MEASUREMENT (tần số, âm thanh, ví dụ như Hz, kHz, dB, ...)",
        "MEASUREMENT (lực, ví dụ như N, ...)",
        "MEASUREMENT (áp lực, ví dụ như atm, mmHg, pascal, N/m2, ...)",
        "MEASUREMENT (nồng độ, ví dụ như mg/ml, mg/dL, mol, ...)",
        "MEASUREMENT (tốc độ, ví dụ như km/h, m/s, ...)",
        "MEASUREMENT (gia tốc, ví dụ như m/s2, ...)",
        "MEASUREMENT (lưu lượng, ví dụ m3/s, L/s, ...)",
        "MEASUREMENT (năng lượng, ví dụ kwh, kCal, ...)",
        "MEASUREMENT (bộ nhớ, ví dụ 2GB, 5TB, ...)",
        "MEASUREMENT (tốc độ dữ liệu, ví dụ Mbps, ...)",
        "MEASUREMENT (nhiệt độ)",
        "MEASUREMENT (cấu trúc: phân số + đơn vị)",
    ],
    ["MONEY"],
    ["TIME"],
    ["DATE"],
    ["TIME_RANGE"],
    ["DATE_RANGE"],
    ["DATE_RANGE_y_y"],
    ["PHONE"],
    ["EMAIL"],
    ["PLATE"],
    ["ADDRESS"],
    [
        "URL",
        "URL (địa chỉ IP)",
        "URL (đường dẫn đầy đủ của website)",
        "URL (domain của đường dẫn web)",
        "URL (đường dẫn trên máy tính Windows)",
        "URL (đường dẫn trên máy tính Ubuntu)",
    ],
    ["FRACTION"],
    ["NUMBER_RANGE", "NUMBER_RANGE (không được có đơn vị đo)", "NUMBER_RANGE (phải có đơn vị đo)"],
    ["DIMENSION", "DIMENSION (không được có đơn vị)", "DIMENSION (phải có đơn vị)"],
    ["SPORT_SCORE"],
    ["MATH_EXPR"],
    ["ROMAN_NUMERAL"],
    ["LEGAL_DOC_ID"],
    [
        "DATE và FRACTION (case khó để phân biệt tốt hơn 2 thực thể này, ví dụ như 6/7 có thể là ngày (dạng DD/MM), có thể là phân số tùy vào ngữ cảnh)"
    ],
]
NEW_VALID_TAGS = [t[0] for t in DETAILED_TAG_GROUPS[:-1]] + ["FOREIGN_WORD", "ALPHANUM_ID", Connector.tag()]

# OLD_VALID_TAGS: list[str] = [Connector.tag()] + sum([[tag] if "__" not in tag else tag.split("__") for tag in TAG_TO_PROMPT], [])

OLD_VALID_TAG_MAP = {
    "DATE_dm": "DATE",
    "DATE_dmy": "DATE",
    "DATE_my": "DATE",
    "DATE_RANGE_dm_dmy": "DATE_RANGE",
    "DATE_RANGE_m_my": "DATE_RANGE",
    "TIME_hm": "TIME",
    "TIME_hms": "TIME",
    "TIME_h": "TIME",
    "ID_NUMBER": "ALPHANUM_ID",
    # ---
    "INTEGER": "INTEGER",
    "FLOAT": "FLOAT",
}


class InvalidTagError(Exception):
    pass


@dataclass
class Pattern:
    content: str
    tag: str
    start: int
    end: int
    normed_content: str = None

    check_tag: bool = True

    def __post_init__(self):
        if self.tag is not None and self.check_tag:
            if self.tag in OLD_VALID_TAG_MAP:
                self.tag = OLD_VALID_TAG_MAP[self.tag]
            if self.tag not in NEW_VALID_TAGS and self.tag not in OLD_VALID_TAG_MAP:
                raise InvalidTagError(f"tag must be one of {NEW_VALID_TAGS}. Got {self.tag}")

        if self.content.strip() == "":
            raise InvalidTagError()

        if self.start < 0:
            raise InvalidTagError(str(self))

        if self.end is None:
            self.end = self.start + len(self.content)

        if self.start > self.end:
            raise InvalidTagError(f"tag = {self.tag}")

        if self.content != self.content.strip():
            raise InvalidTagError(f"tag = {self.tag} || content = |" + self.content + "|")

        if self.content != " ".join(self.content.split()):
            raise InvalidTagError(f"tag = {self.tag} || content = |" + self.content + "|")

        if self.tag == Connector.tag() and self.normed_content is None:
            raise InvalidTagError(f"Pattern {Connector.tag()} must have 'normed_content'")

        if self.normed_content is not None and self.normed_content != self.normed_content.strip():
            raise InvalidTagError(f"tag = {self.tag} || normed_text = |" + self.normed_content + "|")

    def shift_index(self, n: int):
        return replace(self, start=self.start + n, end=self.end + n)

    def reset_start(self, start: int):
        return replace(self, start=start, end=start + len(self.content))

    def reset_tag(self, tag: str):
        return replace(self, tag=tag)

    def refine(self):
        if self.tag in ["INTEGER", "FLOAT"]:
            raise InvalidTagError(f"Invalid tag {self.tag}")
        if self.tag in OLD_VALID_TAG_MAP:
            return self.reset_tag(OLD_VALID_TAG_MAP[self.tag])
        elif self.tag in NEW_VALID_TAGS:
            return self.reset_tag(self.tag)
        raise InvalidTagError(f"Invalid tag {self.tag}")

    def to_dict(self):
        return asdict(self)


class BaseTagger(abc.ABC):
    def __init__(self, strict_validate=True):
        super().__init__()

        self.strict_validate = strict_validate

    @classmethod
    @abc.abstractmethod
    def tag(cls) -> str:
        pass

    @abc.abstractmethod
    def validate(self, pattern_str: str) -> int:
        """Return zero means good"""

    @abc.abstractmethod
    def _augment(self, pattern_str: str) -> Pattern:
        pass

    @classmethod
    def _validate_on_augment(cls) -> bool:
        return True

    def _validate_patterns(self, text: str, patterns: list[Pattern]):
        for pattern in patterns:
            if text[pattern.start : pattern.end] != pattern.content:
                raise InvalidTagError(f"{text[pattern.start : pattern.end]} != {pattern.content}")
            if pattern.tag is None:
                raise InvalidTagError("tag cannot be None")
            if pattern.tag != self.tag():
                if self.tag() in OLD_VALID_TAG_MAP and pattern.tag == OLD_VALID_TAG_MAP[self.tag()]:
                    pass
                else:
                    msg = f"Tag mismatch: {pattern.tag} != {self.tag()}"
                    if self.strict_validate:
                        raise InvalidTagError(msg)
                    warnings.warn(msg)
            else:
                if self.validate(pattern.content) != 0:
                    raise InvalidTagError(
                        f"tag = {pattern.tag} |" + pattern.content + f"| Error code: {self.validate(pattern.content)}"
                    )

    def extract(self, text: str):
        """Extract pattern from text"""
        content, patterns = self._extract(text)
        if len(patterns) <= 0:
            raise InvalidTagError()
        self._validate_patterns(content, patterns)
        return content, patterns

    def _extract(self, text: str):
        return extract_patterns(text, default_tag=self.tag())

    def augment(self, pattern_str: str):
        pattern = self._augment(pattern_str)
        self._validate_patterns(pattern.content, [pattern])
        status = self.validate(pattern.content)
        if status != 0:
            msg = f"{self.__class__.__name__}: Invalid augmentation (error code = {status}): {pattern.content}"
            if self._validate_on_augment():
                raise InvalidTagError(msg)
            else:
                warnings.warn(msg)
        if pattern.content != pattern.content.strip():
            raise InvalidTagError(f"Invalid pattern")
        return pattern

    @staticmethod
    def num_prefix_blank(text: str):
        return len(text) - len(text.lstrip())


@dataclass
class RangePattern(Pattern):
    parts: list[Pattern] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()

        self.parts = [Pattern(**part) if isinstance(part, dict) else part for part in self.parts]
        assert len(self.parts) > 0

        self.validate()

    def validate(self):
        for part in self.parts:
            if part.content != self.content[part.start : part.end]:
                raise InvalidTagError(f"|{part.content}| != |{self.content[part.start : part.end]}|")

    def to_normalized_content(self):
        content = []
        for part in self.parts:
            if part.tag == Connector.tag():
                content.append(part.normed_content)
            else:
                content.append(part.content)
        return " ".join(content)

    def to_normalized_patterns(self) -> tuple[str, list[Pattern]]:
        content = ""
        patterns = []
        for idx, part in enumerate(self.parts):
            if part.tag == Connector.tag():
                content += part.normed_content

                if idx < len(self.parts) - 1 and part.normed_content != "":
                    content += " "

            else:
                patterns.append(
                    Pattern(
                        content=part.content,
                        tag=part.tag,
                        start=len(content),
                        end=None,
                    )
                )
                content += part.content

                if idx < len(self.parts) - 1:
                    content += " "

        if content != content.strip():
            raise InvalidTagError("|" + content + "|")
        for pattern in patterns:
            if pattern.content != content[pattern.start : pattern.end]:
                raise InvalidTagError()

        return content, patterns

    def to_dict(self):
        value = asdict(self)
        value["parts"] = [part.to_dict() for part in self.parts]
        return value


class BaseRangeTagger(BaseTagger):
    def __init__(self, strict=True):
        super().__init__(strict_validate=strict)

    @classmethod
    @abc.abstractmethod
    def connector(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def connector_content(cls) -> str:
        pass

    @abc.abstractmethod
    def _split(self, pattern_str: str) -> list[Pattern]:
        """Split raw range pattern to smaller part, for example DATE_RANGE = DATE - DATE"""

    @abc.abstractmethod
    def _augment(self, pattern_str: str) -> RangePattern:
        pass
    
    @classmethod
    def _validate_on_augment(cls) -> bool:
        return True

    def _validate_patterns(self, text: str, patterns: list[Pattern | RangePattern]):
        for pattern in patterns:
            if text[pattern.start : pattern.end] != pattern.content:
                raise InvalidTagError(f"{text[pattern.start : pattern.end]} != {pattern.content}")
            if pattern.tag is None:
                raise InvalidTagError("tag cannot be None")
            if pattern.tag != self.tag():
                if self.tag() in OLD_VALID_TAG_MAP and pattern.tag == OLD_VALID_TAG_MAP[self.tag()]:
                    pass
                else:
                    msg = f"Tag mismatch: {pattern.tag} != {self.tag()}"
                    if self.strict_validate:
                        raise InvalidTagError(msg)
                    warnings.warn(msg)
            else:
                self.validate(pattern.content)
            if isinstance(pattern, RangePattern):
                pattern.validate()

    def split(self, pattern_str: str):
        status = self.validate(pattern_str)
        if status != 0:
            raise InvalidTagError(f"{self.tag()}: ERROR code {status}: '{pattern_str}'")
        return self._split(pattern_str)

    def extract(self, text: str) -> list[Pattern | RangePattern]:  # type: ignore
        """Extract pattern from text"""
        content, patterns = self._extract(text)
        if len(patterns) <= 0:
            raise InvalidTagError()
        new_patterns = []
        for pattern in patterns:
            if pattern.tag == self.tag():
                new_patterns.append(
                    RangePattern(
                        **pattern.to_dict(),
                        parts=self.split(pattern.content),
                    )
                )
            else:
                new_patterns.append(pattern)
        self._validate_patterns(content, new_patterns)
        return content, new_patterns  # type: ignore

    def _extract(self, text: str):
        return extract_patterns(text, default_tag=self.tag())

    def augment(self, pattern_str: str):
        pattern = self._augment(pattern_str)
        self._validate_patterns(pattern.content, [pattern])
        status = self.validate(pattern.content)
        if status != 0:
            msg = f"{self.__class__.__name__}: Invalid augmentation (error code = {status}): {pattern.content}"
            if self._validate_on_augment():
                raise InvalidTagError(msg)
            else:
                warnings.warn(msg)
        if pattern.content != pattern.content.strip():
            raise InvalidTagError(f"Invalid pattern")
        return pattern


def validate_patterns(text: str, patterns: list[Pattern]):
    for index, pattern in enumerate(patterns):
        if pattern.content != text[pattern.start : pattern.end]:
            raise InvalidTagError(f"Content mismatch: |{pattern.content}| != |{text[pattern.start : pattern.end]}|")
        if index != 0:
            prev = patterns[index - 1]
            if prev.start + len(prev.content) > pattern.start:
                raise InvalidTagError("Overlap pattern")


def find_str(text: str, sub: str):
    start = 0
    pos = []
    while start < len(text):
        start = text.find(sub, start)
        if start == -1:
            break
        pos.append(start)
        start += 1
    return pos


def extract_patterns(raw_content: str, default_tag: str = None):
    pos = find_str(raw_content, "**")
    if len(pos) == 0:
        return raw_content, []

    if len(pos) % 2 != 0:
        raise InvalidTagError(raw_content)
    starts = [pos[i] for i in range(0, len(pos), 2)]
    ends = [pos[i] + 2 for i in range(1, len(pos), 2)]
    if len(starts) != len(ends):
        raise InvalidTagError()

    new_text = raw_content[: starts[0]]
    patterns: list[Pattern] = []
    end = None

    for index in range(len(starts)):
        s = starts[index]
        end = ends[index]

        pattern_str = raw_content[s + 2 : end - 2]
        tag = default_tag
        if end < len(raw_content) and raw_content[end] == "[":
            p = raw_content.find("]", end)
            if p != -1:
                tag = raw_content[end + 1 : p]
                end = p + 1

        patterns.append(
            Pattern(
                content=pattern_str,
                tag=tag,
                start=len(new_text),
                end=len(new_text) + len(pattern_str),
            )
        )
        new_text += pattern_str

        if index + 1 < len(starts):
            new_text += raw_content[end : starts[index + 1]]

    new_text += raw_content[end:]

    validate_patterns(new_text, patterns)

    return new_text, patterns


PUNCTUATION = ".,?!"
READ_PUNCTUATION = "/%-@$&><:+*^=~\\"
SPECIAL_CHARACTERS = "µ°º²³₂₫€£¥₩"
NUMBERS = "0123456789"
CHARACTERS = (
    "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlL"
    "mMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
)
ALL_CHARS = PUNCTUATION + NUMBERS + CHARACTERS + READ_PUNCTUATION + SPECIAL_CHARACTERS
SPACE_NORMALIZER = re.compile(r"\s+")
WORD_NORMALIZER = re.compile(r"[^{}]".format(re.escape(ALL_CHARS)))
PUNC_NORMALIZER = re.compile(r"\s+(([{}])+)($|\s+)".format(re.escape(PUNCTUATION)))


def punc_priority(group_punc):
    if "?" in group_punc:
        return "?"
    if "!" in group_punc:
        return "!"
    if "." in group_punc:
        return "."
    return ","


def clean_content(content: str):
    def _encode_word(text: str, words: list[str]):
        mapping = []
        for index, word in enumerate(words):
            cnt = 0
            while True:
                code = f"E{index}X{cnt}"
                if code not in text:
                    mapping.append((word, code))
                    text = text.replace(word, code)
                    break
                cnt += 1
        return text, mapping

    def _decode_word(text: str, encoded: list):
        for word, code in encoded:
            text = text.replace(code, word)
        return text

    content = content.replace("–", "-")
    content = WORD_NORMALIZER.sub(" ", content)
    content = PUNC_NORMALIZER.sub(r" \1 ", content)
    content = re.sub(r"\s+", " ", content)

    content, encoded = _encode_word(content, ["@", "://", ":\\"])
    content = " ".join(word_tokenize(content))
    content = content.replace("``", '"')
    content = _decode_word(content, encoded)

    content = re.sub(
        r"\s+([{}]+\s*)+".format(re.escape(PUNCTUATION)), lambda x: " {} ".format(punc_priority(x.group(0))), content
    )
    content = re.sub(
        r"\s*([{}]+\s+)+".format(re.escape(PUNCTUATION)), lambda x: " {} ".format(punc_priority(x.group(0))), content
    )
    content = re.sub(r"\s+", " ", content)

    content = content.strip()
    return content


def encode_patterns(content: str, patterns: list[Pattern]):
    """
    Encode patterns as placeholders in content
    Returns: (encoded_content, encoded_mappings)
    """
    if not patterns:
        return content, []

    encoded_mappings = []
    working_content = content

    # Create working copies of patterns to update positions as we go
    working_patterns = [
        Pattern(content=p.content, tag=p.tag, start=p.start, end=p.end, normed_content=p.normed_content) for p in patterns
    ]

    # Sort patterns by position in forward order for natural numbering
    working_patterns.sort(key=lambda x: x.start)

    for i, pattern in enumerate(working_patterns):
        # Generate safe placeholder that won't be affected by text cleaning
        placeholder = f"PATTERN{i}"
        counter = 0
        while placeholder in content:
            counter += 1
            placeholder = f"PATTERN{i}X{counter}"

        # Replace pattern content with placeholder
        old_length = len(pattern.content)
        new_length = len(placeholder)
        working_content = working_content[: pattern.start] + placeholder + working_content[pattern.end :]

        # Calculate length difference for position updates
        length_diff = new_length - old_length

        # Update positions of all subsequent patterns
        for j in range(i + 1, len(working_patterns)):
            if working_patterns[j].start >= pattern.end:
                working_patterns[j].start += length_diff
                working_patterns[j].end += length_diff

        # Store mapping
        encoded_mappings.append(
            {
                "placeholder": placeholder,
                "original_pattern": pattern,
                "position_in_encoded": pattern.start,  # Position where placeholder starts
                "placeholder_length": len(placeholder),
            }
        )

    return working_content, encoded_mappings


def decode_patterns(encoded_content: str, encoded_mappings: list):
    """
    Decode placeholders back to original patterns
    Returns: (decoded_content, updated_patterns)
    """
    # Find all placeholder positions in the encoded content
    placeholder_positions = []
    for mapping in encoded_mappings:
        placeholder = mapping["placeholder"]
        pos = encoded_content.find(placeholder)
        if pos != -1:
            placeholder_positions.append((pos, mapping))

    # Sort by position for correct processing order
    placeholder_positions.sort(key=lambda x: x[0])

    # Replace placeholders with original content, processing in reverse order to avoid index shifting
    final_content = encoded_content

    # Process replacements in reverse order by position to avoid index shifting
    for pos, mapping in reversed(placeholder_positions):
        placeholder = mapping["placeholder"]
        original_pattern = mapping["original_pattern"]

        # Replace placeholder with original content
        final_content = final_content[:pos] + original_pattern.content + final_content[pos + len(placeholder) :]

    # Now create patterns with correct positions by calculating cumulative length differences
    final_patterns = []
    cumulative_diff = 0

    for pos, mapping in placeholder_positions:
        original_pattern = mapping["original_pattern"]
        placeholder = mapping["placeholder"]

        # Calculate actual position accounting for previous replacements
        actual_pos = pos + cumulative_diff

        # Create updated pattern with correct position
        final_pattern = original_pattern.reset_start(actual_pos)
        final_patterns.append(final_pattern)

        # Update cumulative difference for next patterns
        length_diff = len(original_pattern.content) - len(placeholder)
        cumulative_diff += length_diff

    # Sort patterns by position
    final_patterns.sort(key=lambda x: x.start)

    return final_content, final_patterns


def clean_content_with_patterns(content: str, patterns: list[Pattern]) -> tuple[str, list[Pattern]]:
    """
    Clean content using pattern encoding approach

    Steps:
    1. Encode patterns as safe placeholders (PATTERN0, PATTERN1, etc.)
    2. Clean the encoded content (placeholders are immune to cleaning rules)
    3. Decode placeholders back to original patterns with updated positions
    4. Validate final result
    """
    if not patterns:
        return clean_content(content), []

    # Step 1: Encode patterns as placeholders
    encoded_content, encoded_mappings = encode_patterns(content, patterns)

    # print("\nencoded_content =", encoded_content, "\n")

    # Step 2: Clean the encoded content
    # The placeholders (PATTERN0, PATTERN1, etc.) won't be affected by cleaning rules
    cleaned_encoded = clean_content(encoded_content)

    # Step 3: Decode placeholders back to patterns with updated positions
    final_content, final_patterns = decode_patterns(cleaned_encoded, encoded_mappings)

    # Step 4: Validate final result
    validate_patterns(final_content, final_patterns)

    return final_content, final_patterns


def expand_full_word_of_content(content, patterns: list[Pattern]) -> list[Pattern]:
    """
    Handle case like **6.5%**[MEASUREMENT]/năm

    Be careful with case:
        Sau **5**[INTEGER_n] năm Việt Nam gia nhập **WTO**[ALPHANUM_ID] (tức từ **2007-2012**[DATE_RANGE_y_y]),

    => Should clean content first
    """
    expanded_patterns = []
    for pattern in patterns:
        start = pattern.start
        end = pattern.end
        while start - 1 >= 0 and content[start - 1].strip() != "":
            start -= 1

        while end < len(content) and content[end].strip() != "":
            end += 1
        expanded_patterns.append(Pattern(content=content[start:end], tag=pattern.tag, start=start, end=end))

    return expanded_patterns


def convert_to_raw_content(text: str, patterns: list[Pattern]):
    patterns = sorted(patterns, key=lambda x: x.start, reverse=True)
    for pattern in patterns:
        if pattern.tag is not None:
            text = text[: pattern.start] + f"**{pattern.content}**[{pattern.tag}]" + text[pattern.end :]
    return text


def is_alphanum_id(text: str):
    for ch in text:
        if ch.isdigit():
            return True

    has_char = False
    num_upper = 0
    for ch in text:
        if ch.isalpha():
            has_char = True
            if ch.upper() == ch:
                num_upper += 1
    if has_char and num_upper / len(text) >= 0.5:
        return True

    special_puncs = "@#&*"
    for p in special_puncs:
        if p in text:
            return True

    return False


vocab_set: set = None


def get_vocab_vi() -> set:
    global vocab_set
    if vocab_set is not None:
        return vocab_set
    vocab_set = set()
    with open("data/lexicon_vi.tsv") as f:
        for line in f.readlines():
            word = line.split()[0]
            vocab_set.add(word.lower())
    return vocab_set


def is_vietnamese(word: str):
    if word.lower() in get_vocab_vi():
        return True
    return False


def contains_no_number(text: str):
    for ch in text:
        if ch.isdigit():
            return False
    return True


def contains_only_char(word: str):
    for ch in word:
        if ch.isalpha() is False:
            return False
    return True


def calibrate_tags(raw_content: str, return_str=False, default_tag=None) -> str | tuple[str, list[Pattern]]:
    content, patterns = extract_patterns(raw_content, default_tag=default_tag)

    has_none = False
    new_patterns = []
    for pattern in patterns:
        if pattern.tag is None:
            has_none = True  # BUG
            warnings.warn("Has None tag")
        else:
            assert pattern.tag in NEW_VALID_TAGS or pattern.tag in OLD_VALID_TAG_MAP, pattern.tag

        if pattern.tag in ["FOREIGN_WORD", "ALPHANUM_ID"]:
            start = pattern.start
            for word in pattern.content.split():
                if pattern.tag == "ALPHANUM_ID":
                    if is_alphanum_id(word):
                        new_tag = "ALPHANUM_ID"
                    elif contains_no_number(word) and is_vietnamese(word) is False:
                        new_tag = "FOREIGN_WORD"
                    else:
                        new_tag = None
                else:
                    if contains_only_char(word):
                        if is_vietnamese(word):
                            new_tag = None
                        elif is_alphanum_id(word):
                            new_tag = "ALPHANUM_ID"
                        else:
                            new_tag = "FOREIGN_WORD"
                    elif is_alphanum_id(word):
                        new_tag = "ALPHANUM_ID"
                    else:
                        new_tag = None

                if new_tag is not None:
                    new_patterns.append(Pattern(content=word, tag=new_tag, start=start, end=None))

                start += 1 + len(word)

        elif pattern.tag is not None:
            new_patterns.append(pattern)

    validate_patterns(content, new_patterns)
    content, new_patterns = clean_content_with_patterns(content, new_patterns)
    validate_patterns(content, new_patterns)
    new_patterns = expand_full_word_of_content(content, new_patterns)

    if return_str:
        return convert_to_raw_content(content, new_patterns)
    return content, new_patterns


if __name__ == "__main__":
    raw_content = 'Tại khu vực này ("**80**[INTEGER_n]." à), (cũng "**4.5**[FLOAT_n].") việc đỗ xe tạm A10 thời chỉ 12.3,20. được **nguyen @ gmail.com**[EMAIL], phép **80**[INTEGER_n] làm PATTERN1 và PATTERN1X1 trong (**-63.885953 µm/s**[MEASUREMENT].) với giá **1.500đ**[MONEY]/năm'

    raw_content = "Nhà sản xuất âm nhạc **Slim**[FOREIGN_WORD] **V**[ALPHANUM_ID] đã mang cờ Việt Nam lên sân khấu **Tomorrowland**[FOREIGN_WORD] năm **2023**[INTEGER_n], nơi có hơn **400.000**[INTEGER_big] khán giả đến từ **200**[INTEGER_n] quốc gia. Buổi biểu diễn kéo dài từ **21h45**[TIME] đến **23h**[TIME] ngày **29/7**[DATE], thu hút hơn **2.500.000**[INTEGER_big] lượt xem trực tuyến trên **YouTube**[FOREIGN_WORD]. Anh đã sử dụng bộ xử lý **CPU**[ALPHANUM_ID] **i9-13900K**[ALPHANUM_ID] và card âm thanh **Apollo**[ALPHANUM_ID] **X8**[ALPHANUM_ID] trị giá **$3.500**[MONEY], với tổng công suất loa đạt **150.000W**[MEASUREMENT]. Vé VIP được bán với giá **€1.250,50**[MONEY] mỗi người, và đội ngũ kỹ thuật phải làm việc trong khung giờ **9h-18h**[TIME_RANGE] để chuẩn bị. Để biết thêm chi tiết, bạn có thể truy cập **www.slimv.vn**[URL] hoặc gửi email đến **info@slimv.com**[EMAIL]. Tỷ lệ phản hồi của khán giả là **4.7/5**[FRACTION], và sân khấu có kích thước **25x60m**[DIMENSION]."

    content, patterns = extract_patterns(raw_content)

    print("Original patterns:")
    for p in patterns:
        print(f"  {p.content} [{p.tag}] at {p.start}-{p.end}")
    print(f"\nOriginal content: {content}")

    # Test the old cleaning approach
    cleaned_old = clean_content(content)
    print(f"\nCleaned content (old): {cleaned_old}")

    print(calibrate_tags(raw_content, return_str=True))
