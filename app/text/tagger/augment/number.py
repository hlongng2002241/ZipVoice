import abc
import random
import string
from .base import (
    extract_patterns,
    InvalidTagError,
    Connector,
    Pattern,
    RangePattern,
    BaseTagger,
    BaseRangeTagger,
)
from ...normalizer.rule.utils.units import UNITS_DICT, CURRENCY_UNITS, PREFIX_CURRENCY_UNITS, SUFFIX_CURRENCY_UNITS


class MeasurementTagger(BaseRangeTagger):
    UNITS = [u.strip() for u in UNITS_DICT if u not in CURRENCY_UNITS]
    # UNITS_NO_CURRENCY = [u.strip() for u in UNITS_DICT]

    def __init__(self, strict_unit: bool, strict_validate=False):
        super().__init__(strict_validate)

        self.number_taggers: dict[str, BaseTagger] = {
            "f_big": Float_big_Tagger(skip_extract_unit=False, strict_validate=strict_validate),
            "f_n": Float_n_Tagger(skip_extract_unit=False, strict_validate=strict_validate),
            "i_n": Integer_n_Tagger(skip_extract_unit=False, strict_validate=strict_validate),
            "i_big": Integer_big_Tagger(skip_extract_unit=False, strict_validate=strict_validate),
            "fr": FractionTagger(skip_extract_unit=False, strict_validate=strict_validate, aug_with_space=False),
        }

        self.strict_unit = strict_unit

    @classmethod
    def tag(cls):
        return "MEASUREMENT"

    @classmethod
    def connector(cls):
        return ""

    @classmethod
    def connector_content(cls):
        return ""

    @staticmethod
    def _split_num_and_unit(num_and_unit: str, raise_error=True):
        num = "0123456789"
        valid = num + "/.,"
        special_valid = "+-"
        has_special = False
        sep_index = None
        for idx, ch in enumerate(num_and_unit):
            if ch in special_valid:
                if has_special is False:
                    has_special = True
                else:
                    sep_index = idx
                    break
            elif ch in valid:
                pass
            else:
                sep_index = idx
                break
        if sep_index is not None:
            while sep_index > 0 and num_and_unit[sep_index - 1] not in num + "/":
                sep_index -= 1
            while sep_index > 0 and num_and_unit[sep_index - 1] == "/":
                sep_index -= 1
            if sep_index < 0:
                if raise_error:
                    raise InvalidTagError("Invalid: " + num_and_unit)
                else:
                    return None, None
        else:
            sep_index = len(num_and_unit)

        num = num_and_unit[:sep_index]
        unit = num_and_unit[sep_index:]

        if num.strip() == "" and unit[0] in special_valid and unit[1] in special_valid:
            if raise_error:
                raise InvalidTagError("Invalid: " + num_and_unit)
            else:
                return None, None

        return num, unit

    def _find_num_tag(self, num_str: str):
        for tagger in self.number_taggers.values():
            if tagger.validate(num_str) == 0:
                return tagger.tag()

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1
        try:
            num, unit = self._split_num_and_unit(pattern_str)
        except InvalidTagError:
            return 2
        assert unit is not None
        if unit.strip() == "":
            return 3
        if self.strict_unit and unit.strip() not in UNITS_DICT:
            return 4
        if len(num) > 0 and self._find_num_tag(num) is None:
            return 5
        for word in unit.split():
            try:
                _num, _unit = self._split_num_and_unit(word)
            except InvalidTagError:
                return 6
            if len(_num) > 0:
                return 7
        return 0

    def _split(self, pattern_str):
        num, unit = self._split_num_and_unit(pattern_str)
        assert unit is not None
        parts = []
        if unit[0] == " ":
            if len(num) > 0:
                parts.append(Pattern(content=num, tag=self._find_num_tag(num), start=0, end=None))
                start = len(num)
                parts.append(
                    Pattern(content=unit.strip(), tag=self.tag(), start=len(unit) - len(unit.lstrip()), end=None).shift_index(
                        start
                    )
                )
            else:
                parts.append(Pattern(content=pattern_str, tag=self.tag(), start=0, end=None))
        else:
            parts.append(Pattern(content=pattern_str, tag=self.tag(), start=0, end=None))
        return parts

    def _augment_measurement(self, unit: str | None, num_type: str | None = None, num_format: str | None = None):
        if num_type is not None:
            if num_type not in self.number_taggers:
                raise InvalidTagError(num_type)

        unit = random.choice(["", " "]) + random.choice(self.UNITS) if unit is None else unit
        if unit.strip() not in self.UNITS:
            raise InvalidTagError()
        space = unit[: len(unit) - len(unit.lstrip())]
        if space not in ["", " "]:
            raise InvalidTagError("|" + space + "|")

        parts = []
        num = self.number_taggers[num_type].augment(num_format) if num_type is not None else None

        if space == "":
            if num is not None:
                parts.append(Pattern(content=num.content + unit, tag=self.tag(), start=0, end=None))
            else:
                parts.append(Pattern(content=unit, tag=self.tag(), start=0, end=None))
        else:
            if num is not None:
                parts.append(num)
                start = len(num.content) + 1
            else:
                start = 0
            parts.append(Pattern(content=unit.strip(), tag=self.tag(), start=start, end=None))
        return RangePattern(
            content=((num.content if num is not None else "") + unit).strip(),
            tag=self.tag(),
            start=0,
            end=None,
            parts=parts,
        )

    def _augment(self, pattern_str):
        """Example: i_n=-10:10| m"""
        if pattern_str is not None:
            if pattern_str in self.number_taggers:
                unit = None
                num_type = pattern_str
                num_format = None
            else:
                parts = pattern_str.split("|")
                if len(parts) == 1:
                    unit = parts[0]
                    num_type = None
                    num_format = None
                elif len(parts) == 2:
                    num, unit = parts
                    if unit == "":
                        unit = None
                    if "==" in num:
                        num_type, num_format = num.split("==")
                    else:
                        num_type, num_format = num.split("=")
                    if num_format == "":
                        num_format = None
                else:
                    raise InvalidTagError("Invalid format")
        else:
            unit = None
            num_type = None
            if random.random() < 0.5:
                num_type = random.choice([n for n, t in self.number_taggers.items() if isinstance(t, FractionTagger) is False])
            num_format = None
        return self._augment_measurement(unit, num_type, num_format)


class NumberToMeasurement(BaseTagger):
    UNITS = [u.strip() for u in UNITS_DICT]

    def __init__(self, number_tag: str):
        super().__init__(strict_validate=False)

        self.number_tag = number_tag

    @classmethod
    def tag(cls):
        return MeasurementTagger.tag()

    def validate(self, pattern_str):
        if pattern_str.strip() == 0:
            return 1
        return 0

    def _augment(self, pattern_str):
        raise NotImplementedError("Not support augmenting")

    def _extract(self, text):
        content, patterns = extract_patterns(text, default_tag=self.number_tag)

        new_patterns: list[Pattern] = []
        for pattern in patterns:
            if pattern.tag == self.tag():
                new_patterns.append(pattern)
            else:
                unit_str = ""
                index = pattern.end
                has_space = False
                while index < len(content) and content[index] == " ":
                    index += 1
                    has_space = True
                unit_start = index
                while index < len(content):
                    if content[index] == " ":
                        break
                    unit_str += content[index]
                    index += 1
                while len(unit_str) > 0 and unit_str[-1] in "~`@#^&*()-_=+[]{}:;'\",.<>/?":
                    unit_str = unit_str[:-1]

                is_unit = False
                if unit_str in self.UNITS:
                    is_unit = True
                elif "/" in unit_str:
                    is_unit = True
                    # oke = True
                    # for m in unit.split("/"):
                    #     if m not in self.UNITS:
                    #         oke = False
                    #         break
                    # if oke:
                    #     is_unit = True

                if is_unit:
                    if has_space is False:
                        unit_content = content[pattern.start : unit_start + len(unit_str)]
                        if unit_content.startswith(pattern.content) is False:
                            raise InvalidTagError()
                        if unit_content.endswith(unit_str) is False:
                            raise InvalidTagError()
                        new_patterns.append(Pattern(unit_content, tag=self.tag(), start=pattern.start, end=None))
                    else:
                        new_patterns.append(pattern)
                        new_patterns.append(Pattern(unit_str, tag=self.tag(), start=unit_start, end=None))
                else:
                    new_patterns.append(pattern)

        return content, new_patterns


class BaseNumberTagger(BaseTagger):
    def __init__(self, skip_extract_unit: bool, strict_validate=False, ignore_3_sep=False):
        super().__init__(strict_validate)

        self._num_to_measurement = NumberToMeasurement(self.tag())
        self.skip_extract_unit = skip_extract_unit
        self.ignore_3_sep = ignore_3_sep

    def _extract(self, text):
        if self.skip_extract_unit:
            return super()._extract(text)
        return self._num_to_measurement.extract(text)


class BaseNumberRangeTagger(BaseRangeTagger):
    def __init__(self, skip_extract_unit: bool, strict_validate=False):
        super().__init__(strict_validate)

        self.num_to_measurement = NumberToMeasurement(self.tag())
        self.skip_extract_unit = skip_extract_unit

        self.number_taggers: dict[str, BaseNumberTagger] = {
            "f_big": Float_big_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
            "f_n": Float_n_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
            "i_big": Integer_big_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
            "i_n": Integer_n_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
        }

    @classmethod
    @abc.abstractmethod
    def valid_formats(cls) -> list[tuple[str, str]]:
        pass

    def _find_nums_tags(self, parts: list[str]) -> list[str] | None:
        for fmts in self.valid_formats():
            if len(fmts) == len(parts):
                valid = True
                for fmt, part in zip(fmts, parts):
                    if self.number_taggers[fmt].validate(part) != 0:
                        valid = False
                        break
                if valid:
                    return [self.number_taggers[fmt].tag() for fmt in fmts]

    def _find_num_tag(self, num_str: str):
        for tagger in self.number_taggers.values():
            if tagger.validate(num_str) == 0:
                return tagger.tag()

    def _extract(self, text):
        if self.skip_extract_unit:
            return super()._extract(text)
        return self.num_to_measurement.extract(text)


class Integer_n_Tagger(BaseNumberTagger):
    SIGNS = ["-", "+", " "]

    @classmethod
    def tag(cls):
        return "INTEGER_n"

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        num = "1234567890"
        if pattern_str[0] not in num and pattern_str[0] not in self.SIGNS:
            return 2

        for ch in pattern_str[1:]:
            if ch not in num:
                return 3

        pattern_str = pattern_str.strip()

        try:
            int(pattern_str)
        except:
            return 4

        return 0

    def _augment_integer(self, l: int, r: int, add_plus=False):
        int_str = str(random.randint(l, r))
        if add_plus and int_str[0] != "-":
            int_str = "+" + int_str
        return Pattern(content=int_str, tag=self.tag(), start=0, end=None)

    def _augment(self, pattern_str):
        if pattern_str is None:
            left = -10000
            right = 10000
            add_plus = random.choice([True, False])
        else:
            left, right = pattern_str.split(":")
            add_plus = False
            if right[-1] == "+":
                right = right[:-1]
                add_plus = True
            left = int(left.strip())
            right = int(right.strip())
        return self._augment_integer(left, right, add_plus)


def all_equal(values: list, val):
    for v in values:
        if v != val:
            return False
    return True


def all_gt(values: list, val):
    for v in values:
        if v <= val:
            return False
    return True


class Integer_big_Tagger(BaseNumberTagger):
    SIGNS = ["-", "+", " "]

    def __init__(
        self, skip_extract_unit: bool, allow_space_splitter=False, strict_validate=False, ignore_3_sep=False, ignore_comma=False
    ):
        super().__init__(skip_extract_unit, strict_validate, ignore_3_sep)
        if allow_space_splitter:
            self.SPLITTERS = [",", ".", " "]
        elif ignore_comma:
            self.SPLITTERS = ["."]
        else:
            self.SPLITTERS = [",", "."]

    @classmethod
    def tag(cls):
        return "INTEGER_big"

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        cnt = [pattern_str.count(sp) for sp in self.SPLITTERS]

        if all_gt(cnt, 0):
            return 2

        if all_equal(cnt, 0):
            return 3
        num = "1234567890"
        if pattern_str[0] not in num and pattern_str[0] not in self.SIGNS:
            return 4

        for ch in pattern_str[1:]:
            if ch not in num and ch not in self.SPLITTERS:
                return 5

        num_num = 0
        for ch in pattern_str:
            if ch in num:
                num_num += 1
        # if num_num < 4:
        #     return 6
        # BUG 1.000 là int hay float

        splitter = None
        for ch in pattern_str:
            if ch in self.SPLITTERS:
                splitter = ch
                break

        pattern_str_no_splitter = pattern_str.replace(splitter, "")
        if self.ignore_3_sep is False:
            if pattern_str != self.to_big_format(pattern_str_no_splitter, splitter):
                return 7

        try:
            int(pattern_str_no_splitter)
        except:
            return 8

        return 0

    def to_big_format(self, int_str: str, splitter: str):
        if splitter in int_str:
            raise InvalidTagError()
        if splitter not in self.SPLITTERS:
            raise InvalidTagError()
        sign = ""
        if int_str[0] in self.SIGNS:
            sign = int_str[0].strip()
            int_str = int_str[1:]
        parts = []
        while len(int_str) > 0:
            parts.append(int_str[-3:])
            int_str = int_str[:-3]
        return sign + splitter.join(parts[::-1])

    def _augment_integer(self, n: int, splitter: str, sign: str, num_end_zeros: int = 0):
        if n <= num_end_zeros:
            raise InvalidTagError(f"n <= num_end_zeros ({n} <= {num_end_zeros})")
        if n <= 3:
            raise InvalidTagError(n)
        if splitter not in self.SPLITTERS:
            raise InvalidTagError(splitter)
        if sign not in self.SIGNS:
            raise InvalidTagError(sign)

        int_str = ""
        for i in range(n - num_end_zeros):
            if i == 0:
                int_str += str(random.randint(1, 9))
            else:
                int_str += str(random.randint(0, 9))
        int_str += "".join(["0" for _ in range(num_end_zeros)])

        int_str = sign + self.to_big_format(int_str, splitter)
        int_str = int_str.strip()

        return Pattern(content=int_str, tag=self.tag(), start=0, end=None)

    def _augment(self, pattern_str):
        """Example: -3.z4 | +5, | ..."""
        if pattern_str is None:
            sign = random.choice(self.SIGNS)
            splitter = random.choice(self.SPLITTERS)
            n = random.randint(4, 12)
            num_end_zeros = 0
            if random.random() < 0.5:
                num_end_zeros = random.randint(1, n - 1)
        else:
            sign = pattern_str[0]
            if sign == "_":
                sign = random.choice(self.SIGNS)
            num_end_zeros = 0
            if "z" in pattern_str:
                pattern_str, z = pattern_str.split("z")
                num_end_zeros = int(z)
            splitter = pattern_str[-1]
            n = int(pattern_str[1:-1])
            if splitter == "_":
                splitter = random.choice(self.SPLITTERS)
        return self._augment_integer(n, splitter, sign, num_end_zeros)


class Float_n_Tagger(BaseNumberTagger):
    SIGNS = ["-", "+", " "]
    POINTS = [",", "."]

    @classmethod
    def tag(cls):
        return "FLOAT_n"

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        cnt = sum([pattern_str.count(pt) for pt in self.POINTS])
        if cnt != 1:
            return 2

        num = "1234567890"
        if pattern_str[0] not in num and pattern_str[0] not in self.SIGNS:
            return 3

        for ch in pattern_str[1:]:
            if ch not in num and ch not in self.POINTS:
                return 4

        if pattern_str[-1] in self.POINTS:
            return 5

        pattern_str = pattern_str.strip()
        for p in self.POINTS:
            pattern_str = pattern_str.replace(p, ".")

        try:
            float(pattern_str)
        except:
            return 6

        return 0

    def _augment_float(self, l: float, r: float, n_f_digits: int, add_plus=False):
        float_str = f"{random.uniform(l, r):{n_f_digits}f}"
        if add_plus and float_str[0] != "-":
            float_str = "+" + float_str
        if random.random() < 0.5:
            float_str = float_str.replace(".", ",")
        return Pattern(content=float_str, tag=self.tag(), start=0, end=None)

    def _augment(self, pattern_str):
        if pattern_str is None:
            left = -100
            right = 100
            n_f_digits = random.randint(1, 6)
        else:
            left, right = pattern_str.split(":")
            right, n_f_digits = right.split("f")
            left = float(left.strip())
            right = float(right.strip())
            n_f_digits = int(n_f_digits)
        return self._augment_float(left, right, n_f_digits)


class Float_big_Tagger(BaseNumberTagger):
    SIGNS = ["-", "+", " "]
    SPLITTERS = [",", "."]
    POINTS = [",", "."]

    def __init__(self, skip_extract_unit: bool, strict_validate=False, ignore_3_sep=False):
        super().__init__(skip_extract_unit, strict_validate, ignore_3_sep)

        self._int_big = Integer_big_Tagger(skip_extract_unit=skip_extract_unit)

    @classmethod
    def tag(cls):
        return "FLOAT_big"

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        num = "1234567890"
        for idx, p in enumerate(pattern_str):
            if p not in num and p not in self.SPLITTERS and p not in self.POINTS:
                if idx == 0 and p in self.SIGNS:
                    continue
                return 2

        if pattern_str[-1] in self.POINTS:
            return 3

        def find_pos(text: str, ch: str):
            return [p for p, t in enumerate(text) if t == ch]

        point_pos_list = [find_pos(pattern_str, p) for p in self.POINTS]

        if len(point_pos_list[0]) > 1 and len(point_pos_list[1]) > 1:
            return 4

        if len(point_pos_list[0]) == 0 or len(point_pos_list[1]) == 0:
            return 5

        if point_pos_list[0][-1] < point_pos_list[1][0] or point_pos_list[1][0] < point_pos_list[0][-1]:
            pass
        else:
            return 6

        if point_pos_list[0][-1] < point_pos_list[1][-1]:
            splitter, point = self.POINTS[0], self.POINTS[1]
        else:
            splitter, point = self.POINTS[1], self.POINTS[0]

        if self.ignore_3_sep is False:
            decimal = pattern_str.split(point)[0]
            decimal_no_splitter = decimal.replace(splitter, "")
            if decimal != self.to_big_format(decimal_no_splitter, splitter):
                return 7

        pattern_str = pattern_str.strip().replace(splitter, "").replace(point, ".")
        try:
            float(pattern_str)
        except:
            return 8

        return 0

    def to_big_format(self, int_str: str, splitter: str):
        return self._int_big.to_big_format(int_str, splitter)

    def _augment_float(self, n_dc: int, splitter: str, n_fr: int, sign: str, n_dc_end_zeros: int = 0, n_fr_end_zeros: int = 0):
        if n_dc <= n_dc_end_zeros:
            raise InvalidTagError(f"{n_dc} <= {n_dc_end_zeros}")
        if n_dc <= 3:
            raise InvalidTagError(n_dc)
        if n_fr <= 0:
            raise InvalidTagError(n_fr)
        if n_fr < n_fr_end_zeros:
            raise InvalidTagError(f"{n_fr} < {n_fr_end_zeros}")
        if splitter not in self.SPLITTERS:
            raise InvalidTagError(splitter)
        if sign not in self.SIGNS:
            raise InvalidTagError(sign)

        point = self.POINTS[1] if splitter == self.POINTS[0] else self.POINTS[0]

        decimal_str = ""
        for i in range(n_dc - n_dc_end_zeros):
            if i == 0:
                decimal_str += str(random.randint(1, 9))
            else:
                decimal_str += str(random.randint(0, 9))
        decimal_str += "".join(["0" for _ in range(n_dc_end_zeros)])

        decimal_str = sign + self.to_big_format(decimal_str, splitter)
        decimal_str = decimal_str.strip()

        fractional_str = "".join([str(random.randint(0, 9)) for _ in range(n_fr - n_fr_end_zeros)]) + "".join(
            ["0" for _ in range(n_fr_end_zeros)]
        )

        content = decimal_str + point + fractional_str

        return Pattern(content=content, tag=self.tag(), start=0, end=None)

    def _augment(self, pattern_str):
        """Example: -5.f3 | +7,f2zd5zf1"""
        n_dc_end_zeros = 0
        n_fr_end_zeros = 0
        if pattern_str is None:
            n_dc = random.randint(4, 12)
            n_fr = random.randint(1, 6)
            splitter = random.choice(self.SPLITTERS)
            sign = random.choice(self.SIGNS)
            if random.random() < 0.5:
                n_dc_end_zeros = random.randint(1, n_dc - 1)
            if random.random() < 0.5:
                n_fr_end_zeros = random.randint(0, n_fr)
        else:
            if "zf" in pattern_str:
                pattern_str, zf = pattern_str.split("zf")
                n_fr_end_zeros = int(zf)
            if "zd" in pattern_str:
                pattern_str, zd = pattern_str.split("zd")
                n_dc_end_zeros = int(zd)
            decimal, fractional = pattern_str.split("f")
            sign = decimal[0]
            if sign == "_":
                sign = random.choice(self.SIGNS)
            splitter = decimal[-1]
            if splitter == "_":
                splitter = random.choice(self.SPLITTERS)
            n_dc = int(decimal[1:-1])
            n_fr = int(fractional)
        return self._augment_float(
            n_dc=n_dc, splitter=splitter, n_fr=n_fr, sign=sign, n_dc_end_zeros=n_dc_end_zeros, n_fr_end_zeros=n_fr_end_zeros
        )


class FractionTagger(BaseNumberRangeTagger):
    def __init__(self, skip_extract_unit: bool, strict_validate=False, aug_with_space=True):
        super().__init__(skip_extract_unit, strict_validate)
        
        self.aug_with_space = aug_with_space
        
    @classmethod
    def valid_formats(cls):
        return []  # all

    @classmethod
    def tag(cls):
        return "FRACTION"

    @classmethod
    def connector(cls):
        return "/"

    @classmethod
    def connector_content(cls):
        return "phần"

    @classmethod
    def _find_connector_str(cls, pattern_str: str):
        conn_str = cls.connector()
        if conn_str in pattern_str:
            parts = pattern_str.split(conn_str)
            f_type = None
            for index in range(1, len(parts)):
                prev = parts[index - 1]
                part = parts[index]
                if len(prev.strip()) == 0 or len(part.strip()) == 0:
                    return None
                if prev[-1] == " " and part[0] == " ":
                    _type = "space"
                elif prev[-1] != " " and part[0] != " ":
                    _type = "no_space"
                else:
                    return None
                if f_type is None:
                    f_type = _type
                elif f_type != _type:
                    return None
        return conn_str

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        if self.connector() not in pattern_str:
            return 2
        if self._find_connector_str(pattern_str) is None:
            return 3

        parts = [p.strip() for p in pattern_str.split(self.connector())]
        if len(parts) > 3:
            return 4

        for part in parts:
            oke = False
            for tagger in self.number_taggers.values():
                if tagger.validate(part) == 0:
                    oke = True
                    break
            if oke is False:
                return 5

        return 0

    def _split(self, pattern_str):
        patterns = []
        start = 0
        parts = pattern_str.split(self.connector())
        for index, part in enumerate(parts):
            tag = None
            for tagger in self.number_taggers.values():
                if tagger.validate(part) == 0:
                    tag = tagger.tag()
                    break
            if tag is None:
                raise InvalidTagError("tag cannot be None")

            start += len(part) - len(part.lstrip())
            patterns.append(Pattern(part.strip(), tag=tag, start=start, end=None))
            start += len(part)

            if index < len(parts) - 1:
                patterns.append(
                    Pattern(
                        content=self.connector(),
                        tag=Connector.tag(),
                        start=start,
                        end=None,
                        normed_content=self.connector_content(),
                    )
                )
                start += 1
        return patterns

    def _augment_fraction(self, formats: list[tuple[str | None, str | None]]):
        if len(formats) not in [2, 3]:
            raise InvalidTagError(len(formats))

        conn_space = random.choice(["", " "])
        if self.aug_with_space is False:
            conn_space = ""
        conn_str = Connector.random_connector_with_spaces(self.connector(), conn_space)
        parts = []
        content = ""
        start = 0
        for index, (_type, _fmt) in enumerate(formats):
            if _type not in self.number_taggers:
                raise InvalidTagError(f"Invalid type '{_type}'. Expected one of {list(self.number_taggers.keys())}")

            part = self.number_taggers[_type].augment(_fmt)
            parts.append(part.reset_start(start))
            start += len(part.content)
            content += part.content

            if index < len(formats) - 1:
                start += len(conn_space)
                parts.append(
                    Pattern(
                        content=self.connector(),
                        tag=Connector.tag(),
                        start=start,
                        end=None,
                        normed_content=self.connector_content(),
                    )
                )
                start += len(conn_str) - len(conn_space)
                content += conn_str

        return RangePattern(
            content=content,
            tag=self.tag(),
            start=0,
            end=None,
            parts=parts,
        )

    def _augment(self, pattern_str):
        """Example: i_n=-100:100/i_big=+4."""
        if pattern_str is not None:
            parts = []
            for part in pattern_str.split(self.connector()):
                _type, _fmt = part.split("=")
                if _fmt == "":
                    _fmt = None
                parts.append((_type, _fmt))
        else:
            n = 2 if random.random() < 0.9 else 3
            parts = [(random.choice(list(self.number_taggers.keys())), None) for _ in range(n)]
        return self._augment_fraction(parts)


class NumberRangeTagger(BaseNumberRangeTagger):
    @classmethod
    def valid_formats(cls):
        return [
            ("i_n", "i_n"),
            ("i_n", "f_n"),
            ("i_n", "i_big"),
            ("i_n", "f_big"),
            ("f_n", "i_n"),
            # ("f_n", "i_big"),
            ("f_n", "f_n"),
            ("f_n", "f_big"),
            # ("i_big", "i_n"),
            ("i_big", "i_big"),
            # ("i_big", "f_n"),
            ("i_big", "f_big"),
            # ("f_big", "i_n"),
            ("f_big", "i_big"),
            # ("f_big", "f_n"),
            ("f_big", "f_big"),
        ]

    @classmethod
    def tag(cls):
        return "NUMBER_RANGE"

    @classmethod
    def connector(cls):
        return "-"

    @classmethod
    def connector_content(cls):
        return "đến"

    def _extract(self, text):
        return extract_patterns(text, default_tag=self.tag())

    @staticmethod
    def _split_num_and_unit(num_and_unit: str, raise_error=True):
        return MeasurementTagger._split_num_and_unit(num_and_unit, raise_error)

    @classmethod
    def _find_connector_str(cls, pattern_str: str):
        conn_str = " " + cls.connector() + " "
        if conn_str in pattern_str:
            return conn_str

        conn_str = cls.connector()
        puncs = string.punctuation + " "
        if conn_str in pattern_str:
            parts = pattern_str.split(conn_str)
            if len(parts) == 2:
                if parts[0][-1] not in puncs and parts[1][0] not in puncs:
                    return conn_str

    def validate(self, pattern_str):
        """
        Những cái không phải là số thì sẽ là đơn vị
        """
        if pattern_str.strip() == "":
            return 1

        conn_str = self._find_connector_str(pattern_str)
        if conn_str is None:
            return 2

        parts = pattern_str.split(conn_str)
        if len(parts) != 2:
            return 3

        left, right = parts
        try:
            left_num, left_unit = self._split_num_and_unit(left.strip())
        except InvalidTagError:
            return 4
        try:
            right_num, right_unit = self._split_num_and_unit(right.strip())
        except InvalidTagError:
            return 5

        if self._find_nums_tags([left_num, right_num]) is None:
            # print("|", left, "|", left_num, "|", left_unit)
            # print("|", right, "|", right_num, "|", right_unit)
            return 6

        return 0

    def _split(self, pattern_str):
        conn_str = self._find_connector_str(pattern_str)
        assert conn_str is not None
        left, right = pattern_str.split(conn_str)
        left_num, left_unit = self._split_num_and_unit(left.strip())
        assert left_unit is not None
        right_num, right_unit = self._split_num_and_unit(right.strip())
        assert right_unit is not None
        # print("|", left, "|", left_num, "|", left_unit)
        # print("|", right, "|", right_num, "|", right_unit)

        left_num_tag, right_num_tag = self._find_nums_tags([left_num, right_num])  # type: ignore

        parts = []
        start = 0
        if left_unit.strip() in UNITS_DICT:
            if len(left_unit) > 0 and left_unit[0] == " ":
                # Has space between number and unit
                parts.append(Pattern(content=left_num, tag=left_num_tag, start=0, end=None))
                start += len(left_num)
                parts.append(
                    Pattern(
                        content=left_unit.strip(),
                        tag=NumberToMeasurement.tag(),
                        start=len(left_unit) - len(left_unit.lstrip()),
                        end=None,
                    ).shift_index(start)
                )
            else:
                parts.append(Pattern(content=left.strip(), tag=NumberToMeasurement.tag(), start=0, end=None))
        else:
            if len(left_unit) == 0 or left_unit[0] == " ":
                parts.append(Pattern(content=left_num, tag=left_num_tag, start=0, end=None))
            else:
                parts.append(Pattern(content=left.strip(), tag=NumberToMeasurement.tag(), start=0, end=None))

        start = len(left)

        parts.append(
            Pattern(
                content=self.connector(),
                tag=Connector.tag(),
                start=len(conn_str) - len(conn_str.lstrip()),
                end=None,
                normed_content=self.connector_content(),
            ).shift_index(start)
        )
        start += len(conn_str)

        if right_unit.strip() in UNITS_DICT:
            if len(right_unit) > 0 and right_unit[0] == " ":
                parts.append(Pattern(content=right_num, tag=right_num_tag, start=start, end=None))
                start += len(right_num)
                parts.append(
                    Pattern(
                        content=right_unit.strip(),
                        tag=NumberToMeasurement.tag(),
                        start=len(right_unit) - len(right_unit.lstrip()),
                        end=None,
                    ).shift_index(start)
                )
            else:
                parts.append(
                    Pattern(
                        content=right.strip(), tag=NumberToMeasurement.tag(), start=len(right) - len(right.lstrip()), end=None
                    ).shift_index(start)
                )
        else:
            if len(right_unit) == 0 or right_unit[0] == " ":
                parts.append(Pattern(content=right_num, tag=right_num_tag, start=start, end=None))
            else:
                parts.append(
                    Pattern(
                        content=right.strip(), tag=NumberToMeasurement.tag(), start=len(right) - len(right.lstrip()), end=None
                    ).shift_index(start)
                )

        return parts

    def _augment_range(
        self, type_left: str, format_left: str, unit_left: str, type_right: str, format_right: str, unit_right: str
    ):
        if (type_left, type_right) not in self.valid_formats():
            raise InvalidTagError(type_left + " | " + type_right)
        num_left = self.number_taggers[type_left].augment(format_left)
        num_right = self.number_taggers[type_right].augment(format_right)
        if unit_left is None or unit_left.strip() == "":
            unit_left = None
        if unit_right is None or unit_right.strip() == "":
            unit_right = None

        plus = "+"
        if (
            self.connector() in num_left.content
            or self.connector() in num_right.content
            or plus in num_left.content
            or plus in num_right.content
            or (unit_left is not None and unit_left[0] == " ")
            or (unit_left is not None and " " in unit_left)
            or (unit_right is not None and unit_right[0] not in "0123456789")
        ):
            conn_space = " "
        else:
            conn_space = random.choice(["", " "])
        conn_str = Connector.random_connector_with_spaces(self.connector(), conn_space)

        content = ""

        content += num_left.content
        if unit_left is not None:
            content += unit_left
        content += conn_str
        content += num_right.content
        if unit_right is not None:
            content += unit_right

        return RangePattern(
            content=content.strip(),
            tag=self.tag(),
            start=0,
            end=None,
            parts=self.split(content),
        )

    def _augment(self, pattern_str):
        """i_n=-100:100||i_big= 6.z4"""
        if pattern_str is None:
            types = random.choice(self.valid_formats())
            return self._augment_range(
                types[0],
                None,
                random.choice(["", " "]) + random.choice(self.num_to_measurement.UNITS) if random.random() < 0.5 else None,
                types[1],
                None,
                random.choice(["", " "]) + random.choice(self.num_to_measurement.UNITS) if random.random() < 0.5 else None,
            )

        left, right = pattern_str.split("||")

        def _format(fmt: str):
            sep = "|"
            if sep in fmt:
                tf, u = fmt.split("|")
            else:
                tf, u = fmt, None
            t, f = tf.split("=")
            if f == "":
                f = None
            if u == "_":
                if random.random() < 0.5:
                    u = random.choice(self.num_to_measurement.UNITS)
                else:
                    u = None
            return t, f, u

        args = list(_format(left)) + list(_format(right))
        return self._augment_range(*args)


class DimensionTagger(BaseNumberRangeTagger):
    UNITS = list(UNITS_DICT.keys())

    def __init__(self, allow_closed=False, strict_validate=False):
        super().__init__(True, strict_validate)

        self.allow_closed = allow_closed

    @classmethod
    def tag(cls):
        return "DIMENSION"

    @classmethod
    def valid_formats(cls):
        raise NotImplementedError("This function must be blank")

    @classmethod
    def connector(cls):
        return "x"

    @classmethod
    def connector_content(cls):
        return "nhân"

    @staticmethod
    def _split_num_and_unit(num_and_unit: str, raise_error=True):
        return MeasurementTagger._split_num_and_unit(num_and_unit, raise_error)

    @staticmethod
    def _find_connector(text: str):
        conn_str= "x"
        if conn_str not in text:
            return
        parts = text.split(conn_str)
        d_type = None
        for index in range(1, len(parts)):
            prev = parts[index - 1]
            part = parts[index]
            if len(prev) == 0 or len(part) == 0:
                return
            if prev[-1] != " " and part[0] != " ":
                _type = "no_space"
            elif prev[-1] == " " and part[0] == " ":
                _type = "space"
            else:
                return
            if d_type is None:
                d_type = _type
            elif d_type != _type:
                return
        return conn_str

    def validate(self, pattern_str: str):
        if pattern_str.strip() == "":
            return 1

        if self._find_connector(pattern_str) is None:
            return 2
        
        parts = pattern_str.split("x")
        for idx, part in enumerate(parts):
            if len(part.strip()) == 0:
                return 3
            try:
                num, unit = self._split_num_and_unit(part.strip())
            except InvalidTagError:
                return 4
            assert unit is not None
            # print("|" + num + "|" + unit + "|")
            if idx < len(parts) - 1 and unit.strip() != "" and part[-1] not in [" ", "."]:
                if self.allow_closed is False:
                    return 5  # Unit cannot stick close to "x"
            if self._find_num_tag(num) is None:
                return 6

        return 0

    def _split(self, pattern_str):
        components = pattern_str.split("x")
        parts = []
        content = ""
        for idx, component in enumerate(components):
            num, unit = self._split_num_and_unit(component.strip())
            assert num is not None and unit is not None
            _num_tag = self._find_num_tag(num)
            if _num_tag is None:
                raise InvalidTagError((num, unit))

            if unit.strip() != "" and unit[0] == " ":
                parts.append(
                    Pattern(content=num.strip(), tag=_num_tag, start=len(content) + len(num) - len(num.lstrip()), end=None)
                )
                content += num
                parts.append(
                    Pattern(
                        content=unit.strip(),
                        tag=MeasurementTagger.tag(),
                        start=len(content) + len(unit) - len(unit.lstrip()),
                        end=None,
                    )
                )
                content += unit
            elif unit.strip() != "" and unit[0] != " ":
                parts.append(
                    Pattern(
                        content=component.strip(),
                        tag=MeasurementTagger.tag(),
                        start=len(content) + len(component) - len(component.lstrip()),
                        end=None,
                    )
                )
                content += component
            elif unit.strip() == "":
                parts.append(
                    Pattern(content=num.strip(), tag=_num_tag, start=len(content) + len(num) - len(num.lstrip()), end=None)
                )
                content += num
            else:
                raise NotImplementedError()

            if idx < len(components) - 1:
                parts.append(
                    Pattern(
                        content=self.connector(),
                        tag=Connector.tag(),
                        start=len(content),
                        end=None,
                        normed_content=self.connector_content(),
                    )
                )
                content += self.connector()
        return parts

    def _augment_dimension(self, formats: list[tuple[str | None, str | None, str | None]]):
        """
        Parameters
        ----------
        formats: list tuple type, format and unit
        """
        parts = []
        content = ""
        has_unit = False
        for index, (_type, _fmt, _unit) in enumerate(formats):
            if _type not in self.number_taggers:
                raise InvalidTagError(f"Invalid type '{_type}'. Expected one of {list(self.number_taggers.keys())}")
            if _unit is not None and _unit.strip() != "":
                has_unit = True
                break
            
        space = " " if has_unit else random.choice(["", " "])
        
        for index, (_type, _fmt, _unit) in enumerate(formats):

            if _unit is not None:
                _unit = _unit.strip()

            part = self.number_taggers[_type].augment(_fmt)
            if _unit is not None and _unit.strip() != "":
                # Has unit
                # if _unit[-1] != " ":
                #     space = " "
                # else:
                #     space = random.choice(["", " "])
                if _unit[0] == " ":
                    parts.append(part.reset_start(len(content)))
                    content += part.content
                    parts.append(
                        Pattern(
                            content=_unit.strip(),
                            tag=MeasurementTagger.tag(),
                            start=len(content) + self.num_prefix_blank(_unit),
                            end=None,
                        )
                    )
                    content += _unit
                else:
                    parts.append(
                        Pattern(content=(part.content + _unit).strip(), tag=MeasurementTagger.tag(), start=len(content), end=None)
                    )
                    content += part.content + _unit
            else:
                # space = random.choice(["", " "])
                parts.append(part.reset_start(len(content)))
                content += part.content

            if index < len(formats) - 1:
                content += space
                parts.append(
                    Pattern(
                        content=self.connector(),
                        tag=Connector.tag(),
                        start=len(content),
                        end=None,
                        normed_content=self.connector_content(),
                    )
                )
                # content += self.connector() + random.choice(["", " "])
                content += self.connector() + space

        return RangePattern(
            content=content.strip(),
            tag=self.tag(),
            start=0,
            end=None,
            parts=parts,
        )

    def _augment(self, pattern_str):
        """Example: i_n=-100:100||i_big=+4.|mm"""
        parts = []
        if pattern_str is not None:
            for idx, part in enumerate(pattern_str.split("||")):
                if "|" in part:
                    _num, _unit = part.split("|")
                else:
                    _num, _unit = part, None
                if _unit == "_":
                    if random.random() < 0.5:
                        _unit = random.choice(self.UNITS)
                    else:
                        _unit = None
                _type, _fmt = _num.split("=")
                if _fmt == "":
                    _fmt = None
                parts.append((_type, _fmt, _unit))
        else:
            t = random.randint(0, 2)
            n = random.randint(2, 3)
            for i in range(n):
                _type = random.choice(list(self.number_taggers.keys()))
                _fmt = None
                if t == 2:
                    if random.random() < 0.5:
                        _unit = None
                    else:
                        _unit = random.choice(self.UNITS)
                elif t == 1 and i == n - 1:
                    _unit = random.choice(self.UNITS)
                else:
                    _unit = None
                parts.append((_type, _fmt, _unit))
        return self._augment_dimension(parts)


class SportScoreTagger(NumberRangeTagger):
    def __init__(self, strict_validate=False):
        super().__init__(True, strict_validate)

    @classmethod
    def valid_formats(cls):
        return [
            ("i_n", "i_n"),
            # ("i_n", "f_n"),
            # ("i_n", "i_big"),
            # ("i_n", "f_big"),
            # ("f_n", "i_n"),
            # ("f_n", "i_big"),
            # ("f_n", "f_n"),
            # ("f_n", "f_big"),
            # ("i_big", "i_n"),
            # ("i_big", "i_big"),
            # ("i_big", "f_n"),
            # ("i_big", "f_big"),
            # ("f_big", "i_n"),
            # ("f_big", "i_big"),
            # ("f_big", "f_n"),
            # ("f_big", "f_big"),
        ]

    @classmethod
    def tag(cls):  # type: ignore
        return "SPORT_SCORE"

    @classmethod
    def connector(cls):
        return "-"

    @classmethod
    def connector_content(cls):  # type: ignore
        return ""

    def validate(self, pattern_str):  # type: ignore
        """
        Những cái không phải là số thì sẽ là đơn vị
        """
        if pattern_str.strip() == "":
            return 1

        conn_str = self._find_connector_str(pattern_str)
        if conn_str is None:
            return 2

        parts = pattern_str.split(conn_str)
        if len(parts) != 2:
            return 3

        left, right = parts
        try:
            left_num, left_unit = self._split_num_and_unit(left.strip())
        except InvalidTagError:
            return 4
        try:
            right_num, right_unit = self._split_num_and_unit(right.strip())
        except InvalidTagError:
            return 5

        if self._find_nums_tags([left_num, right_num]) is None:
            # print("|", left, "|", left_num, "|", left_unit)
            # print("|", right, "|", right_num, "|", right_unit)
            return 6

        if left_unit != "" or right_unit != "":
            return 7

        return 0

    def _augment(self, pattern_str):
        if pattern_str is None:
            if random.random() < 0.5:
                return self._augment_range("i_n", "0:10", None, "i_n", "0:10", None)
            else:
                return self._augment_range("i_n", "0:100", None, "i_n", "0:100", None)

        left, right = pattern_str.split("||")

        def _format(fmt: str):
            t, f = fmt.split("=")
            if f == "":
                f = None
            return t, f, None

        args = list(_format(left)) + list(_format(right))
        return self._augment_range(*args)


class MoneyTagger(BaseNumberTagger):
    MONEY_UNITS = sorted(list(CURRENCY_UNITS.keys()), key=lambda x: len(x), reverse=True)
    MONEY_UNITS = [M.strip() for M in MONEY_UNITS]
    PREFIX_MONEY_UNITS = [m.strip() for m in PREFIX_CURRENCY_UNITS]
    SUFFIX_MONEY_UNITS = [m.strip() for m in SUFFIX_CURRENCY_UNITS]

    def __init__(self, strict_validate=True):
        super().__init__(True, strict_validate)

        skip_extract_unit = True
        self.number_taggers: dict[str, BaseNumberTagger] = {
            "f_big": Float_big_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
            "i_big": Integer_big_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
            "f_n": Float_n_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
            "i_n": Integer_n_Tagger(strict_validate=strict_validate, skip_extract_unit=skip_extract_unit),
        }

    @classmethod
    def tag(cls) -> str:
        return "MONEY"

    def _find_num_tag(self, num_str: str):
        for tagger in self.number_taggers.values():
            if tagger.validate(num_str) == 0:
                return tagger.tag()

    def _find_currencies(self, money: str):
        units = []
        for crc in self.MONEY_UNITS:
            if crc in money:
                money = money.replace(crc, "")
                units.append(crc)
        return units

    def validate(self, pattern_str: str) -> int:
        """Skip order"""
        if pattern_str.strip() == "":
            return 1

        units = self._find_currencies(pattern_str)
        if len(units) != 1:
            return 2

        unit = units[0]
        cnt = pattern_str.count(unit)
        if cnt != 1:
            return 3

        if pattern_str.startswith(unit) is False and pattern_str.endswith(unit) is False:
            return 4

        if pattern_str.startswith(unit):
            num = pattern_str[len(unit) :]
        elif pattern_str.endswith(unit):
            num = pattern_str[: -len(unit)].strip()
        else:
            return 5

        if len(num) == 0 or num[0] == " ":
            return 6

        if self._find_num_tag(num) is None:
            return 7
        # print(self._find_num_tag(num))

        return 0

    def _augment_money(self, num_type: str, num_fmt: str | None, currency: str, invert=False):
        if currency.strip() not in self.MONEY_UNITS:
            raise InvalidTagError(currency)

        if num_type not in self.number_taggers:
            raise InvalidTagError(f"Not found num_type = '{num_type}'. Expected one of {list(self.number_taggers.keys())}")

        num = self.number_taggers[num_type].augment(num_fmt)
        if currency.strip() in self.PREFIX_MONEY_UNITS:
            if not invert:
                content = currency + num.content
            else:
                content = num.content + currency
        else:
            if not invert:
                content = num.content + currency
            else:
                content = currency + num.content
        return Pattern(content=content.strip(), tag=self.tag(), start=0, end=None)

    def _augment(self, pattern_str: str) -> Pattern:
        """i_big=7.| vnd  |   f_n=-10:10f2| $!"""
        if pattern_str is None:
            currency = random.choice(self.MONEY_UNITS)
            invert = random.random() < 0.05
            if not invert:
                currency = random.choice(["", " "]) + currency
            return self._augment_money(num_type=random.choice(list(self.number_taggers.keys())), num_fmt=None, currency=currency)
        else:
            fmt, crc = pattern_str.split("|")
            _type, fmt = fmt.split("=")
            invert = False
            if crc[-1] == "!":
                crc = crc[:-1]
                invert = True
            return self._augment_money(num_type=_type, num_fmt=fmt, currency=crc, invert=invert)


class MathExprTagger(BaseRangeTagger):
    CALCULATIVE_OPERATORS = {
        "+": "cộng",
        "-": "trừ",
        "*": "nhân",
        "x": "nhân",
        "/": "chia",
        # ":": "chia",
        "^": "mũ",
        "=": "bằng",
        "~": "xấp xỉ",
    }
    COMPARATIVE_OPERATORS = {
        "<": "nhỏ hơn",
        ">": "lớn hơn",
    }
    STAND_ALONE_OPERATORS = {**CALCULATIVE_OPERATORS, **COMPARATIVE_OPERATORS}
    OPEN_BRACKET_OPERATORS = {
        "(": "mở ngoặc",
        "[": "mở ngoặc",
        "{": "mở ngoặc",
    }
    CLOSED_BRACKET_OPERATORS = {
        ")": "đóng ngoặc",
        "]": "đóng ngoặc",
        "}": "đóng ngoặc",
    }
    PAIRS = ["()", "[]", "{}"]
    assert len(OPEN_BRACKET_OPERATORS) == len(CLOSED_BRACKET_OPERATORS)
    OPERATORS_DICT = {**STAND_ALONE_OPERATORS, **OPEN_BRACKET_OPERATORS, **CLOSED_BRACKET_OPERATORS}

    OPERATOR = "operator"
    OPERAND = "operand"

    def __init__(self, strict_validate=True, strict_unit=True):
        super().__init__(strict_validate)

        self.measurement_tagger = MeasurementTagger(strict_unit=strict_unit)
        self.number_taggers: dict[str, BaseNumberTagger] = {
            "f_big": Float_big_Tagger(strict_validate=strict_validate, skip_extract_unit=True),
            "i_big": Integer_big_Tagger(strict_validate=strict_validate, skip_extract_unit=True),
            "f_n": Float_n_Tagger(strict_validate=strict_validate, skip_extract_unit=True),
            "i_n": Integer_n_Tagger(strict_validate=strict_validate, skip_extract_unit=True),
        }

    @classmethod
    def tag(cls):
        return "MATH_EXPR"

    @classmethod
    def connector(cls):  # type: ignore
        return None

    @classmethod
    def connector_content(cls):  # type: ignore
        return None

    def _find_operand_tag(self, operand: str):
        if self.measurement_tagger.validate(operand) == 0:
            return self.measurement_tagger.tag()
        for tagger in self.number_taggers.values():
            if tagger.validate(operand) == 0:
                return tagger.tag()

    def _split_operands(self, math_expr: str):
        parts: list[tuple[str, str]] = []
        operand = ""
        for ch in math_expr:
            if ch in self.OPERATORS_DICT:
                parts.append((operand, self.OPERAND))
                parts.append((ch, self.OPERATOR))
                operand = ""
            else:
                operand += ch
        if operand != "":
            parts.append((operand, self.OPERAND))
        if parts[0][0] == "":
            parts = parts[1:]
        return parts

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        ops = self._split_operands(pattern_str)
        # print(ops)
        p_sum = 0
        for op, tag in ops:
            if tag == self.OPERAND:
                op = op.strip()
                if op == "":
                    continue
                if self._find_operand_tag(op) is None:
                    return 2
            else:
                if op in self.OPEN_BRACKET_OPERATORS:
                    p_sum += 1
                elif op in self.CLOSED_BRACKET_OPERATORS:
                    p_sum -= 1
                if p_sum < 0:
                    return 3
        if p_sum != 0:
            return 4

        return 0

    def _split(self, pattern_str):
        ops = self._split_operands(pattern_str)

        parts = []
        start = 0

        for op, tag in ops:
            if tag == self.OPERAND:
                operand = op.strip()
                if operand != "":
                    operand_tag = self._find_operand_tag(operand)
                    parts.append(Pattern(content=operand, tag=operand_tag, start=start + len(op) - len(op.lstrip()), end=None))
                start += len(op)
            else:
                parts.append(
                    Pattern(content=op, tag=Connector.tag(), start=start, end=None, normed_content=self.OPERATORS_DICT[op])
                )
                start += 1
        return parts

    def _augment(self, pattern_str):
        chosen_tagger = None
        chosen_format = None
        if pattern_str is not None:
            for k, tagger in self.number_taggers.items():
                if pattern_str.startswith(k):
                    chosen_tagger = tagger
                    chosen_format = pattern_str[len(k) :]
                    if chosen_format == "":
                        chosen_format = None
                    else:
                        if chosen_format[0] != "=":
                            raise InvalidTagError(pattern_str)
                        chosen_format = chosen_format[1:]

        n_operators = random.randint(1, 8)
        n_operands = n_operators + 1
        operands = []
        taggers = list(self.number_taggers.values())
        for _ in range(n_operators + 1):
            if chosen_tagger is not None:
                operands.append(chosen_tagger.augment(chosen_format))
            else:
                if random.random() < 0.05:
                    tagger = self.measurement_tagger
                    operands.append(tagger.augment(random.choice(list(self.number_taggers.keys()))))
                else:
                    tagger = random.choice(taggers)
                    operands.append(tagger.augment(None))

        sa_operators = [random.choice(list(self.CALCULATIVE_OPERATORS.keys())) for _ in range(n_operators)]

        p_operators = []
        if random.random() < 0.4:
            P = random.randint(1, n_operands // 2)
        else:
            P = 0
        for _ in range(P):
            t = 3
            p_ops = None
            while t > 0:
                t -= 1
                l = random.randint(0, n_operands - 2)
                r = random.randint(l + 1, n_operands - 1)
                oke = True
                for u, v, _, _ in p_operators:
                    if u == l and v == r:
                        oke = False
                        break
                    elif r < u or v < l:
                        pass
                    elif u <= l and r <= v:
                        pass
                    elif l <= u and r <= v:
                        pass
                    else:
                        oke = False
                        break
                if oke:
                    ps = random.choice(self.PAIRS)
                    p_ops = (l, r, ps[0], ps[1])
                    break
            if p_ops is not None:
                p_operators.append(p_ops)
        p_ops_flatten = []
        L = 0
        R = 1
        for u, v, ou, ov in p_operators:
            p_ops_flatten.append((u, -v, L, ou))
            p_ops_flatten.append((v, -u, R, ov))
        p_ops_flatten = sorted(p_ops_flatten, reverse=True)

        components = [(operands[0], self.OPERAND)]
        for operand, operator in zip(operands[1:], sa_operators):
            components.append((operator, self.OPERATOR))
            components.append((operand, self.OPERAND))

        for i, _, p, op in p_ops_flatten:
            if p == R:
                i = i * 2 + 1
                components.insert(i, (op, self.OPERATOR))
            else:
                i = i * 2
                components.insert(i, (op, self.OPERATOR))

        content = ""
        parts = []
        for idx, (op, code) in enumerate(components):
            if code == self.OPERATOR:
                parts.append(
                    Pattern(
                        content=op.strip(),
                        tag=Connector.tag(),
                        start=len(content),
                        end=None,
                        normed_content=self.OPERATORS_DICT[op],
                    )
                )
                content += op.strip()
            else:
                parts.append(Pattern(content=op.content, tag=op.tag, start=len(content), end=None))
                content += op.content
            if idx < len(components) - 1:
                content += random.choice(["", " "])
        return RangePattern(content=content, tag=self.tag(), start=0, end=None, parts=parts)
    

class MathExprTaggerV2(MathExprTagger):
    @classmethod
    def _validate_on_augment(cls) -> bool:
        return False
    
    def _augment(self, pattern_str):
        pattern = super()._augment(pattern_str)
        operators = [s.strip() for s in self.STAND_ALONE_OPERATORS]
        parts = [p for p in pattern.parts if p.tag != Connector.tag() or p.content in operators]
        
        # for p in parts:
        #     print(p)

        for index, part in enumerate(parts):
            
            if index % 2 == 0:
                if part.tag not in [
                    Integer_n_Tagger.tag(),
                    Integer_big_Tagger.tag(),
                    Float_n_Tagger.tag(),
                    Float_big_Tagger.tag(),
                    MeasurementTagger.tag()
                ]:
                    raise InvalidTagError()
            else:
                if part.tag != Connector.tag():
                    print("ERROR")
                    for p in pattern.parts:
                        print(p)
                    raise InvalidTagError(part.tag)

        spaces = [random.randint(0, 1) for _ in range(len(parts) // 2)]
        splits = [[parts[0]]]
        for index, sp in enumerate(spaces):
            op = parts[index * 2 + 1]
            num = parts[index * 2 + 2]
            if sp == 0 and num.content[0].isdigit():
                splits[-1].append(op)
                splits[-1].append(num)
            else:
                splits.append([op])
                splits.append([num])

        new_parts = []
        content = ""
        for split in splits:
            if len(split) > 1:
                part = Pattern(content="".join([p.content for p in split]), tag=self.tag(), start=len(content), end=None)
                new_parts.append(part)
                content += part.content
            else:
                part = split[0].reset_start(len(content))
                if part.tag == Connector.tag():
                    part = part.reset_tag(self.tag())
                new_parts.append(part)
                content += part.content
            content += " "

        return RangePattern(content=content.strip(), tag=self.tag(), start=0, end=None, parts=new_parts)

