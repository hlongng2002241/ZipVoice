import abc
import random
from .base import (
    Connector,
    InvalidTagError, 
    Pattern,
    RangePattern,
    BaseTagger, 
    BaseRangeTagger,
)
from .number import Integer_n_Tagger


class BaseDateTagger(BaseTagger):
    CONNECTORS = ["/", ".", "-"]

    def _validate_date(self, pattern_str: str, N: int) -> int:
        if pattern_str.strip() == "":
            return 1
        
        found = False
        for connector in self.CONNECTORS:
            if connector in pattern_str:
                if found:
                    return 2
                found = True

                parts = pattern_str.split(connector)
                if len(parts) != N:
                    return 3
                for p in parts:
                    try:
                        p = int(p.strip())
                    except:
                        return 4
        if not found:
            return 5
        return 0
    
    def _augment_date(self, fmt: str, connector: str | list[str] | None = None, space: str | None = None):
        parts = []
        if connector is None:
            connector = random.choice(self.CONNECTORS)
        elif isinstance(connector, list):
            connector = random.choice(connector)
            if connector not in self.CONNECTORS:
                raise InvalidTagError()
        
        if space is None:
            space = random.choice(["", " "])

        conn_str = space + connector + space

        for f in fmt:
            if f == "d":
                parts.append(self.random_day_month(1, 31))
            elif f == "m":
                parts.append(self.random_day_month(1, 12))
            elif f == "y":
                year = random.randint(1800, 1999) if random.random() < 0.3 else random.randint(2000, 2050)
                parts.append(str(year))
            # parts.append(self.random_connector_with_spaces(connector, space))
            parts.append(conn_str)

        content = "".join(parts[:-1])
        return Pattern(content=content, tag=self.tag(), start=0, end=len(content))
    
    @staticmethod
    def random_day_month(l: int, r: int):
        d = str(random.randint(l, r))
        if len(d) == 1 and random.random() < 0.5:
            d = "0" + d
        return d
    
    @classmethod
    def random_connector_with_spaces(cls, connector: str = None, space: str = None):
        if space is None:
            space = random.choice(["", " "])
        if connector is None:
            connector = random.choice(cls.CONNECTORS)
        else:
            if connector not in cls.CONNECTORS:
                raise InvalidTagError()
        return space + connector + space
        # return Connector.random_connector_with_spaces(connector, space)


class Date_dm_Tagger(BaseDateTagger):
    """[ngày]/[tháng]"""

    @classmethod
    def tag(cls):
        return "DATE_dm"
    
    def validate(self, pattern_str):
        return self._validate_date(pattern_str, 2)
    
    def _augment(self, pattern_str):
        return self._augment_date("dm")


class Date_dmy_Tagger(BaseDateTagger):
    """[ngày]/[tháng]/[năm]"""

    @classmethod
    def tag(cls):
        return "DATE_dmy"
    
    def validate(self, pattern_str):
        return self._validate_date(pattern_str, 3)

    def _augment(self, pattern_str):
        return self._augment_date("dmy")
    

class Date_my_Tagger(BaseDateTagger):
    """[tháng]/[năm]"""

    @classmethod
    def tag(cls):
        return "DATE_my"
    
    def validate(self, pattern_str):
        return self._validate_date(pattern_str, 2)

    def _augment(self, pattern_str):
        return self._augment_date("my")


class DateRange_y_y_Tagger(BaseRangeTagger):
    """[năm]-[năm]"""

    @classmethod
    def tag(cls):
        return "DATE_RANGE_y_y"
    
    @classmethod
    def connector(cls):
        return "-"

    @classmethod
    def connector_content(cls):
        return ""
    
    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1
        
        if self.connector() not in pattern_str:
            return 2
        parts = pattern_str.split(self.connector())
        if len(parts) != 2:
            return 3
        for p in parts:
            try:
                int(p.strip())
            except:
                return 4
        return 0
    
    def _split(self, pattern_str):
        if self.validate(pattern_str) != 0:
            raise InvalidTagError()
        y_left, y_right = pattern_str.split(self.connector())
        parts = []
        start = 0
        parts.append(Pattern(y_left.strip(), tag=Integer_n_Tagger.tag(), start=start, end=None))
        start += len(y_left)
        parts.append(Pattern(content=self.connector(), tag=Connector.tag(), start=start, end=None, normed_content=self.connector_content()))
        start += len(self.connector()) + len(y_right) - len(y_right.strip())
        parts.append(Pattern(content=y_right.strip(), tag=Integer_n_Tagger.tag(), start=start, end=None))

        return parts

    def _augment(self, pattern_str):
        content = ""
        y = random.randint(1800, 1999) if random.random() < 0.3 else random.randint(2000, 2050)
        content += str(y)
        conn = BaseDateTagger.random_connector_with_spaces(self.connector())
        content += conn
        y = random.randint(y, 2055) if random.random() < 0.3 else random.randint(max(y, 2000), 2055)
        content += str(y)

        return RangePattern(
            content=content, 
            tag=self.tag(), 
            start=0, 
            end=len(content), 
            parts=self.split(content), 
        )


class BaseDateRangeTagger(BaseRangeTagger):
    """
    INTEGER - DATE_0 = 1-2/6

    INTEGER - DATE_1 = 1-2/6/2025

    DATE_0 - DATE_0 = 1/6-2/7

    DATE_0 - DATE_1 = 1/6-2/7/2025

    DATE_1 - DATE_1 = 1/6/2025-1/7/2025

    INTEGER - DATE_2 = 3-4/2024

    DATE_2 - DATE_2 = 3/2004 - 4/2025
    """

    def __init__(self):
        super().__init__()

        self.format_to_tagger: dict[str, BaseTagger] = {
            "d": Integer_n_Tagger(skip_extract_unit=True),
            "m": Integer_n_Tagger(skip_extract_unit=True),
            "dm": Date_dm_Tagger(),
            "dmy": Date_dmy_Tagger(),
            "my": Date_my_Tagger(),
        }

    @classmethod
    def connector(cls):
        return "-"

    @classmethod
    def connector_content(cls):
        return "đến"
    
    @classmethod
    @abc.abstractmethod
    def valid_formats(cls) -> list[tuple[str, str]]:
        """ Example: ("d", "dm"), ("dm", "dmy") """
    
    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1
        if self.connector() not in pattern_str:
            return 2
        parts = pattern_str.split(self.connector())
        if len(parts) != 2:
            return 3
        left, right = parts

        oke = False
        for fmt_left, fmt_right in self.valid_formats():
            t_left = self.format_to_tagger[fmt_left]
            t_right = self.format_to_tagger[fmt_right]
            if t_left.validate(left.strip()) == 0 and t_right.validate(right.strip()) == 0:
                oke = True
                break
        if not oke:
            return 4
        return 0
    
    def _split(self, pattern_str):
        left, right = pattern_str.split(self.connector())
        parts = []
        start = 0

        tag_left, tag_right = None, None
        for fmt_left, fmt_right in self.valid_formats():
            t_left = self.format_to_tagger[fmt_left]
            t_right = self.format_to_tagger[fmt_right]
            if t_left.validate(left) == 0 and t_right.validate(right) == 0:
                tag_left, tag_right = t_left.tag(), t_right.tag()
        if tag_left is not None and tag_right is not None:
            pass
        else:
            raise InvalidTagError()

        parts.append(Pattern(content=left.strip(), tag=tag_left, start=0, end=None))
        start += len(left)
        parts.append(Pattern(content=self.connector(), tag=Connector.tag(), start=start, end=None, normed_content=self.connector_content()))
        start += len(self.connector()) + len(right) - len(right.strip())
        parts.append(Pattern(content=right.strip(), tag=tag_right, start=start, end=None))

        return parts

    def _augment(self, pattern_str):
        if pattern_str is not None:
            fmt_left, fmt_right = pattern_str.split(self.connector())
            if fmt_left not in self.format_to_tagger:
                raise InvalidTagError()
            if fmt_right not in self.format_to_tagger:
                raise InvalidTagError()
            if (fmt_left, fmt_right) not in self.valid_formats():
                raise InvalidTagError(f"Invalid format '{pattern_str}'")
        else:
            fmt_left, fmt_right = random.choice(self.valid_formats())

        def _augment(tagger, connector, space):
            if isinstance(tagger, Integer_n_Tagger):
                content = BaseDateTagger.random_day_month(1, 31)
                return Pattern(content=content, tag=Integer_n_Tagger.tag(), start=0, end=None)
            if isinstance(tagger, Date_dm_Tagger):
                return tagger._augment_date("dm", connector=connector, space=space)
            if isinstance(tagger, Date_dmy_Tagger):
                return tagger._augment_date("dmy", connector=connector, space=space)
            if isinstance(tagger, Date_my_Tagger):
                return tagger._augment_date("my", connector=connector, space=space)
            raise NotImplementedError()

        if random.random() < 0.5:
            # 1-2 - 1-2-2025 is oke
            if random.random() < 0.5:
                left = _augment(self.format_to_tagger[fmt_left], connector=["-"], space="")
            else:
                left = _augment(self.format_to_tagger[fmt_left], connector=["/", "."], space="")

            conn_str = BaseDateTagger.random_connector_with_spaces(self.connector(), space=" ")

            if random.random() < 0.5:
                right = _augment(self.format_to_tagger[fmt_right], connector=["-"], space="")
            else:
                right = _augment(self.format_to_tagger[fmt_right], connector=["/", "."], space="")
                
        else:
            connectors = ["/", "."]
            space = ""
            left = _augment(self.format_to_tagger[fmt_left], connector=connectors, space=space)
            if random.random() < 0.5:
                conn_str = BaseDateTagger.random_connector_with_spaces(self.connector(), space="")
            else:
                conn_str = BaseDateTagger.random_connector_with_spaces(self.connector(), space=" ")
            right = _augment(self.format_to_tagger[fmt_right], connector=connectors, space=space)

        parts = []
        start = 0

        parts.append(left)
        start += len(left.content)

        p = len(conn_str) - len(conn_str.lstrip())
        parts.append(Pattern(content=self.connector(), tag=Connector.tag(), start=p + start, end=None, normed_content=self.connector_content()))
        start += len(conn_str)

        parts.append(right.shift_index(start))

        content = left.content + conn_str + right.content

        return RangePattern(content, tag=self.tag(), start=0, end=None, parts=parts)


class DateRange_dm_dmy_Tagger(BaseDateRangeTagger):
    @classmethod
    def tag(cls):
        return "DATE_RANGE_dm_dmy"

    @classmethod
    def valid_formats(cls):
        return [("d", "dm"), ("d", "dmy"), ("dm", "dm"), ("dm", "dmy"), ("dmy", "dmy")]
    

class DateRange_m_my_Tagger(BaseDateRangeTagger):
    @classmethod
    def tag(cls):
        return "DATE_RANGE_m_my"

    @classmethod
    def valid_formats(cls):
        return [("m", "my"), ("my", "my")]
