import random
from .base import (
    Pattern,
    InvalidTagError,
    BaseTagger,
    RangePattern,
    BaseRangeTagger,
    Connector,
)
from .number import Integer_n_Tagger, Integer_big_Tagger, Float_n_Tagger


class TimeTagger(BaseTagger):
    CONNECTOR = ":"
    LETTERS_HOUR = ["h", "g"]
    LETTERS_MINUTE = ["p", "m"]
    LETTERS_SECOND = ["s"]

    def __init__(self, strict_validate=True):
        super().__init__(strict_validate)

        self.number_taggers: dict[str, BaseTagger] = {
            "i_n": Integer_n_Tagger(skip_extract_unit=True),
            "i_big": Integer_big_Tagger(skip_extract_unit=True),
            "f_n": Float_n_Tagger(skip_extract_unit=True),
        }

    @classmethod
    def tag(cls):
        return "TIME_hms"

    def _find_num_tag(self, num_str: str):
        for tagger in self.number_taggers.values():
            if tagger.validate(num_str) == 0:
                return tagger.tag()

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        if self.CONNECTOR in pattern_str:
            parts = pattern_str.split(self.CONNECTOR)
            if len(parts) > 3:
                return 2

            for part in parts:
                if part.strip() != part:
                    return 3
                if self._find_num_tag(part.strip()) is None:
                    return 4
        else:
            tmp_str = pattern_str
            sep = "~"
            for ch in self.LETTERS_HOUR + self.LETTERS_MINUTE + self.LETTERS_SECOND:
                tmp_str = tmp_str.replace(ch, sep)

            if sep not in tmp_str:
                return 5

            parts = tmp_str.split(sep)
            if len(parts) > 0 and parts[0] == "":
                parts = parts[1:]
            if len(parts) > 0 and parts[-1] == "":
                parts = parts[:-1]
            for part in parts:
                if self._find_num_tag(part) is None:
                    return 6

            positions = [pattern_str.find(ch) for ch in self.LETTERS_HOUR]
            positions = [p for p in positions if p > -1]
            if len(positions) > 1:
                return 7
            h_pos = positions[0] if len(positions) == 1 else None

            positions = [pattern_str.find(ch) for ch in self.LETTERS_MINUTE]
            positions = [p for p in positions if p > -1]
            if len(positions) > 1:
                return 8
            m_pos = positions[0] if len(positions) == 1 else None

            positions = [pattern_str.find(ch) for ch in self.LETTERS_SECOND]
            positions = [p for p in positions if p > -1]
            if len(positions) > 1:
                return 9
            s_pos = positions[0] if len(positions) == 1 else None

            positions = [p for p in [h_pos, m_pos, s_pos] if p is not None]
            if len(positions) == 0:
                return 10

            if positions != sorted(positions):  # 12m4h => invalid
                return 11

        return 0

    def _random_num(self, _type: str | None, _fmt: str | None):
        if _type is None:
            return
        if _type not in self.number_taggers:
            raise InvalidTagError(_type)
        return self.number_taggers[_type].augment(_fmt)

    def _augment_time(
        self,
        h_type: str | None,
        h_fmt: str | None,
        m_type: str | None,
        m_fmt: str | None,
        s_type: str | None,
        s_fmt: str | None,
        use_connector: bool,
    ):
        components = [
            (self._random_num(h_type, h_fmt), "h"),
            (self._random_num(m_type, m_fmt), "m"),
            (self._random_num(s_type, s_fmt), "s"),
        ]
        num_none = sum([1 if com is None else 0 for com, _ in components])
        if num_none == len(components):
            raise InvalidTagError()
        if num_none == len(components) - 1:
            use_connector = False
        first = True
        content = ""
        for com, t in components:
            if com is not None:
                if first is False:
                    if com.content[0] not in "1234567890":
                        cnt = com.content[1:]
                    else:
                        cnt = com.content
                else:
                    cnt = com.content
                if len(cnt) == 1 and random.random() < 0.5:
                    cnt += "0" + cnt
                content += cnt
                first = False
                if use_connector:
                    content += self.CONNECTOR
                else:
                    content += t + (" " if random.random() < 0.5 else "")
        if content == "":
            raise InvalidTagError()
        if use_connector:
            if content[-1] != self.CONNECTOR:
                raise InvalidTagError()
            content = content[:-1]
        return Pattern(content=content.strip(), tag=self.tag(), start=0, end=None)

    def _augment(self, pattern_str):
        """i_n=1:10#h|i_n#m|f_n#s|c=0"""
        if pattern_str is None:
            index = random.randint(0, 2)
            hms_types = [None for _ in range(3)]
            for i in range(index - 1):
                hms_types[i] = random.choice([None, "i_n"])
            hms_types[index] = random.choice(["i_n", "f_n"])
            return self._augment_time(
                h_type=hms_types[0],
                h_fmt=None,
                m_type=hms_types[1],
                m_fmt=None,
                s_type=hms_types[2],
                s_fmt=None,
                use_connector=random.choice([True, False]),
            )
        elif pattern_str in self.number_taggers:
            fmts = ["h", "hm", "hs", "hms", "m", "ms", "s"]
            fmt = random.choice(fmts)
            type_kwargs = {k + "_type": pattern_str if k in fmt else None for k in "hms"}
            beauty_fmt = {
                "i_n": dict(
                    h_fmt="0:24",
                    m_fmt="0:60",
                    s_fmt="0:60",
                )
            }
            fmt_kwargs = {k + "_fmt": None for k in "hms"}
            if pattern_str in beauty_fmt and random.random() < 0.6:
                fmt_kwargs.update(beauty_fmt[pattern_str])

            return self._augment_time(**type_kwargs, **fmt_kwargs, use_connector=random.choice([True, False]))
        else:
            fmts = ["h", "hm", "hs", "hms", "m", "ms", "s"]
            parts = pattern_str.split("|")
            if len(parts) < 2:
                raise InvalidTagError(pattern_str)
            parts, use_conn = parts[:-1], parts[-1]
            if use_conn == "c=0":
                use_conn = False
            elif use_conn == "c=1":
                use_conn = True
            else:
                raise InvalidTagError(pattern_str)
            type_kwargs = {k: (None, None) for k in "hms"}
            fmt = ""
            for part in parts:
                num, t = part.split("#")
                if "=" in num:
                    _type, _fmt = num.split("=")
                else:
                    _type, _fmt = num, None
                fmt += t
                type_kwargs[t] = (_type, _fmt)
            if fmt not in fmts:
                raise InvalidTagError(pattern_str)
            new_kwargs = {}
            for k, (t, f) in type_kwargs.items():
                new_kwargs[k + "_type"] = t
                new_kwargs[k + "_fmt"] = f
            return self._augment_time(**new_kwargs, use_connector=use_conn)


class TimeRangeTagger(BaseRangeTagger):
    def __init__(self, strict_validate=True):
        super().__init__(strict_validate)

        self.time_tagger = TimeTagger(strict_validate=strict_validate)

    @classmethod
    def tag(cls):
        return "TIME_RANGE"

    @classmethod
    def connector(cls):
        return "-"

    @classmethod
    def connector_content(cls):
        return "đến"
    
    @classmethod
    def _validate_on_augment(cls):
        return False

    def validate(self, pattern_str):
        if pattern_str.strip() == "":
            return 1

        if self.connector() not in pattern_str:
            return 2
        parts = pattern_str.split(self.connector())
        if len(parts) != 2:
            return 3
        if self.time_tagger.validate(parts[0].strip()) != 0:
            return 4
        if self.time_tagger.validate(parts[1].strip()) != 0:
            return 5
        return 0

    def _split(self, pattern_str):
        left, right = pattern_str.split(self.connector())
        parts = []
        start = 0

        parts.append(Pattern(content=left.strip(), tag=TimeTagger.tag(), start=0, end=None))
        start += len(left)
        parts.append(
            Pattern(content=self.connector(), tag=Connector.tag(), start=start, end=None, normed_content=self.connector_content())
        )
        start += 1 + len(right) - len(right.lstrip())
        parts.append(Pattern(content=right.strip(), tag=TimeTagger.tag(), start=start, end=None))

        return parts

    def _augment(self, pattern_str):
        if pattern_str is None:
            left, right = None, None
        else:
            left, right = pattern_str.split("||")

        left = self.time_tagger.augment(left)
        right = self.time_tagger.augment(right)

        if right.content[0] not in "1234567890":
            conn_str = " " + self.connector() + " "
        else:
            # conn_str = Connector.random_connector_with_spaces(self.connector())
            space = random.choice([" ", ""])
            conn_str = space + self.connector() + space

        content = ""
        parts = []

        parts.append(left)
        content += left.content

        parts.append(
            Pattern(
                content=self.connector(),
                tag=Connector.tag(),
                start=len(content) + len(conn_str) - len(conn_str.lstrip()),
                end=None,
                normed_content="đến",
            )
        )
        content += conn_str

        parts.append(right.reset_start(len(content)))
        content += right.content

        return RangePattern(
            content=content,
            tag=self.tag(),
            start=0,
            end=None,
            parts=parts,
        )
