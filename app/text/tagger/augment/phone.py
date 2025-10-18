import random
from .base import (
    Pattern,
    BaseTagger, 
)


class PhoneTagger(BaseTagger):
    SPLITTER = [" ", ".", "-"]

    @classmethod
    def tag(cls) -> str:
        return "PHONE"

    def validate(self, pattern_str: str) -> int:
        valid = list("0123456789x") + self.SPLITTER
        
        cnt = pattern_str.count("+")
        if cnt > 1:
            return 1
        if cnt == 1:
            start = 0
            while start < len(pattern_str) and pattern_str[start] == "(":
                start += 1
            if pattern_str[start] != "+":
                return 2

        s = 0
        for ch in pattern_str:
            if ch == "(":
                s += 1
            elif ch == ")":
                s -= 1
            elif ch != "+" and ch not in valid:
                return 3
            if s not in [0, 1]:
                return 4
        
        return 0
    
    @staticmethod
    def _rand_ints(n: int, first_is_zero=False):
        p = "".join([str(random.randint(0, 9)) for _ in range(n-1)])
        if first_is_zero:
            p = "0" + p
        else:
            p = str(random.randint(0, 9)) + p
        return p
    
    def _augment(self, pattern_str: str) -> Pattern:
        has_plus = (random.random() < 0.1)
        has_bracket = (random.random() < 0.1)
        splitter = random.choice([""] + self.SPLITTER)
        phone_format_dict = {
            3: [(3,)],
            7: [(1, 3, 3)],
            8: [(4, 4)],
            10: [(4, 3, 3), (3, 3, 4), (4, 2, 2, 2)],
            11: [(5, 3, 3), (3, 4, 4), (2, 3, 3, 3)],
        }
        ratio = {
            3: 0.05, 
            7: 0.05,
            8: 0.3,
            10: 0.4,
            11: 0.2
        }
        assert sum(ratio.values()) == 1.0, sum(ratio.values())
        
        rand = random.random()
        s = 0
        N = 10
        for k, r in ratio.items():
            s += r
            if rand < s:
                N = k
                break
        fmt = random.choice(phone_format_dict[N])
        parts = [self._rand_ints(n, first_is_zero=(i == 0 and n != 1)) for i, n in enumerate(fmt)]
        phone = splitter.join(parts)

        if N == 3:
            has_plus = False
        
        if has_plus and has_bracket:
            s = "(+" + self._rand_ints(2) + ")" + random.choice(["", " "])
            phone = s + phone
        elif has_plus:
            s = "+" + self._rand_ints(2)
            phone = s + phone
        elif has_bracket:
            n = random.randint(2, 3)
            if n >= fmt[0]:
                phone = "(" + phone[:n] + ")" + random.choice(["", " "]) + phone[n:]
            else:
                phone = "(" + self._rand_ints(n) + ")" + random.choice(["", " "]) + phone
        
        phone = phone.strip()
        phone = " ".join(phone.split())
        return Pattern(content=phone, tag=self.tag(), start=0, end=None)
        