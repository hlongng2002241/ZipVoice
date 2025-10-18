import re
import random
from .base import (
    Pattern,
    BaseTagger, 
)


class PlateTagger(BaseTagger):
    P_PLATE = (
        r"([0-9]+[\-]{0,1}[a-zA-Z]{1,2}[0-9]*)"
        r"(\s{0,1}[\-]{0,1}\s{0,1})"
        r"([0-9]+\.*[0-9]+)"
    )
    
    @classmethod
    def tag(cls) -> str:
        return "PLATE"
    
    def validate(self, pattern_str: str) -> int:
        m = re.match(self.P_PLATE, pattern_str)
        if m is None:
            return 1
        return 0
    
    def _augment(self, pattern_str: str) -> Pattern:
        n = str(random.randint(1, 99))
        if len(n) == 1 and random.random() < 0.5:
            n = "0" + n
        
        chs = "qwertyuiopasdfghjklzxcvbnm"
        chs = list(chs.upper() + chs)
        c = random.choice(chs)
        if random.random() < 0.6:
            c += str(random.randint(1, 9))
        else:
            c += random.choice(chs)
        
        if random.random() < 0.1:
            s = n + "-" + c
        else:
            s = n + c
        
        e = "".join([str(random.randint(0, 9)) for _ in range(5)])
        if random.random() < 0.5:
            e = e[:3] + "." + e[3:]

        content = (
            s
            + random.choice(["", " "])
            + random.choice(["", "-"])
            + random.choice(["", " "])
            + e
        )
        content = " ".join(content.split())
        return Pattern(content=content, tag=self.tag(), start=0, end=None)
        