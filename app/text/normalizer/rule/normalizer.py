import re
import string
import logging
from roman import fromRoman, InvalidRomanNumeralError
from nltk import word_tokenize

from .cores import *
from .utils.characters import ALPHABET_DICT, NUMBER_DICT
from .utils.units import UNITS_DICT
from .utils.measure import MEASURES_DICT
from .utils.verbatim import VERBATIM
from .utils.names import NAMES_DICT
from .utils.abbre import ABBRE_DICT


logger = logging.getLogger(__name__)


def _load_vocab():
    words = []
    for path in ["data/lexicon_en.tsv", "data/lexicon_vi.tsv"]:
        with open(path) as f:
            for line in f.readlines():
                word, _ = line.strip().split("\t")
                words.append(word.lower())
    return words


LOWER_VOCAB_SET = set(_load_vocab())


class RuleBasedTextNormalizer():
    RAW_ENDING_PUNCTUATIONS_REGEX = r"\s|\.|\,|\)|\;|\:|\?|\!|\]|\}"
    RAW_ENDING_PUNCTUATIONS_NO_SPACE_REGEX = r"\s|\.|\,|\)|\;|\:|\?|\!|\]|\}"
    ENDING_PUNCTUATIONS_REGEX   = r"([" + RAW_ENDING_PUNCTUATIONS_REGEX + r"])"
    # ENDING_PUNCTUATIONS_REGEX = r"\b"

    RAW_LINKING_WORDS_REGEX = (
        r"[Vv]à|"
        r"[Hh]oặc|"
        r"[Vv]ới|"
        r"[Cc]ùng|"
        r"[Hh]ay|"
        r"[Đđ]ến|"
        r"[Cc]ùng [Vv]ới|"
        r"[Hh]oặc [Ll]à|"
        r"[Hh]ay [Ll]à|"
        r"[Hh]ay [Vv]ới|"
        r"[Hh]oặc [Vv]ới|"
        r"[Vv]à [Vv]ới|"
        r"[Tt]hậm chí"
    )
    LINKING_WORDS_REGEX = r"(" + RAW_LINKING_WORDS_REGEX + r")"

    RAW_COMPARATIVE_WORDS_REGEX = r"([Hh]ơn|[Kk]ém|[Bb]ằng)"

    RAW_UNITS_REGEX = "|".join([u.strip() for u in UNITS_DICT.keys()])
    RAW_UNITS_VALUE_REGEX = "|".join([u.strip() for u in UNITS_DICT.values()])

    RAW_MEASUREMENTS_REGEX = "|".join([u.strip() for u in MEASURES_DICT.keys()])

    # r"\s+.*?\s*\(*\s*" # greedy mark .* -> turn into .*?
    IN_BETWEEN_CONTENT_REGEX = r"(\s|\,|\:)+(()|.*?\s+)"

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def lowercase(self, input_str: str):
        return input_str.strip().lower().strip()

    def tokenize(self, input_str: str):
        """
        'abc@gmail.com' --> 'abc @ gmail.com'
        """
        tokens = word_tokenize(input_str)
        output_str = " ".join(tokens)

        return output_str

    def remove_multi_space(self, input_str: str):
        return re.sub(' +', ' ', input_str)

    def separate_numbers_adjacent_chars(self, sentence):
        separated_sentence = re.sub(r'(\d)(?=\D)|(?<=\D)(\d)', r'\1 \2', sentence)
        return separated_sentence

    def remove_special_characters(self, input_str: str):
        input_str = ' ' + input_str + ' '
        input_str = input_str.replace('–', '-')
        # punct = '! " “ \' ( ) ; [ ] * _ ` { | } ~ … 》 ≧ ≦ ‘ ’ · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
        # punct = '! " “ \' ( ) ; [ ] * _ ` { ~ } … 》 ≧ ≦ ‘ ’ ” · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
        punct = '“ … 》 ≧ ≦ ‘ ’ ” · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
        for e in punct:
            e = e.strip()
            input_str = input_str.replace(e, ' ')
        return input_str.strip()
    
    # def remove_special_characters_v2(self, input_str: str):
    #     raise NotImplementedError()
    #     return re.sub(r'[^\w\s,.?]', ' ', input_str)
    
    def normalize_duplicate_word(self, input_str: str):
        """
        Remove some common duplicated words
        """
        # input_str = remove_adjacent_word(input_str, "ngày")
        # input_str = remove_adjacent_word(input_str, "tháng")
        # # input_str = remove_adjacent_word(input_str, "năm")
        # input_str = remove_adjacent_word(input_str, "giờ")
        # input_str = remove_adjacent_word(input_str, "phút")
        # input_str = remove_adjacent_word(input_str, "giây")
        # input_str = remove_adjacent_word(input_str, ".")
        # input_str = remove_adjacent_word(input_str, "?")
        input_str = self.remove_redundant_words(input_str, 'ngày', 'ngày')
        input_str = self.remove_redundant_words(input_str, 'mùng', 'ngày')
        input_str = self.remove_redundant_words(input_str, 'tháng', 'tháng')
        return input_str.strip()
    
    def remove_redundant_words(self, text: str, word1: str, word2: str):
        # print('text: ' ,text)
        text = text.strip()
        new_text = []
        tokens = [i.strip() for i in text.split()]
        # print('tokens: ', tokens)
        if len(tokens) != 0:
            for i in range(len(tokens)-1):
                if tokens[i] == word1 and tokens[i+1] == word2:
                    continue
                else:
                    new_text.append(tokens[i+1])
        # print('tokens after: ', tokens)
        if tokens:
            new_text.insert(0, tokens[0])
        else:
            pass
        
        return ' '.join(new_text)

    def remove_emoji(self, input_str: str):
        """
        Remove emoji in text
        """
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', input_str)
    
    # def remove_emoticons(self, input_str):
    #     """
    #     Remove emoticons in text
    #     """
    #     emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    #     return emoticon_pattern.sub(r'', input_str)

    def remove_urls(self, input_str: str):
        """
        Remove urls in input_str
        """
        url_pattern = re.compile(r'(https|http)?://\S+|www\.\S+|[A-Za-z0-9]*@[A-Za-z]*\.+[A-Za-z0-9]*')
        return url_pattern.sub(r'', input_str)
    
    def normalize_urls(self, input_str: str):
        """
        """
        p = re.compile(r"(https|http)?://\S+|www\.\S+|[A-Za-z0-9]*@[A-Za-z]*\.+[A-Za-z0-9]*")
    
    # def remove_html(self, input_str: str):
    #     """
    #     Remove html in input_str
    #     """
    #     html_pattern = re.compile(r'<.*?>')
    #     return html_pattern.sub(r'', input_str)
    
    # def norm_vnmese_accent(self, input_str: str):
    #     output_str = VietnameseTextNormalizer.Normalize(input_str)
    #     return output_str

    def normalize_phone_number(self, input_str: str):
        """
        Normalize phone numbers
        
        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = (
            r"(\d{8,12}|"
            r"\d{4}\s\d{2}\s\d{2}\s\d{2}|"
            r"\d{2,5}\s\d{2,4}\s\d{2,4}\s\d{2,4}|"
            r"\d{5}\.\d{5}|"
            r"\d{4}\s\d{4}|"
            r"\d{3,5}\s\d{2,4}\s\d{2,4}|"
            r"\d{3}\s\d{3}\s\d{4}|"
            r"\d{3,5}\s\d{5,7}|"
            r"\d{3}\.\d{4}\.\d{3}|"
            r"\d{3}\.\d{3}\.\d{4}|"
            r"\d{4}\.\d{3}\.\d{3}|"
            r"\d{2}\.\d{4}\.\d{4}|"
            r"\d{4}\.\d{2}\.\d{4}|"
            r"\d{4}\.\d{4}\.\d{2}|"
            r"\d{4}\.\d{2}\.\d{2}\.\d{2})"
            # r"\d{3,4})"
        )
        p = re.compile(
            r"([Hh]otline|"
            r"[Tt]ổng [đĐ]ài|"
            r"[Đđ]iện [tT]hoại|"
            r"[sS]ố [đĐ]iện [tT]hoại|"
            r"[sS][dD][tT]|"
            r"[sS][đĐ][tT]|"
            r"[đĐ]ầu [sS]ố|"
            r"[zZ]alo|"
            r"[đĐ]ường [dD]ây [nN]óng|"
            r"[Ll]iên [hH]ệ|"
            r"[gG]ọi|"
            r"[cC]all|"
            # r"[cC]hi [tT]iết|"
            r"[hH]ỗ [tT]rợ|"
            r"[tT]ư [vV]ấn|"
            r"[lL]iên [lL]ạc|"
            # r"[cC]ông [tT]y|"
            # r"[bB]án [hH]àng|"
            # r"[đĐ]ặt [hH]àng"
            r")+"
            + self.IN_BETWEEN_CONTENT_REGEX
            + r"(\+?)"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"
            r"(\+?)"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        phone_number_list = []
        while True:
            phone_number = p.search(temp_str)
            if not phone_number:
                break
            x = phone_number.group(5) + phone_number.group(6)
            phone_number_list.append((x, phone_number.group()))
            temp_str = temp_str[phone_number.span()[1]-1:]

            while True:
                phone_number = p_addons.search(temp_str)
                if not phone_number:
                    break
                x = phone_number.group(2) + phone_number.group(3)
                phone_number_list.append((x, phone_number.group()))
                temp_str = temp_str[phone_number.span()[1]-1:]

        for phone_number, phone_number_repl in phone_number_list:
            phone_number_str = phone2words(phone_number)
            input_str = input_str.replace(phone_number_repl, phone_number_repl.replace(phone_number, ' ' + phone_number_str + ' '))

        return input_str.strip()
    
    def normalize_tag_verbatim(self, input_str: str):
        """
        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        for key, value in VERBATIM.items():
            input_str = input_str.replace(key.strip(), ' ' + value.strip() + ' ')
        return input_str.strip()
    
    def normalize_endline(self, input_str: str):
        """
        Normalize newlines to comma separated format
        """
        return re.sub(r"\n+", " , ", input_str)
    
    def normalize_connector(self, input_str: str):
        """
        Normalize connector characters
        """
        return input_str.replace('/', ' trên ')
    
    def normalize_spacing_puncs(self, input_str: str):
        """
        Add spaces around punctuation marks
        """
        punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^`{}~"
        return re.sub(f"([{punct}])", r" \1 ", input_str)
    
    def normalize_punctuation(self, input_str: str):
        """
        Normalize punctuation marks
        """
        text = input_str
        text = text.replace("?", ".").replace("!", ".")
        text = text.replace(":", ",").replace(";", ",")
        for p in string.punctuation:
            if p not in ".,":
                text = text.replace(p, ",")
        return text
    
    def normalize_AZ09(self, input_str: str):
        """
        Normalize sequences with forms [A-Z]{1}[0-9]{1,2} or [0-9]{1,2}[A-Z]{1},
        i.e. 'chung cư A10', 'cục phòng chống tội phạm công nghệ cao C50'

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        number_letter = re.findall(
            r"([a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}\d{1,10}|"
            r"\d{1,10}[a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}|"
            r"[a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}|"
            r"\d{1,10}[a-zA-Z]{1,10}\d{1,10}|"
            r"[a-zA-Z]{1,10}\d{1,10}|"
            r"\d{1,10}[a-zA-Z]{1,10})",
            
            input_str
        )
        
        for item in number_letter:
            input_str = input_str.replace(item,  ' ' + alpha_num2words(item).strip() + ' ')
                    
        return input_str.strip()

    def normalize_single_stand_letter(self, input_str: str):
        """
        NOTE. LongNH
        """
        for k, v in ALPHABET_DICT.items():
            input_str = re.sub(rf"\b{k}\b", " " + v + " ", input_str)
        return input_str.strip()

    def normalize_date(self, input_str: str):
        """
        Normalize dates.

        NOTE. LongNH
        """
        input_str = re.sub(r'\s+', ' ', input_str)
        input_str = self._norm_date_type_1(input_str)
        input_str = self._norm_date_type_2(input_str)
        input_str = self._norm_date_type_3(input_str)
        input_str = self._norm_date_type_4(input_str)
        input_str = self._norm_date_type_5(input_str)
        return input_str.strip()
    
    DATE_PREFIX_REGEX = (
        r"[Ss]ớm|"
        r"[Đđ]ến hết|"
        r"[Đđ]ợt|"
        r"[Pp]hiên|"
        r"[Nn]gày|"
        r"[Dd]ịp|"
        r"[Ss]áng|"
        r"[Tt]rưa|"
        r"[Cc]hiều|"
        r"[Tt]ối|"
        r"[Đđ]êm|"
        r"[Mm]ùng|"
        r"[Hh]ôm|"
        r"[Ss]áng qua|"
        r"[Tt]ưa qua|"
        r"[Cc]hiều qua|"
        r"[Tt]ối qua|"
        r"[Đđ]êm qua|"
        r"[Hh]ôm qua|"
        r"[Hh]ôm sau|"
        r"[Nn]gày kia|"
        r"[Hh]ôm kia|"
        r"[Vv]ào|"
        r"[Kk]éo dài tới|"
        r"[Dd]ự kiến tới|"
        r"[Nn]ay|"
        r"[Mm]ai|"
        r"[Đđ]ến|"
        r"[Tt]ới|"
        r"[Tt]ừ|"
        r"[Hh]ạn chót|"
        r"[Hh]ạn chót là|"
        r"[Đđ]ầu tuần|"
        r"[Cc]uối tuần|"
        r"[Tt]rước đó"
    )

    def _norm_date_type_1(self, input_str: str):
        """
        Normalize dd/mm/yy[yy] (dmy) form of dates

        Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = (
            r"("
                r"(0?[1-9]|[12]\d|3[01])\s*[\/.]\s*(0?[1-9]|[1][0-2])\s*[\/.]\s*(\d{4}|\d{2})|"
                r"(0?[1-9]|[12]\d|3[01])\s*[\-]\s*(0?[1-9]|[1][0-2])\s*[\-]\s*(\d{4}|\d{2})"
            r")"
        )
        p = re.compile(
            r"([Nn]gày|[Hh]ôm)"
            
            + r".*?\s"
            # + self.IN_BETWEEN_CONTENT_REGEX
            
            + r"\(*\s*"
            
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        dates_dmy = []

        while True:
            date = p.search(temp_str)
            if not date:
                break
            prefix = date.group(1)
            x = date.group(2)
            dates_dmy.append((prefix.strip().lower() in ["ngày", "hôm"], x, date.group()))
            temp_str = temp_str[date.span()[1]-1:]

            while True:
                date = p_addons.search(temp_str)
                if not date:
                    break
                x = date.group(2)
                dates_dmy.append((False, x, date.group()))
                temp_str = temp_str[date.span()[1]-1:]

        for has_prefix, date, date_repl in dates_dmy:
            date_str = date_dmy2words(date, add_day_prefix=not has_prefix)
            input_str = input_str.replace(date_repl, date_repl.replace(date, ' ' + date_str + ' '))
        return input_str.strip()
    
    def _norm_date_type_2(self, input_str: str):
        """
        Normalize dd/mm/yy[yy] (dmy) form of dates

        This function supports function type 1

        Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(
            r"[\s|\(]\s*"
            r"("
                r"(0?[1-9]|[12]\d|3[01])[\/](0?[1-9]|[1][0-2])[\/](\d{4}|\d{2})|"
                r"(0?[1-9]|[12]\d|3[01])[.](0?[1-9]|[1][0-2])[.](\d{4}|\d{2})|"
                r"(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{4}|\d{2})"
            r")"
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        dates_dmy = []
        while True:
            date = p.search(temp_str)
            if not date:
                break
            dates_dmy.append((date.group(1), date.group()))
            temp_str = temp_str[date.span()[1]-1:]
        
        for date, date_repl in dates_dmy:
            date_str = date_dmy2words(date)
            input_str = input_str.replace(date_repl, date_repl.replace(date, ' ' + date_str + ' '))
        return input_str.strip()

    def _norm_date_type_3(self, input_str: str):
        """
        Normalize dd/mm (dm) form of dates

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = r"((0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|[1][0-2]))"
        p = re.compile(
            r"(sau ,|"
            r"mai ,|"
            r"qua ,|"
            r"kia ,|"
            r"nay ,|"
            + self.DATE_PREFIX_REGEX +
            r")"

            + self.IN_BETWEEN_CONTENT_REGEX
            + r"\(*\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        dates_dm = []
        while True:
            date = p.search(temp_str)
            if not date:
                break
            dates_dm.append((date.group(5), date.group()))
            temp_str = temp_str[date.span()[1]-1:]

            while True:
                date = p_addons.search(temp_str)
                if not date:
                    break
                dates_dm.append((date.group(2), date.group()))
                temp_str = temp_str[date.span()[1]-1:]

        # Cases: "số ra các ngày 28 29-2 và 1-3", "ngày 19 và 20.3 tới"
        # dates_dm_special = re.findall(r'[Nn]gày .+ (và|với|cùng)+ (\d{1,2}[\/\-.]\d{1,2})\s', input_str)
        # dates_dm += [(d[1], d[1]) for d in dates_dm_special]

        for date, date_repl in dates_dm:
            date_str = date_dm2words(date, add_day_prefix=False)
            input_str = input_str.replace(date_repl, date_repl.replace(date, ' ' + date_str + ' '))

        return input_str.strip()

    def _norm_date_type_4(self, input_str: str):
        """
        Normalize dd/mm form without clear rules
        
        Cám ơn cha dành cho Ngày của cha (16/6) tới đây của Hồ Ngọc Hà...

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = r"((0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|[1][0-2]))"
        p = re.compile(
            r"([Nn]gày|[Hh]ôm).*?\s\(*\s*"
            
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        dates_dm = []
        temp_str = input_str
        while True:
            date = p.search(temp_str)
            if not date:
                break
            dates_dm.append((date.group(2), date.group()))
            temp_str = temp_str[date.span()[1]-1:]

            while True:
                date = p_addons.search(temp_str)
                if not date:
                    break
                dates_dm.append((date.group(2), date.group()))
                temp_str = temp_str[date.span()[1]-1:]

        for date, date_repl in dates_dm:
            date_str = date_dm2words(date, add_day_prefix=False)
            input_str = input_str.replace(date_repl, date_repl.replace(date, ' ' + date_str + ' '))

        return input_str.strip()

    def _norm_date_type_5(self, input_str: str):
        """
        Normalize mm/yyyy (my) form of dates

        @improve:
        
        những cases không có [Tt]háng ở trước --> thêm 'tháng' ở date_my2word()
        
        nhưng tránh các trường hợp Quý 2/2018, đợt 3/2019, tỷ lệ 1/2000

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = r"((0?[1-9]|[1][0-2])[\/\-.](\d{4}))"
        p = re.compile(
            r"([Tt]háng|[Qq]uý)"
            # r"(.*?)\s"
            r"(\s|\,|\:)+(()|.*?\s+)"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        dates_my = []
        while True:
            date = p.search(temp_str)
            if not date:
                break
            # print(date.groups())
            if (date.group(2) + date.group(3)).strip() == "":
                prefix = None
            else:
                prefix = date.group(1).lower()
            dates_my.append((date.group(5), prefix, date.group()))
            temp_str = temp_str[date.span()[1]-1:]

            while True:
                date = p_addons.search(temp_str)
                if not date:
                    break
                # print(date.groups())
                # if date.group(2) == "":
                #     prefix = None
                # else:
                #     prefix = date.group(1).lower()
                dates_my.append((date.group(2), prefix, date.group()))
                temp_str = temp_str[date.span()[1]-1:]

        for date, prefix, date_repl in dates_my:
            date_str = date_my2words(date, add_month_prefix=False)
            if prefix is not None:
                date_str = " " + prefix + " " + date_str
            input_str = input_str.replace(date_repl, date_repl.replace(date, ' ' + date_str + ' '))

        return input_str.strip()

    def normalize_date_range(self, input_str: str):
        """
        NOTE. LongNH
        """
        input_str = self._norm_date_range_type_1(input_str)
        input_str = self._norm_date_range_type_2(input_str)
        input_str = self._norm_date_range_type_3(input_str)
        input_str = self._norm_date_range_type_4(input_str)
        input_str = self._norm_date_range_type_5(input_str)
        input_str = self._norm_date_range_type_6(input_str)
        input_str = self._norm_date_range_type_7(input_str)
        return input_str.strip()
    
    def __has_ngay_prefix(self, prefix: str):
        if prefix is None:
            return False
        for p in ["ngày", "hôm"]:
            if p in prefix:
                return True
        return False
    
    def __has_thang_prefix(self, prefix: str):
        if prefix is None:
            return False
        for p in ["thang"]:
            if p in prefix:
                return True
        return False
    
    def __should_add_tu(self, prefix: str):
        if prefix is None:
            return True
        
        prefix = prefix.lower()
        if prefix in ["cuối tuần", "đầu tuần"]:
            return True
        return False

    def _norm_date_range_type_1(self, input_str: str):
        """
        Normalize date ranges.
        Normalize yyyy-yyyy forms: 2016-2017, 1912-1982 ngày sinh, v.v.
        Khi nào thì chèn từ vào ví dụ 'năm học 2018-2019' thì đọc luôn tên năm

        NOTE. LongNH
        """
        p = re.compile(
            r"([Nn]ăm)\s+([Hh]ọc)*"
            # r".*?\s*"
            r"(()|.*?\s+)"

            r"((\d{4})\s*\-\s*(\d{4}))"
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            r"((\d{4})\s*\-\s*(\d{4}))"
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        year_range_list = []

        while True:
            year_range = p.search(temp_str)
            if not year_range:
                break
            is_school_year = year_range.group(2) is not None
            year_range_list.append((year_range.group(5), is_school_year, year_range.group()))
            temp_str = temp_str[year_range.span()[1]-1:]

            while True:
                year_range = p_addons.search(temp_str)
                if not year_range:
                    break
                year_range_list.append((year_range.group(2), is_school_year, year_range.group()))
                temp_str = temp_str[year_range.span()[1]-1:]

        for year_range, is_school_year, year_range_repl in year_range_list:
            year_range_norm = year_range.replace('-', ' - ')
            year_range_norm = " ".join(year_range_norm.split())
            start_year = year_range_norm.split('-')[0]
            end_year = year_range_norm.split('-')[1]
            if is_school_year:
                year_range_str = ' ' + num2words_integer(start_year) + ' ' + num2words_integer(end_year)
            else:
                year_range_str = ' ' + num2words_integer(start_year) + ' đến năm ' + num2words_integer(end_year)
            input_str = input_str.replace(year_range_repl, year_range_repl.replace(year_range, ' ' + year_range_str + ' '))

        return input_str.strip()

    def _norm_date_range_type_2(self, input_str: str):
        """
        Normalize dd/mm/yyyy-dd/mm/yyyy forms

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = (
            r"("
                r"(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})"
                r"\s*\-\s*"
                r"(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})"
            r")"
        )
        p = re.compile(
            r"(" + self.DATE_PREFIX_REGEX + r")*\s+"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        date_range_dmy_list = []
        
        while True:
            date_range_dmy = p.search(temp_str)
            if not date_range_dmy:
                break
            prefix = self.__has_ngay_prefix(date_range_dmy.group(1))
            date_range_dmy_list.append((date_range_dmy.group(2), prefix, date_range_dmy.group()))
            temp_str = temp_str[date_range_dmy.span()[1]-1:]

        for date_range_dmy, has_ngay, date_range_dmy_repl in date_range_dmy_list:
            start_date = date_range_dmy.split('-')[0]
            end_date = date_range_dmy.split('-')[1]
            date_range_dmy_str = date_dmy2words(start_date, add_day_prefix=not has_ngay) + ' đến ' + date_dmy2words(end_date)
            input_str = input_str.replace(date_range_dmy_repl, date_range_dmy_repl.replace(date_range_dmy, ' ' + date_range_dmy_str + ' '))
                
        return input_str.strip()

    def _norm_date_range_type_3(self, input_str: str):
        """
        Normalize dd-dd/mm/yyyy forms

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = r"((0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4}))"
        p = re.compile(
            r"(" + self.DATE_PREFIX_REGEX + r")*\s+"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        date_range_dmy_list = []

        while True:
            date_range_dmy = p.search(temp_str)
            if not date_range_dmy:
                break
            # has_prefix = date_range_dmy.group(1) is not None 
            has_prefix = not self.__should_add_tu(date_range_dmy.group(1))
            has_ngay_prefix = self.__has_ngay_prefix(date_range_dmy.group(1))
            date_range_dmy_list.append((
                date_range_dmy.group(2), 
                has_ngay_prefix, 
                has_prefix,
                date_range_dmy.group()),
            )
            temp_str = temp_str[date_range_dmy.span()[1]-1:]

        for date_range_dmy, has_ngay, has_prefix, date_range_repl in date_range_dmy_list:
            start_date = date_range_dmy.split('-')[0]
            end_date = date_range_dmy.split('-')[1]
            date_range_dmy_str = num2words_integer(start_date) + ' đến ' + date_dmy2words(end_date)
            if int(start_date) < 10:
                date_range_dmy_str = " mùng " + date_range_dmy_str
            if not has_ngay:
                date_range_dmy_str = " ngày " + date_range_dmy_str
            if not has_prefix:
                date_range_dmy_str = " từ " + date_range_dmy_str
            input_str = input_str.replace(date_range_repl, date_range_repl.replace(date_range_dmy, ' ' + date_range_dmy_str + ' '))

        return input_str.strip()

    def _norm_date_range_type_4(self, input_str: str):
        """
        Normalize dd/mm-dd/mm/yyyy forms

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = (
            r"("
            r"(0?[1-9]|[12]\d|3[01])[\/](0?[1-9]|[1][0-2])"
            r"\s*\-\s*"
            r"(0?[1-9]|[12]\d|3[01])[\/](0?[1-9]|[1][0-2])[\/](\d{4})"
            r")"
        )
        p = re.compile(
            r"(" + self.DATE_PREFIX_REGEX + r")*\s+"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        date_range_dmy_list = []

        while True:
            date_range_dmy = p.search(temp_str)
            if not date_range_dmy:
                break
            # has_prefix = date_range_dmy.group(1) is not None
            has_prefix = not self.__should_add_tu(date_range_dmy.group(1))
            date_range_dmy_list.append((
                date_range_dmy.group(2),
                self.__has_ngay_prefix(date_range_dmy.group(1)),
                has_prefix,
                date_range_dmy.group())
            )
            temp_str = temp_str[date_range_dmy.span()[1]-1:]

        for date_range_dmy, has_ngay, has_prefix, date_range_repl in date_range_dmy_list:
            start_date = date_range_dmy.split('-')[0]
            end_date = date_range_dmy.split('-')[1]
            date_range_dmy_str = date_dm2words(start_date, add_day_prefix=not has_ngay) + ' đến ' + date_dmy2words(end_date)
            if not has_prefix:
                date_range_dmy_str = " từ " + date_range_dmy_str
            input_str = input_str.replace(date_range_repl, date_range_repl.replace(date_range_dmy, ' ' + date_range_dmy_str + ' '))

        return input_str.strip()

    def _norm_date_range_type_5(self, input_str: str):
        """
        Normalize dd/mm-dd/mm forms: 20/1-18/2

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = (
            r"("
            r"(0?[1-9]|[12]\d|3[01])[\/](0?[1-9]|[1][0-2])"
            r"\s*\-\s*"
            r"(0?[1-9]|[12]\d|3[01])[\/](0?[1-9]|[1][0-2])"
            r")"
        )
        p = re.compile(
            r"(" + self.DATE_PREFIX_REGEX + r")\s+"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        date_range_dm_list = []

        while True:
            date_range_dm = p.search(temp_str)
            if not date_range_dm:
                break
            date_range_dm_list.append((
                date_range_dm.group(2), 
                self.__has_ngay_prefix(date_range_dm.group(1)),
                # True,
                not self.__should_add_tu(date_range_dm.group(1)),
                date_range_dm.group()
            ))
            temp_str = temp_str[date_range_dm.span()[1]-1:]

            while True:
                date_range_dm = p_addons.search(temp_str)
                if not date_range_dm:
                    break
                date_range_dm_list.append((
                    date_range_dm.group(2), 
                    False,
                    False,
                    date_range_dm.group()
                ))
                temp_str = temp_str[date_range_dm.span()[1]-1:]

        for date_range_dm, has_ngay, has_prefix, date_range_repl in date_range_dm_list:
            start_date = date_range_dm.split('-')[0]
            end_date = date_range_dm.split('-')[1]
            date_range_dm_str = date_dm2words(start_date, add_day_prefix=not has_ngay) + ' đến ' + date_dm2words(end_date)
            if not has_prefix:
                date_range_dm_str = " từ " + date_range_dm_str
            input_str = input_str.replace(date_range_repl, date_range_repl.replace(date_range_dm, ' ' + date_range_dm_str + ' '))

        return input_str.strip()

    def _norm_date_range_type_6(self, input_str: str):
        """
        Normalize dd-dd/mm forms: 15-18/6, 15 -18/6, 15- 18/6

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = r"((0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2]))"
        p = re.compile(
            r"(" + self.DATE_PREFIX_REGEX + r")\s+"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        date_range_dm_list = []

        while True:
            date_range_dm = p.search(temp_str)
            if not date_range_dm:
                break
            # has_prefix = date_range_dm.group(1) is not None 
            has_prefix = not self.__should_add_tu(date_range_dm.group(1))
            date_range_dm_list.append((
                date_range_dm.group(2), 
                self.__has_ngay_prefix(date_range_dm.group(1)),
                has_prefix,
                date_range_dm.group()
            ))
            temp_str = temp_str[date_range_dm.span()[1]-1:]

            while True:
                date_range_dm = p_addons.search(temp_str)
                if not date_range_dm:
                    break
                date_range_dm_list.append((
                    date_range_dm.group(2), 
                    self.__has_ngay_prefix(date_range_dm.group(1)),
                    False,
                    date_range_dm.group()
                ))
                temp_str = temp_str[date_range_dm.span()[1]-1:]

        for date_range_dm, has_ngay, has_prefix, date_range_repl in date_range_dm_list:
            start_date = date_range_dm.split('-')[0]
            end_date = date_range_dm.split('-')[1]
            date_range_dm_str = num2words_integer(start_date) + ' đến ' + date_dm2words(end_date)
            if int(start_date) < 10:
                date_range_dm_str = ' mùng ' + date_range_dm_str
            if not has_ngay:
                date_range_dm_str = ' ngày ' + date_range_dm_str
            if not has_prefix:
                date_range_dm_str = ' từ ' + date_range_dm_str
            input_str = input_str.replace(date_range_repl, date_range_repl.replace(date_range_dm, ' ' + date_range_dm_str + ' '))

        return input_str.strip()
    
    def _norm_date_range_type_7(self, input_str: str):
        """
        Normalize mm-mm/yyyy forms

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = r"((0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.]([0-9]{4}))"
        p = re.compile(
            r"([Tt]háng|[Qq]úy|[Qq]uý)\s+"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        date_range_my_list = []

        while True:
            date_range_my = p.search(temp_str)
            if not date_range_my:
                break
            # has_prefix = date_range_my.group(1) is not None 
            has_prefix = not self.__should_add_tu(date_range_my.group(1))
            date_range_my_list.append((
                date_range_my.group(2), 
                self.__has_thang_prefix(date_range_my.group(1)),
                has_prefix,
                date_range_my.group()
            ))
            temp_str = temp_str[date_range_my.span()[1]-1:]

            while True:
                date_range_my = p_addons.search(temp_str)
                if not date_range_my:
                    break
                date_range_my_list.append((
                    date_range_my.group(2), 
                    self.__has_thang_prefix(date_range_my.group(1)),
                    False,
                    date_range_my.group()
                ))
                temp_str = temp_str[date_range_my.span()[1]-1:]

        for date_range_my, has_thang, has_prefix, date_range_repl in date_range_my_list:
            start_date = date_range_my.split('-')[0]
            end_date = date_range_my.split('-')[1]
            date_range_dm_str = num2words_integer(start_date) + ' đến ' + date_my2words(end_date, add_month_prefix=not has_thang)
            if not has_prefix:
                date_range_dm_str = ' từ ' + date_range_dm_str
            input_str = input_str.replace(date_range_repl, date_range_repl.replace(date_range_my, ' ' + date_range_dm_str + ' '))

        return input_str.strip()

    def normalize_tag_roman_num(self, input_str: str):
        input_str = self._norm_tag_roman_num_v1(input_str)
        input_str = self._norm_tag_roman_num_v2(input_str)
        return input_str.strip()

    def _norm_tag_roman_num_v1(self, input_str: str):
        input_str = ' ' + input_str + ' '
        roman_numeral_p = re.compile(
            r"("
            r"\s(\(\s*X{0,3})(IX|IV|V?I{0,3})|"
            r"\s(\(\s*x{0,3})(ix|iv|v?i{0,3})"
            r")"
            + self.ENDING_PUNCTUATIONS_REGEX

            , re.IGNORECASE
        )
        temp_str = input_str
        roman_numeral_list = []

        while True:
            roman_numeral = roman_numeral_p.search(temp_str)
            if not roman_numeral:
                break
            
            roman = roman_numeral.group().strip()
            roman = roman.replace(' ', '')
            for character in '. , ( ) /'.split():
                roman = roman.replace(character, '')
            roman = roman.strip()
            if roman != "":
                roman_numeral_list.append(roman)

            temp_str = temp_str[roman_numeral.span()[1]-1:]

        roman_numeral_list = sorted(roman_numeral_list, key=lambda x: (len(x), x), reverse=True)
        
        for roman in roman_numeral_list:
            try:
                roman2int = fromRoman(roman.upper())
            except InvalidRomanNumeralError:
                continue
            roman_numeral_str = num2words_integer(str(roman2int))
            input_str = input_str.replace(roman, roman_numeral_str)

        return input_str.strip()

    def _norm_tag_roman_num_v2(self, input_str: str):
        """
        Normalize roman numerals.
        """
        # @improve: 'Nữ hoàng Anh Elizabeth II thường đi lại trên chiếc xe của Land
        # Rovers và Jaguars' --> II đọc thành là 'đệ nhị'     
        input_str = ' ' + input_str + ' '   
        p_text = (
            r"("
            r"(X{0,3})(IX|IV|VI{0,3}|V|I{1,3})|"
            r"(x{0,3})(ix|iv|vi{0,3}|v|i{1,3})"
            r")"
        )
        p = re.compile(
            r"(thế hệ|"
            r"số|"
            r"đại hội|"
            r"giai đoạn|"
            r"quý|"
            r"cấp|"
            r"quận|"
            r"kỳ|"
            r"kì|"
            r"khóa|"
            r"quy định|"
            r"vành đai|"
            r"vùng|"
            r"thế kỷ|"
            r"khu vực|"
            r"khu|"
            r"đợt|"
            r"hạng|"
            r"báo động|"
            r"tập|"
            r"lần|"
            r"lần thứ|"
            r"trung ương|"
            r"tw|"
            r"chương|"
            r"bước|"
            r"loại|"
            r"mục lục|"
            r"phụ lục|"
            r"đề mục|"
            r"danh mục|"
            r"mục|"
            r"chương|"
            r"quyển|"
            r"nhóm|"
            r"phần)"
            r"\s+"
            
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
            
            , re.IGNORECASE
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )

        temp_str = input_str
        roman_numeral_list = []
        
        while True:
            roman_numeral = p.search(temp_str)
            if not roman_numeral:
                break
            roman_numeral_list.append((roman_numeral.group(2), roman_numeral.group()))
            temp_str = temp_str[roman_numeral.span()[1]-1:]

            while True:
                roman_numeral_addons = p_addons.search(temp_str)
                if not roman_numeral_addons:
                    break
                roman_numeral_list.append((roman_numeral_addons.group(2), roman_numeral_addons.group()))
                temp_str = temp_str[roman_numeral_addons.span()[1]-1:]

        roman_numeral_list = sorted(roman_numeral_list, key=lambda x: (len(x[0]), x[1]), reverse=True)

        for roman_numeral, roman_numeral_repl in roman_numeral_list:
            try:
                roman2int = fromRoman(roman_numeral.upper())
            except InvalidRomanNumeralError:
                continue
            roman_numeral_str = num2words_integer(str(roman2int))
            input_str = input_str.replace(roman_numeral_repl, roman_numeral_repl.replace(roman_numeral, roman_numeral_str))

        return input_str.strip()

    def normalize_time(self, input_str: str):
        """
        Normalize time and time range

        NOTE. LongNH
        """
        input_str = self._norm_time_type_1(input_str)
        input_str = self._norm_time_type_2(input_str)
        return input_str.strip()

    TIME_REGEX = (
        r"(\d+(?:\.\d+)?)(\:|[hg])(\d+(?:\.\d+)?)(\:|[mp])(\d+(?:\.\d+)?)(s?)|"
        r"(\d+(?:\.\d+)?)(\:|[hg])(\d+(?:\.\d+)?)([mps]?)|"
        r"(\d+(?:\.\d+)?)(\:|[mp])(\d+(?:\.\d+)?)([s]?)|"
        r"(\d+(?:\.\d+)?)[hps]"
    )

    def _norm_time_type_1(self, input_str: str):
        """
        Normalize time range

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        time_range_patterns = re.compile(
            r"(" + self.TIME_REGEX + r")"
            r"\s*-\s*"
            r"(" + self.TIME_REGEX + r")"
        )
        temp_str_time = input_str
        duration_times = []

        while True:
            time_range = time_range_patterns.search(temp_str_time)
            if not time_range:
                break
            duration_times.append(time_range.group())
            temp_str_time = temp_str_time[time_range.span()[1]:]

        time_patterns = re.compile(self.TIME_REGEX)
        for duration_time in duration_times:
            temp_time = duration_time
            times = []

            while True:
                time_range = time_patterns.search(temp_time)
                if not time_range:
                    break
                times.append(time_range.group())
                temp_time = temp_time[time_range.span()[1]:]

            times = [time for time in times if not(time.startswith('24') and (time[3:]>'00')) ]
            time_str = time2words(times[0]) + ' đến ' + time2words(times[1])
            input_str = input_str.replace(duration_time, ' ' + time_str + ' ')

        return input_str.strip()

    def _norm_time_type_2(self, input_str: str):
        """
        Normalize time

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        time_patterns = re.compile(self.TIME_REGEX)
        temp_str = input_str
        times = []

        while True:
            time = time_patterns.search(temp_str)
            if not time:
                break
            times.append(time.group())
            temp_str = temp_str[time.span()[1]:]

        times = [time for time in times if not(time.startswith('24') and (time[3:]>'00')) ]

        for time in times:
            time_str = time2words(time)
            input_str = input_str.replace(time, ' ' + time_str + ' ')
        
        return input_str.strip()

    def normalize_multiply_number(self, input_str: str):
        """
        Normalize ???

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(
            r"[0-9,.]*[0-9]+"
            r"\s*x\s*"
            r"[0-9,.]*[0-9]+"
            r"\s*x\s*"
            r"[0-9,.]*[0-9]|"
            r"[0-9,.]*[0-9]+"
            r"\s*x\s*"
            r"[0-9,.]*[0-9]", 
        )
        number_list = []
        start = 0

        while True:
            mul = p.search(input_str, start)
            if not mul:
                break
            number_list.append((mul.group(), *mul.span()))
            start = mul.span()[1]

        for number, s_idx, e_idx in number_list[::-1]:
            number_str = multiply(number)
            input_str = input_str[: s_idx] + number_str + input_str[e_idx :]

        # vi_numbers = re.findall(
        #     r"[0-9,.]*[0-9]+"
        #     r"\s*x\s*"
        #     r"[0-9,.]*[0-9]+"
        #     r"\s*x\s*"
        #     r"[0-9,.]*[0-9]|"
        #     r"[0-9,.]*[0-9]+"
        #     r"\s*x\s*"
        #     r"[0-9,.]*[0-9]", 
            
        #     input_str
        # )

        # for vi_number in vi_numbers:
        #     vi_number_ver = multiply(vi_number)
        #     input_str = input_str.replace(vi_number, ' ' + vi_number_ver + ' ')

        return input_str.strip()

    def normalize_sport_score(self, input_str: str):
        """
        Normalize sport scores
        """
        input_str = ' ' + input_str + ' '
        temp_str = input_str.lower()
        scores = re.findall(r"(\s[0-9]+(\-|\:)[0-9]+)", temp_str)
        scores = [score[0] for score in scores]
        sport_ngrams = ['tỷ số', 'chiến thắng', 'trận đấu', 'tỉ số', 'bàn thắng', 'trên sân', 'đội bóng', 'thi đấu', 'cầu thủ',
                        'vô địch', 'mùa giải', 'đánh bại', 'đối thủ', 'bóng đá', 'gỡ hòa', 'chung kết', 'bán kết', 'ghi bàn',
                        'chủ nhà', 'tiền đạo', 'dứt điểm', 'tiền vệ', 'tiền đạo', 'thua', 'bị dẫn']
        is_sport = 0
        for item in sport_ngrams:
            if (temp_str.find(item)) != -1:
                is_sport = 1
                break
        if is_sport == 1 and len(scores) > 0:
            for score in scores:
                if '-' in score:
                    lscore = num2words_integer(score.split('-')[0])
                    rscore = num2words_integer(score.split('-')[1])
                    score_norm = lscore + ' ' + rscore
                    input_str = input_str.replace(score, ' ' + score_norm + ' ')
                else:
                    lscore = num2words_integer(score.split(':')[0])
                    rscore = num2words_integer(score.split(':')[1])
                    score_norm = lscore + ' ' + rscore
                    input_str = input_str.replace(score, ' ' + score_norm + ' ')

        return input_str.strip()

    def normalize_number_plate(self, input_str: str):
        """
        Normalize number plate

        NOTE. LongNH
        """
        input_str = self._norm_number_plate_type_1(input_str)
        # input_str = self._norm_number_plate_type_2(input_str)
        # input_str = self._norm_number_plate_type_3(input_str)
        # input_str = self._norm_number_plate_type_4(input_str)
        return input_str.strip()

    def _norm_number_plate_type_1(self, input_str: str):
        """
        Normalize plate

        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = (
            r"([0-9]+[a-zA-Z]+[0-9]*)"
            r"(\s{0,1}[\-]{0,1}\s{0,1})"
            r"([0-9]+\.*[0-9]+)"
        )
        p = re.compile(
            r"([B|b]iển [K|k]iểm [S|s]oát|"
            r"[B|b]iển [S|s]ố [Xx]e|"
            r"[Xx]e|"
            r"[Ôô] tô|"
            r"[Ôô]tô|"
            r"[Bb]iển [Ss]ố|"
            r"[Bb]iển)+"
            # r"\s+(.*?)\s*"
            r"(\s|\,|\:)+(()|.*?\s+)"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        number_plate_list = []
        while True:
            number_plate = p.search(temp_str)
            if not number_plate:
                break
            number_plate_list.append((
                number_plate.group(5), 
                number_plate.group(7), 
                number_plate.group()
            ))
            temp_str = temp_str[number_plate.span()[1]-1:]

            while True:
                number_plate = p_addons.search(temp_str)
                if not number_plate:
                    break
                number_plate_list.append((
                    number_plate.group(2), 
                    number_plate.group(4), 
                    number_plate.group()
                ))
                temp_str = temp_str[number_plate.span()[1]-1:]

        # print(number_plate_list)
        for prefix, suffix, plate_repl in number_plate_list:
            prefix_repl = alpha_num2words(prefix)
            suffix_repl = phone2words(suffix)
            plate_repl_new = (
                plate_repl
                    .replace(prefix, " " + prefix_repl + " ")
                    .replace(suffix, " " + suffix_repl + " ")
            )
            input_str = input_str.replace(plate_repl, plate_repl_new)

        return input_str.strip()

    # def _norm_number_plate_type_2(self, input_str: str):
    #     """
    #     Normalize plate

    #     NOTE. LongNH
    #     """
    #     return input_str.strip()
    #     input_str = ' ' + input_str + ' '
    #     p = re.compile(
    #         r"([B|b]iển [K|k]iểm [S|s]oát|"
    #         r"[B|b]iển [S|s]ố xe|"
    #         r"[B|b]iển [S|s]ố)+"
    #         r"\s*(.*)\s+"
    #         r"([0-9]+[a-zA-Z]+[0-9]*)+"
    #         r"\s*(\-|\.|\s)+\s*"
    #         r"([0-9]+)"
    #         + self.ENDING_PUNCTUATIONS_REGEX
    #     )
    #     temp_str = input_str
    #     number_plate_list = []
    #     while p.search(temp_str):
    #         number_plate = p.search(temp_str)
    #         print(number_plate.groups())
    #         x = number_plate.group()
    #         if '-' in x:
    #             number_plate_list.append(x.split("-")[-1])
    #         elif '.' in x:
    #             number_plate_list.append(x.split(".")[-1])
    #         else:
    #             number_plate_list.append(x.split()[-1])
    #         temp_str = temp_str[number_plate.span()[1]-1:]

    #     return self._postprocess_number_plate(input_str, number_plate_list)

    # def _norm_number_plate_type_3(self, input_str: str):
    #     """
    #     Normalize plate

    #     NOTE. LongNH
    #     """
    #     return input_str.strip()
    #     input_str = ' ' + input_str + ' '
    #     p = re.compile(
    #         r"([B|b]iển [K|k]iểm [S|s]oát|"
    #         r"[B|b]iển [S|s]ố xe|"
    #         r"[B|b]iển [S|s]ố)+"
    #         r"\s*.*\s*"
    #         r"([0-9]+[a-zA-Z]+[0-9]*)"
    #         r"(\s|\-)+"
    #         r"([0-9]+\.[0-9]+)"
    #         r"([\s|.|,|)])"
    #     )
    #     temp_str = input_str
    #     number_plate_list = []
    #     while(p.search(temp_str)):
    #         number_plate = p.search(temp_str)
    #         x = number_plate.group()
    #         number_plate_list.append(x.strip().split("-")[-1] if '-' in x else x.split()[-1])
    #         temp_str = temp_str[number_plate.span()[1]:]

    #     return self._postprocess_number_plate(input_str, number_plate_list)

    # def _norm_number_plate_type_4(self, input_str: str):
    #     """
    #     Normalize plate

    #     NOTE. LongNH
    #     """
    #     return input_str.strip()
    #     input_str = ' ' + input_str + ' '
    #     p = re.compile(
    #         r"([B|b]iển [K|k]iểm [S|s]oát|"
    #         r"[B|b]iển [S|s]ố xe|"
    #         r"[B|b]iển [S|s]ố)+"
    #         r"\s*.*\s*"
    #         r"([0-9]+[a-zA-Z]+[0-9]*)+"
    #         r"\s*(\-|\.|\s)\s*"
    #         r"[0-9]+")
    #     temp_str = input_str
    #     number_plate_list = []
    #     while(p.search(temp_str)):
    #         number_plate = p.search(temp_str)
    #         x = number_plate.group()
    #         if '-' in x:
    #             number_plate_list.append(x.split("-")[-1])
    #         elif '.' in x:
    #             number_plate_list.append(x.split(".")[-1])
    #         else:
    #             number_plate_list.append(x.split()[-1])
    #         temp_str = temp_str[number_plate.span()[1]:]

    #     return self._postprocess_number_plate(input_str, number_plate_list)

    def normalize_id_digit(self, input_str: str):
        """
        Normalize CMT, STK

        NOTE. LongNH
        """
        def has_number(text: str):
            for ch in text:
                if ch in "1234567890":
                    return True
            return False
        

        input_str = ' ' + input_str + ' '
        p_text = r"([0-9A-Z\-]+[0-9A-Za-z\-]*)"
        p = re.compile(
            r"([Cc]hứng [Mm]inh [Nn]hân [Dd]ân|"
            r"[Cc][Mm][Nn][Dd]|"
            r"[Cc]hứng [Mm]inh [Tt]hư|"
            r"[Cc][Mm][Tt]|"
            r"[Mm]ã [Tt]hẻ|"
            r"[Ss]ố [Tt]hẻ|"
            r"[Ss]ố [Tt]ài [Kk]hoản|"
            r"[Ss][Tt][Kk]|"
            r"[Tt]hẻ [Tt]ín [Dd]ụng|"
            r"[Tt]ài [Kk]hoản|"
            r"[Cc]ăn [Cc]ước|"
            r"[Cc]ăn [Cc]ước [Cc]ông [Dd]ân|"
            r"[Cc][Cc][Cc][Dd]|"
            r"[Mm]ã [Ss]ố [Tt]huế|"
            r"[Mm][Ss][Tt]|"
            r"[Mm]ã [Ss]ố|"
            r"[Nn]hân [Vv]iên|"
            r"[Mm]ã)+"

            r"(\s|\,|\:)+(()|.*?\s+)"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        digits_list = []
        while True:
            digit = p.search(temp_str)
            if not digit:
                break
            if has_number(digit.group(5)):
                digits_list.append((digit.group(5), digit.group()))
            temp_str = temp_str[digit.span()[1]-1:]

            while True:
                digit = p_addons.search(temp_str)
                if not digit:
                    break
                if has_number(digit.group(2)):
                    digits_list.append((digit.group(2), digit.group()))
                temp_str = temp_str[digit.span()[1]-1:]

        for digit, digit_repl in digits_list:
            digits_str = alpha_num2words(digit)
            # print(digit, "=", digits_str)
            input_str = input_str.replace(digit_repl, digit_repl.replace(digit, ' ' + digits_str + ' '))

        return input_str.strip()

    # def normalize_negative_number(self, input_str: str):
    #     input_str = ' ' + input_str + ' '
    #     p = re.compile(r'\s\-([0-9]*[,.]*[0-9]+)\s')
    #     # p = re.compile(r"(là|kết quả|âm|dưới|lạnh|xuống|nhiệt độ|áp suất)+\s*\:*\-\s*[0-9]*,*[0-9]+")
    #     temp_str = input_str
    #     neg_numbers = []
    #     while True:
    #         numbers = p.search(temp_str)
    #         if not numbers:
    #             break
    #         term = numbers.group()
    #         neg_numbers.append(term.split("-")[-1])
    #         temp_str = temp_str[numbers.span()[1]:]
    #     neg_numbers = [number.replace(',', '.') for number in neg_numbers]
    #     neg_numbers.sort(key=float)
    #     neg_numbers = [number.replace('.', ',') for number in neg_numbers]

    #     if len(neg_numbers) > 0:
    #         for number in neg_numbers:
    #             # if ',' in number:
    #             #     numbers_str = num2words_float(number)
    #             # else:
    #             #     numbers_str = num2words_fixed(number)
    #             # numbers_str = ' âm ' + numbers_str
    #             numbers_str = ' âm ' + number
    #             input_str = input_str.replace("-" + number, ' ' + numbers_str + ' ')

    #     return input_str.strip()

    @classmethod
    def normalize_abbreviation(cls, input_str: str):
        """
        Normalize abbreviations with standard dictionary
        """
        abbre_dict = {
            **ABBRE_DICT, 
            **NAMES_DICT, 
            **{k.lower() : v for k, v in NAMES_DICT.items()},
            **{k.replace(" ", "-") : v for k, v in NAMES_DICT.items()},
            **{k.replace(" ", "-").lower() : v for k, v in NAMES_DICT.items()},
        }
        return cls._norm_abbreviation(input_str, abbre_dict)
    
    @classmethod
    def _norm_abbreviation(cls, input_str: str, abbre_dict: dict):
        temp_str = input_str

        # not remove punctuation
        words_temp = temp_str.split()
        for i in range(len(words_temp)):
            temp_word = words_temp[i]
            if temp_word[-1] in [',', '.', ')', '}', ']', '!', '?', '/', '-', ':', ';', '(', '{', '[', '"', "'"]:
                temp_word = temp_word[:-1]

            if temp_word in abbre_dict.keys():
                input_str =  input_str.replace(temp_word, str(abbre_dict[temp_word]).strip().lower())

        #remove punctuation
        temp_str = input_str
        for character in [',', '.', ')', '}', ']', '!', '?', '/', '-', '(', '{', '[', '"', "'"]:
            temp_str = temp_str.replace(character, ' ')

        words_inp = temp_str.split()
        for i in range(len(words_inp)):
            temp_word = words_inp[i]

            if temp_word in abbre_dict.keys():
                input_str =  input_str.replace(temp_word, str(abbre_dict[temp_word]).strip())

        return input_str.strip()

    def normalize_tag_fraction(self, input_str: str):
        input_str = self._norm_tag_fraction_type_1(input_str)
        input_str = self._norm_tag_fraction_type_2(input_str)
        input_str = self._norm_tag_fraction_type_3(input_str)
        return input_str.strip()

    def _norm_tag_fraction_type_1(self, input_str: str):
        """
        NOTE. LongNH
        """
        input_str = ' ' + input_str + ' '
        p_text = r"(\-?[0-9]+[0-9,.]*\s*[\/:]\s*[0-9]+[0-9,.]*)"
        p = re.compile(
            r"([Tt]hứ|"
            r"[Hh]ơn|"
            r"[Gg]ần|"
            r":|"
            r"[Hh]ạng|"
            r"[Đđ]ược|"
            r"[Tt]ới|"
            r"[Gg}óp|"
            r"[Ll]à|"
            r"[Cc]ó|"
            r"[Cc]òn|"
            r"[Ll]ên|"
            r"[Bb]ằng|"
            r"[Cc]hiếm|"
            r"[Gg]iảm|"
            r"[Tt]ỷ [Ll]ệ|"
            r"[Tt]ỉ [Ll]ệ|"
            r"[Mm]ật [Đđ]ộ|"
            r"[Nn]ồng [Đđ]ộ|"
            r"[Ll]iều [Ll]ượng|"
            r"[Kk]hoảng|"
            r"[Oo]nline)+\s+"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        ratio_list = []
        while True:
            ratio = p.search(temp_str)
            if not ratio:
                break
            ratio_list.append((ratio.group(2).rstrip(",."), ratio.group()))
            temp_str = temp_str[ratio.span()[1]-1:]

            while True:
                ratio = p_addons.search(temp_str)
                if not ratio:
                    break
                ratio_list.append((ratio.group(2).rstrip(",."), ratio.group()))
                temp_str = temp_str[ratio.span()[1]-1:]

        # for character in '. , ( ) ;'.split():
        #     ratio_list = [item.replace(character, '') for item in ratio_list]
        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        for ratio, ratio_repl in ratio_list:
            first_num  = ratio.replace(":", "/").split('/')[0]
            second_num = ratio.replace(":", "/").split('/')[1]
            ratio_str = num2words_float(first_num) + ' phần ' + num2words_float(second_num)
            input_str = input_str.replace(ratio_repl, ratio_repl.replace(ratio, ' ' + ratio_str + ' '))

        return input_str.strip()

    def _norm_tag_fraction_type_2(self, input_str: str):
        """
        Normalize case 1/3 muỗng

        NOTE: [LongNH]
        """
        input_str = ' ' + input_str + ' '
        p_text = r"([0-9,.]+\s*[\/:]\s*[0-9,.]+)"
        p = re.compile(
            r"\s+"
            + p_text
            + r"\s*([Mm]uỗng|[Tt]hìa|[Ll]y|[Cc]ốc|[Cc]hén|[Cc]hai|[Ll]ọ)"
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"
            
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        ratio_list = []
        while True:
            ratio = p.search(temp_str)
            if not ratio:
                break
            ratio_list.append((ratio.group(1).rstrip(",."), ratio.group()))
            temp_str = temp_str[ratio.span()[1]-1:]

            while True:
                ratio = p_addons.search(temp_str)
                if not ratio:
                    break
                ratio_list.append((ratio.group(2).rstrip(",."), ratio.group()))
                temp_str = temp_str[ratio.span()[1]-1:]

        # for character in '. , ( ) ;'.split():
            # ratio_list = [item.replace(character, '') for item in ratio_list]
        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]

        for ratio, ratio_repl in ratio_list:
            first_num  = ratio.replace(":", "/").split('/')[0].replace(",", "")
            second_num = ratio.replace(":", "/").split('/')[1].replace(",", "")
            ratio_str = num2words_float(first_num) + ' phần ' + num2words_float(second_num)
            input_str = input_str.replace(ratio_repl, ratio_repl.replace(ratio, ' ' + ratio_str + ' '))

        return input_str.strip()

    def _norm_tag_fraction_type_3(self, input_str: str):
        """
        Normalize case trường hợp/100.000 dân

        NOTE: Keep original, not touch
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"\s[AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴA-Zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z]+\/")
        temp_str = input_str
        ratio_list = []
        while True:
            ratio = p.search(temp_str)
            if not ratio:
                break
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            ratio_list.append(x.split()[-1])
            temp_str = temp_str[ratio.span()[1]:]

        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        if len(ratio_list) > 0:
            for ratio in ratio_list:
                ratio_str = ratio.replace('/', ' trên ')
                input_str = input_str.replace(ratio, ' ' + ratio_str + ' ')
        return input_str.strip()
    
    def normalize_legal_id(self, input_str: str):
        """
        Normalize case Điều 48 Nghị định 110/2013/NĐ-CP

        NOTE: Keep original, not touch
        """
        input_str = ' ' + input_str + ' '
        p_text = r"([0-9]+\s*\/\s*[0-9]+)(\/)*[0-9]+(\/)([A-z\-Đ]+)"
        p = re.compile(
            r"([Đđ]iều [Kk]hoản|"
            r"[Đđ]iều [Ll]ệ|"
            r"[Đđ]iều [Ll]uật|"
            r"[Dd]ự [Tt]hảo|"
            r"[Dd]ự [Ll]uật|"
            r"[N|n]ghị [Đ|đ]ịnh|"
            r"[N|n]ghị [Q|q]uyết|"
            r"[T|t]hông [T|t]ư|"
            r"[T|t]hông [T|t]ư [Ll]iên [Tt]ịch)+"
            r"(\s+số)?"
            r"\s+"
            + p_text
            # + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"
            
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )

        temp_str = input_str
        legal_id_list = []
        while True:
            legal_id = p.search(temp_str)
            if not legal_id:
                break
            x = legal_id.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            legal_id_list.append(x.split()[-1])
            temp_str = temp_str[legal_id.span()[1]:]

            while True:
                legal_id = p_addons.search(temp_str)
                if not legal_id:
                    break
                x = legal_id.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
                legal_id_list.append(x.split()[-1])
                temp_str = temp_str[legal_id.span()[1]-1:]

        for character in '. , ( ) ;'.split():
            legal_id_list = [item.replace(character, '') for item in legal_id_list]
        legal_id_list = sorted(legal_id_list, key=len, reverse=True)

        for legal_id in legal_id_list:
            parts = legal_id.split("/")
            if len(parts) <= 2:
                legal_id_str = legal_id.replace('/', ', năm ')
            else:
                legal_id_str = " trên ".join([alpha_num2words(p).strip() for p in parts[:-2]]).strip()
                legal_id_str += " năm " + num2words_integer(parts[-2]).strip() + " ,"
                for p in parts[-1]:
                    if p.strip() == "":
                        continue
                    if p in NUMBER_DICT:
                        legal_id_str += " " + NUMBER_DICT[p]
                    elif p in ALPHABET_DICT:
                        legal_id_str += " " + ALPHABET_DICT[p]
                    else:
                        legal_id_str += " ,"
            legal_id_str = legal_id_str.strip()
            input_str = input_str.replace(legal_id, ' ' + legal_id_str.strip() + ' ')

        return input_str.strip()

    def normalize_address(self, input_str: str):
        """
        Normalize case ngõ 12A/124 
        """
        input_str = ' ' + input_str + ' '
        p_text = (
            r"[0-9A-zĐ]+"
            r"\s*\/\s*"
            r"[0-9A-zĐ]+"
            r"(\s*\/\s*)*"
            r"[0-9A-zĐ]*"
        )
        p = re.compile(
            r"([N|n]gõ|[N|n]gách|[H|h]ẻm|[Đđ]ường|[Tt]ổ)"
            r"\s+"
            + p_text

            # r"[\s|.|,|)|;]"
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"
            
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        
        temp_str = input_str
        address_list = []
        while True:
            address = p.search(temp_str)
            if not address:
                break
            x = address.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            address_list.append(x.split()[-1])
            temp_str = temp_str[address.span()[1] - 1:]
            
            while True:
                address = p_addons.search(temp_str)
                if not address:
                    break
                x = address.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
                address_list.append(x.split()[-1])
                temp_str = temp_str[address.span()[1] - 1:]

        for character in '. , ( ) ;'.split():
            address_list = [item.replace(character, '') for item in address_list]
        address_list = sorted(address_list, key=len, reverse=True)
        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        for address in address_list:
            ratio_str = " trên ".join([alpha_num2words(a, num_mode="full") for a in address.split("/")])
            input_str = input_str.replace(address, ratio_str)
        return input_str.strip()
    
    def normalize_money_number(self, input_str: str):
        """
        NOTE: Keep original, not touch
        """
        # pattern = r"([\d]+đ[\/)\]}]?)\b"
        p = re.compile(
            r"("
            # r"[\d]+"
            r"(-?[\d]+"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*)"
            
            r"([Đđ]|\s*[Vv][Nn][DdĐđ])"
            r"[\/)\]}]*"
            r")\b"
        )
        money_list = []
        start = 0
        
        while True:
            number = p.search(input_str, start)
            if not number:
                break
            # print(number.groups())
            x = number.group(2)
            prefix = ""
            if x[0] == "-":
                x = x[1:]
                prefix = " âm "
            if x[-1] in ",.":
                x = x[:-1]
            unit = number.group(3).strip()
            money_list.append((prefix, x, unit, number.group(), *number.span()))
            start = number.span()[1]
            
        for prefix, number, unit, number_repl, s_idx, e_idx in money_list[::-1]:
            unit_str = "đồng" if unit.lower() == "đ" else "việt nam đồng"
            number_repl = number_repl.replace(unit, " " + unit_str + " ")

            number_str = prefix + money2words(number)
            number_repl = number_repl.replace(number, " " + number_str + " ")

            tmp = number_repl.rstrip("/)]}")
            suffix = " trên " if tmp != number_repl and "/" in number_repl[-len(number_repl) + len(number_repl):] else ""

            number_repl = tmp + suffix

            input_str = input_str[: s_idx] + number_repl + input_str[e_idx :]

        return input_str.strip()

    def normalize_number(self, input_str: str):
        input_str = self._norm_number_type_1(input_str)
        input_str = self._norm_number_type_2(input_str)

        return input_str.strip()

    def _norm_number_type_1(self, input_str: str):
        """
        Normalize number
        """
        # Normalize vi-style numbers: '2.300 Euro', '25.320 vé', etc
        input_str = ' ' + input_str + ' '
        p = re.compile(
            r"[\s|(]"
            r"("
            r"-?[\d]+[\.|\,][\d]+"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*"
            r")"
        )
        number_list = []
        start = 0

        while True:
            number = p.search(input_str, start)
            if not number:
                break
            x = number.group(1)
            prefix = ""
            if x[0] == "-":
                x = x[1:]
                prefix = " âm "
            if x[-1] in ",.":
                x = x[:-1]
            number_list.append((prefix, x, number.group(), *number.span()))
            start = number.span()[1]

        for prefix, number, number_repl, s_idx, e_idx in number_list[::-1]:
            number_str = prefix + num2words_float(number)
            number_repl = number_repl.replace(number, " " + number_str + " ")
            input_str = input_str[: s_idx] + number_repl + input_str[e_idx :]

        return input_str.strip()

    def _norm_number_type_2(self, input_str: str):
        input_str = ' ' + input_str + ' '
        p = re.compile(r"(-?[\d]+)")
        number_list = []
        start = 0
        
        while True:
            number = p.search(input_str, start)
            if not number:
                break
            x = number.group(1)
            prefix = ""
            if x[0] == "-":
                x = x[1:]
                prefix = " âm "
            number_list.append((prefix, x, number.group(), *number.span()))
            start = number.span()[1]

        for prefix, number, number_repl, s_idx, e_idx in number_list[::-1]:
            number_str = prefix + num2words_integer(str(int(number)))
            number_repl = number_repl.replace(number, " " + number_str + " ")
            input_str = input_str[: s_idx] + number_repl + input_str[e_idx :]

        return input_str.strip()

    def normalize_tag_measure(self, input_str: str):
        """
        Normalize unit names and number + unit names. i.e.
        'kg' --> 'ki lô gam', '1000mAh' --> 'một nghìn mi li am pe'
        """
        # Normalize unit names (length, area, volume, information, speed, etc)
        input_str = ' ' + input_str + ' '

        def is_number_str(word: str):
            # has_num, has_alpha = False, False
            # for ch in text:
            #     if ch in "1234567890":
            #         has_num = True
            #     if ch.isalpha():
            #         has_alpha = True
            # if has_num and has_alpha:
            #     return False
            # return True
            nums = "0123456789"
            signs = "+-"
            points = ",."
            has_point = False
            for i, ch in enumerate(word):
                if ch not in nums:
                    if i == 0 and ch in signs:
                        pass
                    elif ch in points:
                        has_point = True
                    else:
                        return False
                    
            if has_point:
                pos_point = [i for i, ch in enumerate(word) if ch == ","]
                pos_dot = [i for i, ch in enumerate(word) if ch == "."]

                if len(pos_dot) > 0 and len(pos_point) > 0:
                    if pos_dot[-1] + 1 < pos_point[0] or pos_point[-1] + 1 < pos_dot[0]:
                        pass
                    else:
                        return False
            
            return True

        units_regex = (
            "|".join(["\\" + k if k.startswith("$") else k for k in UNITS_DICT.keys()])
            # + "|".join(["\\" + k.strip() if k.startswith("$") else k.strip() for k in UNITS_DICT.keys()])
        )

        for _regex, _dict in [(units_regex, UNITS_DICT), (self.RAW_MEASUREMENTS_REGEX, MEASURES_DICT)]:
            p = re.compile(
                r"(\b|[0-9])"
                + r"(" + _regex + r")\b"
            )

            start = 0
            unit_list = []
            while True:
                number = p.search(input_str, start)
                if not number:
                    break
                last_word = input_str[:number.span()[0] + len(number.group(1))]
                last_word = last_word.split()[-1].split("-")[-1] if last_word != "" else ""
                # print(last_word, number.groups())
                if is_number_str(last_word):
                    unit_list.append((number.group(2), number.group(), *number.span()))
                start = number.span()[1]

            for unit, unit_repl, s_idx, e_idx in unit_list[::-1]:
                unit = unit.strip()
                if unit in _dict:
                    unit_str = _dict[unit]
                else:
                    unit_str = _dict[unit + " "]
                unit_repl = unit_repl.replace(unit, " " + unit_str + " ")
                input_str = input_str[: s_idx] + unit_repl + input_str[e_idx :]

        return input_str.strip()

    def _norm_tag_measure_generic(self, input_str: str, pattern: str, term: str, norm_term: str):
        matches = re.findall(pattern, input_str)
        # print(pattern)
        # print(input_str)
        if len(matches) > 0:
            for item in matches:
                item_norm_out = item.replace(term, norm_term)
                input_str = input_str.replace(' ' + item.strip() + ' ', ' ' + item_norm_out.strip() + ' ')
        return input_str.strip()

    def normalize_number_range(self, input_str: str):
        """
        Must run before unit and measurement and number
        """
        input_str = self._norm_number_range_type_1(input_str)
        input_str = self._norm_number_range_type_2(input_str)
        # input_str = self._norm_number_range_type_3(input_str)
        return input_str.strip()

    def _norm_number_range_type_1(self, input_str: str):
        """
        Normalize number ranges

        NOTE. LongNH

        Has not consider version range
        """
        input_str = ' ' + input_str + ' '
        p_number = (
            r"(\-?[0-9]+[0-9,.^\/]*)(\s*)"
            r"(" + self.RAW_UNITS_REGEX + r")*"

            r"(\s*\-\s*)"
            
            r"(\-?[0-9]+[0-9,.^\/]*)(\s*)"
            r"(" + self.RAW_UNITS_REGEX + r")*"
        )
        p = re.compile(
            r"([Hh]ơn|"
            r"[Kk]ém|"
            r"[Gg]ấp|"
            r"[Tt]ăng|"
            r"[Tt]ầm|"
            r"[Gg]iảm|"
            r"[Ll]iệu [Tt]rình|"
            r"[Nn]hất|"
            r"[Tt]ới|"
            r"[Cc]ó|"
            r"[Ss]au|"
            r"[Mm]ức|"
            r"[Tt]uổi|"
            r"[Tt]ừ|"
            r"[Tt]ăng [Tt]ốc|"
            r"[Đđ]ược|"
            r"[Kk]hoảng|"
            r"[Tt]rong|"
            r"[Vv]òng|"
            r"[Dd]ao [Đđ]ộng|"
            r"[Gg]iao [Đđ]ộng|"
            r"[Pp]hiên [Bb]bản"
            r")"
            r"\s+(.*?)\s*"

            # r"[0-9]*(,|\.)*[0-9]+\s*\-\s*[0-9]*(,|\.)*[0-9]+"
            + p_number
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"

            + p_number
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        number_range_list = []
        while True:
            number_range = p.search(temp_str)
            if not number_range:
                break
            first = number_range.group(1).lower().strip()
            second = number_range.group(2).lower().strip()
            is_version = ("phiên bản" in first or "phiên bản" in second)
            number_range_list.append((
                is_version,
                number_range.group(3), # number
                number_range.group(4),
                number_range.group(5), # unit
                number_range.group(6), # đến
                number_range.group(7), # number
                number_range.group(8),
                number_range.group(9), # unit
                number_range.group()   # [ALL]
            ))
            temp_str = temp_str[number_range.span()[1]-1:]

            while True:
                number_range = p_addons.search(temp_str)
                if not number_range:
                    break
                number_range_list.append((
                    is_version,
                    number_range.group(2), # number
                    number_range.group(3),
                    number_range.group(4), # unit
                    number_range.group(5), # đến
                    number_range.group(6), # number
                    number_range.group(7),
                    number_range.group(8), # unit
                    number_range.group()   # [ALL]
                ))
                temp_str = temp_str[number_range.span()[1]-1:]

        for parts in number_range_list:
            (is_version, start_num, _, start_unit, _,
             end_num, _, end_unit, num_range_repl) = parts
            origin = "".join([p for p in parts[1:-1] if p is not None])

            if is_version is False:
                start_num = num2words_mixed(start_num)
                end_num = num2words_mixed(end_num)
            else:
                start_num = version2words(start_num)
                end_num = version2words(end_num)
            start_unit = UNITS_DICT[start_unit] if start_unit is not None else ""
            end_unit = UNITS_DICT[end_unit] if end_unit is not None else ""

            norm = " ".join([start_num, start_unit, "đến", end_num, end_unit])
            input_str = input_str.replace(num_range_repl, num_range_repl.replace(origin, " " + norm + " "))

        return input_str.strip()

    def _norm_number_range_type_2(self, input_str: str):
        """
        NOTE. LongNH
        """
        input_str = " " + input_str + " "
        p = re.compile(
            r"(\-?[0-9]+[0-9,.^\/]*)(\s*)"
            r"(" + self.RAW_UNITS_REGEX + r")*"

            r"(\s*\-\s*)"
            
            r"(\-?[0-9]+[0-9,.^\/]*)(\s*)"
            r"(" + self.RAW_UNITS_REGEX + r"|" 
            + self.RAW_UNITS_VALUE_REGEX 
            + r"|cái|lần|túi|khách|con|chiếc|bộ|đống|nhóm|"
            r"hôm|ngày|tháng|năm|tiếng|giờ|phút|giây|tuổi|"
            r"muỗng|thìa|gói|ca|thùng|"
            r"độ|lít|tấn|tạ|yến|mét|m|cân|"
            r"triệu|tỷ|trăm|nghìn|vạn|hàng)"
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        temp_str = input_str
        number_range_list = []
        tmp_unit_dict = {"m": "mét", **UNITS_DICT}
        while True:
            number_range = p.search(temp_str)
            if not number_range:
                break
            number_range_list.append((
                number_range.group(1), # number
                number_range.group(2),
                number_range.group(3), # unit
                number_range.group(4), # đến
                number_range.group(5), # number
                number_range.group(6),
                number_range.group(7), # unit
                number_range.group()   # [ALL]
            ))
            temp_str = temp_str[number_range.span()[1]-1:]

        for parts in number_range_list:
            (start_num, _, start_unit, _,
             end_num, _, end_unit, num_range_repl) = parts
            origin = "".join([p for p in parts[:-1] if p is not None])

            start_num = num2words_mixed(start_num)
            end_num = num2words_mixed(end_num)

            if start_unit is None:
                start_unit = ""
            elif start_unit in tmp_unit_dict:
                start_unit = tmp_unit_dict[start_unit]

            if end_unit is None:
                end_unit = ""
            elif end_unit in tmp_unit_dict:
                end_unit = tmp_unit_dict[end_unit]

            norm = " ".join([start_num, start_unit, "đến", end_num, end_unit])
            input_str = input_str.replace(num_range_repl, num_range_repl.replace(origin, " " + norm + " "))

        return input_str.strip()

    # def _norm_number_range_type_3(self, input_str: str):
    #     """
    #     Normalize number ranges
    #     """
    #     return input_str.strip()
    #     input_str = ' ' + input_str + ' '
    #     p = re.compile(r"\s+[0-9]*,*[0-9]+\s*\-\s*[0-9]*,*[0-9]+\s*(lần|cái|khách|túi|ki lô gam|kg|hôm|ngày|muỗng|thìa|phút|gói|cái|tháng|năm|tiếng|mét|ca|tuổi|phần trăm|ki lô gam|giờ|giây|xen ti mét|mi li mét|độ|lít|tấn|thùng|cái|con|triệu|gam|hàng)")
    #     temp_str = input_str
    #     number_range_list = []
    #     while(p.search(temp_str)):
    #         number_range = p.search(temp_str)
    #         term = number_range.group()
    #         x = re.search(r"[0-9]*,*[0-9]+\s*\-\s*[0-9]*,*[0-9]+[\s|.|,|)]", term)
    #         number_range_list.append(x.group())
    #         temp_str = temp_str[number_range.span()[1]:]
        
    #     if len(number_range_list) > 0:
    #         for number_range in number_range_list:
    #             start_num = number_range.split('-')[0]
    #             end_num = number_range.split('-')[1]
    #             if ',' in start_num:
    #                 start_num = num2words_float(start_num)
    #             else:
    #                 start_num = num2words_integer(start_num)

    #             if ',' in end_num:
    #                 end_num = num2words_float(end_num)
    #             else:
    #                 end_num = num2words_integer(end_num)

    #             number_range_str = start_num + ' đến ' + end_num
    #             input_str = input_str.replace(number_range, ' ' + number_range_str + ' ')

    #     return input_str.strip()

    def normalize_rate(self, input_str: str):
        """
        Normalize number ranges
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"(đánh giá|rate)\s+[0-9]+[*|⭐|★].")
        temp_str = input_str
        rate_list = []
        while True:
            rate = p.search(temp_str)
            if not rate:
                break
            term = rate.group()
            x = re.search(r"[*|⭐|★]", term)
            if not x:
                break
            rate_list.append(x.group())
            temp_str = temp_str[rate.span()[1]:]
        
        for rate in rate_list:
            input_str = input_str.replace(rate, ' ' + 'sao' + ' ')

        return input_str.strip()

    # def normalize_math_characters(self, input_str: str):
    #     """
    #     Normalize math characters
    #     """
    #     input_str = ' ' + input_str + ' '
    #     input_str = replace_math_characters(input_str)

    #     return input_str.strip()
    
    def normalize_full_upper_word(self, input_str: str):
        EXCLUDES = ["II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XX", "XIX", "XXI"]
        INCLUDES = ["ABC", "ABCD", "XYZ"]
        p = re.compile(r"\b[A-Z]{2,}\b")
        start = 0
        upper_words = []
        while True:
            word = p.search(input_str, start)
            if not word:
                break
            w = word.group()
            if (w.lower() not in LOWER_VOCAB_SET and w not in EXCLUDES) or w in INCLUDES:
                upper_words.append((word.group(), *word.span()))
            start = word.span()[1]

        for word, i_start, i_end in upper_words[::-1]:
            new_word = " " + " ".join([ALPHABET_DICT[ch] for ch in word]) + " "
            input_str = input_str[:i_start] + new_word + input_str[i_end:]
        
        return input_str
    
    def normalize(self, text: str, norm_puncs=False, spacing_puncs=True):
        # Define the normalization pipeline in order
        normalization_pipeline = [
            self.normalize_endline,
            self.remove_emoji,  
            self.remove_special_characters,
            self.normalize_money_number,
            self.normalize_legal_id,
            self.normalize_number_plate,
            self.normalize_abbreviation,
            self.normalize_date_range,
            self.normalize_date,
            self.normalize_number_range,
            self.normalize_tag_measure,
            self.normalize_rate,
            self.normalize_tag_fraction,
            self.normalize_address,
            self.normalize_phone_number,
            self.normalize_multiply_number,
            self.normalize_sport_score,
            self.normalize_time,
            self.normalize_id_digit,
            self.normalize_tag_roman_num,
            self.normalize_AZ09,
            self.normalize_tag_verbatim,
            self.normalize_number,
            self.normalize_tag_measure,  # Second call
            self.normalize_connector,
            self.normalize_number,  # Second call
        ]
        
        # Add conditional normalizations
        if spacing_puncs:
            normalization_pipeline.append(self.normalize_spacing_puncs)
            
        normalization_pipeline.extend([
            self.normalize_tag_measure,  # Third call
            self.normalize_single_stand_letter,
            self.normalize_full_upper_word,
        ])
        
        if norm_puncs:
            normalization_pipeline.append(self.normalize_punctuation)
            
        normalization_pipeline.extend([
            self.normalize_duplicate_word,
            self.remove_multi_space,
        ])
        
        def apply_pipeline(text: str, funcs: list):
            failed_funcs = []
            for func in funcs:
                try:
                    text = func(text)
                except:
                    failed_funcs.append(func)
                    msg = f"{self.__class__.__name__}.normalize(): Failed to normalize with '{func.__name__}'"
                    logger.exception(msg)
                    if self.verbose:
                        print("EXCEPTION:", msg)
            return text, failed_funcs
        
        text, failed_funcs = apply_pipeline(text, normalization_pipeline)
        if len(failed_funcs) > 0:
            text, failed_funcs = apply_pipeline(text, failed_funcs)

            if len(failed_funcs) > 0:
                msg = f"{self.__class__.__name__}.normalize(): Cannot normalize text='{text}' | failed_funcs={failed_funcs}"
                logger.error(msg)
                if self.verbose:
                    print("ERROR:", msg)

        return text.strip()
        
