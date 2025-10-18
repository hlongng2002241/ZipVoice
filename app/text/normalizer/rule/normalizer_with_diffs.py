import re
import string
from dataclasses import dataclass, asdict
from roman import fromRoman, InvalidRomanNumeralError
from nltk import word_tokenize

from .cores import *
from .utils.characters import ALPHABET_DICT, NUMBER_DICT
from .utils.units import UNITS_DICT
from .utils.measure import MEASURES_DICT
from .utils.verbatim import VERBATIM
from .utils.names import NAMES_DICT
from .utils.abbre import ABBRE_DICT


def _load_vocab():
    words = []
    for path in ["data/lexicon_en.tsv", "data/lexicon_vi.tsv"]:
        with open(path) as f:
            for line in f.readlines():
                word, _ = line.strip().split("\t")
                words.append(word.lower())
    return words


LOWER_VOCAB_SET = set(_load_vocab())


@dataclass
class Difference:
    start_index: int
    end_index: int
    original: str
    modified: str
    tag: str = None
    
    root_start_index: int = None
    root_end_index: int = None
    root_original: str = None
    
    def __init__(
        self, 
        start_index: int, 
        end_index: int, 
        original: str, 
        modified: str, 
        tag: str = None, 
        root_start_index: int = None,
        root_end_index: int = None,
        root_original: str = None
    ):
        self.start_index = start_index
        self.end_index = end_index
        self.original = original
        self.modified = modified
        self.tag = tag
        self.root_start_index = start_index if root_start_index is None else root_start_index
        self.root_end_index = end_index if root_end_index is None else root_end_index
        self.root_original = root_original
    
    def range(self):
        return (self.start_index, self.end_index)
    
    def original_range(self):
        return (self.root_start_index, self.root_end_index)
    
    def shift(self, value: int):
        self.start_index += value
        self.end_index += value
        
    def shift_root(self, value: int):
        self.root_start_index += value
        self.root_end_index += value
    
    def __repr__(self):
        return (
             f"{self.__class__.__name__}"
            + "("
            + ", ".join([f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in self.to_dict().items()])
            + ")"
        )

    def to_dict(self):
        return asdict(self)


def shift_starts(starts: list[int], shift: int):
    return [s + shift for s in starts]


def replace_with_starts(text: str, old: str, new: str, start: int=0):
    repl = text[: start] + text[start :].replace(old, new)

    starts = []
    while True:
        start = text.find(old, start)
        if start == -1:
            break
        starts.append(start)
        start += len(old)
    
    for s in starts[::-1]:
        e = s + len(old)
        text = text[: s] + new + text[e :]

    assert text == repl, f"{text} != {repl}"

    return text, starts


def assert_match_with_starts(starts: list[int], text: str, mat: str):
    for s in starts:
        e = s + len(mat)
        assert text[s:e] == mat, f"{text[s:e]} != {mat}"


def assert_differences(text: str, differences: list[Difference]):
    for d in differences:
        assert text[d.start_index : d.end_index] == d.original, (
            f"{text[d.start_index : d.end_index]} != {d.original}"
        )


def shift_differences(differences: list[Difference], shift: int):
    for d in differences:
        d.shift(shift)
        d.shift_root(shift)


def map_position_to_root(pos: int, previous_differences: list[Difference]) -> int:
    """
    Map a position in current text back to position in root text.
    
    Args:
        pos: Position in current text
        previous_differences: Differences that transformed root text to current text
        
    Returns:
        Corresponding position in root text
    """
    # Sort differences by their root position in reverse order
    # Process them from right to left to avoid position shifting issues
    sorted_diffs = sorted(previous_differences, key=lambda d: d.root_start_index, reverse=True)
    
    # Start with the position in current text
    root_pos = pos
    
    # Process each difference from right to left
    # This way, earlier differences don't affect the positions of later ones
    for diff in sorted_diffs:
        # Calculate length change from this difference
        len_change = len(diff.modified) - len(diff.original)
        
        # Calculate where this difference appears in current text
        # Start with the root position
        current_start = diff.root_start_index
        
        # Adjust for all transformations that happened BEFORE this one (to the left)
        for other_diff in previous_differences:
            if other_diff.root_start_index < diff.root_start_index:
                other_len_change = len(other_diff.modified) - len(other_diff.original)
                current_start += other_len_change
        
        current_end = current_start + len(diff.modified)
        
        # Check if our position is affected by this difference
        if pos >= current_start:
            if pos < current_end:
                # Position is within this difference - map to corresponding position in original
                offset = pos - current_start
                if offset >= len(diff.original):
                    # Position is beyond the original text length, map to end
                    root_pos = diff.root_end_index
                else:
                    # Map to corresponding position within original
                    root_pos = diff.root_start_index + offset
                break
            else:
                # Position is after this difference - adjust for the length change
                root_pos -= len_change
    
    return root_pos


def merge_differences(
    previous_differences: list[Difference], 
    current_differences: list[Difference]
) -> list[Difference]:
    """
    Merge differences from sequential normalization steps.
    
    Args:
        previous_differences: Differences from previous normalization step(s)
        current_differences: Differences from current step (need root positions updated)
        
    Returns:
        Combined differences list with correct root positions
    """
    # For current differences, we need to find their true root positions
    # The challenge is that diff.root_start_index and diff.root_end_index are relative to the
    # current function's input text, not the true root text
    
    updated_current_diffs = []
    for diff in current_differences:
        # FIXED APPROACH: Use a simpler position mapping
        # The key insight is that diff.root_start_index and diff.root_end_index 
        # are actually positions in the current function's input text
        
        # Find the true root position by accounting for all previous expansions/contractions
        total_offset = 0
        for prev_diff in previous_differences:
            if prev_diff.root_start_index < diff.root_start_index:
                # Previous difference is to the left, so it shifts our position
                len_change = len(prev_diff.modified) - len(prev_diff.original)
                total_offset += len_change
        
        # The true root positions account for shifts from previous transformations
        true_root_start = diff.root_start_index - total_offset
        true_root_end = diff.root_end_index - total_offset
        
        
        # Create new difference with corrected root positions
        updated_diff = Difference(
            start_index=diff.start_index,
            end_index=diff.end_index,
            original=diff.original,
            modified=diff.modified,
            tag=diff.tag,
            root_start_index=true_root_start,
            root_end_index=true_root_end,
            root_original=diff.root_original
        )
        updated_current_diffs.append(updated_diff)
    
    # Combine all differences and maintain chronological order (by root position)
    all_differences = previous_differences + updated_current_diffs
    all_differences.sort(key=lambda d: d.root_start_index)
    
    return all_differences


class RuleBasedTextNormalizerWithDiffs():
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

    # def remove_special_characters(self, input_str: str):
    #     input_str = ' ' + input_str + ' '
    #     input_str = input_str.replace('–', '-')
    #     # punct = '! " “ \' ( ) ; [ ] * _ ` { | } ~ … 》 ≧ ≦ ‘ ’ · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
    #     # punct = '! " “ \' ( ) ; [ ] * _ ` { ~ } … 》 ≧ ≦ ‘ ’ ” · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
    #     punct = '“ … 》 ≧ ≦ ‘ ’ ” · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
    #     for e in punct:
    #         e = e.strip()
    #         input_str = input_str.replace(e, ' ')
    #     return input_str.strip()
    
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
        return input_str.strip()
        input_str = self.remove_redundant_words(input_str, 'ngày', 'ngày')
        input_str = self.remove_redundant_words(input_str, 'mùng', 'ngày')
        input_str = self.remove_redundant_words(input_str, 'tháng', 'tháng')
        return input_str.strip()
    
    # def remove_redundant_words(self, text: str, word1: str, word2: str):
    #     # print('text: ' ,text)
    #     text = text.strip()
    #     new_text = []
    #     tokens = [i.strip() for i in text.split()]
    #     # print('tokens: ', tokens)
    #     if len(tokens) != 0:
    #         for i in range(len(tokens)-1):
    #             if tokens[i] == word1 and tokens[i+1] == word2:
    #                 continue
    #             else:
    #                 new_text.append(tokens[i+1])
    #     # print('tokens after: ', tokens)
    #     if tokens:
    #         new_text.insert(0, tokens[0])
    #     else:
    #         pass
        
    #     return ' '.join(new_text)

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
        ori_input_str = input_str
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

            # r"\s+(.*?)\s*"
            r"(\s|\,|\:)+(()|.*?\s+)(\+?)"

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
        phone_number_list = []
        start = 0
        while True:
            phone_number = p.search(input_str, start)
            if not phone_number:
                break
            x = phone_number.group(5) + phone_number.group(6)
            phone_number_list.append((x, phone_number.group(), phone_number.span()[0]))
            start = phone_number.span()[1] - 1

            while True:
                phone_number = p_addons.search(input_str, start)
                if not phone_number:
                    break
                x = phone_number.group(2) + phone_number.group(3)
                phone_number_list.append((x, phone_number.group(), phone_number.span()[0]))
                start = phone_number.span()[1] - 1

        differences: list[Difference] = []
        for phone_number, phone_number_repl, start_pos in phone_number_list[::-1]:
            # TODO. Small replacement: replace spoken text within matched text chunk
            # convert to spoken text
            phone_number_str = phone2words(phone_number).strip()
            # replace the text to the spoken text in the bigger replacement phone_number_repl
            phone_number_repl_normed, diff_starts = replace_with_starts(phone_number_repl, phone_number, phone_number_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # TODO. Big replacement: replace entired matched chunk in full string
            # replace
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                phone_number_repl, 
                phone_number_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, phone_number)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(phone_number), 
                original=phone_number, modified=phone_number_str,
                tag="PHONE"
            ) for s in diff_starts]

        # shift -1 because padding is added at first: " " + input_str + " "
        shift_differences(differences, -1)
        # assert differences with the original input str
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]
    
    def normalize_tag_verbatim(self, input_str: str):
        """
        NOTE. LongNH
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        differences: list[Difference] = []
        
        # Collect all verbatim matches and their positions
        all_verbatims = []
        for key, value in VERBATIM.items():
            key_stripped = key.strip()
            start_pos = 0
            while True:
                pos = input_str.find(key_stripped, start_pos)
                if pos == -1:
                    break
                all_verbatims.append({
                    'original': key_stripped,
                    'replacement': value.strip(),
                    'start': pos,
                    'end': pos + len(key_stripped)
                })
                start_pos = pos + 1  # Continue searching for overlapping matches
        
        # Sort by position and length (longer matches first, then by position)
        all_verbatims.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        # Filter out overlapping matches (prefer longer matches)
        filtered_verbatims = []
        for verbatim in all_verbatims:
            # Check if this verbatim overlaps with any already accepted verbatim
            overlaps = False
            for accepted in filtered_verbatims:
                if not (verbatim['end'] <= accepted['start'] or verbatim['start'] >= accepted['end']):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_verbatims.append(verbatim)
        
        # Sort filtered list by position (right to left) for processing
        filtered_verbatims.sort(key=lambda x: x['start'], reverse=True)
        
        # Process replacements from right to left to preserve positions
        for verbatim in filtered_verbatims:
            # Replace in input string
            replacement = ' ' + verbatim['replacement'] + ' '
            input_str = input_str[:verbatim['start']].rstrip() + replacement + input_str[verbatim['end']:].lstrip()
            
            # Track difference (adjust for padding)
            differences.append(Difference(
                start_index=verbatim['start'] - 1,  # Adjust for added space at start
                end_index=verbatim['end'] - 1,
                original=verbatim['original'],
                modified=verbatim['replacement'],
                tag="VERBATIM"
            ))
        
        # Validate results
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]
    
    def normalize_endline(self, input_str: str):
        """
        Normalize newlines to comma separated format
        """
        return re.sub(r"\n+", " , ", input_str)
    
    def normalize_connector(self, input_str: str):
        """
        Normalize connector characters (replace '/' with 'trên')
        """
        ori_input_str = input_str
        differences = []
        
        # Find all occurrences of '/' and track their positions
        connector_positions = []
        start = 0
        while True:
            pos = input_str.find('/', start)
            if pos == -1:
                break
            connector_positions.append(pos)
            start = pos + 1
        
        # Process right-to-left to preserve positions
        for pos in reversed(connector_positions):
            # Replace in input string
            input_str = input_str[:pos].rstrip() + ' trên ' + input_str[pos + 1:].lstrip()
            
            # Track difference
            differences.append(Difference(
                start_index=pos,
                end_index=pos + 1,
                original='/',
                modified='trên',
                tag="CONNECTOR"
            ))
        
        # Validate results
        assert_differences(ori_input_str, differences)
        
        return input_str, differences[::-1]
    
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
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        # Pattern for alphanumeric sequences
        p = re.compile(
            r"([a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}\d{1,10}|"
            r"\d{1,10}[a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}|"
            r"[a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}|"
            r"\d{1,10}[a-zA-Z]{1,10}\d{1,10}|"
            r"[a-zA-Z]{1,10}\d{1,10}|"
            r"\d{1,10}[a-zA-Z]{1,10})"
        )
        
        # Phase 1: Collect all matches with their positions
        az09_list = []
        start = 0
        while True:
            match = p.search(input_str, start)
            if not match:
                break
            az09_list.append((match.group(), match.span()[0]))
            start = match.span()[1]
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for az09_text, start_pos in az09_list[::-1]:
            # Convert to spoken text
            az09_normalized = alpha_num2words(az09_text).strip()
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                az09_text,
                az09_normalized,
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, az09_text)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(az09_text),
                original=az09_text,
                modified=az09_normalized,
                tag="ALPHANUM"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_single_stand_letter(self, input_str: str):
        """
        NOTE. LongNH
        """
        ori_input_str = input_str
        differences: list[Difference] = []
        
        # Collect all standalone letters and their positions
        all_letters = []
        for k, v in ALPHABET_DICT.items():
            pattern = rf"\b{k}\b"
            for match in re.finditer(pattern, input_str):
                all_letters.append({
                    'original': k,
                    'replacement': v.strip(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Sort by position (right to left) for processing
        all_letters.sort(key=lambda x: x['start'], reverse=True)
        
        # Process replacements from right to left to preserve positions
        for letter in all_letters:
            # Replace in input string
            replacement = letter['replacement']
            input_str = input_str[:letter['start']] + replacement + input_str[letter['end']:]
            
            # Track difference
            differences.append(Difference(
                start_index=letter['start'],
                end_index=letter['end'],
                original=letter['original'],
                modified=letter['replacement'],
                tag="SINGLE_LETTER"
            ))
        
        # Validate results
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_date(self, input_str: str):
        """
        Normalize dates.

        NOTE. LongNH
        """
        
        # Define the child functions in order
        date_functions = [
            self._norm_date_type_1,
            self._norm_date_type_2,
            self._norm_date_type_3,
            self._norm_date_type_4,
            self._norm_date_type_5,
        ]
        
        # Process each function and merge differences
        current_text = input_str
        all_differences = []
        
        for date_func in date_functions:
            normalized_text, current_differences = date_func(current_text)
            all_differences = merge_differences(all_differences, current_differences)
            current_text = normalized_text
        
        return current_text, all_differences
    
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
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        dates_dmy_list = []
        start = 0
        
        # First pass: search for dates with prefix
        while True:
            date = p.search(input_str, start)
            if not date:
                break
            prefix = date.group(1)
            x = date.group(2)
            has_prefix = prefix.strip().lower() in ["ngày", "hôm"]
            dates_dmy_list.append((has_prefix, x, date.group(), date.span()[0]))
            addon_start = date.span()[1] - 1
            
            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                date_addon = p_addons.search(remaining_str)
                if not date_addon:
                    break
                x = date_addon.group(2)
                # Convert relative position to absolute position
                abs_start = addon_start + date_addon.span()[0]
                dates_dmy_list.append((False, x, date_addon.group(), abs_start))
                addon_start = addon_start + date_addon.span()[1] - 1
            
            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for has_prefix, date, date_repl, start_pos in dates_dmy_list[::-1]:
            # Convert to spoken text
            date_str = date_dmy2words(date, add_day_prefix=not has_prefix).strip()
            
            # Two-step replacement like phone_number
            date_repl_normed, diff_starts = replace_with_starts(date_repl, date, date_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_repl,
                date_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, date)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date),
                original=date, modified=date_str,
                tag="DATE_DMY"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]
    
    def _norm_date_type_2(self, input_str: str):
        """
        Normalize dd/mm/yy[yy] (dmy) form of dates

        This function supports function type 1

        Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        dates_dmy_list = []
        start = 0
        while True:
            date = p.search(input_str, start)
            if not date:
                break
            dates_dmy_list.append((date.group(1), date.group(), date.span()[0]))
            start = date.span()[1] - 1
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for date, date_repl, start_pos in dates_dmy_list[::-1]:
            # Convert to spoken text
            date_str = date_dmy2words(date).strip()
            
            # Two-step replacement like phone_number
            date_repl_normed, diff_starts = replace_with_starts(date_repl, date, date_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_repl,
                date_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, date)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date),
                original=date, modified=date_str,
                tag="DATE_DMY"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def _norm_date_type_3(self, input_str: str):
        """
        Normalize dd/mm (dm) form of dates

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        dates_dm_list = []
        start = 0
        
        # First pass: search for dates with prefix
        while True:
            date = p.search(input_str, start)
            if not date:
                break
            dates_dm_list.append((date.group(5), date.group(), date.span()[0]))
            addon_start = date.span()[1] - 1
            
            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                date_addon = p_addons.search(remaining_str)
                if not date_addon:
                    break
                # Convert relative position to absolute position
                abs_start = addon_start + date_addon.span()[0]
                dates_dm_list.append((date_addon.group(2), date_addon.group(), abs_start))
                addon_start = addon_start + date_addon.span()[1] - 1
            
            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for date, date_repl, start_pos in dates_dm_list[::-1]:
            # Convert to spoken text
            date_str = date_dm2words(date, add_day_prefix=False).strip()
            
            # Two-step replacement like phone_number
            date_repl_normed, diff_starts = replace_with_starts(date_repl, date, date_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_repl,
                date_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, date)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date),
                original=date, modified=date_str,
                tag="DATE_DM"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def _norm_date_type_4(self, input_str: str):
        """
        Normalize dd/mm form without clear rules
        
        Cám ơn cha dành cho Ngày của cha (16/6) tới đây của Hồ Ngọc Hà...

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        dates_dm_list = []
        start = 0
        
        # First pass: search for dates with prefix
        while True:
            date = p.search(input_str, start)
            if not date:
                break
            dates_dm_list.append((date.group(2), date.group(), date.span()[0]))
            addon_start = date.span()[1] - 1
            
            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                date_addon = p_addons.search(remaining_str)
                if not date_addon:
                    break
                # Convert relative position to absolute position
                abs_start = addon_start + date_addon.span()[0]
                dates_dm_list.append((date_addon.group(2), date_addon.group(), abs_start))
                addon_start = addon_start + date_addon.span()[1] - 1
            
            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for date, date_repl, start_pos in dates_dm_list[::-1]:
            # Convert to spoken text
            date_str = date_dm2words(date, add_day_prefix=False).strip()
            
            # Two-step replacement like phone_number
            date_repl_normed, diff_starts = replace_with_starts(date_repl, date, date_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_repl,
                date_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, date)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date),
                original=date, modified=date_str,
                tag="DATE_DM"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def _norm_date_type_5(self, input_str: str):
        """
        Normalize mm/yyyy (my) form of dates

        @improve:
        
        những cases không có [Tt]háng ở trước --> thêm 'tháng' ở date_my2word()
        
        nhưng tránh các trường hợp Quý 2/2018, đợt 3/2019, tỷ lệ 1/2000

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        dates_my_list = []
        start = 0
        
        # First pass: search for dates with prefix
        while True:
            date = p.search(input_str, start)
            if not date:
                break
            # Handle prefix logic
            if (date.group(2) + date.group(3)).strip() == "":
                prefix = None
            else:
                prefix = date.group(1).lower()
            dates_my_list.append((date.group(5), prefix, date.group(), date.span()[0]))
            addon_start = date.span()[1] - 1
            
            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                date_addon = p_addons.search(remaining_str)
                if not date_addon:
                    break
                # Convert relative position to absolute position
                abs_start = addon_start + date_addon.span()[0]
                dates_my_list.append((date_addon.group(2), prefix, date_addon.group(), abs_start))
                addon_start = addon_start + date_addon.span()[1] - 1
            
            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for date, prefix, date_repl, start_pos in dates_my_list[::-1]:
            # Convert to spoken text
            date_str = date_my2words(date, add_month_prefix=False).strip()
            if prefix is not None:
                date_str = prefix + " " + date_str
                date_str = date_str.strip()
            
            # Two-step replacement like phone_number
            date_repl_normed, diff_starts = replace_with_starts(date_repl, date, date_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_repl,
                date_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, date)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date),
                original=date, modified=date_str,
                tag="DATE_MY"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_date_range(self, input_str: str):
        """
        NOTE. LongNH
        """
        # Define the child functions in order
        date_range_functions = [
            self._norm_date_range_type_1,
            self._norm_date_range_type_2,
            self._norm_date_range_type_3,
            self._norm_date_range_type_4,
            self._norm_date_range_type_5,
            self._norm_date_range_type_6,
            self._norm_date_range_type_7,
        ]
        
        # Process each function and merge differences
        current_text = input_str
        all_differences = []
        
        for date_range_func in date_range_functions:
            normalized_text, current_differences = date_range_func(current_text)
            all_differences = merge_differences(all_differences, current_differences)
            current_text = normalized_text
        
        return current_text, all_differences
    
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
        ori_input_str = input_str
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
        year_range_list = []
        start = 0

        while True:
            year_range = p.search(input_str, start)
            if not year_range:
                break
            is_school_year = year_range.group(2) is not None
            year_range_list.append((year_range.group(5), is_school_year, year_range.group(), year_range.span()[0]))
            addon_start = year_range.span()[1] - 1

            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                year_range_addon = p_addons.search(remaining_str)
                if not year_range_addon:
                    break
                # Convert relative position to absolute position
                abs_start = addon_start + year_range_addon.span()[0]
                year_range_list.append((year_range_addon.group(2), is_school_year, year_range_addon.group(), abs_start))
                addon_start = addon_start + year_range_addon.span()[1] - 1

            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        for year_range, is_school_year, year_range_repl, start_pos in year_range_list[::-1]:
            # convert to spoken text
            year_range_norm = year_range.replace('-', ' - ')
            year_range_norm = " ".join(year_range_norm.split())
            start_year = year_range_norm.split('-')[0]
            end_year = year_range_norm.split('-')[1]
            if is_school_year:
                year_range_str = num2words_integer(start_year) + ' ' + num2words_integer(end_year)
            else:
                year_range_str = num2words_integer(start_year) + ' đến năm ' + num2words_integer(end_year)
            
            # replace the text to the spoken text in the bigger replacement year_range_repl
            year_range_repl_normed, diff_starts = replace_with_starts(year_range_repl, year_range, year_range_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # replace entire matched chunk in full string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                year_range_repl, 
                year_range_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, year_range)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(year_range), 
                original=year_range, modified=year_range_str,
                tag="DATE_RANGE_Y_Y"
            ) for s in diff_starts]

        # assert differences with the original input str
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def _norm_date_range_type_2(self, input_str: str):
        """
        Normalize dd/mm/yyyy-dd/mm/yyyy forms

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        date_range_dmy_list = []
        start = 0
        
        while True:
            date_range_dmy = p.search(input_str, start)
            if not date_range_dmy:
                break
            x = date_range_dmy.group(2)
            has_ngay = self.__has_ngay_prefix(date_range_dmy.group(1))
            date_range_dmy_list.append((x, has_ngay, date_range_dmy.group(), date_range_dmy.span()[0]))
            start = date_range_dmy.span()[1] - 1

        differences: list[Difference] = []
        for date_range_dmy, has_ngay, date_range_dmy_repl, start_pos in date_range_dmy_list[::-1]:
            # convert to spoken text
            start_date = date_range_dmy.split('-')[0]
            end_date = date_range_dmy.split('-')[1]
            date_range_dmy_str = date_dmy2words(start_date, add_day_prefix=not has_ngay).strip() + ' đến ' + date_dmy2words(end_date).strip()
            
            # replace the text to the spoken text in the bigger replacement date_range_dmy_repl
            date_range_dmy_repl_normed, diff_starts = replace_with_starts(date_range_dmy_repl, date_range_dmy, date_range_dmy_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # replace entire matched chunk in full string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_range_dmy_repl, 
                date_range_dmy_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, date_range_dmy)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date_range_dmy), 
                original=date_range_dmy, modified=date_range_dmy_str,
                tag="DATE_RANGE_DMY_DMY"
            ) for s in diff_starts]

        # shift -1 because padding is added at first: " " + input_str + " "
        shift_differences(differences, -1)
        # assert differences with the original input str
        assert_differences(ori_input_str, differences)
                
        return input_str.strip(), differences[::-1]

    def _norm_date_range_type_3(self, input_str: str):
        """
        Normalize dd-dd/mm/yyyy forms

        NOTE. LongNH
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        p_text = r"((0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4}))"
        p = re.compile(
            r"(" + self.DATE_PREFIX_REGEX + r")*\s+"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        date_range_dmy_list = []
        start = 0

        while True:
            date_range_dmy = p.search(input_str, start)
            if not date_range_dmy:
                break
            x = date_range_dmy.group(2)
            # has_prefix = date_range_dmy.group(1) is not None 
            has_prefix = not self.__should_add_tu(date_range_dmy.group(1))
            has_ngay_prefix = self.__has_ngay_prefix(date_range_dmy.group(1))
            date_range_dmy_list.append((x, has_ngay_prefix, has_prefix, date_range_dmy.group(), date_range_dmy.span()[0]))
            start = date_range_dmy.span()[1] - 1

        differences: list[Difference] = []
        for date_range_dmy, has_ngay, has_prefix, date_range_repl, start_pos in date_range_dmy_list[::-1]:
            # convert to spoken text
            start_date = date_range_dmy.split('-')[0]
            end_date = date_range_dmy.split('-')[1]
            date_range_dmy_str = num2words_integer(start_date) + ' đến ' + date_dmy2words(end_date).strip()
            if int(start_date) < 10:
                date_range_dmy_str = "mùng " + date_range_dmy_str
            if not has_ngay:
                date_range_dmy_str = "ngày " + date_range_dmy_str
            if not has_prefix:
                date_range_dmy_str = "từ " + date_range_dmy_str
            
            # replace the text to the spoken text in the bigger replacement date_range_repl
            date_range_repl_normed, diff_starts = replace_with_starts(date_range_repl, date_range_dmy, date_range_dmy_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # replace entire matched chunk in full string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_range_repl, 
                date_range_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, date_range_dmy)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date_range_dmy), 
                original=date_range_dmy, modified=date_range_dmy_str,
                tag="DATE_RANGE_D_DMY"
            ) for s in diff_starts]

        # shift -1 because padding is added at first: " " + input_str + " "
        shift_differences(differences, -1)
        # assert differences with the original input str
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def _norm_date_range_type_4(self, input_str: str):
        """
        Normalize dd/mm-dd/mm/yyyy forms

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        date_range_dmy_list = []
        start = 0

        while True:
            date_range_dmy = p.search(input_str, start)
            if not date_range_dmy:
                break
            x = date_range_dmy.group(2)
            # has_prefix = date_range_dmy.group(1) is not None
            has_prefix = not self.__should_add_tu(date_range_dmy.group(1))
            date_range_dmy_list.append((x, self.__has_ngay_prefix(date_range_dmy.group(1)), has_prefix, date_range_dmy.group(), date_range_dmy.span()[0]))
            start = date_range_dmy.span()[1] - 1

        differences: list[Difference] = []
        for date_range_dmy, has_ngay, has_prefix, date_range_repl, start_pos in date_range_dmy_list[::-1]:
            # convert to spoken text
            start_date = date_range_dmy.split('-')[0]
            end_date = date_range_dmy.split('-')[1]
            date_range_dmy_str = date_dm2words(start_date, add_day_prefix=not has_ngay).strip() + ' đến ' + date_dmy2words(end_date).strip()
            if not has_prefix:
                date_range_dmy_str = "từ " + date_range_dmy_str
            
            # replace the text to the spoken text in the bigger replacement date_range_repl
            date_range_repl_normed, diff_starts = replace_with_starts(date_range_repl, date_range_dmy, date_range_dmy_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # replace entire matched chunk in full string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_range_repl, 
                date_range_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, date_range_dmy)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date_range_dmy), 
                original=date_range_dmy, modified=date_range_dmy_str,
                tag="DATE_RANGE_DM_DMY"
            ) for s in diff_starts]

        # shift -1 because padding is added at first: " " + input_str + " "
        shift_differences(differences, -1)
        # assert differences with the original input str
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def _norm_date_range_type_5(self, input_str: str):
        """
        Normalize dd/mm-dd/mm forms: 20/1-18/2

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        date_range_dm_list = []
        start = 0

        while True:
            date_range_dm = p.search(input_str, start)
            if not date_range_dm:
                break
            x = date_range_dm.group(2)
            date_range_dm_list.append((
                x, 
                self.__has_ngay_prefix(date_range_dm.group(1)), 
                # True,
                not self.__should_add_tu(date_range_dm.group(1)),
                date_range_dm.group(), date_range_dm.span()[0]
            ))
            addon_start = date_range_dm.span()[1] - 1

            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                date_range_addon = p_addons.search(remaining_str)
                if not date_range_addon:
                    break
                x = date_range_addon.group(2)
                # Convert relative position to absolute position
                abs_start = addon_start + date_range_addon.span()[0]
                date_range_dm_list.append((x, False, False, date_range_addon.group(), abs_start))
                addon_start = addon_start + date_range_addon.span()[1] - 1

            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        for date_range_dm, has_ngay, has_prefix, date_range_repl, start_pos in date_range_dm_list[::-1]:
            # convert to spoken text
            start_date = date_range_dm.split('-')[0]
            end_date = date_range_dm.split('-')[1]
            date_range_dm_str = date_dm2words(start_date, add_day_prefix=not has_ngay).strip() + ' đến ' + date_dm2words(end_date).strip()
            if not has_prefix:
                date_range_dm_str = "từ " + date_range_dm_str
            
            # replace the text to the spoken text in the bigger replacement date_range_repl
            date_range_repl_normed, diff_starts = replace_with_starts(date_range_repl, date_range_dm, date_range_dm_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # replace entire matched chunk in full string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_range_repl, 
                date_range_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, date_range_dm)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date_range_dm), 
                original=date_range_dm, modified=date_range_dm_str,
                tag="DATE_RANGE_DM_DM"
            ) for s in diff_starts]

        # shift -1 because padding is added at first: " " + input_str + " "
        shift_differences(differences, -1)
        # assert differences with the original input str
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def _norm_date_range_type_6(self, input_str: str):
        """
        Normalize dd-dd/mm forms: 15-18/6, 15 -18/6, 15- 18/6

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        date_range_dm_list = []
        start = 0

        while True:
            date_range_dm = p.search(input_str, start)
            if not date_range_dm:
                break
            # has_prefix = date_range_dm.group(1) is not None 
            has_prefix = not self.__should_add_tu(date_range_dm.group(1))
            x = date_range_dm.group(2)
            date_range_dm_list.append((
                x, 
                self.__has_ngay_prefix(date_range_dm.group(1)),
                has_prefix,
                date_range_dm.group(),
                date_range_dm.span()[0]
            ))
            addon_start = date_range_dm.span()[1] - 1

            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                date_range_addon = p_addons.search(remaining_str)
                if not date_range_addon:
                    break
                x = date_range_addon.group(2)
                # Convert relative position to absolute position
                abs_start = addon_start + date_range_addon.span()[0]
                date_range_dm_list.append((
                    x, 
                    self.__has_ngay_prefix(date_range_addon.group(1)),
                    False,
                    date_range_addon.group(),
                    abs_start
                ))
                addon_start = addon_start + date_range_addon.span()[1] - 1

            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        for date_range, has_ngay, has_prefix, date_range_repl, start_pos in date_range_dm_list[::-1]:
            # TODO. Small replacement: replace spoken text within matched text chunk
            start_date = date_range.split('-')[0]
            end_date = date_range.split('-')[1]
            date_range_str = num2words_integer(start_date) + ' đến ' + date_dm2words(end_date).strip()
            if int(start_date) < 10:
                date_range_str = 'mùng ' + date_range_str
            if not has_ngay:
                date_range_str = 'ngày ' + date_range_str
            if not has_prefix:
                date_range_str = 'từ ' + date_range_str
            # replace the text to the spoken text in the bigger replacement date_range_repl
            date_range_repl_normed, diff_starts = replace_with_starts(date_range_repl, date_range, date_range_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # TODO. Big replacement: replace entired matched chunk in full string
            # replace
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_range_repl, 
                date_range_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, date_range)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date_range), 
                original=date_range, modified=date_range_str,
                tag="DATE_RANGE_D_DM"
            ) for s in diff_starts]

        # shift -1 because padding is added at first: " " + input_str + " "
        shift_differences(differences, -1)
        # assert differences with the original input str
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]
    
    def _norm_date_range_type_7(self, input_str: str):
        """
        Normalize mm-mm/yyyy forms

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        date_range_my_list = []
        start = 0

        while True:
            date_range_my = p.search(input_str, start)
            if not date_range_my:
                break
            # has_prefix = date_range_my.group(1) is not None
            has_prefix = not self.__should_add_tu(date_range_my.group(1))
            x = date_range_my.group(2)
            date_range_my_list.append((
                x, 
                self.__has_thang_prefix(date_range_my.group(1)),
                has_prefix,
                date_range_my.group(),
                date_range_my.span()[0]
            ))
            addon_start = date_range_my.span()[1] - 1

            # Nested search for additional dates - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                date_range_addon = p_addons.search(remaining_str)
                if not date_range_addon:
                    break
                x = date_range_addon.group(2)
                # Convert relative position to absolute position
                abs_start = addon_start + date_range_addon.span()[0]
                date_range_my_list.append((
                    x, 
                    self.__has_thang_prefix(date_range_addon.group(1)),
                    False,
                    date_range_addon.group(),
                    abs_start
                ))
                addon_start = addon_start + date_range_addon.span()[1] - 1

            # Continue main search from where addons left off
            start = addon_start

        differences: list[Difference] = []
        for date_range, has_thang, has_prefix, date_range_repl, start_pos in date_range_my_list[::-1]:
            # TODO. Small replacement: replace spoken text within matched text chunk
            start_date = date_range.split('-')[0]
            end_date = date_range.split('-')[1]
            date_range_str = num2words_integer(start_date) + ' đến ' + date_my2words(end_date, add_month_prefix=not has_thang).strip()
            if not has_prefix:
                date_range_str = 'từ ' + date_range_str
            # replace the text to the spoken text in the bigger replacement date_range_repl
            date_range_repl_normed, diff_starts = replace_with_starts(date_range_repl, date_range, date_range_str)
            assert len(diff_starts) == 1
            # get the position of spoken text in matched text chunk
            shift = diff_starts[0]
            
            # TODO. Big replacement: replace entired matched chunk in full string
            # replace
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                date_range_repl, 
                date_range_repl_normed,
                start_pos
            )
            # shift the start with the position of spoken text in matched text chunk
            diff_starts = shift_starts(diff_starts, shift)
            # assert to ensure the start of replacement is matched with the input_str
            assert_match_with_starts(diff_starts, input_str, date_range)

            # add differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(date_range), 
                original=date_range, modified=date_range_str,
                tag="DATE_RANGE_M_MY"
            ) for s in diff_starts]

        # shift -1 because padding is added at first: " " + input_str + " "
        shift_differences(differences, -1)
        # assert differences with the original input str
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def normalize_tag_roman_num(self, input_str: str):
        # Define the child functions in order
        roman_functions = [
            self._norm_tag_roman_num_v1,
            self._norm_tag_roman_num_v2,
        ]
        
        # Process each function and merge differences
        current_text = input_str
        all_differences = []
        
        for roman_func in roman_functions:
            normalized_text, current_differences = roman_func(current_text)
            all_differences = merge_differences(all_differences, current_differences)
            current_text = normalized_text
        
        return current_text, all_differences

    def _norm_tag_roman_num_v1(self, input_str: str):
        """
        Normalize roman numerals v1 - with parentheses
        
        NOTE. LongNH
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        roman_numeral_p = re.compile(
            r"("
            r"\s(\(\s*X{0,3})(IX|IV|V?I{0,3})|"
            r"\s(\(\s*x{0,3})(ix|iv|v?i{0,3})"
            r")"
            + self.ENDING_PUNCTUATIONS_REGEX

            , re.IGNORECASE
        )
        
        # Phase 1: Collect all matches with their positions
        roman_numeral_list = []
        temp_str = input_str
        
        while True:
            roman_numeral = roman_numeral_p.search(temp_str)
            if not roman_numeral:
                break
            
            # Clean roman numeral text
            roman = roman_numeral.group().strip()
            roman = roman.replace(' ', '')
            for character in '. , ( ) /'.split():
                roman = roman.replace(character, '')
            roman = roman.strip()
            
            if roman != "":
                # Calculate absolute position in original padded string
                full_match = roman_numeral.group()
                match_start_in_temp = roman_numeral.start()
                processed_length = len(input_str) - len(temp_str)
                abs_start = processed_length + match_start_in_temp
                
                roman_numeral_list.append((roman, full_match, abs_start))
            
            temp_str = temp_str[roman_numeral.span()[1]-1:]
        
        # Sort by length and value like original
        roman_numeral_list = sorted(roman_numeral_list, key=lambda x: (len(x[0]), x[0]), reverse=True)
        
        differences: list[Difference] = []
        
        # Phase 2: Process matches (already sorted by length and value)
        for roman, roman_repl, start_pos in roman_numeral_list:
            # Convert to spoken text
            try:
                roman2int = fromRoman(roman.upper())
            except InvalidRomanNumeralError:
                continue
            roman_numeral_str = num2words_integer(str(roman2int))
            
            # Two-step replacement like date functions
            roman_repl_normed, diff_starts = replace_with_starts(roman_repl, roman, roman_numeral_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                roman_repl,
                roman_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, roman)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(roman),
                original=roman, modified=roman_numeral_str,
                tag="ROMAN"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]
    
    def _norm_tag_roman_num_v2(self, input_str: str):
        """
        Normalize roman numerals v2 - with context checking
        
        @improve: 'Nữ hoàng Anh Elizabeth II thường đi lại trên chiếc xe của Land
        Rovers và Jaguars' --> II đọc thành là 'đệ nhị'
        
        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        roman_numeral_list = []
        start = 0
        
        # First pass: search for roman numerals with context
        while True:
            roman_numeral = p.search(input_str, start)
            if not roman_numeral:
                break
            x = roman_numeral.group(2)
            roman_numeral_list.append((x, roman_numeral.group(), roman_numeral.span()[0]))
            addon_start = roman_numeral.span()[1] - 1
            
            # Nested search for additional roman numerals - search in remaining substring from addon_start
            while True:
                # Search p_addons from the beginning of the remaining substring
                remaining_str = input_str[addon_start:]
                roman_numeral_addon = p_addons.search(remaining_str)
                if not roman_numeral_addon:
                    break
                x = roman_numeral_addon.group(2)
                # Convert relative position to absolute position
                abs_start = addon_start + roman_numeral_addon.span()[0]
                roman_numeral_list.append((x, roman_numeral_addon.group(), abs_start))
                addon_start = addon_start + roman_numeral_addon.span()[1] - 1
            
            # Continue main search from where addons left off
            start = addon_start
        
        # Sort by position in descending order (right-to-left processing)
        roman_numeral_list = sorted(roman_numeral_list, key=lambda x: x[2], reverse=True)
        
        differences: list[Difference] = []
        
        # Phase 2: Process matches (already sorted by length and value)
        for roman_numeral, roman_numeral_repl, start_pos in roman_numeral_list:
            # Convert to spoken text
            try:
                roman2int = fromRoman(roman_numeral.upper())
            except InvalidRomanNumeralError:
                continue
            roman_numeral_str = num2words_integer(str(roman2int))
            
            # Two-step replacement like date functions
            # Note: roman_numeral is stored as-is from regex (could be mixed case like "Xi")
            # but fromRoman expects uppercase, so we use the original case for replacement
            roman_numeral_repl_normed, diff_starts = replace_with_starts(roman_numeral_repl, roman_numeral, roman_numeral_str)
            assert len(diff_starts) == 1, f"Expected 1 match for '{roman_numeral}' in '{roman_numeral_repl}', got {len(diff_starts)}"
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                roman_numeral_repl,
                roman_numeral_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, roman_numeral)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s, end_index=s + len(roman_numeral),
                original=roman_numeral, modified=roman_numeral_str,
                tag="ROMAN"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_time(self, input_str: str):
        """
        Normalize time and time range

        NOTE. LongNH
        """
        # Define the child functions in order
        time_functions = [
            self._norm_time_type_1,
            self._norm_time_type_2,
        ]
        
        # Process each function and merge differences
        current_text = input_str
        all_differences = []
        
        for time_func in time_functions:
            normalized_text, current_differences = time_func(current_text)
            all_differences = merge_differences(all_differences, current_differences)
            current_text = normalized_text
        
        return current_text, all_differences

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
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        # Pattern for individual times within ranges
        single_time_pattern = re.compile(self.TIME_REGEX)
        
        # Pattern for time ranges
        time_range_patterns = re.compile(
            r"(" + self.TIME_REGEX + r")"
            r"\s*-\s*"
            r"(" + self.TIME_REGEX + r")"
        )
        
        # Phase 1: Collect all time ranges with their positions
        duration_times_list = []
        start = 0
        while True:
            time_range = time_range_patterns.search(input_str, start)
            if not time_range:
                break
            duration_times_list.append((time_range.group(), time_range.group(), time_range.span()[0]))
            start = time_range.span()[1]
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for duration_time, duration_time_repl, start_pos in duration_times_list[::-1]:
            # Extract individual times from the range
            times = []
            temp_time = duration_time
            start_time = 0
            while True:
                time_match = single_time_pattern.search(temp_time, start_time)
                if not time_match:
                    break
                times.append(time_match.group())
                start_time = time_match.span()[1]
            
            # Filter out invalid times (24:xx where xx > 00)
            times = [time for time in times if not(time.startswith('24') and (time[3:] > '00'))]
            
            if len(times) >= 2:
                # Convert to spoken text
                time_str = time2words(times[0]).strip() + ' đến ' + time2words(times[1]).strip()
                time_str = time_str.strip()
                
                # Replace in input string
                tmp_str, diff_starts = replace_with_starts(
                    input_str,
                    duration_time_repl,
                    time_str,
                    start_pos
                )
                
                # Verify replacement
                assert_match_with_starts(diff_starts, input_str, duration_time_repl)
                
                # Update string and track differences
                input_str = tmp_str
                differences += [Difference(
                    start_index=s,
                    end_index=s + len(duration_time_repl),
                    original=duration_time_repl,
                    modified=time_str,
                    tag="TIME_RANGE"
                ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def _norm_time_type_2(self, input_str: str):
        """
        Normalize time

        NOTE. LongNH
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        time_patterns = re.compile(self.TIME_REGEX)
        
        # Phase 1: Collect all times with their positions
        times_list = []
        start = 0
        while True:
            time_match = time_patterns.search(input_str, start)
            if not time_match:
                break
            times_list.append((time_match.group(), time_match.group(), time_match.span()[0]))
            start = time_match.span()[1]
        
        # Filter out invalid times (24:xx where xx > 00)
        times_list = [(time, time_repl, start_pos) for time, time_repl, start_pos in times_list 
                      if not(time.startswith('24') and (time[3:] > '00'))]
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for time, time_repl, start_pos in times_list[::-1]:
            # Convert to spoken text
            time_str = time2words(time).strip()
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                time_repl,
                time_str,
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, time_repl)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(time_repl),
                original=time_repl,
                modified=time_str,
                tag="TIME"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_multiply_number(self, input_str: str):
        """
        Normalize ???

        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches
        number_list = []
        start = 0

        while True:
            mul = p.search(input_str, start)
            if not mul:
                break
            number_list.append((mul.group(), *mul.span()))
            start = mul.span()[1]

        differences = []
        
        # Phase 2: Process right-to-left to preserve positions
        for number, s_idx, e_idx in number_list[::-1]:
            number_str = multiply(number).strip()
            
            # Replace in input string
            input_str = input_str[: s_idx] + number_str + input_str[e_idx :]
            
            # Track difference (adjust for padding)
            differences.append(Difference(
                start_index=s_idx - 1,  # Adjust for added space at start
                end_index=e_idx - 1,
                original=number,
                modified=number_str,
                tag="MULTIPLY"
            ))

        # Validate results
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def normalize_sport_score(self, input_str: str):
        """
        Normalize sport scores
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        temp_str = input_str.lower()
        
        # Find all score patterns
        score_matches = re.finditer(r"(\s[0-9]+(\-|\:)[0-9]+)", temp_str)
        score_list = []
        for match in score_matches:
            score_list.append({
                'original': match.group(1),
                'start': match.start(1),
                'end': match.end(1)
            })
        
        # Check if this is sports context
        sport_ngrams = ['tỷ số', 'chiến thắng', 'trận đấu', 'tỉ số', 'bàn thắng', 'trên sân', 'đội bóng', 'thi đấu', 'cầu thủ',
                        'vô địch', 'mùa giải', 'đánh bại', 'đối thủ', 'bóng đá', 'gỡ hòa', 'chung kết', 'bán kết', 'ghi bàn',
                        'chủ nhà', 'tiền đạo', 'dứt điểm', 'tiền vệ', 'tiền đạo', 'thua', 'bị dẫn']
        is_sport = 0
        for item in sport_ngrams:
            if temp_str.find(item) != -1:
                is_sport = 1
                break
        
        differences: list[Difference] = []
        
        # Process scores only if it's sports context
        if is_sport == 1 and len(score_list) > 0:
            # Sort by position (right to left) for processing
            score_list.sort(key=lambda x: x['start'], reverse=True)
            
            for score_info in score_list:
                score = score_info['original']
                if '-' in score:
                    lscore = num2words_integer(score.split('-')[0])
                    rscore = num2words_integer(score.split('-')[1])
                    score_norm = lscore + ' ' + rscore
                else:
                    lscore = num2words_integer(score.split(':')[0])
                    rscore = num2words_integer(score.split(':')[1])
                    score_norm = lscore + ' ' + rscore
                
                # Replace in input string
                replacement = ' ' + score_norm + ' '
                input_str = input_str[:score_info['start']] + replacement + input_str[score_info['end']:]
                
                # Track difference (adjust for padding and leading space)
                differences.append(Difference(
                    start_index=score_info['start'],  # Start after the space
                    end_index=score_info['end'] - 1,  # Adjust for added space at start
                    original=score.strip(),
                    modified=score_norm,
                    tag="SPORT_SCORE"
                ))

        # Validate results
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def normalize_number_plate(self, input_str: str):
        """
        Normalize number plate

        NOTE. LongNH
        """
        input_str, differences = self._norm_number_plate_type_1(input_str)
        # input_str = self._norm_number_plate_type_2(input_str)
        # input_str = self._norm_number_plate_type_3(input_str)
        # input_str = self._norm_number_plate_type_4(input_str)
        return input_str, differences

    def _norm_number_plate_type_1(self, input_str: str):
        """
        Normalize plate

        NOTE. LongNH
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        p_text = (
            r"([0-9]+[a-zA-Z]+[0-9]*)"
            r"(\s{0,1}[\-]{0,1}\s{0,1})"
            r"([0-9]+\.*[0-9]+)"
        )
        p = re.compile(
            r"([Bb]iển [Kk]iểm [Ss]oát|"
            r"[Bb]iển [Ss]ố [Xx]e|"
            r"[Xx]e|"
            r"[Ôô] tô|"
            r"[Ôô]tô|"
            r"[Bb]iển [Ss]ố|"
            r"[Bb]iển)+"
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
        
        # Phase 1: Collect all matches with their positions
        number_plate_list = []
        start = 0
        
        # First pass: search for plates with prefix
        while True:
            number_plate = p.search(input_str, start)
            if not number_plate:
                break
            number_plate_list.append((
                number_plate.group(5) + number_plate.group(6) + number_plate.group(7), 
                number_plate.group(),
                number_plate.span()[0]
            ))
            addon_start = number_plate.span()[1] - 1
            
            # Nested search for additional plates
            while True:
                remaining_str = input_str[addon_start:]
                number_plate_addon = p_addons.search(remaining_str)
                if not number_plate_addon:
                    break
                abs_start = addon_start + number_plate_addon.span()[0]
                number_plate_list.append((
                    number_plate_addon.group(2) + number_plate_addon.group(3) + number_plate_addon.group(4),
                    number_plate_addon.group(),
                    abs_start
                ))
                addon_start = addon_start + number_plate_addon.span()[1] - 1
            
            start = addon_start
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for plate, plate_repl, start_pos in number_plate_list[::-1]:
            # Convert prefix and suffix to spoken text
            plate_str = plate.replace(" ", "-")
            plate_str = alpha_num2words(plate_str, skip_symbols=True)
            
            # Two-step replacement
            plate_repl_normed, diff_starts = replace_with_starts(plate_repl, plate, plate_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                plate_repl,
                plate_repl_normed,
                start_pos
            )
            
            diff_starts = shift_starts(diff_starts, shift)
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, plate)
            
            # Update string and track differences
            input_str = tmp_str
            # Track both prefix and suffix differences
            differences += [Difference(
                start_index=s,
                end_index=s + len(plate),
                original=plate,
                modified=plate_str,
                tag="PLATE"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

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
        
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        digits_list = []
        start = 0
        
        # First pass: search for digits with prefix
        while True:
            digit = p.search(input_str, start)
            if not digit:
                break
            if has_number(digit.group(5)):
                digits_list.append((digit.group(5), digit.group(), digit.span()[0]))
            addon_start = digit.span()[1] - 1
            
            # Nested search for additional digits
            while True:
                remaining_str = input_str[addon_start:]
                digit_addon = p_addons.search(remaining_str)
                if not digit_addon:
                    break
                if has_number(digit_addon.group(2)):
                    abs_start = addon_start + digit_addon.span()[0]
                    digits_list.append((digit_addon.group(2), digit_addon.group(), abs_start))
                addon_start = addon_start + digit_addon.span()[1] - 1
            
            start = addon_start
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for digit, digit_repl, start_pos in digits_list[::-1]:
            # Convert to spoken text
            digits_str = alpha_num2words(digit).strip()
            
            # Two-step replacement
            digit_repl_normed, diff_starts = replace_with_starts(digit_repl, digit, digits_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                digit_repl,
                digit_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, digit)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(digit),
                original=digit,
                modified=digits_str,
                tag="ID_DIGIT"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

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

    def normalize_abbreviation(self, input_str: str):
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
        return self._norm_abbreviation(input_str, abbre_dict)
    
    def _norm_abbreviation(self, input_str: str, abbre_dict: dict):
        ori_input_str = input_str
        all_abbreviations = []
        
        # Phase 1: Collect all abbreviations with punctuation attached
        words_temp = input_str.split()
        word_start = 0
        for full_word in words_temp:
            word_pos = input_str.find(full_word, word_start)
            temp_word = full_word
            if temp_word and temp_word[-1] in [',', '.', ')', '}', ']', '!', '?', '/', '-', ':', ';', '(', '{', '[', '"', "'"]:
                temp_word = temp_word[:-1]

            if temp_word in abbre_dict.keys():
                replacement = str(abbre_dict[temp_word]).strip().lower()
                abbr_pos = input_str.find(temp_word, word_pos)
                if abbr_pos >= 0:
                    all_abbreviations.append({
                        'original': temp_word,
                        'replacement': replacement,
                        'start': abbr_pos,
                        'end': abbr_pos + len(temp_word),
                        'phase': 1
                    })
            
            word_start = word_pos + len(full_word)

        # Phase 2: Collect abbreviations after removing punctuation
        temp_str = ori_input_str  # Use original string for position tracking
        for character in [',', '.', ')', '}', ']', '!', '?', '/', '-', '(', '{', '[', '"', "'"]:
            temp_str = temp_str.replace(character, ' ')

        words_inp = temp_str.split()
        word_start = 0
        for temp_word in words_inp:
            if temp_word in abbre_dict.keys():
                replacement = str(abbre_dict[temp_word]).strip()
                abbr_pos = ori_input_str.find(temp_word, word_start)
                if abbr_pos >= 0:
                    # Check if this abbreviation was already found in phase 1
                    already_found = any(
                        abbr['original'] == temp_word and abbr['start'] == abbr_pos 
                        for abbr in all_abbreviations
                    )
                    if not already_found:
                        all_abbreviations.append({
                            'original': temp_word,
                            'replacement': replacement,
                            'start': abbr_pos,
                            'end': abbr_pos + len(temp_word),
                            'phase': 2
                        })
                
                word_start = abbr_pos + len(temp_word) if abbr_pos >= 0 else word_start

        # Sort by position (right to left) for processing
        all_abbreviations.sort(key=lambda x: x['start'], reverse=True)
        
        # Process replacements from right to left to preserve positions
        differences = []
        for abbr in all_abbreviations:
            # Replace in input string
            input_str = input_str[:abbr['start']] + abbr['replacement'] + input_str[abbr['end']:]
            
            # Track difference
            differences.append(Difference(
                start_index=abbr['start'],
                end_index=abbr['end'],
                original=abbr['original'],
                modified=abbr['replacement'],
                tag="ABBRE"
            ))

        # Validate results
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_tag_fraction(self, input_str: str):
        # Define the child functions in order
        fraction_functions = [
            self._norm_tag_fraction_type_1,
            self._norm_tag_fraction_type_2,
            self._norm_tag_fraction_type_3,
        ]
        
        # Process each function and merge differences
        current_text = input_str
        all_differences = []
        
        for fraction_func in fraction_functions:
            normalized_text, current_differences = fraction_func(current_text)
            all_differences = merge_differences(all_differences, current_differences)
            current_text = normalized_text
        
        return current_text, all_differences

    def _norm_tag_fraction_type_1(self, input_str: str):
        """
        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        ratio_list = []
        start = 0
        
        # First pass: search for fractions with prefix
        while True:
            ratio = p.search(input_str, start)
            if not ratio:
                break
            ratio_list.append((ratio.group(2).rstrip(",."), ratio.group(), ratio.span()[0]))
            addon_start = ratio.span()[1] - 1
            
            # Nested search for additional fractions
            while True:
                remaining_str = input_str[addon_start:]
                ratio_addon = p_addons.search(remaining_str)
                if not ratio_addon:
                    break
                abs_start = addon_start + ratio_addon.span()[0]
                ratio_list.append((ratio_addon.group(2).rstrip(",."), ratio_addon.group(), abs_start))
                addon_start = addon_start + ratio_addon.span()[1] - 1
            
            start = addon_start
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for ratio, ratio_repl, start_pos in ratio_list[::-1]:
            # Convert to spoken text
            first_num  = ratio.replace(":", "/").split('/')[0]
            second_num = ratio.replace(":", "/").split('/')[1]
            ratio_str = num2words_float(first_num) + ' phần ' + num2words_float(second_num)
            ratio_str = ratio_str.strip()
            
            # Two-step replacement
            ratio_repl_normed, diff_starts = replace_with_starts(ratio_repl, ratio, ratio_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                ratio_repl,
                ratio_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, ratio)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(ratio),
                original=ratio,
                modified=ratio_str,
                tag="FRACTION"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def _norm_tag_fraction_type_2(self, input_str: str):
        """
        Normalize case 1/3 muỗng

        NOTE: [LongNH]
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        ratio_list = []
        start = 0
        
        # First pass: search for fractions with suffix units
        while True:
            ratio = p.search(input_str, start)
            if not ratio:
                break
            ratio_list.append((ratio.group(1).rstrip(",."), ratio.group(), ratio.span()[0]))
            addon_start = ratio.span()[1] - 1
            
            # Nested search for additional fractions
            while True:
                remaining_str = input_str[addon_start:]
                ratio_addon = p_addons.search(remaining_str)
                if not ratio_addon:
                    break
                abs_start = addon_start + ratio_addon.span()[0]
                ratio_list.append((ratio_addon.group(2).rstrip(",."), ratio_addon.group(), abs_start))
                addon_start = addon_start + ratio_addon.span()[1] - 1
            
            start = addon_start
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for ratio, ratio_repl, start_pos in ratio_list[::-1]:
            # Convert to spoken text
            first_num  = ratio.replace(":", "/").split('/')[0].replace(",", "")
            second_num = ratio.replace(":", "/").split('/')[1].replace(",", "")
            ratio_str = num2words_float(first_num) + ' phần ' + num2words_float(second_num)
            ratio_str = ratio_str.strip()
            
            # Two-step replacement
            ratio_repl_normed, diff_starts = replace_with_starts(ratio_repl, ratio, ratio_str)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                ratio_repl,
                ratio_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, ratio)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(ratio),
                original=ratio,
                modified=ratio_str,
                tag="FRACTION"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def _norm_tag_fraction_type_3(self, input_str: str):
        """
        Normalize case trường hợp/100.000 dân

        NOTE: Keep original, not touch
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        p = re.compile(r"\s[AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴA-Zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z]+\/")
        
        # Phase 1: Collect all matches with their positions
        ratio_list = []
        start = 0
        while True:
            ratio = p.search(input_str, start)
            if not ratio:
                break
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            ratio_text = x.split()[-1]
            ratio_list.append((ratio_text, ratio.group(), ratio.span()[0]))
            start = ratio.span()[1]
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for ratio_text, ratio_repl, start_pos in ratio_list[::-1]:
            # Convert to spoken text
            ratio_str = ratio_text.replace('/', ' trên ').strip()
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                ratio_repl,
                ratio_repl.replace(ratio_text, ratio_str),
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, ratio_repl)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s + ratio_repl.find(ratio_text),
                end_index=s + ratio_repl.find(ratio_text) + len(ratio_text),
                original=ratio_text,
                modified=ratio_str,
                tag="FRACTION"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]
    
    def normalize_legal_id(self, input_str: str):
        """
        Normalize case Điều 48 Nghị định 110/2013/NĐ-CP

        NOTE: Keep original, not touch
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions  
        legal_id_list = []
        start = 0
        
        # First pass: search for legal IDs with prefix
        while True:
            ratio = p.search(input_str, start)
            if not ratio:
                break
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            legal_id = x.split()[-1]
            legal_id_list.append((legal_id, ratio.group(), ratio.span()[0]))
            addon_start = ratio.span()[1]
            
            # Nested search for additional legal IDs
            while True:
                remaining_str = input_str[addon_start:]
                ratio_addon = p_addons.search(remaining_str)
                if not ratio_addon:
                    break
                x = ratio_addon.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
                legal_id = x.split()[-1]
                abs_start = addon_start + ratio_addon.span()[0]
                legal_id_list.append((legal_id, ratio_addon.group(), abs_start))
                addon_start = addon_start + ratio_addon.span()[1] - 1
            
            start = addon_start
        
        # Clean up ratio list
        legal_id_list_cleaned = []
        for legal_id, legal_id_repl, start_pos in legal_id_list:
            cleaned_id = legal_id
            for character in '. , ( ) ;'.split():
                cleaned_id = cleaned_id.replace(character, '')
            legal_id_list_cleaned.append((cleaned_id, legal_id_repl, start_pos))
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions  
        for legal_id, legal_id_repl, start_pos in legal_id_list_cleaned[::-1]:
            # Convert to spoken text
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
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                legal_id,
                legal_id_str,
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, legal_id)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(legal_id),
                original=legal_id,
                modified=legal_id_str,
                tag="LEGAL_ID"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_address(self, input_str: str):
        """
        Normalize case ngõ 12A/124 
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        # Use updated patterns from the new version
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
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        p_addons = re.compile(
            r"^\s*(" + self.RAW_LINKING_WORDS_REGEX + r"|"
            + self.RAW_ENDING_PUNCTUATIONS_REGEX + r")+\s*"
            + p_text
            + self.ENDING_PUNCTUATIONS_REGEX
        )
        
        # Phase 1: Collect all matches with their positions  
        address_list = []
        temp_str = input_str
        offset = 0
        
        # First pass: search for addresses with prefix
        while True:
            address = p.search(temp_str)
            if not address:
                break
            x = address.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            address_part = x.split()[-1]
            address_list.append((address_part, address.group(), offset + address.span()[0]))
            addon_offset = offset + address.span()[1] - 1
            
            # Nested search for additional addresses
            addon_str = temp_str[address.span()[1] - 1:]
            while True:
                address_addon = p_addons.search(addon_str)
                if not address_addon:
                    break
                x = address_addon.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
                address_part = x.split()[-1]
                address_list.append((address_part, address_addon.group(), addon_offset + address_addon.span()[0]))
                addon_str = addon_str[address_addon.span()[1] - 1:]
                addon_offset += address_addon.span()[1] - 1
            
            temp_str = temp_str[address.span()[1] - 1:]
            offset += address.span()[1] - 1
        
        # Clean up address list
        address_list_cleaned = []
        for address_part, address_repl, start_pos in address_list:
            cleaned_addr = address_part
            for character in '. , ( ) ;'.split():
                cleaned_addr = cleaned_addr.replace(character, '')
            address_list_cleaned.append((cleaned_addr, address_repl, start_pos))
        
        differences: list[Difference] = []
        
        # Sort by length (longest first) for replacement
        address_list_cleaned = sorted(address_list_cleaned, key=lambda x: len(x[0]), reverse=True)
        
        # Phase 2: Process right-to-left to preserve positions  
        for address_part, address_repl, start_pos in address_list_cleaned[::-1]:
            # Convert to spoken text using alpha_num2words like original
            address_spoken = " trên ".join([alpha_num2words(a, num_mode="full") for a in address_part.split("/")])
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                address_part,
                address_spoken,
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, address_part)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(address_part),
                original=address_part,
                modified=address_spoken,
                tag="ADDRESS"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]
    
    def normalize_money_number(self, input_str: str):
        """
        NOTE: Keep original, not touch
        """
        ori_input_str = input_str
        
        p = re.compile(
            r"("
            r"(-?[\d]+"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*"
            r"[\.|\,]*[\d]*)"
            r"([Đđ]|\s*[Vv][Nn][DdĐđ])"
            r"[\/)\]}]*"
            r")\b"
        )
        
        # Phase 1: Collect all matches with their positions
        money_list = []
        start = 0
        
        while True:
            number = p.search(input_str, start)
            if not number:
                break
            x = number.group(2)
            prefix = ""
            if x[0] == "-":
                x = x[1:]
                prefix = "âm "
            if x[-1] in ",.":
                x = x[:-1]
            unit = number.group(3).strip()
            money_list.append((prefix, x, unit, number.group(), number.span()[0]))
            start = number.span()[1]
        
        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for prefix, number, unit, number_repl, start_pos in money_list[::-1]:
            # Convert unit to spoken text
            unit_str = "đồng" if unit.lower() == "đ" else "việt nam đồng"
            
            # Convert number to spoken text
            number_str = prefix + money2words(number)
            number_str = number_str.strip()
            
            # Handle suffix patterns
            tmp = number_repl.rstrip("/)]}")
            suffix = "trên " if tmp != number_repl and "/" in number_repl[-len(number_repl) + len(tmp):] else ""
            
            # Create final replacement
            final_replacement = number_str + " " + unit_str
            if suffix:
                final_replacement += " " + suffix
            # final_replacement = final_replacement.strip()
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                number_repl,
                final_replacement,
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, number_repl)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(number_repl),
                original=number_repl,
                modified=final_replacement,
                tag="MONEY"
            ) for s in diff_starts]
        
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    def normalize_number(self, input_str: str):
        """
        Normalize numbers
        """
        # First normalization pass
        text1, differences1 = self._norm_number_type_1(input_str)
        
        # Second normalization pass
        text2, differences2 = self._norm_number_type_2(text1)
        
        # Merge differences from both passes
        merged_differences = merge_differences(differences1, differences2)
        
        return text2, merged_differences

    def _norm_number_type_1(self, input_str: str):
        """
        Normalize number
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
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
                prefix = "âm "
            if x[-1] in ",.":
                x = x[:-1]
            number_list.append((prefix, x, number.group(), number.span()[0]))
            start = number.span()[1]
        
        differences: list[Difference] = []

        # Phase 2: Process right-to-left to preserve positions
        for prefix, number, number_repl, start_pos in number_list[::-1]:
            # Convert to spoken text
            number_str = prefix + num2words_float(number)
            number_str = number_str.strip()
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                number_repl,
                number_repl.replace(number, number_str),
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, number_repl)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s + number_repl.find(number),
                end_index=s + number_repl.find(number) + len(number),
                original=number,
                modified=number_str,
                tag="NUMBER_INT_FLOAT"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def _norm_number_type_2(self, input_str: str):
        """
        Normalize integer numbers
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        p = re.compile(r"(-?[\d]+)")
        
        # Phase 1: Collect all matches with their positions
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
                prefix = "âm "
            number_list.append((prefix, x, number.group(), number.span()[0]))
            start = number.span()[1]
        
        differences: list[Difference] = []

        # Phase 2: Process right-to-left to preserve positions
        for prefix, number, number_repl, start_pos in number_list[::-1]:
            # Convert to spoken text
            number_str = prefix + num2words_integer(str(int(number)))
            number_str = number_str.strip()
            
            # Replace in input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                number_repl,
                number_str,
                start_pos
            )
            
            # Verify replacement
            assert_match_with_starts(diff_starts, input_str, number_repl)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(number_repl),
                original=number_repl,
                modified=number_str,
                tag="NUMBER_INT"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def normalize_tag_measure(self, input_str: str):
        """
        Normalize unit names and number + unit names. i.e.
        'kg' --> 'ki lô gam', '1000mAh' --> 'một nghìn mi li am pe'
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '

        def is_number_str(word: str):
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

        # units_regex = self.RAW_UNITS_REGEX + "|".join(["\\" + k for k in UNITS_DICT.keys() if k.startswith("$")])
        units_regex = (
            "|".join(["\\" + k if k.startswith("$") else k for k in UNITS_DICT.keys()])
            + "|".join(["\\" + k.strip() if k.startswith("$") else k.strip() for k in UNITS_DICT.keys()])
        )
        all_differences: list[Difference] = []

        # Use only UNITS_DICT to avoid duplicates
        p = re.compile(
            r"(\b|[0-9])"
            + r"(" + units_regex + r")\b"
        )
        
        # Collect all unit matches without modifying string
        start = 0
        while True:
            number = p.search(input_str, start)
            if not number:
                break
            last_word = input_str[:number.span()[0] + len(number.group(1))]
            last_word = last_word.split()[-1].split("-")[-1] if last_word != "" else ""
            
            if is_number_str(last_word):
                unit = number.group(2).strip()
                unit_match = number.group()
                
                if unit in UNITS_DICT:
                    unit_str = UNITS_DICT[unit]
                elif unit + " " in UNITS_DICT:
                    unit_str = UNITS_DICT[unit + " "]
                else:
                    continue  # Skip units not in UNITS_DICT
                unit_str = unit_str.strip()
                
                # Find unit position within the match
                unit_pos_in_match = unit_match.find(unit)
                unit_start = number.span()[0] + unit_pos_in_match
                unit_end = unit_start + len(unit)
                
                all_differences.append(Difference(
                    start_index=unit_start,  # Will be adjusted by shift_differences
                    end_index=unit_end,
                    original=unit,
                    modified=unit_str,
                    tag="MEASURE"
                ))
                
            start = number.span()[1]

        # Sort by position for right-to-left processing
        all_differences.sort(key=lambda d: d.start_index, reverse=True)
        
        # Apply replacements to create normalized text
        for diff in all_differences:
            replacement = " " + diff.modified + " "
            start_pos = diff.start_index
            end_pos = diff.end_index
            input_str = input_str[:start_pos] + replacement + input_str[end_pos:]

        # Adjust for padding
        shift_differences(all_differences, -1)
        assert_differences(ori_input_str, all_differences)

        return input_str.strip(), all_differences[::-1]

    def normalize_number_range(self, input_str: str):
        """
        Must run before unit and measurement and number
        """
        # Define the child functions in order
        number_range_functions = [
            self._norm_number_range_type_1,
            self._norm_number_range_type_2,
            # Note: _norm_number_range_type_3 is commented out in original
        ]
        
        # Process each function and merge differences
        current_text = input_str
        all_differences = []
        
        for number_range_func in number_range_functions:
            normalized_text, current_differences = number_range_func(current_text)
            all_differences = merge_differences(all_differences, current_differences)
            current_text = normalized_text
        
        return current_text, all_differences

    def _norm_number_range_type_1(self, input_str: str):
        """
        Normalize number ranges

        NOTE. LongNH

        Has not consider version range
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        number_range_list = []
        start = 0
        
        # First pass: search for number ranges with prefix
        while True:
            number_range = p.search(input_str, start)
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
                number_range.group(),  # [ALL]
                number_range.span()[0]
            ))
            addon_start = number_range.span()[1] - 1

            # Nested search for additional number ranges
            while True:
                remaining_str = input_str[addon_start:]
                number_range_addon = p_addons.search(remaining_str)
                if not number_range_addon:
                    break
                abs_start = addon_start + number_range_addon.span()[0]
                number_range_list.append((
                    is_version,
                    number_range_addon.group(2), # number
                    number_range_addon.group(3),
                    number_range_addon.group(4), # unit
                    number_range_addon.group(5), # đến
                    number_range_addon.group(6), # number
                    number_range_addon.group(7),
                    number_range_addon.group(8), # unit
                    number_range_addon.group(),  # [ALL]
                    abs_start
                ))
                addon_start = addon_start + number_range_addon.span()[1] - 1

            start = addon_start
        
        differences: list[Difference] = []

        # Phase 2: Process right-to-left to preserve positions
        for parts in number_range_list[::-1]:
            (is_version, start_num, _, start_unit, _,
             end_num, _, end_unit, num_range_repl, start_pos) = parts
            origin = "".join([p for p in parts[1:-2] if p is not None])

            # Convert to spoken text
            if is_version is False:
                start_num_str = num2words_mixed(start_num).strip()
                end_num_str = num2words_mixed(end_num).strip()
            else:
                start_num_str = version2words(start_num).strip()
                end_num_str = version2words(end_num).strip()
            
            start_unit_str = UNITS_DICT[start_unit].strip() if start_unit is not None else None
            end_unit_str = UNITS_DICT[end_unit].strip() if end_unit is not None else None

            parts = [start_num_str, start_unit_str, "đến", end_num_str, end_unit_str]
            norm = " ".join([p for p in parts if p is not None])
            norm = norm.strip()
            
            # Two-step replacement
            num_range_repl_normed, diff_starts = replace_with_starts(num_range_repl, origin, norm)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                num_range_repl,
                num_range_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, origin)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(origin),
                original=origin,
                modified=norm,
                tag="NUMBER_RANGE"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

    def _norm_number_range_type_2(self, input_str: str):
        """
        NOTE. LongNH
        """
        ori_input_str = input_str
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
        
        # Phase 1: Collect all matches with their positions
        number_range_list = []
        tmp_unit_dict = {"m": "mét", **UNITS_DICT}
        start = 0
        
        while True:
            number_range = p.search(input_str, start)
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
                number_range.group(),  # [ALL]
                number_range.span()[0]
            ))
            start = number_range.span()[1] - 1
        
        differences: list[Difference] = []

        # Phase 2: Process right-to-left to preserve positions
        for parts in number_range_list[::-1]:
            (start_num, _, start_unit, _,
             end_num, _, end_unit, num_range_repl, start_pos) = parts
            origin = "".join([p for p in parts[:-2] if p is not None])

            # Convert to spoken text
            start_num_str = num2words_mixed(start_num).strip()
            end_num_str = num2words_mixed(end_num).strip()

            if start_unit is None:
                start_unit_str = None
            elif start_unit in tmp_unit_dict:
                start_unit_str = tmp_unit_dict[start_unit].strip()
            else:
                start_unit_str = start_unit

            if end_unit is None:
                end_unit_str = None
            elif end_unit in tmp_unit_dict:
                end_unit_str = tmp_unit_dict[end_unit].strip()
            else:
                end_unit_str = end_unit

            parts = [start_num_str, start_unit_str, "đến", end_num_str, end_unit_str]
            norm = " ".join([p for p in parts if p is not None and p.strip() != ""])
            norm = norm.strip()
            
            # Two-step replacement
            num_range_repl_normed, diff_starts = replace_with_starts(num_range_repl, origin, norm)
            assert len(diff_starts) == 1
            shift = diff_starts[0]
            
            # Replace in full input string
            tmp_str, diff_starts = replace_with_starts(
                input_str,
                num_range_repl,
                num_range_repl_normed,
                start_pos
            )
            
            # Adjust positions and verify
            diff_starts = shift_starts(diff_starts, shift)
            assert_match_with_starts(diff_starts, input_str, origin)
            
            # Update string and track differences
            input_str = tmp_str
            differences += [Difference(
                start_index=s,
                end_index=s + len(origin),
                original=origin,
                modified=norm,
                tag="NUMBER_RANGE"
            ) for s in diff_starts]
        
        # Adjust for padding
        shift_differences(differences, -1)
        assert_differences(ori_input_str, differences)

        return input_str.strip(), differences[::-1]

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
        Normalize rating expressions (replace star symbols with "sao")
        """
        ori_input_str = input_str
        input_str = ' ' + input_str + ' '
        
        # Phase 1: Collect all star symbols in rating contexts
        p = re.compile(r"(đánh giá|rate)\s+[0-9]+[*|⭐|★].")
        star_list = []
        start = 0
        
        while True:
            rate_match = p.search(input_str, start)
            if not rate_match:
                break
            
            term = rate_match.group()
            star_match = re.search(r"[*|⭐|★]", term)
            if star_match:
                # Calculate absolute position of the star symbol
                star_pos_in_term = star_match.start()
                absolute_star_pos = rate_match.start() + star_pos_in_term
                star_symbol = star_match.group()
                
                star_list.append({
                    'original': star_symbol,
                    'replacement': 'sao',
                    'start': absolute_star_pos,
                    'end': absolute_star_pos + len(star_symbol)
                })
            
            start = rate_match.end()
        
        differences = []
        
        # Phase 2: Process right-to-left to preserve positions
        star_list.sort(key=lambda x: x['start'], reverse=True)
        
        for star_info in star_list:
            # Replace in input string
            replacement = ' ' + star_info['replacement'] + ' '
            input_str = input_str[:star_info['start']].rstrip() + replacement + input_str[star_info['end']:].lstrip()
            
            # Track difference (adjust for padding)
            differences.append(Difference(
                start_index=star_info['start'] - 1,  # Adjust for added space at start
                end_index=star_info['end'] - 1,
                original=star_info['original'],
                modified=star_info['replacement'],
                tag="RATE"
            ))
        
        # Validate results
        assert_differences(ori_input_str, differences)
        
        return input_str.strip(), differences[::-1]

    # def normalize_math_characters(self, input_str: str):
    #     """
    #     Normalize math characters
    #     """
    #     input_str = ' ' + input_str + ' '
    #     input_str = replace_math_characters(input_str)

    #     return input_str.strip()
    
    def normalize_full_upper_word(self, input_str: str):
        ori_input_str = input_str
        EXCLUDES = ["II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XX", "XIX", "XXI"]
        INCLUDES = ["ABC", "ABCD", "XYZ"]
        p = re.compile(r"\b[A-Z]{2,}\b")
        
        # Phase 1: Collect all matches
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

        differences: list[Difference] = []
        
        # Phase 2: Process right-to-left to preserve positions
        for word, i_start, i_end in upper_words[::-1]:
            # Convert to spoken text using alphabet dictionary
            spoken_word = " ".join([ALPHABET_DICT[ch] for ch in word])
            
            # Replace in input string
            input_str = input_str[:i_start] + spoken_word + input_str[i_end:]
            
            # Track difference
            differences.append(Difference(
                start_index=i_start,
                end_index=i_end,
                original=word,
                modified=spoken_word,
                tag="FULL_UPPER_WORD"
            ))
        
        # Validate results
        assert_differences(ori_input_str, differences)
        
        return input_str, differences[::-1]
    
    def _find_correct_position(self, text: str, target_diff: Difference, processed_diffs: list[Difference]) -> int:
        """
        Find the correct position for a difference in the root text, handling duplicates intelligently.
        
        For duplicate strings, this tries to find the occurrence that hasn't been used by previous differences
        and makes the most sense based on the expected position.
        """
        target_string = target_diff.original
        
        # Find all occurrences of the target string in the root text
        occurrences = []
        start = 0
        while True:
            pos = text.find(target_string, start)
            if pos == -1:
                break
            occurrences.append(pos)
            start = pos + 1
        
        if not occurrences:
            return -1
        
        # If there's only one occurrence, use it
        if len(occurrences) == 1:
            return occurrences[0]
        
        # Get positions already used by processed differences
        used_ranges = set()
        for diff in processed_diffs:
            for pos in range(diff.root_start_index, diff.root_end_index):
                used_ranges.add(pos)
        
        # Find the first occurrence that doesn't overlap with used positions
        for pos in occurrences:
            end_pos = pos + len(target_string)
            overlap = False
            for check_pos in range(pos, end_pos):
                if check_pos in used_ranges:
                    overlap = True
                    break
            if not overlap:
                return pos
        
        # If all positions overlap (shouldn't happen in valid text), return the first
        return occurrences[0]

    def normalize(self, text: str):
        # Define the normalization pipeline in order
        normalization_pipeline = [
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
            
        normalization_pipeline.extend([
            self.normalize_tag_measure,  # Third call
            # self.normalize_single_stand_letter, # skip
            self.normalize_full_upper_word,
        ])

        # Initialize for Level 2 Merge
        current_text = text
        all_differences = []
        
        # Process each normalization function in the pipeline
        for i, norm_func in enumerate(normalization_pipeline):
            # Get normalized text and differences from current function
            normalized_text, current_differences = norm_func(current_text)
            
            
            # Merge current differences with accumulated differences
            all_differences = merge_differences(all_differences, current_differences)
            
            # Update text for next iteration
            current_text = normalized_text
        
        # print(current_text)
        
        # Ensure all_differences are correct
        for i, diff in enumerate(all_differences):
            actual_text = text[diff.root_start_index : diff.root_end_index]
            if actual_text != diff.original:
                # Try to find the correct position of the original text in the root text
                # For duplicate strings, we need to find the right occurrence
                correct_pos = self._find_correct_position(text, diff, all_differences[:i])
                if correct_pos != -1:
                    # Update the difference with correct root positions
                    diff.root_start_index = correct_pos
                    diff.root_end_index = correct_pos + len(diff.original)
                    actual_text = text[diff.root_start_index : diff.root_end_index]
            
            assert actual_text == diff.original, (
                f"Position mapping error: text[{diff.root_start_index}:{diff.root_end_index}] = "
                f"'{actual_text}' != '{diff.original}'"
            )
            
            if diff.root_start_index < 0:
                diff.root_start_index = len(text) + diff.root_start_index
            if diff.root_end_index < 0:
                diff.root_end_index = len(text) + diff.root_end_index
                
            assert actual_text == diff.original, (
                f"Position mapping error: text[{diff.root_start_index}:{diff.root_end_index}] = "
                f"'{actual_text}' != '{diff.original}'"
            )

        all_differences = sorted(all_differences, key=lambda x: (x.root_start_index, x.root_end_index))
        
        return current_text.strip(), all_differences
        
