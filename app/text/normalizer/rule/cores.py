import re
from typing import Literal
from vietnam_number import n2w_single
from .utils.characters import ALPHABET_DICT, NUMBER_DICT, SYMBOL_DICT, WORD_BREAK_DICT


def _normalize_vietnamese(text):
    """Remove Vietnamese diacritics for vocabulary matching"""
    # Vietnamese diacritic mapping
    diacritic_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd'
    }
    
    normalized = ''
    for char in text.lower():
        normalized += diacritic_map.get(char, char) # type: ignore
    return normalized


def _load_vocab(path: str):
    vocab = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            word = line.split()[0]
            vocab.append(word)
            
            # For Vietnamese vocabulary, also add normalized form
            if 'vi.tsv' in path:
                normalized = _normalize_vietnamese(word)
                if normalized != word:  # Only add if different
                    vocab.append(normalized)
    return set(vocab)

VIETNAMESE_VOCAB = _load_vocab("data/lexicon_vi.tsv")
ENGLISH_VOCAB = _load_vocab("data/lexicon_en.tsv")


def break_word(text: str):
    assert " " not in text
    text = text.lower()
    
    if len(text) == 0:
        return []
    
    # Check WORD_BREAK_DICT first
    if text in WORD_BREAK_DICT:
        return WORD_BREAK_DICT[text].split()
    
    # Check if entire text is a single word
    if text in VIETNAMESE_VOCAB or text in ENGLISH_VOCAB:
        return [text]
    
    # Use greedy approach with lookahead
    start = 0
    words = []
    
    while start < len(text):
        sub_text = text[start:]
        best_word = None
        best_len = 0
        
        # Check WORD_BREAK_DICT first for substrings
        for end in range(len(sub_text), 0, -1):
            candidate = sub_text[:end]
            if candidate in WORD_BREAK_DICT:
                if end > best_len:
                    best_word = candidate
                    best_len = end
                break  # Take first (longest) match from WORD_BREAK_DICT
        
        # If no match in WORD_BREAK_DICT, try vocabularies
        if not best_word:
            for end in range(len(sub_text), 0, -1):
                candidate = sub_text[:end]
                if candidate in VIETNAMESE_VOCAB or candidate in ENGLISH_VOCAB:
                    if end > best_len:
                        best_word = candidate
                        best_len = end
                    break  # Take first (longest) match
        
        if best_word:
            if best_word in WORD_BREAK_DICT:
                # Split the word using WORD_BREAK_DICT
                words.extend(WORD_BREAK_DICT[best_word].split())
            else:
                words.append(best_word)
            start += best_len
        else:
            # Check if we have a sequence of digits
            remaining = sub_text
            if remaining and remaining[0].isdigit():
                # Collect consecutive digits
                collected = ""
                pos = 0
                while pos < len(remaining) and remaining[pos].isdigit():
                    collected += remaining[pos]
                    pos += 1
                
                if collected:
                    # words.append(collected)
                    words.append(phone2words(collected).strip())
                    start += pos
                    continue
            
            # No word found, take single character
            words.append(sub_text[0])
            start += 1
    
    return words


def _split_text(text: str, seps: str):
    results = []
    ret = ""
    for ch in text:
        if ch not in seps:
            ret += ch
        else:
            results.append(ret)
            results.append(ch)
            ret = ""
    if len(ret) > 0:
        results.append(ret)
    return results


def _all_equal(values: list, val):
    for v in values:
        if v != val:
            return False
    return True


def replace_math_characters(input_str):
    # input_str = input_str.replace("²", " bình phương ")
    input_str = input_str.replace("π", " pi ")
    return input_str


def _convert_group_of_three(num):
    """Convert a group of three digits to words"""
    if num == 0:
        return ''
    
    ones = ['', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín']
    result = []
    
    # Hundreds place
    hundreds = num // 100
    if hundreds > 0:
        result.append(ones[hundreds])
        result.append('trăm')
    
    # Tens and ones place
    remainder = num % 100
    
    if remainder > 0:
        if remainder < 10:
            # Special case: 0X (e.g., 101, 202)
            if hundreds > 0:
                result.append('linh')
            result.append(ones[remainder])
        elif remainder == 10:
            result.append('mười')
        elif remainder < 20:
            # 11-19
            result.append('mười')
            ones_digit = remainder % 10
            if ones_digit == 5:
                result.append('lăm')
            else:
                result.append(ones[ones_digit])
        else:
            # 20-99
            tens = remainder // 10
            ones_digit = remainder % 10
            
            if tens == 1:
                result.append('mười')
            else:
                result.append(ones[tens])
                result.append('mươi')
            
            if ones_digit > 0:
                if ones_digit == 1 and tens > 1:
                    result.append('mốt')
                elif ones_digit == 5 and tens > 0:
                    result.append('lăm')
                else:
                    result.append(ones[ones_digit])
    
    return ' '.join(result)


def _insert_linh_for_gaps(all_groups, result_parts, original_num):
    """
    Insert 'linh' or 'không trăm' based on the last three digits:
    - 'linh': when last 3 digits are "00x" (single digit: 001, 005, 009)
    - 'không trăm': when last 3 digits are "0xx" (tens/ones but no hundreds: 025, 045, 099)
    
    Examples:
    - 1001 → "một nghìn linh một" (last 3 digits: 001)
    - 2025 → "hai nghìn không trăm hai mươi lăm" (last 3 digits: 025)
    - 1000022 → "một triệu linh hai mươi hai" (last 3 digits: 022, but large gap)
    """
    
    # Get the last 3 digits to determine the pattern
    last_three_digits = original_num % 1000
    
    # If number ends in 000, no linh/không trăm needed
    if last_three_digits == 0:
        return ' '.join(result_parts)
    
    # Only handle cases where last non-zero group < 100 and we have multiple groups
    last_group = all_groups[-1]
    if last_group >= 100 or len(result_parts) < 2:
        return ' '.join(result_parts)
    
    # Determine if there are gaps between non-zero groups
    non_zero_positions = [i for i, g in enumerate(all_groups) if g > 0]
    if len(non_zero_positions) < 2:
        return ' '.join(result_parts)
    
    highest_pos = non_zero_positions[0]
    lowest_pos = non_zero_positions[-1]
    
    # Check for major gaps (missing groups between highest and lowest)
    has_major_gaps = lowest_pos - highest_pos > len(non_zero_positions) - 1
    
    # Rule application:
    if last_three_digits < 10:  # 00x pattern (001, 005, 009)
        # Always use "không trăm linh" for single digits 
        return ' '.join(result_parts[:-1]) + ' không trăm linh ' + result_parts[-1]
    elif last_three_digits < 100:  # 0xx pattern (025, 045, 099)  
        if has_major_gaps:
            # Large gaps (millions/billions) still use "linh" even for 0xx
            return ' '.join(result_parts[:-1]) + ' linh ' + result_parts[-1]
        else:
            # Adjacent groups use "không trăm" for 0xx pattern
            return ' '.join(result_parts[:-1]) + ' không trăm ' + result_parts[-1]
    
    return ' '.join(result_parts)


def num2words_integer(num, remove_sep=False):
    """
    Convert an integer to Vietnamese words with comprehensive edge case handling.
    
    Args:
        num: String or integer representation of a number
    
    Returns:
        Vietnamese text representation of the number
    """
    # Handle trailing punctuation
    if isinstance(num, str):
        if num and num[-1] in '! " " \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ' ''.split():
            num = num[:-1]
        if remove_sep:
            num = num.replace(",", "").replace(".", "")
        if num == "0" * len(num):
            return " ".join(["không" for _ in range(len(num))])
        num = int(num)
    
    if num == 0:
        return 'không'
    
    if num < 0:
        return 'âm ' + num2words_integer(-num)
    
    # Split number into groups of 3 digits from right to left
    group_names = ['', 'nghìn', 'triệu', 'tỷ', 'nghìn tỷ', 'triệu tỷ']
    all_groups = []  # Track all groups including zeros for gap detection
    
    temp_num = num
    group_index = 0
    
    while temp_num > 0 and group_index < len(group_names):
        group = temp_num % 1000
        all_groups.append(group)
        temp_num //= 1000
        group_index += 1
    
    # Reverse to get correct order (highest to lowest)
    all_groups.reverse()
    
    # Convert non-zero groups to text
    result_parts = []
    for i, group in enumerate(all_groups):
        if group > 0:
            group_text = _convert_group_of_three(group)
            
            # Add "không trăm" prefix for middle groups with missing hundreds (< 100)
            if group < 100 and i > 0 and i < len(all_groups) - 1:
                if group < 10:
                    group_text = 'không trăm linh ' + group_text
                else:
                    group_text = 'không trăm ' + group_text
            
            group_name = group_names[len(all_groups) - 1 - i]
            if group_name:
                group_text += ' ' + group_name
            result_parts.append(group_text)
    
    # Handle "linh" insertion for gaps
    if len(result_parts) >= 2:
        result = _insert_linh_for_gaps(all_groups, result_parts, num)
    else:
        result = ' '.join(result_parts)
    
    # Clean up round numbers (remove redundant "không trăm")
    if num >= 1000 and num % 1000 == 0:
        result = result.replace('không trăm nghìn', '')
        result = result.replace('không trăm triệu', '')
        if result.endswith('không trăm'):
            result = result[:-10].strip()
    
    return result.strip()


def num2word_fraction_part(fraction: str):
    fraction = fraction.strip()
    while len(fraction) > 0 and fraction[-1] == "0":
        fraction = fraction[:-1]
    
    if len(fraction) == 0:
        return " không "
    
    zeros = ""
    while len(fraction) > 0 and fraction[0] == "0":
        zeros += fraction[0]
        fraction = fraction[1:]
        
    if len(fraction) == 0:
        return " không "

    if len(fraction) <= 3:
        return phone2words(zeros) + num2words_integer(fraction)
    return phone2words(zeros + fraction)


def num2words_float(number, prefer_int=True, prefer_comma_as_int_sep=False):
    """
    Convert a float number to Vietnamese words.
    
    Args:
        number: String representation of the number
        prefer_int: When True, numbers like "12.345" are treated as integers (12345)
                   When False, they are treated as decimals (12.345 → "mười hai phẩy ba bốn lăm")
    
    Returns:
        Vietnamese text representation of the number
    """
    if number[-1] in '! " " \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ' ''.split():
        number = number[:-1]
    number = number.strip()

    # nếu có 1 dấu phẩy thì đọc là phẩy luôn
    # nếu có 1 dấu chấm thì
    # - đọc là phẩy nếu sau fraction khác 3 chữ số
    # - nếu fraction là 3 chữ số
    #   - nếu decimal > 3 chữ số hoặc decimal là 0 thì đọc là phẩy
    #   - nếu prefer_int=True thì bỏ qua chấm (đọc như integer)
    #   - nếu prefer_int=False thì đọc là phẩy
    # nếu toàn phẩy hoặc toàn chấm
    # - bỏ qua hết
    # nếu vừa chấm vừa phẩy
    # - nếu chỉ có 1 dấu phẩy hoặc chỉ có 1 dấu chấm ở cuối thì đọc là phẩy
    # - nếu thứ tự hỗn loạn thì bỏ qua hết
    
    result = ""
    seps = [ch for ch in number if ch in ",."]
    if len(seps) == 0:
        result = num2words_integer(number)
    elif len(seps) == 1:
        decimal, fraction = number.split(seps[0])
        if seps[0] == ",":
            if prefer_comma_as_int_sep is False:
                # Comma always means decimal separator
                result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
            else:
                if len(fraction) != 3:
                    # Not 3 digits after comma → always decimal
                    result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
                else:
                    # 3 digits after comma → depends on context and prefer_int
                    if len(decimal) > 3 or decimal == "0":
                        # Large number or 0,xxx → always decimal
                        result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
                    elif prefer_int:
                        # prefer_int=True → treat as integer (remove comma)
                        result = num2words_integer(number.replace(",", ""))
                    else:
                        # prefer_int=False → treat as decimal
                        result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
        elif seps[0] == ".":
            if len(fraction) != 3:
                # Not 3 digits after dot → always decimal
                result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
            else:
                # 3 digits after dot → depends on context and prefer_int
                if len(decimal) > 3 or decimal == "0":
                    # Large number or 0.xxx → always decimal
                    result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
                elif prefer_int:
                    # prefer_int=True → treat as integer (remove dot)
                    result = num2words_integer(number.replace(".", ""))
                else:
                    # prefer_int=False → treat as decimal
                    result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
            
    elif _all_equal(seps, seps[0]):
        # All separators are the same → treat as thousands separators
        result = num2words_integer(number.replace(seps[0], ""))
    
    elif _all_equal(seps[:-1], seps[0]) and seps[-1] != seps[0]:
        # Mixed separators with last one different → decimal
        decimal, fraction = number.split(seps[-1])
        decimal = decimal.replace(seps[0], "")
        result = num2words_integer(decimal) + " phẩy " + num2word_fraction_part(fraction)
    else:
        if prefer_int:
            # Complex case → treat as integer
            number = number.replace(",", "").replace(".", "")
            result = num2words_integer(number)
        else:
            # Keep last sep as pivot
            sep_pos = number.rfind(seps[-1])
            decimal, fraction = number[:sep_pos], number[sep_pos + 1:]
            result = num2words_integer(decimal, remove_sep=True) + " phẩy " + num2word_fraction_part(fraction)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def num2words_mixed(input_str: str):
    input_str = input_str.strip()
    neg = ""
    if input_str[0] == "-":
        neg = "âm"
        input_str = input_str[1:]
    parts = _split_text(input_str, "^/")
    mapping = {"^": "mũ", "/": "phần"}
    parts = [num2words_float(p) if p not in mapping else mapping[p] for p in parts]
    return " " + neg + " " + " ".join(parts) + " "


def date_dmy2words(date, add_day_prefix=True, support_day_lt10=True) -> str:
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]
    if "/" in date:
        day, month, year = date.split("/") 
    elif "." in date:
        day, month, year = date.split(".") 
    elif "-" in date:
        day, month, year = date.split("-") 
    else:
        raise NotImplementedError()

    date_str = " " + num2words_integer(str(int(day))) + " tháng " + num2words_integer(str(int(month))) + " năm " +  num2words_integer(year)
    if int(day) < 10 and support_day_lt10:
        date_str = " mùng" + date_str 
    if add_day_prefix:
        date_str = " ngày" + date_str
    return date_str


def date_dm2words(date, add_day_prefix=True, support_day_lt10=True):
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]
    if "/" in date:
        day, month = date.split("/") 
    elif "." in date:
        day, month = date.split(".") 
    elif "-" in date:
        day, month = date.split("-") 
    else:
        raise NotImplementedError()
        
    date_str = " " + num2words_integer(str(int(day))) + " tháng " + num2words_integer(str(int(month)))
    if int(day) < 10 and support_day_lt10:
        date_str = " mùng" + date_str 
    if add_day_prefix:
        date_str = " ngày" + date_str
    return date_str


def date_my2words(date, add_month_prefix=True):
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]

    if "/" in date:
        month, year = date.split("/") 
    elif "." in date:
        month, year = date.split(".") 
    elif "-" in date:
        month, year = date.split("-")
    else:
        raise NotImplementedError()

    date_str = num2words_integer(str(int(month))) + " năm " +  num2words_integer(year)
    if add_month_prefix:
        date_str = " tháng " + date_str
    return date_str


def time2words(time_str):
    # Handle colon-separated format (H:M or H:M:S)
    if ":" in time_str:
        # Remove trailing 's' if present
        if time_str[-1] == "s":
            time_str = time_str[:-1]
        time_split = time_str.split(":")
        if len(time_split) == 2:
            hour, minute = time_split
            time_str = num2words_float(hour) + " giờ " +  num2words_float(minute) + " phút "
        elif len(time_split) == 3:
            hour, minute, second = time_split
            time_str = num2words_float(hour) + " giờ " +  num2words_float(minute) + " phút " + num2words_float(second) + " giây "
        else:
            raise NotImplementedError()
    else:
        # Handle letter-based format (XhYpZs, XgYmZs, etc.) with single regex
        # Support floats: 2.5h, 10.3p, 45.7s
        pattern = (
            r"(?:(\d+(?:\.\d+)?)[hg])?"
            r"(?:(\d+(?:\.\d+)?)[pm])?"
            r"(?:(\d+(?:\.\d+)?)s)?"
            r"(?:(\d+(?:\.\d+)?))?"
        )
        match = re.match(pattern, time_str)
        
        if not match or not any(match.groups()):
            raise NotImplementedError(time_str)
        
        hour, minute, second, last = match.groups()

        time_str = ""
        last_unit = None
        
        # Convert hour
        if hour:
            time_str += num2words_float(hour) + " giờ "
            last_unit = "phút"
        
        # Convert minute if present
        if minute:
            time_str += num2words_float(minute) + " phút "
            last_unit = "giây"
        
        # Convert second if present  
        if second:
            time_str += num2words_float(second) + " giây "
            last_unit = None
            
        if last and last_unit:
            time_str += num2words_float(last) # + " " + last_unit + " "
            
    # Clean up zero minutes
    time_str = time_str.replace("không phút", "")
    return time_str.strip()


def multiply(input_str):
    element_split = input_str.split("x")
    multiply_str_list = []
    for element in element_split:
        multiply_str_list.append(num2words_float(element.strip()))
    multiply_str = " nhân ".join(multiply_str_list)
    return multiply_str


def phone2words(number):
    if number.strip() == "":
        return ""
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
    number = number.replace(" ", "")
    return " " + n2w_single(number) + " "


def alpha_num2words(input_str: str, num_mode: Literal["single", "full", "shortest", "longest"] = "shortest", skip_symbols=False):
    parts = []
    num = ""
    for ch in input_str.strip():
        if ch in ALPHABET_DICT:
            if num != "":
                parts.append(num)
                num = ""
            parts.append(ALPHABET_DICT[ch])
        elif ch in SYMBOL_DICT:
            if skip_symbols:
                continue
            if num != "":
                parts.append(num)
                num = ""
            parts.append(SYMBOL_DICT[ch])
        elif ch in "0123456789":
            num += ch
    if num != "":
        parts.append(num)
    
    for i in range(len(parts)):
        num = parts[i]
        if num[0] in "0123456789":
            if num_mode == "single":
                num_str = " ".join([NUMBER_DICT[ch] for ch in num])
            elif num_mode == "full":
                num_str = num2words_integer(num).strip()
            else:
                full_str = num2words_integer(num).strip()
                single_str = " ".join([NUMBER_DICT[ch] for ch in num])
                if num_mode == "shortest":
                    num_str = full_str if len(full_str.split()) <= len(single_str.split()) else single_str
                elif num_mode == "longest":
                    num_str = full_str if len(full_str.split()) >= len(single_str.split()) else single_str
                else:
                    raise NotImplementedError(num_mode)
            parts[i] = num_str
    
    return " ".join(parts)
            

def money2words(number: str):
    seps = [ch for ch in number if ch in ",."]
    if len(seps) == 1 and seps[0] == ",":
        number = number.replace(",", "")
    return num2words_float(number)


def version2words(number: str):
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
        
    parts = number.split(".")
    return " chấm ".join([num2words_integer(p) for p in parts])
