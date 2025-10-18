import re
from nltk import sent_tokenize

from .normalizer import RuleBasedTextNormalizer


def normalize(text, punctuation=False):
    normalizer = RuleBasedTextNormalizer(verbose=True)
    return normalizer.normalize(text, punctuation)


def post_processing(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.\.\s', '. ', text)
    text  = text.lower()
    return text


def load_dict_english(path):
    with open(path, "r+", encoding="utf8") as f:
        lines = f.read().splitlines()
    rs = {}
    for line in lines:
        pair = line.split("|")
        rs[pair[0]] = pair[1]
    return rs


def mapping_eng(text, my_dict):
    text = re.sub(r',\s*|\s*,\s*|\s*,', ' , ', text)
    list_sent = text.split(".")
    rs_sent = []
    for sent in list_sent:
        list_words = sent.split(' ')
        for i in range(len(list_words)):
            if list_words[i] in my_dict.keys():
                # print(key, my_dict[key])
                list_words[i] = my_dict[list_words[i]]
        rs_sent.append(" ".join(list_words))
    return ". ".join(rs_sent)


def load_list_word(path):
    with open(path, "r+", encoding ="utf8") as f:
        lines = f.read().splitlines()
    list_rs = [x for x in lines]
    return list_rs


# def filter_string(text, list_word):
#     text = re.sub(r',\s*|\s*,\s*|\s*,', ' , ', text)
#     list_sent = text.split(".")
#     remove_words = []
#     rs = []
#     for sent in list_sent:
#         list_words = sent.split(" ")    
#         rs_sent = []
#         for i in range(len(list_words)):
#             if list_words[i] in list_word or list_words[i] == ",":
#                   rs_sent.append(list_words[i])
#             else:
#                   remove_words.append(list_words[i])  
#         rs.append(" ".join(rs_sent))
#     return ". ".join(rs), remove_words


text_to_pronounce = {
    "a": "a",
    "ă": "á",
    "â": "ớ",
    "b": "bê",
    "c": "xê",
    "d": "dê",
    "đ":"đê",
    "e": "e",
    "ê": "ê",
    "g": "gờ",
    "h": "hát",
    "i": "i",
    "j": "di",
    "k": "ca",
    "l": "lờ",
    "m": "mờ",
    "n": "nờ",
    "o": "o",
    "ô": "ô",
    "ơ": "ơ",
    "p": "pê",
    "q": "quy",
    "r": "rờ",
    "s": "ét",
    "t": "tê",
    "u": "u",
    "ư": "ư",
    "v": "vê",
    "w": "vê kép",
    "x": "xê",
    "y": "y",
    "z": "dét"
}


def filter_string(text, list_word):
    text = re.sub(r',\s*|\s*,\s*|\s*,', ' , ', text)
    list_sent = text.split(".")
    remove_words = []
    rs = []
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^`{}~"
    for sent in list_sent:
        list_words = sent.split(" ")    
        rs_sent = []
        
        for word in list_words:
            if word in list_word or word in punct:
                rs_sent.append(word)
            else:
                remove_words.append(word)
                pronounced_word = ' '.join([text_to_pronounce.get(char, '') for char in word])
                if pronounced_word:
                    rs_sent.append(pronounced_word)

        rs.append(" ".join(rs_sent))

    return ". ".join(rs), remove_words


def run(text):
    # try:
        norm_text = []
        dict = load_dict_english("english_word_3_v3.txt")
        list_word = load_list_word("vn_dict.txt")
        # print(f'token: {sent_tokenize(text)}')
        for sentence in sent_tokenize(text):
            print('Before norm: ', sentence)
            sentence = normalize(sentence)
            print('After norm: ', sentence)
            norm_text.append(sentence)
        text = '. '.join(norm_text)
        print('Before processing: ', text)
        text = post_processing(text)
        print('After processing: ', text)
        text = mapping_eng(text, dict)
        print("After mapping: ", text)
        text, remove_words = filter_string(text, list_word)
        return text, remove_words
    # except:
    #    return "0001"
