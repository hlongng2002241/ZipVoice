import sys; sys.path.append(".") # fmt: skip
from zipvoice.tokenizer.fusion_tokenizer import FusionTokenizer
from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer


def main():
    l = LanguageModelTokenizer()
    t = FusionTokenizer(lang="en", token_file="data/tokens.txt", lm_tokenizer=l)
    text = "hello world, how are you today?"
    print(t.text_to_artifact(text))


if __name__ == "__main__":
    main()
