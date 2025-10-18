from enum import Enum
from dataclasses import dataclass

import torch

from transformers import AutoTokenizer
from ...tagger.m30_train import AutoTaggerModel
from ...tagger.augment.base import clean_content, Pattern, calibrate_tags, convert_to_raw_content


class InferenceError(Exception):
    pass


@dataclass
class TaggerOutput:
    content: str
    patterns: list[Pattern]
    token_level_outputs: list[tuple[str, str]] | None = None
    errors: list[str] | None = None


class LabelMatching(str, Enum):
    MATCH_GOAL_TAG = "match_goal_tag"
    MATCH_ALL_TAGS = "match_all_tags"


class TaggerInference:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = None,
        label_matching: LabelMatching = LabelMatching.MATCH_ALL_TAGS,
        device="cuda",
        max_batch_size: int = 64,
    ):
        """Initialize NER inference model

        Args:
            checkpoint_path: Path to trained model checkpoint
            model_name: Base model name used for tokenizer
        """
        self.prefix = "▁"
        self.label_matching = label_matching
        self.device = device
        self.max_batch_size = max_batch_size

        if tokenizer_name is None:
            tokenizer_name = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=False)
        self.model = AutoTaggerModel.from_pretrained(model_path)
        self.model.eval().to(device)

        # Load label mappings
        self.label2id = self.model.config.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}

    @torch.inference_mode()
    def inference(self, texts: str | list[str], batch_size: int = None):
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size if batch_size is not None else self.max_batch_size

        texts = [clean_content(text) for text in texts]
        results: list[TaggerOutput] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})

            # input_ids = inputs.input_ids
            # attn_mask = inputs.attention_mask
            pred_tags_list = outputs.tags

            for index in range(len(batch)):
                pred_tags = [self.id2label[p] for p in pred_tags_list[index]]
                # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[index][: attn_mask[index].sum()])
                tokens = self.tokenizer.tokenize(batch[index])
                results.append(self.postprocess(batch[index], tokens, pred_tags[1 : -1]))

        return results

    def postprocess(self, text: str, tokens: list[str], preds: list[str]):
        assert len(tokens) == len(preds), f"{len(tokens)} != {len(preds)}"
        # print(list(zip(tokens, preds)))
        
        errors = []
        patterns: list[Pattern] = []
        content = ""
        first = True

        p_content, p_tag, p_start = "", None, None

        for token, pred in zip(tokens, preds):
            if token in self.tokenizer.all_special_tokens:
                continue

            if token.startswith(self.prefix) and not first:
                content += " "
            first = False

            if pred[0] in ["O", "S", "B"]:
                if p_tag is not None:
                    patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))

                if pred[0] == "S":
                    patterns.append(Pattern(content=token.lstrip(self.prefix), tag=pred[2:], start=len(content), end=None))

                if pred[0] == "B":
                    p_content, p_tag, p_start = token.lstrip(self.prefix), pred[2:], len(content)
                else:
                    p_content, p_tag, p_start = "", None, None

            elif pred[0] in ["I", "E"]:
                c_tag = pred[2:]

                if c_tag != p_tag:
                    if self.label_matching == LabelMatching.MATCH_ALL_TAGS:
                        raise InferenceError(f"Tag conflict: {p_tag} != {c_tag} | text = |{text}|")

                    elif self.label_matching == LabelMatching.MATCH_GOAL_TAG:
                        errors.append(f"Tag conflict: {p_tag} != {c_tag} | {p_content} | {token}")

                        # Close the previous pattern and start the current tag as the new pattern
                        if p_tag is not None:
                            patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))

                        # Create a new pattern
                        p_content, p_tag, p_start = token.lstrip(self.prefix), pred[2:], len(content)

                    else:
                        raise NotImplementedError()

                else:
                    p_content += token.replace(self.prefix, " ")

                if pred[0] == "E":
                    patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))
                    p_content, p_tag, p_start = "", None, None

            else:
                raise NotImplementedError()

            content += token.lstrip(self.prefix)

        if p_tag is not None:
            patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))

        if text != content:
            raise InferenceError(f"Mismatch text: |{text}| != |{content}|")

        # patterns = expand_full_word_of_content(content, patterns)

        return TaggerOutput(
            content=text,
            patterns=patterns,
            token_level_outputs=list(zip(tokens, preds)),
            errors=errors,
        )


def test():
    tagger = TaggerInference(
        model_path="logs/tagger_09_mMiniLM/checkpoint-6010",
        tokenizer_name="microsoft/Multilingual-MiniLM-L12-H384",
        # label_matching=LabelMatching.MATCH_ALL_TAGS,
        label_matching=LabelMatching.MATCH_GOAL_TAG,
    )
    # text = "nay nước chảy tậm (1.7 m3/s) đấy, rất nguy hiểm (tối đa là 2,3m3/s)"
    # print(tagger.inference(text))
    # print()

    text = "hôm qua **15/8/2023**[DATE] lúc **14h30**[TIME] tao đi mua **RTX**[ALPHANUM_ID] **4090**[ALPHANUM_ID] giá **35.000.000đ**[MONEY] ở **123A/456**[ADDRESS] Trần Hưng Đạo, thấy biển số **51F-12345**[PLATE] đậu trước cửa. Chủ shop nói máy chạy được **144fps**[MEASUREMENT] ở độ phân giải **3840x2160px**[DIMENSION], bảo hành **36**[INTEGER_n] tháng từ **1/9-31/12/2026**[DATE_RANGE]. Tao test benchmark **3DMark**[FOREIGN_WORD] được điểm **15.678**[FLOAT_n], nhưng nhiệt độ lên tới **82°C**[MEASUREMENT] sau **2h15p**[TIME] chạy. Email liên hệ là **support@nvidia.com**[EMAIL], hotline **1900-123-456**[PHONE]."
    text, _ = calibrate_tags(text)

    outputs = tagger.inference(text)[0]
    print("Result:", convert_to_raw_content(outputs.content, outputs.patterns), end="\n\n")
    print("Error:", outputs.errors)
    print("\n----\n")

    text = 'Acc @user_2k3 (SN 2k3) ở 102/3A Ngô Tất Tố, P.19, Q.Bình Thạnh đã post link http://localhost:8501/ lên group "Hội >9đ Toán" lúc 19:05, đòi 500k cho file C:/Windows/System32'
    print(clean_content(text), end="\n\n")

    outputs = tagger.inference(text)[0]
    print("Result:", convert_to_raw_content(outputs.content, outputs.patterns), end="\n\n")
    print("Error:", outputs.errors)
    print("\n----\n")

    text = "hôm qua 15/8/2023 lúc 14h30 tao đi mua RTX 4090 giá 35.000.000đ ở 123A/456 Trần Hưng Đạo, thấy biển số 51F-12345 đậu trước cửa. Chủ shop nói máy chạy được 144fps ở độ phân giải 3840x2160px, bảo hành 36 tháng từ 1/9-31/12/2026. Tao test benchmark 3DMark được điểm 15.678, nhưng nhiệt độ lên tới 82°C sau 2h15p chạy. Email liên hệ là support@nvidia.com , hotline 1900-123-456."

    text = """16h ngày 27/9: Tây Tây Bắc ~35 km/h.
Vị trí: 15,4°N – 113,0°E, trên vùng biển Hoàng Sa.
Cường độ: cấp 12, giật 15.
Vùng nguy hiểm: 11,5–17,5°N; Đông 111,0°E.
Rủi ro thiên tai: cấp 3 (Bắc & Giữa Biển Đông, gồm Hoàng Sa).
16h ngày 28/9: Tây Tây Bắc 25–30 km/h, tiếp tục mạnh thêm.
Vị trí: 17,3°N – 107,7°E, vùng biển Nghệ An – Tp. Huế.
Cường độ: cấp 13, giật 16.
Vùng nguy hiểm: 13,5–21,0°N; 108,0–116,5°E.
Rủi ro thiên tai: cấp 3 (Bắc & Giữa Biển Đông, Hoàng Sa, biển Thanh Hoá – Quảng Ngãi, đảo Hòn Ngư, Cồn Cỏ, Lý Sơn, Bắc vịnh Bắc Bộ gồm Bạch Long Vỹ)."""

    text = "1 100 000 CCU là con số rất lớn đấy"

    text = "1.000.000đ * (1 + 1%) = 1.010.000đ"

    outputs = tagger.inference(text)[0]
    print(convert_to_raw_content(outputs.content, outputs.patterns), end="\n\n")
    print(outputs.errors)

    # print(len(target_patterns), len(outputs.patterns))


if __name__ == "__main__":
    test()
