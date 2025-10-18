import os
import json
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from utils.text.tagger.augment.base import Pattern
from utils.text.tagger.m30_train import DebertaV2CrfForTokenClassification, AutoTaggerModel


class InferenceError(Exception):
    pass


@dataclass
class TaggerOutput:
    patterns: list[Pattern]
    token_level_outputs: list[tuple[str, str]] | None = None
    errors: list[str] | None = None


class TaggerInference:
    def __init__(
        self, 
        checkpoint_path: str, 
        model_name: str = "microsoft/mdeberta-v3-base", 
        label_consistency_level: int = 2, 
        device="cuda",
    ):
        """Initialize NER inference model
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            model_name: Base model name used for tokenizer
        """
        self.device = device
        self.label_consistency_level = label_consistency_level
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
        with open(os.path.join(checkpoint_path, "config.json")) as f:
            config = json.load(f)
        if config["architectures"][0] == "DebertaV2CrfForTokenClassification":
            model_cls = DebertaV2CrfForTokenClassification
        else:
            model_cls = AutoModelForTokenClassification
        print("NERInference: using", model_cls)
        self.model = model_cls.from_pretrained(checkpoint_path, token=False)
        self.model.eval().to(device)
        
        # Load label mappings
        self.label2id = self.model.config.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        self.prefix = "▁"
            
    def predict(self, text: str):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        tokens: list[str] = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device))
            
            if hasattr(outputs, "tags"):
                preds = outputs.tags
            else:
                preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
                preds = [self.id2label[p] for p in preds]

        # print(list(zip(tokens, preds)))
        return self.postprocess(text, tokens, preds)
    
    def inference(self, texts: list[str]):
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            if hasattr(outputs, "tags"):
                preds_list = outputs.tags
            else:
                preds_list = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            preds_list = [[self.id2label[p] for p in preds] for preds in preds_list]
            
        results: list[TaggerOutput] = []    
        for text, input_ids, attn_mask, preds in zip(texts, inputs.input_ids, inputs.attention_mask, preds_list):
            S = attn_mask.sum()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[: S])
            preds = preds[: S]
            results.append(self.postprocess(text, tokens, preds))
        return results
        
    def postprocess(self, text: str, tokens: list[str], preds: list[str]):
        errors = []
        patterns: list[Pattern] = []
        content = ""
        first = True

        p_tag = None
        p_content = ""
        p_start = None

        # Ensure always start with B token, if 
        for token, pred in zip(tokens, preds):
            if token in self.tokenizer.all_special_tokens:
                continue

            if token.startswith(self.prefix) and not first:
                content += " "
            first = False

            if pred[0] == "B":
                if p_tag is not None:
                    patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))

                p_tag = pred[2:]
                p_content = token.lstrip(self.prefix)
                p_start = len(content)

            elif pred[0] == "I":
                c_tag = pred[2:]
                resolve = False
                
                if c_tag != p_tag:
                    if self.label_consistency_level == 2:
                        raise InferenceError(f"Tag conflict: {p_tag} != {c_tag} | text = |{text}|")
                    else:
                        errors.append(f"Tag conflict: {p_tag} != {c_tag} | {p_content} | {token}")

                        if self.label_consistency_level == 1:
                            if p_tag is not None:
                                patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))
                            
                            p_tag = pred[2:]
                            p_content = token.lstrip(self.prefix)
                            p_start = len(content)
                            
                            resolve = True

                if not resolve:
                    if token.startswith(self.prefix):
                        p_content += " " + token.lstrip(self.prefix)
                    else:
                        p_content += token.lstrip(self.prefix)

            else:
                if p_tag is not None:
                    patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))

                p_tag = None
                p_content = ""
                p_start = None

            content += token.lstrip(self.prefix)

        if p_tag is not None:
            patterns.append(Pattern(content=p_content, tag=p_tag, start=p_start, end=None))

        if text != content:
            raise InferenceError(f"Mismatch text: |{text}| != |{content}|")
            
        # return patterns, list(zip(tokens, preds)), errors
        return TaggerOutput(
            patterns=patterns,
            token_level_outputs=list(zip(tokens, preds)),
            errors=errors,
        )


def run_predict():
    # To use inference, example:
    inferencer = TaggerInference("logs/tagger_05/checkpoint-17840")
    # text = "Tôi có 100 đồng và 50.5 kg gạo"
    # text = "Tôi có 50.5 kg gạo"
    text = "Tôi có 50.5kg gạo"
    # text = "Tôi có năm kg gạo"
    # text = "dành 1/3 tài sản cho hôm 1/3 đi chơi"
    # text = "dành 1/3 tài sản cho 1/3 đi chơi"
    # text = "dành 1/3 tài sản cho ngày 1/3 đi chơi"
    # text = "dành (1/3) tài sản cho ngày 1-3 đi chơi"
    # text = "dành tài sản cho ngày đi chơi"
    output = inferencer.predict(text)
    for pattern in output.patterns:
        print(f"Entity: '{pattern.content}' | Tag: {pattern.tag} | Position: {pattern.start}-{pattern.end}")
    print(output.patterns)
    
    
if __name__ == "__main__":
    run_predict()