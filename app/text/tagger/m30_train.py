import sys; sys.path.append(".") # fmt: skip
import os
import json
import jsonlines
import random
import hashlib
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
)
from transformers.models.deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchcrf import CRF
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .augment.base import Pattern, SPECIAL_CHARACTERS

# from ....models.envibert.tokenizer import RobertaTokenizer as EnvibertTokenizer


@dataclass
class Word:
    content: str
    start: int
    end: int
    index: int


def startswith(text: str, prefixes: list[str]):
    for pre in prefixes:
        if text.startswith(pre):
            return True
    return False


def punc_priority(group_punc):
    if "?" in group_punc:
        return "?"
    if "!" in group_punc:
        return "!"
    if "." in group_punc:
        return "."
    return ","


class ExtractError(Exception):
    pass


def align_token_labels(content: str, patterns: list[Pattern], tokenizer, style: str, prefix="▁", verbose=False):
    assert style in ["bies", "bi"]

    input_ids = tokenizer(content).input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_tokens_map = set(tokenizer.special_tokens_map.values())

    if verbose:
        print(tokens)

    words: list[Word] = []
    start = 0
    for index, token in enumerate(tokens):
        if token == tokenizer.unk_token:
            raise ExtractError(f"Cannot encode data: {tokens}")

        if token not in special_tokens_map:
            if token.startswith(prefix):
                token = token[len(prefix) :]
                if start > 0:
                    start += 1  # add space
            words.append(Word(content=token, start=start, end=start + len(token), index=index))
            start += len(token)
            if start > len(content):
                raise ExtractError()

    for word in words:
        if word.content != content[word.start : word.end]:
            if startswith(content[word.start : word.end], list(SPECIAL_CHARACTERS)) is False:
                raise ExtractError(f"|{word.content}| != |{content[word.start : word.end]}|")

    labels = ["O" for _ in range(len(input_ids))]
    for pattern in patterns:
        inner_words: list[Word] = []

        for word in words:
            if word.end <= pattern.start or pattern.end <= word.start:
                pass
            else:
                inner_words.append(word)

        if len(inner_words) == 0:
            raise ExtractError(f"Not found inner words of pattern {pattern}")

        if len(inner_words) == 1:
            labels[inner_words[0].index] = "S-" + pattern.tag
        else:
            labels[inner_words[0].index] = "B-" + pattern.tag
            labels[inner_words[-1].index] = "E-" + pattern.tag
            for word in inner_words[1:-1]:
                labels[word.index] = "I-" + pattern.tag

        start = inner_words[0].start
        end = inner_words[-1].end
        if start != pattern.start or end != pattern.end:
            raise ValueError(
                f"{start} != {pattern.start} # {end} != {pattern.end} # |{content[start : end]}| != |{content[pattern.start : pattern.end]}|"
            )

    if style == "bi":
        for index in range(len(labels)):
            if labels[index][0] == "S":
                labels[index] = "B" + labels[index][1:]
            elif labels[index][0] == "E":
                labels[index] = "I" + labels[index][1:]

    if verbose:
        print(list(zip(tokens, labels)))

    return input_ids, labels


def load_dataset(path):
    dataset = []
    with jsonlines.open(path) as f:
        for item in f:
            dataset.append(dict(content=item["content"], patterns=[Pattern(**p) for p in item["patterns"]]))
    return dataset


def _worker(**kwargs):
    try:
        result = align_token_labels(**kwargs)
    except:
        result = None
    kwargs.pop("tokenizer")
    return dict(**kwargs, result=result)


def run_validate_dataset_and_tokenizer(tokenizer, ds_path: str, output_dir: str = None):
    from quick_utils.common.mp_utils import execute_parallel

    text = "(hay là 80, )"
    text = "(hay là 80.)"
    text = "( 80 à )"
    text = '("80" à)'

    # print(tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids))
    # return

    if False:
        ex = {
            "content": "Tại khu vực này, việc đỗ xe tạm thời chỉ được phép trong -63.885953 µm/s - -64.5gr, và vị trí đỗ cần cách cột mốc giao thông ít nhất -46.577,840 - -8.831.893,68757 /tấn.",
            "patterns": [
                Pattern(content="-63.885953 µm/s - -64.5gr", tag="NUMBER_RANGE", start=57, end=82),
                Pattern(content="-46.577,840 - -8.831.893,68757 /tấn", tag="NUMBER_RANGE", start=133, end=168),
            ],
            "result": None,
        }
        align_token_labels(ex["content"], ex["patterns"], tokenizer, verbose=True)
        return

    ds = load_dataset(ds_path)

    task_inputs = [{**ex, "tokenizer": tokenizer} for ex in ds]

    errors = []
    success = []
    for result in execute_parallel(_worker, task_inputs, max_workers=8, use_process=False, batch_size=10000):
        if result["result"] is None:
            errors.append(
                dict(
                    content=result["content"],
                    patterns=[p.to_dict() for p in result["patterns"]],
                )
            )
        else:
            success.append(
                dict(
                    content=result["content"],
                    patterns=[p.to_dict() for p in result["patterns"]],
                )
            )

    success = sorted(success, key=lambda x: hashlib.sha256(json.dumps(x).encode()).hexdigest())
    random.seed(8686)
    random.shuffle(success)

    for e in errors[:3]:
        print(e)
        print()

    print("success =", len(success))
    print("errors =", len(errors))

    if output_dir is None:
        output_dir = os.path.dirname(ds_path)
    os.makedirs(output_dir, exist_ok=True)
    fn, ext = os.path.splitext(ds_path)

    fn_s = fn + "_validated" + ext
    fn_s = os.path.join(output_dir, os.path.split(fn_s)[1])
    print("Save success to", fn_s)
    with jsonlines.open(fn_s, "w") as f:
        f.write_all(success)

    fn_e = fn + "_failed" + ext
    fn_e = os.path.join(output_dir, os.path.split(fn_e)[1])
    print("Save errors to", fn_e)
    with jsonlines.open(fn_e, "w") as f:
        f.write_all(errors)


def run_prepare_label2id(paths: list[str], output_path: str):
    labels = set(["O"])
    for path in paths:
        with jsonlines.open(path) as f:
            for item in tqdm(f, desc=path):
                for p in item["patterns"]:
                    labels.add("B-" + p["tag"])
                    labels.add("I-" + p["tag"])
                    labels.add("E-" + p["tag"])
                    labels.add("S-" + p["tag"])
    labels = {l: i for i, l in enumerate(sorted(labels))}
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)


def rename_with_hash(path: str, length=12):
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:length]
    fn, ext = os.path.splitext(path)
    new_path = fn + "_" + digest + ext
    os.rename(path, new_path)
    print("Rename", path, "to", new_path)


def validate(content: str, patterns: list[Pattern]):
    for pattern in patterns:
        if content[pattern.start : pattern.end] != pattern.content:
            print("DIFF = |" + content[pattern.start : pattern.end] + "| != |" + pattern.content + "|")
            return False
    return True


class MyDataset(Dataset):
    """
    {
        "content": "Để báo cáo chi tiết về tình hình tài chính hiện tại, quý vị có thể liên hệ số 028 7300 1234 và yêu cầu mã số báo cáo tài chính 55.777.999.",
        "patterns": [
            {
                "content": "028 7300 1234",
                "tag": "PHONE",
                "start": 78,
                "end": 91,
                "normed_content": null
            },
            {
                "content": "55.777.999",
                "tag": "INTEGER_big",
                "start": 127,
                "end": 137,
                "normed_content": null
            }
        ]
    }
    """

    def __init__(
        self,
        data_paths: str | list[str],
        label_id_path: str,
        tokenizer,
        prefix: str,
        concat_candidates_range: tuple[int, int] = (1, 4),
        concat_prob: float = 0.0,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.prefix = prefix
        self.concat_candidates_range = concat_candidates_range
        self.concat_prob = concat_prob

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.items = []
        for path in data_paths:
            self.items.extend(load_dataset(path))

        self._indexes_pool = []

        with open(label_id_path) as f:
            self.label2id = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.prepare(self.items[index])

    def next_indexes(self, n: int):
        ids = []
        if len(self._indexes_pool) < n:
            ids = self._indexes_pool
            n -= len(ids)
            self._indexes_pool = list(range(len(self.items)))
            random.shuffle(self._indexes_pool)
        ids += self._indexes_pool[:n]
        self._indexes_pool = self._indexes_pool[n:]
        return ids

    def prepare(self, item: dict):
        if self.concat_prob > 0 and random.random() < self.concat_prob:
            n = random.randint(self.concat_candidates_range[0], self.concat_candidates_range[1])
            aug_items = [item] + [self.items[idx] for idx in self.next_indexes(n)]
            content = ""
            patterns = []
            for it in aug_items:
                if content == "":
                    shifted = len(content)
                    content += it["content"]
                else:
                    shifted = len(content) + 1
                    content += " " + it["content"]

                for ptn in it["patterns"]:
                    ptn: Pattern
                    patterns.append(ptn.shift_index(shifted))

            if validate(content, patterns):
                item = dict(content=content, patterns=patterns)
            else:
                print("--- CONCAT FAILED ---")
                print(content)
                print(patterns)
                print("----")
                raise ValueError("Concat failed")

        input_ids, labels = align_token_labels(**item, tokenizer=self.tokenizer, prefix=self.prefix)
        labels = [self.label2id[l] for l in labels]
        return dict(input_ids=input_ids, labels=labels)


def pad_1d(tensors: list[torch.Tensor], pad_value):
    max_len = max([t.size(0) for t in tensors])
    padded = torch.full((len(tensors), max_len), fill_value=pad_value, dtype=tensors[0].dtype)

    for idx, tensor in enumerate(tensors):
        padded[idx, : tensor.size(0)] = tensor

    return padded


class Collator:
    LABEL_IGNORED_ID = -100
    MAX_LENGTH = 512

    def __init__(self, tokenizer: AutoTokenizer, label_pad_id: int = LABEL_IGNORED_ID):
        self.pad_token_id = tokenizer.pad_token_id  # type: ignore
        self.eos_token_id = tokenizer.eos_token_id  # type: ignore
        self.label_pad_id = label_pad_id

        print("pad_token_id =", self.pad_token_id)
        print("eos_token_id =", self.eos_token_id)

    def __call__(self, samples: list[dict]):
        input_ids = []
        attention_mask = []
        labels = []

        for sample in samples:
            inp = sample["input_ids"]
            lbl = sample["labels"]

            if len(inp) > self.MAX_LENGTH:
                inp = inp[: self.MAX_LENGTH - 1] + [self.eos_token_id]
                lbl = lbl[: self.MAX_LENGTH - 1] + [self.label_pad_id]

            input_ids.append(torch.LongTensor(inp))
            attention_mask.append(torch.ones((len(inp),)).long())
            labels.append(torch.LongTensor(lbl))

        return dict(
            input_ids=pad_1d(input_ids, self.pad_token_id),
            attention_mask=pad_1d(attention_mask, 0),
            labels=pad_1d(labels, self.label_pad_id),
        )


@dataclass
class CrfTokenClassifierOutput(TokenClassifierOutput):
    """
    Output class for our CRF token classifier.
    It contains the loss, logits, and the Viterbi-decoded tags.
    """

    tags: Optional[List[List[int]]] = None  # Our new field for the decoded tags


def get_crf_constraints(label2id: dict):
    """Create constraint mask for BIES tagging to prevent invalid transitions"""
    num_tags = len(label2id)
    constraints = torch.zeros((num_tags, num_tags), dtype=torch.bool)

    # Allow all transitions by default
    constraints.fill_(True)

    id2label = {v: k for k, v in label2id.items()}

    for from_id in range(num_tags):
        for to_id in range(num_tags):
            from_label = id2label[from_id]
            to_label = id2label[to_id]

            # Extract prefix and entity type
            from_prefix = from_label[0] if from_label != "O" else "O"
            to_prefix = to_label[0] if to_label != "O" else "O"
            from_type = from_label[2:] if from_label != "O" else ""
            to_type = to_label[2:] if to_label != "O" else ""

            # Block invalid BIES transitions
            if from_prefix == "B":  # After B-, only I- or E- of same type allowed
                if to_prefix not in ["I", "E"] or to_type != from_type:
                    constraints[from_id, to_id] = False
            elif from_prefix == "I":  # After I-, only I- or E- of same type allowed
                if to_prefix not in ["I", "E"] or to_type != from_type:
                    constraints[from_id, to_id] = False
            elif from_prefix in ["E", "S", "O"]:  # After E-/S-/O, cannot transition to I-
                if to_prefix == "I":
                    constraints[from_id, to_id] = False

    return constraints


class AutoTaggerModel(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "model"

    def __init__(self, config: PretrainedConfig, pretrain_model=None):
        super().__init__(config)

        if pretrain_model is None:
            pretrain_model = AutoModel.from_config(config)
        self.model = pretrain_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Apply BIES constraints to prevent invalid tag transitions
        constraints = get_crf_constraints(config.label2id)
        # Mask invalid transitions with very large negative values
        self.crf.transitions.data.masked_fill_(~constraints, -10000.0)

    @classmethod
    def from_kwargs(cls, model_name_or_path: str, **kwargs):
        pretrained_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        return cls(pretrained_model.config, pretrained_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None
    ) -> CrfTokenClassifierOutput:
        # print(input_ids.shape)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        tags = None
        crf_mask = attention_mask.bool()

        # Calculate loss if labels are provided
        if labels is not None:
            loss = -self.crf(emissions=logits, tags=labels, mask=crf_mask, reduction="mean")

            if torch.isnan(loss):
                print(loss)
                import pdb; pdb.set_trace() # fmt: skip

        # Always perform decoding when not in training mode or when labels are not provided
        # This makes the model ready for inference by default
        if not self.training or labels is None:
            tags = self.crf.decode(emissions=logits, mask=crf_mask)

        return CrfTokenClassifierOutput(
            loss=loss, logits=logits, tags=tags, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class CrfTrainer(Trainer):
    def prediction_step(  # type: ignore
        self,
        model: AutoTaggerModel,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        labels = inputs.get("labels")
        assert labels is not None

        decoded_predictions = outputs.tags

        # The rest of the logic to pad the predictions remains the same.
        padded_predictions = []
        ignored_label_id = -100
        for i, (pred_seq, label_seq) in enumerate(zip(decoded_predictions, labels)):
            padded_seq = pred_seq + [ignored_label_id] * (labels.shape[1] - len(pred_seq))
            padded_predictions.append(padded_seq)
            # import pdb; pdb.set_trace() # fmt: skip

        return (None, torch.tensor(padded_predictions, device=outputs.logits.device), labels)


def compute_report(y_true, y_pred, label_names, save_report_path: str = None):
    classification_rep = ""
    try:
        classification_rep = classification_report(y_true=y_true, y_pred=y_pred, labels=label_names, zero_division=0)
    except Exception as e:
        classification_rep = f"Error generating classification report: {str(e)}"

    if save_report_path:
        report = classification_report(y_true=y_true, y_pred=y_pred, labels=label_names, zero_division=0, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_excel(save_report_path, index=True)

    return f"<pre>{classification_rep}</pre>"


def compute_metrics(eval_pred, label_O_id: int, label2id: dict, save_report_dir: str = None):
    label_ignored_id = -100

    id2label = {v: k for k, v in label2id.items()}
    label_O = id2label[label_O_id]

    predictions, labels = eval_pred

    # Remove ignored index (padding) positions
    raw_preds = []
    raw_labels = []

    for prediction, label in zip(predictions, labels):
        for pred_id, label_id in zip(prediction, label):
            if pred_id != label_ignored_id and label_id != label_ignored_id:
                raw_preds.append(id2label[pred_id])
                raw_labels.append(id2label[label_id])

    def convert_to_goal_labels(labels: list[str], label_O: str):
        return [l if l == label_O else l[2:] for l in labels]

    goal_preds = convert_to_goal_labels(raw_preds, label_O)
    goal_labels = convert_to_goal_labels(raw_labels, label_O)

    def filter_non_O_pair(preds: list[str], labels: list[str], label_O):
        non_o_preds, non_o_labels = [], []
        assert len(preds) == len(labels)
        for p, l in zip(preds, labels):
            if p != label_O or l != label_O:
                non_o_preds.append(p)
                non_o_labels.append(l)
        return non_o_preds, non_o_labels

    raw_non_o_preds, raw_non_o_labels = filter_non_O_pair(raw_preds, raw_labels, label_O)
    goal_non_o_preds, goal_non_o_labels = filter_non_O_pair(goal_preds, goal_labels, label_O)

    full_tag_names = sorted(label2id.keys())
    goal_tag_names = convert_to_goal_labels(full_tag_names, label_O)
    goal_tag_names = sorted(set(goal_tag_names))
    metrics = {}

    def add_metrics(name: str, preds, labels, tag_names):
        save_path = None
        if save_report_dir is not None:
            save_path = os.path.join(save_report_dir, name + ".xlsx")
        assert len(preds) == len(labels)
        if len(preds) == 0:
            return
        metrics[f"{name}_acc"] = accuracy_score(labels, preds)
        metrics[f"{name}_f1"] = f1_score(labels, preds, average="weighted")
        metrics[f"{name}_report"] = compute_report(labels, preds, tag_names, save_report_path=save_path)

    add_metrics("raw_tag", raw_preds, raw_labels, full_tag_names)
    add_metrics("goal_tag", goal_preds, goal_labels, goal_tag_names)
    add_metrics("raw_non_o_tag", raw_non_o_preds, raw_non_o_labels, full_tag_names)
    add_metrics("goal_non_o_tag", goal_non_o_preds, goal_non_o_labels, goal_tag_names)

    return metrics


def run_evaluate(model_path: str, tokenizer_name: str, test_data_path: str, label_id_path: str, multiple=False):
    if multiple:
        ckpt_prefix = "checkpoint-"
        model_paths = [
            (int(folder[len(ckpt_prefix) :]), folder) for folder in os.listdir(model_path) if folder.startswith(ckpt_prefix)
        ]
        model_paths = sorted(model_paths)
        model_paths = [(step, os.path.join(model_path, folder)) for step, folder in model_paths]
        model_paths = [(step, folder) for step, folder in model_paths if os.path.isdir(folder)]
        report_to = "tensorboard"
    else:
        model_paths = [(0, model_path)]
        report_to = "none"

    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=False)
    test_ds = MyDataset(test_data_path, label_id_path, tokenizer, prefix="▁")
    collator = Collator(tokenizer, label_pad_id=test_ds.label2id["O"])

    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir="logs/temp_eval",
        per_device_eval_batch_size=32,
        do_train=False,
        do_eval=True,
        report_to=report_to,
    )

    trainer = CrfTrainer(
        args=training_args,
        model=AutoTaggerModel.from_pretrained(model_paths[0][1], token=False),
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=lambda x: compute_metrics(
            x, test_ds.label2id["O"], test_ds.label2id, save_report_dir=training_args.output_dir
        ),
    )

    for step, model_path in model_paths:
        print(f"Evaluating checkpoint: {model_path}")
        trainer._load_from_checkpoint(model_path)
        trainer.state.global_step = step
        metrics = trainer.evaluate()
        if len(model_paths) == 1:
            print(json.dumps(metrics, indent=4, ensure_ascii=False))


def run_train(
    train_paths: list[str],
    valid_paths: list[str],
    label_id_path: str,
    exp_dir: str,
    model_name: str,
    tokenizer_name: str | None,
    tokenizer_token_prefix: str,
):
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=False)

    train_ds = MyDataset(
        data_paths=train_paths,
        label_id_path=label_id_path,
        tokenizer=tokenizer,
        prefix=tokenizer_token_prefix,
        concat_prob=0.4,
        concat_candidates_range=(1, 4),
    )
    test_ds = MyDataset(
        data_paths=valid_paths,
        label_id_path=label_id_path,
        tokenizer=tokenizer,
        prefix=tokenizer_token_prefix,
    )
    collator = Collator(tokenizer, label_pad_id=test_ds.label2id["O"])

    # model = DebertaV2CrfForTokenClassification.from_pretrained(
    model = AutoTaggerModel.from_kwargs(
        model_name,
        num_labels=len(test_ds.label2id),
        label2id=test_ds.label2id,
        token=False,
    )
    assert model.config.label2id == test_ds.label2id
    print(model)

    # Fix CRF parameter initialization if using CRF model
    # model.crf.reset_parameters() # <= This erases the constraints you just applied, so comment it

    training_args = TrainingArguments(
        output_dir=exp_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        # eval_strategy="steps",
        # eval_steps=1,
        warmup_ratio=0.1,
        num_train_epochs=10,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        # eval_on_start=True,
        save_strategy="epoch",
        log_level="info",
        logging_steps=10,
        report_to="tensorboard",
        optim="adamw_torch",
        learning_rate=5e-5,
        # learning_rate=1e-4,
        lr_scheduler_type="cosine_with_restarts",
        auto_find_batch_size=True,
    )
    trainer = CrfTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=lambda x: compute_metrics(x, test_ds.label2id["O"], label2id=test_ds.label2id),
    )

    # trainer.save_model()
    trainer.train()


def test_dataset():
    model_name = "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)

    train_ds = MyDataset("data/train/train_validated.jsonl", "", tokenizer, prefix="▁", concat_prob=1)
    for d in train_ds:
        print(d)
        break


def test_tokenizer():
    from src.utils.text.tagger.augment.base import calibrate_tags, SPECIAL_CHARACTERS

    # tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", token=False)

    # tokenizer = EnvibertTokenizer.special_init("checkpoints/envibert")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384", token=False)

    text = "Anh giải thích **hotro@startupsupport.co**[EMAIL] đến 17đ/ngày tùy kinh nghiệm **+3090đ**[MONEY] thế nào khi tỉ lệ **#100**[MONEY] giữ chân người dùng chỉ là **25%**[MEASUREMENT] trong khi mục tiêu đề ra là trên **40%**[MEASUREMENT]"
    content, patterns = calibrate_tags(text + " " + SPECIAL_CHARACTERS)
    print(content)
    align_token_labels(content, patterns, tokenizer, verbose=True)
    return

    if False:
        # Test comprehensive cases
        test_cases = [
            ("µm", "Special char at start"),
            ("µ m", "Special char with space after"),
            ("10µm", "Direct attachment"),
            ("10 µm", "With space before special char"),
            ("được phép trong -63.885953 Ym/s", "Vietnamese text without special char"),
            ("được phép trong -63.885953 µm/s", "Vietnamese text with special char"),
            ("30°C", "Degree symbol attached"),
            ("30 °C", "Degree symbol with space"),
        ]

        print("\n=== Comprehensive Tokenization Tests ===")
        for text, description in test_cases:
            tokens = tokenizer(text).input_ids
            tokens = tokenizer.convert_ids_to_tokens(tokens)
            print(f"{description}: '{text}'")
            print(f"  → {tokens}")
            print()

        return

    null = None
    # data = {"content": "Nếu anh/chị muốn nâng hạn mức thẻ tín dụng lên 262.000 µm/giờ cùng với mmol/l, chúng tôi cần sao kê lương trung bình đạt 1.390.000,47oF trong 84,068 µm/giờ gần nhất.", "patterns": [{"content": "262.000 µm/giờ", "tag": "MEASUREMENT", "start": 47, "end": 61, "normed_content": None}, {"content": "mmol/l", "tag": "MEASUREMENT", "start": 71, "end": 77, "normed_content": None}, {"content": "1.390.000,47oF", "tag": "MEASUREMENT", "start": 121, "end": 135, "normed_content": None}, {"content": "84,068 µm/giờ", "tag": "MEASUREMENT", "start": 142, "end": 155, "normed_content": None}]}

    # data = {"content": "(Kỹ thuật viên, Khẩn trương) Vui lòng kiểm tra lại độ phân giải của video đầu ra; nó bắt buộc phải là 5 nhân 3 nhân 4 hoặc 0,717621x10ºC pixel, bất kỳ định dạng nào khác sẽ không được chấp nhận!", "patterns": [{"content": "5", "tag": "INTEGER_n", "start": 102, "end": 103, "normed_content": null}, {"content": "3", "tag": "INTEGER_n", "start": 109, "end": 110, "normed_content": null}, {"content": "4", "tag": "INTEGER_n", "start": 116, "end": 117, "normed_content": null}, {"content": "0,717621x10ºC", "tag": "DIMENSION", "start": 123, "end": 136, "normed_content": null}]}

    # data = {"content": "Chúng tôi thẩm định diện tích thường ở mức 80-120m², nhưng các trường hợp đặc biệt thường yêu cầu 150-200m² tùy vị trí.", "patterns": [{"content": "80-120m²", "tag": "NUMBER_RANGE", "start": 43, "end": 51, "normed_content": null}, {"content": "150-200m²", "tag": "NUMBER_RANGE", "start": 98, "end": 107, "normed_content": null}]}

    data = [
        {
            "content": "Theo hồ sơ hiện có , diện tích mặt sàn của cơ sở kinh doanh này là 120.5m .",
            "patterns": [{"content": "120.5m", "tag": "MEASUREMENT", "start": 67, "end": 73, "normed_content": null}],
        },
        {
            "content": "Theo hồ sơ hiện có , diện tích mặt sàn của cơ sở kinh doanh này là 120.5m² và -63.885953 µm/s năm 1.500₫ .",
            "patterns": [{"content": "120.5m²", "tag": "MEASUREMENT", "start": 67, "end": 74, "normed_content": null}],
        },
        {
            "content": "Sản phẩm này không chỉ giúp giảm lượng khí thải CO₂ tới -3916tr/CP mà còn có thể tái sử dụng tới 89,278 usd/năm vật liệu .",
            "patterns": [
                {"content": "-3916tr/CP", "tag": "MEASUREMENT", "start": 56, "end": 66, "normed_content": null},
                {"content": "89,278", "tag": "INTEGER_big", "start": 97, "end": 103, "normed_content": null},
                {"content": "usd/năm", "tag": "MEASUREMENT", "start": 104, "end": 111, "normed_content": null},
            ],
        },
    ]

    ex = data[2]
    content = ex["content"]
    patterns = [Pattern(**p) for p in ex["patterns"]]

    align_token_labels(content, patterns, tokenizer, verbose=True)


def main():
    # test_tokenizer()

    # run_validate_dataset_and_tokenizer("data/train/dataset_news.jsonl")
    # run_validate_dataset_and_tokenizer("data/train/train.jsonl")
    # run_validate_dataset_and_tokenizer("data/train/test.jsonl")

    # rename_with_hash("data/train/train_validated.jsonl")
    # rename_with_hash("data/train/test_validated.jsonl")
    # rename_with_hash("data/train/dataset_news_validated.jsonl")

    if False:
        run_prepare_label2id(
            [
                # "data/train/train_validated.jsonl",
                "data/train/train_validated_e7f6eb4ddfcc.jsonl",
                # "data/train/test_validated.jsonl",
                "data/train/test_validated_e60c99fcf3d1.jsonl",
                "data/train/dataset_news_validated_bc0cd9d98815.jsonl",
            ]
        )
        rename_with_hash("data/train/label2id.json")

    # test_dataset()

    run_train(
        train_paths=[
            "data/train/train_validated_e7f6eb4ddfcc.jsonl",
            "data/train/dataset_news_validated_bc0cd9d98815.jsonl",
        ],
        valid_paths=[
            "data/train/test_validated_e60c99fcf3d1.jsonl",
        ],
        label_id_path="data/train/label2id_71c45107fad0.json",
        exp_dir="logs/tagger_07",
        model_name="microsoft/mdeberta-v3-base",
        tokenizer_name="microsoft/mdeberta-v3-base",
        tokenizer_token_prefix="▁",
    )
    # run_evaluate("logs/tagger_03/checkpoint-17670", model_name="microsoft/mdeberta-v3-base")


def main_0925_mMiniLM(stage: int):
    tokenizer_name = "microsoft/Multilingual-MiniLM-L12-H384"
    working_dir = "data/train_10"
    data_dir = os.path.join(working_dir, "mMiniLM")

    label_id_path = os.path.join(data_dir, "label2id.json")
    train_path = os.path.join(data_dir, "train_validated.jsonl")
    test_path = os.path.join(data_dir, "test_validated.jsonl")

    if stage == 0:
        test_tokenizer()

    if stage == 1:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        run_validate_dataset_and_tokenizer(tokenizer, os.path.join(working_dir, "train.jsonl"), output_dir=data_dir)
        run_validate_dataset_and_tokenizer(tokenizer, os.path.join(working_dir, "test.jsonl"), output_dir=data_dir)

    if stage == 2:
        run_prepare_label2id([train_path, test_path], output_path=label_id_path)
        return

    if stage == 3:
        run_train(
            train_paths=[train_path],
            valid_paths=[test_path],
            label_id_path=label_id_path,
            exp_dir="logs/tagger_10_mMimiLM_v2",
            model_name=tokenizer_name,
            tokenizer_name=None,
            tokenizer_token_prefix="▁",
        )

    if stage == 4:
        run_evaluate(
            model_path="logs/tagger_08_mMimiLM_v3/checkpoint-4770",
            tokenizer_name=tokenizer_name,
            test_data_path=test_path,
            label_id_path=label_id_path,
            # multiple=True,
        )


def main_0925_mDeberta(stage: int):
    tokenizer_name = "microsoft/mdeberta-v3-base"
    working_dir = "data/train_10"
    data_dir = os.path.join(working_dir, "mdeberta")

    label_id_path = os.path.join(data_dir, "label2id.json")
    train_path = os.path.join(data_dir, "train_validated.jsonl")
    test_path = os.path.join(data_dir, "test_validated.jsonl")

    if stage == 0:
        test_tokenizer()

    if stage == 1:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        run_validate_dataset_and_tokenizer(tokenizer, os.path.join(working_dir, "train.jsonl"), output_dir=data_dir)
        run_validate_dataset_and_tokenizer(tokenizer, os.path.join(working_dir, "test.jsonl"), output_dir=data_dir)

    if stage == 2:
        run_prepare_label2id([train_path, test_path], output_path=label_id_path)
        return

    if stage == 3:
        run_train(
            train_paths=[train_path],
            valid_paths=[test_path],
            label_id_path=label_id_path,
            exp_dir="logs/tagger_10_mdeberta",
            model_name=tokenizer_name,
            tokenizer_name=None,
            tokenizer_token_prefix="▁",
        )

    if stage == 4:
        run_evaluate(
            model_path="logs/tagger_10_mdeberta/checkpoint-4207",
            tokenizer_name=tokenizer_name,
            test_data_path=test_path,
            label_id_path=label_id_path,
            # multiple=True,
        )


if __name__ == "__main__":
    # main()
    main_0925_mMiniLM(3)
    # main_0925_mDeberta(3)
