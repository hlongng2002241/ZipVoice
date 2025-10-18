import os
import sys
sys.path.append(".")
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import jsonlines
from collections import defaultdict

from quick_utils.data_synthesis.client import Client
from quick_utils.data_synthesis.utils import format_text, auto_extract
from quick_utils.data_synthesis import (
    BaseWorkflow,
    WorkflowConfig,
    WorkflowTask,
    WorkflowTaskManager,
    BaseInputItem,
    BaseOutputItem,
    BaseWorkflowInputArgs,
)
from utils.text.tagger.prompts.contexts import CONTEXTS
from utils.text.tagger.prompts.elite import TAG_TO_PROMPT
from utils.text.tagger.prompts.template import REASONING_TEMPLATE_V1


working_dir = "data/batches"


def apply_generate_examples_single_pattern_reasoning_template(
    tag: str,

    context: str,
    situations: str,
    roles: str,
    emotions: str,
    
    good_examples: list[str] | None = None,
    
    num_new_examples = 5,
    num_special_examples = 2,
    num_min_repeat = 2,

    index=None,
):   
    good_examples_str = ""
    if good_examples is not None and len(good_examples) > 0:
        good_examples_str = "\n\n## VÍ DỤ MẪU TỐT:\n" + "\n".join(["- " + e for e in good_examples])
    
    # example_output_str = "\n".join([f"- [Câu ví dụ {i + 1}, không cần phần giải thích]" for i in range(num_new_examples)])
    example_output_str = f"- [Câu ví dụ 1, không cần phần giải thích]\n- ... (và {num_new_examples - 1} câu còn lại)"

    user_message = format_text(
        REASONING_TEMPLATE_V1,
        inputs=TAG_TO_PROMPT[tag],
        situations=situations,
        context=context,
        roles=roles,
        emotions=emotions,
        example_output_str=example_output_str,
        num_new_examples=str(num_new_examples),
        num_special_examples=str(num_special_examples),
        num_min_repeat=str(num_min_repeat),
    )
    return [dict(role="user", content=user_message)]


class InputItem(BaseInputItem):
    tag: str

    context: str
    situations: str
    roles: str
    emotions: str
    
    num_new_examples: int
    num_special_examples: int
    num_min_repeat: int

    good_examples: list[str] | None = None

    index: int
    

class OutputItem(BaseOutputItem):
    tag: str

    context: str
    situations: str
    roles: str
    emotions: str

    num_new_examples: int
    num_special_examples: int
    num_min_repeat: int
    
    good_examples: list[str] | None = None

    index: int

    examples: list[str] | None = None


class InputArgs(BaseWorkflowInputArgs):
    contexts: list[dict[str, str]]
    num_loops: int
    num_new_examples: int
    num_special_examples: int
    tags: list[str] | None = None


class Generate(BaseWorkflow):
    __input_args_class__ = InputArgs
    __input_class__ = InputItem
    __output_class__ = OutputItem

    def prepare_inputs(self, input_args: InputArgs) -> list[InputItem]: # type: ignore
        if input_args.tags is None:
            tags = list(TAG_TO_PROMPT.keys())
        else:
            tags = input_args.tags

        items: list[InputItem] = []
        for tag in tags:
            for context in input_args.contexts:
                for index in range(input_args.num_loops):
                    items.append(InputItem(
                        allow_auto_id=True,

                        tag=tag,
                        **context,

                        num_new_examples=input_args.num_new_examples,
                        num_special_examples=input_args.num_special_examples,
                        num_min_repeat=3 if "__" in tag else 2,
                        
                        index=index,
                    ))
        return items
    
    def extract_output_content(self, content):
        return dict(examples=self.auto_extract(content, "examples", split_lines=True))


def attempt():
    print("contexts =", len(CONTEXTS))

    # client = Client("openai"); model = "gpt4.1-mini"
    client = Client("gemini"); model = "gemini-2.5-flash-0520"

    user_message = apply_generate_examples_single_pattern_reasoning_template(
        tag="ROMAN_NUMERAL__ALPHANUM_ID",
        **CONTEXTS[-2],
        num_new_examples=10, 
        num_special_examples=5,
    )[0]["content"]
    print(user_message)
    print(len(user_message.split()))
    # exit()

    messages = [{"role": "user", "content": user_message}]
    response = client.messages.generate(model, messages, thinking_budget=2400)
    client.pprint(response, model=model)

    exit()


def generate():
    print("contexts =", len(CONTEXTS))

    # client = Client("openai"); model = "gpt4.1-mini"
    client = Client("gemini"); model = "gemini-2.5-flash-0520"

    g = Generate(
        config=WorkflowConfig(
            working_dir="data/generations",
            use_batch_api=False,

            num_workers=16,
            generation_config=dict(
                thinking_budget=2400,
            )
        ),
        client=client,
        template=apply_generate_examples_single_pattern_reasoning_template,
        task_manager=WorkflowTaskManager({
            model: WorkflowTask(
                output_file="data/results/training.jsonl",
                finished_batch_ids=[
                    # => bad, have to filter manually
                    # "gembatch_ec5ee6e6681ac2f4fcc3fff569ade189",
                    # "gembatch_23f6d3c3746220e9a76c07917923ad6b",
                    # "gembatch_18fa3283909e331e7dd128f1ef3570b7",
                    # "gembatch_bd0ad195dd60fc7def95c338ded21491",
                    
                    # "gembatch_d465e130ba50b5dafedbf6df6c0b0341",
                    # "gembatch_7a8e8634c88cd778ac1a687efd326ba4",

                    # "gembatch_851a54cd8af193cd04b2e3bd05fbace7",

                    # prompt test p2 => bad, have to filter manually
                    # "gembatch_1e1a19acb80fa8d6e8879aef437d3587",

                    # prompt test p3 => bad, have to filter manually
                    # "gembatch_df1fef4e23d7ae4f2b532bbb8681ac48",

                    # prompt test p4 => good
                    # "gembatch_d0a04a714f0f0829b68fd64b6d834a73",

                    # "gembatch_accef2d86bfe2df1a922307dd6536950",

                    # "gembatch_4af62b8ad7c9b42af52dbdd7f43fe240"

                    # context 3
                    "gembatch_e4180c01269b5a26835d520de3fee223",
                    "gembatch_aadecbd724c151c18c14ff81c8a78dd4",

                    # Full, exclude context 3
                    "gembatch_d5e57021fc050a1ef3715f76eb172002",
                    "gembatch_5a997ad096ce195e8bf2ae3e7b38a6b3",

                    # Full
                    "gembatch_8255f9bebdf522cc49845055420446cb",
                ]
            )
        })
    )
    g.run(
        model=model,
        input_args=InputArgs(
            contexts=CONTEXTS,
            num_loops=4,
            num_new_examples=10,
            num_special_examples=5,
            # tags=["DATE_RANGE__MATH_EXPR"]
        ),
        inputs_offset=0,
        num_input_samples=None,
        parse_responses=True,
        dry_run=True,
    )
    return g


def export():
    group_by_tag = defaultdict(list)
    with (
        jsonlines.open("data/results/training.jsonl") as f_in,
        open("data/results/training.md", "w") as f_out
    ):
        for item in f_in:
            group_by_tag[item["tag"]].extend(item["examples"])
            
        first = True
        for k, vs in sorted(group_by_tag.items()):
            if not first:
                print(file=f_out)
            first = False
            print("# " + k, file=f_out)
            for v in vs:
                print("- " + v, file=f_out)


def group_exported(path: str):
    groups = defaultdict(list)
    tag = None
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            if line[0] == "#":
                tag = line[1:].strip()
            elif line[0] == "-":
                assert tag is not None
                sentence = line[1:].strip()
                groups[tag].append(sentence)
    
    fn, ext = os.path.splitext(path)
    output_path = fn + "_sorted" + ext
    with open(output_path, "w", encoding="utf-8") as f:
        for tag in sorted(groups):
            print("# " + tag, file=f)
            sents = sorted(set(groups[tag]))
            for s in sents:
                print("- " + s, file=f)
            print(file=f)

            u, v = len(sents), len(groups[tag])
            if u < v:
                print(tag, "=", u, "/", v)


# attempt()
generate()
export()
group_exported("data/results/training.md")
# group_exported("data/human_validated.md")
