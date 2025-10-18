import os
import sys
sys.path.append(".")
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import jsonlines
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from quick_utils.data_synthesis.client import Client
from quick_utils.data_synthesis.utils import format_text, auto_extract
from utils.text.tagger.prompts.template import REASONING_TEMPLATE_V1
from utils.text.tagger.prompts.elite import TAG_TO_PROMPT
from utils.text.tagger.prompts.contexts import CONTEXTS


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
):   
    good_examples_str = ""
    if good_examples is not None and len(good_examples) > 0:
        good_examples_str = "\n\n## VÍ DỤ MẪU TỐT:\n" + "\n".join(["- " + e for e in good_examples])
    
    # example_output_str = "\n".join([f"- [Câu ví dụ {i + 1}]" for i in range(num_new_examples)])
    example_output_str = f"- [Câu ví dụ 1]\n- ... (và {num_new_examples - 1} câu còn lại)"

    return format_text(
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


def sample(pool: list[str], k: int) -> list[str]:
    pool = [p for p in pool]
    random.shuffle(pool)
    unique = set()
    index = 0
    while len(unique) < k and index < len(pool):
        unique.add(pool[index])
        index += 1
    return list(unique)


class Loop:
    def __init__(
        self, 
        client: Client, 
        model: str, 
        contexts: list[dict[str, str]],
        num_loop: int, 
        pool_dir: str, 
        template_fn: Callable,
        num_given_examples: int = 5,
        num_new_examples: int = 10,
        num_special_examples: int = 5,
        generation_config: dict = {},
    ):
        self.client = client
        self.model = model
        self.contexts = contexts
        self.pool_dir = pool_dir
        self.template_fn = template_fn
        self.num_loop = num_loop
        self.num_given_examples = num_given_examples
        self.generation_config = generation_config
        self.num_new_examples = num_new_examples
        self.num_special_examples = num_special_examples

    def start(self, items: list[dict], max_workers: int):
        os.makedirs(working_dir, exist_ok=True)

        with self.client.messages.open_pool(
            num_workers=max_workers,
            working_dir=working_dir,
            save_responses=True,
            disable_tqdm=False,
            tqdm_total=len(items) * len(self.contexts) * self.num_loop,
        ) as batch_info:
            with ThreadPoolExecutor(max_workers=max_workers) as exec:
                futures = []
                for item in items:
                    if "__" in item["file_name"]:
                        num_min_repeat = 3
                    else:
                        num_min_repeat = 2

                    # if "__" not in item["file_name"]:
                    #     continue
                    
                    # if item["file_name"] == "NUMBER_RANGE__SCORE_0":
                    # if item["file_name"] != "PHONE__INTEGER_0":
                    #     continue

                    futures.append(exec.submit(
                        self.worker,
                        item=item,
                        contexts=self.contexts[:],
                        template_kwargs=dict(
                            num_new_examples=self.num_new_examples, 
                            num_special_examples=self.num_special_examples,
                            num_min_repeat=num_min_repeat,
                        )
                    ))

                for future in as_completed(futures):
                    future.result()

        return batch_info

    def worker(self, item: dict, contexts: list[dict], template_kwargs: dict):
        file_name = item["file_name"]
        pool_file = os.path.join(self.pool_dir, file_name + ".jsonl")
        excludes = ["file_name", "simple_examples", "enhanced_examples"]

        sentence_pool = []
        is_new = True
        if os.path.exists(pool_file):
            with jsonlines.open(pool_file) as f_in:
                for line in f_in:
                    sentence_pool.append(line["sentence"])
        if len(sentence_pool) != 0:
            is_new = False

        simple_examples = item["simple_examples"]
        enhanced_examples = item["enhanced_examples"]
        item = {k: v for k, v in item.items() if k not in excludes}

        os.makedirs(self.pool_dir, exist_ok=True)
        with jsonlines.open(pool_file, "a") as f_out:
            if is_new:
                for ex in simple_examples:
                    f_out.write({"context": "simple_example", "sentence": ex})
                for ex in enhanced_examples:
                    f_out.write({"context": "enhanced_example", "sentence": ex})
                    sentence_pool.append(ex)

            for context in contexts:
                for _ in range(self.num_loop):
                    item["examples"] = random.choices(simple_examples, k=2) + sample(sentence_pool, k=self.num_given_examples - 1)
                    
                    user_message = self.template_fn(**item, **template_kwargs, **context)
                    messages = [{"role": "user", "content": user_message}]
                    # print(user_message)
                    # exit(0)
                
                    try:
                        response = self.client.messages.pool.generate(
                            self.model, messages=messages, **self.generation_config
                        )
                        # client.pprint(response, model=model)

                        outputs = extract(response.content)
                        sentence_pool.extend(outputs)

                        for out in outputs:
                            f_out.write({"context": context, "sentence": out})
                    except Exception as e:
                        print("ERROR:", str(e))
                    

def extract(content: str) -> list[str]:
    try:
        examples: list[str] = auto_extract(content, "examples", split_lines=True)
    except KeyError as e:
        print(e)
        return []
    return examples


def main(start_index: int=None, end_index: int=None, dry_run=True):
    print("contexts =", len(CONTEXTS))

    # client = Client("openai"); model = "gpt4.1-mini"
    client = Client("gemini"); model = "gemini-2.5-flash-0520"

    user_message = apply_generate_examples_single_pattern_reasoning_template(
        tag="DATE_dmy",
        **CONTEXTS[-2],
        num_new_examples=10, 
        num_special_examples=5,
    )
    # print(user_message)
    # print(len(user_message.split()))
    # return

    messages = [{"role": "user", "content": user_message}]
    response = client.messages.generate(model, messages, thinking_budget=2400)
    client.pprint(response, model=model)


main()
