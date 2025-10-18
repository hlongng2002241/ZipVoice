from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import random
import jsonlines

from quick_utils.data_synthesis import (
    BaseWorkflow,
    WorkflowConfig,
    WorkflowTask,
    WorkflowTaskManager,
    BaseInputItem,
    BaseOutputItem,
    BaseWorkflowInputArgs,
    BaseTemplate,
    Client,
    LiteGenerationConfig,
)
from .augment.base import InvalidTagError, DETAILED_TAG_GROUPS, calibrate_tags


def create_super_hard_generation_prompt(version: str, topics: list[str] = None, focus_tags: list[str] = None):
    """Create a prompt for generating super hard Vietnamese examples with multiple entities."""

    with open(f"src/utils/text/tagger/prompts/generate/{version}.md") as f:
        prompt = f.read().strip()

    if focus_tags is not None and len(focus_tags) > 0:
        focus_tags = " Đặc biệt lưu ý, tất cả các câu phải có các thực thể đặc biệt sau: " + ", ".join(focus_tags)
        focus_tags += ". Nên suy nghĩ cách thêm các thực thể này sao cho tự nhiên nhất."
    else:
        focus_tags = ""

    if topics is not None and len(topics) > 0:
        topic_str = "\n\n".join(["- " + t for t in topics])
        topic_section = f"""
## CÁC CHỦ ĐỀ DÙNG ĐỂ ĐẶT CÂU
Bạn hãy sử dụng các chủ đề sau để đặt câu:
{topic_str}
""".lstrip()
        topic_note = "; để tránh trùng lặp, hãy sử dụng các chủ đề đã được cung cấp ở phía dưới để đặt câu"
    else:
        topic_section = ""
        topic_note = ""

    user_message = BaseTemplate.format_text(
        prompt,
        topic_section=topic_section,
        topic_note=topic_note,
        focus_tags=focus_tags,
    )

    return [dict(role="user", content=user_message)]


def prepare_news_data():
    import numpy as np
    from collections import defaultdict
    import random

    lengths = []
    N = 100000000000
    split_by_lengths = defaultdict(list)

    with open("data/corpus-title.txt") as f:
        for line in tqdm(f):
            line = line.strip()
            # lengths.append(len(line.split()))
            L = len(line.split())
            if len(split_by_lengths[L]) < N:
                split_by_lengths[L].append(line)

    for L in [16, 18, 20]:
        print(L, len(split_by_lengths[L]))
        for v in split_by_lengths[L][:10]:
            print(v)

    if len(lengths) > 0:
        lengths = np.array(lengths)
        print(lengths.min(), lengths.max(), lengths.mean(), lengths.std())

    topics = []
    for L in [16, 18, 20]:
        topics.extend(split_by_lengths[L])
    random.seed(8686)
    random.shuffle(topics)
    with open("data/generate_8/news_topics.txt", "w") as f:
        for t in topics:
            print(t.strip(), file=f)


def select_random_topics(k: int):
    with open("data/generate_8/news_topics.txt") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]
    return random.choices(lines, k=k)


def attempt():
    client = Client("google-genai"); model = "gemini-2.5-flash" # fmt: skip
    # client = Client("google-genai"); model = "gemini-2.5-pro" # fmt: skip

    # messages = create_super_hard_generation_prompt()
    # messages = create_super_hard_generation_prompt(focus_tags=["NUMBER_RANGE", "PHONE", "LEGAL_DOC_ID"])
    # messages = create_super_hard_generation_prompt(version="v2", focus_tags=["URL", "PLATE", "ROMAN_NUMERAL"])
    messages = create_super_hard_generation_prompt(
        version="v3",
        topics=select_random_topics(10),
        focus_tags=[
            # "MEASUREMENT (lực, ví dụ như N, ...)",
            # "MEASUREMENT (cấu trúc: phân số + đơn vị)",
            # "DATE_RANGE",
            # "PLATE",
            # "ADDRESS",
            # "ROMAN_NUMERAL",
            "DIMENSION",
            "MATH_EXPR",
            "SPORT_SCORE",
            # "DATE và FRACTION (case khó ví dụ như 6/7 để phân biệt tốt hơn 2 thực thể này, 6/7 có thể là ngày, có thể là phân số tùy vào ngữ cảnh)",
        ],
    )
    # if False: # fmt: skip
    if True: # fmt: skip
        print(messages[0]["content"])
        input_tokens = client.messages._api_client._api_client.models.count_tokens(
            model=model, contents=messages[0]["content"]
        ).total_tokens
        print(input_tokens)
        print(client.cost(model, input_tokens=input_tokens, output_tokens=0))
        return

    response = client.messages.generate(model=model, messages=messages, generation_config=dict(thinking_budget=2800))
    client.pprint(response, model)
    exit(0)


def has_tag(values: list[str], tag):
    for v in values:
        if v.startswith(tag):
            return True
    return False


def generate_focus_tags(N: int, G: int):
    random.seed(8686)
    results = []
    for _ in range(N):
        tag_groups = random.choices(DETAILED_TAG_GROUPS, k=G)
        tag_groups = [random.choice(tg) for tg in tag_groups]
        results.append(tag_groups)
    return results


class InputItem(BaseInputItem):
    version: str
    topics: list[str]
    focus_tags: list[str]


class OutputItem(BaseOutputItem):
    version: str
    topics: list[str]
    focus_tags: list[str]

    sentences: list[tuple[str | None, str]] | None = None


class InputArgs(BaseWorkflowInputArgs):
    version: str
    num_generations: int
    num_focus_tags: int
    calibrate_tags: bool = True


class Generate(BaseWorkflow):
    __input_args_class__ = InputArgs
    __input_class__ = InputItem
    __output_class__ = OutputItem

    def prepare_inputs(self, input_args: InputArgs) -> list[InputItem]:  # type: ignore
        focus_tags_list = generate_focus_tags(input_args.num_generations, input_args.num_focus_tags)
        items: list[InputItem] = []

        with open("data/generate_8/news_topics.txt") as f:
            topics = [t.strip() for t in f.readlines() if t.strip() != ""]
        print("topics =", len(topics))

        C = 10
        N = input_args.num_generations
        for i in range(N):
            items.append(
                InputItem(  # type: ignore
                    allow_auto_id=True,
                    version=input_args.version,
                    topics=topics[i * C : (i + 1) * C],
                    focus_tags=focus_tags_list[i],
                )
            )

        return items

    def extract_output_content(self, content: str, input_args: InputArgs = None) -> dict:  # type: ignore
        sentences = self.ext.extract_to_lines(content, tag="output", split_lines=True, strip_asterisk=False)
        sentences = [s.strip() for s in sentences if s.strip() != ""]
        if input_args.calibrate_tags:
            new_sents = []
            for i, s in enumerate(sentences):
                try:
                    new_sents.append((None, calibrate_tags(s, return_str=True)))
                except InvalidTagError as e:
                    # print(s)
                    # raise ValueError(f"Failed at sentence {i}: {e.args}")
                    new_sents.append(("ERROR = " + e.args[0], s))
            sentences = new_sents
        return dict(sentences=sentences)


def generate(dry_run=True):
    client = Client("google-genai", use_inline_requests=False); model = "gemini-2.5-flash" # fmt: skip
    # print(client.api_client._api_client._api_client.api_key[-8:])

    g = Generate(
        config=WorkflowConfig(
            working_dir="data/generate_8/generate",
            use_batch_api=True,
            wait_for_batch_api=True,
            num_workers=16,
            generation_kwargs=dict(
                generation_config=LiteGenerationConfig(
                    thinking_budget=2800,
                ),
            ),
        ),
        client=client,
        template=create_super_hard_generation_prompt,
        task_manager=WorkflowTaskManager(
            {
                model: WorkflowTask(
                    output_file="data/generate_8/generations.jsonl",
                    finished_batch_ids=[
                        # old
                        "gembatch_8db5926c80df92383966be6e7f05df5e",
                        "gembatch_c0625418c0ed9c25b75568bdd73be2c7",
                        # new
                        "gembatch_vnwtgrfp72g9hqbfsef81i5dwaj1jod1vfe4",
                        "gembatch_b641d032504242fce414d139e2058672",
                        "gembatch_52d447c3ba96c2e82985091cb97ad7a9",
                        "gembatch_5312a164416fb10ee4c51a26d9fa7925",
                        "gembatch_6d6b4884e654177e56d4f3218435baf9",
                        # v3
                        "gembatch_lxj6h4lq4posebrdte8tiy7envflel94nk7e",
                        "gembatch_8cvk6rabjs6nmismcmp1nsmedypy2aejaoll",
                    ],
                )
            }
        ),
    )
    # g.wait_for_batch("gembatch_vnwtgrfp72g9hqbfsef81i5dwaj1jod1vfe4")
    g.run(
        model=model,
        input_args=InputArgs(version="v3", num_generations=5000, num_focus_tags=3, calibrate_tags=True),
        inputs_offset=500,
        num_input_samples=1000,
        parse_responses=True,
        parse_responses_kwargs=dict(
            ignore_input=True, dummy_input=InputItem(custom_id="dummy", version="", topics=[], focus_tags=[])
        ),
        dry_run=dry_run,
    )
    return g


def visualize(input_file: str, output_file: str, calib_tags=False):
    sentences = []
    errors = []
    with jsonlines.open(input_file) as f:
        for item in f:
            for e, s in item["sentences"]:
                if e is None:
                    if calib_tags:
                        s = calibrate_tags(s, return_str=True)
                    sentences.append(s)
                else:
                    errors.append((e, s))

    for e, s in errors[:3]:
        print(e)
        print(s)
        print()
    print("errors =", len(errors))

    with open(output_file, "w") as f:
        for s in sentences:
            print("-", s + "\n", file=f)


if __name__ == "__main__":
    # attempt()
    # prepare_news_data()
    # text = "- Mộ mẹ vua **Dục Đức** bị xới tung nghi tìm kho báu: thông tin mới nhất cho thấy kẻ gian đã đào bới khu mộ có kích thước **3x2m**[DIMENSION] vào khoảng **1h-3h**[TIME_RANGE] đêm **22/6/2024**[DATE], gây thiệt hại ước tính **1.024.125,75**[FLOAT_big] đồng và đang được các chuyên gia từ **Viện Khảo Cổ Học** đánh giá lại, đồng thời công an đã phong tỏa khu vực và kiểm tra dữ liệu camera tại **http://security.palace.vn**[URL]."
    # print(calibrate_tags(text))
    # generate(dry_run=False)
    visualize("data/generate_8/generations.jsonl", "data/generate_8/generations.md", calib_tags=False)
