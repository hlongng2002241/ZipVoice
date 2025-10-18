#!/usr/bin/env python3
import sys; sys.path.append("./src") # fmt: skip
import re
import os
import json
import time
import jsonlines
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langextract_custom.gemini import GeminiLanguageModelCustom
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
from utils.text.tagger.augment.number import Float_n_Tagger, Integer_big_Tagger, DimensionTagger
from utils.text.tagger.augment.time import TimeTagger


api_key = os.getenv("LANGEXTRACT_API_KEY")
model_id = "gemini-2.5-flash"
model = GeminiLanguageModelCustom(model_id=model_id, api_key=api_key)
# model.set_thinking_budget(2800)

VERIFI_PROMPT = """
Bạn là một chuyên gia đánh giá dữ liệu và kiểm tra xem các thực thể dưới đây đã được gán nhãn đúng hay chưa dựa theo các thẻ được định nghĩa bên dưới.
Mỗi thực thể được định dạng như sau: **text**[tag], trong đó **text** là dạng viết được in đậm, [tag] là thẻ nhãn của thực thể.
Một thực thể được coi là đúng nếu:
- *text** (dạng viết) được gán đúng nhãn [tag].

Nhiệm vụ của bạn là đánh từng thực thể trong đoạn văn vản INPUT_TEXT được cung cấp và trả lời là "Đúng" nếu điều kiện trên được thỏa mãn, và "Sai" nếu điều kiện trên sai. Trong trường hợp "Sai" hãy giải thích
Trả về kết quả theo định dạng dưới đây:
OUTPUT:
- Entity 1: **text**[tag]: 
    + Đánh giá: Đúng
    + Giải thích: ""
- Entity 2: **text**[tag]: 
    + Đánh giá: Sai
    + Giải thích: giải thích vì sao sai
......

VÍ DỤ:
INPUT_TEXT: bây giờ là **18:30**[TIME] ngày **26/08/2001**[DATE]
OUTPUT:
- Entity 1: **18:30**[TIME]
    + Đánh giá: Đúng
    + Giải thích: ""
- Entity 2: **26/08/2001**[DATE]
    + Đánh giá: Đúng
    + Giải thích: ""
......


ĐỊNH NGHĨA CÁC THẺ:
MEASUREMENT: PHẢI có số + ký hiệu đơn vị (%, km/h, °C, GB, MHz...), bao gồm cả đơn vị phức hợp có dấu "/" (tỷ lệ/thời gian), không chấp nhận đơn vị được viết đầy đủ (lít, ki lô gam, ...)
- Cấu trúc: [Số] + [Ký hiệu đơn vị] hoặc chỉ [Ký hiệu đơn vị]
- Ngữ cảnh: tốc độ xe, dung lượng file, trọng lượng, tần số, đo lường vật lý, công suất, điện áp, năng lượng, tốc độ, nhiệt độ, tỷ lệ tăng trưởng, ...
- Ví dụ: **120km/h**, **5 km/h**, **50%**, **82°C**, **3.8GHz**, **6,5%/năm**, **m**, **cm**
- Lưu ý:
    + Phân biệt với MONEY: Các giá trị tiền tệ (ví dụ: 100.000 đồng) không thuộc nhãn MEASUREMENT. Chúng có nhãn riêng là MONEY.
    + Phân biệt với NUMBER_RANGE: MEASUREMENT chỉ áp dụng cho một giá trị đơn lẻ, trong khi một khoảng giá trị (ví dụ: 10-12%) thuộc nhãn NUMBER_RANGE.
    + Phân biệt với INTEGER: Nhãn này không áp dụng cho các từ diễn tả số lượng của một vật thể hay cấp độ, như 4 xi-lanh, 6 cấp hay 5 người. Những trường hợp này nên được gán nhãn là INTEGER + đơn vị đếm, không phải MEASUREMENT
    + Phân biệt với DURATION: Không sử dụng nhãn này cho các đơn vị thời gian (ví dụ: 8,2 giây, 2 giờ).

INTEGER_big: Số nguyên lớn có dấu chấm phân cách hàng nghìn
- Cấu trúc: Số có dấu chấm cách 3 chữ số từ phải sang
- Ví dụ: **1.000.000**, **20.000**
- Lưu ý: 
    + Không có phần thập phân, không dùng dấu phẩy
    + Phân biệt với MONEY: nếu dùng trong ngữ cảnh liên quan đến tài chính thì là MONEY, nếu đi với các đơn vị đếm được không liên quan đến tài chính thì là INTEGER_big

INTEGER_n: Số nguyên thông thường không có dấu phân cách
- Cấu trúc: Số nguyên đơn giản
- Ví dụ: **123**, **45**, **113**

FLOAT_n: Số thập phân với dấu phẩy hoặc dấu chấm làm dấu thập phân
- Cấu trúc: Số có dấu phẩy hoặc dấu chấm cho phần thập phân
- Ví dụ: **12,5**, **0,75**

FLOAT_big: Số thập phân lớn có dấu chấm phân cách nghìn và dấu phẩy thập phân
- Cấu trúc: Dấu chấm cho hàng nghìn, dấu phẩy cho thập phân
- Ví dụ: **1.250,50**

FRACTION: Phân số toán học
- Cấu trúc: Định dạng Số/Số
- Ví dụ: **1/2**, **3/4**, **2/3**, **1/6**

MONEY: Số tiền với ký hiệu tiền tệ
- Cấu trúc: [Số][Ký hiệu tiền tệ] hoặc [Ký hiệu][Số] hoặc chỉ số khi ngữ cảnh là tiền
- Ví dụ: **500.000đ**, **$100**, **50000vnd**, **100.000**

PHONE: Số điện thoại định dạng Việt Nam
- Cấu trúc: Định dạng số điện thoại Việt Nam hoặc số khẩn cấp
- Ví dụ: **0123456789**, **+84123456789**, **113**

DATE: Ngày tháng năm định dạng ngày/tháng, tháng/năm, ngày/tháng/năm
- Cấu trúc: DD/MM, MM/YYYY, MM/YY, D/M hoặc DD/MM/YYYY
- Ví dụ: **15/3**, **03/2023**, **12/22**, **6-7**, 15/3/2023**, **2-12-22**, **2.9.1945**

TIME: Thời gian định dạng giờ:phút, giờ:phút:giây, giờ
- Cấu trúc: HH:MM, H:MM, HHhMM, HhMM, HH:MM:SS, H:MM:SS, HHpMMs, HpMMs
- Ví dụ: **14:30**, **9:15**, **10h30**, **8h45**, **14:30:45**, **11p50s**, **10h**

TIME_RANGE: Khoảng thời gian
- Cấu trúc: Thời gian bắt đầu - Thời gian kết thúc
- Ví dụ: **14h-15h**, **9:00-10:30**, **8h30-9h45**

EMAIL: Địa chỉ email
- Cấu trúc: Định dạng email chuẩn
- Ví dụ: **user@domain.com**, **test@gmail.com**

URL: Địa chỉ web
- Cấu trúc: Định dạng URL chuẩn
- Ví dụ: **https://example.com**, **www.google.com**

PLATE: Biển số xe
- Cấu trúc: Định dạng biển số xe Việt Nam
- Ví dụ: **30A-12345**, **51B-678.90**, **30B-11111**

ADDRESS: Địa chỉ đường phố với số nhà, tên đường, hẻm, ngõ ngách, quận
- Cấu trúc: Định dạng địa chỉ Việt Nam
- Ví dụ: **123A**, **12/45**, **67B/89**, **TP.**[ADDRESS] Thủ Đức, **Q.5**[ADDRESS] (Quận 5)

ROMAN_NUMERAL: Số La Mã
- Cấu trúc: Định dạng số La Mã
- Ví dụ: **III**, **XVI**, **XXI**

ALPHANUM_ID: Mã định danh chữ số
- Cấu trúc: Kết hợp chữ cái và số hoặc mã sản phẩm
- Ví dụ: **ABC123**, **XYZ789**, **A100**, **GPU**

MATH_EXPR: Biểu thức toán học
- Cấu trúc: Phép toán với toán tử
- Ví dụ: **2+3**, **5*7**, **10-4**

DIMENSION: Kích thước đo lường
- Cấu trúc: Số x Số với đơn vị tùy chọn
- Ví dụ: **1920x1080**, **5x7cm**

SPORT_SCORE: Điểm số trận đấu hoặc thi cử
- Cấu trúc: Định dạng điểm số với gạch ngang hoặc gạch chéo
- Ví dụ: **3-2**, **10-0**, **85/100**, **2-3**

NUMBER_RANGE: Một khoảng số có dạng "từ X đến Y", thường kèm theo đơn vị đo lường.
- Cấu trúc: [số] + [đơn vị (tùy chọn)] + [dấu gạch ngang/từ nối/khoảng trắng] + [số] + [đơn vị (tùy chọn)].
- Ví dụ: **5-10**, **100-200**, **1,5-2,5**, **1,7ha-4,5ha**, **100-200đ**, **100 200m**
- Lưu ý:
    + Toàn bộ cụm từ biểu thị khoảng số, bao gồm các con số, dấu gạch ngang và đơn vị đo lường (ví dụ: %, kg, km, ha, đ), cần được gán nhãn là một thực thể NUMBER_RANGE duy nhất.
    + Nếu chỉ có một số kèm theo ký hiệu phần trăm (ví dụ: 12%), nó thuộc về thẻ MEASUREMENT, không phải NUMBER_RANGE.

ID_NUMBER: Số chứng minh thư/căn cước, số tài khoản, mã số định danh, hoặc các dãy số dùng để nhận dạng một cá nhân hoặc một thực thể.
- Cấu trúc: Thường là một dãy số, có thể bao gồm các chữ cái.
- Ví dụ: **012345678901** (số CCCD), **123456789** (số CMND), **0987654321** (số điện thoại), **1234567890** (số tài khoản ngân hàng), **250495368** (mã số thuế).

LEGAL_DOC_ID: Mã văn bản pháp lý
- Cấu trúc: Định dạng văn bản pháp lý Việt Nam
- Ví dụ: **123/2023/NĐ-CP**, **456/QĐ-TTg**

DATE_RANGE_y_y: Khoảng năm
- Cấu trúc: Năm bắt đầu - Năm kết thúc
- Ví dụ: **2020-2023**, **1990-2000**

DATE_RANGE: Khoảng ngày tháng, tháng năm
- Cấu trúc: Ngày/tháng - Ngày/tháng/năm, Tháng - Tháng/năm
- Ví dụ: **15/3-20/3/2023**, **1/1-31/12/2023**, **3-12/2023**, **6-8/2024**

FOREIGN_WORD: Từ/cụm không thuộc từ vựng tiếng Việt, chỉ chứa chữ cái (không chứa số hoặc ký hiệu)
- Cấu trúc: Từ/cụm từ tiếng Anh, tên riêng, tên thương hiệu, tên địa danh, thuật ngữ chuyên ngành nước ngoài.
- Ví dụ: **NVIDIA**, **Apple**, **Championship**, **VinGroup**

DURATION: Khoảng thời gian, được biểu thị bằng một giá trị số và một đơn vị thời gian.
- Cấu trúc: [Số] + [Đơn vị thời gian] (dạng văn nói) hoặc [Số] + [Ký hiệu thời gian] (dạng viết).
- Ngữ cảnh: Dùng để mô tả một khoảng thời gian trôi qua, tuổi thọ, thời gian chạy của một chương trình, thời lượng một sự kiện, v.v.
- Ví dụ: **13 tháng**, **3 thập kỷ**, **3 ngày 3 đêm**, **8,2 giây**, **15 phút 30 giây**, **2 giờ**.
- Lưu ý: Nhãn này được ưu tiên cho các đơn vị thời gian. Nếu một thực thể bao gồm cả giá trị số và đơn vị thời gian mang ngữ cảnh chỉ khoảng thời gian, nó phải được gán nhãn DURATION thay vì MEASUREMENT.

Dưới đây là đoạn văn bản được cung cấp.
INPUT_TEXT:
{input_text}
"""


def format_text(text, extractions):
    prev_end_pos = None
    tmp_str = ""
    for extraction in extractions:
        extraction_class = extraction["extraction_class"]
        extraction_text = extraction["extraction_text"]
        attributes = extraction["attributes"]
        start_pos = extraction["start_pos"]
        end_pos = extraction["end_pos"]

        assert "written_form" in attributes
        written_form = attributes["written_form"]

        if prev_end_pos is None:
            prev_end_pos = 0
        assert text[start_pos:end_pos] == extraction_text

        formated_text = f"**{extraction_text}**[{extraction_class}][{written_form}]"

        tmp_str = tmp_str + text[prev_end_pos:start_pos] + formated_text
        prev_end_pos = end_pos

    tmp_str = tmp_str + text[prev_end_pos:]

    return tmp_str


def split_data_to_batch(data, batch_size):
    n_samples = len(data)
    if n_samples % batch_size == 0:
        n_batches = int(len(data) / batch_size)
    else:
        n_batches = int(len(data) / batch_size) + 1

    batches = []
    for i in range(n_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        batches.append(batch)

    return batches


def extract_entities_to_list_of_dict(input_text):
    pattern_block = r"- Entity (\d+): (.+?)\n\s*\+ Đánh giá: (Đúng|Sai)\n\s*\+ Giải thích: (.+?)(?=\n- Entity|\Z)"
    matches = re.findall(pattern_block, input_text, re.DOTALL)

    entities = []
    for match in matches:
        entity_id = int(match[0])
        entity_full = match[1].strip()
        evaluation = match[2].strip()
        explanation = match[3].strip()

        # if evaluation != "Đúng":
        #     print(match)

        entities.append({"entity_id": entity_id, "entity": entity_full, "evaluation": evaluation, "explanation": explanation})

    return entities


def compute_acc(data):
    total, correct = 0, 0
    for sample in data:
        evaluations = sample["evaluations"]

        for evaluation in evaluations:
            pred = evaluation["evaluation"]

            if pred.lower() == "đúng":
                correct += 1

            total += 1

    return correct / total


def calibrate_rule(evaluation: dict):
    float_n_tagger = Float_n_Tagger(skip_extract_unit=True, ignore_3_sep=False)
    integer_big_tagger = Integer_big_Tagger(skip_extract_unit=True, ignore_3_sep=False)
    time_tagger = TimeTagger()
    dim_tagger = DimensionTagger()

    eval_content = evaluation["explanation"]
    entity = re.search(r"\*\*(.*?)\*\*", evaluation["entity"]).group(1)

    if evaluation["entity"].upper().endswith("[ADDRESS]"):
        if (
            entity.lower() in ["tp.", "q.", "tt.", "p.", "quận", "huyện", "tỉnh", "thành phố", "xã", "phường", "thị trấn", "tp.hcm", "hcm"] # fmt: skip
            or re.fullmatch(r"\d+", entity)
        ):
            return "ADDRESS"

    elif evaluation["entity"].upper().endswith("[ALPHANUM_ID]"):
        if "FOREIGN_WORD" in evaluation["explanation"] or "viết tắt" in evaluation["explanation"] or re.fullmatch(r"\d+", entity):
            return "ALPHANUM_ID"

    elif evaluation["entity"].upper().endswith("[DATE]"):
        if re.fullmatch(r"Q\d+\/\d{4}", entity):
            return "DATE"

    elif evaluation["entity"].upper().endswith("[TIME]"):
        if "DURATION" in evaluation["explanation"] or time_tagger.validate(entity) == 0:
            return "TIME"

    elif evaluation["entity"].upper().endswith("[URL]"):
        for p in ["D:", "C:", "E:", "F:", "G:", "H:"]:
            if entity.startswith(p):
                return "URL"
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", entity):
            return "URL"

    elif evaluation["entity"].upper().endswith("[INTEGER_N]"):
        if "DURATION" in evaluation["explanation"] and re.fullmatch(r"\d+", entity):
            return "INTEGER_n"

    elif evaluation["entity"].upper().endswith("[FLOAT_N]"):
        # if float_n_tagger.validate(entity) == 0:
        if re.fullmatch(r"\d+[\.\,]\d+", entity):
            return "FLOAT_n"

    elif evaluation["entity"].upper().endswith("[INTEGER_BIG]"):
        if "MONEY" in evaluation["explanation"] and integer_big_tagger.validate(entity) == 0:
            return "INTEGER_big"

    elif evaluation["entity"].upper().endswith("[DATE_RANGE]"):
        if re.fullmatch(r"Q\d{1}\/\d{4}\s*-\s*Q\d{1}\/\d{4}", entity):
            return "DATE_RANGE"

    elif evaluation["entity"].upper().endswith("[MONEY]"):
        if re.fullmatch(r"\d+[,.0-9]+\s{0,1}đ/.*", entity):
            return "MONEY"

    elif evaluation["entity"].upper().endswith("[PHONE]"):
        if re.fullmatch(r"[+]?\d+\-\d{3}\-\d{3}\-\d{4}", entity) or re.fullmatch(r"[+]?\d{4}\-\d{3}\-\d{4}", entity):
            return "PHONE"

    elif evaluation["entity"].upper().endswith("[DIMENSION]"):
        if ("sử dụng dấu chấm" in eval_content.lower() or "dùng dấu chấm" in eval_content.lower()) and dim_tagger.validate(
            entity
        ) == 0:
            return "DIMENSION"


def compute_acc_calibrated(data, verbose=False, is_files=False):
    if isinstance(data, str):
        with jsonlines.open(data) as f:
            data = list(f)
    elif isinstance(data, list) and is_files:
        items = []
        for d in data:
            with jsonlines.open(d) as f:
                items.extend(list(f))
        data = items

    total, correct, resolve = 0, 0, 0
    wrong_tags = []
    all_tags = []
    cost = 0
    for sample in data:
        evaluations = sample["evaluations"]
        cost += sample["cost"]

        for evaluation in evaluations:
            pred = evaluation["evaluation"]
            tag = re.search(r"\[(.*?)\]", evaluation["entity"]).group(1)

            if pred.lower() == "đúng":
                correct += 1
            else:
                if calibrate_rule(evaluation):
                    resolve += 1
                    correct += 1
                else:
                    wrong_tags.append(tag)
                    if verbose:
                        print(evaluation)

            total += 1
            all_tags.append(tag)

    if verbose:
        print()
        print("Total samples:", len(data))
        print("Total cost:", cost)
        print("Cost per sample:", cost / len(data))
        print("Acc:", correct, "/", total, "=", correct / total)
        print("Number of resolved tags:", resolve)
        wrong_tags_cnt = Counter(wrong_tags)
        all_tags_cnt = Counter(all_tags)
        all_tags_cnt = {k: (v - wrong_tags_cnt[k], v) for k, v in all_tags_cnt.items()}
        all_tags_cnt = dict(sorted(all_tags_cnt.items()))
        df = []
        for k, (r, v) in all_tags_cnt.items():
            df.append((k, r, v, r / v))
        import pandas as pd

        pd.DataFrame(df, columns=["Tag", "Correct", "Total", "Acc"]).to_excel("data/generate_8/first_eval.xlsx", index=False)

    return correct / total


def run(metadata, output_filepath):
    with open(output_filepath, "w") as f:
        verified_outputs = []
        for formated_text in tqdm(metadata, desc="Verify", postfix=f"pid={os.getpid()}"):

            prompt = VERIFI_PROMPT.format(input_text=formated_text)
            try:
                output, usages = model._process_single_prompt(prompt, config={}, compute_cost=True)
            except:
                print("Overload....")
                time.sleep(10)
                continue

            lite_usages = [Client.convert_to_lite_usage_(u) for u in usages]
            cost = sum([Client.cost_(model.model_id, **u.model_dump(), provider="google-genai") for u in lite_usages])

            output = output.output
            evaluations = extract_entities_to_list_of_dict(output)

            output = {
                "formated_text": formated_text,
                "evaluations": evaluations,
                "cost": cost,
            }
            print("###cost: ", cost)
            verified_outputs.append(output)

            json_obj = json.dumps(output, ensure_ascii=False)
            f.write(json_obj + "\n")

        acc = compute_acc(verified_outputs)
        print(f"(pid={os.getpid()}): ACC: {acc}")

    print(f"(pid={os.getpid()}): Saved metadata to {output_filepath}")


class InputItem(BaseInputItem):
    messages: list | str
    # text: str


class OutputItem(BaseOutputItem):
    messages: list | str
    # text: str

    evaluations: list[dict]


class InputArgs(BaseWorkflowInputArgs):
    input_filepath: str


class Evaluate(BaseWorkflow):
    __input_args_class__ = InputArgs
    __input_class__ = InputItem
    __output_class__ = OutputItem

    def prepare_inputs(self, input_args: InputArgs) -> list[InputItem]:  # type: ignore
        metadata = open(input_args.input_filepath).readlines()
        metadata = [line for line in metadata if len(line.strip()) > 10]

        inputs = []
        base_config = None

        for formated_text in metadata:
            prompt = VERIFI_PROMPT.format(input_text=formated_text)
            inp = model._process_single_prompt(prompt, config={}, process_input_only=True)
            if base_config is None:
                base_config = inp["config"]
            else:
                assert base_config == inp["config"]
            inputs.append(
                InputItem(
                    custom_id=None,
                    allow_auto_id=True,
                    messages=[dict(role="user", content=inp["contents"])],  # text=formated_text,
                )
            )

        print("Base config:", base_config)

        return inputs

    def extract_output_content(self, content: str, input_args: BaseWorkflowInputArgs = None) -> dict:
        evaluations = extract_entities_to_list_of_dict(content)
        return dict(evaluations=evaluations)


def evaluate_batch(dry_run=True):
    client = Client("google-genai", use_inline_requests=False); model = "gemini-2.5-flash" # fmt: skip
    e = Evaluate(
        config=WorkflowConfig(
            working_dir="data/generate_8/evaluate",  # <= folder lưu batch requests và responses
            use_batch_api=True,  # <= dùng batch api thay vì mở luồng
            wait_for_batch_api=True,  # <= đợi đến khi batch api hoàn thành
            num_workers=8,  # <= số lượng workers để chạy đa luồng
        ),
        template=lambda messages: messages,
        client=client,
        task_manager=WorkflowTaskManager(
            {
                model: WorkflowTask(
                    output_file="data/generate_8/evaluations.jsonl",  # <= kết quả đánh giá được lưu ở đây
                    finished_batch_ids=[  # <= thêm batch id đã chạy vào đây
                        # "gembatch_4567c55717fe6a895c0cbc3b957d24cd",
                        "gembatch_b0a92c29db5f68f1742272157fdfb0df",
                        "gembatch_cjw5puqiuua29az4o6vvtaij02tmpr9a50vw",
                        "gembatch_r5llp8b0pssak8j04fx7xguwpduo2eg65hao",
                        "gembatch_raz5ycuhk0b1bvwrjvx20j6f31bdqmh3qvis",
                        "gembatch_7cgiaih40jjemycswwerjd21fu2ype8fgcee",
                        "gembatch_uvyatas7u8bybak5rb40kx45qr6p992mna76",
                    ],
                )
            }
        ),
        log_stdout=False,
    )
    e.run(
        model=model,
        input_args=InputArgs(
            input_filepath="data/generate_8/generations.md",  # <= input file ở đây
        ),
        inputs_offset=4000,
        num_input_samples=4000,  # <= số lượng chạy, None là chạy toàn bộ
        parse_responses=True,
        dry_run=dry_run,  # <= dry_run = False thì mới chạy
    )
    return e


def main():
    # compute_acc_calibrated(
    #     [
    #         # 0 - 200
    #         "data/generate_8/evaluation_part1/metadata.jsonl",
    #         "data/generate_8/evaluation_part2/metadata.jsonl",
    #         "data/generate_8/evaluation_part3/metadata.jsonl",
    #         "data/generate_8/evaluation_part4/metadata.jsonl",
    #         #
    #     ],
    #     verbose=True,
    #     is_files=True,
    # )
    # exit()

    # input_filepath = "/home/tuyendv/norm/human_validated.md"
    # output_dir = "/home/tuyendv/norm/dev/label_data/outputs/s2_llm_tts_verifications/"
    input_filepath = "data/generate_8/generations.md"
    output_dir = "data/generate_8/evaluation_part5"

    n_workers = 1

    # load data
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    metadata = open(input_filepath).readlines()
    metadata = [line for line in metadata if len(line.strip()) > 10]

    print(len(metadata))

    metadata = metadata[200:201]

    # split data
    n_samples_per_split = int(len(metadata) / n_workers) + 1
    splits = split_data_to_batch(data=metadata, batch_size=n_samples_per_split)

    # run
    output_filepath = "{output_dir}/metadata-{idx}.jsonl"
    params = [(split, output_filepath.format(idx=idx, output_dir=output_dir)) for idx, split in enumerate(splits)]
    with Pool(processes=n_workers) as pool:
        pool.starmap(func=run, iterable=params)

    output_filepath = f"{output_dir}/metadata.jsonl"
    with open(output_filepath, "w") as f:
        for _, filepath in params:
            for line in open(filepath).readlines():
                line = line.strip()

                f.write(line + "\n")
        print(f"###saved metadata to {output_filepath}")


if __name__ == "__main__":
    # main()
    evaluate_batch(dry_run=False)
