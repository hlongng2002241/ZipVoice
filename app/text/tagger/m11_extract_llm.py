#!/usr/bin/env python3
"""LLM-based entity extraction for text normalization tags."""

import os
import sys

sys.path.append("./src")
import textwrap
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import langextract as lx
from langextract_custom.gemini import GeminiLanguageModelCustom

from quick_utils.data_synthesis.client import Client


# Single-entity tags (excluding multi-entity tags with "__")
SINGLE_ENTITY_TAGS = [
    "MEASUREMENT",
    "DATE_dm",
    "DATE_dmy",
    "DATE_my",
    "DATE_RANGE_y_y",
    "DATE_RANGE_dm_dmy",
    "DATE_RANGE_m_my",
    "TIME_hm",
    "TIME_hms",
    "TIME_h",
    "TIME_RANGE",
    "INTEGER_n",
    "INTEGER_big",
    "FLOAT_n",
    "FLOAT_big",
    "FRACTION",
    "NUMBER_RANGE",
    "MATH_EXPR",
    "PHONE",
    "ID_NUMBER",
    "LEGAL_DOC_ID",
    "PLATE",
    "ALPHANUM_ID",
    "ROMAN_NUMERAL",
    "ADDRESS",
    "MONEY",
    "DIMENSION",
    "SPORT_SCORE",
    "EMAIL",
    "URL",
]


def create_extraction_prompt() -> str:
    return """
Trích xuất các thực thể từ văn bản tiếng Việt dựa trên các định nghĩa thẻ sau. Mỗi thực thể được đánh dấu bằng định dạng in đậm (**text**) trong văn bản gốc.

ĐỊNH NGHĨA CÁC THẺ:

MEASUREMENT: Đơn vị đo lường với số và ký hiệu đơn vị (ví dụ: **120km/h**, **25MB**, **1.2kg**, **m**, **cm**)
- Cấu trúc: [Số] + [Ký hiệu đơn vị] hoặc chỉ [Ký hiệu đơn vị]
- Dùng ký hiệu (kg, m, GB) không dùng từ đầy đủ (kilogram, mét, gigabyte)

INTEGER_big: Số nguyên lớn có dấu chấm phân cách hàng nghìn (ví dụ: **1.000.000**, **20.000**)
- Cấu trúc: Số có dấu chấm cách 3 chữ số từ phải sang
- Không có phần thập phân, không dùng dấu phẩy

INTEGER_n: Số nguyên thông thường không có dấu phân cách (ví dụ: **123**, **45**, **113**)
- Cấu trúc: Số nguyên đơn giản

FLOAT_n: Số thập phân với dấu phẩy làm dấu thập phân (ví dụ: **12,5**, **0,75**)
- Cấu trúc: Số có dấu phẩy cho phần thập phân

FLOAT_big: Số thập phân lớn có dấu chấm phân cách nghìn và dấu phẩy thập phân (ví dụ: **1.250,50**)
- Cấu trúc: Dấu chấm cho hàng nghìn, dấu phẩy cho thập phân

FRACTION: Phân số toán học (ví dụ: **1/2**, **3/4**, **2/3**, **1/6**)
- Cấu trúc: Định dạng Số/Số

MONEY: Số tiền với ký hiệu tiền tệ (ví dụ: **500.000đ**, **$100**, **50000vnd**, **100.000**)
- Cấu trúc: [Số][Ký hiệu tiền tệ] hoặc [Ký hiệu][Số] hoặc chỉ số khi ngữ cảnh là tiền
- Không có khoảng trắng giữa số và ký hiệu

PHONE: Số điện thoại định dạng Việt Nam (ví dụ: **0123456789**, **+84123456789**, **113**)
- Cấu trúc: Định dạng số điện thoại Việt Nam hoặc số khẩn cấp

DATE_dm: Ngày tháng định dạng ngày/tháng (ví dụ: **15/3**, **02/12**, **6-7**)
- Cấu trúc: DD/MM hoặc D/M

DATE_dmy: Ngày tháng năm định dạng ngày/tháng/năm (ví dụ: **15/3/2023**, **2-12-22**, **2.9.1945**)
- Cấu trúc: DD/MM/YYYY hoặc các biến thể

DATE_my: Tháng năm định dạng tháng/năm (ví dụ: **03/2023**, **12/22**)
- Cấu trúc: MM/YYYY hoặc MM/YY

TIME_hm: Thời gian định dạng giờ:phút (ví dụ: **14:30**, **9:15**, **10h30**, **8h45**)
- Cấu trúc: HH:MM, H:MM, HHhMM, HhMM

TIME_hms: Thời gian định dạng giờ:phút:giây (ví dụ: **14:30:45**, **9:15:30**, **11p50s**, **5p30s**, **9p12s**)
- Cấu trúc: HH:MM:SS, H:MM:SS, HHpMMs, HpMMs

TIME_h: Chỉ giờ (ví dụ: **14h**, **9h**, **10h**)
- Cấu trúc: Số theo sau bởi 'h'

TIME_RANGE: Khoảng thời gian (ví dụ: **14h-15h**, **9:00-10:30**, **8h30-9h45**)
- Cấu trúc: Thời gian bắt đầu - Thời gian kết thúc

EMAIL: Địa chỉ email (ví dụ: **user@domain.com**, **test@gmail.com**)
- Cấu trúc: Định dạng email chuẩn

URL: Địa chỉ web (ví dụ: **https://example.com**, **www.google.com**)
- Cấu trúc: Định dạng URL chuẩn

ID_NUMBER: Số chứng minh thư/căn cước (ví dụ: **012345678901**, **123456789**)
- Cấu trúc: Định dạng số CMND/CCCD Việt Nam

PLATE: Biển số xe (ví dụ: **30A-12345**, **51B-678.90**, **30B-11111**)
- Cấu trúc: Định dạng biển số xe Việt Nam

ADDRESS: Địa chỉ đường phố với số nhà và tên đường (ví dụ: **123A**, **12/45**, **67B/89**, **44/12**)
- Cấu trúc: Định dạng địa chỉ Việt Nam

ROMAN_NUMERAL: Số La Mã (ví dụ: **III**, **XVI**, **XXI**)
- Cấu trúc: Định dạng số La Mã

ALPHANUM_ID: Mã định danh chữ số (ví dụ: **ABC123**, **XYZ789**, **A100**, **GPU**)
- Cấu trúc: Kết hợp chữ cái và số hoặc mã sản phẩm

MATH_EXPR: Biểu thức toán học (ví dụ: **2+3**, **5*7**, **10-4**)
- Cấu trúc: Phép toán với toán tử

DIMENSION: Kích thước đo lường (ví dụ: **1920x1080**, **5x7cm**)
- Cấu trúc: Số x Số với đơn vị tùy chọn

SPORT_SCORE: Điểm số trận đấu hoặc thi cử (ví dụ: **3-2**, **10-0**, **85/100**, **2-3**)
- Cấu trúc: Định dạng điểm số với gạch ngang hoặc gạch chéo

NUMBER_RANGE: Khoảng số (ví dụ: **5-10**, **100-200**, **1,5-2,5**)
- Cấu trúc: Số đầu - Số cuối

DATE_RANGE_y_y: Khoảng năm (ví dụ: **2020-2023**, **1990-2000**)
- Cấu trúc: Năm bắt đầu - Năm kết thúc

DATE_RANGE_dm_dmy: Khoảng ngày tháng (ví dụ: **15/3-20/3/2023**, **1/1-31/12/2023**)
- Cấu trúc: Ngày/tháng - Ngày/tháng/năm

DATE_RANGE_m_my: Khoảng tháng năm (ví dụ: **3-12/2023**, **6-8/2024**)
- Cấu trúc: Tháng - Tháng/năm

LEGAL_DOC_ID: Mã văn bản pháp lý (ví dụ: **123/2023/NĐ-CP**, **456/QĐ-TTg**)
- Cấu trúc: Định dạng văn bản pháp lý Việt Nam

Trích xuất tất cả các thực thể khớp với các mẫu này và phân loại chúng với thẻ phù hợp. Với mỗi thực thể, trả về attribute "spoken_form" của nó (dạng văn nói tiếng Việt).
""".strip()


def create_examples() -> List[lx.data.ExampleData]:
    """Create example data for training the extraction model."""
    return [
        lx.data.ExampleData(
            text="tôi lái xe tốc độ 120km/h với trọng lượng 1.500kg.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="MEASUREMENT",
                    extraction_text="120km/h",
                    attributes={"spoken_form": "một trăm hai mươi ki lô mét trên giờ"},
                ),
                lx.data.Extraction(
                    extraction_class="MEASUREMENT",
                    extraction_text="1.500kg",
                    attributes={"spoken_form": "một nghìn năm trăm ki lô gam"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="giải thưởng trị giá 1.000.000đ được trao vào ngày 15/3/2023 lúc 14:30h số 12A/123 Phan Bội Châu",
            extractions=[
                lx.data.Extraction(
                    extraction_class="MONEY", extraction_text="1.000.000đ", attributes={"spoken_form": "một triệu đồng"}
                ),
                lx.data.Extraction(
                    extraction_class="DATE_dmy",
                    extraction_text="15/3/2023",
                    attributes={"spoken_form": "ngày mười lăm tháng ba năm hai không hai ba"},
                ),
                lx.data.Extraction(
                    extraction_class="TIME_h",
                    extraction_text="14:30h",
                    attributes={"spoken_form": "mười bốn giờ ba mươi phút"},
                ),
                lx.data.Extraction(
                    extraction_class="ADDRESS",
                    extraction_text="12A/123",
                    attributes={"spoken_form": "số mười hai a trên một trăm hai ba"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Liên hệ email support@company.com hoặc số điện thoại 0123456789 vào khung giờ 14h-15h.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="EMAIL",
                    extraction_text="support@company.com",
                    attributes={"spoken_form": "sắp pót a còng com pa ni chấm com"},
                ),
                lx.data.Extraction(
                    extraction_class="PHONE",
                    extraction_text="0123456789",
                    attributes={"spoken_form": "không một hai ba bốn năm sáu bảy tám chín"},
                ),
                lx.data.Extraction(
                    extraction_class="TIME_RANGE",
                    extraction_text="14h-15h",
                    attributes={"spoken_form": "mười bốn giờ đến mười lăm giờ"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="tôi dành 1/6 số tiền tích cóp để đi chơi vào 1/6",
            extractions=[
                lx.data.Extraction(
                    extraction_class="DATE_dm", 
                    extraction_text="1/6", 
                    attributes={"spoken_form": "ngày mùng một tháng sáu"}
                ),
                lx.data.Extraction(
                    extraction_class="FRACTION",
                    extraction_text="1/6",
                    attributes={"spoken_form": "một phần sáu"},
                ),
            ],
        ),
    ]


class EntityExtractor:
    """LLM-based entity extractor for Vietnamese text normalization."""

    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: str = None):
        """Initialize the extractor with a language model."""
        if api_key is None:
            api_key = os.getenv("LANGEXTRACT_API_KEY")

        self.model = GeminiLanguageModelCustom(model_id=model_id, api_key=api_key)
        self.model.set_thinking_budget(thinking_budget=1600)
        self.prompt = create_extraction_prompt()
        self.examples = create_examples()

    def extract(self, text: str, extraction_passes: int = 1) -> tuple:
        """Extract entities from text and return results with usage info."""
        try:
            result, usages = self.model.lx_extract(
                text_or_documents=text,
                prompt_description=self.prompt,
                examples=self.examples,
                extraction_passes=extraction_passes,
            )

            # Calculate cost
            lite_usages = [Client.convert_to_lite_usage_(u) for u in usages]
            cost = sum([Client.cost_(self.model.model_id, **u.model_dump(), provider="google-genai") for u in lite_usages])

            return result, usages, cost

        except Exception as e:
            print(f"Extraction error: {e}")
            return None, [], 0.0

    def extract_by_tag(self, text: str, target_tags: List[str] = None) -> Dict[str, List[str]]:
        """Extract entities and group them by tag type."""
        result, _, _ = self.extract(text)

        if result is None:
            return {}

        grouped = {}
        for extraction in result.extractions:
            tag = extraction.extraction_class
            if target_tags is None or tag in target_tags:
                if tag not in grouped:
                    grouped[tag] = []
                grouped[tag].append(extraction.extraction_text)

        return grouped


def test_extractor():
    """Test the entity extractor with sample Vietnamese text."""
    extractor = EntityExtractor()

    test_texts = [
        "thằng cha xe 30B-11111 phóng nhanh thật, 2-3 cảnh sát thấy mà ko cản được, cũng phải 100km/s chứ ko đùa, ăn đứt kỷ lục 9p12s đấy. t cá 1/2 bát phở, m thua thì phải đưa t 100.000 đồng đấy, hoặc 1 con GPU A100 cũng ngon. Trốn là t gọi 113 đấy, có mười mày cũng ko chạy được. nhà m ở 44/12 hồ tây, t biết rồi nhé, hạn là 6/7 đấy.",
        #
        "hôm qua 15/8/2023 lúc 14h30 tao đi mua RTX 4090 giá 35.000.000đ ở 123A/456 Trần Hưng Đạo, thấy biển số 51F-12345 đậu trước cửa. Chủ shop nói máy chạy được 144fps ở độ phân giải 3840x2160px, bảo hành 36 tháng từ 1/9-31/12/2026. Tao test benchmark 3DMark được điểm 15.678, nhưng nhiệt độ lên tới 82°C sau 2h15p chạy. Email liên hệ là support@nvidia.com, hotline 1900-123-456.",
        #
        "mấy đứa nhỏ đá bóng sân 7A-11 từ 16h-18h30, tỷ số 5-3 team áo đỏ thắng. Sau đó đi ăn phở 45.000 đồng/tô ở quán 67/89 Nguyễn Du, uống thêm 2,5l nước ngọt. Về nhà lúc 20h45p, xem tin tức thấy vàng tăng lên 68.500.000đ/lượng, USD giảm về 24.125 VNĐ. Check email thấy hóa đơn điện 1.250.000đ tháng 7/2023, tiêu thụ 456kWh. Ngày mai 16/8 phải nộp thuế theo QĐ 123/2023/NĐ-CP.",
        #
        "đêm qua 3h15p sáng nghe tiếng động lạ, ra xem thấy xe máy 29H-98765 đậu dưới sân. CPU của máy tính chạy ở tần số 3,8GHz, RAM 32GB DDR5, ổ cứng SSD 2TB. Download file 4,2GB mất 15p30s với tốc độ 5MB/s. Wifi password là ABC123def, IP address 192.168.1.1. Hôm nay 2/9 phải chuyển khoản 2.500.000đ vào STK 0123456789 VCB trước 17h. Tỷ giá EUR/VND là 25.850, Bitcoin ở mức $26.500.",
        #
        "cuối tuần 25-26/11/2023 đi Đà Lạt, ở khách sạn 12B/34C Hùng Vương. Giá phòng 1.200.000đ/đêm, diện tích 25m², view hồ Xuân Hương. Thuê xe máy SH 125cc biển 47A-88899 giá 300.000đ/ngày. Đi chợ mua đặc sản: café 250.000đ/kg, mứt dâu 150.000đ/hộp, rượu vang 680.000đ/chai. Nhiệt độ ban ngày 22-25°C, đêm xuống 15-18°C. Check-in Facebook lúc 10h30, có 157 likes và 23 comments. Tài xế Grab tên Minh số 0987654321 rất nhiệt tình.",
        #
        "startup công nghệ gọi vốn series A 15.5 triệu USD từ quỹ đầu tư, định giá 75 triệu USD. Văn phòng thuê tầng 8-12 tòa nhà 88/90 Điện Biên Phủ, diện tích 1.500m², giá thuê 45USD/m²/tháng. Team dev 25 người, designer 8 người, QA/QC 12 người. Sản phẩm mobile app có 2.3 triệu users, rating 4,7/5 sao trên AppStore. Server AWS chi phí $12.500/tháng, bandwidth 500GB/ngày. Domain .com giá $15/năm, SSL certificate $199/năm. Hợp đồng B2B ký với 156 doanh nghiệp, doanh thu Q3/2023 đạt 45,8 tỷ đồng.",
    ]

    for i, text in enumerate(test_texts[2:3], 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Input: {text}")

        result, usages, cost = extractor.extract(text)
        print(result)

        if result:
            print("\nExtractions:")
            for extraction in result.extractions:
                print(f"  - {extraction.extraction_class}: '{extraction.extraction_text}' | {extraction.attributes}")

            print(usages)

            print(f"\nUsage cost: ${cost:.6f}")
        else:
            print("No extractions found or error occurred.")


if __name__ == "__main__":
    test_extractor()
