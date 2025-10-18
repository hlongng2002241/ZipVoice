TAG_TO_PROMPT: dict[str, str] = {}

# ============================================= #

TAG_TO_PROMPT["MEASUREMENT"] = """
# YÊU CẦU SINH DỮ LIỆU ĐO LƯỜNG

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin đo lường. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo cấu trúc: `[Số] (tùy chọn) + [Ký hiệu Đơn vị] (bắt buộc)`
* **Số**: Có thể xuất hiện hoặc không.
* **Ký hiệu Đơn vị**: Luôn luôn phải có mặt.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `100`, `2.5`, `0.5`).
    * KHÔNG DÙNG: Số viết bằng chữ (ví dụ: ~~một trăm~~, ~~hai phẩy năm~~).
* **ĐƠN VỊ**: Bắt buộc phải là **ký hiệu** quốc tế (ví dụ: `kg`, `m`, `GB`, `km/h`).
    * KHÔNG DÙNG: Tên đơn vị viết đầy đủ (ví dụ: ~~ki-lô-gam~~, ~~mét~~, ~~gigabyte~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, và các từ xung quanh làm nổi bật ý nghĩa đo lường.
* Không cần một từ khóa cố định nào.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** biểu thị một **khoảng giá trị**.
    * KHÔNG DÙNG: Các dạng như `5-7kg`, `từ 1.5m đến 2m`, `10 - 15 km`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Định dạng Số
* **ĐÚNG**: Tốc độ tối đa là **120km/h**.
* **SAI**: Tốc độ tối đa là **một trăm hai mươi km/h**. (Lý do: Số viết bằng chữ)

### Về Định dạng Đơn vị
* **ĐÚNG**: Vui lòng gửi file dưới **25MB**.
* **SAI**: Vui lòng gửi file dưới **25 megabyte**. (Lý do: Đơn vị viết đầy đủ)

### Về Khoảng Giá trị
* **ĐÚNG**: Nhiệt độ phòng là **26°C**.
* **SAI**: Nhiệt độ phòng là **25-27°C**. (Lý do: Sử dụng khoảng giá trị)

### Về Trường hợp không có Số
* **ĐÚNG**: Bạn muốn đo bằng **m** hay **cm**?
* **SAI**: Bạn muốn đo bằng **mét** hay **cm**? (Lý do: Đơn vị viết đầy đủ)

### Về Lỗi Kết hợp
* **ĐÚNG**: Quả bưởi nặng **1.2kg**.
* **SAI**: Quả bưởi nặng **một phẩy hai ki-lô-gam**. (Lý do: Sai cả định dạng số và đơn vị)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Quả bưởi nặng **1.2kg**.
* **SAI**: Quả bưởi nặng 1.2kg. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DATE_dm"] = """
# YÊU CẦU SINH DỮ LIỆU NGÀY THÁNG

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin ngày và tháng. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Ngày]/[Tháng]`
* **Ngày, Tháng**: Luôn phải có mặt.
* **Dấu phân cách**: Bắt buộc phải là dấu gạch chéo (`/`).

### 2. Quy Tắc Định Dạng
* **NGÀY và THÁNG**: Bắt buộc phải là **chữ số** và là ngày tháng hợp lệ (ví dụ: `1/5`, `29/2`, `31/12`).
    * KHÔNG DÙNG: Ngày tháng viết bằng chữ (ví dụ: ~~mồng một tháng năm~~).
* **NĂM**: **TUYỆT ĐỐI KHÔNG** được thêm năm vào sau (ví dụ: `1/5/2025`).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, và các từ xung quanh làm nổi bật ý nghĩa về thời gian, lịch hẹn, sự kiện.
* Không cần một từ khóa cố định nào.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** biểu thị một **khoảng thời gian**.
    * KHÔNG DÙNG: Các dạng như `1/5-5/5`, `từ 2/9 đến 10/9`.
* **TUYỆT ĐỐI KHÔNG** viết đầy đủ kiểu "ngày [số] tháng [số]".
    * KHÔNG DÙNG: `ngày 1 tháng 5`. Chỉ dùng `1/5`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc (Dấu Phân Cách)
* **ĐÚNG**: Hạn chót nộp hồ sơ là **15/8**.
* **SAI**: Hạn chót nộp hồ sơ là **15-8**. (Lý do: Sai dấu phân cách, phải là `/`)

### Về Việc Thêm Năm
* **ĐÚNG**: Sự kiện sẽ diễn ra vào **2/9** tại Hội trường Thống nhất.
* **SAI**: Sự kiện sẽ diễn ra vào **2/9/2025** tại Hội trường Thống nhất. (Lý do: Chứa thông tin năm, không đúng cấu trúc yêu cầu)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Hẹn gặp lại cậu vào **10/10** nhé.
* **SAI**: Hẹn gặp lại cậu vào **mùng 10 tháng 10** nhé. (Lý do: Dùng chữ thay vì định dạng `Số/Số`)

### Về Khoảng Thời Gian
* **ĐÚNG**: Chúng ta cần hoàn thành trước ngày **30/4**.
* **SAI**: Chúng ta cần hoàn thành trong khoảng **25/4-30/4**. (Lý do: Biểu thị một khoảng thời gian)

### Về Lỗi Kết Hợp
* **ĐÚNG**: Vui lòng xác nhận tham dự trước **1/6**.
* **SAI**: Vui lòng xác nhận tham dự trước ngày **một tháng sáu**. (Lý do: Sai cả cấu trúc và định dạng số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Vui lòng xác nhận tham dự trước **1/6**.
* **SAI**: Vui lòng xác nhận tham dự trước 1/6. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DATE_dmy"] = """
# YÊU CẦU SINH DỮ LIỆU NGÀY THÁNG NĂM ĐẦY ĐỦ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin ngày tháng năm đầy đủ. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Ngày]/[Tháng]/[Năm]`
* **Ngày, Tháng, Năm**: Luôn phải có mặt đầy đủ.
* **Dấu phân cách**: Bắt buộc phải là dấu gạch chéo (`/`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `2/9/1945`, `05/09/2025`). Có thể có số 0 đứng trước cho ngày và tháng có một chữ số.
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~ngày hai tháng chín năm một chín bốn lăm~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, và các từ xung quanh làm nổi bật ý nghĩa về một mốc thời gian cụ thể.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** bỏ bớt bất kỳ thành phần nào (ngày, tháng, hoặc năm).
* **TUYỆT ĐỐI KHÔNG** sử dụng dấu phân cách khác ngoài `/` (ví dụ: ~~02-09-1945~~, ~~02.09.1945~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Định Dạng Đầy Đủ
* **ĐÚNG**: Hợp đồng sẽ hết hạn vào ngày **31/12/2026**.
* **SAI**: Hợp đồng sẽ hết hạn vào ngày **31/12**. (Lý do: Thiếu thông tin năm, không đúng cấu trúc yêu cầu)

### Về Dấu Phân Cách
* **ĐÚNG**: Ngày khai giảng năm học mới là **05/09/2025**.
* **SAI**: Ngày khai giảng năm học mới là **05.09.2025**. (Lý do: Sai dấu phân cách, phải là `/`)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Nước Việt Nam Dân chủ Cộng hòa ra đời ngày **02/09/1945**.
* **SAI**: Nước Việt Nam Dân chủ Cộng hòa ra đời **ngày 2 tháng 9 năm 1945**. (Lý do: Dùng chữ viết và cấu trúc diễn giải thay vì định dạng `Số/Số/Số`)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Nước Việt Nam Dân chủ Cộng hòa ra đời ngày **02/09/1945**.
* **SAI**: Nước Việt Nam Dân chủ Cộng hòa ra đời ngày 02/09/1945. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DATE_my"] = """
# YÊU CẦU SINH DỮ LIỆU THÁNG VÀ NĂM

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin tháng và năm. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Tháng]/[Năm]`
* **Tháng, Năm**: Luôn phải có mặt.
* **Dấu phân cách**: Bắt buộc phải là dấu gạch chéo (`/`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `4/2025`, `12/2027`).
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~tháng tư năm hai không hai lăm~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, làm nổi bật ý nghĩa về một kỳ, giai đoạn, quý, hoặc một tháng cụ thể trong năm.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thêm ngày vào trước (ví dụ: ~~01/04/2025~~).
* **TUYỆT ĐỐI KHÔNG** sử dụng dấu phân cách khác ngoài `/` (ví dụ: ~~4-2025~~).
* **TUYỆT ĐỐI KHÔNG** biểu thị một khoảng thời gian (ví dụ: ~~4/2025 - 6/2025~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Báo cáo tài chính quý này sẽ chốt số liệu vào **3/2025**.
* **SAI**: Báo cáo tài chính quý này sẽ chốt số liệu vào ngày **31/3/2025**. (Lý do: Chứa thông tin ngày, không đúng cấu trúc `Tháng/Năm`)

### Về Dấu Phân Cách
* **ĐÚNG**: Thẻ tín dụng của bạn sẽ hết hạn vào **12/2027**.
* **SAI**: Thẻ tín dụng của bạn sẽ hết hạn vào **12-2027**. (Lý do: Sai dấu phân cách, phải là `/`)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Chúng ta cần nỗ lực hơn nữa trong quý **4/2025**.
* **SAI**: Chúng ta cần nỗ lực hơn nữa trong **quý IV năm 2025**. (Lý do: Dùng chữ La Mã và cấu trúc diễn giải thay vì định dạng `Số/Số`)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Chúng ta cần nỗ lực hơn nữa trong quý **4/2025**.
* **SAI**: Chúng ta cần nỗ lực hơn nữa trong quý 4/2025. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DATE_RANGE_y_y"] = """
# YÊU CẦU SINH DỮ LIỆU KHOẢNG NĂM

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin về một khoảng thời gian kéo dài nhiều năm. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Năm bắt đầu]-[Năm kết thúc]`
* **Năm bắt đầu, Năm kết thúc**: Luôn phải có mặt.
* **Dấu phân cách**: Bắt buộc phải là dấu gạch nối (`-`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** có 4 chữ số (ví dụ: `2021-2025`). `Năm kết thúc` phải lớn hơn `Năm bắt đầu`.
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~hai không hai mốt đến hai không hai lăm~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả một giai đoạn, nhiệm kỳ, kế hoạch, năm học, mùa giải... kéo dài qua nhiều năm.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thêm tháng hoặc ngày vào (ví dụ: ~~1/2021-1/2025~~).
* **TUYỆT ĐỐI KHÔNG** sử dụng các từ nối như `đến`, `tới` thay cho dấu `-`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Kế hoạch phát triển kinh tế giai đoạn **2021-2025** đã được phê duyệt.
* **SAI**: Kế hoạch phát triển kinh tế giai đoạn **năm 2021 đến năm 2025** đã được phê duyệt. (Lý do: Dùng từ nối thay vì dấu `-`)

### Về Thành Phần
* **ĐÚNG**: Nhiệm kỳ của chủ tịch là **2020-2025**.
* **SAI**: Nhiệm kỳ của chủ tịch là **1/2020-1/2025**. (Lý do: Chứa thông tin tháng, không đúng cấu trúc `Năm-Năm`)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Những năm tháng đại học **2018-2022** thật đáng nhớ.
* **SAI**: Những năm tháng đại học từ **hai không mười tám đến hai không hai hai** thật đáng nhớ. (Lý do: Sai định dạng số và cấu trúc)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Những năm tháng đại học **2018-2022** thật đáng nhớ.
* **SAI**: Những năm tháng đại học 2018-2022 thật đáng nhớ. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DATE_RANGE_dm_dmy"] = """
# YÊU CẦU SINH DỮ LIỆU KHOẢNG NGÀY THÁNG (CÓ NĂM)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một khoảng thời gian xác định từ ngày/tháng đến ngày/tháng/năm. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Ngày]/[Tháng]-[Ngày]/[Tháng]/[Năm]`
* **Mốc bắt đầu (d/m), Mốc kết thúc (d/m/y)**: Luôn phải có mặt đầy đủ.
* **Dấu phân cách**: Dấu gạch chéo (`/`) cho ngày tháng, dấu gạch nối (`-`) cho khoảng thời gian.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**. Mốc thời gian phải hợp lệ.
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~hai mươi tháng một đến hai hai tháng hai~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả một lịch trình, sự kiện, chương trình kéo dài từ một ngày trong quá khứ/hiện tại đến một ngày cụ thể trong tương lai.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thay đổi thứ tự các thành phần.
* **TUYỆT ĐỐI KHÔNG** sử dụng các từ nối như `đến`, `tới` thay cho dấu `-`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Chuỗi sự kiện "Chào Xuân" sẽ diễn ra từ **20/1-22/2/2025**.
* **SAI**: Chuỗi sự kiện "Chào Xuân" sẽ diễn ra từ **20/01 đến 22/02/2025**. (Lý do: Dùng từ nối thay vì dấu `-`)

### Về Thành Phần
* **ĐÚNG**: Chuyến đi Vũng Tàu của chúng tôi dự kiến là **4/5-6/5/2025**.
* **SAI**: Chuyến đi Vũng Tàu của chúng tôi dự kiến là **4/5-6/5**. (Lý do: Mốc kết thúc thiếu năm)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Chương trình khuyến mãi áp dụng **1/8-1/9/2025**.
* **SAI**: Chương trình khuyến mãi áp dụng từ **mùng 1 tháng 8 đến mùng 1 tháng 9 năm 2025**. (Lý do: Sai định dạng số và cấu trúc)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Chương trình khuyến mãi áp dụng **1/8-1/9/2025**.
* **SAI**: Chương trình khuyến mãi áp dụng 1/8-1/9/2025. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DATE_RANGE_m_my"] = """
# YÊU CẦU SINH DỮ LIỆU KHOẢNG THÁNG NĂM

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một khoảng thời gian xác định từ tháng/năm đến tháng/năm. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Tháng]/[Năm]-[Tháng]/[Năm]`
* **Mốc bắt đầu (m/y), Mốc kết thúc (m/y)**: Luôn phải có mặt. Mốc kết thúc phải bằng hoặc sau mốc bắt đầu.
* **Dấu phân cách**: Dấu gạch chéo (`/`) cho tháng năm, dấu gạch nối (`-`) cho khoảng thời gian.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**.
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~tháng một năm hai tư đến tháng hai năm hai lăm~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả một giai đoạn, kế hoạch, chương trình, thời hạn, hợp đồng... kéo dài nhiều tháng.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thêm ngày vào (ví dụ: ~~1/1/2024-1/2/2025~~).
* **TUYỆT ĐỐI KHÔNG** sử dụng các từ nối như `đến`, `tới` thay cho dấu `-`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Kế hoạch phát triển sản phẩm giai đoạn **9/2025-6/2026** đã được phê duyệt.
* **SAI**: Kế hoạch phát triển sản phẩm giai đoạn **từ 9/2025 đến 6/2026** đã được phê duyệt. (Lý do: Dùng từ nối thay vì dấu `-`)

### Về Thành Phần
* **ĐÚNG**: Hợp đồng có hiệu lực từ **3/2023-3/2026**.
* **SAI**: Hợp đồng có hiệu lực từ **3/2023**. (Lý do: Thiếu mốc kết thúc)

### Về Lỗi Kết Hợp
* **ĐÚNG**: Thời hạn bảo hành là **1/2025-1/2027**.
* **SAI**: Thời hạn bảo hành là từ **tháng một 2025 tới tháng một 2027**. (Lý do: Sai định dạng số và cấu trúc)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Thời hạn bảo hành là **1/2025-1/2027**.
* **SAI**: Thời hạn bảo hành là 1/2025-1/2027. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["TIME_hm"] = """
# YÊU CẦU SINH DỮ LIỆU GIỜ PHÚT

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin về một mốc thời gian giờ và phút trong ngày. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Giờ]:[Phút]`
* **Giờ, Phút**: Luôn phải có mặt.
* **Dấu phân cách**: Bắt buộc phải là dấu hai chấm (`:`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `10:30`, `6:59`, `07:05`).
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~mười giờ ba mươi~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả một thời điểm cụ thể trong ngày như lịch hẹn, giờ giấc, thời gian biểu.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thêm giây vào sau (ví dụ: ~~10:30:15~~).
* **TUYỆT ĐỐI KHÔNG** sử dụng các từ nối như `giờ`, `phút` trong cấu trúc.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Cuộc họp sẽ bắt đầu lúc **9:00**.
* **SAI**: Cuộc họp sẽ bắt đầu lúc **9 giờ 00 phút**. (Lý do: Dùng từ diễn giải thay vì cấu trúc `Số:Số`)

### Về Dấu Phân Cách
* **ĐÚNG**: Tôi đến nơi đúng **6:59**, may quá!
* **SAI**: Tôi đến nơi đúng **6.59**, may quá! (Lý do: Sai dấu phân cách, phải là `:`)

### Về Thành Phần
* **ĐÚNG**: Tôi cho bạn đến **10:30** để hoàn thành công việc.
* **SAI**: Tôi cho bạn đến **10:30:00** để hoàn thành công việc. (Lý do: Chứa thông tin giây, không đúng cấu trúc)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Tôi cho bạn đến **10:30** để hoàn thành công việc.
* **SAI**: Tôi cho bạn đến 10:30 để hoàn thành công việc. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["TIME_hms"] = """
# YÊU CẦU SINH DỮ LIỆU GIỜ PHÚT GIÂY

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin về một mốc thời gian chính xác đến từng giây. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Giờ]:[Phút]:[Giây]`
* **Giờ, Phút, Giây**: Luôn phải có mặt đầy đủ.
* **Dấu phân cách**: Bắt buộc phải là dấu hai chấm (`:`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `10:30:15`, `06:59:00`).
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~mười giờ ba mươi phút mười lăm giây~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả một thời điểm cực kỳ chính xác, thường liên quan đến hệ thống máy tính, thi đấu thể thao, sự kiện khoa học.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** bỏ bớt bất kỳ thành phần nào.
* **TUYỆT ĐỐI KHÔNG** sử dụng dấu phân cách khác.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Định Dạng Đầy Đủ
* **ĐÚNG**: Vận động viên về đích với thành tích **09:58:32**.
* **SAI**: Vận động viên về đích với thành tích **09:58**. (Lý do: Thiếu thông tin giây)

### Về Dấu Phân Cách
* **ĐÚNG**: Hệ thống sẽ đóng vào lúc **10:30:15**.
* **SAI**: Hệ thống sẽ đóng vào lúc **10.30.15**. (Lý do: Sai dấu phân cách, phải là `:`)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Đồng hồ điện tử nhảy sang **6:59:00** đúng lúc tôi đến.
* **SAI**: Đồng hồ điện tử nhảy sang **sáu giờ năm mươi chín phút không giây** đúng lúc tôi đến. (Lý do: Sai định dạng số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Đồng hồ điện tử nhảy sang **6:59:00** đúng lúc tôi đến.
* **SAI**: Đồng hồ điện tử nhảy sang 6:59:00 đúng lúc tôi đến. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["TIME_h"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ GIỜ (KHOẢNG THỜI GIAN)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin về một khoảng thời gian được đo bằng giờ. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Số]h`
* **Số**: Luôn phải có mặt, có thể là số nguyên hoặc thập phân.
* **Ký hiệu 'h'**: Luôn phải có mặt, viết liền sau số.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `3h`, `1.5h`, `10h`).
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~ba giờ~~, ~~mười tiếng~~).
* **KÝ HIỆU**: Bắt buộc phải là `h`.
    * KHÔNG DÙNG: Các từ như `giờ`, `tiếng` (ví dụ: ~~3 giờ~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả độ dài của một hành động, sự kiện hoặc một mốc thời gian (deadline).

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** tách rời số và chữ `h` (ví dụ: ~~3 h~~).
* **TUYỆT ĐỐI KHÔNG** biểu thị một mốc thời gian có phút (ví dụ: ~~10:30~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Chuyến bay này kéo dài khoảng **3h**.
* **SAI**: Chuyến bay này kéo dài khoảng **3 giờ**. (Lý do: Dùng từ "giờ" thay vì ký hiệu `h`)

### Về Ký Hiệu
* **ĐÚNG**: Bạn phải hoàn thành công việc trước **10h** sáng nay.
* **SAI**: Bạn phải hoàn thành công việc trước **10 giờ** sáng nay. (Lý do: Sai ký hiệu đơn vị)

### Về Định Dạng Viết Liền
* **ĐÚNG**: Tôi đã ngủ một giấc **1.5h**.
* **SAI**: Tôi đã ngủ một giấc **1.5 h**. (Lý do: Có khoảng trắng giữa số và ký hiệu `h`)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Bạn phải hoàn thành công việc trước **10h** sáng nay.
* **SAI**: Bạn phải hoàn thành công việc trước 10h sáng nay. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["TIME_RANGE"] = """
# YÊU CẦU SINH DỮ LIỆU KHOẢNG THỜI GIAN (GIỜ:PHÚT)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một khoảng thời gian trong ngày (từ giờ:phút đến giờ:phút). Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Giờ]:[Phút]-[Giờ]:[Phút]`
* **Mốc bắt đầu, Mốc kết thúc**: Luôn phải có mặt. Mốc kết thúc phải sau mốc bắt đầu.
* **Dấu phân cách**: Dấu hai chấm (`:`) cho giờ phút, dấu gạch nối (`-`) cho khoảng thời gian.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**.
    * KHÔNG DÙNG: Viết bằng chữ.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả một khung giờ làm việc, cuộc họp, giờ mở cửa, lịch trình sự kiện...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thêm giây vào (ví dụ: ~~9:00:00-10:30:00~~).
* **TUYỆT ĐỐI KHÔNG** sử dụng các từ nối như `đến`, `tới` thay cho dấu `-`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Cuộc họp hôm nay sẽ diễn ra trong khoảng **9:00-10:30**.
* **SAI**: Cuộc họp hôm nay sẽ diễn ra từ **9 giờ đến 10 giờ 30**. (Lý do: Dùng từ nối và diễn giải thay vì cấu trúc)

### Về Dấu Phân Cách
* **ĐÚNG**: Giờ nghỉ trưa của công ty là **12:00-13:00**.
* **SAI**: Giờ nghỉ trưa của công ty là **12:00 đến 13:00**. (Lý do: Dùng từ "đến" thay vì dấu `-`)

### Về Thành Phần
* **ĐÚNG**: Cửa hàng tạm đóng cửa để kiểm kê kho lúc **13:00-14:30**.
* **SAI**: Cửa hàng tạm đóng cửa để kiểm kê kho lúc **13:00**. (Lý do: Thiếu mốc thời gian kết thúc)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Cửa hàng tạm đóng cửa lúc **13:00-14:30**.
* **SAI**: Cửa hàng tạm đóng cửa lúc 13:00-14:30. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["INTEGER_n"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ NGUYÊN ĐƠN GIẢN

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một số nguyên đơn giản (không có dấu phân cách hàng nghìn). Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa: `[Số Nguyên]`
* **Số Nguyên**: Một chuỗi chữ số liên tục, có thể có dấu âm (`-`) ở đầu.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `30`, `20`, `-1`).
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~ba mươi~~).
* **DẤU PHÂN CÁCH**: **TUYỆT ĐỐI KHÔNG** chứa dấu phân cách hàng nghìn (`.`) hoặc dấu thập phân (`,`).
* **ĐƠN VỊ**: Nếu có đơn vị đi cùng số thì đơn vị phải được viết dưới dạng **chữ đầy đủ** chứ không phải dạng ký hiệu. Đơn vị không được nằm trong cấu trúc cốt lõi được in đậm.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi (phần số) trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, sử dụng số trong các ngữ cảnh đếm thông thường (số lượng, số tầng, số thứ tự...).

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** dùng số có dấu phân cách (ví dụ: ~~1.000~~).
* **TUYỆT ĐỐI KHÔNG** dùng số thập phân (ví dụ: ~~20.5~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dấu Phân Cách Hàng Nghìn
* **ĐÚNG**: Có hơn **1000** người tham gia.
* **SAI**: Có hơn **1.000** người tham gia. (Lý do: Chứa dấu phân cách hàng nghìn)

### Về Dấu Thập Phân
* **ĐÚNG**: Tòa nhà này có **20** tầng.
* **SAI**: Tòa nhà này cao **20.0** mét. (Lý do: Là số thập phân, không phải số nguyên)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Lớp tôi có **30** học sinh.
* **SAI**: Lớp tôi có **ba mươi** học sinh. (Lý do: Dùng chữ viết thay vì số)

### Về đơn vị đi cùng (nếu có)
* **ĐÚNG**: Dải tần đáp ứng là **20** héc.
* **SAI**: Dải tần đáp ứng là **20** Hz. (Lý do: Dùng đơn vị ở dạng ký hiệu)
* **SAI**: Dải tần đáp ứng là **20 héc**. (Lý do: In đậm các từ khác ngoài số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Tòa nhà này có **20** tầng.
* **SAI**: Tòa nhà này có 20 tầng. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["INTEGER_big"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ NGUYÊN LỚN (DẤU CHẤM)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một số nguyên lớn, sử dụng dấu chấm (`.`) làm dấu phân cách hàng nghìn. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `([Số]\.)+[Số]{3}`
* **Số**: Một chuỗi chữ số.
* **Dấu phân cách**: Bắt buộc phải là dấu chấm (`.`) để ngăn cách mỗi 3 chữ số từ phải qua trái.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `1.000`, `2.000.000`).
    * KHÔNG DÙNG: Viết bằng chữ.
* **DẤU PHÂN CÁCH**: Bắt buộc là dấu chấm (`.`).
    * KHÔNG DÙNG: Dấu phẩy (`,`) hoặc khoảng trắng.
* **ĐƠN VỊ**: Nếu có đơn vị đi cùng số thì đơn vị phải được viết dưới dạng **chữ đầy đủ** chứ không phải dạng ký hiệu. Đơn vị không được nằm trong cấu trúc cốt lõi được in đậm.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, sử dụng số trong các ngữ cảnh chỉ số lượng lớn, tiền tệ, dân số...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** có phần thập phân (ví dụ: ~~1.000,5~~).
* **TUYỆT ĐỐI KHÔNG** sử dụng sai dấu phân cách (ví dụ: ~~1,000~~, ~~1 000 000~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dấu Phân Cách
* **ĐÚNG**: Giải thưởng trị giá **1.000.000** đồng.
* **SAI**: Giải thưởng trị giá **1,000,000** đồng. (Lý do: Sai dấu phân cách, phải là `.`)

### Về Phần Thập Phân
* **ĐÚNG**: Chiến dịch trồng **2.000.000** cây xanh.
* **SAI**: Chiến dịch trồng **2.000.000,5** cây xanh. (Lý do: Chứa phần thập phân, không phải số nguyên)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Sự kiện thu hút hơn **1.000** người.
* **SAI**: Sự kiện thu hút hơn **một nghìn** người. (Lý do: Dùng chữ viết thay vì số)

### Về đơn vị đi cùng (nếu có)
* **ĐÚNG**: Dải tần đáp ứng là **20.000** héc.
* **SAI**: Dải tần đáp ứng là **20.000** Hz. (Lý do: Dùng đơn vị ở dạng ký hiệu)
* **SAI**: Dải tần đáp ứng là **20.000 héc**. (Lý do: In đậm các từ khác ngoài số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Sự kiện thu hút hơn **1.000** người.
* **SAI**: Sự kiện thu hút hơn 1.000 người. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["FLOAT_n"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ THẬP PHÂN (DẤU CHẤM)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa số thập phân sử dụng dấu chấm (`.`) làm dấu ngăn cách. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa: `[Số Nguyên].[Số Thập Phân]`
* **Số nguyên, Số thập phân**: Các chuỗi chữ số.
* **Dấu phân cách**: Bắt buộc phải là dấu chấm (`.`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `1.25`, `5.5`, `9.5`, `8.5`).
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~chín phẩy năm~~).
* **DẤU PHÂN CÁCH**: **TUYỆT ĐỐI KHÔNG** dùng dấu phẩy (`,`) làm dấu ngăn cách thập phân.
* **ĐƠN VỊ**: Nếu có đơn vị đi cùng số thì đơn vị phải được viết dưới dạng **chữ đầy đủ** chứ không phải dạng ký hiệu. Đơn vị không được nằm trong cấu trúc cốt lõi được in đậm.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả các số liệu kỹ thuật, khoa học, tài chính, kinh tế, điểm số...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** dùng dấu phẩy (`,`) cho phần thập phân (ví dụ: ~~9,5~~).
* **TUYỆT ĐỐI KHÔNG** thêm dấu phân cách hàng nghìn vào phần nguyên.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dấu Phân Cách Thập Phân
* **ĐÚNG**: Chi tiết máy này dài **1.25** mét.
* **SAI**: Chi tiết máy này dài **1,25** mét. (Lý do: Sai dấu phân cách thập phân, phải là `.`)

### Về Phân Cách Hàng Nghìn
* **ĐÚNG**: Hệ số là **1000.5**.
* **SAI**: Hệ số là **1,000.5**. (Lý do: Chứa dấu phẩy phân cách hàng nghìn, không đúng cấu trúc yêu cầu)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Anh ấy đạt **9.5** điểm.
* **SAI**: Anh ấy đạt **chín phẩy năm** điểm. (Lý do: Dùng chữ viết thay vì số)

### Về đơn vị đi cùng (nếu có)
* **ĐÚNG**: Dải tần đáp ứng là **20.0** héc.
* **SAI**: Dải tần đáp ứng là **20.0** Hz. (Lý do: Dùng đơn vị ở dạng ký hiệu)
* **SAI**: Dải tần đáp ứng là **20.0 héc**. (Lý do: In đậm các từ khác ngoài số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Anh ấy đạt **9.5** điểm.
* **SAI**: Anh ấy đạt 9.5 điểm. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["FLOAT_big"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ THẬP PHÂN LỚN

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa số thập phân lớn, kết hợp dấu chấm (`.`) cho hàng nghìn và dấu phẩy (`,`) cho phần thập phân. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `([Số]\.)+[Số]{3},[Số]+`
* **Phần nguyên**: Dùng dấu chấm (`.`) ngăn cách mỗi 3 chữ số.
* **Phần thập phân**: Dùng dấu phẩy (`,`) để ngăn cách.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `50.000.000,45`, `1.000,5`). Phần nguyên phải có ít nhất 4 chữ số
    * KHÔNG DÙNG: Viết bằng chữ.
* **DẤU PHÂN CÁCH**: Tuân thủ nghiêm ngặt: `.` cho hàng nghìn, `,` cho thập phân.
* **ĐƠN VỊ**: Nếu có đơn vị đi cùng số thì đơn vị phải được viết dưới dạng **chữ đầy đủ** chứ không phải dạng ký hiệu. Đơn vị không được nằm trong cấu trúc cốt lõi được in đậm.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả các số liệu tài chính, khoa học có giá trị lớn và độ chính xác cao.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** hoán đổi vai trò của dấu `.` và dấu `,`.
* **TUYỆT ĐỐI KHÔNG** bỏ sót một trong hai loại dấu nếu cấu trúc số yêu cầu.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Vai Trò Của Dấu Phân Cách
* **ĐÚNG**: Số tiền chính xác là **50.000.000,45** đồng.
* **SAI**: Số tiền chính xác là **50,000,000.45** đồng. (Lý do: Hoán đổi vai trò của dấu `.` và `,`)

### Về Cấu Trúc
* **ĐÚNG**: Mảnh đất rộng **1.000,5** ha.
* **SAI**: Mảnh đất rộng **1000,5** ha. (Lý do: Thiếu dấu `.` phân cách hàng nghìn)
* **SAI**: Mảnh đất rộng **10,5** ha. (Lý do: Đây là số thập phân bình thường)

### Về Lỗi Kết Hợp
* **ĐÚNG**: Giá trị đo được là **1.234.567,89**.
* **SAI**: Giá trị đo được là **1,234,567.89**. (Lý do: Sai cả hai loại dấu phân cách)

### Về đơn vị đi cùng (nếu có)
* **ĐÚNG**: Dải tần đáp ứng là **20.000,4** héc.
* **SAI**: Dải tần đáp ứng là **20.000,4** Hz. (Lý do: Dùng đơn vị ở dạng ký hiệu)
* **SAI**: Dải tần đáp ứng là **20.000,4 héc**. (Lý do: In đậm các từ khác ngoài số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Giá trị đo được là **1.234.567,89**.
* **SAI**: Giá trị đo được là 1.234.567,89. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["FRACTION"] = """
# YÊU CẦU SINH DỮ LIỆU PHÂN SỐ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa phân số. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Tử số]/[Mẫu số]`
* **Tử số, Mẫu số**: Luôn phải có mặt.
* **Dấu phân cách**: Bắt buộc phải là dấu gạch chéo (`/`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**. Tử số và mẫu số có thể là số nguyên đơn giản hoặc số nguyên lớn có dấu `.` phân cách (ví dụ: `1/2`, `3/4`, `1/1.000.000`).
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~một phần hai~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, sử dụng phân số trong ngữ cảnh nấu ăn, chia chác, thống kê, tỷ lệ, xác suất...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** sử dụng dấu phân cách khác (ví dụ: `\` hoặc `:`).
* **TUYỆT ĐỐI KHÔNG** diễn giải bằng lời (ví dụ: ~~một nửa~~, ~~ba phần tư~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Cho thêm **1/2** muỗng cà phê.
* **SAI**: Cho thêm **một nửa** muỗng cà phê. (Lý do: Dùng từ diễn giải thay vì cấu trúc `Số/Số`)

### Về Thành Phần
* **ĐÚNG**: Khả năng xảy ra là **1/1.000.000**.
* **SAI**: Khả năng xảy ra là **1 trên 1.000.000**. (Lý do: Dùng từ nối thay vì dấu `/`)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Công ty chiếm **3/4** thị phần.
* **SAI**: Công ty chiếm **ba phần tư** thị phần. (Lý do: Dùng chữ viết thay vì số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Công ty chiếm **3/4** thị phần.
* **SAI**: Công ty chiếm 3/4 thị phần. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["NUMBER_RANGE"] = """
# YÊU CẦU SINH DỮ LIỆU KHOẢNG SỐ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một khoảng giá trị số. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Số bắt đầu]-[Số kết thúc]`
* **Số bắt đầu, Số kết thúc**: Luôn phải có mặt. Số kết thúc phải lớn hơn hoặc bằng số bắt đầu.
* **Dấu phân cách**: Bắt buộc phải là dấu gạch nối (`-`).
* **Đơn vị**: Có thể có hoặc không. Nếu có, nó thường đứng sau số kết thúc (ví dụ `25-30°C`) hoặc sau cả hai số (ví dụ `1.2m - 1.5m`).
    * CHÚ Ý: Nếu có đơn vị ở cả hai số thì giữa hai số và dấu phân cách phải có khoảng trống (ví dụ `1.2m - 1.5m`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `25-30`, `1.2-2.3`, `-1-1`).
    * KHÔNG DÙNG: Viết bằng chữ.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả sự dao động, biến thiên của một giá trị (nhiệt độ, lãi suất, vận tốc, số lượng...).

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** sử dụng các từ nối như `đến`, `tới` thay cho dấu `-`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Nhiệt độ dao động **25-30°C**.
* **SAI**: Nhiệt độ dao động từ **25 đến 30°C**. (Lý do: Dùng từ nối thay vì cấu trúc `Số-Số`)

### Về Dấu Phân Cách
* **ĐÚNG**: Lãi suất dự kiến là **1.2-2.3%**.
* **SAI**: Lãi suất dự kiến là **1.2 ~ 2.3%**. (Lý do: Sai dấu phân cách, phải là `-`)
* **ĐÚNG**: Ưu tiên trẻ em cao **1.2m - 1.5m**.
* **SAI**: Ưu tiên trẻ em cao **1.2m-1.5m**. (Lý do: Phải có khoảng trống quanh dấu `-` khi đơn vị xuất hiện ở cả hai số.)

### Về Đơn Vị
* **ĐÚNG**: Vận tốc của nó là **1.2m - 1.5m/s**.
* **SAI**: Vận tốc của nó là từ **1.2 đến 1.5** mét trên giây. (Lý do: Sai định dạng số, cấu trúc và đơn vị)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Vận tốc của nó là **1.2m - 1.5m/s**.
* **SAI**: Vận tốc của nó là 1.2m - 1.5m/s. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["MATH_EXPR"] = """
# YÊU CẦU SINH DỮ LIỆU BIỂU THỨC TOÁN HỌC

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một biểu thức toán học hoàn chỉnh. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa một chuỗi các số và toán tử: `[Số] [Toán tử] [Số] ...`
* **Toán tử**: Các ký hiệu phổ biến như `+`, `-`, `*`, `/`, `^`, `=`.
* **Số**: Có thể là số nguyên, số thập phân, phân số.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `2`, `1/2`, `25.000`, `10%`).
    * KHÔNG DÙNG: Viết bằng chữ.
* **TOÁN TỬ:** Bắt buộc phải xuất hiện ít nhất một toán tử
* **SPACING**: Nên có khoảng trắng giữa các số và toán tử để dễ đọc.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, sử dụng biểu thức để minh họa một phép tính, một công thức, một quy tắc...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** diễn giải phép tính bằng lời (ví dụ: ~~hai cộng hai bằng bốn~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Vấn đề này cũng đơn giản như **2 + 2 = 4** thôi.
* **SAI**: Vấn đề này cũng đơn giản như **hai cộng hai bằng bốn** thôi. (Lý do: Diễn giải phép tính bằng lời)

### Về Thành Phần
* **ĐÚNG**: Giá mới sẽ là **25.000 * (1 + 10%) = 27.500** đồng.
* **SAI**: Giá mới sẽ là **25.000 nhân với 1 cộng 10 phần trăm**. (Lý do: Dùng từ ngữ thay cho ký hiệu toán tử)

### Về toán tử
* **ĐÚNG:** Lỡ quên giấy tờ, mỗi lần thiếu thì bị phạt **100.000 * 5% = 5.000** đồng.
* **SAI:** Lỡ quên giấy tờ, mỗi lần thiếu thì bị phạt **5%** số tiền. (Lý do: không xuất hiện bất kì toán tử nào)

### Về Ngữ Cảnh
* **ĐÚNG**: Tôi không hiểu sao công thức **1/2 + 1/2 * 5 = 3** lại sai.
* **SAI**: **1/2 + 1/2 * 5 = 3**. (Lý do: Chỉ có biểu thức, thiếu câu văn ngữ cảnh tự nhiên)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Tôi không hiểu sao công thức **1/2 + 1/2 * 5 = 3** lại sai.
* **SAI**: Tôi không hiểu sao công thức 1/2 + 1/2 * 5 = 3 lại sai. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["PHONE"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ ĐIỆN THOẠI

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa số điện thoại. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa một chuỗi số là số điện thoại hợp lệ của Việt Nam.
* **Cấu trúc**: Có thể bắt đầu bằng mã quốc gia `+84` (thay cho số `0` ở đầu) hoặc không.
* **Dấu phân cách**: Có thể không có dấu, hoặc dùng dấu chấm (`.`), khoảng trắng (` `), hoặc dấu gạch nối (`-`) để ngăn cách các nhóm số.

### 2. Quy Tắc Định Dạng
* **KÝ TỰ HỢP LỆ**: Bắt buộc phải là **chữ số**. Các ký tự hợp lệ khác bao gồm dấu cộng `+` (chỉ ở đầu), dấu chấm `.`, khoảng trắng ` `, dấu gạch nối `-` và dấu ngoặc đơn `()`.
    * KHÔNG DÙNG: Viết bằng chữ.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập đến việc liên lạc, gọi điện, nhắn tin...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** dùng số điện thoại không hợp lệ hoặc không có thật ở Việt Nam.
* **TUYỆT ĐỐI KHÔNG** dùng các ký tự lạ khác ngoài danh sách ký tự hợp lệ đã nêu (ví dụ: `/`, `\`, `,`).
* **TUYỆT ĐỐI KHÔNG** dùng đồng thời `+84` và số `0` ở đầu số di động (ví dụ: ~~+84 09...~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dấu Phân Cách
* **ĐÚNG**: Xin lỗi, tôi cần xác nhận lại số điện thoại của người làm chứng: có phải là **0868.123.456** không?
* **ĐÚNG**: Gọi cho tôi qua số **0987 654 321**.
* **ĐÚNG**: Liên hệ qua số **090-123-4567**. (Lý do: Chấp nhận dấu `-` làm dấu phân cách)
* **SAI**: Số của tôi là **0912/345/678**. (Lý do: Sử dụng dấu `/` làm dấu phân cách không hợp lệ)

### Về Đầu Số Quốc Gia (+84)
* **ĐÚNG**: Bạn có thể liên hệ cho cô ấy qua di động, hình như là **+84 912 345 678**.
* **SAI**: Bạn có thể liên hệ tôi qua số **+84 0912 345 678**. (Lý do: Thừa số `0` sau mã quốc gia `+84`)
* **SAI**: Bạn có thể liên hệ tôi qua số **84912345678**. (Lý do: Thiếu dấu `+` cho mã quốc gia)

### Về Định Dạng Máy Bàn
* **ĐÚNG**: Vui lòng liên hệ qua tổng đài **(024) 3943-9999**.
* **SAI**: Vui lòng liên hệ qua tổng đài **02439439999**. (Lý do: Thiếu dấu ngoặc, dấu phân cách khiến khó đọc và sai chuẩn)

### Về Ngữ Cảnh
* **ĐÚNG**: Zalo của anh là **0398765432**.
* **SAI**: **0398765432**. (Lý do: Thiếu câu văn ngữ cảnh)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Zalo của anh là **0398765432**.
* **SAI**: Zalo của anh là 0398765432. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["ID_NUMBER"] = """
# YÊU CẦU SINH DỮ LIỆU MÃ SỐ NHẬN DẠNG

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một mã số nhận dạng (số tài khoản, mã số thuế, CCCD, mã nhân viên...). Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa một chuỗi `[Mã số]` là một dãy ký tự (thường là số, hoặc kết hợp chữ và số).
* **Dấu phân cách**: Chuỗi mã số **có thể** chứa dấu gạch nối (`-`) nếu đó là định dạng chuẩn của loại mã đó (ví dụ: một số mã CCCD hoặc mã số thuế cũ).

### 2. Quy Tắc Định Dạng
* **KÝ TỰ**: Bắt buộc phải là dạng ký tự gốc, không diễn giải bằng lời.
* **ĐA DẠNG**: Các mã số có thể có hoặc không có dấu gạch nối, độ dài và cấu trúc khác nhau.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập đến ngữ cảnh sử dụng mã số đó (giao dịch, định danh, chấm công...).

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thêm các ký tự lạ hoặc các dấu phân cách khác (như `.`, `,`, ` `) không có trong mã số gốc.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dấu Phân Cách
* **ĐÚNG**: Mã số thuế của công ty là **01234567-890**. (Lý do: Chấp nhận dấu gạch nối `-` làm dấu phân cách hợp lệ)
* **SAI**: Mã số thuế của công ty là **01234567.890**. (Lý do: Sử dụng dấu `.` làm dấu phân cách không hợp lệ)

### Về Mã Không Có Dấu Phân Cách
* **ĐÚNG**: Chuyển tiền vào số tài khoản **124123441** nhé. (Lý do: Nhiều loại mã không có dấu ngăn cách)
* **SAI**: Chuyển tiền vào số tài khoản **một hai bốn một hai ba bốn bốn một** nhé. (Lý do: Diễn giải bằng chữ)

### Về Mã Chữ và Số
* **ĐÚNG**: Mã nhân viên của bạn là **F0125G**.
* **SAI**: **F0125G**. (Lý do: Thiếu câu văn ngữ cảnh)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Mã nhân viên của bạn là **F0125G**.
* **SAI**: Mã nhân viên của bạn là F0125G. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["LEGAL_DOC_ID"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ HIỆU VĂN BẢN PHÁP LUẬT

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có trích dẫn số hiệu của một văn bản pháp luật. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Số]/[Năm]/[Mã]`
* **Số**: Số thứ tự của văn bản.
* **Năm**: Năm ban hành.
* **Mã**: Mã loại văn bản và cơ quan ban hành (ví dụ: NĐ-CP, TT-BGDĐT, QĐ-UBND).

### 2. Quy Tắc Định Dạng
* **ĐỊNH DẠNG**: Bắt buộc giữ nguyên cấu trúc và các ký tự gốc (`/`, `-`).
    * KHÔNG DÙNG: Diễn giải bằng lời.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập hoặc trích dẫn một điều luật, quy định từ một văn bản cụ thể.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thay đổi cấu trúc hoặc các ký tự phân cách.
* **TUYỆT ĐỐI KHÔNG** viết tắt hay biến tấu số hiệu.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Căn cứ theo Nghị định **110/2013/NĐ-CP**.
* **SAI**: Căn cứ theo Nghị định **110, năm 2013 của Chính Phủ**. (Lý do: Diễn giải thay vì dùng số hiệu chuẩn)

### Về Ký Tự
* **ĐÚNG**: Theo Thông tư **01/2025/TT-BGDĐT**.
* **SAI**: Theo Thông tư **01-2025-TT-BGDĐT**. (Lý do: Sai ký tự phân cách `/`)

### Về Ngữ Cảnh
* **ĐÚNG**: Quy trình mới sẽ tuân theo văn bản **23/2024/AABC**.
* **SAI**: **23/2024/AABC**. (Lý do: Thiếu câu văn ngữ cảnh)
""".strip()

# ============================================= #

TAG_TO_PROMPT["PLATE"] = """
# YÊU CẦU SINH DỮ LIỆU BIỂN SỐ XE

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin biển số xe hợp lệ của Việt Nam. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa một chuỗi `[Biển số xe]` theo các định dạng chuẩn của Việt Nam.
* **Ví dụ**: `29-F1 123.45`, `51C-888.88`, `30A-123.45`.

### 2. Quy Tắc Định Dạng
* **ĐỊNH DẠNG**: Giữ nguyên các ký tự, số, dấu chấm, dấu gạch nối và khoảng trắng của biển số gốc.
    * KHÔNG DÙNG: Viết liền không có định dạng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập đến một chiếc xe cụ thể trong các bối cảnh giao thông, mua bán, hoặc bình luận.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** sử dụng biển số không đúng chuẩn định dạng (ví dụ: thiếu dấu chấm, sai vị trí gạch nối).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Định Dạng
* **ĐÚNG**: Chiếc xe đó có biển số **29-F1 123.45**.
* **SAI**: Chiếc xe đó có biển số **29F112345**. (Lý do: Thiếu dấu gạch nối, khoảng trắng và dấu chấm)

### Về Cấu Trúc
* **ĐÚNG**: Công nhận con xe **30A-555.55** có biển số đẹp thật.
* **SAI**: Công nhận con xe **30A 55555** có biển số đẹp thật. (Lý do: Thiếu dấu gạch nối và dấu chấm)

### Về Ngữ Cảnh
* **ĐÚNG**: Tôi vừa thấy chiếc **51C-888.88** chạy ngoài đường.
* **SAI**: **51C-888.88**. (Lý do: Thiếu câu văn ngữ cảnh)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Tôi vừa thấy chiếc **51C-888.88** chạy ngoài đường.
* **SAI**: Tôi vừa thấy chiếc 51C-888.88 chạy ngoài đường. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["ALPHANUM_ID"] = """
# YÊU CẦU SINH DỮ LIỆU MÃ CHỮ VÀ SỐ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một chuỗi mã kết hợp ngẫu nhiên giữa chữ, số và ký tự đặc biệt. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa một chuỗi `[Mã chữ và số]` đa dạng.
* **Thành phần**: Kết hợp chữ cái (hoa, thường), số, và có thể có ký tự đặc biệt (`#`, `&`, `*`, `%`...).

### 2. Quy Tắc Định Dạng
* **ĐỊNH DẠNG**: Giữ nguyên chuỗi ký tự gốc, phân biệt chữ hoa và chữ thường.
    * KHÔNG DÙNG: Diễn giải bằng lời.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập đến mã đặt chỗ, mã hiệu, số serial, tên phiên bản, mật khẩu, mã truy cập...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** thêm khoảng trắng vào giữa chuỗi mã nếu mã gốc không có.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Vui lòng giữ lại mã đặt chỗ **A1B2C34**.
* **SAI**: Vui lòng giữ lại mã đặt chỗ **A 1 B 2 C 34**. (Lý do: Thêm khoảng trắng không cần thiết)

### Về Ký Tự Đặc Biệt
* **ĐÚNG**: Nhập mã khuyến mãi **SALE#50%** để được giảm giá.
* **SAI**: Nhập mã khuyến mãi **SALE thăng 50 phần trăm** để được giảm giá. (Lý do: Dùng chữ diễn giải thay vì các ký tự đặc biệt `#` và `%`)

### Về Ngữ Cảnh
* **ĐÚNG**: Hãy nâng cấp lên phiên bản **v1.2a** để vá lỗi.
* **SAI**: **v1.2a**. (Lý do: Thiếu câu văn ngữ cảnh)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Hãy nâng cấp lên phiên bản **v1.2a** để vá lỗi.
* **SAI**: Hãy nâng cấp lên phiên bản v1.2a để vá lỗi. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["ROMAN_NUMERAL"] = """
# YÊU CẦU SINH DỮ LIỆU SỐ LA MÃ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa số La Mã. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa một `[Số La Mã]` hợp lệ.
* **Ví dụ**: `I`, `V`, `IX`, `XIV`, `XXI`.

### 2. Quy Tắc Định Dạng
* **KÝ TỰ**: Bắt buộc phải là các chữ cái La Mã viết hoa (`I, V, X, L, C, D, M`).
    * KHÔNG DÙNG: Chữ thường (ví dụ: ~~xxi~~) hoặc diễn giải bằng lời (ví dụ: ~~thế kỷ hai mươi mốt~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, sử dụng số La Mã để chỉ tên vua, thế kỷ, chương mục, đại hội, quý trong năm...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** dùng các chuỗi không phải là số La Mã hợp lệ (ví dụ: ~~IIII~~, ~~VX~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Định Dạng
* **ĐÚNG**: Đó là một sự kiện của thế kỷ **XXI**.
* **SAI**: Đó là một sự kiện của thế kỷ **21**. (Lý do: Dùng số Ả Rập thay vì số La Mã)

### Về Ký Tự
* **ĐÚNG**: Vua Louis **XIV** được mệnh danh là Vua Mặt Trời.
* **SAI**: Vua Louis **mười bốn** được mệnh danh là Vua Mặt Trời. (Lý do: Diễn giải bằng chữ)

### Về Ngữ Cảnh
* **ĐÚNG**: Vui lòng đọc kỹ mục **I**, chương **V** của giáo trình.
* **SAI**: **I**, **V**. (Lý do: Thiếu câu văn ngữ cảnh)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Vui lòng đọc kỹ mục **I**, chương **V** của giáo trình.
* **SAI**: Vui lòng đọc kỹ mục I, chương V của giáo trình. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["ADDRESS"] = """
# YÊU CẦU SINH DỮ LIỆU KÝ HIỆU ĐỊA CHỈ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa các **ký hiệu và cách viết tắt đặc trưng** của địa chỉ. Mục tiêu chính là để mô hình nhận diện được các ký hiệu đặc biệt này, chứ không phải toàn bộ chuỗi địa chỉ.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cần Tạo
Tập trung vào việc sinh ra các ký hiệu sau trong một ngữ cảnh tự nhiên:
* **Ký hiệu Ngõ/Hẻm**: Dấu gạch chéo (`/`) nằm giữa các con số. Ví dụ: `123/45`.
* **Viết tắt Phường**: `P.` theo sau là tên phường. Ví dụ: `P. Bến Nghé`.
* **Viết tắt Quận**: `Q.` theo sau là tên hoặc số của quận. Ví dụ: `Q.1`, `Q. Ba Đình`.
* **Viết tắt Thành phố**: `TP.` theo sau là tên thành phố. Ví dụ: `TP. Hà Nội`.

### 2. Quy Tắc Định Dạng & Highlight
* **QUAN TRỌNG NHẤT**: Chỉ **in đậm (highlight)** đúng phần ký hiệu đặc biệt (`/`, `P.`, `Q.`, `TP.`) và con số hoặc tên đi liền ngay sau nó nếu có.
* **KHÔNG** in đậm toàn bộ địa chỉ.

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên và các từ xung quanh phải làm rõ đây là một thông tin về địa chỉ.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** in đậm toàn bộ cụm từ chỉ địa chỉ.
* **TUYỆT ĐỐI KHÔNG** diễn giải các ký hiệu ra thành chữ đầy đủ (ví dụ: không viết "ngõ", "phường", "quận"...).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Ký Hiệu Ngõ/Hẻm (/)
* **ĐÚNG**: Cậu cứ đến số nhà **123/45** đường Lê Lợi rồi gọi tớ nhé.
* **SAI**: Cậu cứ đến **số nhà 123/45 đường Lê Lợi** rồi gọi tớ nhé. (Lý do: In đậm cả địa chỉ thay vì chỉ ký hiệu)
* **SAI**: Cậu cứ đến số nhà 123 ngõ 45 đường Lê Lợi rồi gọi tớ nhé. (Lý do: Diễn giải bằng chữ "ngõ" thay vì dùng ký hiệu `/`)

### Về Ký Hiệu Viết Tắt Phường (P.)
* **ĐÚNG**: Địa chỉ công ty ở **P.** Trung Hòa, quận Cầu Giấy.
* **SAI**: Địa chỉ công ty ở **Phường Trung Hòa**, quận Cầu Giấy. (Lý do: Viết đầy đủ chữ "Phường" thay vì ký hiệu `P.`)
* **SAI**: Địa chỉ công ty ở **P. Trung Hòa, quận Cầu Giấy**. (Lý do: In đậm quá nhiều thông tin không cần thiết)

### Về Ký Hiệu Viết Tắt Quận (Q.)
* **ĐÚNG**: Anh ấy sống ở **Q.3** từ khi còn nhỏ.
* **SAI**: Anh ấy sống ở **Quận 3** từ khi còn nhỏ. (Lý do: Viết đầy đủ chữ "Quận" thay vì ký hiệu `Q.`)

### Về Ký Hiệu Viết Tắt Thành Phố (TP.)
* **ĐÚNG**: Sự kiện sẽ được tổ chức tại **TP.** Hồ Chí Minh.
* **SAI**: Sự kiện sẽ được tổ chức tại Thành phố Hồ Chí Minh. (Lý do: Thiếu ký hiệu viết tắt `TP.`)

### Về Lỗi Kết Hợp
* **ĐÚNG**: Bưu kiện cần được gửi về **P.4**, **Q.** Tân Bình, **TP.** Hồ Chí Minh.
* **SAI**: Bưu kiện cần được gửi về **P.4, Q.Tân Bình, TP.Hồ Chí Minh**. (Lý do: In đậm toàn bộ chuỗi địa chỉ)
""".strip()

# ============================================= #

TAG_TO_PROMPT["MONEY"] = """
# YÊU CẦU SINH DỮ LIỆU TIỀN TỆ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin về một số tiền. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Số][Ký hiệu tiền tệ]`
* **Số**: Một số nguyên hoặc thập phân.
* **Ký hiệu tiền tệ**: Một ký hiệu chuẩn như `đ`, `vnđ`, `$`, `€`...

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc là **chữ số**. Có thể dùng `.` để phân cách hàng nghìn hoặc `,` để phân cách thập phân tùy theo quy ước của đơn vị tiền tệ.
* **KÝ HIỆU**: Đứng liền sau số (`500.000đ`) hoặc đứng trước số (`$100`).
    * KHÔNG DÙNG: Viết đầy đủ tên đơn vị (ví dụ: ~~đồng~~, ~~đô la~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả giá cả, chi phí, lương bổng, hóa đơn...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** có khoảng trắng giữa số và ký hiệu tiền tệ (ví dụ: ~~500.000 đ~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Ký Hiệu
* **ĐÚNG**: Cái áo này giá **500.000đ**.
* **SAI**: Cái áo này giá **500.000 đồng**. (Lý do: Dùng tên đầy đủ thay vì ký hiệu)

### Về Vị Trí Ký Hiệu
* **ĐÚNG**: Nó có giá **$100**.
* **SAI**: Nó có giá **100$**. (Lý do: Sai vị trí ký hiệu `$` theo quy ước quốc tế)

### Về Khoảng Trắng
* **ĐÚNG**: Lương của tôi là **20.000.000vnđ**.
* **SAI**: Lương của tôi là **20.000.000 vnđ**. (Lý do: Có khoảng trắng giữa số và ký hiệu)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Lương của tôi là **20.000.000vnđ**.
* **SAI**: Lương của tôi là 20.000.000vnđ. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DIMENSION"] = """
# YÊU CẦU SINH DỮ LIỆU KÍCH THƯỚC (2 CHIỀU)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin về kích thước hai chiều (dài x rộng). Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc có dạng `[Số]x[Số]`.
* **Số**: Có thể là số nguyên hoặc số thập phân.
* **Dấu phân cách**: Bắt buộc phải là chữ `x` viết thường.
* **Đơn vị**: Có thể được đặt ở cuối cùng (ví dụ: `5x20m`) hoặc đặt ngay sau mỗi số (ví dụ: `5mx20m`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số** (ví dụ: `20.5x30`, `1920x1080`, `5mx20m`).
    * KHÔNG DÙNG: Viết bằng chữ.
* **KÝ HIỆU**: Bắt buộc là `x`.
    * KHÔNG DÙNG: Dấu nhân `*` hoặc chữ `X` hoa.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, mô tả về diện tích, kích thước ảnh, độ phân giải, kích thước lô đất...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** có khoảng trắng xung quanh chữ `x`.
* **TUYỆT ĐỐI KHÔNG** đặt đơn vị ở vị trí không hợp lệ (ví dụ: ~~cm50xcm20~~).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dấu Phân Cách
* **ĐÚNG**: Ảnh phải có độ phân giải **1920x1080** pixel.
* **SAI**: Ảnh phải có độ phân giải **1920*1080** pixel. (Lý do: Sai dấu phân cách, phải là `x`)

### Về Khoảng Trắng
* **ĐÚNG**: Tôi cần tìm mảnh đất **5x20m**.
* **SAI**: Tôi cần tìm mảnh đất **5 x 20m**. (Lý do: Có khoảng trắng quanh dấu `x`)

### Về Vị Trí Đơn Vị
* **ĐÚNG**: Tấm ván có kích thước **50cmx120cm**. (Lý do: Chấp nhận đơn vị đặt sau mỗi số)
* **ĐÚNG**: Tấm ván có kích thước **50x120cm**. (Lý do: Chấp nhận đơn vị đặt ở cuối)
* **SAI**: Tấm ván có kích thước **cm50xcm120**. (Lý do: Đặt sai vị trí đơn vị, phải đặt sau số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Tôi cần tìm mảnh đất **5x20m**.
* **SAI**: Tôi cần tìm mảnh đất 5x20m. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["SPORT_SCORE"] = """
# YÊU CẦU SINH DỮ LIỆU TỶ SỐ

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa thông tin về tỷ số của một trận đấu. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[Số]-[Số]`
* **Số**: Các số nguyên chỉ tỷ số của hai đội.
* **Dấu phân cách**: Bắt buộc phải là dấu gạch nối (`-`).

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**.
    * KHÔNG DÙNG: Viết bằng chữ (ví dụ: ~~hai một~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập đến kết quả của một trận đấu thể thao như bóng đá, bóng rổ, tennis...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** sử dụng các từ nối như `đấu với`, `và` thay cho dấu `-`.
* **TUYỆT ĐỐI KHÔNG** có khoảng trắng xung quanh dấu `-`.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Việt Nam thắng **2-1**.
* **SAI**: Việt Nam thắng **2 và 1**. (Lý do: Sai cấu trúc, không dùng từ nối)

### Về Dấu Phân Cách
* **ĐÚNG**: Trận đấu kết thúc với tỷ số **0-0**.
* **SAI**: Trận đấu kết thúc với tỷ số **0 - 0**. (Lý do: Có khoảng trắng quanh dấu `-`)

### Về Định Dạng Chữ/Số
* **ĐÚNG**: Đội bạn thắng áp đảo **30-12**.
* **SAI**: Đội bạn thắng áp đảo **ba mươi mười hai**. (Lý do: Dùng chữ viết thay vì số)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Đội bạn thắng áp đảo **30-12**.
* **SAI**: Đội bạn thắng áp đảo 30-12. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["EMAIL"] = """
# YÊU CẦU SINH DỮ LIỆU ĐỊA CHỈ EMAIL

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một địa chỉ email hợp lệ. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải theo đúng cấu trúc: `[tên]@[miền].[đuôi]`
* **Thành phần**: Bao gồm tên người dùng, ký tự `@`, tên miền và đuôi miền (`.com`, `.vn`, `.org`...).

### 2. Quy Tắc Định Dạng
* **ĐỊNH DẠNG**: Giữ nguyên cấu trúc, không có khoảng trắng.
    * KHÔNG DÙNG: Diễn giải bằng lời (ví dụ: ~~nguyenvana a còng gmail chấm com~~).
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập đến việc gửi/nhận email, liên lạc, đăng ký tài khoản...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** sử dụng các địa chỉ email không hợp lệ (ví dụ: thiếu `@`, thiếu đuôi miền).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc Hợp Lệ
* **ĐÚNG**: Gửi CV về địa chỉ **example.user@email.com**.
* **SAI**: Gửi CV về địa chỉ **example.user.email.com**. (Lý do: Thiếu ký tự `@`, cấu trúc không hợp lệ)

### Về Khoảng Trắng
* **ĐÚNG**: Email của tôi là **nguyenvana123@gmail.com**.
* **SAI**: Email của tôi là **nguyen van a 123 @ gmail.com**. (Lý do: Chứa khoảng trắng không hợp lệ)

### Về Ngữ Cảnh
* **ĐÚNG**: Liên hệ hỗ trợ tại **support@company.vn**.
* **SAI**: **support@company.vn**. (Lý do: Thiếu câu văn ngữ cảnh)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Liên hệ hỗ trợ tại **support@company.vn**.
* **SAI**: Liên hệ hỗ trợ tại support@company.vn. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["URL"] = """
# YÊU CẦU SINH DỮ LIỆU ĐỊA CHỈ WEB (URL)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên có chứa một địa chỉ web (URL) hợp lệ. Các câu này phải tuân thủ nghiêm ngặt các quy tắc định dạng được nêu dưới đây.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi mẫu phải chứa một chuỗi `[URL]` có cấu trúc: `(http[s]?://)?(www\.)?[miền].[đuôi]/[đường_dẫn]?`
* **Thành phần**: Có thể bao gồm hoặc không bao gồm `http(s)://`, `www.`, và đường dẫn.

### 2. Quy Tắc Định Dạng
* **ĐỊNH DẠNG**: Giữ nguyên cấu trúc, không có khoảng trắng ở những vị trí không hợp lệ.
    * KHÔNG DÙNG: Diễn giải bằng lời.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ cấu trúc cốt lõi trong câu phải được bôi đậm bằng hai dấu sao ở mỗi bên (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, đề cập đến một trang web, tài liệu tham khảo, đường link...

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** sử dụng URL không hợp lệ (ví dụ: có khoảng trắng, thiếu đuôi miền).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Cấu Trúc
* **ĐÚNG**: Xem thêm tại **https://www.example.com/products**.
* **SAI**: Xem thêm tại **https www example com products**. (Lý do: Thiếu các dấu `://`, `.` và `/`)

### Về Khoảng Trắng
* **ĐÚNG**: Truy cập **google.com.vn** để biết thêm chi tiết.
* **SAI**: Truy cập **google. com. vn** để biết thêm chi tiết. (Lý do: Chứa khoảng trắng không hợp lệ)

### Về Ngữ Cảnh
* **ĐÚNG**: Link video nằm ở **http://googleusercontent.com/youtube/0**.
* **SAI**: **http://googleusercontent.com/youtube/0**. (Lý do: Thiếu câu văn ngữ cảnh)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Link video nằm ở **http://googleusercontent.com/youtube/0**.
* **SAI**: Link video nằm ở http://googleusercontent.com/youtube/0. (Lý do: Thiếu in đậm cho cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["DATE__FRACTION"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (NGÀY THÁNG & PHÂN SỐ)

## MỤC TIÊU CHÍNH

Tạo ra một câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể có cùng pattern `Số/Số`: một là **Ngày tháng (DATE)** và một là **Phân số (FRACTION)**. Câu phải được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi câu phải chứa:
* Một thực thể `[Số]/[Số][DATE]` trong ngữ cảnh ngày tháng.
* Một thực thể `[Số]/[Số][FRACTION]` trong ngữ cảnh tỷ lệ, phần trăm, chia chác.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**.
* **GÁN NHÃN**: Nhãn `[DATE]` hoặc `[FRACTION]` phải được đặt ngay sau thực thể tương ứng, không có khoảng trắng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, logic, kết nối hai thực thể một cách hợp lý.
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** tạo câu mà cả hai thực thể đều là ngày tháng hoặc đều là phân số. Phải có sự khác biệt về ngữ cảnh.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Ngữ Cảnh và Gán Nhãn
* **ĐÚNG**: Trong chương trình **1/6**[DATE], nhóm chúng tôi đã quyên góp được **1/2**[FRACTION] số quà tặng dự kiến.
* **SAI**: Trong chương trình **1/6**[FRACTION], nhóm chúng tôi đã quyên góp được **1/2**[DATE] số quà tặng dự kiến. (Lý do: Gán nhãn sai ngữ cảnh)

### Về Sự Tồn Tại Của Cả Hai Nhãn
* **ĐÚNG**: Mới có **3/4**[FRACTION] số sinh viên nộp bài trong khi hạn chót là **20/11**[DATE].
* **SAI**: Hạn chót nộp bài là **20/11**[DATE]. (Lý do: Thiếu thực thể `FRACTION`, không đáp ứng yêu cầu sinh dữ liệu kết hợp)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Mới có **3/4**[FRACTION] số sinh viên nộp bài trong khi hạn chót là **20/11**[DATE].
* **SAI**: Mới có 3/4[FRACTION] số sinh viên nộp bài trong khi hạn chót là 20/11[DATE]. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["PHONE__INTEGER"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (SĐT & SỐ NGUYÊN)

## MỤC TIÊU CHÍNH

Tạo ra một câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể số: một là **Số điện thoại (PHONE)** và một là **Số nguyên (INTEGER)**. Câu phải được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi câu phải chứa:
* Một thực thể là chuỗi số trong ngữ cảnh số điện thoại, được gán nhãn `[PHONE]`.
* Một thực thể là chuỗi số trong ngữ cảnh đếm số lượng, được gán nhãn `[INTEGER]`.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**. Có thể chứa khoảng trắng hoặc dấu phân cách hàng nghìn (`.`) tùy theo loại thực thể.
* **GÁN NHÃN**: Nhãn `[PHONE]` hoặc `[INTEGER]` phải được đặt ngay sau thực thể tương ứng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, kết nối hai thực thể một cách logic. Ví dụ: gọi đến tổng đài để nhận giải thưởng, số lượng cuộc gọi đến một đầu số...
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** tạo câu mà hai thực thể có thể bị nhầm lẫn về ngữ cảnh (ví dụ: hai số điện thoại hoặc hai số đếm).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Ngữ Cảnh và Gán Nhãn
* **ĐÚNG**: Gọi tổng đài **1800 1008**[PHONE] để có cơ hội nhận giải thưởng **500.000**[INTEGER] đồng.
* **SAI**: Gọi tổng đài **1800 1008**[INTEGER] để có cơ hội nhận giải thưởng **500.000**[PHONE] đồng. (Lý do: Gán nhãn sai ngữ cảnh)

### Về Sự Tồn Tại Của Cả Hai Nhãn
* **ĐÚNG**: Hơn **10.000**[INTEGER] cuộc gọi trong ngày hôm qua được gọi đến **1900 1234**[PHONE].
* **SAI**: Hơn **10.000**[INTEGER] cuộc gọi được ghi nhận. (Lý do: Thiếu thực thể `PHONE`, không đáp ứng yêu cầu)

### Về Ngữ Cảnh Dễ Gây Nhầm Lẫn
* **ĐÚNG**: Hãy gọi **113**[PHONE] nếu bạn thấy có **113**[INTEGER] người đang tụ tập.
* **SAI**: Hãy gọi **113**[PHONE] nếu bạn thấy số điện thoại **0987654321**[PHONE]. (Lý do: Chứa 2 thực thể PHONE, không đúng yêu cầu)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Hơn **10.000**[INTEGER] cuộc gọi trong ngày hôm qua được gọi đến **1900 1234**[PHONE].
* **SAI**: Hơn 10.000[INTEGER] cuộc gọi trong ngày hôm qua được gọi đến 1900 1234[PHONE]. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["INTEGER__FLOAT"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (SỐ NGUYÊN & SỐ THẬP PHÂN)

## MỤC TIÊU CHÍNH

Tạo ra một câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể số: một là **Số nguyên lớn (INTEGER)** có dấu `.` ngăn cách hàng nghìn, và một là **Số thập phân (FLOAT)** có dấu `,` ngăn cách phần thập phân. Câu phải được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi câu phải chứa:
* Một thực thể `[INTEGER]` có dạng `([Số]\.)+[Số]{3}`.
* Một thực thể `[FLOAT]` có dạng `[Số],[Số]`.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**.
* **DẤU PHÂN CÁCH**: `.` cho hàng nghìn (INTEGER), `,` cho thập phân (FLOAT).
* **GÁN NHÃN**: Nhãn `[INTEGER]` hoặc `[FLOAT]` phải được đặt ngay sau thực thể tương ứng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, kết nối hai thực thể một cách logic, thường trong bối cảnh tài chính, khoa học, kỹ thuật.
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** hoán đổi vai trò của dấu `.` và `,`.
* **TUYỆT ĐỐI KHÔNG** tạo câu chỉ chứa một loại thực thể.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Ngữ Cảnh và Gán Nhãn
* **ĐÚNG**: Lô hàng trị giá **25.000.000**[INTEGER] đồng có tỷ trọng là **0,8**[FLOAT].
* **SAI**: Lô hàng trị giá **25.000.000**[FLOAT] đồng có tỷ trọng là **0,8**[INTEGER]. (Lý do: Gán nhãn sai ngữ cảnh và định dạng)

### Về Định Dạng
* **ĐÚNG**: Anh ấy đạt hệ số lợi nhuận **2,5**[FLOAT] từ **1.000**[INTEGER] đô la tiền vốn.
* **SAI**: Anh ấy đạt hệ số lợi nhuận **2.5**[FLOAT] từ **1,000**[INTEGER] đô la tiền vốn. (Lý do: Sai quy ước dấu phân cách cho cả hai thực thể)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Có **1.000**[INTEGER] người ra kết quả **1,25**[FLOAT] cho bài toán này.
* **SAI**: Có 1.000[INTEGER] người ra kết quả 1,25[FLOAT] cho bài toán này. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["NUMBER_RANGE__SPORT_SCORE"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (KHOẢNG SỐ & TỶ SỐ)

## MỤC TIÊU CHÍNH

Tạo ra một câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể có cùng pattern `Số-Số`: một là **Khoảng giá trị (NUMBER_RANGE)** và một là **Tỷ số (SPORT_SCORE)**. Câu phải được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi câu phải chứa:
* Một thực thể `[Số]-[Số][NUMBER_RANGE]` trong ngữ cảnh khoảng giá trị, số lượng, thời gian...
* Một thực thể `[Số]-[Số][SPORT_SCORE]` trong ngữ cảnh tỷ số trận đấu thể thao.

### 2. Quy Tắc Định Dạng
* **SỐ**: Bắt buộc phải là **chữ số**.
* **DẤU PHÂN CÁCH**: Bắt buộc là dấu gạch nối (`-`).
* **GÁN NHÃN**: Nhãn `[NUMBER_RANGE]` hoặc `[SPORT_SCORE]` phải được đặt ngay sau thực thể tương ứng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, kết nối hai thực thể một cách logic.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** tạo câu mà cả hai thực thể đều là khoảng số hoặc đều là tỷ số.
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Ngữ Cảnh và Gán Nhãn
* **ĐÚNG**: Trận đấu diễn ra lúc **19-21**[NUMBER_RANGE] giờ đã kết thúc với tỉ số **2-1**[SPORT_SCORE].
* **SAI**: Trận đấu diễn ra lúc **19-21**[SPORT_SCORE] giờ đã kết thúc với tỉ số **2-1**[NUMBER_RANGE]. (Lý do: Gán nhãn sai ngữ cảnh)

### Về Sự Tồn Tại Của Cả Hai Nhãn
* **ĐÚNG**: Với tỷ số **3-0**[SPORT_SCORE], sẽ có khoảng **3-5**[NUMBER_RANGE] người được nhận quà.
* **SAI**: Với tỷ số **3-0**[SPORT_SCORE], nhiều người sẽ được nhận quà. (Lý do: Thiếu thực thể `NUMBER_RANGE`)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Với tỷ số **3-0**[SPORT_SCORE], sẽ có khoảng **3-5**[NUMBER_RANGE] người được nhận quà.
* **SAI**: Với tỷ số 3-0[SPORT_SCORE], sẽ có khoảng 3-5[NUMBER_RANGE] người được nhận quà. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["ROMAN_NUMERAL__ALPHANUM_ID"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (SỐ LA MÃ & MÃ CHỮ)

## MỤC TIÊU CHÍNH

Tạo ra một câu văn tiếng Việt tự nhiên trong đó cùng một chuỗi chữ cái La Mã được sử dụng với hai ý nghĩa khác nhau: một là **Số La Mã (ROMAN_NUMERAL)** và một là **Mã chữ (ALPHANUM_ID)**. Câu phải được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi câu phải chứa cùng một chuỗi ký tự, nhưng được gán nhãn khác nhau dựa trên ngữ cảnh:
* Một thực thể là `[chuỗi][ROMAN_NUMERAL]` trong ngữ cảnh số thứ tự (vua, thế kỷ, chương...).
* Một thực thể là `[chuỗi][ALPHANUM_ID]` trong ngữ cảnh mã hiệu, ký hiệu, mật mã...

### 2. Quy Tắc Định Dạng
* **CHUỖI KÝ TỰ**: Bắt buộc phải là chuỗi chữ cái La Mã viết hoa hợp lệ.
* **GÁN NHÃN**: Nhãn `[ROMAN_NUMERAL]` hoặc `[ALPHANUM_ID]` phải được đặt ngay sau thực thể.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

### 3. Ngữ Cảnh
* Câu văn phải tự nhiên, tạo ra một sự đối lập hoặc liên kết thú vị giữa hai ý nghĩa của cùng một chuỗi ký tự.

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** sử dụng hai chuỗi ký tự khác nhau. Phải là cùng một chuỗi.
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Ngữ Cảnh và Gán Nhãn
* **ĐÚNG**: Các nhà sử học tin rằng Vua Louis **XIV**[ROMAN_NUMERAL] đã dùng chuỗi **XIV**[ALPHANUM_ID] làm mật mã.
* **SAI**: Các nhà sử học tin rằng Vua Louis **XIV**[ALPHANUM_ID] đã dùng chuỗi **XIV**[ROMAN_NUMERAL] làm mật mã. (Lý do: Gán nhãn sai ngữ cảnh)

### Về Việc Dùng Cùng Một Chuỗi
* **ĐÚNG**: Nghị quyết có mã hiệu **IX**[ALPHANUM_ID] đã được thông qua tại Đại hội Đảng lần thứ **IX**[ROMAN_NUMERAL].
* **SAI**: Nghị quyết có mã hiệu **IX**[ALPHANUM_ID] đã được thông qua tại Đại hội Đảng lần thứ **X**[ROMAN_NUMERAL]. (Lý do: Sử dụng hai chuỗi ký tự khác nhau, không đúng yêu cầu)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Nghị quyết có mã hiệu **IX**[ALPHANUM_ID] đã được thông qua tại Đại hội Đảng lần thứ **IX**[ROMAN_NUMERAL].
* **SAI**: Nghị quyết có mã hiệu IX[ALPHANUM_ID] đã được thông qua tại Đại hội Đảng lần thứ IX[ROMAN_NUMERAL]. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["MATH_EXPR__NUMBER_RANGE"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (PHÉP TRỪ & KHOẢNG SỐ)

## MỤC TIÊU CHÍNH

Tạo ra một câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể có cùng pattern `Số-Số`:
1.  Một là **Phép trừ (MATH_EXPR)**.
2.  Hai là **Khoảng số (NUMBER_RANGE)**.

Câu văn phải được xây dựng để làm nổi bật sự khác biệt về ngữ nghĩa và được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi câu phải chứa:
* Một thực thể `MATH_EXPR` trong ngữ cảnh một phép tính trừ. Thực thể này phải ở dạng biểu thức (`Số-Số`).
* Một thực thể `NUMBER_RANGE` (`Số-Số`) trong ngữ cảnh một khoảng giá trị, số lượng.

### 2. Yêu Cầu Về Ngữ Cảnh
* **Bắt buộc** phải có các từ khóa gợi ý để phân biệt rõ ràng hai ngữ cảnh.
    * **Với MATH_EXPR**: Dùng các từ như "phép tính", "tính toán", "kết quả", "bằng", "là âm", "trừ đi"...
    * **Với NUMBER_RANGE**: Dùng các từ như "khoảng", "dao động", "từ...đến", "mất khoảng", "khoảng chừng"...
* Khuyến khích tạo ra các câu có tính so sánh, đối chiếu trực tiếp giữa hai ý nghĩa.

### 3. Quy Tắc Định Dạng & Gán Nhãn
* **SỐ**: Bắt buộc phải là **chữ số**.
* **DẤU PHÂN CÁCH**: Bắt buộc là dấu gạch nối (`-`).
* **GÁN NHÃN**: Nhãn `[MATH_EXPR]` hoặc `[NUMBER_RANGE]` phải được đặt ngay sau thực thể tương ứng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

### 4. Điều Cấm
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dạng Biểu Thức (Không Cần Dấu Bằng)
* **ĐÚNG**: Anh ấy mất khoảng **2-3**[NUMBER_RANGE] giây để thực hiện phép tính **3-2**[MATH_EXPR].
* **SAI**: Anh ấy mất khoảng **2-3**[MATH_EXPR] giây để thực hiện phép tính **3-2**[NUMBER_RANGE]. (Lý do: Gán nhãn sai ngữ cảnh. "2-3 giây" là khoảng thời gian, "phép tính 3-2" là biểu thức toán học)

### Về Dạng Phương Trình (Có Dấu Bằng)
* **ĐÚNG**: Nhiệt độ không phải là kết quả của phép tính **5-10=-5**[MATH_EXPR] độ, mà nó đang dao động trong khoảng **5-10**[NUMBER_RANGE] độ C.
* **SAI**: Nhiệt độ không phải là kết quả của phép tính **5-10=-5**[NUMBER_RANGE] độ. (Lý do: Gán nhãn sai, đây là một phương trình toán học)

### Về Sự Rõ Ràng Của Ngữ Cảnh
* **ĐÚNG**: Đừng nhầm lẫn, số lượng nhân sự chỉ còn **50-20**[MATH_EXPR] người, chứ không phải là tuyển thêm trong khoảng **20-50**[NUMBER_RANGE] người.
* **SAI**: Công ty có **50-20**[MATH_EXPR] người và cần thêm **20-50**[NUMBER_RANGE] người. (Lý do: Câu văn không tự nhiên, ngữ cảnh yếu và không rõ ràng)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Đừng nhầm lẫn, số lượng nhân sự chỉ còn **50-20**[MATH_EXPR] người, chứ không phải là tuyển thêm trong khoảng **20-50**[NUMBER_RANGE] người.
* **SAI**: Đừng nhầm lẫn, số lượng nhân sự chỉ còn 50-20[MATH_EXPR] người, chứ không phải là tuyển thêm trong khoảng 20-50[NUMBER_RANGE] người. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #

TAG_TO_PROMPT["MATH_EXPR__SPORT_SCORE"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (PHÉP TRỪ & TỶ SỐ)

## MỤC TIÊU CHÍNH

Tạo ra một câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể có cùng pattern `Số-Số`:
1.  Một là **Phép trừ (MATH_EXPR)**.
2.  Hai là **Tỷ số (SPORT_SCORE)**.

Câu văn phải được xây dựng để làm nổi bật sự khác biệt về ngữ nghĩa và được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi
Mỗi câu phải chứa:
* Một thực thể `MATH_EXPR` trong ngữ cảnh một phép tính trừ. Có thể ở dạng biểu thức (`Số-Số`) hoặc phương trình (`Số-Số=Số`).
* Một thực thể `SPORT_SCORE` (`Số-Số`) trong ngữ cảnh tỷ số một trận đấu.

### 2. Yêu Cầu Về Ngữ Cảnh
* **Bắt buộc** phải có các từ khóa gợi ý để phân biệt rõ ràng hai ngữ cảnh.
    * **Với MATH_EXPR**: Dùng các từ như "phép tính", "tính toán", "kết quả", "bằng", "trừ đi"...
    * **Với SPORT_SCORE**: Dùng các từ như "tỷ số", "thắng", "thua", "hòa", "dẫn trước", "trận đấu"...
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

### 3. Quy Tắc Định Dạng & Gán Nhãn
* **SỐ**: Bắt buộc phải là **chữ số**.
* **DẤU PHÂN CÁCH**: Bắt buộc là dấu gạch nối (`-`).
* **GÁN NHÃN**: Nhãn `[MATH_EXPR]` hoặc `[SPORT_SCORE]` phải được đặt ngay sau thực thể tương ứng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

---

## VÍ DỤ SO SÁNH CỤ THỂ

### Về Dạng Biểu Thức (Không Cần Dấu Bằng)
* **ĐÚNG**: Đội nhà đang dẫn trước với tỷ số **2-0**[SPORT_SCORE], một thế trận dễ dàng như làm phép tính **2-0**[MATH_EXPR] vậy.
* **SAI**: Đội nhà đang dẫn trước với tỷ số **2-0**[MATH_EXPR]. (Lý do: Gán nhãn sai, "tỷ số" và "dẫn trước" là ngữ cảnh của SPORT_SCORE)

### Về Dạng Phương Trình (Có Dấu Bằng)
* **ĐÚNG**: Kết quả trận đấu là **3-1**[SPORT_SCORE], chứ không phải là phép tính **3-1=2**[MATH_EXPR].
* **SAI**: Kết quả trận đấu là **3-1**[MATH_EXPR]. (Lý do: Gán nhãn sai, "kết quả trận đấu" là ngữ cảnh của SPORT_SCORE)

### Về Ngữ Cảnh và Gán Nhãn
* **ĐÚNG**: Thắng một trận đấu với tỷ số **1-0**[SPORT_SCORE] mang lại nhiều cảm xúc hơn việc giải đúng bài toán **1-0=1**[MATH_EXPR].
* **SAI**: Thắng một trận đấu với tỷ số **1-0**[SPORT_SCORE] mang lại nhiều cảm xúc hơn việc giải đúng bài toán **1-0=1**[SPORT_SCORE]. (Lý do: Gán nhãn sai cho thực thể thứ hai)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Thắng một trận đấu với tỷ số **1-0**[SPORT_SCORE] mang lại nhiều cảm xúc hơn việc giải đúng bài toán **1-0=1**[MATH_EXPR].
* **SAI**: Thắng một trận đấu với tỷ số 1-0[SPORT_SCORE] mang lại nhiều cảm xúc hơn việc giải đúng bài toán 1-0=1[MATH_EXPR]. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #

# TAG_TO_PROMPT["MATH_EXPR__TIME_RANGE"] = """
# # YÊU CẦU SINH DỮ LIỆU KẾT HỢP (PHÉP TRỪ & KHUNG GIỜ)

# ## MỤC TIÊU CHÍNH

# Tạo ra một câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể có cùng pattern `Số-Số`:
# 1.  Một là **Phép trừ (MATH_EXPR)**.
# 2.  Hai là **Khung giờ (TIME_RANGE)**.

# Câu văn phải được xây dựng để làm nổi bật sự khác biệt về ngữ nghĩa và được gán nhãn chính xác.

# ---

# ## BỘ QUY TẮC

# ### 1. Cấu Trúc Cốt Lõi
# Mỗi câu phải chứa:
# * Một thực thể `MATH_EXPR` trong ngữ cảnh một phép tính trừ. Có thể ở dạng biểu thức (`Số-Số`) hoặc phương trình (`Số-Số=Số`).
# * Một thực thể `TIME_RANGE` (`Số-Số`) trong ngữ cảnh một khung giờ trong ngày.

# ### 2. Yêu Cầu Về Ngữ Cảnh
# * **Bắt buộc** phải có các từ khóa gợi ý để phân biệt rõ ràng hai ngữ cảnh.
#     * **Với MATH_EXPR**: Dùng các từ như "phép tính", "tính nhẩm", "kết quả", "bằng"...
#     * **Với TIME_RANGE**: Dùng các từ như "giờ", "khung giờ", "làm việc", "cuộc họp", "ca làm", "từ...đến"...
# * **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

# ### 3. Quy Tắc Định Dạng & Gán Nhãn
# * **SỐ**: Bắt buộc phải là **chữ số**.
# * **DẤU PHÂN CÁCH**: Bắt buộc là dấu gạch nối (`-`).
# * **GÁN NHÃN**: Nhãn `[MATH_EXPR]` hoặc `[TIME_RANGE]` phải được đặt ngay sau thực thể tương ứng.
# * **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

# ---

# ## VÍ DỤ SO SÁNH CỤ THỂ

# ### Về Dạng Biểu Thức (Không Cần Dấu Bằng)
# * **ĐÚNG**: Sếp yêu cầu tôi tính lại phép trừ **11-9**[MATH_EXPR] trong khung giờ làm việc **9-11**[TIME_RANGE].
# * **SAI**: Sếp yêu cầu tôi tính lại phép trừ **11-9**[TIME_RANGE] trong khung giờ làm việc **9-11**[MATH_EXPR]. (Lý do: Gán nhãn sai ngữ cảnh)

# ### Về Dạng Phương Trình (Có Dấu Bằng)
# * **ĐÚNG**: Cuộc họp sẽ diễn ra trong khoảng **2-4**[TIME_RANGE] giờ chiều, đừng nhầm lẫn nó với kết quả của phép tính **2-4=-2**[MATH_EXPR] nhé.
# * **SAI**: Cuộc họp sẽ diễn ra trong khoảng **2-4**[MATH_EXPR] giờ chiều. (Lý do: Gán nhãn sai, "giờ chiều" và "khoảng" là ngữ cảnh của TIME_RANGE)

# ### Về Sự Tồn Tại Của Cả Hai Nhãn
# * **ĐÚNG**: Tôi phải làm việc ca **8-12**[TIME_RANGE] và phải giải quyết xong bài toán **8-12=-4**[MATH_EXPR] này.
# * **SAI**: Tôi phải làm việc ca **8-12**[TIME_RANGE] và phải giải quyết xong bài toán khó này. (Lý do: Thiếu thực thể `MATH_EXPR`)

# ### Về Lỗi In Đậm (Highlight)
# * **ĐÚNG**: Tôi phải làm việc ca **8-12**[TIME_RANGE] và phải giải quyết xong bài toán **8-12=-4**[MATH_EXPR] này.
# * **SAI**: Tôi phải làm việc ca 8-12[TIME_RANGE] và phải giải quyết xong bài toán 8-12=-4[MATH_EXPR] này. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
# """.strip()

# ============================================= #

TAG_TO_PROMPT["DATE_RANGE__MATH_EXPR"] = """
# YÊU CẦU SINH DỮ LIỆU KẾT HỢP (KHOẢNG NGÀY & PHÉP TÍNH TRỪ - DẠNG NÂNG CAO)

## MỤC TIÊU CHÍNH

Tạo ra các câu văn tiếng Việt tự nhiên chứa đồng thời hai loại thực thể có **cấu trúc bề mặt phức tạp và dễ gây nhầm lẫn**:
1.  Một là **Khoảng ngày (DATE_RANGE)**.
2.  Hai là **Phép tính trừ (MATH_EXPR)**.

Câu văn phải được xây dựng để AI buộc phải dựa vào ngữ cảnh để phân biệt và phải được gán nhãn chính xác.

---

## BỘ QUY TẮC

### 1. Cấu Trúc Cốt Lõi Bắt Buộc
Prompt này chỉ tập trung vào việc tạo ra các câu chứa các cặp thực thể mơ hồ sau:
* **Dạng 1**: `[Số]/[Số]-[Số]/[Số]`
* **Dạng 2**: `[Số]-[Số]/[Số]`

Trong mỗi câu, một thực thể phải được diễn giải là `DATE_RANGE` và một thực thể là `MATH_EXPR`.

### 2. Yêu Cầu Về Ngữ Cảnh
* **Bắt buộc** phải có các từ khóa gợi ý để phân biệt rõ ràng hai ngữ cảnh.
    * **Với DATE_RANGE**: Dùng các từ như "sự kiện", "lịch trình", "diễn ra từ ngày...đến ngày...", "kéo dài", "tháng", "áp dụng từ...đến..."
    * **Với MATH_EXPR**: Dùng các từ như "phép tính", "biểu thức", "tính toán", "kết quả", "bằng", "trừ đi"...
* Khuyến khích tạo ra các câu có tính so sánh, đối chiếu trực tiếp hai ý nghĩa.
* **TUYỆT ĐỐI KHÔNG** tạo câu mà ngữ cảnh mập mờ, không thể phân biệt được hai loại thực thể.

### 3. Quy Tắc Định Dạng & Gán Nhãn
* **SỐ**: Bắt buộc phải là **chữ số**.
* **DẤU PHÂN CÁCH**: Sử dụng đúng các dấu `/` và `-` theo từng cấu trúc.
* **GÁN NHÃN**: Nhãn `[DATE_RANGE]` hoặc `[MATH_EXPR]` phải được đặt ngay sau thực thể tương ứng.
* **IN ĐẬM (HIGHLIGHT)**: Toàn bộ các cấu trúc cốt lõi trong câu phải được bôi đậm (`**...**`).

---

## VÍ DỤ SO SÁNH CỤ THỂ CHO TỪNG TRƯỜNG HỢP

### Về Pattern `[Số]/[Số] - [Số]/[Số]`
* **ĐÚNG**: Trong thời gian từ **1/4-1/5**[DATE_RANGE], chúng ta phải hoàn thành việc tính toán giá trị biểu thức **1/4-1/5**[MATH_EXPR].
* **SAI**: Trong thời gian từ **1/4-1/5**[DATE_RANGE], chúng ta phải hoàn thành việc tính toán giá trị biểu thức **1/4-1/5**[DATE_RANGE]. (Lý do: Gán nhãn sai cho thực thể thứ hai)
* **ĐÚNG**: Hãy tính xem **1/2-1/4**[MATH_EXPR] bằng bao nhiêu trong khi chờ đợi đến đợt khuyến mãi **1/2-1/4**[DATE_RANGE].
* **SAI**: Hãy tính xem **1/2-1/4**[MATH_EXPR] bằng bao nhiêu trong khi chờ đợi đến đợt khuyến mãi sắp tới. (Lý do: Thiếu thực thể `DATE_RANGE` theo yêu cầu)

### Về Pattern `[Số] - [Số]/[Số]`
* **ĐÚNG**: Lịch nghỉ từ ngày **1-2/3**[DATE_RANGE] được ấn định sau khi mọi người đã nộp bài giải cho phép tính **1-2/3**[MATH_EXPR].
* **SAI**: Lịch nghỉ từ ngày **1-2/3**[MATH_EXPR] đã được ấn định. (Lý do: Gán nhãn sai, "lịch nghỉ từ ngày" là ngữ cảnh của DATE_RANGE)
* **ĐÚNG**: Sự kiện diễn ra từ **1-15/4**[DATE_RANGE] có phần thi giải đáp án của bài toán **1-1/4**[MATH_EXPR].
* **SAI**: Sự kiện diễn ra từ **1-15/4**[DATE_RANGE] có phần thi giải đáp án của một bài toán khó. (Lý do: Thiếu thực thể `MATH_EXPR` theo yêu cầu)

### Về Lỗi In Đậm (Highlight)
* **ĐÚNG**: Hãy tính xem **1/2-1/4**[MATH_EXPR] bằng bao nhiêu trong khi chờ đợi đến đợt khuyến mãi **1/2-1/4**[DATE_RANGE].
* **SAI**: Hãy tính xem 1/2-1/4[MATH_EXPR] bằng bao nhiêu trong khi chờ đợi đến đợt khuyến mãi 1/2-1/4[DATE_RANGE]. (Lý do: Thiếu in đậm cho các cấu trúc cốt lõi)
""".strip()

# ============================================= #
# ============================================= #
# ============================================= #
# ============================================= #