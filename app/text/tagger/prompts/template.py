REASONING_TEMPLATE_V0 = """Bạn là một chuyên gia ngôn ngữ và nhà văn sáng tạo, am hiểu sâu sắc về ngữ pháp và văn phong Tiếng Việt. Nhiệm vụ của bạn là tạo ra {{num_new_examples}} câu văn mẫu độc đáo, tự nhiên và chính xác dựa trên các yêu cầu cụ thể được cung cấp.

---

{{inputs}}

---

# QUY TRÌNH THỰC HIỆN

**Bước 1: Brainstorming (Phân tích và lên ý tưởng)**
Trước khi viết câu, hãy suy nghĩ và phác thảo {{num_new_examples}} ngữ cảnh (context) **khác biệt và không trùng lặp** với các ví dụ đã cho. {{context}} Với mỗi ngữ cảnh, hãy:
1.  **Mô tả ngắn gọn:** Nêu rõ tình huống giao tiếp là gì (ví dụ: {{situations}}), sau đó lựa chọn vai diễn (ví dụ: {{roles}}) và cảm xúc sẽ truyền đạt (ví dụ: {{emotions}}).
2.  **Quyết định có bao nhiêu lần Cấu trúc cần tạo xuất hiện trong câu.
3.  **Lựa chọn kiểu câu sẽ dùng:** câu kể, câu hỏi, câu cảm thán,... Các kiểu câu không nên lặp lại nhau.
4.  **Trùng lặp:** không được giống các ví dụ mẫu.
4.  **Phân tích tính tự nhiên:** Giải thích ngắn gọn tại sao việc chèn **Cấu trúc cần tạo** vào ngữ cảnh này lại tự nhiên và không vi phạm các quy tắc "cần tránh".

**Bước 2: Generating Examples (Tạo câu mẫu)**
Dựa trên các ngữ cảnh đã phân tích, hãy viết {{num_new_examples}} câu hoàn chỉnh. Đảm bảo rằng:
* Mỗi câu sử dụng một ngữ cảnh mà bạn đã phân tích và lên ý tưởng.
* Tất cả Cấu trúc cần tạo được **bôi đậm** chính xác.
* Câu văn phải tự nhiên, phù hợp với văn phong giao tiếp và đúng ngữ pháp Tiếng Việt. Vì tôi cần một ví dụ khó nên phải Tránh sử dụng những từ khóa để nhận biết cấu trúc cần tạo. Ví dụ với cấu trúc cần tạo liên quan đến ngày tháng, không nên sử dụng trực tiếp các từ đơn giản như "ngày", "hôm",...
* Ngoài Cấu trúc cần tạo, trong câu **tuyệt đối không được chứa bất kỳ chữ số nào khác** dưới mọi hình thức.
* Cố gắng tạo ra **ít nhất {{num_special_examples}} câu** chứa Cấu trúc cần tạo lặp lại từ {{num_min_repeat}} lần trở lên.

---

**Sau khi thực hiện đầy đủ quy trình trên, vui lòng trả về kết quả theo đúng định dạng sau và không cần giải thích gì thêm:**
<examples>
{{example_output_str}}
</examples>
""".strip()

# ============================================= #

REASONING_TEMPLATE_V1 = """Bạn là một chuyên gia ngôn ngữ và nhà văn sáng tạo, am hiểu sâu sắc về ngữ pháp và văn phong Tiếng Việt. Nhiệm vụ của bạn là tạo ra {{num_new_examples}} câu văn mẫu độc đáo, tự nhiên và chính xác dựa trên các yêu cầu cụ thể được cung cấp.

---

{{inputs}}

---

## YÊU CẦU VỀ SỰ SÁNG TẠO VÀ ĐA DẠNG
* **Đa dạng ngữ cảnh:** Không lặp lại các tình huống. Hãy sáng tạo các bối cảnh khác nhau về {{context}}.
* **Nhập vai và thể hiện cảm xúc:** Thay đổi văn phong giữa các câu. Đặt mình dưới các góc nhìn khác nhau ({{roles}}) với các trạng thái cảm xúc khác nhau ({{emotions}}) để câu văn có hồn và có chiều sâu hơn.
* **Đa dạng loại câu:** Kết hợp giữa câu kể, câu hỏi, câu cảm thán và câu mệnh lệnh để kết quả không bị đơn điệu.
* **Tăng độ phức tạp:** Cố gắng tạo ra **ít nhất {{num_special_examples}} câu** mà trong đó một hoặc cả hai dạng cấu trúc cốt lõi được lặp lại từ {{num_min_repeat}} lần trở lên.

---

## ĐIỀU CẤM (bổ sung):
* **TUYỆT ĐỐI KHÔNG** chứa bất kỳ chữ số nào khác ngoài các chữ số nằm trong cấu trúc cốt lõi được in đậm.

* **TUYỆT ĐỐI KHÔNG** sử dụng câu dạng trích dẫn trong dấu ngoặc kép ("...").
    * **VÍ DỤ ĐÚNG:** Anh chị yên tâm, ...
    * **VÍ DỤ SAI:** Nhân viên bán hàng tự tin: "Anh chị yên tâm, ..." (Lý do: Là câu trích dẫn và có phần giải thích "Nhân viên bán hàng tự tin")

* **TUYỆT ĐỐI KHÔNG** giải thich gì thêm về câu văn
    * **VÍ DỤ ĐÚNG:** Hãy cung cấp mã số nhân sự ...
    * **VÍ DỤ SAI** (Nhà đầu tư thận trọng) Hãy cung cấp mã số nhân sự ... (Lý do: Có phần giải thích "Nhà đầu tư thận trọng")
    * **VÍ DỤ SAI** Nhà đầu tư nghiêm giọng: "Tôi cần anh giải thích tại sao ..." (Lý do: Có phần giải thích "Nhà đầu tư nghiêm giọng" và là câu trích dẫn)

---

## YÊU CẦU VỀ KẾT QUẢ
**Trả về kết quả trong thẻ <examples> như định dạng dưới đây, không được thêm phần giải thích ngữ cảnh hay lời giải thích gì khác:**

<examples>
{{example_output_str}}
</examples>
""".strip()