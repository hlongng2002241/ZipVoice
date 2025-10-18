DATE_0 = """
Pattern: [ngày]/[tháng]
Cách sinh: Tạo ngày, tháng hợp lệ (ở dạng số, không phải dạng chữ). Nối chúng bằng "/". Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố.

## Ví dụ:
Hạn chót là **1/5**
Sự kiện diễn ra **2/9**
Hẹn gặp lại **10/10** nhé.
Toàn bộ đội ngũ cần lưu ý rằng hạn chót để gửi báo cáo là **1/5**, tuyệt đối không trễ hẹn.
Mình phải nộp bài luận cuối kỳ trước **1/5** nên tuần này chắc là sẽ bận lắm.
Nhóm mình định đi cắm trại ở Ba Vì dịp **2/9** này, cậu có muốn tham gia không?
Cảm ơn anh vì buổi trao đổi hôm nay, hẹn gặp lại anh trong cuộc họp dự án vào **10/10** nhé.
"""

# ============================================= #

DATE_1 = """
Pattern: [ngày]/[tháng]/[năm]
Cách sinh: Tạo ngày, tháng, năm đầy đủ (ở dạng số, không phải dạng chữ). Nối chúng bằng "/". Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố.

## Ví dụ:
Quốc khánh **02/09/1945** là thời khắc khai sinh ra nước Việt Nam
Để ý nhé, **05/09/2025** là ngày khai giảng đó.
Các hoạt động kỷ niệm Quốc khánh **02/09/1945** sẽ được tổ chức trang trọng trên cả nước.
Mẹ dặn này, con nhớ chuẩn bị xong sách vở và đồng phục trước **05/09/2025** nhé, đó là ngày khai giảng năm học mới đấy.
Em cần lưu ý rằng hợp đồng lao động hiện tại của mình sẽ hết hạn vào ngày **31/12/2026**, chúng ta cần thảo luận về việc gia hạn trước thời điểm đó.
"""

# ============================================= #

DATE_2 = """
Pattern: [tháng]/[năm]
Cách sinh: Tạo tháng, năm hợp lệ (ở dạng số, không phải dạng chữ). Nối bằng "/". Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố, có thể là mô tả tháng, quý, giai đoạn, ...

## Ví dụ:
hãy cố gắng trong quý **4/2025** mọi người ơi.
doanh thu giai đoạn **3/2025** khá ổn đó 
Báo cáo tài chính còn một chặng đường dài, hãy cố gắng hết sức trong quý **4/2025** mọi người ơi!
Mặc dù thị trường có nhiều biến động, thật đáng mừng là doanh thu giai đoạn **3/2025** của chúng ta vẫn khá ổn.
Ngân hàng xin thông báo thẻ tín dụng của quý khách sẽ hết hạn vào tháng **12/2027**. Chúng tôi sẽ tự động phát hành và gửi thẻ mới đến địa chỉ của quý khách.
"""

# ============================================= #

DATE_RANGE_0 = """
Pattern: yyyy-yyyy
Cách sinh: Tạo hai năm (năm sau > năm trước, ở dạng số, không phải dạng chữ). Nối bằng "-". Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố, có thể là mô tả nhiệm kỳ, giai đoạn, mùa giải, năm học, ...

## Ví dụ:
giai đoạn **2021-2025** là cả quá trình cố gắng
Kế hoạch phát triển kinh tế - xã hội của quốc gia giai đoạn **2021-2025** đặt mục tiêu tập trung vào chuyển đổi số và phát triển bền vững.
Đối với tôi, những năm tháng đại học **2021-2025** không chỉ là kiến thức trên giảng đường, mà còn là cả một quá trình nỗ lực để trưởng thành và định vị bản thân.
"""

# ============================================= #

DATE_RANGE_1 = """
Pattern: [ngày]/[tháng] - [ngày]/[tháng]/[năm]
Cách sinh: Tạo một ngày/tháng (d/m) và một ngày đầy đủ (d/m/y). Lưu ý, ngày, tháng và năm phải ở dạng số, không phải dạng chữ. Nối chúng lại bằng dấu "-". Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố, có thể là mô tả lịch trình, sự kiện ...

## Ví dụ:
sự kiện **20/1-22/2/2025** thật thú vị
chụẩn bị **4/5-6/5/2024** đi du lịch nhé
Nhớ chuyến đi Vũng Tàu hồi **4/5-6/5/2024** ghê, tuy ngắn ngày nhưng mà vui thật sự.
Chuỗi sự kiện "Chào Xuân Giáp Thìn" sẽ diễn ra từ **20/1-22/2/2025** với nhiều hoạt động văn hóa, nghệ thuật đặc sắc hứa hẹn sẽ mang đến cho du khách những trải nghiệm thật thú vị.
"""

# ============================================= #

DATE_RANGE_2 = """
Pattern: [tháng]/[năm] - [tháng]/[năm]
Cách sinh: Tạo hai mốc tháng/năm hợp lệ, trong đó mốc đầu tiên phải trước hoặc bằng mốc thứ hai. Lưu ý, tháng và năm phải ở dạng số, không phải dạng chữ. Nối chúng bằng " - ". Đặt vào câu có ngữ cảnh về một khoảng thời gian một cách tự nhiên, ví dụ như mô tả một giai đoạn, kế hoạch, chương trình, thời hạn, v.v.

## Ví dụ:
sự kiện **1/2024 - 2/2025** thật thú vị
chụẩn bị **5/2024 - 5/2025** sang dự án mới nhé
Chương trình khuyến mãi này sẽ kéo dài từ **11/2024 - 2/2025**.
Kế hoạch phát triển sản phẩm giai đoạn **9/2025 - 6/2026** đã được phê duyệt.
Thời hạn bảo hành cho thiết bị này là từ **01/2025 - 01/2027**, quý khách vui lòng lưu ý.
Chúng ta cần phân tích dữ liệu bán hàng trong khoảng thời gian **7/2024 - 12/2024** để lập báo cáo cuối năm.
Hợp đồng lao động của anh ấy có hiệu lực từ **03/2023 - 03/2026**.
"""

# ============================================= #

TIME_0 = """
Pattern: [giờ]:[phút]
Cách sinh: Tạo giờ và phút, lưu ý phải ở dạng số, không phải dạng chữ. Nối chúng bằng dấu ":". Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố

## Ví dụ:
lúc **10:30**
tôi đến trường suýt thì muộn, tầm **6:59**, may quá
cho bạn đến **10:30** phải hoàn thành công việc đấy
Em đã đặt lịch khám tổng quát cho mình tại bệnh viện vào lúc **10:30** thứ Hai tuần sau rồi ạ, anh nhớ đi nhé.
Mày ơi tao vừa qua được cổng trường xong. Tý nữa thì bị ghi tên, tao đến nơi đúng **6:59**, hú hồn!
Sáng nay đồng hồ báo thức không kêu, tôi cuống cuồng chạy đến trường, nhìn đồng hồ đã là **6:59**, suýt thì muộn giờ vào lớp, may quá!
Báo cáo này rất quan trọng, tôi cho anh đến đúng **10:30** sáng nay để hoàn thành và gửi lại cho tôi nhé.
"""

# ============================================= #

TIME_1 = """
Pattern: [giờ]:[phút]:[giây]
Cách sinh: Tạo giờ, phút và giây, lưu ý phải ở dạng số, không phải dạng chữ. Nối chúng bằng dấu ":". Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố

## Ví dụ:
 cho bạn đến **10:30:15** phải hoàn thành công việc đấy
tôi đến trường suýt thì muộn, tầm **6:59:00**, may quá
Hệ thống sẽ tự động đóng phiên đấu thầu vào lúc **10:30:15** sáng mai, mọi hồ sơ nộp sau thời điểm này sẽ không được chấp nhận.
Vận động viên đã hoàn thành phần thi bơi của mình với thành tích **6:59:00**, chỉ kém kỷ lục quốc gia vài phần trăm giây.
Tôi liếc nhìn đồng hồ điện tử, nó nhảy sang **6:59:00** đúng lúc tôi lách qua được.
"""

# ============================================= #

TIME_2 = """
Pattern: [số]h
Cách sinh: Tạo một số giờ, lưu ý phải ở dạng số, không phải dạng chữ. Thêm chữ "h" ngay sau. Đặt chúng vào câu có ngữ cảnh thời gian một cách tự nhiên mà không ép buộc tiền tố

## Ví dụ:
bay khoảng **3h**
cho bạn đến **10h** phải hoàn thành công việc đấy
tôi đến trường suýt thì muộn, tầm **7h**, may quá
Chuyến bay từ Hà Nội vào Đà Nẵng sẽ bay khoảng **3h** nếu tính cả thời gian làm thủ tục và chờ đợi.
Task này ưu tiên cao nhất, em phải hoàn thành và gửi lại cho anh trước **10h** sáng nay nhé.
Hú hồn mày ơi, tao vừa chạy thục mạng đến trường. Tưởng muộn rồi cơ, lúc đến nơi là gần **7h**, may thế!
"""

# ============================================= #

TIME_RANGE_0 = """
Pattern: [giờ]:[phút]-[giờ]:[phút]
Cách sinh: Tạo hai mốc thời gian (giờ:phút) hợp lệ, trong đó thời gian bắt đầu phải sớm hơn thời gian kết thúc. Lưu ý giờ và phút phải ở dạng số, không phải dạng chữ. Nối chúng bằng "-". Đặt vào câu có ngữ cảnh về một khoảng thời gian trong ngày một cách tự nhiên như ca làm việc, cuộc họp, giờ mở cửa, lịch trình sự kiện,...

## Ví dụ:
làm việc từ **9:30-11:30**
Cuộc họp hôm nay sẽ diễn ra trong khoảng **9:00-10:30** nhé mọi người.
Tôi có lịch bận từ **14:00-15:30** chiều nay.
Giờ nghỉ trưa của công ty là từ **12:00-13:00**.
Nhân viên đó chỉ làm việc bán thời gian, ca sáng từ **8:30-11:30** hàng ngày.
Cửa hàng sẽ tạm đóng cửa để kiểm kê kho trong khung giờ **13:00-14:30** ngày mai.
"""

# ============================================= #

INTEGER_0 = """
Pattern: [số_nguyên]
Cách sinh: Tạo một số nguyên (có thể âm hoặc dương) không chứa dấu ngăn cách, bắt buộc phải ở dạng số, không phải dạng chữ. Đặt trong ngữ cảnh số đếm thông thường một cách tự nhiên mà không ép buộc tiền tố

## Ví dụ:
có **30** học sinh
tòa nhà **20** tầng
kết quả chắc là **-1** đó
Chuyến dã ngoại lần này của chúng ta có tổng cộng **30** học sinh tham gia, các em nhớ điểm danh đầy đủ nhé.
Dự án chung cư này bao gồm hai tòa nhà **20** tầng với đầy đủ tiện ích như hồ bơi, phòng gym và khu vui chơi trẻ em.
"""

# ============================================= #

INTEGER_1 = """
Pattern: ([số]\.)+[số]{3}
Cách sinh: Tạo một số nguyên rất lớn. Dùng dấu "." làm dấu ngăn cách lặp lại cho mỗi nhóm 3 chữ số từ phải sang trái. Đặt trong ngữ cảnh sử dụng số một cách tự nhiên mà không ép buộc tiền tố. Chú ý bắt buộc phải ở dạng số, không phải dạng chữ.

## Ví dụ:
hơn **1.000** người
**1.000.000** đồng
hơn **2.000.000** cây được trồng
Đêm nhạc gây quỹ từ thiện đã thành công ngoài mong đợi khi thu hút hơn **1.000** người đến tham dự và ủng hộ.
Người thắng cuộc trong cuộc thi sáng tạo video lần này sẽ nhận được giải thưởng tiền mặt trị giá **1.000.000** đồng.
Chiến dịch "Vì một Việt Nam xanh" đã đặt mục tiêu trồng mới **2.000.000** cây xanh trên khắp cả nước trong năm nay.
"""

# ============================================= #

FLOAT_0 = """
Pattern: [số].[số]
Cách sinh: Tạo số thập phân dùng dấu "." làm dấu ngăn. Đặt trong ngữ cảnh sử dụng số một cách tự nhiên mà không ép buộc tiền tố, có thể mô tả đo lường kỹ thuật, khoa học, tài chính, kinh tế, ... Chú ý bắt buộc phải ở dạng số, không phải dạng chữ.

## Ví dụ:
chiều dài **1.25** mét
**5.5**
**9.5** điểm
lãi suất **8.5**%
Theo bản vẽ thiết kế, chi tiết máy này phải có chiều dài chính xác là **1.25**m
Bộ phim được giới chuyên môn đánh giá ở mức trung bình với số điểm **5.5**
Với bài thi luận gần như hoàn hảo, Lan đã xuất sắc giành được **9.5** điểm và trở thành thủ khoa môn Văn của trường
Ngân hàng chúng tôi vừa ra mắt sản phẩm tiết kiệm với mức lãi suất ưu đãi lên tới **8.5**% một năm.
"""

# ============================================= #

FLOAT_1 = """
Pattern: ([số]\.)+[số]{3},[số]+
Cách sinh: Tạo số có phần nguyên lớn và phần thập phân. Với phần nguyên, dùng dấu "." lặp lại để ngăn cách mỗi nhóm 3 chữ số. Dùng dấu "," làm dấu ngăn thập phân. Đặt trong ngữ cảnh sử dụng số một cách tự nhiên mà không ép buộc tiền tố. Chú ý bắt buộc phải ở dạng số, không phải dạng chữ.

## Ví dụ:
chuyển khoản **50.000.000,45** đồng
mảnh đất rộng **1.000,5** ha
Sau khi đối soát, bộ phận kế toán xác nhận số tiền công ty cần thanh toán cho đối tác chính xác là **50.000.000,45** đồng, bao gồm cả các chi phí phát sinh lẻ
Chính phủ vừa phê duyệt chủ trương chuyển đổi mục đích sử dụng của một khu đất nông nghiệp rộng **1.000,5** ha để xây dựng khu công nghệ cao.
"""

# ============================================= #

FRACTION_0 = """
Pattern: [số]/[số]
Cách sinh: Tạo một phân số. Đặt nó vào các ngữ cảnh đa dạng như nấu ăn, chia chác, thống kê, tỷ lệ, ... mà không ép buộc tiền tố. Chú ý bắt buộc phải ở dạng số, không phải dạng chữ.

## Ví dụ:
**1/2** muỗng cà phê
chiếm **3/4** thị phần
Tỷ lệ nam/nữ là **3/4**
khả năng xảy là **1/1000**
khả năng xảy là **1/1.000.000**
Để món thịt kho đậm đà hơn, bạn hãy cho thêm **1/2** muỗng cà phê bột quế vào cùng lúc ướp gia vị.
Báo cáo phân tích thị trường chỉ ra rằng chỉ riêng ba ông lớn trong ngành đã chiếm **3/4** thị phần, không để lại nhiều cơ hội cho các doanh nghiệp nhỏ.
Nghiên cứu trên loài chim này cho thấy trong một đàn, tỷ lệ nam/nữ thường duy trì ở mức xấp xỉ **3/4** để đảm bảo sự phát triển của bầy.
Mặc dù khả năng trúng giải đặc biệt chỉ là **1/1000**, rất nhiều người vẫn hào hứng tham gia vì giá trị giải thưởng quá lớn.
Công ty bảo hiểm tính toán rằng khả năng xảy ra một vụ tai nạn máy bay do lỗi kỹ thuật chỉ là **1/1.000.000**, nên mức phí bảo hiểm hàng không tương đối thấp.
Hiện tại, quỹ đầu tư của chúng tôi phân bổ **1/3** danh mục vào cổ phiếu, **1/3** vào trái phiếu và **1/3** còn lại vào bất động sản để đảm bảo an toàn và tối ưu lợi nhuận.
"""

# ============================================= #

MEASUREMENT_0 = """
Pattern: [số]? + [đơn_vị]
Cách sinh: Chọn một ký hiệu đơn vị (m, kg, GB...), chú ý bắt buộc phải sử dụng dạng ký hiệu, không được phiên âm sang dạng chữ. Đặt nó vào câu văn tự nhiên. Một số trường hợp có số đứng trước (số ở dạng số, không phải dạng chữ), một số trường hợp không có. Quan trọng: Không ép buộc phải có từ khóa cố định, hãy để các từ xung quanh tạo ra ngữ cảnh đo lường. Chú ý không được tạo một khoảng số, ví dụ như "2m - 3m" hoặc "1.5 - 2 m", ...

## Ví dụ:
Quãng đường này dài **10km**.
Anh ấy nặng **75kg**,
Dung lượng của file là **2GB**
Đơn vị chuẩn là **m**, không phải **cm**.
Nồng độ cồn **50 mg/100ml**.
Mục tiêu của tôi sáng nay là hoàn thành cự ly chạy bộ **10km** quanh công viên trong vòng một tiếng
Để được thi đấu ở hạng cân này, các vận động viên phải đảm bảo trọng lượng cơ thể không vượt quá **75kg**.
Ổ đĩa của tôi báo sắp đầy rồi, chỉ còn trống có **2GB** thôi, phải dọn dẹp bớt file rác mới được.
Anh xem lại bản vẽ đi, tất cả các kích thước trong đây đều đang tính theo đơn vị chuẩn là **m**, không phải **cm** đâu, kẻo thi công lại nhầm lẫn.
"""

# ============================================= #

NUMBER_RANGE_0 = """
Pattern: [số] + [đơn_vị]? "-" [số] + [đơn_vị]?
Cách sinh: Tạo hai số (số ở dạng số, không phải dạng chữ). Nối chúng bằng dấu "-". Có thể thêm đơn vị ở cuối. Đặt trong ngữ cảnh sử dụng khoảng số một cách tự nhiên mà không ép buộc tiền tố.

## Ví dụ:
dao động **25-30 độ C**
lãi suất **1.2 - 2.3%**
vận tốc **1.2m - 1.5m/s**
biến thiên trong khoảng **-1 - 1**
tính ra mất **2-3** cái 
Để ấp trứng thành công, máy ấp phải duy trì nhiệt độ ổn định trong ngưỡng **25-30 độ C**
Các chuyên gia nhận định, lạm phát có thể sẽ biến động trong khoảng **1.2 - 2.3%** trong quý tới do ảnh hưởng từ giá xăng dầu.
Ngân hàng nhà nước vừa công bố gói vay ưu đãi cho doanh nghiệp vừa và nhỏ với mức lãi suất chỉ từ **1.2 - 2.3%** một năm.
Cánh tay robot này có thể di chuyển các vật thể với vận tốc tùy chỉnh, khoảng **1.2m - 1.5m/s**, tùy thuộc vào khối lượng của chúng.
Giá trị của hàm số sin và cosin luôn luôn biến thiên trong khoảng **-1 - 1**, không bao giờ vượt ra ngoài đoạn này.
Cái bản lề cửa này hay bị gãy chốt nhựa lắm, mỗi lần thay là tôi tính ra mất **2-3** cái dự phòng cho chắc.
"""

# ============================================= #

MATH_EXPR_0 = """
Pattern: [số] + [toán_tử] + [số] ...
Cách sinh: Tạo một biểu thức gồm hai hay nhiều số dạng số tự nhiên, số thập phân, phân số và các toán tử ở giữa (+, -, *, /, ^, =). Đặt trong ngữ cảnh sử dụng phép tính hay công thức một cách tự nhiên mà không ép buộc tiền tố, có thể mô tả đo lường kỹ thuật, khoa học, tài chính, kinh tế, ... Chú ý bắt buộc phải ở dạng số, không phải dạng chữ.

## Ví dụ:
hãy làm bài **2 + 2 = 4**
có thể **1/2 + 1/2 * 5 = 1** được không
của bạn hết **25.000 * (1 + 10%) = 27.500** đồng
Anh không cần giải thích những điều phức tạp. Vấn đề này cũng đơn giản như việc **2 + 2 = 4** thôi, ai cũng có thể hiểu được.
Tôi đang kiểm tra lại công thức trong file excel mà không hiểu sao nó cứ trả về kết quả sai. Chẳng lẽ logic **1/2 + 1/2 * 5 = 1** của tôi có vấn đề ở đâu đó?
Nếu nó tăng thêm thì giá mới sẽ là **25.000 * (1 + 10%) = 27.500** đồng. Anh thấy tiềm năng tăng trưởng của mã này thế nào?
"""

# ============================================= #

PHONE_0 = """
Pattern: [số_điện_thoại]
Cách sinh: Tạo một SĐT hợp lệ. Đặt nó trong câu có bối cảnh liên quan đến liên lạc một cách tự nhiên mà không ép buộc tiền tố, có thể sử dụng các từ như "số điện thoại", "gọi cho", "Zalo của", "liên hệ qua số"

## Ví dụ:
SĐT: **0987654321**
Gọi cho tôi qua số **0912345678**.
Zalo của anh là **0398765432**.
bạn nên để ý đầu số **024** gọi đến nhé
nhắn qua **(0225) 394399** nhé 
Để được tư vấn trực tiếp về sản phẩm, quý khách vui lòng liên hệ chuyên viên của chúng tôi qua số điện thoại **0987654321**.
Nếu có bất kỳ vấn đề gì phát sinh trong quá trình lắp đặt, anh cứ gọi cho tôi qua số **0912345678**, tôi sẽ xử lý ngay.
Để tiện trao đổi tài liệu và hình ảnh, anh cứ kết bạn nhé, Zalo của anh là **0398765432**.
Dạo này có nhiều cuộc gọi lừa đảo lắm, mẹ cứ thấy có đầu số lạ, đặc biệt là đầu số **024** không quen biết, gọi đến thì đừng nghe nhé.
Để đặt bánh pizza, chị vui lòng gọi trực tiếp đến cửa hàng qua số máy bàn **(0225) 394399** nhé, bên em không nhận đơn qua tin nhắn ạ.
"""

# ============================================= #

ID_NUMBER_0 = """
Pattern: [chuỗi_id]
Cách sinh: Tạo một chuỗi ID. Đặt nó vào bối cảnh sử dụng loại ID đó một cách tự nhiên mà không ép buộc tiền tố, có thể mô tả CCCD, mã số thuế, số tài khoản, ...

## Ví dụ:
Mã số thuế: **0123456789**
tài khoản ngân hàng của tôi là **124123441**
mã nhân viên của bạn là **F0125G** nhé
Khi điền tờ khai quyết toán thuế thu nhập cá nhân, anh chị nhớ điền chính xác mã số thuế **0123456789** của mình vào mục tương ứng.
Cậu chuyển tiền cơm trưa cho tớ vào số tài khoản **124123441**, ngân hàng Techcombank, chủ tài khoản là Nguyễn Thu Trang nhé.
Chị ơi, phòng nhân sự vừa gửi email xác nhận thông tin, chị kiểm tra lại giúp em xem số tài khoản ngân hàng của tôi là **124123441** đã đúng chưa ạ.
Để chấm công trên app, cậu phải điền mã nhân viên vào đấy. Mã nhân viên của bạn là **F0125G** phải không, tớ thấy trên danh sách này.
Để chấm công trên app, cậu phải điền mã nhân viên vào đấy. **F0125G** là mã nhân viên của bạn phải không, tớ thấy trên danh sách này.
"""

# ============================================= #

LEGAL_DOC_ID_0 = """
Pattern: [số]/[năm]/[mã]
Cách sinh: Tạo một chuỗi mã văn bản pháp luật. Đặt trong bối cảnh nói về luật pháp một cách tự nhiên mà không ép buộc tiền tố, có thể mô tả hay đề cập đến điều luật nào đó

## Ví dụ:
Nghị định **110/2013/NĐ-CP** cho phép làm như vậy
tuân theo văn bản **23/2024/AABC** được thi hành
Theo luật sư, căn cứ vào khoản 2, Điều 8 của Nghị định **110/2013/NĐ-CP** quy định về quản lý và sử dụng con dấu, công ty chúng ta hoàn toàn được phép làm như vậy.
Kể từ ngày mai, toàn thể nhân viên trong công ty chúng ta sẽ bắt đầu tuân theo quy trình làm việc mới theo văn bản **23/2024/AABC** vừa được ban giám đốc ký và thi hành.
Dựa trên Thông tư **01/2025/TT-BGDĐT** mới nhất, kỳ thi tốt nghiệp THPT năm nay sẽ có một vài thay đổi trong cấu trúc đề thi môn Lịch sử.
Theo Quyết định **999/2025/QĐ-UBND** của thành phố, khu đất công nghiệp cũ sẽ được di dời để xây dựng công viên cây xanh và hồ điều hòa.
"""

# ============================================= #

PLATE_0 = """
Pattern: [mẫu_biển_số]
Cách sinh: Tạo một chuỗi biển số xe hợp lệ (ví dụ: 29-F1 123.45, 51C-888.88). Đặt trong các ngữ cảnh về giao thông, phương tiện.

## Ví dụ:
Chiếc xe đó có biển số **29-F1 123.45**
con xe **30A4 55555** chạy ẩu thật
biển đẹp **30A 12345** hợp với xe đó 
Thưa anh công an, tôi vừa chứng kiến một vụ va chạm rồi bỏ chạy, chiếc xe đó có biển số **29-F1 123.45**, hiệu Toyota Vios màu đen ạ.
Tôi vừa xem video trên mạng, con xe **30A4 55555** phóng bạt mạng trên cao tốc, gây nguy hiểm cho bao nhiêu người khác, cần phải phạt nguội ngay.
Thằng Hưng mới bốc được biển số cho con xe mới của nó kìa. Công nhận may thật, bốc được đúng biển đẹp **30A 12345**, vừa dễ nhớ lại còn hợp với xe.
"""

# ============================================= #

ALPHANUM_ID_0 = """
Pattern: [chữ/số/dấu hỗn tạp]
Cách sinh: Tạo chuỗi ngẫu nhiên kết hợp chữ cái (hoa, thường) và số, với cấu trúc đa dạng (A1, 12B, H4A1, A1B2C34). Đặt trong ngữ cảnh mã hiệu, số serial, tên phiên bản, tên viết tắt, ... mà không ép buộc tiền tố

## Ví dụ:
mã **A1B2C34**
phiên bản **v1.2a**
mã **a22#bf**
mật khẩu mới là **44at&f#f**
Email xác nhận đã được gửi, quý khách vui lòng giữ lại mã đặt chỗ **A1B2C34** để làm thủ tục check-in tại sân bay.
Ứng dụng của bạn vừa có bản cập nhật mới. Hãy nâng cấp lên phiên bản **v1.2a** để vá lỗi bảo mật và trải nghiệm các tính năng mới nhất.
À, Wi-Fi của quán anh không có mật khẩu, em chỉ cần nhập mã truy cập **a22#bf** ở trang chào là được nhé.
Hệ thống đã đặt lại mật khẩu của bạn. Mật khẩu tạm thời của bạn là **44at&f#f**. Vui lòng đăng nhập và đổi mật khẩu ngay trong lần đầu tiên.
"""

# ============================================= #

ROMAN_NUMERAL_0 = """
Pattern: [số_la_mã]
Cách sinh: Tạo một số La Mã hợp lệ (ví dụ: I, V, IX, XXI). Đặt nó trong các ngữ cảnh tự nhiên thường đi kèm với số La Mã như tên vua, tên giáo hoàng, số thứ tự của thế kỷ, chương, phần, đại hội, quý...

## Ví dụ:
Thế kỷ **XXI**
mục **I** điều **V** trong sách
Một trong những thách thức lớn nhất của nhân loại trong thế kỷ **XXI** chính là vấn đề biến đổi khí hậu và an ninh năng lượng.
Để hiểu rõ hơn về luận điểm này, các bạn sinh viên vui lòng đọc kỹ phần giải thích ở mục **I**, chương **V** của giáo trình chính.
Vua Louis **XIV**, người được mệnh danh là "Vua Mặt Trời", có một triều đại trị vì lâu dài và để lại nhiều di sản kiến trúc huy hoàng cho nước Pháp.
Vua Louis đệ **IV**, người được mệnh danh là "Vua Mặt Trời", có một triều đại trị vì lâu dài và để lại nhiều di sản kiến trúc huy hoàng cho nước Pháp.
"""

# ============================================= #

ADDRESS_0 = """
Pattern: [cấu_trúc_địa_chỉ_linh_hoạt]
Cách sinh: Tạo một địa chỉ bằng cách kết hợp linh hoạt nhiều thành phần: [số nhà], [tên đường], [phường/xã], [quận/huyện], và [tỉnh/thành phố]. Không phải lúc nào cũng cần đầy đủ các thành phần. Hãy tạo ra các biến thể khác nhau, từ địa chỉ rất ngắn gọn đến địa chỉ đầy đủ chi tiết. Đặt trong bối cảnh tự nhiên cần đề cập đến địa chỉ

## Ví dụ:
bạn đến **số 10 Lý Thường Kiệt, Hoàn Kiếm, Hà Nội** rồi hỏi nhé
tôi có nhà ở **123/45/6 Lê Lợi, P. Bến Nghé, Q.1, TP.HCM**
**Quận Ba Đình, Hà Nội** là địa chỉ nổi tiếng đó
Anh tài xế cứ chở tôi đến đúng **số 10 Lý Thường Kiệt, Hoàn Kiếm, Hà Nội**, đó là trụ sở chính của ngân hàng, tôi sẽ xuống ở cổng bảo vệ.
Nhà tớ trong hẻm sâu lắm, địa chỉ là **123/45/6 Lê Lợi, P. Bến Nghé, Q.1, TP.HCM**, lúc nào gần đến nơi thì gọi tớ ra đầu hẻm đón nhé.
Thưa quý khách, chúng ta đang tiến vào **Quận Ba Đình, Hà Nội**, trung tâm chính trị của cả nước, nơi có Lăng Bác và nhiều cơ quan quan trọng khác.
"""

# ============================================= #

MONEY_0 = """
Pattern: [số] + [ký_hiệu_tiền]
Cách sinh: Tạo một số. Nối với ký hiệu tiền tệ (đ, vnđ, $). Đặt trong các ngữ cảnh đa dạng như giá cả, chi phí, lương bổng.

## Ví dụ:
Cái áo này giá **500.000đ**.
Lương của tôi là **20.000.000 vnđ** đó.
Nó có giá **100$**.
tôi trả **$1.5** cho cái này
Em rất thích chiếc áo này nhưng nhìn giá **500.000đ** thì lại phải đắn đo suy nghĩ lại.
Trong hợp đồng lao động có ghi rõ, mức lương khởi điểm của bạn sẽ là **20.000.000 vnđ** một tháng, chưa bao gồm các khoản phụ cấp.
ôi giày này là phiên bản giới hạn, tôi đã mua nó ở sân bay Changi trong chuyến đi Singapore, giá quy ra tiền Việt khá cao nhưng niêm yết là **100$**
Phí vận chuyển cho đơn hàng quốc tế này khá rẻ, tôi chỉ phải trả thêm **$1.5** để hàng được giao tận nhà.
"""

# ============================================= #

DIMENSION_0 = """
Pattern: [số]x[số]
Cách sinh: Tạo hai số (có thể là số thập phân). Nối chúng bằng chữ x. Đặt trong các ngữ cảnh một cách tự nhiên, có thể mô tả về diện tích, kích thước ảnh, độ phân giải, ...

## Ví dụ:
kích thước **20.5x30**
Theo bản thiết kế, chúng ta cần sử dụng loại gạch lát nền có kích thước chuẩn là **20.5x30** để đảm bảo tính thẩm mỹ cho công trình.
Để banner hiển thị sắc nét trên trang chủ, anh phải thiết kế ảnh với độ phân giải tối thiểu là **1920x1080** pixel.
Tôi đang muốn tìm mua một mảnh đất thổ cư, mặt tiền tối thiểu **5x20m**, hướng Đông Nam là đẹp nhất
"""

# ============================================= #

SCORE_0 = """
Pattern: [số]-[số]
Cách sinh: Tạo một tỷ số (số phải ở dạng số, không phải dạng chữ). Đặt trong các ngữ cảnh thể thao, thi đấu.

## Ví dụ:
Việt Nam thắng **2-1**.
Trận đấu kết thúc với tỷ số **30-12**.
Với chiến thắng quan trọng **2-1** trước đối thủ Thái Lan, đội tuyển Việt Nam đã chính thức giành vé vào vòng chung kết.
Đội bóng rổ trường tôi đã có một trận đấu áp đảo hoàn toàn và giành chiến thắng thuyết phục trước đội bạn với tỷ số cách biệt **30-12**.
Thật là một trận đấu nhàm chán, cả hai đội đều chơi quá an toàn và không tạo ra được cơ hội nào nguy hiểm, cuối cùng đành hòa nhau **0-0**.
"""

# ============================================= #

EMAIL_0 = """
Pattern: [tên]@[miền].[đuôi]
Cách sinh: Tạo một địa chỉ email bằng cách kết hợp tên người dùng, ký tự @, tên miền và đuôi miền (ví dụ: .com, .vn). Đặt trong ngữ cảnh liên lạc, đăng ký.

## Ví dụ:
Gửi CV về địa chỉ **example.user@email.com**.
Email của tôi là **nguyenvana123@gmail.com**.
Liên hệ hỗ trợ tại **support@company.vn**.
Các ứng viên quan tâm vui lòng gửi bộ hồ sơ năng lực và CV mới nhất của mình về địa chỉ email **example.user@email.com** trước chiều nay
Bài trình bày của anh rất hay, anh có thể gửi cho tôi bản slide được không? Email của tôi là **nguyenvana123@gmail.com**.
Bài trình bày của anh rất hay, anh có thể gửi cho tôi bản slide được không? Tới **nguyenvana123@gmail.com** nhé
Email của bạn đã được ghi nhận. Do quá tải, chúng tôi sẽ phản hồi trong 24 giờ tới. Trong trường hợp khẩn cấp, vui lòng liên hệ hỗ trợ kỹ thuật tại **support@company.vn**
"""

# ============================================= #

URL_0 = """
Pattern: (http[s]?://)?(www\.)?[miền].[đuôi]/[đường_dẫn]?
Cách sinh: Tạo một địa chỉ web, có thể bao gồm hoặc không bao gồm http(s)://, www., và đường dẫn phía sau. Đặt trong ngữ cảnh trang web, tài liệu tham khảo.

## Ví dụ:
Xem thêm tại **https://www.example.com/products**.
Truy cập **google.com.vn** để biết thêm chi tiết.
Hãy vào **www.youtube.com** để xem video.
Bộ sưu tập Xuân Hè đã chính thức lên kệ, mời bạn xem thêm toàn bộ sản phẩm tại **https://www.example.com/products** và chọn cho mình một bộ cánh thật ưng ý.
Về các chính sách bảo hành chung của công ty, anh có thể truy cập **google.com.vn** và tìm kiếm tên công ty chúng tôi để biết thêm chi tiết ạ.
File video này không nằm trên server chính thức, anh phải truy cập vào đường link tài nguyên tạm tại **www.youtube.com** mới có thể tải về được.
"""

# ============================================= #

DATE__FRACTION_0 = """
Pattern: [số]/[số]
Cách sinh: Tạo ra một câu phức, trong đó pattern d/m xuất hiện ít nhất hai lần (Chú ý d và m phải ở dạng số, không phải dạng chữ). Một lần trong ngữ cảnh ngày tháng (ví dụ: đi kèm các từ "ngày", "hôm", "sự kiện") để được gán nhãn "DATE". Lần còn lại trong ngữ cảnh tỉ số, điểm số, hoặc tỉ lệ (ví dụ: đi kèm các từ "điểm", "tỉ lệ", "đạt") để được gán nhãn "FRACTION". Nhớ phải gán nhãn [DATE] hoặc [FRACTION] thật chuẩn xác như trong ví dụ.

## Ví dụ:
Vào ngày **1/6**[DATE] tôi cho bé đi chơi và bé đã ăn được **1/2**[FRACTION] cái bánh.
Hiện tại mới chỉ có **3/4**[FRACTION] số sinh viên đã nộp trong khi hạn chót nộp bài tập là **20/11**[DATE], 
Trong chương trình "Vui Tết thiếu nhi" **1/6**[DATE], nhóm chúng tôi đã quyên góp được **1/2**[FRACTION] số quà tặng dự kiến cho các em nhỏ có hoàn cảnh khó khăn.
Báo cáo tiến độ cho thấy mới có **3/4**[FRACTION] các hạng mục được hoàn thành, trong khi deadline tổng của dự án là **20/11**[DATE], chúng ta đang bị chậm tiến độ.
Trong trận chung kết bóng rổ diễn ra vào sáng **10/10**[DATE], đội tuyển của chúng ta đã không thể làm nên bất ngờ khi chỉ ghi được **1/3**[FRACTION] tổng số điểm của đối phương.
"""

# ============================================= #

PHONE__INTEGER_0 = """
Pattern: [dãy số]
Cách sinh: Tạo câu chứa một chuỗi số là số điện thoại (ví dụ, số tổng đài 1800 1008, số điện thoại cá nhân, đầu số điện thoại) để gán nhãn PHONE, và một chuỗi số khác trong ngữ cảnh số đếm, số lượng để gán nhãn INTEGER, chú ý đây là số nguyên, không phải số thập phân. Lưu ý chuỗi số phải ở dạng số, không phải dạng chữ. Nhớ phải gán nhãn [PHONE] hoặc [INTEGER] thật chuẩn xác như trong ví dụ.

## Ví dụ:
Gọi tổng đài **1800 1008**[PHONE] để có cơ hội nhận giải thưởng **500 000**[INTEGER] đồng.
Hơn **10 000**[INTEGER] cuộc gọi trong ngày hôm qua được gọi đến **1900 1234**[PHONE] đấy
cái số **024**[PHONE] cứ gọi đến tôi một ngày **24**[PHONE] cuộc
Hãy gọi **113**[PHONE] nếu bạn thấy có **113**[INTEGER] người đang tụ tập gây rối.
người chơi số **112**[INTEGER] dùng điện thoại đầu số **031**[PHONE] đấy
Hãy nhanh tay gọi về tổng đài **1800 1008**[PHONE] của chúng tôi, chỉ cần trả lời đúng một câu hỏi, bạn sẽ có cơ hội nhận ngay giải thưởng tiền mặt trị giá **500 000**[INTEGER] đồng
Báo cáo từ trung tâm chăm sóc khách hàng cho thấy, sau khi sản phẩm mới ra mắt, đã có hơn **10 000**[INTEGER] cuộc gọi được ghi nhận chỉ trong một ngày qua tổng đài **1900 1234**[PHONE].
Dạo này tôi khổ sở vì một số điện thoại có đầu số **024**[PHONE], nó cứ nhá máy cho tôi liên tục, tính ra một ngày phải đến **24**[INTEGER] lần.
Trong buổi diễn tập phòng chống bạo loạn, tình huống giả định là có một nhóm **113**[INTEGER] phần tử quá khích đang đập phá, và đội cơ động phải có mặt sau khi nhận được cuộc gọi đến số **113**[PHONE].
À, Hưng 'voi' số báo danh **112**[INTEGER] đây mà, nhớ chứ, nhà nó hồi xưa ở Hải Phòng, vẫn dùng số điện thoại đầu **031**[PHONE] ấy
"""

# ============================================= #

INTEGER__FLOAT_0 = """
Pattern: [số][dấu phân cách][số]
Cách sinh: Tạo câu mà trong đó một chuỗi số dùng dấu "." làm dấu ngăn cách hàng nghìn (đọc là số nguyên) và được gán nhãn INTEGER. Một chuỗi số khác (dùng dấu ".") trong ngữ cảnh cần đến số thập phân, ví dụ như kỹ thuật, tài chính (hệ số, tỉ giá), ..., được đọc là số thập phân và gán nhãn FLOAT. Lưu ý chuỗi số phải ở dạng số, không phải dạng chữ. Nhớ phải gán nhãn [INTEGER] hoặc [FLOAT] thật chuẩn xác như trong ví dụ.

## Ví dụ:
Lô hàng trị giá **25.000.000**[INTEGER] đồng có khối lượng riêng là **0,8**[FLOAT].
anh ấy đã đạt được hệ số lợi nhuận là **2.5**[FLOAT] sau một năm với **1.000**[INTEGER] đô la tiền vốn
có **1.000**[INTEGER] người ra kết quả **1.25**[FLOAT] cho bài toán này
Biên bản kiểm định ghi rõ: lô hàng dầu ăn này trị giá **25.000.000**[INTEGER] đồng và có tỷ trọng là **0.8**[FLOAT] so với nước.
Báo cáo tài chính cho thấy quỹ đầu tư đã đạt được hệ số sinh lời ấn tượng là **2.5**[FLOAT], biến mỗi **1.000**[INTEGER] đô la vốn ban đầu thành **2.500**[INTEGER] đô la.
Thật đáng ngạc nhiên, trong số **1.000**[INTEGER] học sinh được kiểm tra, không em nào tính ra được hằng số chính xác là **1.25**[FLOAT] cho thí nghiệm vật lý này.
"""

# ============================================= #

NUMBER_RANGE__SCORE_0 = """
Pattern: X-Y
Cách sinh: Tạo câu phức chứa pattern X-Y với các ngữ cảnh khác nhau (Lưu ý X và Y phải ở dạng số, không phải dạng chữ). Trong câu, một lần để chỉ tỉ số thể thao như bóng đá, cầu lông, bóng bàn, ... (gán nhãn SCORE), một lần để chỉ một khoảng giá trị (gán nhãn NUMBER_RANGE). Hãy cố gắng sáng tạo để có thể kết hợp ngữ cảnh phù hợp cho hai tag này. Nhớ phải gán nhãn [NUMBER_RANGE] hoặc [SCORE] thật chuẩn xác như trong ví dụ.

## Ví dụ:
Trận đấu diễn ra lúc **7-9**[NUMBER_RANGE] giờ tối đã kết thúc với tỉ số **2-1**[SCORE].
Nếu Việt Nam thắng với tỉ số **3-0**[SCORE], Cửa hàng sẽ giảm giá cho **3-5**[NUMBER_RANGE] sản phẩm
Trận cầu tâm điểm diễn ra trong khoảng **7-9**[NUMBER_RANGE] giờ tối qua đã có một kết quả kịch tính, đội chủ nhà đã lội ngược dòng thành công với tỉ số **2-1**[SCORE].
Nghe nói cửa hàng kia có khuyến mãi hay lắm, chỉ cần đội nhà thắng **3-0**[SCORE] là sẽ có cơ hội mua được **3-5**[NUMBER_RANGE] món đồ giảm giá đó.
Theo nhận định của tôi, đây là một trận đấu cân tài cân sức, tôi dự đoán tỉ số sẽ là **2-2**[SCORE] và với lối chơi quyết liệt của hai đội, sẽ có khoảng **4-5**[NUMBER_RANGE] thẻ vàng được rút ra.
"""

# ============================================= #

ROMAN_NUMERAL__ALPHANUM_ID_0 = """
Pattern: [Số La Mã] (ví dụ: II, V, X)
Cách sinh: Tạo câu phức trong đó cùng một chuỗi số La Mã được dùng với hai vai trò. Một là số thứ tự (chỉ vua, thế kỷ, chương mục, cấp bậc...) và được gán nhãn ROMAN_NUMERAL. Hai là một chuỗi các chữ cái Latin đứng riêng lẻ (thường trong mã hiệu, ký hiệu...) và được gán nhãn ALPHANUM_ID. Nhớ phải gán nhãn [ROMAN_NUMERAL] hoặc [ALPHANUM_ID] thật chuẩn xác như trong ví dụ.

## Ví dụ:
Vua Louis **XIV**[ROMAN_NUMERAL] trong lịch sử Pháp thường ký các mật thư bằng chuỗi ký tự **XIV**[ALPHANUM_ID].
Các nhà sử học đã giải mã được rằng Vua Louis **XIV**[ROMAN_NUMERAL] của Pháp đã sử dụng một hệ thống mật mã phức tạp, trong đó chuỗi ký tự **XIV**[ALPHANUM_ID] được dùng để chỉ các điệp viên hoạt động tại Anh.
Bế mạc Đại hội Đảng lần thứ **IX**[ROMAN_NUMERAL], một trong những văn kiện quan trọng nhất được thông qua là Nghị quyết Trung ương với mã hiệu **IX**[ALPHANUM_ID], đặt nền móng cho chiến lược công nghiệp hóa.
Mô hình này tái hiện tàu con thoi trong nhiệm vụ Apollo **XI**[ROMAN_NUMERAL] lịch sử; quý khách có thể thấy module dịch vụ với mã hiệu sản xuất là **XI**[ALPHANUM_ID] được khắc trên thân.
"""

# ============================================= #

