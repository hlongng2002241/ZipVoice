CONTEXTS = [
    dict(
        context="Cuộc trò chuyện giữa nhân viên bán hàng và khách hàng về Ngân hàng / Tín dụng - Quy trình cho vay",
        situations="nhân viên tư vấn các gói vay, khách hàng cung cấp hồ sơ, thẩm định tài sản đảm bảo, thẩm định phương án kinh doanh/nguồn thu nhập, thông báo kết quả thẩm định, ký hợp đồng tín dụng và giải ngân",
        roles="Nhân viên tín dụng, Khách hàng vay",
        emotions="Hy vọng, Lo lắng, Căng thẳng, Hồi hộp, Vui mừng, Nhẹ nhõm, Thất vọng, Cẩn trọng, Trách nhiệm, Áp lực"
    ),
    dict(
        context="Cuộc trò chuyện giữa nhân viên bán hàng và khách hàng về Thẩm định khoản vay tại nhà/cơ sở kinh doanh",
        situations="chuyên viên thẩm định giới thiệu, hỏi về lịch sử hình thành tài sản, xác minh tình trạng tài sản đảm bảo, phỏng vấn người có liên quan (hàng xóm, tổ trưởng), chụp ảnh hiện trạng, lập biên bản thẩm định",
        roles="Chuyên viên thẩm định, Khách hàng vay, Người có liên quan",
        emotions="Thận trọng, Lo lắng, Căng thẳng, Tò mò, Hoài nghi, Bất an, Tự tin, Hợp tác"
    ),
    dict(
        context="Cuộc trò chuyện giữa nhân viên bán hàng và khách hàng về Tư vấn các sản phẩm cho vay",
        situations="tìm hiểu nhu cầu khách hàng (vay mua nhà, vay kinh doanh, vay tiêu dùng), giới thiệu các sản phẩm phù hợp, so sánh lãi suất và phí, giải thích điều kiện và quy trình vay, tính toán số tiền trả góp hàng tháng",
        roles="Nhân viên tư vấn, Khách hàng",
        emotions="Tò mò, Phân vân, Hy vọng, Lo lắng, Bối rối, Tin tưởng, Nhiệt tình, Kiên nhẫn, Đồng cảm"
    ),
    dict(
        context="Cuộc trò chuyện giữa nhân viên bán hàng và khách hàng",
        situations="tư vấn thông số kỹ thuật, báo giá sản phẩm, hẹn lịch giao hàng, giải thích chính sách bảo hành, thông báo chương trình khuyến mãi, xử lý khiếu nại",
        roles="Nhân viên bán hàng, Khách hàng",
        emotions="Tự tin, Nhiệt tình, Hào hứng, Phân vân, Hài lòng, Bực bội, Thất vọng, Kiên nhẫn, Tò mò"
    ),
    dict(
        context="Thuyết trình kêu gọi đầu tư",
        situations="giới thiệu đội ngũ sáng lập, trình bày vấn đề thị trường, giới thiệu sản phẩm và giải pháp, phân tích đối thủ cạnh tranh, trình bày mô hình kinh doanh, dự phóng tài chính, kêu gọi số vốn và kế hoạch sử dụng, phiên hỏi đáp với nhà đầu tư",
        roles="Người thuyết trình, Nhà đầu tư",
        emotions="Đam mê, Nhiệt huyết, Tự tin, Căng thẳng, Hồi hộp, Hy vọng, Tò mò, Hoài nghi, Hứng thú, Thận trọng, Ấn tượng"
    ),
    dict(
        context="Nhà đầu tư chất vấn người sáng lập",
        situations="yêu cầu cung cấp số liệu chi tiết, hỏi về các giả định trong kế hoạch kinh doanh, kiểm tra pháp lý doanh nghiệp, phỏng vấn đội ngũ chủ chốt, đánh giá công nghệ và tài sản trí tuệ, đàm phán các điều khoản đầu tư",
        roles="Nhà đầu tư, Nhà sáng lập",
        emotions="Căng thẳng, Áp lực, Hoài nghi, Thận trọng, Tập trung, Tự tin, Lo lắng, Bị chất vấn, Minh bạch"
    ),
    dict(
        context="Hỏi và trả lời về luật giao thông",
        situations="Hỏi về mức phạt cho các lỗi cụ thể, Xử lý khi xảy ra va chạm/tai nạn, Thắc mắc về quy trình xử phạt, Hỏi về \"phạt nguội\", Quy định về giấy tờ và phương tiện, Khiếu nại và tranh chấp",
        roles="Người dân, người điều khiển phương tiện, cảnh sát giao thông, luật sư",
        emotions="Căng thẳng, Áp lực, Hoài nghi, Thận trọng, Tập trung, Tự tin, Lo lắng, Minh bạch, Khẩn trương, Gấp gáp"
    ),
    dict(
        context="Tự do",
        situations="trò chuyện, tán gẫu, bất kì tình huống nào",
        roles="Tự chọn",
        emotions="Tự chọn"
    ),

    # dict(
    #     context="Cuộc trò chuyện về Nhà hàng / Quán ăn",
    #     situations="gọi món, yêu cầu hóa đơn thanh toán, phàn nàn về chất lượng món ăn, đặt bàn trước, nhờ nhân viên giới thiệu món đặc sản, yêu cầu thêm gia vị"
    # ),
    # dict(
    #     context="Cuộc trò chuyện về Trường học",
    #     situations="thảo luận nhóm làm bài tập, xin phép nghỉ học, hỏi giảng viên về bài giảng, đăng ký môn học, mượn sách ở thư viện, họp phụ huynh"
    # ),
    # dict(
    #     context="Cuộc trò chuyện về Bệnh viện / Phòng khám",
    #     situations="đặt lịch hẹn khám bệnh, mô tả triệu chứng cho bác sĩ, nhận tư vấn về cách dùng thuốc, làm thủ tục nhập viện, thanh toán viện phí, thăm người bệnh"
    # ),
    # dict(
    #     context="Cuộc trò chuyện về Công sở / Văn phòng",
    #     situations="họp giao ban đầu tuần, báo cáo tiến độ dự án, trình bày ý tưởng mới, xin ý kiến chỉ đạo của cấp trên, hướng dẫn nhân viên mới, giải quyết mâu thuẫn với đồng nghiệp"
    # ),
    # dict(
    #     context="Cuộc trò chuyện về Sân bay",
    #     situations="làm thủ tục check-in, ký gửi hành lý, hỏi đường đến cổng ra máy bay, qua cửa kiểm tra an ninh, mua hàng miễn thuế, thông báo thất lạc hành lý"
    # ),
    # dict(
    #     context="Cuộc trò chuyện về Ngân hàng",
    #     situations="mở tài khoản thanh toán, thực hiện giao dịch gửi/rút tiền, yêu cầu sao kê tài khoản, tư vấn về các gói vay vốn, khóa thẻ khẩn cấp, khiếu nại về giao dịch lỗi"
    # ),
    # dict(
    #     context="Cuộc trò chuyện về Gia đình",
    #     situations="trò chuyện trong bữa cơm tối, bàn bạc kế hoạch đi du lịch cuối tuần, cha mẹ dạy con học bài, phân công việc nhà, hỏi thăm sức khỏe ông bà"
    # ),
]