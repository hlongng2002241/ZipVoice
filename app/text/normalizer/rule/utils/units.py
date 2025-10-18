# fmt: off
UNITS_DICT = {}

HUMAN_BODY_UNITS = {
	"bpm" 		: " nhịp trên phút ",
}
UNITS_DICT.update(HUMAN_BODY_UNITS)

LENGTH_UNITS = {
	# Độ dài
	"nm" 		: " na nô mét ",
	"µm" 		: " mi cờ rô mét ",
	"mm" 		: " mi li mét ",
	"cm" 		: " xen ti mét ",
	"dm" 		: " đề xi mét ",
	"m" 		: " mét ",
	"dam" 		: " đề ca mét ",
	"hm" 		: " héc tô mét ",
	"km" 		: " ki lô mét ",
}
UNITS_DICT.update(LENGTH_UNITS)
    
_squared_units = ["2", "^2", "²"]
AREA_UNITS = {
	# Diện tích
	**{
        L.strip() + S : " " + L_spk.strip() + " vuông " 
        for L, L_spk in LENGTH_UNITS.items() 
        for S in _squared_units
    },
	"ha" 		: " héc ta ",
	"hecta" 	: " héc ta ",
}
UNITS_DICT.update(AREA_UNITS)

_cubed_units = ["3", "^3", "³"]
BASIC_VOLUME_UNITS = {
	# Thể tích
	L.strip() + S : " " + L_spk.strip() + " khối " 
	for L, L_spk in LENGTH_UNITS.items() 
	for S in _cubed_units
    
}
LIQUID_VOLUME_UNITS = {
    "cc" 		: " xê xê ",
    "ml" 		: " mi li lít ",
    "l "		: " lít ",
    "L "		: " lít ",
    "dL"		: " đề xi lít ",
    "dl"		: " đề xi lít ",
}
VOLUME_UNITS = {
    **BASIC_VOLUME_UNITS,
    **LIQUID_VOLUME_UNITS,
}
UNITS_DICT.update(VOLUME_UNITS)
    
WEIGHT_UNITS = {
	# Khối lượng
	"kg" 		: " ki lô gam ",
	"Kg" 		: " ki lô gam ",
	"mg" 		: " mi li gam ",
	"grams" 	: " gờ ram ",
    "gram" 		: " gờ ram ",
    "gr" 		: " gờ ram ",
    "g " 		: " gam ",
    "ounce" 	: " ao ",
    "oz"        : " ao ",
}
UNITS_DICT.update(WEIGHT_UNITS)
    
FREQUENCY_UNITS = {
	# Tần số
	"Hz"  		: " héc ",
	"kHz" 		: " ki lô héc ",
	"MHz" 		: " mê ga héc ",
	"GHz" 		: " ghi ga héc ",
    "dB"		: " đề xi ben ",
}
UNITS_DICT.update(FREQUENCY_UNITS)

FORCE_UNITS = {
	# Lực
    "N "		: " Niu tơn ",
}
UNITS_DICT.update(FORCE_UNITS)
    
PRESSURE_UNITS = {
	# Áp lực
    **{
        "N/" + A : " Niu tơn trên " + A_spk.strip() + " " 
        for A, A_spk in AREA_UNITS.items()
	},
    "Nm" 		: " Niu tơn mét ",
    "Pa"		: " Pát can ",
    "atm"		: " át mốt phe ", 
    "mmHg"		: " mi li mét thủy ngân ", 
}
UNITS_DICT.update(PRESSURE_UNITS)

MOLE_UNITS = {
    "mol"		: " mon ",
    "mmol"		: " mi li mon ",
}

CONCENTRATION_UNITS  = {
	# Nồng độ (weight / liquid volume)
    W.strip() + "/" + V.strip() : f" {W_spk.strip()} trên {V_spk.strip()} "
    for W, W_spk in list(WEIGHT_UNITS.items()) + list(MOLE_UNITS.items())
    for V, V_spk in list(LIQUID_VOLUME_UNITS.items()) + [("100ml", "một trăm mi li lít")]
}
UNITS_DICT.update(CONCENTRATION_UNITS)

# This is different from _time_units
TIME_UNITS = {
	"h "			: " giờ ",
	"/h"			: " trên giờ ",
	"/p"			: " trên phút ",
	"/s"			: " trên giây ",
	"h/"			: " giờ trên ",
	"p/"			: " phút trên ",
	"s/"			: " giây trên ",
}
UNITS_DICT.update(TIME_UNITS)

_time_units = {
    "h"			: " giờ ",
    "p"			: " phút ",
    "s"			: " giây ",
}

VELOCITY_UNITS = {
    # Tốc độ (độ dài/thời gian)
	L.strip() + "/" + T: f" {L_spk.strip()} trên {T_spk.strip()} "
	for L, L_spk in LENGTH_UNITS.items()
	for T, T_spk in list(_time_units.items()) + [(T_spk.strip(), T_spk.strip()) for T_spk in _time_units.values()]
}
UNITS_DICT.update(VELOCITY_UNITS)

ACCELERATION_UNITS = {
	# Gia tốc
    **{
        L.strip() + "/" + T + S : f" {L_spk.strip()} trên {T_spk.strip()} bình "
        for L, L_spk in LENGTH_UNITS.items()
        for T, T_spk in _time_units.items()
        for S in _squared_units
	}
}
UNITS_DICT.update(ACCELERATION_UNITS)
	
FLOW_RATE_UNITS = {
	# Lưu lượng (volume/time)
	V.strip() + "/" + T : f" {V_spk.strip()} trên {T_spk.strip()} "
	for V, V_spk in VOLUME_UNITS.items()
	for T, T_spk in list(_time_units.items()) + [(T_spk.strip(), T_spk.strip()) for T_spk in _time_units.values()]
	
}
UNITS_DICT.update(FLOW_RATE_UNITS)

ENERGY_UNITS = {
	# Năng lượng
	"kW" 		: " ki lô goát ",
    "kWh"       : " ki lô oát giờ ",
    "Wh"        : " oát giờ ",
    "kcal"		: " ki lô ca lo ",
    "kCal"		: " ki lô ca lo ",
    "cal"		: " ca lo ",
    "Cal"		: " ca lo ",
}
UNITS_DICT.update(ENERGY_UNITS)

LIGHT_UNITS = {
    "nits"		: " nít ",
    "lux"		: " lắc ",
}
UNITS_DICT.update(LIGHT_UNITS)

PREFIX_CURRENCY_UNITS = {
	"€"			: " ơ rô ",
	"£"			: " bảng anh ",
	"¥"			: " yên ",
	"₩"			: " uôn ",
	"₹"			: " ru pi ",
	"฿"			: " bạt ",
	"៛"			: " ri en ",
	"₭"			: " kíp lào ",
	"S$"		: " đô la xinh ga po ",
	"元"		: " nhân dân tệ ",
	"Fr"		: " phơ răng ",
	"₺"			: " lia ra thổ nhĩ kỳ ",
	"₦"			: " nai ra ni giê ri a ",
	"د.إ"		: " đia ram các tiểu vương quốc ả rập thống nhất ",
	"R$"		: " rê an bờ ra xin ",
	"$"			: " đô la ",
}
 
SUFFIX_CURRENCY_UNITS = {
	"₽"			: " rúp ",
	"Kč"		: " co ru na séc ",
	"zł"		: " dờ lo ti ba lan ",
	"₴"			: " hờ ríp nia u cờ rai na ",
    "dolar"		: " đô la ",
	"USD"		: " u ét đê ",
	"usd"		: " u ét đê ",
	"Euro"		: " ơ rô ",
	"euro"		: " ơ rô ",
	"Eur"		: " ơ rô ",
	"eur"		: " ơ rô ",
	"đ "		: " đồng ",
	"₫"			: " đồng ",
	"VNĐ"		: " việt nam đồng ",
	"VND"		: " việt nam đồng ",
	"vnđ"		: " việt nam đồng ",
	"vnd"		: " việt nam đồng ",
    "K " 		:  " nghìn ",
    "k " 		:  " nghìn ",
    "tr" 		:  " triệu ",
}

CURRENCY_UNITS = {**PREFIX_CURRENCY_UNITS, **SUFFIX_CURRENCY_UNITS}
UNITS_DICT.update(CURRENCY_UNITS)

BASIC_ANY_OVER_UNITS = {
	"/kg" 		: " trên một ki lô gam ",
    
	"đồng/" 	: " đồng trên ",
	"USD/" 		: " u ét đê trên ",
    
	"người/" 	: " người trên ",
	"giờ/" 		: " giờ trên ",
	"lít/" 		: " lít trên ",
	"mmol/l" 	: " mi li mon trên lít ",
	"mg/" 		: " mi li gam trên ",
	"kg/" 		: " ki lô gam trên ",
	"cái/" 		: " cái trên ",
	"triệu/" 	: " triệu trên ",
}
ANY_OVER_UNITS = {
	**BASIC_ANY_OVER_UNITS
}
UNITS_DICT.update(ANY_OVER_UNITS)

OVER_TIME_UNITS = {
	"/giây"		: " trên giây ",
    "/phút"		: " trên phút ",
    "/giờ"		: " trên giờ ",
    "/ngày"		: " trên ngày ",
    "/tháng"	: " trên tháng ",
    "/năm"		: " trên năm ",
}
OVER_QUANTITY_UNITS = {
	"/CP" 		: " trên một cổ phiếu ",
    "/cổ phiếu"	: " trên một cổ phiếu ",
    "/sản phẩm"	: " trên một sản phẩm ",
	"/lượt" 	: " trên một lượt ",
    "/thùng" 	: " trên một thùng ",
	"/căn" 		: " trên một căn ",
	"/bàn" 		: " trên một bàn ",
	"/ghế" 		: " trên một ghế ",
	"/bát" 		: " trên một bát ",
	"/cái" 		: " trên một cái ",
	"/con" 		: " trên một con ",
	"/lần" 		: " trên một lần ",
	"/người"	: " trên một người ",
}
OVER_ANY_UNITS = {
    "/lít" 		: " trên lít ",
	"/tấn" 		: " trên tấn ",
 
    **OVER_QUANTITY_UNITS,
    **OVER_TIME_UNITS,
     
    **{"/" + k : " trên " + v.strip() + " " for k, v in LENGTH_UNITS.items()},
    **{"/" + k : " trên " + v.strip() + " " for k, v in AREA_UNITS.items()},
    **{"/" + k : " trên " + v.strip() + " " for k, v in VOLUME_UNITS.items()},
    **{"/" + k : " trên " + v.strip() + " " for k, v in WEIGHT_UNITS.items()},
    **{"/" + k : " trên " + v.strip() + " " for k, v in FREQUENCY_UNITS.items()},
}
UNITS_DICT.update(OVER_ANY_UNITS)

CURRENCY_OVER_ANY_UNITS = {
	# Tiền tệ trên đơn vị
    **{
        C.strip() + O.strip() : f" {C_spk.strip()} {O_spk.strip()} "
		for C, C_spk in list(CURRENCY_UNITS.items()) + [("triệu", "triệu"), ("tỷ", "tỷ"), ("tỉ", "tỉ")]
        for O, O_spk in OVER_ANY_UNITS.items()
	},
    
	# Tỷ giá
	"USD/VND"   : " u ét đê trên việt nam đồng ",
    "VND/USD"   : " việt nam đồng trên u ét đê ",
}
UNITS_DICT.update(CURRENCY_OVER_ANY_UNITS)
	
PERCENTAGE_OVER_ANY_UNITS = {
    "%" + O.strip() : f" phần trăm {O_spk.strip()}"
	for O, O_spk in OVER_ANY_UNITS.items()
}
UNITS_DICT.update(PERCENTAGE_OVER_ANY_UNITS)
    
DEGREE_UNITS = ["º", "°"]
COMMON_UNITS = {
	# Đơn vị chung
    **{D : " độ " for D in DEGREE_UNITS},
	"%" 		: " phần trăm ",
	"mAh" 		: " mi li am pe giờ ",
	"Ah"        : " am pe giờ ",
    "MP" 		: " mê ga píc seo ",
    "Mpx"     	: " mê ga píc seo ",
    "pixel"     : " píc seo ",
    "px"     	: " píc seo ",
    "dpi"       : " đi pi ai ",
    "DPI"       : " đi pi ai ",
	"g/km" 		: " gam trên ki lô mét ",
}
UNITS_DICT.update(COMMON_UNITS)
    
DATA_USAGE_UNITS = {
	# Đơn vị lưu trữ dữ liệu
	"byte" 		: " bai ",
	"B "        : " bai ",
	"KB" 		: " ki lô bai ",
	"Mb" 		: " mê ga bai ",
	"MB" 		: " mê ga bai ",
	"mb" 		: " mê ga bai ",
	"Gb" 		: " ghi ga bai ",
	"GB" 		: " ghi ga bai ",
	"gb" 		: " ghi ga bai ",
	"TB" 		: " tê ra bai ",
    "Tb" 		: " tê ra bai ",
}
UNITS_DICT.update(DATA_USAGE_UNITS)
	
DATA_THROUGHPUT_UNITS = {
	# Đơn vị tốc độ dữ liệu (phân biệt bit và byte)
    "bps"       : " bít trên giây ",
    "kbps"      : " ki lô bít trên giây ",
    "Kbps"      : " ki lô bít trên giây ",
    "Mbps"      : " mê ga bít trên giây ",
    "Gbps"      : " ghi ga bít trên giây ",
    "B/s"       : " bai trên giây ",
    "KB/s"      : " ki lô bai trên giây ",
    "MB/s"      : " mê ga bai trên giây ",
    "GB/s"      : " ghi ga bai trên giây ",
    # Các biến thể thường gặp
    "Mb/s"      : " mê ga bít trên giây ",
    "Gb/s"      : " ghi ga bít trên giây ",
}
UNITS_DICT.update(DATA_THROUGHPUT_UNITS)
	
	#"2G":"hai gờ",
	#"3G":"ba gờ",
	#"4G":"bốn gờ",
	#"5G":"năm gờ",
 	
TEMPERATURE_UNITS = {
	# Units of temperature 
    D + T : f" độ {T_spk.strip()} "
    for T, T_spk in [("C", "xê"), ("F", "ép")]
    for D in DEGREE_UNITS + ["o"]
}
UNITS_DICT.update(TEMPERATURE_UNITS)

COORDINATE_UNITS = {
    D + T : f" độ {T_spk.strip()} "
    for T, T_spk in [("N", "vĩ bắc"), ("E", "kinh đông")]
    for D in DEGREE_UNITS
}
UNITS_DICT.update(COORDINATE_UNITS)