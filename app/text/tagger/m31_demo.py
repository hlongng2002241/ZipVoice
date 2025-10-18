import sys; sys.path.append(".") # fmt: skip
import streamlit as st
from src.utils.text.normalizer.models.inference_v2 import TaggerInference


CSS = """
    <style>
        html, body, [class*="css"] {
            font-family: 'Verdana', sans-serif;  /* Change your font family */
            font-size: 18px;  /* Change your font size */
        }
        
        .st-emotion-cache-1c7y2kd {
            background-color: #C0E5FF;  /* Change the background color for the user's messages */
            border-radius: 10px;  /* Optional: Add border-radius for rounded corners */
            padding: 10px;  /* Optional: Add padding for spacing */
            margin: 5px 0;  /* Optional: Add margin for spacing */
            position: relative;
        }
            
        [data-testid=stSidebar] {
            background: linear-gradient(155deg, #1E3A8A 0%, #3B82F6 50%, #1D4ED8 100%);
            color: white;
        }
                
        /* Add these lines to target the text color */
        [data-testid=stSidebar] label {
            color: white !important;
        }
        [data-testid=stSidebar] title {
            color: white !important;
        }
        [data-testid=stSidebar] .stMarkdown {
            color: white !important;
        }
        [data-testid=stSidebar] .stMarkdown p {
            color: white !important;
        }
        [data-testid=stSidebar] .stMarkdown h1,
        [data-testid=stSidebar] .stMarkdown h2,
        [data-testid=stSidebar] .stMarkdown h3 {
            color: white !important;
        }
        [data-testid=stAlert] {
            background-color: #71B2F0;
        }

        div[data-testid=stSelectbox]{
            color: white;
        }

        .stButton > button {
            display: block;
            margin: 0 auto;
        }     
    </style>
""".strip()
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the NER model with caching for better performance"""
    return TaggerInference(
        "logs/tagger_09_mMiniLM/checkpoint-6010",
        tokenizer_name="microsoft/Multilingual-MiniLM-L12-H384",
        label_matching="match_goal_tag",
    )


def highlight_text_with_tags(content: str, patterns: list):
    """Create highlighted HTML with tags displayed next to patterns"""
    if not patterns:
        return content

    # Sort patterns by start position to process them in order
    sorted_patterns = sorted(patterns, key=lambda p: p.start)

    highlighted_html = ""
    last_end = 0

    # Color mapping for different tags - darker colors with white text
    tag_colors = {
        "ADDRESS": "#8B4513",
        "ALPHANUM_ID": "#4B0082",
        "DATE": "#1E3A8A",
        "DATE_RANGE": "#0F172A",
        "DATE_RANGE_y_y": "#334155",
        "DIMENSION": "#1E40AF",
        "EMAIL": "#0369A1",
        "FLOAT_big": "#1E293B",
        "FLOAT_n": "#0F172A",
        "FRACTION": "#475569",
        "INTEGER_big": "#374151",
        "INTEGER_n": "#1F2937",
        "LEGAL_DOC_ID": "#6B21A8",
        "MATH_EXPR": "#059669",
        "MEASUREMENT": "#0D9488",
        "MONEY": "#047857",
        "FOREIGN_WORD": "#7C2D12",
        "NUMBER_RANGE": "#4338CA",
        "PHONE": "#7C3AED",
        "PLATE": "#DC2626",
        "ROMAN_NUMERAL": "#BE123C",
        "SPORT_SCORE": "#C2410C",
        "TIME": "#1D4ED8",
        "TIME_RANGE": "#2563EB",
        "URL": "#1565C0",
    }

    for pattern in sorted_patterns:
        # Add text before the pattern
        highlighted_html += content[last_end : pattern.start]

        # Get color for this tag
        color = tag_colors.get(pattern.tag, "#eee")

        # Add highlighted pattern with tag (following m06_demo.py style)
        highlighted_html += f'<span style="background-color: {color}; color: white; padding: 0.2em 0.4em; margin: 0 0.2em; line-height: 2; border-radius: 0.3em;">{pattern.content} <strong>{pattern.tag}</strong></span>'

        last_end = pattern.start + len(pattern.content)

    # Add remaining text
    highlighted_html += content[last_end:]

    return highlighted_html


def main():
    st.set_page_config(page_title="NER Model Demo", page_icon="🏷️", layout="wide")

    st.title("🏷️ Named Entity Recognition Demo")
    st.markdown("Demo for Vietnamese text NER model that identifies and tags various entities.")

    # Load model
    with st.spinner("Loading NER model..."):
        try:
            tagger = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return

    # Input section
    st.header("Input Text")

    # Provide some example texts
    examples = [
        "hôm qua 15/8/2023 lúc 14h30 tao đi mua RTX 4090 giá 35.000.000đ ở 123A/456 Trần Hưng Đạo",
        "Acc @user_2k3 (SN 2k3) ở 102/3A Ngô Tất Tố, P.19, Q.Bình Thạnh đã post link http://localhost:8501/",
        "Liên hệ qua email support@nvidia.com hoặc hotline 1900-123-456 để được hỗ trợ",
        "hôm qua 15/8/2023 lúc 14h30 tao đi mua RTX 4090 giá 35.000.000đ ở 123A/456 Trần Hưng Đạo, thấy biển số 51F-12345 đậu trước cửa. Chủ shop nói máy chạy được 144fps ở độ phân giải 3840x2160px, bảo hành 36 tháng từ 1/9-31/12/2026. Tao test benchmark 3DMark được điểm 15.678, nhưng nhiệt độ lên tới 82°C sau 2h15p chạy. Email liên hệ là support@nvidia.com , hotline 1900-123-456.",
    ]

    selected_example = st.selectbox("Choose an example:", ["Custom input"] + examples)
    height = 300
    if selected_example == "Custom input":
        text_input = st.text_area(
            "Enter Vietnamese text to analyze:", height=height, placeholder="Type your Vietnamese text here..."
        )
    else:
        text_input = st.text_area("Enter Vietnamese text to analyze:", value=selected_example, height=height)

    # Analysis section
    if st.button("Analyze Text", type="primary") and text_input.strip():
        with st.spinner("Analyzing text..."):
            # Run inference
            output = tagger.inference(text_input.strip())[0]

            # Display results
            st.header("Results")

            # Show highlighted text
            st.subheader("Highlighted Text with Tags")
            highlighted_html = highlight_text_with_tags(output.content, output.patterns)
            st.markdown(highlighted_html, unsafe_allow_html=True)

            if not output.patterns:
                st.info("No entities detected in the text.")

            # Show errors if any
            # if output.errors:
            #     st.subheader("Warnings/Errors")
            #     for error in output.errors:
            #         st.warning(error)

            # Show token-level outputs in expander
            if output.token_level_outputs:
                with st.expander("Token-level Analysis"):
                    token_data = []
                    for token, tag in output.token_level_outputs:
                        if token not in tagger.tokenizer.all_special_tokens:
                            token_data.append({"Token": token, "Predicted Tag": tag})

                    st.dataframe(token_data, use_container_width=True)

    # Information section
    with st.sidebar:
        st.header("Model Information")
        st.markdown(
            """
        **Model**: Vietnamese NER Tagger

        **Supported Entity Types**:
        - 📍 ADDRESS: Addresses
        - 🔤 ALPHANUM_ID: Alphanumeric IDs
        - 📅 DATE: Dates
        - 📅 DATE_RANGE: Date ranges
        - 📅 DATE_RANGE_y_y: Year-to-year ranges
        - 📐 DIMENSION: Dimensions
        - 📧 EMAIL: Email addresses
        - 🔢 FLOAT_big: Large float numbers
        - 🔢 FLOAT_n: Float numbers
        - ➗ FRACTION: Fractions
        - 🔢 INTEGER_big: Large integers
        - 🔢 INTEGER_n: Integer numbers
        - 📋 LEGAL_DOC_ID: Legal document IDs
        - ➕ MATH_EXPR: Math expressions
        - 📏 MEASUREMENT: Measurements
        - 💰 MONEY: Monetary amounts
        - 🌐 FOREIGN_WORD: Non-Vietnamese text
        - 🔢 NUMBER_RANGE: Number ranges
        - 📞 PHONE: Phone numbers
        - 🚗 PLATE: License plates
        - 🏛️ ROMAN_NUMERAL: Roman numerals
        - ⚽ SPORT_SCORE: Sport scores
        - ⏰ TIME: Time expressions
        - ⏰ TIME_RANGE: Time ranges
        - 🌐 URL: URLs
        """
        )


if __name__ == "__main__":
    main()
