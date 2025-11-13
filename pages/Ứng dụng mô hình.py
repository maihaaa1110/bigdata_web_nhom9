import streamlit as st
import joblib
import pandas as pd

# ============== 1. XỬ LÝ BAN ĐẦU ==============
# --- Load model ---
model_data = joblib.load("best_model.pkl")
model = model_data["model"]
numeric_cols = model_data["numeric_cols"]
label_mapping = model_data["label_mapping"]
best_model_name = model_data["best_model_name"]

# --- Page setup ---
st.markdown("<h1 style='text-align:center;text-transform: uppercase; color:#4CAF50;'> DỰ ĐOÁN XU HƯỚNG THỊ TRƯỜNG</h1>", unsafe_allow_html=True)
st.write(f"<p style='text-align:center; color:gray;'>Model: <b>{best_model_name}</b></p>", unsafe_allow_html=True)
st.markdown("---")

# --- CSS nền dark blue ---
st.markdown(
    """
    <style>
        /* Toàn bộ nền ứng dụng */
        .stApp {
            background-color: #0b132b; /* xanh đậm */
            background-image: linear-gradient(160deg, #0b132b 0%, #1c2541 50%, #3a506b 100%);
            color: #e0e0e0;
        }

        /* Header và sidebar */
        [data-testid="stHeader"], [data-testid="stSidebar"] {
            background-color: #1c2541 !important;
        }

        /* Màu chữ mặc định */
        * {
            color: #f5f6fa;
        }

        /* Tiêu đề */
        h1, h2, h3, h4, h5, h6 {
            color: #5bc0be !important;
        }

        /* Nút bấm */
        .stButton button {
            background: linear-gradient(90deg, #5bc0be, #3a506b);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #3a506b, #5bc0be);
            transform: scale(1.05);
        }

        /* Input box & selectbox */
        .stTextInput, .stNumberInput, .stSelectbox {
            background-color: #1c2541;
            color: white;
        }

        /* Kẻ bảng */
        div[data-testid="stDataFrame"] table {
            background-color: #1c2541;
            color: #f5f6fa;
        }

        /* Thanh cuộn */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #5bc0be;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #0b132b;
        }
    </style>
    """,
    unsafe_allow_html=True
)


    # Tiêu đề và menu chọn dữ liệu

st.sidebar.markdown("### Phục vụ dự đoán:")

dashboard_option = st.sidebar.selectbox(
    "Chọn chế độ:",
    (
        "Mô hình dự đoán",
        "Xử lý chấm điểm cảm xúc",
        "Tải dữ liệu thô ban đầu",
    )
)

# Tiêu đề chính theo lựa chọn
st.markdown(
f"<h2 style='text-align: center; text-transform: uppercase;'>{dashboard_option}</h2>",
unsafe_allow_html=True
)

# Đường ngăn cách (divider) bên dưới menu
st.sidebar.markdown("---")

# ============== 2. TAB1 ==============
if dashboard_option == "Mô hình dự đoán":
    # --- Chọn chế độ nhập liệu ---
    mode1, mode2 = st.tabs(["🔹 Nhập thủ công", "📁 Upload file dữ liệu"])
    st.markdown("---")

    # ============================================================
    # 1. Nhập thủ công
    # ============================================================
    with mode1:
        with st.form("manual_form"):
            st.markdown("### Financial Indicators")
            col1, col2, col3 = st.columns(3)

            with col1:
                sp500_open = st.number_input("S&P500 Open", value=3810.0)
                sp500_high = st.number_input("S&P500 High", value=3850.0)
                sp500_low = st.number_input("S&P500 Low", value=3800.0)
                sp500_close = st.number_input("S&P500 Close", value=3840.0)
                sp500_volume = st.number_input("S&P500 Volume", value=3900000000.0)
                sp500_return = st.number_input("SP500 % Thay đổi", value= 0)
                sp500_range = st.number_input("SP500 Biên độ", value=37.0)
                sp500_return_lag1 = st.number_input("SP500 % hôm trước", value=0)
                spy_open = st.number_input("SPY Open", value=370.0)
                spy_high = st.number_input("SPY High", value=375.0)
                spy_low = st.number_input("SPY Low", value=365.0)
                spy_close = st.number_input("SPY Close", value=372.0)
                spy_volume = st.number_input("SPY Volume", value=85900000.0)
                oil_return = st.number_input("Dầu % Thay đổi",value=0)
                oil_range = st.number_input("Dầu Biên độ", value=4.6)

            with col2:
                vix_open = st.number_input("VIX Open", value=22.0)
                vix_high = st.number_input("VIX High", value=23.0)
                vix_low = st.number_input("VIX Low", value=21.9)
                vix_close = st.number_input("VIX Close", value=22.5)
                vix_volume = st.number_input("VIX Volume", value=0)
                vix_close_lag1 = st.number_input("VIX hôm trước", value=22.5)
                gold_open = st.number_input("Gold Open", value=1850.0)
                gold_high = st.number_input("Gold High", value=1880.0)
                gold_low = st.number_input("Gold Low", value=1849.0)
                gold_close = st.number_input("Gold Close", value=1872.0)
                gold_volume = st.number_input("Gold Volume", value=62.0)
                gold_return = st.number_input("Vàng % Thay đổi", value=0)
                gold_range = st.number_input("Vàng Biên độ", value=13)
                gold_return_lag1 = st.number_input("Vàng % hôm trước", value=0)


            with col3:
                oil_open = st.number_input("Oil Open", value=77.2)
                oil_high = st.number_input("Oil High", value=77.4)
                oil_low = st.number_input("Oil Low", value=72.8)
                oil_close = st.number_input("Oil Close", value=73.5)
                oil_volume = st.number_input("Oil Volume", value=350000.0)

                usd_index_open = st.number_input("USD Index Open", value=103.0)
                usd_index_high = st.number_input("USD Index High", value=104.0)
                usd_index_low = st.number_input("USD Index Low", value=102.5)
                usd_index_close = st.number_input("USD Index Close", value=103.5)
                usd_index_volume = st.number_input("USD Index Volume", value=0)
                uup_open = st.number_input("UUP Open", value=25.2)
                uup_high = st.number_input("UUP High", value=25.3)
                uup_low = st.number_input("UUP Low", value=25.1)
                uup_close = st.number_input("UUP Close", value=25.28)
                uup_volume = st.number_input("UUP Volume", value=4400000)  

            st.markdown("---")
            st.markdown("### Sentiment & Text Features")

            col4, col5, col6 = st.columns(3)
            with col4:
                n_articles = st.number_input("Number of Articles", value=50.0)
                n_positive = st.number_input("Positive Articles", value=36.0)
                n_neutral = st.number_input("Neutral Articles", value=8.0)
                n_negative = st.number_input("Negative Articles", value=6.0)
                prop_positive = st.number_input("Prop. Positive", value=0.72)
            with col5:
                prop_neutral = st.number_input("Prop. Neutral", value=0.16)
                prop_negative = st.number_input("Prop. Negative", value=0.12)
                mean_sentiment_score = st.number_input("Mean Sentiment Score", value=0.56)
                weighted_sentiment_score = st.number_input("Weighted Sentiment Score", value=0.9657)
                mean_sentiment_prob = st.number_input("Mean Sentiment Prob", value=0.9)
            with col6:
                median_sentiment_prob = st.number_input("Median Sentiment Prob", value=0.75)
                std_sentiment_score = st.number_input("Std Sentiment Score", value=0.57)
                avg_text_len = st.number_input("Avg Text Length", value=225.0)
                median_text_len = st.number_input("Median Text Length", value=186.0)

            submit_manual = st.form_submit_button(" Predict Manually")

        if submit_manual:
            # gom dữ liệu thành DataFrame
            input_data = pd.DataFrame([[locals()[col] for col in numeric_cols]], columns=numeric_cols)
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0]
            inv_map = {v: k for k, v in label_mapping.items()}
            pred_label = inv_map[pred]

            color = "#2E8B57" if pred_label == "up" else "#C0392B"
            st.markdown(f"<h3 style='text-align:center; color:{color};'> Prediction: {pred_label.upper()}</h3>", unsafe_allow_html=True)
            st.progress(prob[pred])
            st.write(f"**Confidence:** {prob[pred]*100:.2f}%")

    # ============================================================
    # 2. Upload file dữ liệu
    # ============================================================
    with mode2:
        st.markdown("###  Upload File Dữ Liệu")
        uploaded_file = st.file_uploader("Tải file dữ liệu", type=["parquet"])

        if uploaded_file is not None:
            df1 = pd.read_parquet(uploaded_file)
            st.dataframe(df1)
            st.write(" File đã đọc thành công!")
            df_new = df1.dropna()

            st.dataframe(df_new)

            # Kiểm tra cột hợp lệ
            missing_cols = [c for c in numeric_cols if c not in df_new.columns]
            if missing_cols:
                st.error(f"⚠️ File thiếu các cột cần thiết: {missing_cols}")
            else:
                st.markdown("---")
                st.markdown("<h4 style='color:#4CAF50;'> Bấm nút dưới đây để thực hiện dự đoán:</h4>", unsafe_allow_html=True)
                
                # Nút kích hoạt dự đoán
                predict_button = st.button(" Dự đoán từ File")

                if predict_button:
                    preds = model.predict(df_new[numeric_cols])
                    probs = model.predict_proba(df_new[numeric_cols])
                    inv_map = {v: k for k, v in label_mapping.items()}

                    # Sau khi dự đoán
                    df_new["Prediction"] = [inv_map[p] for p in preds]
                    df_new["Confidence (%)"] = (probs.max(axis=1) * 100).round(2)

                    st.success(" Dự đoán hoàn tất!")
                    st.markdown("###  Kết quả dự đoán:")

                    # Hiển thị kết quả an toàn
                
                    cols_to_add = df_new.columns.difference(["Prediction", "Confidence (%)"])
                    df_to_show = df_new[["Prediction", "Confidence (%)"]].join(df_new[cols_to_add])
                #   df_to_show["Prediction"] = df_to_show["Prediction"].shift(-1)  # shift lên 1 ô
                
                #   df_to_show["Prediction"].iloc[-1] = None 
                
                #   df_to_show=df_to_show.dropna()
                    # Hàm highlight cột
                    def highlight_col(col):
                        return ['background-color: yellow; font-weight: bold' if col.name == "sp500_close" else '' for _ in col]

                    # Hiển thị DataFrame trên Streamlit với highlight
                    st.dataframe(df_to_show.style.apply(highlight_col))

                    # Nút tải về
                    csv = df_new.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=" Tải kết quả (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

# if dashboard_option == "Xử lý chấm điểm cảm xúc":
    