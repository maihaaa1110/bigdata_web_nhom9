import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px

st.markdown(
    """
    <style>
        /* 🌌 NỀN CHÍNH */
        .stApp {
            background-color: #0b132b;
            background-image: linear-gradient(180deg, #0b132b 0%, #1b263b 100%);
            color: #e0e6ed;
        }

        /* 🧭 THANH SIDEBAR (tương phản hơn nền chính) */
        [data-testid="stSidebar"] {
            background-color: #1c2541 !important;
            border-right: 1px solid #3a506b;
        }

        /* HEADER (giữ đồng bộ với sidebar) */
        [data-testid="stHeader"] {
            background-color: #1c2541 !important;
            border-bottom: 1px solid #3a506b;
        }

        /* 🧾 CÁC KHỐI VĂN BẢN & CARD */
        .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox, .stDataFrame {
            background-color: transparent;
            color: #e0e6ed;
        }

        /* 🔘 NÚT BẤM */
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

        /* 📊 BẢNG DỮ LIỆU */
        div[data-testid="stDataFrame"] table {
            background-color: #1b263b;
            color: #e0e6ed;
            border: 1px solid #3a506b;
        }

        /* 🎛️ INPUT FIELD */
        input, select, textarea {
            background-color: #1b263b !important;
            color: #e0e6ed !important;
            border-radius: 5px !important;
            border: 1px solid #3a506b !important;
        }

        /* 🎨 TITLES */
        h1, h2, h3, h4, h5, h6 {
            color: #5bc0be !important;
        }

        /* 🧭 SIDEBAR TITLE */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #5bc0be !important;
        }

        /* 🧱 CARD / CONTAINER */
        div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
            color: #5bc0be !important;
        }

        /* 🔻 SCROLLBAR */
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

        /* 💬 ĐƯỜNG LINK */
        a {
            color: #5bc0be;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Hàm lấy data clean
def load_daily_parquet():
    file_path = os.path.join("data", "daily_merged.parquet")

    # ✅ Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        st.error(f"❌ Không tìm thấy file: {file_path}")
        return None

    try:
        df = pd.read_parquet(file_path)

        # ✅ Đổi tên cột sang tiếng Việt dễ hiểu
        rename_columns = {
            "date": "Ngày",
            "sp500_open": "SP500 Mở cửa",
            "sp500_high": "SP500 Cao nhất",
            "sp500_low": "SP500 Thấp nhất",
            "sp500_close": "SP500 Đóng cửa",
            "sp500_volume": "SP500 Khối lượng",

            "spy_open": "SPY Mở cửa",
            "spy_high": "SPY Cao nhất",
            "spy_low": "SPY Thấp nhất",
            "spy_close": "SPY Đóng cửa",
            "spy_volume": "SPY Khối lượng",

            "vix_open": "VIX Mở cửa",
            "vix_high": "VIX Cao nhất",
            "vix_low": "VIX Thấp nhất",
            "vix_close": "VIX Đóng cửa",
            "vix_volume": "VIX Khối lượng",

            "gold_open": "Vàng Mở cửa",
            "gold_high": "Vàng Cao nhất",
            "gold_low": "Vàng Thấp nhất",
            "gold_close": "Vàng Đóng cửa",
            "gold_volume": "Vàng Khối lượng",

            "oil_open": "Dầu Mở cửa",
            "oil_high": "Dầu Cao nhất",
            "oil_low": "Dầu Thấp nhất",
            "oil_close": "Dầu Đóng cửa",
            "oil_volume": "Dầu Khối lượng",

            "usd_index_open": "USD Index Mở cửa",
            "usd_index_high": "USD Index Cao nhất",
            "usd_index_low": "USD Index Thấp nhất",
            "usd_index_close": "USD Index Đóng cửa",
            "usd_index_volume": "USD Index Khối lượng",

            "uup_open": "UUP Mở cửa",
            "uup_high": "UUP Cao nhất",
            "uup_low": "UUP Thấp nhất",
            "uup_close": "UUP Đóng cửa",
            "uup_volume": "UUP Khối lượng",

            "sp500_return": "SP500 % Thay đổi",
            "sp500_range": "SP500 Biên độ",
            "gold_return": "Vàng % Thay đổi",
            "gold_range": "Vàng Biên độ",
            "oil_return": "Dầu % Thay đổi",
            "oil_range": "Dầu Biên độ",

            "sp500_return_lag1": "SP500 % hôm trước",
            "vix_close_lag1": "VIX hôm trước",
            "gold_return_lag1": "Vàng % hôm trước",

            "market_direction": "Xu hướng thị trường",
                
            # Các cột thêm từ xử lý tin tức / sentiment:
            "n_articles": "Số lượng bài báo",
            "n_positive": "Số bài tích cực",
            "n_neutral": "Số bài trung lập",
            "n_negative": "Số bài tiêu cực",

            "prop_positive": "Tỷ lệ tích cực",
            "prop_neutral": "Tỷ lệ trung lập",
            "prop_negative": "Tỷ lệ tiêu cực",

            "mean_sentiment_score": "Điểm sentiment trung bình",
            "mean_sentiment_prob": "Xác suất sentiment trung bình",
            "median_sentiment_prob": "Xác suất sentiment trung vị",
            "std_sentiment_score": "Độ lệch chuẩn sentiment",
            "weighted_sentiment_score": "Điểm sentiment trọng số",

            "avg_text_len": "Độ dài văn bản trung bình",
            "median_text_len": "Độ dài văn bản trung vị"
        }
        
        df1=df.copy()

        df1.rename(columns=rename_columns, inplace=True)

        # đưa cột "Xu hướng thị trường" xuống cuối
        col = "Xu hướng thị trường"

        # nếu cột tồn tại
        if col in df1.columns:
            cols = [c for c in df1.columns if c != col] + [col]
            df1 = df1[cols]

        return df,df1


    except Exception as e:
        st.error(f"🔥 Lỗi khi đọc parquet: {e}")
        return None

# Hàm lấy data tin tức
def load_data_news(date_str):
    """
    Tải dữ liệu từ file Excel dựa trên chuỗi ngày đã nhập (YYYYMMDD).
    File được đọc từ dòng 8 đến dòng 27 (bỏ qua 7 dòng đầu, chỉ lấy 20 dòng).
    Dòng đầu (dòng 8) làm header, sau đó loại bỏ đuôi "L2" ở cột A nếu có.
    """
    file_path = os.path.join("data", "news", f"sentiment_{date_str}.parquet")

    # Kiểm tra sự tồn tại của file
    if not os.path.exists(file_path):
        st.error(f"❌ File không tồn tại: {file_path}")
        return None
    try:
        df_news = pd.read_parquet(file_path)
        df = df_news[1:].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# Hàm vẽ biểu đồ tổng hợp đường
def a():
    df_merged, df_full = load_daily_parquet()
    print(df_merged.columns.tolist())

    col_map = {}
    for col in df_merged.columns:
        c = col.lower()
        if 'date' in c:
            col_map['date'] = col
        elif 'price' in c or 'close' in c:
            col_map['price'] = col
        elif 'sentiment' in c or 'compound' in c or 'score' in c:
            col_map['sentiment_score'] = col

    print("\nĐã dò được mapping cột:", col_map)

    required = {'date', 'price', 'sentiment_score'}
    if not required.issubset(col_map.keys()):
        raise KeyError(f"Thiếu các cột cần thiết! Các cột hiện có: {df_merged.columns.tolist()}")

    df = df_merged.rename(columns={
        col_map['date']: 'date',
        col_map['price']: 'price',
        col_map['sentiment_score']: 'sentiment_score'
    })[['date', 'price', 'sentiment_score']]

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    return df

# ======================================================
# ĐỊNH NGHĨA HÀM TRỰC QUAN HÓA
# ======================================================


def plot_single_timeseries_plotly(df, date_col, value_col):
    """
    Vẽ biểu đồ 1 đường theo thời gian
    df: DataFrame
    date_col: tên cột ngày ('date')
    value_col: tên cột giá trị cần vẽ (đã được lọc từ hàm trên)
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode='lines',
            name=value_col,
            line=dict(width=2)
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{value_col} theo thời gian</b>", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Date", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title=value_col, showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        width=900,
        height=450
    )

    return fig


import streamlit as st

def filter_columns_by_selection(df1):
    """
    df1: DataFrame khi chưa đổi tên cột

    Hàm tạo 2 selectbox:
    - Selectbox 1: Chọn nhóm (SP500, SPY, VIX...)
    - Selectbox 2: Chọn loại dữ liệu (Mở cửa, Cao nhất, ...)
    """

    # Tất cả keywords VIẾT HOA như bạn yêu cầu
    keywords_display = ["SP500", "SPY", "VIX", "GOLD", "OIL", "USD_INDEX", "UUP"]

    # Chọn keyword (hiển thị đẹp)
    st.sidebar.markdown("### Chọn nhóm dữ liệu")
    keyword_display = st.sidebar.selectbox("Chọn nhóm (keyword):", keywords_display)

    # Convert sang lowercase để tìm cột trong df1
    keyword = keyword_display.lower()

    # Các loại dữ liệu (mapping)
    feature_options = {
        "Mở cửa": "open",
        "Cao nhất": "high",
        "Thấp nhất": "low",
        "Đóng cửa": "close",
        "Khối lượng": "volume"
    }

    feature_choice = st.sidebar.selectbox("Chọn loại dữ liệu:", list(feature_options.keys()))
    feature_suffix = feature_options[feature_choice]

    # Tìm cột thuộc keyword + loại dữ liệu
    filtered_cols = [col for col in df1.columns if keyword in col and feature_suffix in col]

    if len(filtered_cols) == 0:
        st.warning("⚠️ Không tìm thấy cột phù hợp trong dataset.")
        return None

    return filtered_cols[0]

def get_default_corr_columns(df1):
    default_cols = [col for col in df1.columns if "close" in col.lower()]

    # Thêm cột cảm xúc nếu có
    extra_sentiment_cols = [
        "mean_sentiment_score", 
        "weighted_sentiment_score",
        "mean_sentiment_prob"
    ]

    for col in extra_sentiment_cols:
        if col in df1.columns:
            default_cols.append(col)

    return default_cols

import streamlit as st

def select_corr_variables(df1):

    default_cols = get_default_corr_columns(df1)

    selected_cols = st.sidebar.multiselect(
        "Chọn biến để tính tương quan",
        options=list(df1.columns),
        default=default_cols
    )

    return selected_cols

def plot_corr_heatmap(df, columns):
    corr_matrix = df[columns].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            reversescale=True
        )
    )

    fig.update_layout(
        title=dict(text="<b>Ma trận tương quan</b>", x=0.5, font=dict(size=16)),
        height=700,
        template="plotly_white",
        xaxis=dict(side="top")
    )

    return fig

def split_finance_vs_sentiment(df):
    """
    Trả về:
    - df_fin: nhóm dữ liệu tài chính (SP500, SPY, VIX, GOLD, OIL, USD Index, UUP)
    - df_sent: nhóm dữ liệu còn lại (cảm xúc + đặc trưng NLP + metadata)
    """

    keywords = ["SP500", "SPY", "VIX", "GOLD", "OIL", "USD_INDEX", "UUP"]
    date_col = "date"

    # Nhóm tài chính
    matched_fin_cols = [col for col in df.columns if any(k in col.upper() for k in keywords)]
    if date_col in df.columns:
        matched_fin_cols = [date_col] + matched_fin_cols

    df_fin = df[matched_fin_cols]

    # Nhóm cảm xúc & tin tức
    df_sent_cols = [col for col in df.columns 
                    if col not in matched_fin_cols and col.lower() != "market_direction"]

    if date_col in df.columns:
        df_sent_cols = [date_col] + df_sent_cols

    df_sent = df[df_sent_cols]

    return df_sent


def select_sentiment_column(df_sent):
    st.sidebar.markdown("### Chọn biến cảm xúc để xem phân phối")

    sentiment_cols = [
        col for col in df_sent.columns
        if col not in ["date"]  # bỏ cột ngày
    ]

    # chọn mặc định: mean_sentiment_score nếu có
    default = "mean_sentiment_score" if "mean_sentiment_score" in sentiment_cols else sentiment_cols[0]

    selected = st.sidebar.selectbox(
        "Chọn chỉ số cảm xúc:",
        sentiment_cols,
        index=sentiment_cols.index(default)
    )
    return selected

import plotly.graph_objects as go

def plot_sentiment_distribution_plotly(df, column):
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df[column],
            nbinsx=30,
            opacity=0.85
        )
    )

    fig.update_layout(
        title=f"<b>Phân phối của {column}</b>",
        xaxis_title=column,
        yaxis_title="Tần suất",
        template="plotly_white",
        height=450
    )

    return fig



# Trang web
def tab1():
    dashboard_option = st.sidebar.selectbox(
        "Chọn dữ liệu bạn muốn xem:", 
        (
            "Dữ liệu tổng hợp đã qua xử lý",
            "Dữ liệu tài chính", 
            "Dữ liệu cảm xúc tin tức",
        )
    )
    st.title(f"{dashboard_option}")

    df1, df_full = load_daily_parquet()

    # Copy toàn bộ DataFrame
    df_fin = df_full.copy()

    # Danh sách keywords
    keywords = ["SP500", "SPY", "VIX", "Vàng", "Dầu", "USD Index", "UUP"]
    
    # Cột ngày
    date_col = "Ngày"

    # Lọc cột chứa keyword
    matched_columns = [col for col in df_fin.columns if any(k in col for k in keywords)]

    # Luôn giữ cột Ngày
    matched_columns = [date_col] + matched_columns if date_col in df_fin.columns else matched_columns

    # Chỉ giữ các cột matched trong df_fin
    df_fin = df_fin[matched_columns]

    # df_fin đã là các cột liên quan keyword
    df_fin_columns = df_fin.columns.tolist()

    # Lấy tất cả các cột còn lại trong df_full
    df_news_columns = [col for col in df_full.columns 
                    if col not in df_fin_columns and col != "Xu hướng thị trường"]

    # Luôn giữ cột Ngày trong df_news
    df_news_columns = [date_col] + df_news_columns if date_col in df_full.columns else df_news_columns

    # Tạo df_news chỉ với các cột còn lại
    df_news = df_full[df_news_columns]

    table_placeholder = st.empty()
    
    # WEB Clean data
    if dashboard_option == "Dữ liệu tổng hợp đã qua xử lý":
        if df_full is not None:
            table_placeholder.dataframe(df_full)

            # --- VẼ MA TRẬN TƯƠNG QUAN ---
            corr_cols = select_corr_variables(df1)

            if len(corr_cols) >= 2:
                fig_corr = plot_corr_heatmap(df1, corr_cols)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("⚠️ Cần chọn ít nhất 2 biến để tính tương quan.")

    
    if dashboard_option == "Dữ liệu tài chính":
        table_placeholder.dataframe(df_fin)

        filter_fi = st.sidebar.multiselect(

            "Lọc theo Dữ liệu tài chính:",
            options=keywords
        )

        selected_col = filter_columns_by_selection(df1)

        if selected_col:
            fig = plot_single_timeseries_plotly(df1, "date", selected_col)
            st.plotly_chart(fig, use_container_width=True)


        filtered_df = df_fin.copy()

        # ✅ Step 2 + 3: khi chọn keyword → tìm các cột chứa keyword → hiển thị dataframe
        if filter_fi:
            matcol = ["Ngày"] + [
            col for col in filtered_df.columns 
            if any(k in col for k in filter_fi)
            ]
            filtered_df = df_fin[matcol]
            table_placeholder.dataframe(filtered_df)




    # Tin tức
    if dashboard_option == "Dữ liệu cảm xúc tin tức":
        st.sidebar.header("Bộ lọc dữ liệu")
        selected_category = st.sidebar.selectbox(
            "Chọn loại dữ liệu", ["Điểm cảm xúc", "Tin tức gốc"]
        )
        if selected_category == "Điểm cảm xúc":
            table_placeholder.dataframe(df_news)

            df_sent = split_finance_vs_sentiment(df1)

            # Sidebar chọn biến cảm xúc
            selected_sent_col = select_sentiment_column(df_sent)

            # Vẽ biểu đồ phân phối
            fig_dist = plot_sentiment_distribution_plotly(df_sent, selected_sent_col)
            st.plotly_chart(fig_dist, use_container_width=True)


        if selected_category == "Tin tức gốc":

            st.markdown("### Hiển thị tin tức gốc")
 
            date_str = st.sidebar.text_input("Nhập ngày (ví dụ: 2023-01-30):", value="2023-01-01", key="txn_date")

            df_newss = load_data_news(date_str)
            if df_newss is not None:
                st.dataframe(df_newss)

# Sidebar


  #  st.sidebar.slider("Chọn mức độ", min_value=0, max_value=100, value=50)

    # Nội dung chính
 #   st.write(f"Bạn đã chọn: {selected_category}")
#    st.write("Đây là nội dung trang Home.")

if __name__ == '__main__':
    tab1()