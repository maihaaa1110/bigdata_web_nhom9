import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


st.markdown(
    """
    <style>
        /* NỀN CHÍNH */
        .stApp {
            background-color: #0b132b;
            background-image: linear-gradient(180deg, #0b132b 0%, #1b263b 100%);
            color: #e0e6ed;
        }

        /* THANH SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: #1c2541 !important;
            border-right: 1px solid #3a506b;
        }

        /* HEADER */
        [data-testid="stHeader"] {
            background-color: #1c2541 !important;
            border-bottom: 1px solid #3a506b;
        }

        /* TEXT BLOCKS */
        .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox, .stDataFrame {
            background-color: transparent;
            color: #e0e6ed;
        }

        /* BUTTON */
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

        /* TABLE */
        div[data-testid="stDataFrame"] table {
            background-color: #1b263b;
            color: #e0e6ed;
            border: 1px solid #3a506b;
        }

        /* INPUT */
        input, select, textarea {
            background-color: #1b263b !important;
            color: #e0e6ed !important;
            border-radius: 5px !important;
            border: 1px solid #3a506b !important;
        }

        /* TITLES */
        h1, h2, h3, h4, h5, h6 {
            color: #5bc0be !important;
        }

        /* SIDEBAR TITLES */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #5bc0be !important;
        }

        /* METRICS */
        div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
            color: #5bc0be !important;
        }

        /* SCROLLBAR */
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

        /* LINK */
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

# ============== 1. HÀM BỔ TRỢ XỬ LÝ DỮ LIỆU ==============

# Hàm đổi tên cột để hiển thị
def rename_columns_if_exist_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đổi tên các cột trong DataFrame nếu cột đó có tồn tại.
    Danh sách cột đổi tên được định nghĩa sẵn trong hàm.
    
    Args:
        df (pd.DataFrame): DataFrame cần đổi tên.
    
    Returns:
        pd.DataFrame: DataFrame sau khi đổi tên cột.
    """
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
            
        # Các cột sentiment / tin tức
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

    # Lọc ra những cột thật sự có trong df
    valid_map = {old: new for old, new in rename_columns.items() if old in df.columns}
    
    # Đổi tên
    return df.rename(columns=valid_map)

# Hàm lấy data clean
def load_daily_parquet():
    file_path = os.path.join("data", "daily_merged.parquet")

    # Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        st.error(f"❌ Không tìm thấy file: {file_path}")
        return None

    try:
        df = pd.read_parquet(file_path)

        # đưa cột "Xu hướng thị trường" xuống cuối
        col = "market_direction"

        # nếu cột tồn tại
        if col in df.columns:
            cols = [c for c in df.columns if c != col] + [col]
            df = df[cols]

        return df


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

# Danh sách từ khóa tài chính
keywords = ["SP500", "SPY", "VIX", "Vàng", "Dầu", "USD Index", "UUP"]

# Hàm tách dữ liệu tài chính và cảm xúc
def split_finance_and_news(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tách DataFrame đã đổi tên cột thành 2 phần:
      1.Dữ liệu tài chính (df_fin)
      2.Dữ liệu tin tức / sentiment (df_news)

    - Giữ cột 'Ngày' nếu có.
    - Lọc df_fin theo danh sách keyword.
    - df_news gồm các cột còn lại (trừ 'Xu hướng thị trường').
    """

    date_col = "Ngày"

    # --- Tạo df_fin ---
    matched_columns = [
        col for col in df.columns if any(k in col for k in keywords)
    ]
    if date_col in df.columns:
        matched_columns = [date_col] + matched_columns

    df_fin = df[matched_columns].copy()

    # --- Tạo df_news ---
    df_fin_columns = df_fin.columns.tolist()

    df_news_columns = [
        col for col in df.columns if col not in df_fin_columns and col != "Xu hướng thị trường"
    ]
    if date_col in df.columns:
        df_news_columns = [date_col] + df_news_columns

    df_news = df[df_news_columns].copy()

    return df_fin, df_news

# Hàm đổi tên cột tin tức gốc
def rename_columns_if_exist_news(df):
    # Từ điển ánh xạ tên cột tiếng Anh sang tiếng Việt
    rename_col = {
        "id": "Mã bài viết",
        "title": "Tiêu đề",
        "author": "Tác giả",
        "published_utc": "Thời gian xuất bản (UTC)",
        "article_url": "Đường dẫn bài viết",
        "tickers": "Mã chứng khoán liên quan",
        "image_url": "Hình ảnh minh họa",
        "description": "Mô tả ngắn",
        "keywords": "Từ khóa",
        "source": "Nguồn tin",
        "published_date": "Ngày xuất bản",
        "amp_url": "Liên kết AMP",
        "publisher.name": "Tên nhà xuất bản",
        "publisher.homepage_url": "Trang chủ nhà xuất bản",
        "publisher.logo_url": "Logo nhà xuất bản",
        "publisher.favicon_url": "Biểu tượng (favicon)",
        "text_for_sentiment": "Nội dung phân tích cảm xúc",
        "sentiment_label": "Nhãn cảm xúc (tích cực/trung lập/tiêu cực)",
        "sentiment_score_prob": "Xác suất cảm xúc",
        "sentiment_score": "Điểm cảm xúc"
    }
    existing_cols = {col: rename_col[col] for col in df.columns if col in rename_col}
    return df.rename(columns=existing_cols)
    
# ============== 2. ĐỊNH NGHĨA HÀM TRỰC QUAN HÓA ==============

# ------------------- TAB 1 ---------------------
# Hàm lấy danh sách các cột tạo ma trận ban đầu
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

# Hàm tạo select
def select_corr_variables(df1):

    default_cols = get_default_corr_columns(df1)

    selected_cols = st.sidebar.multiselect(
        "Chọn biến để tính tương quan",
        options=list(df1.columns),
        default=default_cols
    )

    return selected_cols

# Vẽ ma trận tương quan
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

# ------------------- TAB 2 ---------------------
# Hàm 

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
    keyword_display = st.sidebar.selectbox("Chọn nhóm tài sản:", keywords_display)

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

    feature_choice = st.sidebar.selectbox("Chọn loại thuộc tính:", list(feature_options.keys()))
    feature_suffix = feature_options[feature_choice]

    # Tìm cột thuộc keyword + loại dữ liệu
    filtered_cols = [col for col in df1.columns if keyword in col and feature_suffix in col]

    if len(filtered_cols) == 0:
        st.warning("⚠️ Không tìm thấy cột phù hợp trong dataset.")
        return None

    return filtered_cols[0]







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
    st.sidebar.markdown("---")

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



# ============== 3. HIỂN THỊ WEB ==============

def tabdata():

    # Tiêu đề và menu chọn dữ liệu
    st.sidebar.markdown("### Chọn dữ liệu bạn muốn xem:")
    dashboard_option = st.sidebar.selectbox(
        "Danh sách dữ liệu:",
        (
            "Dữ liệu tổng hợp đã qua xử lý",
            "Dữ liệu tài chính",
            "Dữ liệu cảm xúc tin tức",
        )
    )

    # Tiêu đề chính theo lựa chọn
    st.markdown(
    f"<h1 style='text-align: center; text-transform: uppercase;'>{dashboard_option}</h1>",
    unsafe_allow_html=True
)

    # Đường ngăn cách (divider) bên dưới menu
    st.sidebar.markdown("---")

    # Chạy data đầu tiên
    df1 = load_daily_parquet()
    df1_full = rename_columns_if_exist_clean(df1)

    # Gọi hàm tách data 2 phần
    df_fin, df_news = split_finance_and_news(df1_full)

    # TAB1
    if dashboard_option == "Dữ liệu tổng hợp đã qua xử lý":

        tab = st.radio(
            "Chọn chế độ hiển thị:",   # label hiển thị trên giao diện
            ["📋 Dữ liệu chi tiết", "📈 Phân tích trực quan"],  # danh sách lựa chọn
            horizontal=True  # (tuỳ chọn, nếu bạn dùng Streamlit >=1.31)
        )



        if tab == "📋 Dữ liệu chi tiết":
            st.markdown("""
        Sau khi xử lý và ghép nối **dữ liệu tin tức** với **dữ liệu tài chính**, tập dữ liệu được lưu trữ dưới dạng `DataFrame`, 
        bao gồm **740 quan sát (tương ứng với 740 ngày giao dịch)** và **60 biến (thuộc tính)**.  
        Mỗi dòng dữ liệu biểu diễn **một ngày giao dịch** của thị trường Mỹ, kết hợp giữa:
        - Thông tin tài chính từ các chỉ số chính như **S&P 500**, **Vàng**, **Dầu**, **USD Index**, **VIX**, v.v.  
        - Các **chỉ số cảm xúc tổng hợp** (sentiment) được tính toán từ hàng trăm bài viết tin tức trong cùng ngày.

        ---
                        """, unsafe_allow_html=True)
            if df1_full is not None:

                st.dataframe(df1_full)

        else:
            st.markdown("""
        Phân tích trực quan bằng lựa chọn các biến trong `Sidebar` để tính **ma trận tương quan**, khám phá mối quan hệ giữa  
        các chỉ số tài chính và cảm xúc tin tức.
                        
        ---
                        """, unsafe_allow_html=True)

            # VẼ MA TRẬN TƯƠNG QUAN
            corr_cols = select_corr_variables(df1)

            if len(corr_cols) >= 2:
                fig_corr = plot_corr_heatmap(df1, corr_cols)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("⚠️ Cần chọn ít nhất 2 biến để tính tương quan.")


    # TAB2
    if dashboard_option == "Dữ liệu tài chính":
        tab = st.radio(
            "Chọn chế độ hiển thị:",   # label hiển thị trên giao diện
            ["📋 Dữ liệu chi tiết", "📈 Phân tích biến động"],  # danh sách lựa chọn
            horizontal=True
        )

        if tab == "📋 Dữ liệu chi tiết": 
            st.markdown("""
            Dữ liệu tài chính được **thu thập và tổng hợp từ các nguồn công khai**, bao gồm **các chỉ số thị trường chính**, **hàng hóa chiến lược**, và **các thước đo biến động vĩ mô**.  
            Mỗi quan sát đại diện cho **một ngày giao dịch của thị trường Mỹ**, phản ánh **trạng thái tổng thể của nền kinh tế** thông qua biến động giá, khối lượng giao dịch và mối quan hệ giữa các loại tài sản.

            **Bộ dữ liệu tài chính bao gồm các nhóm chỉ số quan trọng:**
            -  **Chỉ số thị trường:** S&P 500 *(sp500_open, sp500_close, …)*
            -  **ETF theo dõi chỉ số:** SPY *(spy_open, spy_close, …)*
            -  **Chỉ số biến động:** VIX *(vix_open, vix_close, …)*
            -  **Hàng hóa:** Vàng *(gold_*)* và Dầu *(oil_*)*
            -  **Tiền tệ:** USD Index và ETF UUP *(usd_index_*, uup_*)*
                        
            ---
            """)

            table_placeholder = st.empty()
            table_placeholder.dataframe(df_fin)

            filter_fi = st.sidebar.multiselect(

                "Lọc theo Dữ liệu tài chính:",
                options=keywords
            )
            filtered_df = df_fin.copy()

            # Chọn keyword → tìm các cột chứa keyword → hiển thị dataframe
            if filter_fi:
                matcol = ["Ngày"] + [
                col for col in filtered_df.columns 
                if any(k in col for k in filter_fi)
                ]
                filtered_df = df_fin[matcol]
                table_placeholder.dataframe(filtered_df)

        else:
            st.markdown("""

            Thông qua **biểu đồ đường (line chart)** giúp **quan sát xu hướng và mối quan hệ động của các biến tài chính theo thời gian**.
                        
            **Tùy chọn linh hoạt các biến hiển thị** bằng `Selectbox` trong `Sidebar`:
            - Chọn **loại tài sản:** S&P 500, Vàng, Dầu, USD Index, v.v.  
            - Chọn **thuộc tính hiển thị:** Giá mở cửa *(Open)*, đóng cửa *(Close)*, hoặc khối lượng *(Volume)*.
            
            ---
            """)
            selected_col = filter_columns_by_selection(df1)

            if selected_col:
                fig = plot_single_timeseries_plotly(df1, "date", selected_col)
                st.plotly_chart(fig, use_container_width=True)







    # TAB3
    if dashboard_option == "Dữ liệu cảm xúc tin tức":

        selected_category = st.sidebar.selectbox(
            "Chọn loại dữ liệu", ["Điểm cảm xúc", "Tin tức gốc"]
        )
        if selected_category == "Điểm cảm xúc":
            tab = st.radio(
            "Chọn chế độ hiển thị:",   # label hiển thị trên giao diện
            ["📋 Dữ liệu chi tiết", "📈 Phân tích phân phối"],  # danh sách lựa chọn
            horizontal=True
        )

            if tab == "📋 Dữ liệu chi tiết": 
                st.markdown("""
            Dữ liệu tin tức được **thu thập, xử lý và tổng hợp từ các nguồn truyền thông uy tín**, nhằm phản ánh **tâm lý và cảm xúc của thị trường tài chính** qua từng ngày giao dịch.  
            Mỗi quan sát tương ứng với **một ngày**, tổng hợp các bài viết liên quan đến thị trường Mỹ, cổ phiếu, vàng, dầu, và các chủ đề kinh tế vĩ mô khác.

            **Bộ dữ liệu tin tức bao gồm các nhóm biến chính sau:**

            -  **Thống kê bài viết:** Số lượng bài viết tổng hợp mỗi ngày; Số bài **tích cực**, **trung lập**, **tiêu cực**
            -  **Tỷ trọng cảm xúc:** Tỷ lệ bài viết theo cảm xúc
            -  **Chỉ số cảm xúc tổng hợp:** Điểm trung bình và điểm trọng số cảm xúc 
            -  **Đặc trưng chi tiết về xác suất và nội dung:** Xác suất trung bình, trung vị, độ lệch chuẩn cảm xúc; Độ dài trung bình và trung vị của văn bản

            Dữ liệu này đóng vai trò quan trọng trong việc **phân tích mối quan hệ giữa tin tức và biến động tài chính**, giúp nhận diện **ảnh hưởng của tâm lý thị trường** đối với giá tài sản.
            
            ---                
            """)
                st.dataframe(df_news)
            else:

                st.markdown("""
            Phần này cung cấp **biểu đồ trực quan về phân phối cảm xúc tin tức** theo từng ngày hoặc toàn bộ giai đoạn nghiên cứu.  
            Mục tiêu là giúp người dùng **nắm bắt xu hướng tâm lý thị trường** và **đánh giá mức độ lạc quan, trung lập hoặc bi quan** trong dòng thông tin tài chính.

            Thông qua các **biểu đồ dạng cột*, bạn có thể:

            -  **Quan sát phân bố cảm xúc** tích cực – trung lập – tiêu cực theo thời gian.  
            -  **So sánh tỷ trọng cảm xúc** giữa các nhóm chủ đề.  
            -  **Phát hiện giai đoạn tâm lý bất thường** (ví dụ: khi tin tiêu cực tăng mạnh so với trung bình).  
            -  **Liên hệ biến động cảm xúc** với **xu hướng của thị trường tài chính** để tìm hiểu khả năng dự báo.

            Người dùng có thể **tùy chọn biến để biểu đồ hiển thị** để trực quan hóa dữ liệu theo nhu cầu phân tích.

            ---
            """)
                df_sent = split_finance_vs_sentiment(df1)

                # Sidebar chọn biến cảm xúc
                selected_sent_col = select_sentiment_column(df_sent)

                # Vẽ biểu đồ phân phối
                fig_dist = plot_sentiment_distribution_plotly(df_sent, selected_sent_col)
                st.plotly_chart(fig_dist, use_container_width=True)


        if selected_category == "Tin tức gốc":

            st.markdown("""
            <div style="display:flex; justify-content:center; margin-top:0px; margin-bottom:0px;">
                <div style="height:2.5px; width:190px; background-color:#1E90FF; border-radius:2px;"></div>
            </div>
            <h2 style='text-align:center; color:#1E90FF; margin-top:0;'>HIỂN THỊ TIN TỨC GỐC</h2>
            """, unsafe_allow_html=True)

            date_str = st.sidebar.text_input("Nhập ngày (ví dụ: 2023-01-30):", value="2023-01-01", key="txn_date")

            df_newss = load_data_news(date_str)
            if df_newss is not None:
                st.markdown("""
            Phần này cung cấp **dữ liệu tin tức ban đầu** được thu thập từ Polygon API, bao gồm các bài báo kinh tế – tài chính xuất bản từ năm 2023 đến tháng 10/2025.  
            Mỗi dòng dữ liệu đại diện cho ** thông tin của một bài viết được công bố trong ngày giao dịch**.

            Bộ dữ liệu bao gồm các thông tin chính:
            -  **Tiêu đề, tác giả, nguồn xuất bản và thời điểm đăng tải**.  
            -  **Từ khóa và mô tả ngắn** giúp nhận diện chủ đề bài viết.  
            -  **Chỉ số cảm xúc (Sentiment)** được trích xuất từ nội dung bài viết, gồm:
            - Nhãn cảm xúc: *tích cực / trung lập / tiêu cực*.  
            - Điểm xác suất và điểm cảm xúc tổng hợp.  

            Bạn có thể **lựa chọn ngày cụ thể trong thanh `Sidebar`** để xem các bài viết được đăng trong ngày đó.  

            ---
            """)


                # Thực hiện đổi tên
                df_newsss = df_newss.copy()
                df_newsss = rename_columns_if_exist_news(df_newsss)
                st.dataframe(df_newsss)

# Sidebar


  #  st.sidebar.slider("Chọn mức độ", min_value=0, max_value=100, value=50)

    # Nội dung chính
 #   st.write(f"Bạn đã chọn: {selected_category}")
#    st.write("Đây là nội dung trang Home.")

if __name__ == '__main__':
    tabdata()