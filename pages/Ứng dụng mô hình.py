import streamlit as st
import joblib
import pandas as pd
import os
import time
from io import BytesIO
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
import numpy as np
import yfinance as yf
from typing import List, Dict
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import io
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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

# CONFIG
SYMBOL_MAP = {
    '^GSPC': 'sp500',
    'SPY': 'spy',
    '^VIX': 'vix',
    'GC=F': 'gold',
    'CL=F': 'oil',
    'DX-Y.NYB': 'usd_index',
    'UUP': 'uup'
}

POLYGON_API_KEY = "KXHaneBxKmIC0_oLJdUKqhRh4if7DsCz"  # thay bằng API key của bạn
MODEL_NAME = "yiyanghkust/finbert-tone"  # FinBERT
LOCAL_MODEL_DIR = "./models/finbert-tone/"
# Mapping symbol → tên chuẩn
SYMBOL_NAME_MAP = {
    "^GSPC": "sp500",
    "SPY": "spy",
    "^VIX": "vix",
    "GC=F": "gold",
    "CL=F": "oil",
    "DX-Y.NYB": "usd_index",
    "UUP": "uup"
}

# Cấu hình trang
st.sidebar.markdown("### Phục vụ dự đoán:")

dashboard_option = st.sidebar.selectbox(
    "Chọn chế độ:",
    (
        "Mô hình dự đoán",
        "Tải dữ liệu TEST"
    )
)

# Tiêu đề chính theo lựa chọn
st.markdown(
f"<h2 style='text-align: center; text-transform: uppercase;'>{dashboard_option}</h2>",
unsafe_allow_html=True
)

# Đường ngăn cách (divider) bên dưới menu
st.sidebar.markdown("---")

# ============== 2. Hàm hỗ trợ xử lý ===============
# ---------------FINANCE------------------
# Hàm đổi tên cột để hiển thị
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
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

def normalize_columns(df):
    """
    Chuẩn hoá tên cột tài chính theo format:
    sp500_open, gold_close, oil_volume, usd_index_close, ...
    và giữ nguyên các cột sentiment.
    """

    import re

    col_map = {}
    for col in df.columns:

        # 1) Giữ nguyên các cột sentiment, số lượng bài, return...
        if col in [
            "date", "sp500_return", "sp500_range", "sp500_return_lag1",
            "gold_return", "gold_range", "gold_return_lag1",
            "oil_return", "oil_range", "vix_close_lag1",
            "n_articles", "n_positive", "n_neutral", "n_negative",
            "prop_positive", "prop_neutral", "prop_negative",
            "mean_sentiment_score", "mean_sentiment_prob",
            "median_sentiment_prob", "std_sentiment_score",
            "weighted_sentiment_score", "avg_text_len", "median_text_len"
        ]:
            col_map[col] = col
            continue

        # 2) Map theo tiền tố tài sản
        if "_^GSPC" in col: prefix = "sp500"
        elif "_SPY" in col: prefix = "spy"
        elif "_^VIX" in col: prefix = "vix"
        elif "_GC=F" in col: prefix = "gold"
        elif "_CL=F" in col: prefix = "oil"
        elif "_DX-Y.NYB" in col: prefix = "usd_index"
        elif "_UUP" in col: prefix = "uup"
        else:
            col_map[col] = col
            continue

        # 3) Lấy loại giá: open, close, high, low, volume
        m = re.search(r"(open|close|high|low|volume)", col)
        if m:
            suffix = m.group(1)
        else:
            col_map[col] = col
            continue

        # 4) Gộp tên mới
        new_name = f"{prefix}_{suffix}"
        col_map[col] = new_name

    # 5) Đổi tên toàn DataFrame
    df = df.rename(columns=col_map)

    return df

# Hàm thêm các cột return, range, lag1
def add_financial_features(df):
    df = df.copy()

    # ----- RETURN -----
    if 'sp500_close' in df.columns:
        df['sp500_return'] = df['sp500_close'].pct_change()
    if 'gold_close' in df.columns:
        df['gold_return'] = df['gold_close'].pct_change()
    if 'oil_close' in df.columns:
        df['oil_return'] = df['oil_close'].pct_change()

    # ----- RANGE -----
    if 'sp500_high' in df.columns and 'sp500_low' in df.columns:
        df['sp500_range'] = df['sp500_high'] - df['sp500_low']
    if 'gold_high' in df.columns and 'gold_low' in df.columns:
        df['gold_range'] = df['gold_high'] - df['gold_low']
    if 'oil_high' in df.columns and 'oil_low' in df.columns:
        df['oil_range'] = df['oil_high'] - df['oil_low']

    # ----- LAG 1 -----
    if 'sp500_return' in df.columns:
        df['sp500_return_lag1'] = df['sp500_return'].shift(1)
    if 'vix_close' in df.columns:
        df['vix_close_lag1'] = df['vix_close'].shift(1)
    if 'gold_return' in df.columns:
        df['gold_return_lag1'] = df['gold_return'].shift(1)

    return df

# Hàm lấy dữ liệu từ Yahoo Finance
def fetch_financial_data(symbol_map, start_date, end_date):
    # --- FIX: cộng thêm 1 ngày vào end_date để Yahoo Finance lấy đủ ---
    end_date_fixed = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    all_data = []
    for symbol, name in symbol_map.items():
        data = yf.download(symbol, start=start_date, end=end_date_fixed, progress=False)

        if data.empty:
            st.warning(f"⚠️ Không có dữ liệu cho {symbol}.")
            continue
        data = data.rename(columns={
            'Open': f'{name}_open',
            'High': f'{name}_high',
            'Low': f'{name}_low',
            'Close': f'{name}_close',
            'Adj Close': f'{name}_adj_close',
            'Volume': f'{name}_volume'
        }).reset_index().rename(columns={'Date': 'date'})
        all_data.append(data)

    if not all_data:
        return pd.DataFrame()

    df_merged = all_data[0]
    for df in all_data[1:]:
        df_merged = pd.merge(df_merged, df, on='date', how='outer')

    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    return df_merged

# ---------------NEWS---------------------------
# Hàm hỗ trợ tải bài báo an toàn
def safe_request(url, params=None, headers=None, max_retry=4, sleep_sec=2, timeout=500):
    placeholder = st.empty()  # tạo placeholder để hiển thị trạng thái
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503):
                wait = sleep_sec * (2 ** (attempt - 1))
                placeholder.info(f"⚠ Retry {attempt}/{max_retry} sau {wait}s do status {r.status_code}")
                time.sleep(wait)
                continue
            r.raise_for_status()
            placeholder.empty()  # xóa thông báo khi thành công
            return r.json()
        except Exception as e:
            wait = sleep_sec * (2 ** (attempt - 1))
            placeholder.info(f"⚠ Attempt {attempt} thất bại: {e}. Retry sau {wait}s")
            time.sleep(wait)
    placeholder.empty()  # xóa nếu hết retry
    return None

# Hàm tải bài báo từ Polygon API
def fetch_news_for_date(date_str: str, limit: int = 50) -> List[Dict]:
    """Lấy bài báo trong ngày từ Polygon"""
    url = "https://api.polygon.io/v2/reference/news"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    params = {
        "published_utc.gte": f"{date_str}T00:00:00Z",
        "published_utc.lte": f"{date_str}T23:59:59Z",
        "limit": limit
    }
    resp = safe_request(url, params=params, headers=headers)
    articles = []
    if resp and "results" in resp:
        for a in resp["results"]:
            a["source"] = "Polygon"
            a["published_date"] = a.get("published_utc", "")[:10]
        articles = resp["results"]
        st.write(f"{date_str}: Tìm được {len(articles)} bài báo")
    else:
        st.info(f"{date_str}: Không có bài báo được tìm thấy")
    return articles

# Hàm chạy FinBERT để chấm điểm
def load_sentiment_pipeline(model_name: str = MODEL_NAME):
    st.info(" Đang tải FinBERT (ưu tiên bản local)...")

    # ---- 1) Load LOCAL trước ----
    if os.path.exists(LOCAL_MODEL_DIR):
        try:
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
            nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
            st.success(" FinBERT loaded từ local!")
            return nlp
        except:
            st.warning("⚠ Không load được local model, chuyển sang tải online...")

    # ---- 2) Fallback ONLINE ----
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    st.info(" Đang tải model từ HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Lưu lại LOCAL
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

    st.success(" Tải thành công (đã lưu local để dùng lần sau)")
    return nlp

# Hàm chấm điểm cảm xúc 
def infer_sentiment(nlp, texts: List[str], batch_size: int = 16) -> List[tuple]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        cleaned = [t if (t and str(t).strip() != "") else "" for t in batch]
        try:
            preds = nlp(cleaned)
            for p in preds:
                label = p.get("label", "NEUTRAL").upper()
                score = float(p.get("score", 0.0))
                score_num = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}.get(label, 0)
                results.append((label, score, score_num))
        except Exception as e:
            st.warning(f"Inference batch failed: {e}")
            for _ in batch:
                results.append(("NEUTRAL", 0.0, 0))
    return results

def aggregate_daily_sentiment(records: List[Dict], nlp=None) -> Dict:
    """Tổng hợp sentiment + thêm các thống kê nâng cao."""
    if not records:
        return {
            "date": None,
            "n_articles": 0,
            "n_positive": 0,
            "n_neutral": 0,
            "n_negative": 0,
            "prop_positive": float("nan"),
            "prop_neutral": float("nan"),
            "prop_negative": float("nan"),
            "mean_sentiment_score": float("nan"),
            "mean_sentiment_prob": float("nan"),
            "median_sentiment_prob": float("nan"),
            "std_sentiment_score": float("nan"),
            "weighted_sentiment_score": float("nan"),
            "avg_text_len": float("nan"),
            "median_text_len": float("nan"),
            "articles_df": pd.DataFrame()
        }

    df = pd.json_normalize(records)

    # Chuẩn hóa text
    def prepare_text(r):
        title = r.get("title") or r.get("headline") or ""
        desc = r.get("description") or r.get("summary") or ""
        return f"{title}. {desc}".strip()

    df["text_for_sentiment"] = df.apply(lambda row: prepare_text(row.to_dict()), axis=1)
    df["text_len"] = df["text_for_sentiment"].apply(lambda x: len(str(x)))

    texts = df["text_for_sentiment"].tolist()

    if nlp is None:
        nlp = load_sentiment_pipeline()

    preds = infer_sentiment(nlp, texts)
    labels, probs, nums = zip(*preds)

    df["sentiment_label"] = labels
    df["sentiment_score_prob"] = probs
    df["sentiment_score"] = nums

    # --- Metrics ---
    n = len(df)
    n_pos = int((df["sentiment_label"] == "POSITIVE").sum())
    n_neu = int((df["sentiment_label"] == "NEUTRAL").sum())
    n_neg = int((df["sentiment_label"] == "NEGATIVE").sum())

    prop_pos = n_pos / n
    prop_neu = n_neu / n
    prop_neg = n_neg / n

    mean_score = float(df["sentiment_score"].mean())
    mean_prob = float(df["sentiment_score_prob"].mean())
    median_prob = float(df["sentiment_score_prob"].median())
    std_score = float(df["sentiment_score"].std())

    # Sentiment có trọng số theo độ dài text
    weighted_sentiment = float((df["sentiment_score"] * df["text_len"]).sum() / df["text_len"].sum())

    avg_text_len = float(df["text_len"].mean())
    median_text_len = float(df["text_len"].median())

    return {
        "date": df["published_date"].iloc[0] if "published_date" in df.columns else None,

        "n_articles": n,
        "n_positive": n_pos,
        "n_neutral": n_neu,
        "n_negative": n_neg,

        "prop_positive": prop_pos,
        "prop_neutral": prop_neu,
        "prop_negative": prop_neg,

        "mean_sentiment_score": mean_score,
        "mean_sentiment_prob": mean_prob,
        "median_sentiment_prob": median_prob,
        "std_sentiment_score": std_score,
        "weighted_sentiment_score": weighted_sentiment,

        "avg_text_len": avg_text_len,
        "median_text_len": median_text_len,

        "articles_df": df
    }





# -------------- TỔNG HỢP -----------------
def df_to_parquet_bytes(df):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df)
    return pq.write_table(table, where=None)



# ============== 3. TAB1 ==============
if dashboard_option == "Mô hình dự đoán":
    # --- Chọn chế độ nhập liệu ---
    mode1, mode2 = st.tabs(["🔹 Nhập thủ công", "📁 Upload file dữ liệu"])
    st.markdown("---")
    st.sidebar.info("Nếu bạn muốn dùng file dữ liệu để dự đoán mà chưa có sẵn file, hãy chuyển sang tab **Tải dữ liệu TEST**")

    # ============================================================
    # 1. Nhập thủ công
    # ============================================================
    with mode1:
        with st.form("manual_form"):
            st.markdown("### Nhập Chỉ số tài chính")
            col1, col2, col3 = st.columns(3)

            with col1:
                sp500_open = st.number_input("SP500 Mở cửa", value=3810.0)
                sp500_high = st.number_input("SP500 Cao nhất", value=3850.0)
                sp500_low = st.number_input("SP500 Thấp nhất", value=3800.0)
                sp500_close = st.number_input("SP500 Đóng cửa", value=3840.0)
                sp500_volume = st.number_input("SP500 Khối lượng", value=3900000000.0)

                sp500_return = st.number_input("SP500 % Thay đổi", value=0.0)
                sp500_range = st.number_input("SP500 Biên độ", value=37.0)
                sp500_return_lag1 = st.number_input("SP500 % hôm trước", value=0.0)

                spy_open = st.number_input("SPY Mở cửa", value=370.0)
                spy_high = st.number_input("SPY Cao nhất", value=375.0)
                spy_low = st.number_input("SPY Thấp nhất", value=365.0)
                spy_close = st.number_input("SPY Đóng cửa", value=372.0)
                spy_volume = st.number_input("SPY Khối lượng", value=85900000.0)

                oil_return = st.number_input("Dầu % Thay đổi", value=0.0)
                oil_range = st.number_input("Dầu Biên độ", value=4.6)


            with col2:
                vix_open = st.number_input("VIX Mở cửa", value=22.0)
                vix_high = st.number_input("VIX Cao nhất", value=23.0)
                vix_low = st.number_input("VIX Thấp nhất", value=21.9)
                vix_close = st.number_input("VIX Đóng cửa", value=22.5)
                vix_volume = st.number_input("VIX Khối lượng", value=0.0)

                vix_close_lag1 = st.number_input("VIX hôm trước", value=22.5)

                gold_open = st.number_input("Vàng Mở cửa", value=1850.0)
                gold_high = st.number_input("Vàng Cao nhất", value=1880.0)
                gold_low = st.number_input("Vàng Thấp nhất", value=1849.0)
                gold_close = st.number_input("Vàng Đóng cửa", value=1872.0)
                gold_volume = st.number_input("Vàng Khối lượng", value=62.0)

                gold_return = st.number_input("Vàng % Thay đổi", value=0.0)
                gold_range = st.number_input("Vàng Biên độ", value=13.0)
                gold_return_lag1 = st.number_input("Vàng % hôm trước", value=0.0)



            with col3:
                oil_open = st.number_input("Dầu Mở cửa", value=77.2)
                oil_high = st.number_input("Dầu Cao nhất", value=77.4)
                oil_low = st.number_input("Dầu Thấp nhất", value=72.8)
                oil_close = st.number_input("Dầu Đóng cửa", value=73.5)
                oil_volume = st.number_input("Dầu Khối lượng", value=350000.0)

                usd_index_open = st.number_input("USD Index Mở cửa", value=103.0)
                usd_index_high = st.number_input("USD Index Cao nhất", value=104.0)
                usd_index_low = st.number_input("USD Index Thấp nhất", value=102.5)
                usd_index_close = st.number_input("USD Index Đóng cửa", value=103.5)
                usd_index_volume = st.number_input("USD Index Khối lượng", value=0.0)

                uup_open = st.number_input("UUP Mở cửa", value=25.2)
                uup_high = st.number_input("UUP Cao nhất", value=25.3)
                uup_low = st.number_input("UUP Thấp nhất", value=25.1)
                uup_close = st.number_input("UUP Đóng cửa", value=25.28)
                uup_volume = st.number_input("UUP Khối lượng", value=4400000.0)


            st.markdown("---")
            st.markdown("### Nhập Thông tin tin tức và Chỉ số cảm xúc")

            col4, col5, col6 = st.columns(3)
            with col4:
                n_articles = st.number_input("Số lượng bài báo", value=50.0)
                n_positive = st.number_input("Số bài tích cực", value=36.0)
                n_neutral = st.number_input("Số bài trung lập", value=8.0)
                n_negative = st.number_input("Số bài tiêu cực", value=6.0)
                prop_positive = st.number_input("Tỷ lệ tích cực", value=0.72)

            with col5:
                prop_neutral = st.number_input("Tỷ lệ trung lập", value=0.16)
                prop_negative = st.number_input("Tỷ lệ tiêu cực", value=0.12)
                mean_sentiment_score = st.number_input("Điểm sentiment trung bình", value=0.56)
                weighted_sentiment_score = st.number_input("Điểm sentiment trọng số", value=0.9657)
                mean_sentiment_prob = st.number_input("Xác suất sentiment trung bình", value=0.9)

            with col6:
                median_sentiment_prob = st.number_input("Xác suất sentiment trung vị", value=0.75)
                std_sentiment_score = st.number_input("Độ lệch chuẩn sentiment", value=0.57)
                avg_text_len = st.number_input("Độ dài văn bản trung bình", value=225.0)
                median_text_len = st.number_input("Độ dài văn bản trung vị", value=186.0)


            submit_manual = st.form_submit_button(" Bắt đầu dự đoán")

        if submit_manual:
            # gom dữ liệu thành DataFrame
            input_data = pd.DataFrame([[locals()[col] for col in numeric_cols]], columns=numeric_cols)
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0]
            inv_map = {v: k for k, v in label_mapping.items()}
            pred_label = inv_map[pred]

            st.success(" Dự đoán hoàn tất!")

            color = "#2E8B57" if pred_label == "up" else "#C0392B"
            st.markdown(f"<h3 style='text-align:center; color:{color};'> Dự đoán Xu hướng thị trường: {pred_label.upper()}</h3>", unsafe_allow_html=True)
            st.progress(prob[pred])
            st.write(f"**Mức độ chính xác:** {prob[pred]*100:.2f}%")

    # ============================================================
    # 2. Upload file dữ liệu
    # ============================================================
    with mode2:
        st.markdown("###  Upload File Dữ Liệu để thực hiện dự đoán")
        st.markdown("""
            Phần này cần dùng file dữ liệu có định dạng phù hợp để dự đoán. 
            Nếu chưa có sẵn file dữ liệu, hãy Chọn chế độ *Tải dữ liệu TEST* trong `Sidebar` để tải dữ liệu.

            ---
            """)

        uploaded_file = st.file_uploader("Tải file dữ liệu", type=["parquet"])

        if uploaded_file is not None:
            df1 = pd.read_parquet(uploaded_file)
            st.dataframe(df1)
            st.success(" File đã đọc thành công!")
            df_new = df1.dropna()

            # Kiểm tra cột hợp lệ
            missing_cols = [c for c in numeric_cols if c not in df_new.columns]
            if missing_cols:
                st.error(f"⚠️ File thiếu các cột cần thiết: {missing_cols}")
            else:
                st.markdown("""
                                
                Để thực hiện dự đoán cần bỏ qua những hàng dữ liệu bị **NaN**, dưới đây là dữ liệu hợp lệ sau xử lý từ dữ liệu được tải lên để thực hiện dự đoán.

                ---
                """)

                st.dataframe(df_new)

                st.markdown("---")

                st.markdown("<h4 style='color:#4CAF50;'> Bấm nút dưới đây để thực hiện dự đoán:</h4>", unsafe_allow_html=True)
                
                # Nút kích hoạt dự đoán
                predict_button = st.button(" Bắt đầu dự đoán")

                if predict_button:
                    preds = model.predict(df_new[numeric_cols])
                    probs = model.predict_proba(df_new[numeric_cols])
                    inv_map = {v: k for k, v in label_mapping.items()}

                    # Sau khi dự đoán
                    df_new["Prediction"] = [inv_map[p] for p in preds]
                    df_new["Confidence (%)"] = (probs.max(axis=1) * 100).round(2)

                    st.success(" Dự đoán hoàn tất!")
                    st.markdown("###  Kết quả dự đoán:")

                    st.markdown("""
            - Cột *Prediction* hiển thị **Dự đoán Xu hướng thị trường** ngày tiếp theo. 
            - Cột *Confidence (%)* hiện thị **Mức độ tin cậy của dự đoán**.
            - Bạn có thể kiểm chứng kết quả dự đoán bằng cách xem *sp500_close* của ngày hôm sau.
    
            """)

                    # Hiển thị kết quả an toàn
                
                    cols_to_add = df_new.columns.difference(["Prediction", "Confidence (%)"])
                    df_to_show = df_new[["Prediction", "Confidence (%)"]].join(df_new[cols_to_add])

                    # Đưa sp500_close lên vị trí thứ 3
                    cols = list(df_to_show.columns)

                    if "sp500_close" in cols:
                        cols.remove("sp500_close")
                        cols.insert(2, "sp500_close")  # vị trí thứ 3 (index 2)
                        df_to_show = df_to_show[cols]

                    # Hàm highlight cột
                    def highlight_col(col):
                        styles = []
                        for _ in col:
                            if col.name == "sp500_close":
                                styles.append('background-color: yellow; font-weight: bold')
                            elif col.name == "Prediction":
                                styles.append('background-color: blue; font-weight: bold')
                            elif col.name == "Confidence (%)":
                                styles.append('background-color: green; font-weight: bold')
                            else:
                                styles.append('')
                        return styles
                    
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

if dashboard_option == "Tải dữ liệu TEST":
    # --- Chọn chế độ nhập liệu ---
    mode1, mode2, mode3 = st.tabs(["🔹 Dữ liệu tài chính", "🔹 Dữ liệu tin tức", "📁 Tổng hợp dữ liệu"])
    st.markdown("---")

    # --- Sidebar cấu hình ---
    start_date = st.sidebar.date_input("Chọn ngày bắt đầu", value=date(2025, 11, 1))
    end_date = st.sidebar.date_input("Chọn ngày kết thúc", value=date.today() - timedelta(days=1))

    if start_date > end_date:
        st.error("Ngày bắt đầu phải <= ngày kết thúc")
        st.stop()

    # --- Khởi tạo version và reset khi đổi ngày ---
    if "last_start" not in st.session_state:
        st.session_state.last_start = start_date
    if "last_end" not in st.session_state:
        st.session_state.last_end = end_date
    if "data_version" not in st.session_state:
        st.session_state.data_version = 0
    if "last_data_version" not in st.session_state:
        st.session_state.last_data_version = 0

    if start_date != st.session_state.last_start or end_date != st.session_state.last_end:
        # Reset Mode1
        st.session_state.data_loaded = False
        st.session_state.features_generated = False
        st.session_state.df_raw = pd.DataFrame()

        # Reset Mode2
        st.session_state.step1_done = False
        st.session_state.step2_done = False
        st.session_state.news_data = {}
        st.session_state.sentiment_data = {}
        st.session_state.df_sentiment_summary = pd.DataFrame()

        # Tăng version để Mode2 nhận biết dữ liệu mới
        st.session_state.data_version += 1
        st.session_state.last_data_version = st.session_state.data_version

        # Cập nhật lại ngày
        st.session_state.last_start = start_date
        st.session_state.last_end = end_date


    # =================TAB1==================
    with mode1:

        st.markdown("###  Bước 1: Tải dữ liệu tài chính từ Yahoo Finance")
        
        st.markdown("""
            Để bắt đầu, hãy chọn khoảng thời gian bạn muốn tải dữ liệu tại `Sidebar` và bấm nút dưới đây.
            """)

        # --- Bước 1: Nút tải dữ liệu ---
        if st.button("Tải dữ liệu tài chính"):
            # --- Tăng version mỗi khi dữ liệu mới được tải ---
            st.session_state.data_version = st.session_state.get("data_version", 0) + 1
            with st.spinner("Đang tải dữ liệu..."):
                df_raw = fetch_financial_data(SYMBOL_MAP, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                df_raw.columns = ['_'.join(col).strip() for col in df_raw.columns.values]



            if df_raw.empty:
                st.error("Không có dữ liệu trong khoảng thời gian này.")
                st.session_state.data_loaded = False
            else:
                st.session_state.df_raw = df_raw
                st.session_state.data_loaded = True
                st.session_state.features_generated = False
                st.success(f"Đã tải {len(df_raw)} ngày dữ liệu.")

        # --- Hiển thị bảng 1 nếu có dữ liệu ---
        if st.session_state.get("data_loaded", False):
            st.markdown(
                "<h3 style='text-align: center;'>Bảng 1: Dữ liệu gốc</h3>",
                unsafe_allow_html=True
            )

            st.dataframe(st.session_state.df_raw, use_container_width=True)

            # --- Nút tiếp tục xử lý ---
            if st.button("Tiếp tục xử lý"):
                with st.spinner("Đang xử lý dữ liệu..."):
                    # 1) Tạo các cột features cơ bản

                    df_features_1 = st.session_state.df_raw.copy()
                    df_features_1 = normalize_columns(df_features_1)
                
                    df_features = add_financial_features(df_features_1)

                    #df_features_extended = pd.DataFrame(list(df_features))

                # Lưu vào session
                st.session_state.df_features = df_features
                st.session_state.features_generated = True

        # --- Hiển thị bảng 2 nếu đã xử lý ---
        if st.session_state.get("features_generated", False):
            st.markdown(
                "<h3 style='text-align: center;'>Bảng 2 : Dữ liệu sau xử lý</h3>",
                unsafe_allow_html=True
            )
            st.dataframe(st.session_state.df_features, use_container_width=True)
            st.success(
                "Bạn đã hoàn thành tải và xử lý dữ liệu tài chính, hãy kéo lên đầu trang và chuyển sang tab **Dữ liệu tin tức** để làm bước tiếp theo"
            )


    with mode2:
        
        st.markdown("###  Bước 2: Tải dữ liệu tin tức và chấm điểm cảm xúc")

        if "initialized_news" not in st.session_state:
            st.session_state.initialized_news = True
            st.session_state.last_data_version = st.session_state.get("data_version", 0)

        # Nếu dữ liệu tài chính thay đổi → reset TAB2
        if st.session_state.get("last_data_version") != st.session_state.get("data_version"):
            st.session_state.step1_done = False
            st.session_state.step2_done = False
            st.session_state.news_data = {}
            st.session_state.sentiment_data = {}

            # cập nhật version đã xử lý
            st.session_state.last_data_version = st.session_state.get("data_version")


        # --- Kiểm tra bước 1 ---
        if not st.session_state.get("data_loaded", False) or not st.session_state.get("features_generated", False):
            st.warning("⚠ Bạn cần hoàn tất Bước 1 (tải và xử lý dữ liệu tài chính) trước khi đến đây.")
        else:
            # Nội dung chính bước 2 chỉ hiện khi bước 1 đã xong
            st.markdown("""
                Để tiếp tục, chúng tôi xác nhận ngày bạn chọn từ "Bước 1".
            """)


            # --- Hiển thị xác nhận ngày ---
            st.info(f"**Khoảng thời gian đã chọn:** từ **{start_date.strftime('%d/%m/%Y')}** đến **{end_date.strftime('%d/%m/%Y')}**")

            # Chọn khoảng ngày
            limit_articles = st.number_input("Chọn số bài tối đa mỗi ngày", min_value=1, max_value=100, value=50, step=1)

            # --- Tạo các vùng giao diện cố định ---
            box_step1 = st.container()
            box_step2 = st.container()
            box_step3 = st.container()

            # --- Step 1: Tải dữ liệu ---
            with box_step1:
                if "step1_done" not in st.session_state:
                    st.session_state.step1_done = False
                if "news_data" not in st.session_state:
                    st.session_state.news_data = {}

                if not st.session_state.step1_done:
                    if st.button(" Tải dữ liệu tin tức"):
                        st.session_state.news_data = {}
                        cur_date = start_date
                        total_days = (end_date - start_date).days + 1
                        pbar = st.progress(0)
                        day_idx = 0

                        while cur_date <= end_date:
                            date_str = cur_date.strftime("%Y-%m-%d")
                            articles = fetch_news_for_date(date_str, limit=limit_articles)
                            st.session_state.news_data[date_str] = articles or []

                            day_idx += 1
                            pbar.progress(day_idx / total_days)
                            cur_date += timedelta(days=1)

                        st.session_state.step1_done = True
                        st.success(" Đã tải xong dữ liệu.")

                # Hiển thị dữ liệu nếu đã tải xong
                if st.session_state.step1_done:
                    with st.expander(" Xem dữ liệu tin tức từng ngày"):
                        for date_str, articles in st.session_state.news_data.items():
                            st.markdown(f"**{date_str}** - {len(articles)} bài")
                            if articles:
                                st.dataframe(
                                    pd.DataFrame(articles)[["published_date", "title", "description"]]
                                )

            # --- Step 2: Chấm sentiment ---
            with box_step2:
                st.markdown("---")

                if st.session_state.step1_done and not st.session_state.step2_done:
                    if st.button(" Chấm điểm Sentiment"):
                        nlp_model = load_sentiment_pipeline()
                        st.session_state.sentiment_data = {}

                        for date_str, articles in st.session_state.news_data.items():
                            if articles:
                                daily_sentiment = aggregate_daily_sentiment(articles, nlp=nlp_model)
                                st.session_state.sentiment_data[date_str] = daily_sentiment

                                with st.expander(f" Sentiment ngày {date_str}"):
                                    st.write(f"Số bài: {daily_sentiment['n_articles']}")
                                    st.write(pd.DataFrame([{
                                        k: v for k, v in daily_sentiment.items() if k != "articles_df"
                                    }]))

                        st.session_state.step2_done = True
                        st.success(" Hoàn tất chấm điểm sentiment từng ngày.")

                elif st.session_state.step2_done:

                    # --- Hiển thị lại toàn bộ expander và bảng dữ liệu ---
                    for date_str, daily_sentiment in st.session_state.sentiment_data.items():
                        with st.expander(f" Sentiment ngày {date_str}"):
                            st.write(f"Số bài: {daily_sentiment['n_articles']}")
                            st.write(pd.DataFrame([{
                                k: v for k, v in daily_sentiment.items() if k != "articles_df"
                            }]))


            # --- Step 3: Tổng hợp sentiment ---

                with box_step3:
                    st.markdown("---")
                    if st.session_state.step2_done:
                        if st.button(" Tổng hợp sentiment theo ngày"):
                            st.session_state.df_sentiment_summary = pd.DataFrame([
                                {k: v for k, v in s.items() if k != "articles_df"}
                                for s in st.session_state.sentiment_data.values()
                            ])
                            st.subheader(" Summary tổng hợp tất cả ngày")
                            st.dataframe(st.session_state.df_sentiment_summary)
                            if "df_sentiment_summary" not in st.session_state:
                                st.session_state.df_sentiment_summary = pd.DataFrame()

                            st.success(
                "Bạn đã hoàn thành tải và xử lý dữ liệu tin tức, hãy kéo lên đầu trang và chuyển sang tab **Tổng hợp dữ liệu** để làm bước tiếp theo"
            )



    with mode3:
        st.markdown("###  Bước 3: Tổng hợp dữ liệu tài chính và tin tức đã xử lý")
        # --- Kiểm tra Bước 2 + version ---
        if st.session_state.get("last_data_version", 0) != st.session_state.get("data_version", 0):
            st.warning("⚠ Bạn cần hoàn tất Bước 2 với dữ liệu tài chính mới trước khi đến đây.")
        elif not st.session_state.get("step2_done", False) or st.session_state.get("df_sentiment_summary") is None:
            st.warning("⚠ Bạn cần hoàn tất Bước 2 (tin tức + chấm sentiment + tổng hợp) trước khi đến đây.")
        else:

            if st.session_state.get("step2_done", False):
                st.markdown("""
            Sau khi bạn đã tải dữ liệu tài chính và tin tức và xử lý thành công, hãy bấm nút dưới đây để tổng hợp lại thành một file sẵn sàng thực hiện dự đoán

            ---
            """)
                if st.button(" Chạy tổng hợp dữ liệu thành bản Final sẵn sàng dự đoán"):


                    # Lấy dữ liệu từ session_state
                    df_all = st.session_state.df_sentiment_summary
                    df_features = st.session_state.df_features.copy()

                    df_all['date'] = pd.to_datetime(df_all['date'], format="%Y-%m-%d")
                    df_features['date_'] = pd.to_datetime(df_features['date_'], format="%Y-%m-%d")
                    df_features = df_features.rename(columns={'date_': 'date'})

                    df_merged = pd.merge(df_features, df_all, on='date', how='inner')
                    st.dataframe(df_merged)


                    # Chuyển df thành file parquet trong RAM
                    buffer = io.BytesIO()
                    table = pa.Table.from_pandas(df_merged)
                    pq.write_table(table, buffer)
                    buffer.seek(0)

                    # Tạo tên file từ ngày bắt đầu / kết thúc
                    file_name = f"summary_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.parquet"

                    # Nút tải file parquet
                    st.download_button(
                        label=" Tải về file Parquet",
                        data=buffer,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )



