import streamlit as st
import pandas as pd
import os

# --- TẠO FILE CONFIG ĐỂ BẬT DARK MODE MẶC ĐỊNH ---
config_dir = ".streamlit"
config_path = f"{config_dir}/config.toml"

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

# Ghi file config (mỗi lần chạy sẽ đảm bảo theme là dark)
with open(config_path, "w", encoding="utf-8") as f:
    f.write("""
[theme]
base="dark"
primaryColor="#5bc0be"
backgroundColor="#0b132b"
secondaryBackgroundColor="#1c2541"
textColor="#e0e6ed"
    """)

# --- CSS TÙY CHỈNH ---
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

# --- TIÊU ĐỀ CHÍNH ---
st.markdown(
    "<h1 style='text-align:center; font-weight:700;'>ỨNG DỤNG KỸ THUẬT BIG DATA - CLOUD COMPUTING</h1>",
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown(
    "<h2 style='text-align:center; font-weight:600; margin-top:-10px;'>DỰ ĐOÁN BIẾN ĐỘNG THỊ TRƯỜNG THÔNG QUA CHỈ SỐ TÀI CHÍNH VÀ CẢM XÚC THỊ TRƯỜNG</h2>",
    unsafe_allow_html=True
)

st.markdown("---")

# --- BẢNG TÊN THÀNH VIÊN ---
st.subheader("Danh sách thành viên thực hiện - Nhóm 9")
members = [
    ["Ngô Lam Giang", "K224141656"],
    ["Mai Thị Thanh Hà", "K224141659"],
    ["Lê Thị Hương", "K224141667"],
    ["Lê Thị Mỹ Uyên", "K224141707"],
    ["Phạm Thị Điệp Vân", "K224141708"],
]

# Tạo bảng có STT
df_members = pd.DataFrame(
    members,
    columns=["Họ và tên", "MSSV"]
)

df_members.index = df_members.index + 1  # bắt đầu STT từ 1
df_members.index.name = "STT"

st.table(df_members)

st.markdown("---")

# --- GIỚI THIỆU TRANG WEB ---
st.subheader("Giới thiệu hệ thống")

intro_text = """
Trang web này được xây dựng nhằm trình bày bộ dữ liệu sử dụng trong quá trình huấn luyện mô hình dự đoán biến động thị trường. 
Hệ thống gồm hai phần chính:

- **Trang Trực quan dữ liệu:** Hiển thị đầy đủ tập dữ liệu tài chính và dữ liệu cảm xúc trước khi đưa vào xử lý.
- **Trang Ứng dụng mô hình:** Áp dụng mô hình học máy có hiệu suất tốt nhất sau quá trình huấn luyện để đưa ra dự đoán theo thời gian thực.

Mục tiêu của hệ thống là cung cấp một cái nhìn trực quan về dữ liệu cũng như minh họa khả năng dự báo của mô hình.
"""

st.markdown(intro_text)