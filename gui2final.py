# gui2final.py
# Merged: original gui2_final.py + posting/admin features from final.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from pathlib import Path

# Added imports from final.py
from io import BytesIO
import os
import uuid
import numpy as np
from utils.tim_xe_tuong_tu_utils import Tim_xe, Tim_xe_ID
# Helpers & config - ensure these modules exist in your project (as in final.py)
try:
    from utils_clean_data import clean_motobike_data
    from utils_anomaly import run_price_anomaly_detection_with_reason
    from helpers import (df_to_excel_bytes, qtv_login, _read_xlsx_if_exists,
                         _save_xlsx, normalize_datetime_like_columns, make_post_record,
                         save_post_record, reorder_columns, rename_columns_vn,
                         ensure_post_id)
    from config import *
except Exception as e:
    # If imports fail, show a readable message but allow the app to load (some pages will error if used)
    st.warning(f"Không thể import một số helper/config: {e}")

# --- Các hàm xử lý---
def format_with_color(label, value, default_text="Chưa có thông tin", color=None):
    if value == default_text or not color:
        return f"**{label}:** {value}"
    return f"**{label}:** <span style='color: {color}'>{value}</span>"

def show_list_card(df_ket_qua):
    st.markdown("""
    <style>
        .bike-card {
            border: 1px solid #e0e0e0;
            border-radius: 14px;
            padding: 18px;
            margin: 20px 0;
            background: #ffffff;
            transition: 0.2s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .bike-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        }
        .bike-header {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 12px;
            color: #0A6EBD;
        }
        .bike-info {
            font-size: 15px;
            margin-bottom: 12px;
        }
        .bike-info p {
            margin: 8px 0;
        }
        .desc-box {
            background: #f4f7fb;
            padding: 12px;
            border-radius: 10px;
            font-size: 15px;
            border-left: 4px solid #0A6EBD;
            white-space: normal;
            word-wrap: break-word;
            max-height: 220px;
            max-width: 100%;
            overflow-y: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    for idx, (_, row) in enumerate(df_ket_qua.iterrows(), 1):
        with st.container():
            st.markdown(f"##### 🔊 {row['Tiêu đề']}")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    f"##### <span style='color: #0A6EBD; font-weight: bold;'>#{idx}</span> - Mã xe: {row['id']}",
                    unsafe_allow_html=True
                )
                price = f"{float(row['Giá']):.1f}".replace(".", ",")
                min_price = f"{float(row['Khoảng giá min']):.1f}".replace(".", ",")
                max_price = f"{float(row['Khoảng giá max']):.1f}".replace(".", ",")
                raw_mileage = row.get('Số Km đã đi')
                if pd.notna(raw_mileage) and str(raw_mileage).strip() and str(raw_mileage).strip().isdigit():
                    mileage = f"{int(raw_mileage):,}".replace(",", ".")
                else:
                    mileage = 'Chưa cập nhật'
                registration_year = row.get('Năm đăng ký', 'Chưa cập nhật')
                mileage_display = f"<span style='color: orange; font-weight: bold;'>{mileage}</span>" if mileage != 'Chưa cập nhật' else mileage
                st.markdown(f"""
                **💰 Giá:** <span style='color: red;'>{price} triệu</span>  
                    **Giá min:** {min_price} triệu  
                    **Giá max:** {max_price} triệu  
                **🛣️ Số Km đã đi:** {mileage_display}  
                **📅 Năm đăng ký:** {registration_year}
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("##### 📝 Mô tả chi tiết")
                brand_model = f"{row.get('Thương hiệu', '')} - {row.get('Dòng xe', '')} - {row.get('Xuất xứ', '')}".strip(' -')
                raw_desc = str(row['Mô tả chi tiết'] or "")
                lines = [ln.strip() for ln in raw_desc.splitlines() if ln.strip()]
                description = " ".join(lines)
                st.markdown(f"""
                <div class='desc-box' style='padding: 8px 10px; max-width: 90%;'>
                    <div style="font-weight:bold; color:#0A6EBD; margin-bottom:4px;">
                        {brand_model if brand_model != '-' else 'Chưa cập nhật thông tin'}
                    </div>
                    <div style="line-height:1.4; margin:0;">
                        {description}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

def load_models():
    import numpy as np
    from pathlib import Path
    models_dir = Path('models')
    model_paths = {
        'best_model_0': models_dir / 'best_regressor_cluster_0.pkl',
        'best_model_1': models_dir / 'best_regressor_cluster_1.pkl',
        'best_model_2': models_dir / 'best_regressor_cluster_2.pkl'
    }
    loaded_models = {}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {path.absolute()}")
    for name, path in model_paths.items():
        try:
            with open(path, 'rb') as f:
                loaded_models[name] = pickle.load(f)
        except Exception as e:
            raise Exception(f"Lỗi khi tải mô hình {name} từ {path}: {str(e)}")
    return (
        loaded_models['best_model_0'],
        loaded_models['best_model_1'],
        loaded_models['best_model_2']
    )

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Hệ thống gợi ý & dự đoán giá xe máy cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container {
        max-width: 90%;
        padding: 2rem 5%;
    }
    section[data-testid="stSidebar"] {
        width: 250px !important;
    }
    div[data-testid="stExpander"] {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("🏠 Xe máy cũ")

# append new menu items as requested
nav = st.sidebar.radio(
    "Chọn chức năng:",
    ["Giới thiệu",
    "Gợi ý tìm xe", 
    "Phân cụm và Dự đoán giá xe", 
    "Tin đăng bán", 
    "Tin đăng mua", 
    "Duyệt tin (QTV)", 
    "Thông tin tác giả"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ")
st.sidebar.info("Hệ thống hỗ trợ tìm xe phù hợp và dự đoán giá xe máy cũ từ dữ liệu Chợ Tốt.")

# --- Load dữ liệu đã lưu ---
@st.cache_data
def load_data():
    try:
        from pathlib import Path
        base = Path(__file__).parent
        file_path = base / "data" / "data_content_cleaned.xlsx"
        if not file_path.exists():
            st.error(f"❌ Không tìm thấy file: {file_path}")
            return None
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi đọc file: {e}")
        return None

df = load_data()

@st.cache_resource
def load_pipeline_and_data():
    with open("models/clustering_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    cluster_summary = pd.read_csv("data/cluster_summary.csv")
    clustered_data = pd.read_csv("data/clustered_data.csv")

    cluster_categorical_mode = None
    cluster_categorical_distributions = {}

    try:
        cluster_categorical_mode = pd.read_csv("data/cluster_categorical_mode.csv")
    except Exception:
        cluster_categorical_mode = None

    categorical_cols = [
        "Thương_hiệu",
        "Dòng_xe",
        "Loại_xe",
        "Dung_tích_xe",
        "Xuất_xứ",
        "is_moi",
        "is_do_xe",
        "is_su_dung_nhieu",
        "is_bao_duong",
        "is_do_ben",
        "is_phap_ly",
    ]

    for col in categorical_cols:
        try:
            dist_df = pd.read_csv(f"data/cluster_categorical_dist_{col}.csv")
            cluster_categorical_distributions[col] = dist_df
        except Exception:
            continue

    return (
        pipeline,
        cluster_summary,
        clustered_data,
        cluster_categorical_mode,
        cluster_categorical_distributions,
    )

def prepare_input_dataframe(pipeline, **kwargs):
    important_original_features = pipeline["important_original_features"]
    input_df = pd.DataFrame([kwargs])
    input_df = input_df[important_original_features]
    return input_df

def predict_cluster_and_price(pipeline, cluster_summary, input_df):
    import numpy as np
    numeric_features = pipeline["numeric_features"]
    categorical_features = pipeline["categorical_features"]
    ohe = pipeline["ohe"]
    scaler = pipeline["scaler"]
    model = pipeline["model"]
    important_ohe_features = pipeline["important_ohe_features"]
    input_num = input_df[numeric_features]
    input_cat = input_df[categorical_features].copy()
    for col in categorical_features:
        input_cat[col] = input_cat[col].astype(str).fillna("")
    input_cat = input_cat.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    try:
        encoded_array = ohe.transform(input_cat)
    except Exception as e:
        raise ValueError(f"Lỗi OHE transform: {e}\nGiá trị input: {input_cat}")
    encoded_cols = ["E_" + name for name in ohe.get_feature_names_out(categorical_features)]
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=input_df.index)
    input_num_scaled = pd.DataFrame(
        scaler.transform(input_num),
        columns=numeric_features,
        index=input_df.index,
    )
    full_encoded_scaled = pd.concat([input_num_scaled, encoded_df], axis=1)
    X_input = full_encoded_scaled[important_ohe_features].astype(float)
    cluster_id = int(model.predict(X_input)[0])
    row = cluster_summary[cluster_summary["cluster_id"] == cluster_id]
    price_info = None
    if not row.empty:
        price_info = {
            "count": int(row["Giá_count"].iloc[0]),
            "mean": float(row["Giá_mean"].iloc[0]),
            "min": float(row["Giá_min"].iloc[0]),
            "max": float(row["Giá_max"].iloc[0]),
        }
    return cluster_id, price_info

# --- Load models and data (only once) ---
if 'models_loaded' not in st.session_state:
    loading_placeholder = st.empty()
    try:
        with loading_placeholder.container():
            st.info("🔄 Đang tải các mô hình và dữ liệu...")
        (
            st.session_state.pipeline,
            st.session_state.cluster_summary,
            st.session_state.clustered_data,
            st.session_state.cluster_categorical_mode,
            st.session_state.cluster_categorical_distributions,
        ) = load_pipeline_and_data()
        best_model_0, best_model_1, best_model_2 = load_models()
        st.session_state.best_model_0 = best_model_0
        st.session_state.best_model_1 = best_model_1
        st.session_state.best_model_2 = best_model_2
        st.session_state.models_loaded = True
        with loading_placeholder.container():
            st.success("✅ Đã tải xong tất cả các mô hình và dữ liệu!")
            time.sleep(1)
        loading_placeholder.empty()
    except FileNotFoundError as e:
        loading_placeholder.empty()
        st.error(f"❌ Lỗi: Không tìm thấy file mô hình hoặc dữ liệu. Vui lòng kiểm tra đường dẫn: {str(e)}")
        st.stop()
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"❌ Lỗi khi tải mô hình hoặc dữ liệu: {str(e)}")
        st.stop()

pipeline = st.session_state.get('pipeline')
cluster_summary = st.session_state.get('cluster_summary')
clustered_data = st.session_state.get('clustered_data')
cluster_categorical_mode = st.session_state.get('cluster_categorical_mode')
cluster_categorical_distributions = st.session_state.get('cluster_categorical_distributions')

# --------------------- Initialize posts session_state (from final.py) ---------------------
# These variables rely on helpers/config (POSTS_* constants)
if "last_clean" not in st.session_state:
    st.session_state["last_clean"] = None
if "predicted_df" not in st.session_state:
    st.session_state["predicted_df"] = None
if "last_predict" not in st.session_state:
    st.session_state["last_predict"] = None

# load persisted posts from excel if exist (ensure post_id & normalize datetimes)
if "posts_sell" not in st.session_state:
    posts = _read_xlsx_if_exists(POSTS_SELL_XLSX)
    posts = ensure_post_id(posts)
    posts = normalize_datetime_like_columns(posts)
    st.session_state["posts_sell"] = posts
if "posts_buy" not in st.session_state:
    posts = _read_xlsx_if_exists(POSTS_BUY_XLSX)
    posts = ensure_post_id(posts)
    posts = normalize_datetime_like_columns(posts)
    st.session_state["posts_buy"] = posts
if "pending_notifications" not in st.session_state:
    st.session_state["pending_notifications"] = []

# ----------GIAO DIỆN CHÍNH ------------------------
if nav == "Giới thiệu":
    st.title("🏍️ Hệ thống gợi ý xe máy và dự đoán giá xe máy cũ")
    st.markdown("---")
    st.markdown("""
    ## 🌟 Giới thiệu hệ thống

    **Chợ Tốt** là một trong những nền tảng mua bán trực tuyến lớn nhất Việt Nam, 
    nơi mỗi ngày có hàng ngàn tin đăng về xe máy. Điều này khiến người dùng gặp khó khăn khi:

    - Tìm kiếm chiếc xe phù hợp giữa vô số tin đăng.
    - Đánh giá xem **mức giá người bán đưa ra có hợp lý hay không**.

    Để hỗ trợ trải nghiệm người dùng, hệ thống này được xây dựng với hai chức năng chính:
    """)

    st.markdown("""
    ---

    ## 🚀 1. Gợi ý xe máy tương tự

    Hệ thống gợi ý danh sách các xe có đặc điểm tương tự với lựa chọn của người dùng:

    - Người dùng chọn thông tin mô tả chiếc xe mong muốn.
    - Hệ thống truy vấn và trả về danh sách xe tương tự.
    - Có thể tuỳ chọn số lượng xe muốn hiển thị.

    """)

    st.markdown("""
    ---

    ## 💰 2. Dự đoán giá xe máy cũ

    Hệ thống hỗ trợ định giá dựa trên các yếu tố như:

    - Thương hiệu
    - Độ phổ biến
    - Giá tham khảo
    - Năm sản xuất
    - Tình trạng sử dụng  
    - Các đặc điểm kỹ thuật khác

    Hệ thống áp dụng các kỹ thuật **phân cụm (clustering)** để phân chia xe vào những phân khúc thị trường riêng biệt trước khi dự đoán, giúp mô hình đưa ra mức giá ước lượng **chính xác và phù hợp hơn**.

    ---
    """)
    st.info("""✨ Hệ thống được xây dựng nhằm hỗ trợ người dùng lựa chọn xe dễ dàng hơn và tham khảo mức giá hợp lý trên thị trường.

        Thực hiện bởi nhóm sinh viên 
            Data Science Class - TTTH ĐH Khoa học Tư nhiên:
            - Nguyễn Thị Tuyết Anh
            - Nguyễn Văn Cường
            - Hồ Thị Quỳnh Như
            
            Giáo viên hướng dẫn: ThS. Khuất Thùy Phương
        """)

# ========================= TÌM XE TƯƠNG TỰ ============================
elif nav == "Gợi ý tìm xe":
    st.title("🔎 Tìm xe theo nội dung gợi ý")
    st.markdown("---")
    search_type = st.radio(
        "Chọn phương thức tìm kiếm:",
        ["Tìm theo mô tả", "Tìm theo xe đã đăng"]
    )
    so_luong_xe = st.number_input("Số xe muốn tìm", min_value=1, max_value=10, value=5, step=1)
    if search_type == "Tìm theo mô tả":
        noi_dung = st.text_input("Nhập nội dung cần tìm", placeholder="VD: Vision, còn mới, giấy tờ đầy đủ...")
    else:
        df['display_text'] = df.apply(
            lambda row: f"Xe {row.name}: {row.get('Dòng xe', '')} - {row.get('Thương hiệu', '')} - {row.get('Giá', '')} triệu - {row.get('Tiêu đề', '')}",
            axis=1
        )
        sorted_df = df.sort_index()
        selected_display = st.selectbox(
            "Chọn xe có sẵn từ hệ thống:",
            options=sorted_df['display_text'].tolist(),
            index=0,
            format_func=lambda x: x,
            key="bike_selector"
        )
        selected_id = int(selected_display.split(':')[0].replace('Xe', '').strip()) if selected_display else None
        if selected_id is not None and selected_id in df.index:
            selected_bike = df.loc[selected_id]
            st.markdown("##### ⭐ Thông tin xe được chọn")
            st.markdown(f"#### 🔊 {selected_bike.get('Tiêu đề')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Thương hiệu:** {selected_bike.get('Thương hiệu', 'Chưa có thông tin')}")
                st.markdown(format_with_color("Dòng xe", selected_bike.get('Dòng xe', 'Chưa có thông tin'), color='blue'), unsafe_allow_html=True)
                st.markdown(f"**Loại xe:** {selected_bike.get('Loại xe', 'Chưa có thông tin')}")
                st.markdown(format_with_color("Xuất xứ", selected_bike.get('Xuất xứ', 'Chưa có thông tin'), color='green'), unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Dung tích xe:** {selected_bike.get('Dung tích xe', 'Chưa có thông tin')}")
                st.markdown(f"**Năm đăng ký:** {selected_bike.get('Năm đăng ký', 'Chưa có thông tin')}")
                raw_mileage = selected_bike.get('Số Km đã đi')
                if pd.notna(raw_mileage) and str(raw_mileage).strip() and str(raw_mileage).strip().isdigit():
                    formatted_mileage = f"{int(raw_mileage):,}".replace(",", ".")
                else:
                    formatted_mileage = 'Chưa cập nhật'
                st.markdown(format_with_color("Số km đã đi", formatted_mileage, color='orange'), unsafe_allow_html=True)
                price = selected_bike.get('Giá')
                if pd.notna(price) and str(price).strip() and str(price).replace('.', '').isdigit():
                    formatted_price = f"{float(price):.1f} triệu".replace(".", ",")
                else:
                    formatted_price = 'Chưa có thông tin'
                st.markdown(format_with_color("Giá bán", formatted_price, color='red'), unsafe_allow_html=True)
            st.markdown("<h5 style='margin-bottom: 0.5rem;'><span style='font-size: 1em;'>📝</span> Mô tả chi tiết</h5>", unsafe_allow_html=True)
            st.markdown(f"{selected_bike.get('Mô tả chi tiết', 'Không có mô tả chi tiết')}")
    if st.button("🔍 Tìm xe tương tự"):
        if search_type == "Tìm theo mô tả":
            if not noi_dung.strip():
                st.warning("Vui lòng nhập nội dung tìm kiếm")
            else:
                try:
                    df_ket_qua = Tim_xe(df, noi_dung, top_n=so_luong_xe)
                    st.markdown("### 📌 Kết quả tìm kiếm")
                    st.success(f"🎉 Tìm thấy {len(df_ket_qua)} xe phù hợp!")
                    show_list_card(df_ket_qua)
                except Exception as e:
                    st.error(f"Lỗi khi tìm xe: {e}")
        else:
            if 'selected_bike' not in locals() or selected_bike is None:
                st.warning("Vui lòng chọn một xe từ danh sách")
            else:
                try:
                    df_ket_qua = Tim_xe_ID(df, selected_id, top_n=so_luong_xe)
                    if len(df_ket_qua) > 0:
                        st.markdown("### 📌 Kết quả tìm kiếm")
                        st.success(f"🎉 Tìm thấy {len(df_ket_qua)} xe tương tự!")
                        show_list_card(df_ket_qua)
                    else:
                        st.warning("Không tìm thấy xe tương tự.")
                except Exception as e:
                    st.error(f"Lỗi khi tìm xe tương tự: {e}")

# ========================= ĐỊNH GIÁ – FORM INPUT ============================
elif nav == "Phân cụm và Dự đoán giá xe":
    
    st.title("💲 Phân cụm và Dự đoán giá xe")
    st.markdown("---")

    with st.container():
        # Your existing tab content here
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-panel"] {
            padding: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Khám phá nhóm xe", "Gợi ý giá xe"])
    with tab1:
        #st.set_page_config(page_title="So sánh 3 nhóm xe", layout="wide")

        # ==== STYLE CSS ====
        st.markdown("""
        <style>
        .card {
            background: #ffffff;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .card h3 {
            margin-top: 0;
        }
        .table-container {
            border-radius: 12px;
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table thead {
            background: #f0f2f6;
            font-weight: bold;
        }
        table td, table th {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)


        # ==== TIÊU ĐỀ ====
        st.subheader("⭐ So sánh 3 nhóm xe máy – Lựa chọn phù hợp nhất cho bạn")

        st.write("""
        Bảng so sánh giúp bạn nắm nhanh sự khác biệt giữa xe tay ga cao cấp, tay ga phổ thông 
        và xe số bền bỉ – để dễ dàng chọn đúng dòng xe phù hợp.
        """)

        # ==== BẢNG SO SÁNH ====
        table_html = """
        <div class="table-container">
        <table>
            <thead style="text-align:center;">
                <tr>
                    <th>Danh mục</th>
                    <th>Tay ga cao cấp 💎</th>
                    <th>Tay ga phổ thông 🌟</th>
                    <th>Xe số bền bỉ 🔧</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Độ sang trọng</td>
                    <td>⭐⭐⭐⭐⭐</td>
                    <td>⭐⭐⭐</td>
                    <td>⭐⭐</td>
                </tr>
                <tr>
                    <td>Tiết kiệm nhiên liệu</td>
                    <td>⭐⭐⭐</td>
                    <td>⭐⭐⭐⭐</td>
                    <td>⭐⭐⭐⭐⭐</td>
                </tr>
                <tr>
                    <td>Giá thành</td>
                    <td>💰💰💰</td>
                    <td>💰💰</td>
                    <td>💰</td>
                </tr>
                <tr>
                    <td>Độ bền</td>
                    <td>⭐⭐⭐⭐</td>
                    <td>⭐⭐⭐</td>
                    <td>⭐⭐⭐⭐⭐</td>
                </tr>
                <tr>
                    <td>Phù hợp với ai?</td>
                    <td>Người thích sự đẳng cấp</td>
                    <td>HS-SV, nhân viên, gia đình</td>
                    <td>Người chạy nhiều, tiết kiệm tối đa</td>
                </tr>
            </tbody>
        </table>
        </div>
        """

        st.markdown(table_html, unsafe_allow_html=True)


        # ==== CARD THÔNG TIN ====
        st.subheader("🧩 Chi tiết từng nhóm xe")

        col1, col2, col3 = st.columns(3)

        # === Card 1 ===
        with col1:
            st.markdown("""
            <div class="card">
                <h3>💎 Nhóm 1 – Xe tay ga cao cấp</h3>
                <ul>
                    <li>✨ Thiết kế sang trọng, hiện đại</li>
                    <li>🚀 Động cơ mạnh 100–175cc</li>
                    <li>💠 Hoàn thiện cao cấp</li>
                    <li>👑 Tôn lên phong cách & đẳng cấp</li>
                </ul>
                <b>Phù hợp với:</b> Người muốn xe bền – mạnh – nổi bật.
            </div>
            """, unsafe_allow_html=True)

        # === Card 2 ===
        with col2:
            st.markdown("""
            <div class="card">
                <h3>🌟 Nhóm 2 – Xe tay ga phổ thông</h3>
                <ul>
                    <li>💸 Giá hợp túi tiền</li>
                    <li>⛽ Siêu tiết kiệm nhiên liệu</li>
                    <li>🎨 Thiết kế trẻ trung</li>
                    <li>👍 Dễ chạy – dễ bảo dưỡng</li>
                </ul>
                <b>Phù hợp với:</b> HS-SV, nhân viên văn phòng, gia đình.
            </div>
            """, unsafe_allow_html=True)

        # === Card 3 ===
        with col3:
            st.markdown("""
            <div class="card">
                <h3>🔧 Nhóm 3 – Xe số bền bỉ</h3>
                <ul>
                    <li>💰 Giá rất rẻ</li>
                    <li>⛽ Cực tiết kiệm xăng</li>
                    <li>🛣️ Đi đường dài ổn định</li>
                    <li>🧰 Ít hỏng vặt – dễ sửa</li>
                </ul>
                <b>Phù hợp với:</b> Người chạy nhiều, cần xe bền & tiết kiệm.
            </div>
            """, unsafe_allow_html=True)


        st.subheader("📖 Tham khảo danh sách xe theo nhóm")
        if clustered_data is not None and 'cluster_id' in clustered_data.columns:
            # Get unique cluster IDs and sort them
            unique_clusters = sorted(clustered_data["cluster_id"].unique())
            # Create display names (add 1 to each cluster ID for display)
            cluster_options = [f'Nhóm {i+1}' for i in unique_clusters]
            
            # Show selectbox with display names
            selected_display = st.selectbox(
                "Chọn nhóm để xem chi tiết",
                options=cluster_options,
                format_func=lambda x: x
            )
            
            # Get the actual cluster ID (subtract 1 from the selected display index)
            selected_index = cluster_options.index(selected_display)
            selected_cluster = unique_clusters[selected_index]
            
            # Filter and show data
            filtered = clustered_data[clustered_data["cluster_id"] == selected_cluster]
            st.write(f"Số lượng xe trong {selected_display}: {len(filtered)}")
            
            # Select specific columns to display
            columns_to_show = [
                'Tiêu_đề', 'Giá',
                'Dòng_xe', 'Thương_hieu', 'Mô_tả_chi_tiết',
                'Dung_tích_xe', 'Năm_đăng_ký', 'Số_Km_đã_đi'
            ]
            
            # Only show columns that exist in the dataframe
            available_columns = [col for col in columns_to_show if col in filtered.columns]
            
            # Display the filtered dataframe with selected columns
            if available_columns:
                # Add search functionality
                search_term = st.text_input("🔍 Tìm kiếm trong danh sách xe:", "")
                
                # Apply search filter if search term is not empty
                if search_term:
                    search_columns = [col for col in available_columns if filtered[col].dtype == 'object']  # Only search in text columns
                    if search_columns:
                        mask = filtered[search_columns].apply(
                            lambda x: x.astype(str).str.contains(search_term, case=False, na=False)
                        ).any(axis=1)
                        filtered = filtered[mask]
                        st.info(f"Tìm thấy {len(filtered)} kết quả phù hợp với từ khóa: '{search_term}'")
                
                # Display the dataframe with pagination
                st.dataframe(
                    filtered[available_columns],
                    use_container_width=True,
                    height=400
                )
                
                # Show total number of records
                st.caption(f"Tổng số xe: {len(filtered)}")
                
            else:
                st.warning("Không tìm thấy cột nào để hiển thị. Vui lòng kiểm tra tên cột.")
                st.dataframe(filtered.head())  # Show first few rows with all columns as fallback
            
        else:
            st.error("Không thể tải dữ liệu nhóm xe. Vui lòng kiểm tra file dữ liệu.")
            if clustered_data is None:
                st.error("Lỗi: Không tải được dữ liệu nhóm xe (clustered_data is None)")
            else:
                st.error(f"Lỗi: Cột 'cluster_id' không tồn tại trong dữ liệu. Các cột có sẵn: {', '.join(clustered_data.columns)}")

    with tab2:
        st.subheader("Nhập các thông tin xe để hệ thống gợi ý giá")

        # init safe flags
        st.session_state.setdefault("posted", False)    
        st.session_state.setdefault("posting_in_progress", False)
        st.session_state.setdefault("last_price_info", None)
        st.session_state.setdefault("last_input_kwargs", None)

        # ====================== FORM INPUT DỮ LIỆU ======================
        with st.form("form_goi_y", clear_on_submit=False):
            st.markdown("#### 💰 Giá mong muốn")
            col1, col2, col3 = st.columns(3)
            with col1:
                gia_mong_muon = st.number_input("Giá mong muốn (triệu VND)", min_value=0.0, value=30.0, step=0.5, key="giamong")
            with col2:
                gia_min = st.number_input("Giá tối thiểu", min_value=0.0, value=10.0, step=0.5, key="giamin")
            with col3:
                gia_max = st.number_input("Giá tối đa", min_value=0.0, value=50.0, step=0.5, key="giamax")

            st.markdown("#### 🏍️ Thông tin xe")
            def options(col): return sorted(df[col].dropna().unique()) if col in df else []

            col1, col2 = st.columns(2)
            with col1:
                thuong_hieu = st.selectbox("Thương hiệu", options("Thương hiệu"), key="in_thuong_hieu")
                loai_xe = st.selectbox("Loại xe", options("Loại xe"), key="in_loai_xe")
                xuat_xu = st.selectbox("Xuất xứ", options("Xuất xứ"), key="in_xuat_xu")
                dong_xe = st.selectbox("Dòng xe", options("Dòng xe"), key="in_dong_xe")
            with col2:
                cc_numeric = st.number_input("Dung tích xe (cc_numeric)", 
                                    min_value=0, 
                                    step=1, 
                                    value=137, 
                                    key="inp_cc")
                tuoi_xe = st.number_input("Tuổi xe", min_value=0, max_value=50, value=5, key="in_tuoi")
                so_km = st.number_input("Số Km đã đi", min_value=0, value=100000, key="in_km")
                phan_khuc = st.selectbox("Phân khúc giá", options("Phân khúc giá"), key="in_phan_khuc")

            # Tình trạng xe
            st.markdown("#### ⚙️ Tình trạng xe")
            column_mapping = {
                "is_moi": "Còn mới",
                "is_do_xe": "Có độ xe",
                "is_su_dung_nhieu": "Sử dụng nhiều",
                "is_bao_duong": "Bảo dưỡng định kỳ",
                "is_do_ben": "Độ bền tốt",
                "is_phap_ly": "Giấy tờ đầy đủ"
            }

            is_cols = [c for c in df.columns if c.startswith("is_")]
            tinh_trang = {}
            col1, col2 = st.columns(2)
            half = (len(is_cols) + 1) // 2
            with col1:
                for col in is_cols[:half]:
                    tinh_trang[col] = st.checkbox(column_mapping.get(col, col), key=f"cb_{col}")
            with col2:
                for col in is_cols[half:]:
                    tinh_trang[col] = st.checkbox(column_mapping.get(col, col), key=f"cb_{col}")

            submitted = st.form_submit_button("⏳ Định giá xe")

        # ====================== SAU KHI SUBMIT FORM ======================
        if submitted:
            price_segment_map = {'Phổ Thông': 1, 'Cận Cao Cấp': 2, 'Cao Cấp': 3}
            input_kwargs = {
                "Thương_hiệu": str(thuong_hieu),
                "Xuất_xứ": str(xuat_xu),
                "Dòng_xe": str(dong_xe),
                "Loại_xe": str(loai_xe),
                "cc_numeric": float(cc_numeric) if cc_numeric is not None else 0.0,
                "Dung_tích_xe": float(cc_numeric) if cc_numeric is not None else 0.0,
                "price_segment_code": price_segment_map.get(phan_khuc, 1),
                "age": int(tuoi_xe),
                "Số_Km_đã_đi": int(so_km),
            }
            for k, v in tinh_trang.items():
                input_kwargs[k] = int(v)

            try:
                input_df = prepare_input_dataframe(pipeline, **input_kwargs)
                cluster_id, price_info = predict_cluster_and_price(pipeline, cluster_summary, input_df)

                st.success(f"Xe thuộc cụm: {cluster_id}")
                st.write(
                    f"Số xe trong cụm: {price_info['count']:,} | "
                    f"Giá TB: {price_info['mean']:.1f} triệu | "
                    f"Khoảng: {price_info['min']:.1f} - {price_info['max']:.1f} triệu"
                )

                # -------------------------------
                # Reset toàn bộ trạng thái ĐĂNG TIN
                # -------------------------------
                st.session_state["posted"] = False
                st.session_state["posting_in_progress"] = False
                st.session_state["force_reset_post_form"] = True

                # khi có dự đoán mới → set lại giá trị mới
                st.session_state["last_price_info"] = price_info
                st.session_state["last_input_kwargs"] = input_kwargs

            except Exception as e:
                st.error(f"Lỗi định giá: {e}")
                import traceback
                st.text(traceback.format_exc())
                st.stop()

        # ==========================
        # FORM ĐĂNG TIN
        st.markdown("---")
        st.subheader("📣 Đăng tin bán / mua xe")

        # ---------------------------
        # FIX
        # ---------------------------
        if st.session_state.get("force_reset_post_form", False):
            st.session_state["posted"] = False
            st.session_state["posting_in_progress"] = False
            st.session_state["force_reset_post_form"] = False


        # -------------------------------------------------------------------
        # 1. Nếu đã đăng rồi thì không hiện form
        # -------------------------------------------------------------------
        if st.session_state.get("posted", False):
            st.success("🎉 Bạn đã gửi tin — chờ QTV duyệt.")
            st.stop()

        # -------------------------------------------------------------------
        # 2. Nếu CHƯA có dự đoán → yêu cầu chạy định giá
        # -------------------------------------------------------------------
        price_info = st.session_state.get("last_price_info")
        if price_info is None:
            st.info("Hãy nhấn '⏳ Định giá xe' trước khi đăng tin để hệ thống gợi ý giá.")
            st.stop()


        # -------------------------------------------------------------------
        # 3. Có dự đoán → Hiện form đăng tin
        # -------------------------------------------------------------------
        gia_du_doan = float(price_info["mean"])

        with st.form("form_dang_tin_v2", clear_on_submit=False):

            chon_gia = st.radio(
                "Chọn giá đăng:",
                ("Giữ giá đã nhập", "Dùng giá model dự đoán"),
                key="ft_chon_gia"
            )
            gia_dang = gia_mong_muon if chon_gia == "Giữ giá đã nhập" else gia_du_doan

            st.success(f"📌 Giá đăng: **{gia_dang:.1f} triệu**")
            st.success(f"📌 Giá dự đoán: **{gia_du_doan:.1f} triệu**")
            
            user_id = st.text_input("ID người đăng", key="ft_user_id")
            user_note = st.text_input("Ghi chú thêm", key="ft_user_note")
            loai_dang = st.radio("Hình thức đăng", ("Đăng bán", "Đăng mua"), key="ft_loai_dang")

            gui_tin = st.form_submit_button("✅ Gửi tin lên hệ thống")


        # -------------------------------------------------------------------
        # 4. Xử lý gửi tin
        # -------------------------------------------------------------------
        if gui_tin:

            # tránh double-submit
            if st.session_state.get("posting_in_progress"):
                st.warning("Đang xử lý gửi tin, vui lòng đợi...")
                st.stop()

            st.session_state["posting_in_progress"] = True

            # map phân khúc
            price_segment_map = {
                'Phổ Thông': 1,
                'Cận Cao Cấp': 2,
                'Cao Cấp': 3
            }

            # tạo record
            record = {
                "post_id": str(uuid.uuid4()),
                "user_id": user_id if user_id else "anonymous",
                "note": user_note,
                "post_type": "sell" if loai_dang == "Đăng bán" else "buy",
                "price_final": float(gia_dang),

                # ------- THÔNG TIN XE -------
                "brand": thuong_hieu,
                "model": dong_xe,
                "vehicle_type": loai_xe,
                "origin": xuat_xu,
                "cc_numeric": float(cc_numeric) if cc_numeric is not None else 0.0,
                "age": int(tuoi_xe),
                "year_reg": int(max(1900, 2025 - int(tuoi_xe))),
                "km_driven": int(so_km),
                "price_min": float(gia_min),
                "price_max": float(gia_max),
                "price_segment_code": price_segment_map.get(phan_khuc, 1),
                "predicted_price": float(price_info["mean"]) if price_info is not None else np.nan,

                # trạng thái
                "status": "pending",
            }
            # -------------------------------------------------------------------
            # 5. Lưu record + reset trạng thái
            # -------------------------------------------------------------------
            try:
                save_post_record(record)

                st.session_state["posted"] = True
                st.session_state["posting_in_progress"] = False

                # XÓA prediction → lần sau phải chạy lại model, KHÔNG dùng kết quả cũ
                st.session_state.pop("last_price_info", None)
                st.session_state.pop("last_input_kwargs", None)

                st.success("🎉 Tin đã được gửi và chờ QTV duyệt!")

                st.stop()

            except Exception as e:
                st.session_state["posting_in_progress"] = False
                st.error(f"❌ Lỗi khi lưu tin: {e}")
                import traceback
                st.text(traceback.format_exc())


# ========================= Tin đăng bán ============================
elif nav == "Tin đăng bán":
    st.header("📢 Tin đăng bán (Người dùng)")
    try:
        posts = _read_xlsx_if_exists(APPROVED_SELL_XLSX)
        posts = normalize_datetime_like_columns(posts)
    except Exception:
        posts = pd.DataFrame()
    if posts.empty:
        st.info("Hiện chưa có tin đăng bán.")
    else:
        st.write(f"Tổng: {len(posts)} tin")
        show_cols = [
            "user_id", "note", "price_final", "year_reg",
            "km_driven", "brand", "model", "cc_numeric",
            "origin", "vehicle_type"
        ]
        posts_show = posts.copy()
        posts_show = posts_show[[c for c in show_cols if c in posts_show.columns]]
        try:
            posts_show = rename_columns_vn(posts_show, mode="sell")
        except Exception:
            pass
        st.dataframe(posts_show.reset_index(drop=True), use_container_width=True)
        try:
            st.download_button("⬇️ Tải tin đăng bán (Excel)", df_to_excel_bytes(posts), file_name="posts_sell.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass

# ========================= Tin đăng mua ============================
elif nav == "Tin đăng mua":
    st.header("📣 Tin đăng mua (Người dùng)")
    try:
        posts = _read_xlsx_if_exists(APPROVED_BUY_XLSX)
        posts = normalize_datetime_like_columns(posts)
    except Exception:
        posts = pd.DataFrame()
    if posts.empty:
        st.info("Hiện chưa có tin đăng mua.")
    else:
        st.write(f"Tổng: {len(posts)} tin")
        show_cols = [
            "user_id", "note", "price_final", "year_reg",
            "km_driven", "brand", "model", "cc_numeric",
            "origin", "vehicle_type"
        ]
        posts_show = posts.copy()
        posts_show = posts_show[[c for c in show_cols if c in posts_show.columns]]
        try:
            posts_show = rename_columns_vn(posts_show, mode="buy")
        except Exception:
            pass
        st.dataframe(posts_show.reset_index(drop=True), use_container_width=True)
        try:
            st.download_button("⬇️ Tải tin đăng mua (Excel)", df_to_excel_bytes(posts), file_name="posts_buy.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass

# ========================= Duyệt tin (QTV) ============================
elif nav == "Duyệt tin (QTV)":
    # require qtv_login from helpers
    try:
        if "qtv_logged_in" not in st.session_state or st.session_state.get("qtv_logged_in") is False:
            qtv_login()
            st.stop()
    except Exception:
        # If qtv_login not available, show admin warning but continue (for local dev)
        st.warning("Lỗi khi đăng nhập QTV. Vui lòng kiểm tra lại.")
    st.header("🔧 Duyệt tin — Quản trị viên")
    pending = len(st.session_state.get("pending_notifications", []))
    st.markdown(f"**Tin chờ duyệt:** {pending}")
    manage_sell = st.checkbox("Quản lý tin đăng bán", value=True)
    manage_buy = st.checkbox("Quản lý tin đăng mua", value=False)

    if manage_sell:
        st.subheader("📦 Tin đăng bán (chờ duyệt)")
        df_sell = st.session_state.get("posts_sell", pd.DataFrame()).copy()
        if df_sell.empty:
            st.info("Không có tin đăng bán nào.")
        else:
            df_sell_display = df_sell.copy()
            df_sell_display["selected"] = False
            try:
                df_sell_display = reorder_columns(df_sell_display)
                df_sell_display = rename_columns_vn(df_sell_display, mode="sell")
            except Exception:
                pass
            edited_sell = st.data_editor(
                df_sell_display,
                use_container_width=True,
                hide_index=True,
                key="editor_sell"
            )
            selected_sell = edited_sell[edited_sell["Chọn"] == True] if "Chọn" in edited_sell.columns else pd.DataFrame()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✔️ Duyệt tin bán"):
                    if selected_sell.empty:
                        st.warning("Chưa chọn dòng để duyệt.")
                    else:
                        post_ids = selected_sell["post_id"].tolist() if "post_id" in selected_sell.columns else []
                        try:
                            approved = _read_xlsx_if_exists(APPROVED_SELL_XLSX)
                            approved = pd.concat(
                                [approved, df_sell[df_sell["post_id"].isin(post_ids)]],
                                ignore_index=True
                            )
                            _save_xlsx(approved, APPROVED_SELL_XLSX)
                            df_sell_new = df_sell[~df_sell["post_id"].isin(post_ids)]
                            st.session_state["posts_sell"] = df_sell_new
                            _save_xlsx(df_sell_new, POSTS_SELL_XLSX)
                            for pid in post_ids:
                                if pid in st.session_state["pending_notifications"]:
                                    st.session_state["pending_notifications"].remove(pid)
                            st.success(f"Đã duyệt {len(post_ids)} tin bán.")
                        except Exception as e:
                            st.error(f"Lỗi khi duyệt: {e}")
            with col2:
                if st.button("❌ Từ chối tin bán"):
                    if selected_sell.empty:
                        st.warning("Chưa chọn dòng để từ chối.")
                    else:
                        post_ids = selected_sell["post_id"].tolist() if "post_id" in selected_sell.columns else []
                        try:
                            rejected = _read_xlsx_if_exists(REJECTED_XLSX)
                            rejected = pd.concat(
                                [rejected, df_sell[df_sell["post_id"].isin(post_ids)]],
                                ignore_index=True
                            )
                            _save_xlsx(rejected, REJECTED_XLSX)
                            df_sell_new = df_sell[~df_sell["post_id"].isin(post_ids)]
                            st.session_state["posts_sell"] = df_sell_new
                            _save_xlsx(df_sell_new, POSTS_SELL_XLSX)
                            for pid in post_ids:
                                if pid in st.session_state["pending_notifications"]:
                                    st.session_state["pending_notifications"].remove(pid)
                            st.success(f"Đã từ chối {len(post_ids)} tin bán.")
                        except Exception as e:
                            st.error(f"Lỗi khi từ chối: {e}")

    st.markdown("---")

    if manage_buy:
        st.subheader("🛒 Tin đăng mua (chờ duyệt)")
        df_buy = st.session_state.get("posts_buy", pd.DataFrame()).copy()
        if df_buy.empty:
            st.info("Không có tin đăng mua nào.")
        else:
            df_buy_display = df_buy.copy()
            df_buy_display["selected"] = False
            try:
                df_buy_display = reorder_columns(df_buy_display)
                df_buy_display = rename_columns_vn(df_buy_display, mode="buy")
            except Exception:
                pass
            edited_buy = st.data_editor(
                df_buy_display,
                use_container_width=True,
                hide_index=True,
                key="editor_buy"
            )
            selected_buy = edited_buy[edited_buy["Chọn"] == True] if "Chọn" in edited_buy.columns else pd.DataFrame()
            col3, col4 = st.columns(2)
            with col3:
                if st.button("✔️ Duyệt tin mua"):
                    if selected_buy.empty:
                        st.warning("Chưa chọn dòng để duyệt.")
                    else:
                        post_ids = selected_buy["post_id"].tolist() if "post_id" in selected_buy.columns else []
                        try:
                            approved = _read_xlsx_if_exists(APPROVED_BUY_XLSX)
                            approved = pd.concat(
                                [approved, df_buy[df_buy["post_id"].isin(post_ids)]],
                                ignore_index=True
                            )
                            _save_xlsx(approved, APPROVED_BUY_XLSX)
                            df_buy_new = df_buy[~df_buy["post_id"].isin(post_ids)]
                            st.session_state["posts_buy"] = df_buy_new
                            _save_xlsx(df_buy_new, POSTS_BUY_XLSX)
                            for pid in post_ids:
                                if pid in st.session_state["pending_notifications"]:
                                    st.session_state["pending_notifications"].remove(pid)
                            st.success(f"Đã duyệt {len(post_ids)} tin mua.")
                        except Exception as e:
                            st.error(f"Lỗi khi duyệt: {e}")
            with col4:
                if st.button("❌ Từ chối tin mua"):
                    if selected_buy.empty:
                        st.warning("Chưa chọn dòng để từ chối.")
                    else:
                        post_ids = selected_buy["post_id"].tolist() if "post_id" in selected_buy.columns else []
                        try:
                            rejected = _read_xlsx_if_exists(REJECTED_XLSX)
                            rejected = pd.concat(
                                [rejected, df_buy[df_buy["post_id"].isin(post_ids)]],
                                ignore_index=True
                            )
                            _save_xlsx(rejected, REJECTED_XLSX)
                            df_buy_new = df_buy[~df_buy["post_id"].isin(post_ids)]
                            st.session_state["posts_buy"] = df_buy_new
                            _save_xlsx(df_buy_new, POSTS_BUY_XLSX)
                            for pid in post_ids:
                                if pid in st.session_state["pending_notifications"]:
                                    st.session_state["pending_notifications"].remove(pid)
                            st.success(f"Đã từ chối {len(post_ids)} tin mua.")
                        except Exception as e:
                            st.error(f"Lỗi khi từ chối: {e}")

# ========================= AUTHOR PAGE ============================
elif nav == "Thông tin tác giả":
    st.header("👤 Nhóm tác giả dự án")
    st.write("""
    **Hồ Thị Quỳnh Như**  
    **Nguyễn Văn Cường**  
    **Nguyễn Thị Tuyết Anh**  
    """)
