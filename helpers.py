import os
import uuid
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from config import MODEL_PATH, QTV_ACCOUNTS, POSTS_SELL_XLSX, POSTS_BUY_XLSX, num_cols, flag_cols, cat_cols

@st.cache_resource
def load_pipeline(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Không load được model từ `{path}`: {e}")
        return None

def qtv_login():
    st.subheader("🔐 Đăng nhập quản trị viên (QTV)")

    user = st.text_input("ID QTV:", key="qtv_user")
    pw = st.text_input("Mật khẩu:", type="password", key="qtv_pw")

    if st.button("Đăng nhập", key="qtv_login_btn"):
        if user in QTV_ACCOUNTS and pw == QTV_ACCOUNTS[user]:
            st.session_state["qtv_logged_in"] = True
            st.success("Đăng nhập thành công!")
            st.rerun()
        else:
            st.error("Sai ID hoặc mật khẩu!")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    try:
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="posts")
        return bio.getvalue()
    except Exception:
        return df_to_csv_bytes(df)


def _read_xlsx_if_exists(path):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_xlsx(df, path):
    try:
        # Ensure directory exists
        dirp = os.path.dirname(path)
        if dirp and not os.path.exists(dirp):
            os.makedirs(dirp, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu file {path}: {e}")
        return False


def safe_prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    for c in num_cols + flag_cols + cat_cols:
        if c not in dfc.columns:
            if c in flag_cols:
                dfc[c] = 0
            elif c in num_cols:
                dfc[c] = 0.0
            else:
                dfc[c] = ""
    for n in ["km_driven", "cc_numeric", "age", "price_segment_code", "year_reg", "price_min", "price_max"]:
        if n in dfc.columns:
            dfc[n] = pd.to_numeric(dfc[n], errors="coerce").fillna(0.0)
    for f in flag_cols:
        if f in dfc.columns:
            dfc[f] = dfc[f].apply(lambda x: 1 if (str(x).lower() in ["1","true","yes","có","co"]) or x==1 or x is True else 0).astype(int)
    return dfc


def compute_risk_score_strict(row, last_clean_brand_models=None, anomaly_reason=None):
    score = 0.0
    try:
        price = float(row.get("price", 0.0))
        pred = float(row.get("predicted_price", 0.0))
        if pred > 0:
            diff_pct = abs(price - pred) / pred
            score += min(50.0, diff_pct * 100.0 * 0.5)
    except Exception:
        pass
    if anomaly_reason and isinstance(anomaly_reason, str) and anomaly_reason != "Không có dấu hiệu bất thường":
        score += 25.0
    try:
        km = float(row.get("km_driven", 0.0))
        age = float(row.get("age", 0.0))
        if age >= 5 and km < 2000:
            score += 20.0
        elif age >= 8 and km < 5000:
            score += 30.0
    except Exception:
        pass
    try:
        if int(row.get("is_moi", 0)) == 1 and float(row.get("age", 0.0)) > 3:
            score += 10.0
    except Exception:
        pass
    if last_clean_brand_models and isinstance(last_clean_brand_models, dict):
        brand = str(row.get("brand", "")).strip()
        model = str(row.get("model", "")).strip()
        if brand in last_clean_brand_models:
            known_models = last_clean_brand_models.get(brand, [])
            if model and (model not in known_models):
                score += 30.0
    score = min(100.0, score)
    return round(score, 2)


def risk_level_from_score(score):
    if score >= 70:
        return "Nguy hiểm"
    elif score >= 40:
        return "Đáng chú ý"
    else:
        return "An toàn"


def make_post_record(df_row: pd.DataFrame, post_type: str, chosen_price: float, user_id: str = "anonymous", note: str = ""):
    rec = df_row.iloc[0].to_dict()
    rec["post_id"] = str(uuid.uuid4())[:8]
    rec["post_time"] = pd.Timestamp.now()
    rec["post_type"] = post_type
    rec["price_input"] = rec.get("price", np.nan)
    rec["price_pred"] = rec.get("predicted_price", np.nan)
    rec["price_final"] = chosen_price
    rec["status"] = "pending"
    rec["user_id"] = user_id
    rec["note"] = note
    rec["anomaly_reason"] = rec.get("anomaly_reason", "")
    rec["risk_score"] = rec.get("risk_score", np.nan)
    return rec


def ensure_post_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out is None or out.empty:
        return out
    if "post_id" not in out.columns:
        out["post_id"] = [str(uuid.uuid4())[:8] for _ in range(len(out))]
    # ensure dtype str
    out["post_id"] = out["post_id"].astype(str)
    return out


def normalize_datetime_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out is None or out.empty:
        return out
    for col in out.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype(str)
        except Exception:
            # ignore problematic columns
            pass
    return out


def prepare_for_aggrid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out is None or out.empty:
        return out
    # convert datetime-like columns to string
    out = normalize_datetime_like_columns(out)
    # convert object columns to str (except post_id)
    for col in out.columns:
        if col != "post_id" and out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out


def save_post_record(record: dict):
    df_new = pd.DataFrame([record])
    if record.get("post_type") == "sell":
        posts = _read_xlsx_if_exists(POSTS_SELL_XLSX)
        posts = pd.concat([posts, df_new], ignore_index=True)
        posts = ensure_post_id(posts)
        _save_xlsx(posts, POSTS_SELL_XLSX)
        st.session_state["posts_sell"] = posts.copy()
    else:
        posts = _read_xlsx_if_exists(POSTS_BUY_XLSX)
        posts = pd.concat([posts, df_new], ignore_index=True)
        posts = ensure_post_id(posts)
        _save_xlsx(posts, POSTS_BUY_XLSX)
        st.session_state["posts_buy"] = posts.copy()
    st.session_state.setdefault("pending_notifications", [])
    st.session_state["pending_notifications"].append(record.get("post_id"))


def rename_columns_vn(df: pd.DataFrame, mode="general"):
    """
    mode = "sell"  -> Giá bán
    mode = "buy"   -> Giá mua
    mode = "general" -> Giá bán / Giá mua (dùng cho QTV)
    """

    if mode == "sell":
        price_name = "Giá bán (triệu đồng)"
    elif mode == "buy":
        price_name = "Giá mua (triệu đồng)"
    else:
        price_name = "Giá bán / Giá mua (triệu đồng)"
    col_map = {
        "selected": "Chọn",
        "user_id": "ID người dùng",
        "note": "Mô tả",
        "price_final": price_name,
        "year_reg": "Năm đăng ký",
        "km_driven": "Km đã đi",
        "brand": "Hãng xe",
        "model": "Dòng xe",
        "cc_numeric": "Dung tích xe (cc)",
        "origin": "Xuất xứ",
        "vehicle_type": "Loại xe",
    }
    df = df.rename(columns=col_map)
    return df

def reorder_columns(df: pd.DataFrame):
    front_cols = ["selected", "user_id", "note"]
    other_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + other_cols]