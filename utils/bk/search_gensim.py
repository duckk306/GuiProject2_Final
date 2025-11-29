import streamlit as st
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities

from . import utils_clean_text as uct

# ============================================
# HÀM KHỞI TẠO TF-IDF MÔ TẢ XE
# Gọi hàm này 1 lần duy nhất rồi lưu vào session_state
# ============================================
def build_tfidf_model(df, text_col="mo_ta_chi_tiet"):
    # Tokenize mô tả chi tiết
    content_gem = [[text for text in x.split()] for x in df[text_col]]

    # Tạo từ điển từ mô tả chi tiết
    dictionary = corpora.Dictionary(content_gem)

    # Tạo bag-of-words corpus
    corpus = [dictionary.doc2bow(doc) for doc in content_gem]

    # Xây dựng mô hình TF-IDF
    tfidf_model = models.TfidfModel(corpus)

    # Tạo ma trận tương đồng
    index = similarities.SparseMatrixSimilarity(tfidf_model[corpus], num_features=len(dictionary))

    return dictionary, tfidf_model, index

# ============================================
# HÀM TÌM XE THEO NỘI DUNG USER NHẬP
# ============================================
def Tim_xe(df, user_text, top_n=10, text_col="clean_text"):
    if user_text.strip() == "":
        return df.copy()  # trả bản sao của df nếu không nhập gì

    # Tạo mô hình nếu chưa có trong session_state
    if "tfidf_dict" not in st.session_state:
        st.session_state["tfidf_dict"], \
        st.session_state["tfidf_model"], \
        st.session_state["tfidf_index"] = build_tfidf_model(df, text_col)

    dictionary = st.session_state["tfidf_dict"]
    tfidf_model = st.session_state["tfidf_model"]
    index = st.session_state["tfidf_index"]

    # Tiền xử lý text user nhập
    query = uct.clean_text(user_text).split()

    # Chuyển thành vector TF-IDF
    bow = dictionary.doc2bow(query)
    tfidf_vec = tfidf_model[bow]

    # Tính similarities
    sims = index[tfidf_vec]  # array gồm similarity score

    # Lấy top N xe
    top_idx = np.argsort(sims)[::-1][:top_n]  # sort giảm dần

    # Trả về dataframe
    result = df.iloc[top_idx].copy()
    result["similarity"] = sims[top_idx]

    # Sắp xếp theo similarity giảm dần
    result = result.sort_values("similarity", ascending=False)

    return result

def Tim_xe_ID(df, bike_id, top_n=10, text_col="clean_text"):
    """
    Tìm xe tương tự dựa trên ID xe sử dụng ma trận tương đồng đã tính toán
    
    Args:
        df: DataFrame chứa dữ liệu xe
        bike_id: ID của xe cần tìm xe tương tự
        top_n: Số lượng xe tương tự cần tìm
        text_col: Tên cột chứa nội dung mô tả
        
    Returns:
        DataFrame chứa các xe tương tự (không bao gồm xe gốc)
    """
    # Kiểm tra xem ID có tồn tại không
    if bike_id not in df.index:
        raise ValueError(f"Không tìm thấy xe với ID: {bike_id}")
    
    # Đảm bảo model đã được train
    if "tfidf_index" not in st.session_state:
        st.session_state["tfidf_dict"], \
        st.session_state["tfidf_model"], \
        st.session_state["tfidf_index"] = build_tfidf_model(df, text_col)
    
    # Lấy index của xe trong ma trận tương đồng
    bike_idx = df.index.get_loc(bike_id)
    
    # Lấy ma trận tương đồng
    index = st.session_state["tfidf_index"]
    
    # Lấy độ tương đồng của xe hiện tại với tất cả các xe khác
    bike_text = str(df[text_col].iloc[bike_idx])
    vec_bow = st.session_state["tfidf_dict"].doc2bow(bike_text.split())
    vec_tfidf = st.session_state["tfidf_model"][vec_bow]
    sims = index[vec_tfidf]
    
    # Lấy top N xe tương tự (bỏ qua chính nó)
    top_indices = np.argsort(-sims)[1:top_n+1]  # bỏ qua chính nó
    
    # Lấy kết quả
    result = df.iloc[top_indices].copy()
    result["similarity"] = sims[top_indices]
    
    return result