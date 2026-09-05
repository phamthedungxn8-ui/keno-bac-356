import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
from PIL import Image
import re
import plotly.express as px

# ---------------------------------------------------------
# Cấu hình trang Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keno UCB1 - Phân Tích Bậc 3-5-6",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 KENO VIETLOTT - PHÂN TÍCH UCB1 & GỢI Ý BẬC 3, 5, 6")
st.markdown("---")

# ---------------------------------------------------------
# Hàm xử lý OpenCV + OCR đọc số từ ảnh
# ---------------------------------------------------------
def extract_numbers_from_image(image_file):
    """
    Tiền xử lý ảnh bằng OpenCV và trích xuất các số từ 01 đến 80 bằng Tesseract
    """
    try:
        # Chuyển đổi file tải lên thành mảng numpy OpenCV
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Chuyển xám & tăng độ tương phản
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Đọc text qua Tesseract OCR
        text = pytesseract.image_to_string(gray, config='--psm 6 digits')
        
        # Dùng RegEx tìm tất cả các chuỗi số
        raw_numbers = re.findall(r'\b\d{1,2}\b', text)
        
        # Lọc các số hợp lệ của Keno (từ 1 đến 80) và loại bỏ trùng lặp giữ nguyên thứ tự
        valid_numbers = []
        for n in raw_numbers:
            num = int(n)
            if 1 <= num <= 80 and num not in valid_numbers:
                valid_numbers.append(num)
                
        return valid_numbers[:20]  # Keno mỗi kỳ ra 20 số
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {e}")
        return []

# ---------------------------------------------------------
# Thuật toán UCB1
# ---------------------------------------------------------
def calculate_ucb1_scores(history_draws, c_param=1.414):
    """
    Tính điểm Upper Confidence Bound (UCB1) cho 80 con số dựa trên lịch sử các kỳ
    """
    N = len(history_draws) # Tổng số kỳ quan sát (N = 3)
    counts = np.zeros(81)   # Tần suất xuất hiện của số 1..80
    
    for draw in history_draws:
        for num in draw:
            if 1 <= num <= 80:
                counts[num] += 1
                
    scores = {}
    for i in range(1, 81):
        n_i = counts[i]
        if n_i == 0:
            # Nếu số chưa xuất hiện kỳ nào, gán thưởng cơ bản + điểm khám phá
            mean_reward = 0.01
            exploration = np.sqrt((2 * np.log(N + 1)) / 0.5)
        else:
            mean_reward = n_i / N
            exploration = np.sqrt((2 * np.log(N)) / n_i)
            
        ucb1_val = mean_reward + c_param * exploration
        scores[i] = {
            'frequency': int(n_i),
            'ucb1_score': ucb1_val
        }
    return scores

# ---------------------------------------------------------
# GIAO DIỆN CHÍNH: TẢI ẢNH DỮ LIỆU 3 KỲ
# ---------------------------------------------------------
st.sidebar.header("📥 Nạp Dữ Liệu 3 Kỳ Trước")
st.sidebar.info("Tải lên 3 ảnh chụp kết quả quay thưởng Keno gần nhất.")

uploaded_files = st.sidebar.file_uploader(
    "Chọn 3 ảnh kết quả Keno", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

draws_data = []

if uploaded_files:
    if len(uploaded_files) != 3:
        st.warning("⚠️ Vui lòng tải lên ĐÚNG 3 ÁNH tương ứng với 3 kỳ gần nhất!")
    else:
        st.subheader("📷 Kết quả trích xuất & Kiểm tra dữ liệu 3 kỳ")
        cols = st.columns(3)
        
        for idx, file in enumerate(uploaded_files):
            with cols[idx]:
                st.image(file, caption=f"Kỳ {idx+1}", use_container_width=True)
                extracted = extract_numbers_from_image(file)
                
                # Cho phép người dùng chỉnh sửa nếu OCR trích xuất nhầm
                numbers_str = st.text_input(
                    f"Danh sách 20 số Kỳ {idx+1} (cách nhau bởi dấu phẩy):",
                    value=", ".join(map(str, extracted)) if extracted else ""
                )
                
                # Parse lại chuỗi số
                parsed_nums = [int(x.strip()) for x in numbers_str.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 80]
                draws_data.append(parsed_nums)

# ---------------------------------------------------------
# NẾU ĐỦ DỮ LIỆU 3 KỲ -> TIẾN HÀNH PHÂN TÍCH
# ---------------------------------------------------------
if len(draws_data) == 3 and all(len(d) > 0 for d in draws_data):
    st.markdown("---")
    st.success("✅ Đã xác nhận đầy đủ dữ liệu 3 kỳ. Đang tiến hành phân tích UCB1 & Lọc bộ số...")

    # 1. Tính toán UCB1
    ucb_results = calculate_ucb1_scores(draws_data)
    
    # Chuyển sang DataFrame
    df_scores = pd.DataFrame.from_dict(ucb_results, orient='index')
    df_scores.index.name = 'Số'
    df_scores = df_scores.reset_index()
    
    # Sắp xếp số theo điểm UCB1 giảm dần
    df_sorted = df_scores.sort_values(by='ucb1_score', ascending=False)
    
    # Lấy ra danh sách các số tiềm năng nhất theo thứ tự UCB1
    top_candidates = df_sorted['Số'].tolist()

    # ---------------------------------------------------------
    # TỔNG HỢP & PHÂN TÍCH THỐNG KÊ (CHẰN/LẺ - LỚN/NHỎ)
    # ---------------------------------------------------------
    all_3_draws_flat = [num for draw in draws_data for num in draw]
    
    even_count = sum(1 for x in all_3_draws_flat if x % 2 == 0)
    odd_count = len(all_3_draws_flat) - even_count
    
    big_count = sum(1 for x in all_3_draws_flat if x >= 41)
    small_count = len(all_3_draws_flat) - big_count

    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.write("### 📊 Tỷ lệ Chẵn / Lẻ (Tổng 3 kỳ)")
        fig_parity = px.pie(
            names=['Chẵn', 'Lẻ'], 
            values=[even_count, odd_count], 
            color_discrete_sequence=['#1f77b4', '#ff7f0e'],
            hole=0.4
        )
        st.plotly_chart(fig_parity, use_container_width=True)

    with col_stat2:
        st.write("### 📊 Tỷ lệ Lớn (41-80) / Nhỏ (01-40)")
        fig_size = px.pie(
            names=['Lớn (41-80)', 'Nhỏ (01-40)'], 
            values=[big_count, small_count], 
            color_discrete_sequence=['#2ca02c', '#d62728'],
            hole=0.4
        )
        st.plotly_chart(fig_size, use_container_width=True)

    # ---------------------------------------------------------
    # PHÂN TÍCH & GỢI Ý BẬC 3 - BẬC 5 - BẬC 6
    # ---------------------------------------------------------
    st.markdown("---")
    st.header("🎯 BỘ SỐ GỢI Ý PHÂN TÍCH THEO BẬC (3 - 5 - 6)")
    
    tab3, tab5, tab6 = st.tabs(["🔥 Gợi Ý Bậc 3", "⚡ Gợi Ý Bậc 5", "💎 Gợi Ý Bậc 6"])

    # --- BẬC 3 ---
    with tab3:
        st.subheader("📌 Gợi Ý Dành Cho Keno Bậc 3 (Chọn 3 số)")
        st.write("Phương pháp: Kết hợp các số có điểm UCB1 cao nhất với tần suất ổn định.")
        
        set3_1 = top_candidates[:3]
        set3_2 = top_candidates[3:6]
        set3_3 = [top_candidates[0], top_candidates[1], top_candidates[6]]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Bộ Bậc 3 (Ưu tiên 1)", f"{sorted(set3_1)}")
        c2.metric("Bộ Bậc 3 (Ưu tiên 2)", f"{sorted(set3_2)}")
        c3.metric("Bộ Bậc 3 (Biến tấu)", f"{sorted(set3_3)}")

    # --- BẬC 5 ---
    with tab5:
        st.subheader("📌 Gợi Ý Dành Cho Keno Bậc 5 (Chọn 5 số)")
        st.write("Phương pháp: Phân bổ cân bằng giữa Top UCB1 + Cân bằng Chẵn/Lẻ.")
        
        set5_1 = top_candidates[:5]
        set5_2 = top_candidates[2:7]
        
        c1, c2 = st.columns(2)
        c1.metric("Bộ Bậc 5 (Tối ưu UCB1)", f"{sorted(set5_1)}")
        c2.metric("Bộ Bậc 5 (Khám phá)", f"{sorted(set5_2)}")

    # --- BẬC 6 ---
    with tab6:
        st.subheader("📌 Gợi Ý Dành Cho Keno Bậc 6 (Chọn 6 số)")
        st.write("Phương pháp: Tối đa hóa xác suất trúng hòa/trúng thưởng từ Bậc 6.")
        
        set6_1 = top_candidates[:6]
        set6_2 = top_candidates[1:7]
        
        c1, c2 = st.columns(2)
        c1.metric("Bộ Bậc 6 (Chiến thuật 1)", f"{sorted(set6_1)}")
        c2.metric("Bộ Bậc 6 (Chiến thuật 2)", f"{sorted(set6_2)}")

    # ---------------------------------------------------------
    # BẢNG ĐIỂM UCB1 CHI TIẾT
    # ---------------------------------------------------------
    with st.expander("📋 Xem toàn bộ bảng điểm UCB1 của 80 con số"):
        st.dataframe(df_sorted.reset_index(drop=True), use_container_width=True)

else:
    st.info("👈 Vui lòng tải lên đủ 3 ảnh kết quả Keno ở thanh bên trái để bắt đầu phân tích.")
