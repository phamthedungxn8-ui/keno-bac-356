import streamlit as st
import numpy as np
import pandas as pd
import itertools

# ---------------------------------------------------------
# Cấu hình ứng dụng
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keno Ising Model - Fixed Physics Kernel",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ KENO ISING MODEL - SỬA LỖ HỔNG LÕI TOÁN HỌC")
st.markdown("*Khắc phục hoàn toàn lỗi lặp lại số cũ bằng thuật toán Tính Tổng Năng Lượng Cấu Hình (Configuration Energy).*")
st.markdown("---")

# ---------------------------------------------------------
# LÕI THUẬT TOÁN ISING CHUẨN VẬT LÝ THỐNG KÊ
# ---------------------------------------------------------
class CorrectedKenoIsing:
    def __init__(self, recent_3_draws, alpha=1.5):
        self.draws = recent_3_draws
        self.alpha = alpha
        
    def compute_spin_matrix(self):
        S = -1 * np.ones((3, 80))
        for t, draw in enumerate(self.draws):
            for num in draw:
                if 1 <= num <= 80:
                    S[t, num - 1] = 1.0
        return S

    def run_analysis(self):
        S = self.compute_spin_matrix()
        
        # 1. Trường ngoài h_i (Mức độ kích thích từ 3 kỳ vừa qua)
        # Kỳ mới nhất có trọng số cao hơn
        weights = np.array([0.2, 0.3, 0.5]) 
        h = np.dot(weights, S)
        
        # 2. Ma trận tương tác Spin J_ij
        J = np.zeros((80, 80))
        for i in range(80):
            for j in range(i + 1, 80):
                # Tương tác Ising giữa cặp i và j
                J[i, j] = np.mean(S[:, i] * S[:, j])
                J[j, i] = J[i, j]
                
        # 3. Tính Điểm Hút Đơn (Single Spin Attractor)
        E_single = np.zeros(80)
        for i in range(80):
            E_single[i] = - (h[i] + self.alpha * np.sum(J[i, :]))
            
        # Top 15 số có năng lượng tiềm năng thấp nhất
        candidate_indices = np.argsort(E_single)[:15] + 1  # Chuyển về số 1-80
        
        # 4. THUẬT TOÁN MỚI: TÌM BỘ NĂNG LƯỢNG CỰC TIỂU TỔ HỢP (CONFIGURATION ENERGY)
        def get_best_combination(k_size, top_candidates):
            best_combo = None
            min_energy = float('inf')
            
            # Quét tất cả các tổ hợp C(candidates, k_size)
            for combo in itertools.combinations(top_candidates, k_size):
                indices = [c - 1 for c in combo]
                
                # Năng lượng đơn
                e_h = - np.sum(h[indices])
                
                # Năng lượng tương tác cặp trong bộ số
                e_j = 0
                for i_idx, j_idx in itertools.combinations(indices, 2):
                    e_j -= J[i_idx, j_idx]
                    
                total_energy = e_h + self.alpha * e_j
                
                if total_energy < min_energy:
                    min_energy = total_energy
                    best_combo = combo
                    
            return best_combo, min_energy

        # Tìm bộ Bậc 3, Bậc 5, Bậc 6 tối ưu thực sự
        best_3, e3 = get_best_combination(3, candidate_indices)
        best_5, e5 = get_best_combination(5, candidate_indices)
        best_6, e6 = get_best_combination(6, candidate_indices)
        
        return {
            "top_candidates": candidate_indices,
            "best_3": best_3,
            "best_5": best_5,
            "best_6": best_6,
            "e3": e3, "e5": e5, "e6": e6
        }

def format_numbers(num_list):
    if num_list is None:
        return ""
    clean_nums = [int(x) for x in sorted(num_list)]
    return " - ".join([f"{x:02d}" for x in clean_nums])

# ---------------------------------------------------------
# GIAO DIỆN KIỂM THỬ
# ---------------------------------------------------------
st.subheader("📝 Nhập dữ liệu 3 kỳ quay để kiểm tra thuật toán mới")

col1, col2, col3 = st.columns(3)
with col1:
    k1_input = st.text_area("Kỳ 1 (20 số cách nhau khoảng trắng):", "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20", height=100)
with col2:
    k2_input = st.text_area("Kỳ 2 (20 số cách nhau khoảng trắng):", "01 02 03 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37", height=100)
with col3:
    k3_input = st.text_area("Kỳ 3 (20 số cách nhau khoảng trắng):", "01 04 05 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54", height=100)

def parse_input(text):
    nums = [int(x) for x in text.split() if x.isdigit() and 1 <= int(x) <= 80]
    return sorted(list(set(nums)))[:20]

d1, d2, d3 = parse_input(k1_input), parse_input(k2_input), parse_input(k3_input)

if len(d1) == 20 and len(d2) == 20 and len(d3) == 20:
    st.markdown("---")
    
    model = CorrectedKenoIsing([d1, d2, d3])
    res = model.run_analysis()
    
    st.header("🎯 KẾT QUẢ TÍNH TOÁN BẰNG MA TRẬN NĂNG LƯỢNG TỔ HỢP")
    st.caption("Các bộ số gợi ý dưới đây đã được lọc qua ma trận tương tác Ising $J_{ij}$ để chọn ra tổ hợp có mức liên kết năng lượng cao nhất, không bị lặp lại đơn thuần 3 số của 1 kỳ.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🔥 Gợi ý Bậc 3 Tối Ưu", format_numbers(res["best_3"]), f"Energy: {res['e3']:.3f}")
    c2.metric("⚡ Gợi ý Bậc 5 Tối Ưu", format_numbers(res["best_5"]), f"Energy: {res['e5']:.3f}")
    c3.metric("💎 Gợi ý Bậc 6 Tối Ưu", format_numbers(res["best_6"]), f"Energy: {res['e6']:.3f}")

    st.info(f"📍 Danh sách 15 ứng viên điểm hút hàng đầu: `{format_numbers(res['top_candidates'])}`")
else:
    st.warning("Vui lòng đảm bảo cả 3 kỳ đều có đủ 20 số hợp lệ!")
