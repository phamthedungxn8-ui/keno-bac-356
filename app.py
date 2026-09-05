import streamlit as st
import numpy as np
import pandas as pd
import itertools
import re

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keno Ising Model - Strict 3-Tier Architecture",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ KENO ISING MODEL - NGUYÊN TẮC DỮ LIỆU THỰC")
st.caption("1️⃣ Real Historical Baseline (J) | 2️⃣ Impulse Perturbation (h) | 3️⃣ Combinatorial Energy Solver")
st.markdown("---")

# ---------------------------------------------------------
# HÀM XỬ LÝ DỮ LIỆU THÔNG MINH (TỰ ĐỘNG TÁCH SỐ DÍNH LIỀN)
# ---------------------------------------------------------
def extract_numbers_from_text(text):
    """
    Tự động trích xuất các số từ 1 đến 80:
    - Hỗ trợ chuỗi dính liền từ minhchinh.com (VD: 06070911 -> 06, 07, 09, 11)
    - Hỗ trợ văn bản có khoảng trắng, dấu phẩy, dấu gạch ngang
    """
    if not text:
        return []
    
    # 1. Nếu có khoảng trắng hoặc ký tự phân cách, tách theo phân cách trước
    tokens = re.split(r'[\s,\t\n\-]+', text.strip())
    
    extracted_nums = []
    for token in tokens:
        clean_token = re.sub(r'\D', '', token) # Chỉ giữ lại chữ số
        if not clean_token:
            continue
            
        # Nếu đoạn chữ số dài > 2 (VD: dính liền kiểu 06070911...), cứ 2 chữ số cắt 1 lần
        if len(clean_token) > 2 and len(clean_token) % 2 == 0:
            for i in range(0, len(clean_token), 2):
                num = int(clean_token[i:i+2])
                if 1 <= num <= 80:
                    extracted_nums.append(num)
        else:
            num = int(clean_token)
            if 1 <= num <= 80:
                extracted_nums.append(num)
                
    return extracted_nums

def parse_multi_line_input(text):
    """Xử lý nhập nhiều kỳ cho Tầng 1"""
    lines = text.strip().split('\n')
    draws = []
    for line in lines:
        nums = extract_numbers_from_text(line)
        # Loại bỏ trùng lặp trong cùng 1 kỳ nhưng vẫn giữ thứ tự / sắp xếp
        unique_nums = sorted(list(dict.fromkeys(nums)))
        if len(unique_nums) == 20:
            draws.append(unique_nums)
    return draws

def parse_input_string(text):
    """Xử lý nhập 1 kỳ (20 số) cho Tầng 2"""
    nums = extract_numbers_from_text(text)
    unique_nums = sorted(list(dict.fromkeys(nums)))
    return unique_nums[:20]

def format_numbers(num_list):
    if not num_list:
        return ""
    return " - ".join([f"{x:02d}" for x in num_list])

# ---------------------------------------------------------
# KHUNG KIẾN TRÚC 3 TẦNG ĐỘC LẬP
# ---------------------------------------------------------

class HistoricalBaselineEngine:
    def __init__(self, history_draws: list[list[int]]):
        self.history_draws = history_draws
        self.N = len(history_draws)
        self.S_history = self._build_spin_matrix()

    def _build_spin_matrix(self) -> np.ndarray:
        S = -1 * np.ones((self.N, 80))
        for t, draw in enumerate(self.history_draws):
            for num in draw:
                if 1 <= num <= 80:
                    S[t, num - 1] = 1.0
        return S

    def compute_coupling_matrix(self) -> np.ndarray:
        J = np.zeros((80, 80))
        mean_spins = np.mean(self.S_history, axis=0)
        
        for i in range(80):
            for j in range(i + 1, 80):
                cov = np.mean(self.S_history[:, i] * self.S_history[:, j]) - (mean_spins[i] * mean_spins[j])
                J[i, j] = cov
                J[j, i] = cov
        return J

class ImpulseFieldEngine:
    def __init__(self, recent_3_draws: list[list[int]]):
        self.recent_3_draws = recent_3_draws
        self.S_recent = self._build_spin_matrix()

    def _build_spin_matrix(self) -> np.ndarray:
        S = -1 * np.ones((3, 80))
        for t, draw in enumerate(self.recent_3_draws):
            for num in draw:
                if 1 <= num <= 80:
                    S[t, num - 1] = 1.0
        return S

    def compute_external_field(self) -> np.ndarray:
        weights = np.array([0.2, 0.3, 0.5])
        return np.dot(weights, self.S_recent)

class ConfigurationEnergySolver:
    def __init__(self, J_matrix: np.ndarray, h_field: np.ndarray, alpha: float = 1.5):
        self.J = J_matrix
        self.h = h_field
        self.alpha = alpha

    def _compute_single_spin_potentials(self) -> np.ndarray:
        E_single = np.zeros(80)
        for i in range(80):
            spin_interaction = np.sum(self.J[i, :])
            E_single[i] = - (self.h[i] + self.alpha * spin_interaction)
        return E_single

    def find_optimal_configuration(self, k_size: int, candidate_pool_size: int = 18):
        E_single = self._compute_single_spin_potentials()
        top_candidates = np.argsort(E_single)[:candidate_pool_size] + 1

        best_combo = None
        min_energy = float('inf')

        for combo in itertools.combinations(top_candidates, k_size):
            indices = [c - 1 for c in combo]
            e_h = - np.sum(self.h[indices])
            
            e_j = 0
            for i_idx, j_idx in itertools.combinations(indices, 2):
                e_j -= self.J[i_idx, j_idx]
                
            total_energy = e_h + self.alpha * e_j
            
            if total_energy < min_energy:
                min_energy = total_energy
                best_combo = combo

        return sorted(list(best_combo)), min_energy, sorted(list(top_candidates))

# ---------------------------------------------------------
# GIAO DIỆN ĐIỀU KHIỂN
# ---------------------------------------------------------

# =========================================================
# 1️⃣ TẦNG 1: DỮ LIỆU NỀN LỊCH SỬ THỰC TẾ
# =========================================================
st.header("1️⃣ TẦNG 1: Nạp Dữ Liệu Nền Lịch Sử Thực (Bắt Buộc)")
st.caption("Ma trận J_ij chỉ được phép tính khi có dữ liệu lịch sử thực tế (Tối thiểu 10 kỳ).")

tab_manual, tab_file = st.tabs(["📝 Nhập Văn Bản Lịch Sử", "📁 Tải File CSV Lịch Sử"])

history_data = []

with tab_manual:
    manual_text = st.text_area(
        "Dán chuỗi kết quả lịch sử (Hỗ trợ copy dính liền từ minhchinh.com):",
        placeholder="Dán chuỗi 06070911121719... hoặc dạng có khoảng trắng",
        height=150
    )
    if manual_text.strip():
        history_data = parse_multi_line_input(manual_text)

with tab_file:
    uploaded_file = st.file_uploader("Tải file CSV/TXT chứa lịch sử các kỳ", type=["csv", "txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        history_data = parse_multi_line_input(content)

# Kiểm duyệt dữ liệu Tầng 1
if len(history_data) >= 10:
    st.success(f"✅ TẦNG 1 HỢP LỆ: Đã nhận diện thành công **{len(history_data)} kỳ** đủ 20 số.")
else:
    st.error(f"❌ TẦNG 1 CHƯA ĐỦ DỮ LIỆU: Đã nhận diện **{len(history_data)} kỳ** chuẩn. Cần tối thiểu **10 kỳ** (mỗi kỳ đủ 20 số).")

st.markdown("---")

# =========================================================
# 2️⃣ TẦNG 2: KÍCH THÍCH TỨC THỜI
# =========================================================
st.header("2️⃣ TẦNG 2: Nhập 3 Kỳ Quay Kích Thích (Trường Ngoài h_i)")

col1, col2, col3 = st.columns(3)
with col1:
    k1_text = st.text_area("Kỳ 1 (Dán dính liền hoặc khoảng trắng):", "", height=100)
    d1 = parse_input_string(k1_text)
    if len(d1) == 20:
        st.caption(f"✓ Đã tách: `{format_numbers(d1)}`")
with col2:
    k2_text = st.text_area("Kỳ 2 (Dán dính liền hoặc khoảng trắng):", "", height=100)
    d2 = parse_input_string(k2_text)
    if len(d2) == 20:
        st.caption(f"✓ Đã tách: `{format_numbers(d2)}`")
with col3:
    k3_text = st.text_area("Kỳ 3 (Dán dính liền hoặc khoảng trắng):", "", height=100)
    d3 = parse_input_string(k3_text)
    if len(d3) == 20:
        st.caption(f"✓ Đã tách: `{format_numbers(d3)}`")

st.markdown("---")

# =========================================================
# 3️⃣ TẦNG 3: TỐI ƯU NĂNG LƯỢNG CẤU HÌNH (E_min)
# =========================================================
st.header("3️⃣ TẦNG 3: Kết Quả Dự Đoán Tối Ưu Năng LƯợng")

if len(history_data) < 10:
    st.warning("🔒 Tầng 3 bị khóa: Vui lòng nạp đủ 10 kỳ ở Tầng 1.")
elif len(d1) != 20 or len(d2) != 20 or len(d3) != 20:
    st.warning("🔒 Tầng 3 bị khóa: Vui lòng dán đủ 20 số cho từng kỳ ở Tầng 2.")
else:
    baseline_engine = HistoricalBaselineEngine(history_data)
    J_matrix = baseline_engine.compute_coupling_matrix()

    impulse_engine = ImpulseFieldEngine([d1, d2, d3])
    h_field = impulse_engine.compute_external_field()

    solver = ConfigurationEnergySolver(J_matrix, h_field, alpha=1.5)

    b3, e3, pool = solver.find_optimal_configuration(k_size=3)
    b5, e5, _ = solver.find_optimal_configuration(k_size=5)
    b6, e6, _ = solver.find_optimal_configuration(k_size=6)

    m1, m2, m3 = st.columns(3)
    m1.metric("🔥 Bậc 3 Tối Ưu", format_numbers(b3), f"Energy: {e3:.4f}")
    m2.metric("⚡ Bậc 5 Tối Ưu", format_numbers(b5), f"Energy: {e5:.4f}")
    m3.metric("💎 Bậc 6 Tối Ưu", format_numbers(b6), f"Energy: {e6:.4f}")

    st.write(f"• **Ứng viên điểm hút (Pool 18):** `{format_numbers(pool)}`")
