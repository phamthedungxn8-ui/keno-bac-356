import streamlit as st
import numpy as np
import pandas as pd
import itertools
import re

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN
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
# THUẬT TOÁN BÓC TÁCH SỐ THÔNG MINH (XỬ LÝ DÍNH LIỀN 100%)
# ---------------------------------------------------------
def extract_all_valid_numbers(text):
    """
    Bóc tách toàn bộ danh sách các số từ 1-80 trong văn bản.
    Hỗ trợ dính liền kiểu 060709... hoặc có khoảng trắng, phẩy, gạch ngang.
    """
    if not text:
        return []
    
    # 1. Nếu văn bản là chuỗi dài toàn chữ số dính liền (không khoảng trắng)
    raw_digits = re.sub(r'\D', '', text)
    
    # Nếu là chuỗi chữ số dính liền có độ dài chẵn (mỗi số 2 chữ số)
    extracted_nums = []
    if len(raw_digits) >= 2:
        for i in range(0, len(raw_digits) - 1, 2):
            num_str = raw_digits[i:i+2]
            num = int(num_str)
            if 1 <= num <= 80:
                extracted_nums.append(num)
                
    return extracted_nums

def parse_multi_line_input(text):
    """
    Tự động nhóm cứ đúng 20 số hợp lệ thành 1 kỳ hoàn chỉnh.
    Bất kể người dùng dán 1 dòng dài hay nhiều dòng.
    """
    all_nums = extract_all_valid_numbers(text)
    draws = []
    
    # Cắt danh sách tổng thành từng cụm 20 số
    for i in range(0, len(all_nums), 20):
        chunk = all_nums[i:i+20]
        # Đảm bảo cụm đủ 20 số và không trùng lặp trong cùng kỳ
        unique_chunk = sorted(list(set(chunk)))
        if len(chunk) == 20 and len(unique_chunk) == 20:
            draws.append(unique_chunk)
            
    return draws

def parse_single_draw_input(text):
    """Trích xuất duy nhất 1 kỳ (20 số) cho Tầng 2"""
    nums = extract_all_valid_numbers(text)
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
st.caption("Ma trận J_ij chỉ được tính khi có đủ dữ liệu lịch sử thực tế (Tối thiểu 10 kỳ).")

tab_manual, tab_file = st.tabs(["📝 Nhập Văn Bản Lịch Sử", "📁 Tải File CSV Lịch Sử"])

history_data = []

with tab_manual:
    manual_text = st.text_area(
        "Dán chuỗi kết quả lịch sử (Dán dính liền liên tục nhiều kỳ thoải mái):",
        placeholder="Dán chuỗi 0607091112171924283435384247495062687072...",
        height=150
    )
    if manual_text.strip():
        history_data = parse_multi_line_input(manual_text)

with tab_file:
    uploaded_file = st.file_uploader("Tải file CSV/TXT chứa lịch sử các kỳ", type=["csv", "txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        history_data = parse_multi_line_input(content)

# Trạng thái nhận diện Tầng 1
if len(history_data) >= 10:
    st.success(f"✅ TẦNG 1 HỢP LỆ: Đã tự động bóc tách thành công **{len(history_data)} kỳ chuẩn** (mỗi kỳ 20 số).")
else:
    st.error(f"❌ TẦNG 1 CHƯA ĐỦ DỮ LIỆU: Hiện đã bóc tách được **{len(history_data)} kỳ chuẩn**. Cần dán thêm để đạt tối thiểu **10 kỳ**.")

st.markdown("---")

# =========================================================
# 2️⃣ TẦNG 2: KÍCH THÍCH TỨC THỜI
# =========================================================
st.header("2️⃣ TẦNG 2: Nhập 3 Kỳ Quay Kích Thích (Trường Ngoài h_i)")

col1, col2, col3 = st.columns(3)
with col1:
    k1_text = st.text_area("Kỳ 1 (20 số dính liền hoặc thưa):", "", height=90)
    d1 = parse_single_draw_input(k1_text)
    if len(d1) == 20:
        st.caption(f"✓ Kỳ 1: `{format_numbers(d1)}`")
    else:
        st.caption(f"⚠️ Mới nhận: {len(d1)}/20 số")

with col2:
    k2_text = st.text_area("Kỳ 2 (20 số dính liền hoặc thưa):", "", height=90)
    d2 = parse_single_draw_input(k2_text)
    if len(d2) == 20:
        st.caption(f"✓ Kỳ 2: `{format_numbers(d2)}`")
    else:
        st.caption(f"⚠️ Mới nhận: {len(d2)}/20 số")

with col3:
    k3_text = st.text_area("Kỳ 3 (20 số dính liền hoặc thưa):", "", height=90)
    d3 = parse_single_draw_input(k3_text)
    if len(d3) == 20:
        st.caption(f"✓ Kỳ 3: `{format_numbers(d3)}`")
    else:
        st.caption(f"⚠️ Mới nhận: {len(d3)}/20 số")

st.markdown("---")

# =========================================================
# 3️⃣ TẦNG 3: TỐI ƯU NĂNG LƯỢNG CẤU HÌNH (E_min)
# =========================================================
st.header("3️⃣ TẦNG 3: Kết Quả Dự Đoán Tối Ưu Năng Lượng")

if len(history_data) < 10:
    st.warning("🔒 Tầng 3 bị khóa: Vui lòng nạp đủ tối thiểu 10 kỳ ở Tầng 1.")
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
