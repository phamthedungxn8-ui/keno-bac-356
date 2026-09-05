import streamlit as st
import numpy as np
import pandas as pd
import itertools

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keno Ising Model - 3-Tier Architecture",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ KENO ISING MODEL - KHUNG KIẾN TRÚC 3 TẦNG")
st.caption("1️⃣ Historical Baseline Matrix (J) | 2️⃣ Impulse Perturbation (h) | 3️⃣ Combinatorial Energy Solver")
st.markdown("---")

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
# TIỆN ÍCH
# ---------------------------------------------------------
def generate_synthetic_history(n_draws=50):
    np.random.seed(42)
    history = []
    for _ in range(n_draws):
        draw = np.random.choice(np.arange(1, 81), size=20, replace=False)
        history.append(sorted(draw))
    return history

def parse_input_string(text):
    nums = [int(x) for x in text.split() if x.isdigit() and 1 <= int(x) <= 80]
    return sorted(list(set(nums)))[:20]

def format_numbers(num_list):
    if not num_list:
        return ""
    return " - ".join([f"{x:02d}" for x in num_list])

# ---------------------------------------------------------
# HIỂN THỊ CHUẨN 3 TẦNG RÕ RÀNG TRÊN MÀN HÌNH CHÍNH
# ---------------------------------------------------------

# =========================================================
# 1️⃣ TẦNG 1: DỮ LIỆU NỀN LỊCH SỬ (HISTORICAL BASELINE)
# =========================================================
st.header("1️⃣ TẦNG 1: Dữ Liệu Nền Lịch Sử (Ma Trận Tương Tác J_ij)")
st.caption("Thiết lập ma trận liên kết tĩnh giữa 80 con số dựa trên chuỗi thời gian dài hạn.")

col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    history_length = st.slider("Số kỳ lịch sử dùng để đo độ liên kết J_ij:", min_value=30, max_value=100, value=50, step=10)
with col_h2:
    st.info(f"📊 Trạng thái Tầng 1: Đã sẵn sàng **{history_length} kỳ** làm nền.")

st.markdown("---")

# =========================================================
# 2️⃣ TẦNG 2: KÍCH THÍCH TỨC THỜI (IMPULSE PERTURBATION)
# =========================================================
st.header("2️⃣ TẦNG 2: Nhập 3 Kỳ Quay Kích Thích (Trường Ngoài h_i)")
st.caption("3 kỳ gần nhất làm xung động làm lệch mặt bằng năng lượng hệ thống.")

col1, col2, col3 = st.columns(3)
with col1:
    k1_text = st.text_area("Kỳ 1 (20 số):", "01 03 05 08 12 15 19 22 25 30 33 41 45 50 55 60 62 70 73 80", height=80)
with col2:
    k2_text = st.text_area("Kỳ 2 (20 số):", "02 03 07 10 12 18 22 28 30 35 40 45 51 55 61 68 70 74 77 80", height=80)
with col3:
    k3_text = st.text_area("Kỳ 3 (20 số):", "01 04 05 11 15 20 25 31 33 42 45 50 56 60 63 70 72 75 78 79", height=80)

d1, d2, d3 = parse_input_string(k1_text), parse_input_string(k2_text), parse_input_string(k3_text)

st.markdown("---")

# =========================================================
# 3️⃣ TẦNG 3: TỐI ƯU NĂNG LƯỢNG CẤU HÌNH (COMBINATORIAL SOLVER)
# =========================================================
st.header("3️⃣ TẦNG 3: Tối Ưu Năng Lượng Cấu Hình Tổ Hợp (E_min)")

if len(d1) == 20 and len(d2) == 20 and len(d3) == 20:
    # Chạy Tầng 1
    history_data = generate_synthetic_history(n_draws=history_length)
    baseline_engine = HistoricalBaselineEngine(history_data)
    J_matrix = baseline_engine.compute_coupling_matrix()

    # Chạy Tầng 2
    impulse_engine = ImpulseFieldEngine([d1, d2, d3])
    h_field = impulse_engine.compute_external_field()

    # Chạy Tầng 3
    solver = ConfigurationEnergySolver(J_matrix, h_field, alpha=1.5)

    b3, e3, pool = solver.find_optimal_configuration(k_size=3)
    b5, e5, _ = solver.find_optimal_configuration(k_size=5)
    b6, e6, _ = solver.find_optimal_configuration(k_size=6)

    m1, m2, m3 = st.columns(3)
    m1.metric("🔥 Bậc 3 Tối Ưu", format_numbers(b3), f"Energy: {e3:.4f}")
    m2.metric("⚡ Bậc 5 Tối Ưu", format_numbers(b5), f"Energy: {e5:.4f}")
    m3.metric("💎 Bậc 6 Tối Ưu", format_numbers(b6), f"Energy: {e6:.4f}")

    st.write(f"• **Ứng viên điểm hút (Pool 18):** `{format_numbers(pool)}`")
else:
    st.error("⚠️ Vui lòng kiểm tra lại 3 kỳ quay ở Tầng 2 để đảm bảo đủ 20 số/kỳ.")
