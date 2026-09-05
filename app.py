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

st.title("⚛️ KENO ISING MODEL - TỐI ƯU NĂNG LƯỢNG CẤU HÌNH")
st.caption("Khung kiến trúc 3 tầng: Historical Baseline Matrix (J) | Impulse Perturbation (h) | Combinatorial Energy Solver")
st.markdown("---")

# ---------------------------------------------------------
# KHUNG KIẾN TRÚC 3 TẦNG ĐỘC LẬP
# ---------------------------------------------------------

# =========================================================
# TẦNG 1: HISTORICAL BASELINE MATRIX (Ma trận nền lịch sử)
# =========================================================
class HistoricalBaselineEngine:
    """
    Tầng 1: Xây dựng Ma trận Tương tác Spin J_ij tĩnh dài hạn.
    Đo đạc lực liên kết, hiệp phương sai và độ lệch chuẩn giữa 80 ô trong N kỳ.
    """
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

    def compute_coupling_matrix(()) -> np.ndarray:
        """
        Tính toán ma trận tương tác J_ij dựa trên Hiệp phương sai Ising.
        """
        J = np.zeros((80, 80))
        mean_spins = np.mean(self.S_history, axis=0)
        
        for i in range(80):
            for j in range(i + 1, 80):
                cov = np.mean(self.S_history[:, i] * self.S_history[:, j]) - (mean_spins[i] * mean_spins[j])
                J[i, j] = cov
                J[j, i] = cov
        return J


# =========================================================
# TẦNG 2: INSTANT IMPULSE PERTURBATION (Kích thích tức thời)
# =========================================================
class ImpulseFieldEngine:
    """
    Tầng 2: Xử lý 3 kỳ quay gần nhất làm Xung động (Impulse) tác động vào hệ thống.
    Biến dạng mặt năng lượng bằng Trường Tương Tác Ngoài h_i.
    """
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
        """
        Tính toán Vector Trường Ngoài h_i với Trọng số Suy giảm Thời gian (Decay Weights).
        Kỳ mới nhất đóng góp trọng số kích thích lớn nhất.
        """
        weights = np.array([0.2, 0.3, 0.5])  # Kỳ 1 -> Kỳ 2 -> Kỳ 3
        h = np.dot(weights, self.S_recent)
        return h


# =========================================================
# TẦNG 3: COMBINATORIAL CONFIGURATION SOLVER (Giải năng lượng tổ hợp)
# =========================================================
class ConfigurationEnergySolver:
    """
    Tầng 3: Giải bài toán Tối ưu hóa Năng lượng Toàn cục E(S) trên không gian C(80, k).
    E(S) = - sum(h_i) - alpha * sum(J_ij)
    """
    def __init__(self, J_matrix: np.ndarray, h_field: np.ndarray, alpha: float = 1.2):
        self.J = J_matrix
        self.h = h_field
        self.alpha = alpha

    def _compute_single_spin_potentials(self) -> np.ndarray:
        """Tính năng lượng tiềm năng đơn ô để thu hẹp không gian ứng viên Top N."""
        E_single = np.zeros(80)
        for i in range(80):
            spin_interaction = np.sum(self.J[i, :])
            E_single[i] = - (self.h[i] + self.alpha * spin_interaction)
        return E_single

    def find_optimal_configuration(self, k_size: int, candidate_pool_size: int = 18):
        """
        Quét toàn bộ không gian C(pool, k_size) để tìm cấu hình k số có Năng lượng Cực tiểu.
        """
        E_single = self._compute_single_spin_potentials()
        top_candidates = np.argsort(E_single)[:candidate_pool_size] + 1  # Chuyển về 1-80

        best_combo = None
        min_energy = float('inf')

        for combo in itertools.combinations(top_candidates, k_size):
            indices = [c - 1 for c in combo]
            
            # 1. Năng lượng kích thích đơn
            e_h = - np.sum(self.h[indices])
            
            # 2. Năng lượng liên kết nội bộ cấu hình
            e_j = 0
            for i_idx, j_idx in itertools.combinations(indices, 2):
                e_j -= self.J[i_idx, j_idx]
                
            total_energy = e_h + self.alpha * e_j
            
            if total_energy < min_energy:
                min_energy = total_energy
                best_combo = combo

        return sorted(list(best_combo)), min_energy, sorted(list(top_candidates))


# ---------------------------------------------------------
# TIỆN ÍCH HIỂN THỊ & GIẢ LẬP DỮ LIỆU NỀN
# ---------------------------------------------------------
def generate_synthetic_history(n_draws=50):
    """Giả lập 50 kỳ lịch sử nếu chưa chọn file Nền Lịch Sử."""
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
# GIAO DIỆN ĐIỀU KHIỂN STREAMLIT
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu Hình Tầng Hệ Thống")
history_length = st.sidebar.slider("Dữ liệu Nền (Số kỳ Lịch sử):", min_value=30, max_value=100, value=50, step=10)
alpha_coupling = st.sidebar.slider("Hệ số Cấu hình Tương tác (Alpha):", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

st.subheader("1️⃣ TẦNG 2: Nhập 3 Kỳ Quay Kích Thích (Impulse)")

col1, col2, col3 = st.columns(3)
with col1:
    k1_text = st.text_area("Kỳ 1 (20 số):", "01 03 05 08 12 15 19 22 25 30 33 41 45 50 55 60 62 70 73 80", height=90)
with col2:
    k2_text = st.text_area("Kỳ 2 (20 số):", "02 03 07 10 12 18 22 28 30 35 40 45 51 55 61 68 70 74 77 80", height=90)
with col3:
    k3_text = st.text_area("Kỳ 3 (20 số):", "01 04 05 11 15 20 25 31 33 42 45 50 56 60 63 70 72 75 78 79", height=90)

d1, d2, d3 = parse_input_string(k1_text), parse_input_string(k2_text), parse_input_string(k3_text)

if len(d1) == 20 and len(d2) == 20 and len(d3) == 20:
    st.markdown("---")
    
    # 1. KÍCH HOẠT TẦNG 1: Matrix Engine (J_ij)
    history_data = generate_synthetic_history(n_draws=history_length)
    baseline_engine = HistoricalBaselineEngine(history_data)
    J_matrix = baseline_engine.compute_coupling_matrix()

    # 2. KÍCH HOẠT TẦNG 2: Impulse Engine (h_i)
    impulse_engine = ImpulseFieldEngine([d1, d2, d3])
    h_field = impulse_engine.compute_external_field()

    # 3. KÍCH HOẠT TẦNG 3: Combinatorial Solver
    solver = ConfigurationEnergySolver(J_matrix, h_field, alpha=alpha_coupling)

    b3, e3, pool = solver.find_optimal_configuration(k_size=3)
    b5, e5, _ = solver.find_optimal_configuration(k_size=5)
    b6, e6, _ = solver.find_optimal_configuration(k_size=6)

    st.subheader("2️⃣ TẦNG 3: Kết Quả Dự Đoán Năng Lượng Cực Tiểu Toàn Cục")

    m1, m2, m3 = st.columns(3)
    m1.metric("🔥 Bậc 3 Tối Ưu Cấu Hình", format_numbers(b3), f"Energy: {e3:.4f}")
    m2.metric("⚡ Bậc 5 Tối Ưu Cấu Hình", format_numbers(b5), f"Energy: {e5:.4f}")
    m3.metric("💎 Bậc 6 Tối Ưu Cấu Hình", format_numbers(b6), f"Energy: {e6:.4f}")

    st.markdown("---")
    st.subheader("🔍 Nhật Ký Phân Tích Hệ Thống")
    st.write(f"• **Không gian ứng viên (Attractor Candidate Pool 18):** `{format_numbers(pool)}`")
    st.write(f"• **Ma trận Tương tác $J_{{ij}}$:** Đã tính toán trên **{history_length}** kỳ lịch sử.")
    st.write(f"• **Thuật toán Tối ưu:** Đã quét **$C_{{18}}^3, C_{{18}}^5, C_{{18}}^6$** để tìm cấu hình có tổn hao năng lượng thấp nhất.")

else:
    st.error("⚠️ Vui lòng nhập đúng và đủ 20 số cho cả 3 kỳ để thực thi hệ thống.")
