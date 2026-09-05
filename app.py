import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# Cấu hình giao diện Streamlit Tối ưu Mobile
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keno Ising Model - Bàn Phím Tối Ưu Mobile",
    page_icon="⚛️",
    layout="wide"
)

# CSS tùy chỉnh để làm nút bấm to hơn, dễ chạm trên điện thoại
st.markdown("""
    <style>
    div.stButton > button {
        height: 3em;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 2px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚛️ KENO ISING MODEL - BÀN PHÍM TỐI ƯU CẢM ỨNG")
st.markdown("*Bố trí Lưới 5 cột - Nút bấm to, phân vùng màu dễ chọn nhất trên điện thoại.*")
st.markdown("---")

# ---------------------------------------------------------
# QUẢN LÝ TRẠNG THÁI BỘ CHỌN SỐ (SESSION STATE)
# ---------------------------------------------------------
for k in [1, 2, 3]:
    if f'selected_k{k}' not in st.session_state:
        st.session_state[f'selected_k{k}'] = []

def toggle_number(key_name, num):
    if num in st.session_state[key_name]:
        st.session_state[key_name].remove(num)
    else:
        if len(st.session_state[key_name]) < 20:
            st.session_state[key_name].append(num)

def clear_selected(key_name):
    st.session_state[key_name] = []

# ---------------------------------------------------------
# 1. LÕI THUẬT TOÁN: ISING SPIN GLASS
# ---------------------------------------------------------
class KenoIsingDynamics:
    def __init__(self, draws, alpha=1.2):
        self.draws = draws
        self.N_draws = len(draws)
        self.alpha = alpha
        self.spins_history = self._build_spin_matrix()
        
    def _build_spin_matrix(self):
        S = -1 * np.ones((self.N_draws, 80))
        for t, draw in enumerate(self.draws):
            for num in draw:
                if 1 <= num <= 80:
                    S[t, num - 1] = 1.0
        return S

    def calculate_external_field(self):
        weights = np.exp(np.linspace(-0.5, 0, self.N_draws))
        weights /= weights.sum()
        return np.dot(weights, self.spins_history)

    def calculate_coupling_matrix(self):
        J = np.zeros((80, 80))
        for i in range(80):
            for j in range(i + 1, 80):
                cov = np.mean(self.spins_history[:, i] * self.spins_history[:, j]) - \
                      np.mean(self.spins_history[:, i]) * np.mean(self.spins_history[:, j])
                J[i, j] = cov
                J[j, i] = cov
        return J

    def compute_attractor_energies(self):
        h = self.calculate_external_field()
        J = self.calculate_coupling_matrix()
        delta_E = np.zeros(80)
        for i in range(80):
            spin_interaction = np.sum(J[i, :] * np.mean(self.spins_history, axis=0))
            delta_E[i] = - (h[i] + self.alpha * spin_interaction)
        return delta_E, h, J

    def compute_entropy_state(self):
        p_active = np.mean(self.spins_history == 1, axis=0)
        p_active = np.clip(p_active, 1e-5, 1 - 1e-5)
        entropy_per_spin = - (p_active * np.log2(p_active) + (1 - p_active) * np.log2(1 - p_active))
        return np.mean(entropy_per_spin)

def format_numbers(num_list):
    clean_nums = [int(x) for x in sorted(num_list)]
    return " - ".join([f"{x:02d}" for x in clean_nums])

# ---------------------------------------------------------
# 2. GIAO DIỆN LƯỚI 5 CỘT TỐI ƯU THAO TÁC
# ---------------------------------------------------------
st.header("🎮 Chọn 20 Số Cho 3 Kỳ Vừa Quay")

tab_k1, tab_k2, tab_k3 = st.tabs(["📌 Kỳ 1 (Kỳ xa nhất)", "📌 Kỳ 2 (Kỳ giữa)", "📌 Kỳ 3 (Kỳ mới nhất)"])
tabs = [tab_k1, tab_k2, tab_k3]

for idx, tab in enumerate(tabs):
    k_num = idx + 1
    key_name = f'selected_k{k_num}'
    
    with tab:
        curr_selected = st.session_state[key_name]
        
        # Thanh trạng thái tiến độ
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.subheader(f"Đã chọn: {len(curr_selected)}/20 số")
            if curr_selected:
                st.info(f"👉 `{format_numbers(curr_selected)}`")
        with col_btn:
            if st.button(f"🗑️ Xóa Kỳ {k_num}", key=f"btn_clear_{k_num}", use_container_width=True):
                clear_selected(key_name)
                st.rerun()

        st.markdown("---")
        
        # TẠO LƯỚI 5 CỘT (Mỗi hàng 5 nút - Phù hợp ngón tay bấm trên điện thoại)
        cols_per_row = 5
        total_numbers = 80
        
        for row in range(total_numbers // cols_per_row):
            # Cứ sau 4 hàng (20 số) sẽ chèn tiêu đề nhóm hàng chục để dễ nhìn
            start_num = row * cols_per_row + 1
            if (start_num - 1) % 20 == 0:
                group_start = start_num
                group_end = start_num + 19
                st.caption(f"🔻 **Vùng số {group_start:02d} đến {group_end:02d}**")

            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                num = row * cols_per_row + col + 1
                is_selected = num in curr_selected
                
                # Nhãn hiển thị trực quan
                btn_label = f"✅ {num:02d}" if is_selected else f"{num:02d}"
                btn_type = "primary" if is_selected else "secondary"
                
                if cols[col].button(btn_label, key=f"btn_k{k_num}_{num}", type=btn_type, use_container_width=True):
                    toggle_number(key_name, num)
                    st.rerun()

# ---------------------------------------------------------
# 3. CHẠY PHÂN TÍCH ISING DYNAMICAL MODEL
# ---------------------------------------------------------
draws_data = [st.session_state['selected_k1'], st.session_state['selected_k2'], st.session_state['selected_k3']]

if all(len(d) == 20 for d in draws_data):
    st.markdown("---")
    st.success("✅ Đã chọn đủ 20/20 số cho 3 kỳ! Đang tính toán ma trận Ising...")

    model = KenoIsingDynamics(draws_data)
    delta_E, h_field, J_matrix = model.compute_attractor_energies()
    system_entropy = model.compute_entropy_state()

    sorted_indices = np.argsort(delta_E)
    top_attractors = [int(idx + 1) for idx in sorted_indices]

    # Hiển thị chỉ số
    st.header("🌀 Trạng Thái Hệ Thống & Bộ Số Gợi Ý")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Entropy Shannon Toàn Cục", f"{system_entropy:.4f} bits")
    col_m2.metric("Năng Lượng Cực Tiểu (E_min)", f"{float(delta_E[sorted_indices[0]]):.3f}")

    # Gợi ý Bậc 3 - 5 - 6
    st.markdown("---")
    st.header("🎯 BỘ SỐ GỢI Ý ĐỌC TỪ ĐIỂM HÚT (ATTRACTOR SETS)")
    
    tab3, tab5, tab6 = st.tabs(["🔥 Gợi Ý Bậc 3", "⚡ Gợi Ý Bậc 5", "💎 Gợi Ý Bậc 6"])

    with tab3:
        st.subheader("📌 Bộ Số Bậc 3")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bậc 3 - Điểm hút 1", format_numbers(top_attractors[:3]))
        c2.metric("Bậc 3 - Điểm hút 2", format_numbers(top_attractors[3:6]))
        c3.metric("Bậc 3 - Cân bằng", format_numbers([top_attractors[0], top_attractors[1], top_attractors[6]]))

    with tab5:
        st.subheader("📌 Bộ Số Bậc 5")
        c1, c2 = st.columns(2)
        c1.metric("Bậc 5 - Cấu hình Năng lượng thấp nhất", format_numbers(top_attractors[:5]))
        c2.metric("Bậc 5 - Cấu hình Mở rộng", format_numbers(top_attractors[2:7]))

    with tab6:
        st.subheader("📌 Bộ Số Bậc 6")
        c1, c2 = st.columns(2)
        c1.metric("Bậc 6 - Điểm hút Chính", format_numbers(top_attractors[:6]))
        c2.metric("Bậc 6 - Điểm hút Phụ", format_numbers(top_attractors[1:7]))

else:
    st.info("👈 Vui lòng bấm chọn ĐỦ 20 SỐ cho cả 3 kỳ ở trên để hiển thị dự đoán Bậc 3, 5, 6.")
