import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import re

# ---------------------------------------------------------
# Cấu hình giao diện Streamlit Tối ưu Mobile Compact
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keno Ising Model - Compact Pad",
    page_icon="⚛️",
    layout="wide"
)

# CSS Tùy chỉnh ép siêu nhỏ lề và nút bấm để vừa 1 màn hình
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    div[data-testid="column"] {
        padding: 0px 1px !important;
    }
    div.stButton > button {
        height: 2.2rem !important;
        padding: 0px !important;
        font-size: 13px !important;
        font-weight: bold !important;
        margin: 1px 0px !important;
        border-radius: 4px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚛️ KENO ISING MODEL - COMPACT MOBILE")
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

def parse_text_input(key_name, text):
    raw_nums = re.findall(r'\b\d{1,2}\b', text)
    valid = []
    for n in raw_nums:
        val = int(n)
        if 1 <= val <= 80 and val not in valid:
            valid.append(val)
    st.session_state[key_name] = sorted(valid[:20])

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
# 2. GIAO DIỆN COMPACT PAD (NHỎ GỌN TRONG 1 MÀN HÌNH)
# ---------------------------------------------------------
st.subheader("🎮 Nhập 20 Số Cho 3 Kỳ")

tab_k1, tab_k2, tab_k3 = st.tabs(["📌 Kỳ 1", "📌 Kỳ 2", "📌 Kỳ 3"])
tabs = [tab_k1, tab_k2, tab_k3]

for idx, tab in enumerate(tabs):
    k_num = idx + 1
    key_name = f'selected_k{k_num}'
    
    with tab:
        curr_selected = st.session_state[key_name]
        
        # Ô Dán Nhanh Text / Hiển thị số đã chọn
        col_txt, col_clr = st.columns([4, 1])
        with col_txt:
            raw_text = st.text_input(
                f"Kỳ {k_num} ({len(curr_selected)}/20 số):",
                value=" ".join([f"{x:02d}" for x in sorted(curr_selected)]),
                key=f"txt_k{k_num}",
                placeholder="Dán hoặc gõ chuỗi số vào đây..."
            )
            # Tự đồng bộ nếu người dùng dán text
            parsed = [int(x) for x in re.findall(r'\b\d{1,2}\b', raw_text) if 1 <= int(x) <= 80]
            if sorted(parsed[:20]) != sorted(curr_selected):
                st.session_state[key_name] = sorted(list(set(parsed[:20])))
                st.rerun()

        with col_clr:
            st.write("") # Căn dòng
            if st.button(f"🗑️ Xóa", key=f"btn_clear_{k_num}", use_container_width=True):
                clear_selected(key_name)
                st.rerun()

        # MA TRẬN 10 CỘT X 8 HÀNG - ÉP SIÊU GỌN
        cols_per_row = 10
        for row in range(8):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                num = row * cols_per_row + col + 1
                is_selected = num in curr_selected
                
                btn_label = f"✓{num:02d}" if is_selected else f"{num:02d}"
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
    st.success("✅ Đã đủ 20/20 số cho 3 kỳ! Đang tính toán...")

    model = KenoIsingDynamics(draws_data)
    delta_E, h_field, J_matrix = model.compute_attractor_energies()
    system_entropy = model.compute_entropy_state()

    sorted_indices = np.argsort(delta_E)
    top_attractors = [int(idx + 1) for idx in sorted_indices]

    st.header("🌀 Trạng Thái Hệ Thống & Dự Đoán")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Entropy Shannon", f"{system_entropy:.4f} bits")
    col_m2.metric("Năng Lượng Cực Tiểu (E_min)", f"{float(delta_E[sorted_indices[0]]):.3f}")

    st.markdown("---")
    st.header("🎯 BỘ SỐ GỢI Ý (ATTRACTOR SETS)")
    
    tab3, tab5, tab6 = st.tabs(["🔥 Bậc 3", "⚡ Bậc 5", "💎 Bậc 6"])

    with tab3:
        c1, c2, c3 = st.columns(3)
        c1.metric("Bậc 3 - ĐH 1", format_numbers(top_attractors[:3]))
        c2.metric("Bậc 3 - ĐH 2", format_numbers(top_attractors[3:6]))
        c3.metric("Bậc 3 - CB", format_numbers([top_attractors[0], top_attractors[1], top_attractors[6]]))

    with tab5:
        c1, c2 = st.columns(2)
        c1.metric("Bậc 5 - Chuẩn", format_numbers(top_attractors[:5]))
        c2.metric("Bậc 5 - Mở rộng", format_numbers(top_attractors[2:7]))

    with tab6:
        c1, c2 = st.columns(2)
        c1.metric("Bậc 6 - Chính", format_numbers(top_attractors[:6]))
        c2.metric("Bậc 6 - Phụ", format_numbers(top_attractors[1:7]))

else:
    st.info("👈 Bấm chọn hoặc dán đủ 20 số cho cả 3 kỳ để xem gợi ý Bậc 3, 5, 6.")
