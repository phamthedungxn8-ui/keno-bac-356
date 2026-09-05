import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
from PIL import Image
import re
import plotly.express as px

# ---------------------------------------------------------
# Cấu hình giao diện Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keno Ising Model - Động Lực Học Phi Tuyến",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ KENO ISING MODEL - ĐỘNG LỰC HỌC PHI TUYẾN & BẬC 3, 5, 6")
st.markdown("*Chuyển đổi bài toán Keno từ Thống kê Tĩnh sang Hệ Cơ học Thống kê Phi tuyến (Attractor & Phase Transitions).*")
st.markdown("---")

# ---------------------------------------------------------
# 1. HÀM OCR TỐI ƯU CHO GIAO DIỆN BẢNG VÀNG / NỀN TRẮNG
# ---------------------------------------------------------
def extract_numbers_from_image(image_file):
    try:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Chuyển ảnh sang mức xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Phân ngưỡng Otsu tách chữ đen rõ nét trên nền trắng/vàng
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Cấu hình OCR Tesseract
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Lọc các số nguyên từ 1 đến 80
        raw_numbers = re.findall(r'\b\d{1,2}\b', text)
        
        valid_numbers = []
        for n in raw_numbers:
            num = int(n)
            if 1 <= num <= 80 and num not in valid_numbers:
                valid_numbers.append(num)
                
        valid_numbers.sort()
        return valid_numbers[:20]
    except Exception as e:
        return []

# ---------------------------------------------------------
# 2. LÕI THUẬT TOÁN: ISING SPIN GLASS & NONLINEAR DYNAMICS
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
        h = np.dot(weights, self.spins_history)
        return h

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
        system_entropy = np.mean(entropy_per_spin)
        return system_entropy, entropy_per_spin

# Hàm định dạng danh sách số nguyên hiển thị đẹp
def format_numbers(num_list):
    clean_nums = [int(x) for x in sorted(num_list)]
    return " - ".join([f"{x:02d}" for x in clean_nums])

# ---------------------------------------------------------
# 3. GIAO DIỆN NẠP DỮ LIỆU
# ---------------------------------------------------------
st.sidebar.header("📥 Nạp Dữ Liệu 3 Kỳ Keno")
st.sidebar.info("Tải lên 3 ảnh chụp kết quả Keno liên tiếp.")

uploaded_files = st.sidebar.file_uploader(
    "Chọn 3 ảnh Keno", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

draws_data = []

if uploaded_files:
    if len(uploaded_files) != 3:
        st.warning("⚠️ Vui lòng tải lên ĐÚNG 3 MẪU ẢNH của 3 kỳ quay liên tiếp!")
    else:
        st.subheader("📷 Trích xuất & Xác nhận Ma Trận Kỳ")
        cols = st.columns(3)
        
        for idx, file in enumerate(uploaded_files):
            with cols[idx]:
                st.image(file, caption=f"Kỳ {idx+1}", use_container_width=True)
                extracted = extract_numbers_from_image(file)
                
                numbers_str = st.text_input(
                    f"20 số Kỳ {idx+1}:",
                    value=", ".join(map(str, extracted)) if extracted else ""
                )
                
                parsed_nums = [int(x.strip()) for x in numbers_str.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 80]
                draws_data.append(parsed_nums)

# ---------------------------------------------------------
# 4. CHẠY MÔ HÌNH ĐỘNG LỰC HỌC PHI TUYẾN
# ---------------------------------------------------------
if len(draws_data) == 3 and all(len(d) > 0 for d in draws_data):
    st.markdown("---")
    st.success("✅ Đã khởi tạo thành công Không gian Trạng thái Spin từ dữ liệu 3 kỳ!")

    model = KenoIsingDynamics(draws_data)
    delta_E, h_field, J_matrix = model.compute_attractor_energies()
    system_entropy, spin_entropies = model.compute_entropy_state()

    sorted_indices = np.argsort(delta_E)
    top_attractors = [int(idx + 1) for idx in sorted_indices]

    # Hiển thị chỉ số
    st.header("🌀 Mức Độ Hỗn Loạn & Pha Trạng Thái (Phase State)")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Entropy Shannon Toàn Cục", f"{system_entropy:.4f} bits")
    
    if system_entropy < 0.65:
        phase_status = "🟢 Pha Điểm Hút (Attractor) - Độ ổn định cao"
    elif system_entropy < 0.85:
        phase_status = "🟡 Pha Chuyển Tiếp (Transition)"
    else:
        phase_status = "🔴 Pha Hỗn Loạn (Chaos) - Biên độ biến động rộng"
        
    col_m2.metric("Trạng Thái Pha Hệ Thống", phase_status)
    col_m3.metric("Năng Lượng Cực Tiểu (E_min)", f"{float(delta_E[sorted_indices[0]]):.3f}")

    # Gợi ý Bậc 3 - 5 - 6
    st.markdown("---")
    st.header("🎯 BỘ SỐ GỢI Ý ĐỌC TỪ ĐIỂM HÚT (ATTRACTOR SETS)")
    
    tab3, tab5, tab6 = st.tabs(["🔥 Gợi Ý Bậc 3", "⚡ Gợi Ý Bậc 5", "💎 Gợi Ý Bậc 6"])

    with tab3:
        st.subheader("📌 Bộ Số Bậc 3 (Năng lượng cực tiểu & Liên kết cặp tối ưu)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bậc 3 - Điểm hút 1", format_numbers(top_attractors[:3]))
        c2.metric("Bậc 3 - Điểm hút 2", format_numbers(top_attractors[3:6]))
        c3.metric("Bậc 3 - Cân bằng", format_numbers([top_attractors[0], top_attractors[1], top_attractors[6]]))

    with tab5:
        st.subheader("📌 Bộ Số Bậc 5 (Tương tác từ trường chéo)")
        c1, c2 = st.columns(2)
        c1.metric("Bậc 5 - Cấu hình Năng lượng thấp nhất", format_numbers(top_attractors[:5]))
        c2.metric("Bậc 5 - Cấu hình Mở rộng", format_numbers(top_attractors[2:7]))

    with tab6:
        st.subheader("📌 Bộ Số Bậc 6 (Cực tiểu hóa Hamiltonian)")
        c1, c2 = st.columns(2)
        c1.metric("Bậc 6 - Điểm hút Chính", format_numbers(top_attractors[:6]))
        c2.metric("Bậc 6 - Điểm hút Phụ", format_numbers(top_attractors[1:7]))

    # Trực quan hóa
    st.markdown("---")
    st.header("📊 Trực Quan Hóa Động Lực Học Hệ Thống")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Bản đồ Năng lượng Spin $E_i$ (80 Hạt)")
        df_energy = pd.DataFrame({'Số': [i + 1 for i in range(80)], 'Năng lượng E_i': delta_E})
        fig_energy = px.bar(df_energy, x='Số', y='Năng lượng E_i', color='Năng lượng E_i', color_continuous_scale='Viridis_r')
        st.plotly_chart(fig_energy, use_container_width=True)

    with col_v2:
        st.subheader("Ma Trận Liên Kết Tương Tác Cặp $J_{ij}$")
        fig_heatmap = px.imshow(J_matrix, x=[i + 1 for i in range(80)], y=[i + 1 for i in range(80)], color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    st.info("👈 Vui lòng tải đủ 3 ảnh Keno ở bảng bên trái để chạy mô hình Động lực học Ising.")
