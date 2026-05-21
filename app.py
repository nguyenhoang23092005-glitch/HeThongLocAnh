import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

# Thiết lập giao diện rộng tràn màn hình
st.set_page_config(page_title="Hệ Thống Lọc Ảnh Nâng Cao", layout="wide")

st.title("Hệ Thống Lọc & Khôi Phục Ảnh Nâng Cao")

# =========================================================
# KHU VỰC TẢI ẢNH VÀ XỬ LÝ ĐẦU VÀO
# =========================================================
uploaded_file = st.file_uploader("Tải ảnh của bạn lên (JPG, PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Đọc và chuyển đổi ảnh sang định dạng OpenCV (NumPy array)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # OpenCV dùng hệ màu BGR thay vì RGB
    if len(img_array.shape) == 3:
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv2 = img_array

    # =========================================================
    # SIDEBAR: BỘ TẠO NHIỄU / MỜ
    # =========================================================
    st.sidebar.header("🛠️ Giả lập nhiễu/mờ")
    st.sidebar.markdown("Biến dạng ảnh gốc để kiểm tra hiệu quả của các bộ lọc.")
    
    noise_choice = st.sidebar.radio(
        "Chọn loại suy biến:",
        ("Không thêm nhiễu", "Nhiễu Gaussian", "Nhiễu Muối Tiêu", "Nhiễu Chu Kỳ", "Mờ chuyển động (Motion Blur)")
    )
    
    img_input = img_cv2.copy()
    
    if noise_choice == "Nhiễu Gaussian":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Nhiễu Gaussian:**")
        std = st.sidebar.slider("Cường độ nhiễu (Sigma):", 5, 100, 25)
        
        noise = np.random.normal(0, std, img_input.shape).astype(np.float32)
        img_input = cv2.add(img_input.astype(np.float32), noise)
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)
        
    elif noise_choice == "Nhiễu Muối Tiêu":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Nhiễu Muối Tiêu:**")
        prob = st.sidebar.slider("Mật độ nhiễu (Probability):", 0.01, 0.50, 0.05, step=0.01)
        
        rnd = np.random.rand(*img_input.shape[:2])
        img_input[rnd < prob/2] = 0
        img_input[rnd > 1 - prob/2] = 255
        
    elif noise_choice == "Nhiễu Chu Kỳ":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Nhiễu Chu Kỳ:**")
        amp = st.sidebar.slider("Biên độ nhiễu (Độ đậm):", 10, 100, 30)
        u0_noise = st.sidebar.slider("Tần số trục X (u0):", 1, 100, 15)
        v0_noise = st.sidebar.slider("Tần số trục Y (v0):", 1, 100, 15)
        
        rows, cols = img_input.shape[:2]
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        noise = amp * np.sin(2 * np.pi * u0_noise * X / cols + 2 * np.pi * v0_noise * Y / rows)
        
        if len(img_input.shape) == 3:
            noise = noise[:, :, np.newaxis] 
            
        img_input = img_input.astype(np.float32) + noise.astype(np.float32)
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)

    elif noise_choice == "Mờ chuyển động (Motion Blur)":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Vệt mờ:**")
        blur_length = st.sidebar.slider("Chiều dài vệt mờ (pixel):", 1, 100, 30)
        blur_angle = st.sidebar.slider("Góc mờ (độ):", 0, 180, 0)
        
        if blur_length > 1:
            psf = np.zeros((blur_length, blur_length), dtype=np.float32)
            center = blur_length // 2
            cv2.line(psf, (0, center), (blur_length - 1, center), 1, 1)
            M = cv2.getRotationMatrix2D((center, center), blur_angle, 1.0)
            psf = cv2.warpAffine(psf, M, (blur_length, blur_length))
            psf = psf / np.sum(psf)
            img_input = cv2.filter2D(img_input, -1, psf)

    # =========================================================
    # KHU VỰC 1: ĐIỀU KHIỂN BỘ LỌC (NẰM Ở TRÊN CÙNG)
    # =========================================================
    st.markdown("---")
    st.header("⚙️ Cấu hình Bộ Lọc Khôi Phục")
    
    filter_type = st.selectbox(
        "Chọn bộ lọc:", 
        ["Bilateral Filter (Khử nhiễu Gaussian)",
         "Median Filter (Trị nhiễu Muối Tiêu)",
         "Non-Local Means",
         "Optimum Notch Filter (Nhiễu chu kỳ)",
         "Wiener Filter (Khôi phục ảnh mờ)", 
         "AI FSRCNN"]
    )
    
    # Khởi tạo biến lưu ảnh kết quả
    processed_img = None
    display_img = None

    if filter_type == "Bilateral Filter (Khử nhiễu Gaussian)":
        col_p1, col_p2, col_p3 = st.columns(3)
        d = col_p1.slider("Đường kính pixel (d):", 1, 15, 9)
        sigma_color = col_p2.slider("Sigma Color:", 10, 150, 75)
        sigma_space = col_p3.slider("Sigma Space:", 10, 150, 75)
        
        processed_img = cv2.bilateralFilter(img_input, d, sigma_color, sigma_space)
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    elif filter_type == "Median Filter (Trị nhiễu Muối Tiêu)":
        ksize = st.slider("Kích thước vùng lọc (ksize):", 3, 15, 3, step=2)
        
        processed_img = cv2.medianBlur(img_input, ksize)
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    elif filter_type == "Non-Local Means":
        h = st.slider("Cường độ lọc (h):", 1, 30, 10)
        
        processed_img = cv2.fastNlMeansDenoisingColored(img_input, None, h, h, 7, 21)
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    elif filter_type == "Wiener Filter (Khôi phục ảnh mờ)":
        st.info("💡 Wiener Filter xử lý đa kênh. Khắc phục nhiễu viền bằng thuật toán Đệm viền phản xạ.")

        col_controls, col_psf = st.columns([2, 1])
        with col_controls:
            length = st.slider("Chiều dài vệt mờ khôi phục (pixel):", 1, 100, 30)
            angle = st.slider("Góc mờ khôi phục (độ):", 0, 180, 0)
            K = st.slider("Hệ số nhiễu K:", 0.0001, 0.1, 0.01, format="%.4f", step=0.001)

        psf = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        cv2.line(psf, (0, center), (length - 1, center), 1, 1)
        M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        psf = cv2.warpAffine(psf, M, (length, length))
        psf = psf / (np.sum(psf) + 1e-8)

        with col_psf:
            psf_display = cv2.resize(psf, (150, 150), interpolation=cv2.INTER_NEAREST)
            psf_display = cv2.normalize(psf_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(psf_display, caption="Ma trận suy biến PSF", use_container_width=False)

        def process_wiener_channel(channel, psf, K):
            pad_size = length
            ch_padded = cv2.copyMakeBorder(channel, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101)
            ch_fft = np.fft.fft2(ch_padded.astype(np.float32))
            
            psf_padded = np.zeros_like(ch_padded, dtype=np.float32)
            kh, kw = psf.shape
            psf_padded[:kh, :kw] = psf
            psf_padded = np.roll(psf_padded, -kh//2, axis=0)
            psf_padded = np.roll(psf_padded, -kw//2, axis=1)
            psf_fft = np.fft.fft2(psf_padded)
            
            wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + K)
            processed_padded = np.real(np.fft.ifft2(ch_fft * wiener_filter))
            
            processed = processed_padded[pad_size:-pad_size, pad_size:-pad_size]
            return np.clip(processed, 0, 255).astype(np.uint8)

        if len(img_input.shape) == 3:
            b, g, r = cv2.split(img_input)
            res_b = process_wiener_channel(b, psf, K)
            res_g = process_wiener_channel(g, psf, K)
            res_r = process_wiener_channel(r, psf, K)
            processed_img = cv2.merge([res_b, res_g, res_r])
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        else:
            processed_img = process_wiener_channel(img_input, psf, K)
            display_img = processed_img

    elif filter_type == "AI FSRCNN":
        st.success("🤖 Trí tuệ nhân tạo (FSRCNN) đang nội suy nâng cấp kích thước ảnh x2.")
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "FSRCNN_x2.pb"
            sr.readModel(model_path)
            sr.setModel("fsrcnn", 2)
            
            height, width = img_input.shape[:2]
            processed_img = sr.upsample(img_input)
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.caption(f"Kích thước ảnh gốc: {width}x{height} ➡️ Kích thước AI nội suy: {processed_img.shape[1]}x{processed_img.shape[0]}")
        except Exception as e:
            st.error(f"❌ Không thể chạy AI. Lỗi chi tiết: {e}")

    elif filter_type == "Optimum Notch Filter (Nhiễu chu kỳ)":
        if len(img_input.shape) == 3:
            gray_for_spectrum = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_spectrum = img_input
            
        rows, cols = gray_for_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        
        st.write("Điều chỉnh tọa độ để 'đục lỗ' triệt tiêu nhiễu:")
        col_n1, col_n2, col_n3 = st.columns(3)
        u0 = col_n1.slider("Tọa độ nhiễu theo trục dọc (u0):", -crow, crow, 30)
        v0 = col_n2.slider("Tọa độ nhiễu theo trục ngang (v0):", -ccol, ccol, 53)
        d0 = col_n3.slider("Bán kính che nhiễu (D0):", 1, 50, 15)
        
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol + v0, crow + u0), d0, 0, -1)
        cv2.circle(mask, (ccol - v0, crow - u0), d0, 0, -1)
        
        if len(img_input.shape) == 3:
            channels = cv2.split(img_input)
            processed_channels = []
            for ch in channels:
                f = np.fft.fft2(ch)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                processed_channels.append(np.real(img_back))
            
            merged_back = cv2.merge(processed_channels)
            processed_img = cv2.normalize(merged_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        else:
            f = np.fft.fft2(img_input)
            fshift = np.fft.fftshift(f)
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.real(np.fft.ifft2(f_ishift))
            processed_img = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display_img = processed_img

    # =========================================================
    # KHU VỰC 2: HIỂN THỊ HÌNH ẢNH SONG SONG CỐ ĐỊNH
    # =========================================================
    st.markdown("---")
    st.subheader("📊 KẾT QUẢ SO SÁNH TRỰC QUAN")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("<h4 style='text-align: center; color: #64748b;'>Ảnh Đầu Vào</h4>", unsafe_allow_html=True)
            
        if len(img_input.shape) == 3:
            disp_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        else:
            disp_input = img_input
        st.image(disp_input, use_container_width=True)

    with col_img2:
        st.markdown("<h4 style='text-align: center; color: #16a34a;'>Ảnh Đã Xử Lý</h4>", unsafe_allow_html=True)
        if display_img is not None:
            st.image(display_img, use_container_width=True)
        else:
            st.info("👆 Hãy thiết lập thông số bộ lọc ở phía trên để xem kết quả.")

    # =========================================================
    # KHU VỰC 3: NÚT TẢI XUỐNG
    # =========================================================
    if processed_img is not None:
        st.markdown("---")
        if len(processed_img.shape) == 3:
            final_save = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        else:
            final_save = processed_img
        
        result_pil = Image.fromarray(final_save)
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        byte_im = buffer.getvalue()

        st.download_button(
            label="💾 TẢI HÌNH ẢNH ĐÃ KHÔI PHỤC VỀ MÁY",
            data=byte_im,
            file_name="anh_da_khoi_phuc.png",
            mime="image/png",
            use_container_width=True
        )
