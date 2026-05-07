import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

# Thiết lập giao diện rộng
st.set_page_config(page_title="Hệ Thống Lọc Ảnh Nâng Cao", layout="wide")

st.title("Hệ Thống Lọc & Khôi Phục Ảnh Nâng Cao")

# Khu vực tải ảnh lên
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

    # --- BỘ TẠO NHIỄU (SIDEBAR) ---
    st.sidebar.header("🛠️ Giả lập nhiễu/mờ")
    st.sidebar.markdown("Biến dạng ảnh gốc để kiểm tra hiệu quả của các bộ lọc.")
    
    # 1. BỔ SUNG THÊM TÙY CHỌN "Mờ chuyển động" VÀO MENU
    noise_choice = st.sidebar.radio(
        "Chọn loại suy biến:",
        ("Không thêm nhiễu", "Nhiễu Gaussian", "Nhiễu Muối Tiêu", "Nhiễu Chu Kỳ", "Mờ chuyển động (Motion Blur)")
    )
    
    img_input = img_cv2.copy()
    
    if noise_choice == "Nhiễu Gaussian":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Nhiễu Gaussian:**")
        # Thêm thanh trượt để chỉnh cường độ nhiễu thay vì fix cứng số 25
        std = st.sidebar.slider("Cường độ nhiễu (Sigma):", 5, 100, 25)
        
        noise = np.random.normal(0, std, img_input.shape).astype(np.float32)
        img_input = cv2.add(img_input.astype(np.float32), noise)
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)
        
    elif noise_choice == "Nhiễu Muối Tiêu":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Nhiễu Muối Tiêu:**")
        # Thêm thanh trượt chỉnh mật độ hạt nhiễu (từ 1% đến 50%)
        prob = st.sidebar.slider("Mật độ nhiễu (Probability):", 0.01, 0.50, 0.05, step=0.01)
        
        rnd = np.random.rand(*img_input.shape[:2])
        img_input[rnd < prob/2] = 0
        img_input[rnd > 1 - prob/2] = 255
        
    elif noise_choice == "Nhiễu Chu Kỳ":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Nhiễu Chu Kỳ:**")
        # Thêm các thanh trượt để chỉnh độ đậm và khoảng cách các vân sọc
        amp = st.sidebar.slider("Biên độ nhiễu (Độ đậm):", 10, 100, 30)
        u0_noise = st.sidebar.slider("Tần số trục X (u0):", 1, 100, 15)
        v0_noise = st.sidebar.slider("Tần số trục Y (v0):", 1, 100, 15)
        
        rows, cols = img_input.shape[:2]
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Cập nhật công thức sử dụng biến từ thanh trượt
        noise = amp * np.sin(2 * np.pi * u0_noise * X / cols + 2 * np.pi * v0_noise * Y / rows)
        
        if len(img_input.shape) == 3:
            noise = noise[:, :, np.newaxis] 
            
        img_input = img_input.astype(np.float32) + noise.astype(np.float32)
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)

    # 2. THÊM KHỐI CODE XỬ LÝ LÀM MỜ CHUYỂN ĐỘNG
    elif noise_choice == "Mờ chuyển động (Motion Blur)":
        st.sidebar.markdown("---")
        st.sidebar.write("💡 **Cấu hình Vệt mờ:**")
        blur_length = st.sidebar.slider("Chiều dài vệt mờ (pixel):", 1, 100, 30)
        blur_angle = st.sidebar.slider("Góc mờ (độ):", 0, 180, 0)
        
        if blur_length > 1:
            # Tạo ma trận điểm suy biến (PSF) giống hệt lúc khôi phục
            psf = np.zeros((blur_length, blur_length), dtype=np.float32)
            center = blur_length // 2
            cv2.line(psf, (0, center), (blur_length - 1, center), 1, 1)
            M = cv2.getRotationMatrix2D((center, center), blur_angle, 1.0)
            psf = cv2.warpAffine(psf, M, (blur_length, blur_length))
            psf = psf / np.sum(psf)
            
            # Dùng cv2.filter2D để chập ma trận mờ vào ảnh
            img_input = cv2.filter2D(img_input, -1, psf)

    # Chia đôi màn hình để so sánh
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Ảnh Đầu Vào")
        if noise_choice != "Không thêm nhiễu":
            st.warning(f"Đang hiển thị ảnh đã thêm: {noise_choice}")
            
        # Hiển thị ảnh đầu vào (đã có nhiễu hoặc chưa)
        if len(img_input.shape) == 3:
            disp_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        else:
            disp_input = img_input
        st.image(disp_input, use_container_width=True)

    with col2:
        st.header("Ảnh Xử Lý")
        
        # Dropdown chọn bộ lọc
        filter_type = st.selectbox(
            "Chọn bộ lọc:", 
            ["Bilateral Filter (Khử nhiễu Gaussian)",
             "Median Filter (Trị nhiễu Muối Tiêu)",
             "Non-Local Means",
             "Optimum Notch Filter (Nhiễu chu kỳ)",
             "Wiener Filter (Khôi phục ảnh mờ)", 
             "AI FSRCNN"]
        )
        
        # --- ÁP DỤNG BỘ LỌC NGAY LẬP TỨC ---
        if filter_type == "Bilateral Filter (Khử nhiễu Gaussian)":
            d = st.slider("Đường kính pixel (d):", 1, 15, 9)
            sigma_color = st.slider("Sigma Color:", 10, 150, 75)
            sigma_space = st.slider("Sigma Space:", 10, 150, 75)
            
            processed_img = cv2.bilateralFilter(img_input, d, sigma_color, sigma_space)
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_container_width=True)

        elif filter_type == "Median Filter (Trị nhiễu Muối Tiêu)":
            # Kích thước ô xét duyệt (kernel) bắt buộc phải là số lẻ: 3, 5, 7, 9...
            ksize = st.slider("Kích thước vùng lọc (ksize):", 3, 15, 3, step=2)
            
            processed_img = cv2.medianBlur(img_input, ksize)
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_container_width=True)

        elif filter_type == "Non-Local Means":
            h = st.slider("Cường độ lọc (h):", 1, 30, 10)
            
            processed_img = cv2.fastNlMeansDenoisingColored(img_input, None, h, h, 7, 21)
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_container_width=True)

        elif filter_type == "Wiener Filter (Khôi phục ảnh mờ)":
            st.info("💡 Wiener Filter xử lý đa kênh. Khắc phục nhiễu viền bằng thuật toán Đệm viền phản xạ (Reflect Padding).")

            col_controls, col_psf = st.columns([2, 1])
            with col_controls:
                length = st.slider("Chiều dài vệt mờ (pixel):", 1, 100, 30)
                angle = st.slider("Góc mờ (độ):", 0, 180, 0)
                K = st.slider("Hệ số nhiễu K:", 0.0001, 0.1, 0.01, format="%.4f", step=0.001)

            # --- BƯỚC 1: TẠO MA TRẬN PSF ---
            psf = np.zeros((length, length), dtype=np.float32)
            center = length // 2
            cv2.line(psf, (0, center), (length - 1, center), 1, 1)
            M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            psf = cv2.warpAffine(psf, M, (length, length))
            psf = psf / (np.sum(psf) + 1e-8) # Tránh chia cho 0

            with col_psf:
                psf_display = cv2.resize(psf, (150, 150), interpolation=cv2.INTER_NEAREST)
                psf_display = cv2.normalize(psf_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # --- BƯỚC 2: HÀM GIẢI CHẬP TỐI ƯU CHO TỪNG KÊNH ---
            def process_wiener_channel(channel, psf, K):
                # Đệm viền phản xạ (Reflect) để triệt tiêu hiệu ứng sọc viền FFT
                pad_size = length
                ch_padded = cv2.copyMakeBorder(channel, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101)
                
                # FFT ảnh
                ch_fft = np.fft.fft2(ch_padded.astype(np.float32))
                
                # FFT PSF
                psf_padded = np.zeros_like(ch_padded, dtype=np.float32)
                kh, kw = psf.shape
                psf_padded[:kh, :kw] = psf
                psf_padded = np.roll(psf_padded, -kh//2, axis=0)
                psf_padded = np.roll(psf_padded, -kw//2, axis=1)
                psf_fft = np.fft.fft2(psf_padded)
                
                # Giải chập
                wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + K)
                processed_padded = np.real(np.fft.ifft2(ch_fft * wiener_filter))
                
                # Cắt bỏ phần viền ảo và trả về kích thước gốc
                processed = processed_padded[pad_size:-pad_size, pad_size:-pad_size]
                return np.clip(processed, 0, 255).astype(np.uint8)

            # --- BƯỚC 3: ÁP DỤNG LÊN 3 KÊNH MÀU ---
            if len(img_input.shape) == 3:
                b, g, r = cv2.split(img_input)
                # Xử lý đồng thời 3 kênh để không bị "bóng ma màu"
                res_b = process_wiener_channel(b, psf, K)
                res_g = process_wiener_channel(g, psf, K)
                res_r = process_wiener_channel(r, psf, K)
                
                processed_img = cv2.merge([res_b, res_g, res_r])
                display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            else:
                display_img = process_wiener_channel(img_input, psf, K)
            
            st.image(display_img, use_container_width=True)

        elif filter_type == "AI FSRCNN":
            st.success("🤖 Trí tuệ nhân tạo (FSRCNN) đang 'đoán' và vẽ lại chi tiết, đồng thời tăng kích thước ảnh x2.")
            
            try:
                # Khởi tạo và đọc mô hình AI
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                model_path = "FSRCNN_x2.pb"
                sr.readModel(model_path)
                sr.setModel("fsrcnn", 2)
                
                # Kiểm tra kích thước ảnh tránh tràn RAM trên Streamlit
                height, width = img_input.shape[:2]
    
                processed_img = sr.upsample(img_input)
                
                # Hiển thị ảnh
                display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                st.image(display_img, use_container_width=True)
                st.caption(f"Kích thước ảnh gốc: {width}x{height} ➡️ Kích thước AI nội suy: {processed_img.shape[1]}x{processed_img.shape[0]}")
                
            except Exception as e:
                st.error(f"❌ Không thể chạy AI. Bạn kiểm tra lại đã up file FSRCNN_x2.pb lên GitHub chưa nhé. Lỗi chi tiết: {e}")
            
        elif filter_type == "Optimum Notch Filter (Nhiễu chu kỳ)":
            # Để vẽ biểu đồ phổ tần số minh họa, ta chỉ cần dùng ảnh xám
            if len(img_input.shape) == 3:
                gray_for_spectrum = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            else:
                gray_for_spectrum = img_input
                
            rows, cols = gray_for_spectrum.shape
            crow, ccol = rows // 2, cols // 2
            
            st.write("Điều chỉnh tọa độ để 'đục lỗ' triệt tiêu nhiễu:")
            
            # Đã set sẵn tọa độ vàng (30, 53) làm mặc định cho ảnh 800x450
            u0 = st.slider("Tọa độ nhiễu theo trục dọc (u0):", -crow, crow, 30)
            v0 = st.slider("Tọa độ nhiễu theo trục ngang (v0):", -ccol, ccol, 53)
            d0 = st.slider("Bán kính che nhiễu (D0):", 1, 50, 15)
            
            # Tạo Mask chung
            mask = np.ones((rows, cols), np.uint8)
            cv2.circle(mask, (ccol + v0, crow + u0), d0, 0, -1)
            cv2.circle(mask, (ccol - v0, crow - u0), d0, 0, -1)
            
            # --- BẮT ĐẦU XỬ LÝ ẢNH MÀU ---
            if len(img_input.shape) == 3:
                # Tách 3 kênh màu (B, G, R)
                channels = cv2.split(img_input)
                processed_channels = []
                
                # Áp dụng Notch Filter cho TỪNG kênh
                for ch in channels:
                    f = np.fft.fft2(ch)
                    fshift = np.fft.fftshift(f)
                    fshift_filtered = fshift * mask # Áp lỗ đen
                    
                    f_ishift = np.fft.ifftshift(fshift_filtered)
                    img_back = np.fft.ifft2(f_ishift)
                    processed_channels.append(np.real(img_back))
                
                # Gộp 3 kênh lại thành ảnh màu
                merged_back = cv2.merge(processed_channels)
                
                # Chuẩn hóa để hiển thị
                processed_img = cv2.normalize(merged_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                
            else:
                # Xử lý nếu ảnh gốc vốn dĩ là ảnh xám
                f = np.fft.fft2(img_input)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * mask
                
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.real(np.fft.ifft2(f_ishift))
                processed_img = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                display_img = processed_img

            # Hiển thị kết quả
            col_res = st.columns(2)
            st.caption("Ảnh Đã Lọc")
            st.image(display_img, use_container_width=True)

        # PHẦN TẢI ẢNH VỀ
        st.markdown("---")
        
        if 'processed_img' in locals():
            if len(processed_img.shape) == 3:
                final_save = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            else:
                final_save = processed_img
            
            result_pil = Image.fromarray(final_save)
            buffer = io.BytesIO()
            result_pil.save(buffer, format="PNG")
            byte_im = buffer.getvalue()

            st.download_button(
                label="Tải ảnh về máy",
                data=byte_im,
                file_name="anh_da_khoi_phuc.png",
                mime="image/png",
                use_container_width=True
            )
