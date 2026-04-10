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
    st.sidebar.header("🛠️ Giả lập nhiễu")
    st.sidebar.markdown("Thêm nhiễu vào ảnh gốc để kiểm tra hiệu quả của các bộ lọc.")
    
    noise_choice = st.sidebar.radio(
        "Chọn loại nhiễu:",
        ("Không thêm nhiễu", "Nhiễu Gaussian", "Nhiễu Muối Tiêu", "Nhiễu Chu Kỳ")
    )
    
    img_input = img_cv2.copy()
    
    if noise_choice == "Nhiễu Gaussian":
        # Thêm nhiễu phân bố chuẩn (Gaussian)
        noise = np.random.normal(0, 25, img_input.shape).astype(np.float32)
        img_input = cv2.add(img_input.astype(np.float32), noise)
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)
        
    elif noise_choice == "Nhiễu Muối Tiêu":
        # Thêm nhiễu Salt & Pepper (5% ảnh bị ảnh hưởng)
        prob = 0.05
        rnd = np.random.rand(*img_input.shape[:2])
        img_input[rnd < prob/2] = 0
        img_input[rnd > 1 - prob/2] = 255
        
    elif noise_choice == "Nhiễu Chu Kỳ":
        # Thêm nhiễu hình sin/cos tạo vằn sọc
        rows, cols = img_input.shape[:2]
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        noise = 30 * np.sin(2 * np.pi * X / 15 + 2 * np.pi * Y / 15)
        
        if len(img_input.shape) == 3:
            # Biến noise thành mảng 3 chiều (rows, cols, 1) để tự động khớp với ảnh màu 3 kênh
            noise = noise[:, :, np.newaxis] 
            
        # Dùng phép cộng của NumPy (+) thay cho cv2.add để tránh lỗi không khớp số kênh màu
        img_input = img_input.astype(np.float32) + noise.astype(np.float32)
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)

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
             "Non-Local Means (Khử nhiễu mạnh)", 
             "Wiener Filter (Khôi phục ảnh mờ)", 
             "Optimum Notch Filter (Nhiễu chu kỳ)"]
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

        elif filter_type == "Non-Local Means (Khử nhiễu mạnh)":
            h = st.slider("Cường độ lọc (h):", 1, 30, 10)
            
            processed_img = cv2.fastNlMeansDenoisingColored(img_input, None, h, h, 7, 21)
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_container_width=True)

        elif filter_type == "Wiener Filter (Khôi phục ảnh mờ)":
            st.markdown("---")
            st.header("🔬 Tùy chọn khôi phục ảnh mờ")
            
            restoration_mode = st.radio(
                "Chọn phương pháp:",
                ("Toán học truyền thống (Wiener Deconvolution)", "AI Deep Learning (FSRCNN)")
            )

            if restoration_mode == "Toán học truyền thống (Wiener Deconvolution)":
                st.info("💡 Wiener truyền thống phù hợp với ảnh chỉ mờ nhẹ và biết rõ hướng mờ.")
                
                # Chuyển ảnh sang kênh xám
                if len(img_input.shape) == 3:
                    gray_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = img_input

                length = st.slider("Chiều dài vệt mờ (pixel):", 1, 100, 30)
                angle = st.slider("Góc mờ (độ):", 0, 180, 0)
                K = st.slider("Hệ số nhiễu K:", 0.0001, 0.1, 0.01, format="%.4f", step=0.001)

                # Tạo ma trận PSF
                psf = np.zeros((length, length), dtype=np.float32)
                center = length // 2
                cv2.line(psf, (0, center), (length - 1, center), 1, 1)
                M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
                psf = cv2.warpAffine(psf, M, (length, length))
                psf = psf / np.sum(psf)

                # Giải chập Wiener
                img_fft = np.fft.fft2(gray_img)
                psf_padded = np.zeros_like(gray_img, dtype=np.float32)
                kh, kw = psf.shape
                psf_padded[:kh, :kw] = psf
                psf_padded = np.roll(psf_padded, -kh//2, axis=0)
                psf_padded = np.roll(psf_padded, -kw//2, axis=1)
                psf_fft = np.fft.fft2(psf_padded)
                
                wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + K)
                processed_img = np.real(np.fft.ifft2(img_fft * wiener_filter))
                processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                st.image(processed_img, use_container_width=True, channels="GRAY")

            elif restoration_mode == "AI Deep Learning (FSRCNN)":
                st.success("🤖 Sử dụng mô hình FSRCNN để làm sắc nét và tăng độ phân giải x2.")
                
                # Tích hợp AI Super Resolution từ OpenCV
                try:
                    # 1. Khởi tạo đối tượng Super Resolution
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    
                    # 2. Đọc file mô hình .pb đã upload lên GitHub
                    model_path = "FSRCNN_x2.pb"
                    sr.readModel(model_path)
                    
                    # 3. Thiết lập mô hình và tỷ lệ phóng (fsrcnn, x2)
                    sr.setModel("fsrcnn", 2)
                    
                    # 4. Chạy AI để xử lý ảnh màu
                    height, width = img_input.shape[:2]
                    if height * width > 1000000: # Ví dụ > 1 Megapixel
                        st.warning("⚠️ Ảnh quá lớn, AI đang tự động thu nhỏ ảnh trước khi xử lý để tránh lỗi server.")
                        img_input_small = cv2.resize(img_input, (0,0), fx=0.5, fy=0.5)
                        processed_img = sr.upsample(img_input_small)
                    else:
                        with st.spinner("🤖 AI đang tính toán và vẽ lại chi tiết..."):
                            processed_img = sr.upsample(img_input)
                    
                    # Chuyển BGR sang RGB để hiển thị màu gốc
                    display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    st.image(display_img, use_container_width=True)
                    st.caption(f"Ảnh kết quả AI có kích thước: {processed_img.shape[1]}x{processed_img.shape[0]}")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy mô hình AI: {e}")
                    st.warning("💡 Đảm bảo bạn đã upload file 'FSRCNN_x2.pb' lên cùng thư mục với app.py trên GitHub.")
            
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
            col_freq, col_res = st.columns(2)
            with col_freq:
                st.caption("Phổ Tần Số (Minh họa)")
                # Tính phổ tần số của ảnh xám để người dùng nhìn thấy lỗ đen
                f_gray = np.fft.fft2(gray_for_spectrum)
                fshift_gray = np.fft.fftshift(f_gray)
                magnitude_spectrum = 20 * np.log(np.abs(fshift_gray) + 1)
                magnitude_spectrum_filtered = magnitude_spectrum * mask
                mag_display = cv2.normalize(magnitude_spectrum_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                st.image(mag_display, use_container_width=True, channels="GRAY")
                
            with col_res:
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
