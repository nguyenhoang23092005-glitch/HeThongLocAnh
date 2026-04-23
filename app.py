import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

# ===================== HÀM HỖ TRỢ (HELPER FUNCTIONS) =====================

def detect_periodic_peaks(mag, num_peaks=4, exclude_radius=25, suppress_radius=18):
    """
    Phát hiện các đỉnh nhiễu mạnh trong phổ tần số.
    Trả về danh sách tọa độ tương đối so với tâm: [(u0, v0), ...]
    """
    rows, cols = mag.shape
    crow, ccol = rows // 2, cols // 2

    work = mag.copy().astype(np.float32)

    # Loại vùng trung tâm để không lấy DC component
    cv2.circle(work, (ccol, crow), exclude_radius, 0, -1)

    peaks = []
    for _ in range(num_peaks):
        idx = np.unravel_index(np.argmax(work), work.shape)
        if work[idx] <= 0:
            break

        r, c = idx
        u0 = r - crow
        v0 = c - ccol
        peaks.append((u0, v0))

        # Suppress vùng quanh đỉnh vừa tìm và đỉnh đối xứng
        cv2.circle(work, (c, r), suppress_radius, 0, -1)
        sr, sc = 2 * crow - r, 2 * ccol - c
        if 0 <= sr < rows and 0 <= sc < cols:
            cv2.circle(work, (sc, sr), suppress_radius, 0, -1)

    # Loại trùng đơn giản
    unique = []
    seen = set()
    for u, v in peaks:
        key = (int(u), int(v))
        if key not in seen and (-key[0], -key[1]) not in seen:
            unique.append((u, v))
            seen.add(key)

    return unique


def build_gaussian_notch_reject_mask(shape, centers, d0=15):
    """
    Mặt nạ Gaussian Notch Reject:
    H(u,v) = Π [1 - exp(-Dk^2 / (2D0^2))] [1 - exp(-Dk'^2 / (2D0^2))]
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    U, V = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    mask = np.ones((rows, cols), dtype=np.float32)

    for u0, v0 in centers:
        # notch tại (crow+u0, ccol+v0)
        d1_sq = (U - (crow + u0))**2 + (V - (ccol + v0))**2
        # notch đối xứng tại (crow-u0, ccol-v0)
        d2_sq = (U - (crow - u0))**2 + (V - (ccol - v0))**2

        H1 = 1.0 - np.exp(-d1_sq / (2.0 * (d0**2)))
        H2 = 1.0 - np.exp(-d2_sq / (2.0 * (d0**2)))

        mask *= (H1 * H2)

    return mask


# ===================== MAIN APP =====================

# Thiết lập giao diện rộng
st.set_page_config(page_title="Hệ Thống Lọc Ảnh Nâng Cao", layout="wide")

st.title("Hệ Thống Lọc & Khôi Phục Ảnh Nâng Cao")

# Khu vực tải ảnh lên
uploaded_file = st.file_uploader("Tải ảnh của bạn lên (JPG, PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    try:
        # 1. Đọc file thẳng từ bộ nhớ đệm thành mảng byte (Chống lỗi PIL)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # 2. Dùng OpenCV giải mã trực tiếp
        img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) 
        
        if img_cv2 is None:
            st.error("🚨 File tải lên bị hỏng dữ liệu hoặc không phải là ảnh hợp lệ.")
            st.stop()
            
    except Exception as e:
        st.error(f"Lỗi hệ thống khi đọc ảnh: {e}")
        st.stop()

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
             "Optimum Notch Filter (Nhiễu chu kỳ)",
             "Wiener Filter (Khôi phục ảnh mờ)", 
             "AI Deep Learning (Siêu nét FSRCNN)"]
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
            st.info("💡 Wiener truyền thống giải chập bằng Toán học. Dựa trên giả định biết trước hướng di chuyển của máy ảnh.")
            
            if len(img_input.shape) == 3:
                gray_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img_input

            col_controls, col_psf = st.columns([2, 1])
            
            with col_controls:
                st.write("**1. Xây dựng Hàm suy biến (PSF - H(u,v)):**")
                length = st.slider("Chiều dài vệt mờ (pixel):", 1, 100, 30)
                angle = st.slider("Góc mờ (độ):", 0, 180, 0)
                
                st.write("**2. Trọng số kìm hãm nhiễu:**")
                K = st.slider("Hệ số nhiễu K:", 0.0001, 0.1, 0.01, format="%.4f", step=0.001)

            # Tạo ma trận PSF
            psf = np.zeros((length, length), dtype=np.float32)
            center = length // 2
            cv2.line(psf, (0, center), (length - 1, center), 1, 1)
            M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            psf = cv2.warpAffine(psf, M, (length, length))
            psf = psf / np.sum(psf)

            with col_psf:
                st.write("**Ma trận PSF:**")
                psf_display = cv2.resize(psf, (150, 150), interpolation=cv2.INTER_NEAREST)
                psf_display = cv2.normalize(psf_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                st.image(psf_display, caption=f"Kích thước thật: {length}x{length}", use_container_width=False)

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

        elif filter_type == "AI Deep Learning (Siêu nét FSRCNN)":
            st.success("🤖 Trí tuệ nhân tạo (FSRCNN) đang 'đoán' và vẽ lại chi tiết, đồng thời tăng kích thước ảnh x2.")
            
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                model_path = "FSRCNN_x2.pb"
                sr.readModel(model_path)
                sr.setModel("fsrcnn", 2)
                
                height, width = img_input.shape[:2]
                if height * width > 1000000:
                    st.warning("⚠️ Ảnh khá lớn, AI đang tự động thu nhỏ một chút trước khi xử lý để chống sập web...")
                    img_input_small = cv2.resize(img_input, (0,0), fx=0.5, fy=0.5)
                    processed_img = sr.upsample(img_input_small)
                else:
                    with st.spinner("🤖 AI đang phân tích và tái tạo ảnh..."):
                        processed_img = sr.upsample(img_input)
                
                display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                st.image(display_img, use_container_width=True)
                st.caption(f"Kích thước ảnh gốc: {width}x{height} ➡️ Kích thước AI nội suy: {processed_img.shape[1]}x{processed_img.shape[0]}")
                
            except Exception as e:
                st.error(f"❌ Không thể chạy AI. Bạn kiểm tra lại đã up file FSRCNN_x2.pb lên GitHub chưa nhé. Lỗi chi tiết: {e}")

        elif filter_type == "Optimum Notch Filter (Nhiễu chu kỳ)":
            st.info("💡 Optimum Notch Filter tự động dò tìm và triệt tiêu các đỉnh nhiễu chu kỳ bằng mặt nạ Gaussian trên không gian màu YCrCb.")
            
            if len(img_input.shape) == 3:
                # Khuyến nghị lọc trên kênh sáng Y để giữ màu tốt hơn
                img_ycrcb = cv2.cvtColor(img_input, cv2.COLOR_BGR2YCrCb)
                y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)
                img_for_filter = y_channel
            else:
                img_for_filter = img_input

            rows, cols = img_for_filter.shape
            crow, ccol = rows // 2, cols // 2

            # FFT ảnh gốc
            f = np.fft.fft2(img_for_filter)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log1p(np.abs(fshift))

            st.write("**Điều chỉnh tham số cho Optimum Notch Filter:**")

            auto_detect = st.checkbox("Tự động phát hiện đỉnh nhiễu", value=True)

            if auto_detect:
                num_peaks = st.slider("Số đỉnh nhiễu cần triệt", 1, 10, 3)
                exclude_radius = st.slider("Bán kính bỏ qua vùng tâm", 5, min(rows, cols) // 4, 25)
                suppress_radius = st.slider("Bán kính triệt vùng lân cận đỉnh", 5, 50, 18)

                peaks = detect_periodic_peaks(
                    magnitude_spectrum,
                    num_peaks=num_peaks,
                    exclude_radius=exclude_radius,
                    suppress_radius=suppress_radius
                )
            else:
                # Nếu muốn nhập tay vẫn hỗ trợ
                u0 = st.slider("Tọa độ nhiễu theo trục dọc (u0):", -crow, crow, 30)
                v0 = st.slider("Tọa độ nhiễu theo trục ngang (v0):", -ccol, ccol, 53)
                peaks = [(u0, v0)]

            d0 = st.slider("Bán kính notch (D0):", 1, 80, 15)

            # Tạo mask notch
            mask = build_gaussian_notch_reject_mask((rows, cols), peaks, d0=d0)

            # Áp mask trong miền tần số
            fshift_filtered = fshift * mask

            # Biến đổi ngược
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.real(np.fft.ifft2(f_ishift))

            # Chuẩn hóa ảnh kết quả
            processed_gray = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Phục hồi ảnh màu nếu đầu vào là màu
            if len(img_input.shape) == 3:
                restored_ycrcb = cv2.merge([processed_gray, cr_channel, cb_channel])
                processed_img = cv2.cvtColor(restored_ycrcb, cv2.COLOR_YCrCb2BGR)
                display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            else:
                processed_img = processed_gray
                display_img = processed_gray

            # Phổ sau lọc để minh họa
            magnitude_spectrum_filtered = np.log1p(np.abs(fshift_filtered))
            mag_display = cv2.normalize(magnitude_spectrum_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Hiển thị kết quả
            col_freq, col_res = st.columns(2)

            with col_freq:
                st.caption("Phổ tần số sau khi áp Notch Filter")
                st.image(mag_display, use_container_width=True, channels="GRAY")

            with col_res:
                st.caption("Ảnh đã khôi phục")
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
