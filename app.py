import streamlit as st
import cv2
import numpy as np
from scipy.signal import wiener
import os

st.title("🖼️ Ứng dụng xử lý ảnh số")

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    st.subheader("Ảnh gốc")
    st.image(img_rgb, use_container_width=True)

    filter_type = st.selectbox(
        "Chọn phương pháp xử lý:",
        (
            "Mean Filter",
            "Median Filter",
            "Gaussian Filter",
            "Bilateral Filter",
            "Laplacian Sharpen",
            "Motion Blur",
            "Wiener Filter (Khôi phục ảnh mờ)",
            "AI Super Resolution (FSRCNN)"
        )
    )

    # -----------------------------
    if filter_type == "Mean Filter":

        k = st.slider("Kernel size",3,15,5,step=2)

        processed_img = cv2.blur(img_input,(k,k))

        st.image(cv2.cvtColor(processed_img,cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # -----------------------------
    elif filter_type == "Median Filter":

        k = st.slider("Kernel size",3,15,5,step=2)

        processed_img = cv2.medianBlur(img_input,k)

        st.image(cv2.cvtColor(processed_img,cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # -----------------------------
    elif filter_type == "Gaussian Filter":

        k = st.slider("Kernel size",3,15,5,step=2)

        processed_img = cv2.GaussianBlur(img_input,(k,k),0)

        st.image(cv2.cvtColor(processed_img,cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # -----------------------------
    elif filter_type == "Bilateral Filter":

        d = st.slider("Diameter",5,15,9)
        sigmaColor = st.slider("Sigma Color",10,150,75)
        sigmaSpace = st.slider("Sigma Space",10,150,75)

        processed_img = cv2.bilateralFilter(img_input,d,sigmaColor,sigmaSpace)

        st.image(cv2.cvtColor(processed_img,cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # -----------------------------
    elif filter_type == "Laplacian Sharpen":

        gray = cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray,cv2.CV_64F)

        sharp = gray - lap

        sharp = cv2.normalize(sharp,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

        st.image(sharp,channels="GRAY",use_container_width=True)

    # -----------------------------
    elif filter_type == "Motion Blur":

        length = st.slider("Chiều dài blur",5,100,30)
        angle = st.slider("Góc blur",0,180,0)

        psf = np.zeros((length,length))
        center = length//2

        cv2.line(psf,(0,center),(length-1,center),1,1)

        M = cv2.getRotationMatrix2D((center,center),angle,1)
        psf = cv2.warpAffine(psf,M,(length,length))

        psf = psf/np.sum(psf)

        blur = cv2.filter2D(img_input,-1,psf)

        st.image(cv2.cvtColor(blur,cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # -----------------------------
    elif filter_type == "Wiener Filter (Khôi phục ảnh mờ)":
        # Slider tham số
        length = st.slider("Chiều dài vệt mờ", 5, 100, 30)
        angle = st.slider("Góc mờ", 0, 180, 0)
        K = st.slider("Hệ số nhiễu K", 0.0001, 0.1, 0.01)

        # Tạo PSF
        psf = np.zeros((length, length), dtype=np.float32)
        center = length // 2

        cv2.line(psf, (0, center), (length-1, center), 1, 1)

        M = cv2.getRotationMatrix2D((center, center), angle, 1)
        psf = cv2.warpAffine(psf, M, (length, length))

        if np.sum(psf) != 0:
            psf = psf / np.sum(psf)

        # Chuyển ảnh xám
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
    
        # FFT ảnh
        img_fft = np.fft.fft2(gray)

        # Padding PSF
        psf_pad = np.zeros((h, w))
        psf_h, psf_w = psf.shape

        psf_pad[:psf_h, :psf_w] = psf

        psf_pad = np.roll(psf_pad, -psf_h//2, axis=0)
        psf_pad = np.roll(psf_pad, -psf_w//2, axis=1)

        # FFT PSF
        psf_fft = np.fft.fft2(psf_pad)
    
        # Wiener filter
        wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft)**2 + K + 1e-8)

        # Khôi phục ảnh
        result = np.real(np.fft.ifft2(img_fft * wiener_filter))

        # Chuẩn hóa ảnh
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        result = result.astype(np.uint8)

        # Hiển thị
        st.image(result, channels="GRAY", use_container_width=True)

    # -----------------------------
    elif filter_type == "AI Super Resolution (FSRCNN)":

        try:

            sr = cv2.dnn_superres.DnnSuperResImpl_create()

            model = "FSRCNN_x2.pb"

            if not os.path.exists(model):
                st.error("Thiếu file FSRCNN_x2.pb")
            else:

                sr.readModel(model)

                sr.setModel("fsrcnn",2)

                with st.spinner("AI đang xử lý..."):

                    result = sr.upsample(img_input)

                st.image(cv2.cvtColor(result,cv2.COLOR_BGR2RGB),
                         use_container_width=True)

        except Exception as e:

            st.error(e)
