import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

# CẤU HÌNH TRANG 
st.set_page_config(
    page_title="DA Face Recognition",
    page_icon="🎯",
    layout="wide"
)

#  CUSTOM CSS 
st.markdown("""
    <style>
    /* Tổng thể */
    .main { background-color: #f8f9fa; }
    
    /* Tiêu đề chính */
    .main-title {
        color: #1E3A8A;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }

    /* Thẻ kết quả (Card) */
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-left: 10px solid #3B82F6;
        margin-top: 20px;
    }

    /* Tên người được nhận diện */
    .name-text {
        color: #1E40AF;
        font-size: 24px;
        font-weight: bold;
    }

    /* Sidebar */
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR 
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1698/1698535.png", width=100) # Logo giả định
    st.title("Settings")
    st.info("Hệ thống sử dụng mạng CNN để nhận diện thành viên lớp DA0001.")
    st.markdown("---")
    st.write("**Hướng dẫn:**")
    st.write("1. Cho phép truy cập Camera.")
    st.write("2. Căn chỉnh mặt vào giữa khung hình.")
    st.write("3. Nhấn 'Take Photo'.")
    st.markdown("---")
    st.caption("Developed by FACE REC DA © 2026")

# XỬ LÝ MODEL 
FILE_ID = '1NsI5vwZaYedAin7xMbXW6lQO62uQTAu4' 
MODEL_PATH = 'models/face_reg2.h5'

@st.cache_resource
def load_ai_model():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 Đang tải mô hình từ Cloud (lần đầu)..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    return load_model(MODEL_PATH)

try:
    model = load_ai_model()
    st.sidebar.success("Model: Ready")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải model: {e}")
    model = None

#  GIAO DIỆN CHÍNH 
st.markdown("<h1 class='main-title'>DA FACE RECOGNITION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Công nghệ nhận diện khuôn mặt thời gian thực dựa trên Deep Learning</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.write("### 📸 Camera Input")
    img_file_buffer = st.camera_input("")

with col2:
    st.write("### 🎯 Kết Quả Phân Tích")
    if img_file_buffer is not None:
        if model is not None:
            # Tiền xử lý ảnh
            image = Image.open(img_file_buffer)
            img_resized = image.resize((200, 200)) 
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  

            with st.spinner("🕵️ Đang quét đặc trưng..."):
                prediction = model.predict(img_array)
                idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                

                CLASS_NAMES = [
                    'DANG NGUYEN PHUONG NGHI', 'HA PHUONG THAO', 'HOANG BAO TRAN', 
                    'HOANG BUI TRA MY', 'LE MINH TRIET', 'LE THAI BAO', 
                    'LE THI NHU QUYNH', 'LE TRAN QUY ANH', 'LE TRONG DAI', 
                    'NGUYEN BAO HAN', 'NGUYEN DONG HAI', 'NGUYEN HOANG BAO', 
                    'NGUYEN HUU TOAN', 'NGUYEN KHAC LUU VU', 'NGUYEN NGOC KHANH UYEN', 
                    'NGUYEN NGOC KIM TUYET', 'NGUYEN THI THANH HA', 'NGUYEN TRONG MINH', 
                    'PHAM LY BAO LAM', 'PHAM MAI PHUONG', 'THAI TUAN PHAT', 
                    'TRAN GIA HAN', 'TRAN MINH HOANG', 'TRAN NGOC THAO ANH', 
                    'TRINH THUY HANG'
                ]
                
                result_name = CLASS_NAMES[idx]


            if confidence > 50:
                st.balloons()
                st.markdown(f"""
                    <div class='result-card' style='border-left-color: #10B981;'>
                        <p style='margin-bottom:5px; color:#6B7280;'>SINH VIÊN:</p>
                        <p class='name-text'>{result_name}</p>
                        <p style='color:#059669; font-weight:600;'> Độ tin cậy: {confidence:.2f}%</p>
                        <p style='font-size: 0.9rem; color: #6B7280;'>Trạng thái: <b>Đã xác minh</b></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='result-card' style='border-left-color: #EF4444;'>
                        <p style='margin-bottom:5px; color:#6B7280;'>KẾT QUẢ:</p>
                        <p class='name-text' style='color:#EF4444;'>Không thể xác định</p>
                        <p style='color:#DC2626;'>⚠️ Độ tin cậy thấp ({confidence:.2f}%)</p>
                        <p style='font-size: 0.8rem;'>Vui lòng điều chỉnh ánh sáng và thử lại.</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Hệ thống chưa sẵn sàng. Vui lòng kiểm tra lại Model.")
    else:
        st.info("Vui lòng chụp ảnh để bắt đầu nhận diện.")

#  FOOTER 
st.markdown("---")
st.caption("Hệ thống được tối ưu hóa cho trình duyệt Chrome và Edge.")
