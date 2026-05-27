import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

# CẤU HÌNH HỆ THỐNG 
FILE_ID = '1oikl54xNw6qa-p2nZoyBezuLznnSQxO9' 
MODEL_PATH = 'models/face_reg.h5'

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

@st.cache_resource
def load_our_model():
    if not os.path.exists('models'):
        os.makedirs('models')
   
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Đang tải mô hình từ Google Drive (chỉ tải lần đầu)...'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Lỗi khi tải model: {e}")
                return None
    
    return load_model(MODEL_PATH)

model = load_our_model()

# GIAO DIỆN STREAMLIT 
st.set_page_config(page_title="Nhận Diện Khuôn Mặt Lớp DA", page_icon="👤", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1E3A8A; text-align: center; font-family: 'Helvetica Neue', sans-serif; }
    p.subtitle { text-align: center; color: #6B7280; font-size: 18px; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>HỆ THỐNG NHẬN DIỆN KHUÔN MẶT CNN</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Bật camera và chụp ảnh để hệ thống nhận diện danh tính</p>", unsafe_allow_html=True)

st.write("---")
st.write("### 📸 Khu vực nhận diện")
img_file_buffer = st.camera_input("Bấm nút phía dưới để chụp ảnh")

if img_file_buffer is not None and model is not None:
    image = Image.open(img_file_buffer)
    img_resized = image.resize((200, 200)) 
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  

    with st.spinner("🧠 Hệ thống đang phân tích khuôn mặt..."):
        prediction = model.predict(img_array)
        idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
    st.write("---")
    st.write("### 🎯 Kết quả dự đoán:")
    
    if confidence > 50: 
        st.balloons()
        st.markdown(f"""
            <div style="background-color:#D1FAE5; padding:20px; border-radius:12px; border-left: 6px solid #10B981;">
                <h2 style="color:#065F46; margin:0; font-size: 24px;">👤 Thành viên: {CLASS_NAMES[idx]}</h2>
                <p style="color:#047857; margin:8px 0 0 0; font-size:18px;"><b>Độ tin cậy:</b> {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Không thể xác định rõ danh tính. Vui lòng chụp lại trong điều kiện ánh sáng tốt hơn.")