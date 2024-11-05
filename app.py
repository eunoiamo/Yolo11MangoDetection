import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Fungsi untuk memuat model YOLO
@st.cache_resource
def load_yolo_model(model_path="best.pt"):
    model = YOLO(model_path)  # Memuat model YOLOv8
    return model

# Tampilan utama aplikasi
st.title("YOLOv8 Image Classification App")
st.write("Unggah gambar untuk mendeteksi dan mengklasifikasikan objek dengan model YOLOv8.")

# Memuat model
model_path = "best.pt"  # Ganti dengan path model Anda
model = load_yolo_model(model_path)

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Konversi gambar yang diunggah ke format PIL
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("Memprediksi...")

    # Konversi gambar ke format numpy untuk YOLO
    img_np = np.array(image)

    # Jalankan prediksi dengan YOLOv8
    results = model.predict(img_np)

    # Ambil gambar hasil prediksi
    result_img = results[0].plot()  # Menampilkan gambar dengan bounding boxes

    # Tampilkan hasil gambar dengan bounding boxes dan label
    st.image(result_img, caption="Hasil Klasifikasi", use_column_width=True)
