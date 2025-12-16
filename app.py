import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

# =========================================================
# KONFIGURASI MODEL
# =========================================================
MODEL_PATH = "model/cnn_eeg_grasp_lift.keras"
THRESHOLD = 0.3  # threshold hasil tuning pada tahap evaluasi

LABELS = [
    "HandStart",
    "FirstDigitTouch",
    "BothStartLoadPhase",
    "LiftOff",
    "Replace",
    "BothReleased",
]

EVENT_EXPLANATION = {
    "HandStart": "Menandakan awal gerakan tangan menuju objek.",
    "FirstDigitTouch": "Jari pertama mulai menyentuh objek.",
    "BothStartLoadPhase": "Kedua tangan mulai memberikan gaya untuk mengangkat objek.",
    "LiftOff": "Objek mulai terangkat dari permukaan.",
    "Replace": "Objek diletakkan kembali ke permukaan.",
    "BothReleased": "Objek dilepaskan sepenuhnya oleh tangan.",
}


# =========================================================
# LOAD MODEL (CACHE)
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# =========================================================
# UI SETUP
# =========================================================
st.set_page_config(page_title="EEG Grasp-and-Lift Event Detection", layout="wide")

st.title("EEG Grasp-and-Lift Event Detection")

st.markdown(
    """
    Aplikasi ini merupakan **demo sederhana** untuk mendeteksi
    **event grasp-and-lift** berdasarkan **sinyal EEG** menggunakan
    **model Deep Learning (CNN 1D)**.

    🔹 Model **tidak menggunakan data mentah**,
    🔹 tetapi **data EEG yang sudah dipreprocessing dan disegmentasi**.
    """
)

# =========================================================
# PENJELASAN SINGKAT (UNTUK ORANG AWAM)
# =========================================================
with st.expander("Penjelasan Singkat"):
    st.markdown(
        """
        **Apa itu EEG?**
        EEG (Electroencephalography) adalah sinyal listrik yang direkam dari otak
        menggunakan beberapa elektroda (channel).

        **Apa itu event grasp-and-lift?**
        Event ini merepresentasikan tahapan gerakan tangan saat
        mengambil (grasp) dan mengangkat (lift) sebuah objek.

        **Apa yang diprediksi aplikasi ini?**
        Aplikasi ini memprediksi **apakah suatu event terjadi atau tidak**
        pada satu segmen kecil sinyal EEG (epoch).
        """
    )

# =========================================================
# INPUT SECTION
# =========================================================
st.header("Input Data EEG")

st.markdown(
    """
    **Format input yang diterima:**
    - File **`.npy`**
    - Berisi **seluruh data EEG test**
    - Sudah melalui preprocessing & segmentasi
    - Bentuk data:
      `(N_epoch, 500, 32)`
    """
)

uploaded_file = st.file_uploader(
    "Upload file EEG (.npy) hasil preprocessing", type=["npy"]
)

if uploaded_file is None:
    st.info("⬆️ Silakan upload file .npy untuk memulai.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================
X_all = np.load(uploaded_file)

if X_all.ndim != 3 or X_all.shape[1:] != (500, 32):
    st.error(
        "❌ Format data tidak valid.\n\nData harus memiliki shape:\n(N_epoch, 500, 32)"
    )
    st.stop()

st.success(f"File berhasil dimuat. Total segmen EEG: {X_all.shape[0]} epoch")

# =========================================================
# PENJELASAN EPOCH
# =========================================================
with st.expander("Apa itu Epoch?"):
    st.markdown(
        """
        **Epoch** adalah potongan kecil sinyal EEG dengan durasi tertentu.

        Pada aplikasi ini:
        - 1 epoch = **1 detik sinyal EEG**
        - 500 sampel (sampling rate 500 Hz)
        - 32 channel EEG

        Model melakukan prediksi **per epoch**, bukan per sinyal panjang.
        """
    )

# =========================================================
# PILIH EPOCH
# =========================================================
st.header("Pilih Segmen EEG (Epoch)")

epoch_index = st.slider(
    "Pilih index epoch yang ingin dianalisis:",
    min_value=0,
    max_value=X_all.shape[0] - 1,
    value=0,
)

epoch = X_all[epoch_index]

# =========================================================
# VISUALISASI EEG + PREDIKSI (SAMPING-SAMPING)
# =========================================================
col_left, col_space, col_right = st.columns([1.2, 0.08, 1])

# =======================
# KOLOM KIRI: VISUALISASI EEG
# =======================
with col_left:
    st.subheader("Visualisasi Sinyal EEG (Epoch Terpilih)")

    fig, ax = plt.subplots(figsize=(14, 6))
    for ch in range(10):  # tetap 10 channel, isi tidak diubah
        ax.plot(epoch[:, ch] + ch * 5, label=f"Ch {ch + 1}")

    ax.set_title(f"Contoh Sinyal EEG (10 Channel), Epoch {epoch_index}")
    ax.set_xlabel("Waktu (sample)")
    ax.set_ylabel("Amplitudo (offset)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)

# =======================
# KOLOM KANAN: PREDIKSI & INTERPRETASI
# =======================
with col_right:
    st.header("Hasil Prediksi Model")

    X_input = epoch[np.newaxis, :, :]
    probs = model.predict(X_input, verbose=0)[0]

    # -----------------------
    # Probabilitas
    # -----------------------
    st.subheader("Probabilitas Event")

    st.markdown(
        "Nilai di bawah menunjukkan **tingkat keyakinan model** "
        "bahwa suatu event sedang terjadi pada epoch yang dipilih."
    )

    for label, p in zip(LABELS, probs):
        st.write(f"- **{label}** : `{p:.3f}`")

    # -----------------------
    # Klasifikasi + Penjelasan
    # -----------------------
st.subheader("Keputusan & Interpretasi Event")

st.markdown(
    f"""
    Keputusan klasifikasi diperoleh dengan **threshold = {THRESHOLD}**
    (ditentukan berdasarkan hasil evaluasi model (treshold tuning)).
    """
)

for label, p in zip(LABELS, probs):
    if p >= THRESHOLD:
        st.success(f"{label} AKTIF → {EVENT_EXPLANATION[label]}")
    else:
        st.write(f"{label}: tidak aktif")

# =========================================================
# PREDIKSI SELURUH EPOCH (TIMELINE)
# =========================================================
st.header("Prediksi Seluruh Epoch (Timeline Event)")

st.markdown(
    """
    Visualisasi berikut menunjukkan **kapan suatu event terdeteksi**
    sepanjang seluruh segmen EEG (berdasarkan index epoch).
    """
)

if st.button("Jalankan Prediksi Semua Epoch"):
    y_prob_all = model.predict(X_all, batch_size=64, verbose=0)
    y_pred_all = (y_prob_all >= THRESHOLD).astype(int)

    fig, ax = plt.subplots(figsize=(14, 4))

    for i, label in enumerate(LABELS):
        active_epochs = np.where(y_pred_all[:, i] == 1)[0]
        ax.scatter(active_epochs, np.full_like(active_epochs, i), s=10, label=label)

    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Index Epoch (waktu relatif)")
    ax.set_title("Timeline Deteksi Event EEG")
    ax.grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)

    st.info(
        "Setiap titik menunjukkan epoch di mana suatu event terdeteksi. "
        "Visualisasi ini membantu memahami urutan dan distribusi event "
        "sepanjang sinyal EEG."
    )

# =========================================================
# CATATAN AKHIR
# =========================================================
st.info(
    """
    **Catatan Penting:**
    - Prediksi dilakukan **per segmen EEG (epoch)**.
    - File `.npy` mencakup **seluruh data EEG test** yang sudah disegmentasi.
    - Aplikasi ini digunakan untuk **demonstrasi inferensi model**,
      bukan untuk evaluasi performa model.
    """
)
