# EEG Grasp-and-Lift Event Detection App

## Deskripsi Singkat

Repositori ini berisi aplikasi sederhana berbasis **Streamlit** untuk mendemonstrasikan penggunaan **model Deep Learning (CNN 1D)** dalam mendeteksi **event grasp-and-lift** dari sinyal **EEG**.
Aplikasi ini menggunakan **data EEG yang telah melalui preprocessing dan segmentasi (epoch-level)**, bukan data mentah.

Aplikasi dikembangkan sebagai bagian dari tugas analisis dan pemodelan sinyal EEG, dengan fokus pada **inferensi model dan visualisasi hasil prediksi**, bukan pada pelatihan ulang model.

---

## Dataset

Dataset yang digunakan berasal dari kompetisi publik:

**Grasp-and-Lift EEG Detection (Kaggle)**

Karakteristik utama dataset:

* 12 subjek
* 32 channel EEG
* Sampling rate: 500 Hz
* Data direpresentasikan dalam bentuk **epoch**

  * 1 epoch = 1 detik sinyal EEG
  * Shape per epoch: `(500, 32)`

Event yang dideteksi (multilabel):

1. HandStart
2. FirstDigitTouch
3. BothStartLoadPhase
4. LiftOff
5. Replace
6. BothReleased

---

## Format Input Aplikasi

Aplikasi menerima input berupa file **`.npy`** dengan spesifikasi berikut:

* Berisi **seluruh data EEG test** yang telah dipreprocessing
* Sudah melalui:

  * Band-pass filtering
  * Segmentasi (sliding window)
  * Normalisasi
* Bentuk data:

  ```
  (N_epoch, 500, 32)
  ```

Setiap file `.npy` merepresentasikan satu subjek dan satu sesi, dengan banyak epoch di dalamnya.

---

## Fitur Aplikasi

Fitur utama yang tersedia:

1. **Pemilihan Epoch**

   * Pengguna dapat memilih index epoch tertentu untuk dianalisis.

2. **Visualisasi Sinyal EEG**

   * Menampilkan sinyal EEG dari beberapa channel pada epoch terpilih.

3. **Prediksi Event per Epoch**

   * Menampilkan probabilitas setiap event.
   * Menampilkan status event (aktif / tidak aktif) berdasarkan threshold.

4. **Interpretasi Event**

   * Memberikan penjelasan singkat mengenai makna setiap event yang terdeteksi.

5. **Prediksi Seluruh Epoch**

   * Menampilkan visualisasi timeline berwarna yang menunjukkan distribusi event sepanjang seluruh epoch.

---

## Arsitektur Model

Model yang digunakan adalah **CNN 1D** dengan karakteristik umum:

* Input: `(500, 32)`
* Output: 6 neuron dengan aktivasi sigmoid (multilabel classification)
* Threshold klasifikasi: `0.3` (hasil tuning pada tahap evaluasi)

Model disimpan dalam format:

```
cnn_eeg_grasp_lift.keras
```

---

## Cara Menjalankan Aplikasi

### 1. Persiapan Lingkungan

Pastikan Python sudah terpasang (disarankan Python 3.9+).

Install dependensi:

```bash
pip install -r requirements.txt
```

### 2. Struktur Folder Minimal

```
project/
│
├── app.py
├── model/
│   └── cnn_eeg_grasp_lift.keras
├── requirements.txt
```

### 3. Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka melalui browser secara otomatis.

---

## Catatan Penting

* Aplikasi ini **tidak melakukan preprocessing data mentah**.
* Input harus berupa data EEG yang sudah dipreprocessing sebelumnya.
* Aplikasi digunakan untuk **demonstrasi inferensi model**, bukan untuk evaluasi performa atau pelatihan ulang.
* Prediksi dilakukan pada level **epoch**, bukan sinyal EEG kontinyu.

---
