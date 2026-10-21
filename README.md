# UTS Machine Learning – Linear Regression & Logistic Regression

- **Nama** : Walmar Wildan
- **NIM** : 241011402756
- **Kelas** : Reg B
- **random_state** : 56 (dua digit terakhir NIM)

## Isi Repositori
| File | Keterangan |
|---|---|
| `tugas_UTS.ipynb` | Notebook jawaban lengkap, sudah dijalankan (semua output dan grafik tampil) |
| `Ads_Data.csv` | Dataset biaya iklan vs penjualan (40 data) |
| `Email_Data.csv` | Dataset email spam (60 data) |

Semua model dan metrik (`train_test_split`, `LinearRegression`, `LinearRegressionGD`, `LogisticRegression`, `accuracy_score`, `confusion_matrix`, MAE/MSE/RMSE/R², precision/recall/F1) diimplementasikan sendiri dengan NumPy. scikit-learn hanya dipakai pada soal bonus.

## Ringkasan Hasil

### Bagian A – Linear Regression (Ads_Data, 32 train / 8 test)
- Persamaan (normal equation): **ŷ = 28.5405 + 8.9851 · BiayaIklan**
  - Setiap kenaikan biaya iklan 1 juta Rp, penjualan naik rata-rata sekitar 8,99 juta Rp.

| | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Training | 5.1345 | 43.2469 | 6.5762 | 0.9276 |
| Testing | 2.2890 | 8.2377 | 2.8701 | 0.9873 |

- RMSE mengukur besarnya kesalahan prediksi dalam juta Rp (lebih sensitif terhadap kesalahan besar). R² testing lebih tinggi daripada R² training, tetapi dengan hanya 8 data testing hal ini tidak membuktikan model bebas dari overfitting.
- Gradient descent (5000 iterasi): lr = 0.001 terlalu lambat (b = 25.72, m = 9.44), lr = 0.02 konvergen sama dengan normal equation (b = 28.5405, m = 8.9851), lr = 0.1 divergen (overflow pada iterasi ke-214, parameter NaN). Batas learning rate stabil sekitar 0,0326.
- Prediksi: 4,5 juta → 68,97 juta; 7 juta → 91,44 juta (di dalam rentang data, tetapi tidak otomatis akurat); 12 juta → 136,36 juta (ekstrapolasi karena di luar rentang 0,8–9,9 juta, kurang dapat dipercaya).

### Bagian B – Logistic Regression (Email_Data, 45 train / 15 test)
- Model: **P(Spam) = σ(−9.0614 + 0.8171 · JumlahLink + 0.8041 · KataPromo)**
- Log loss training: 0,6931 → 0,2890 (konvergen).
- Evaluasi data testing (threshold 0,5): accuracy **0.9333**, precision **0.8000**, recall **1.0000**, F1 **0.8889**; confusion matrix TN = 10, FP = 1, FN = 0, TP = 4.
- Decision boundary: KataPromo = 11.2693 − 1.0162 · JumlahLink. Satu email salah klasifikasi (JumlahLink = 3, KataPromo = 10).

| Threshold | FP | FN | Precision | Recall |
|---|---|---|---|---|
| 0.3 | 3 | 0 | 0.5714 | 1.0 |
| 0.5 | 1 | 0 | 0.8000 | 1.0 |
| 0.7 | 1 | 0 | 0.8000 | 1.0 |

- Threshold pilihan: **0,7** (FP lebih merugikan pengguna daripada FN). Pada data testing yang tersedia, hasilnya identik dengan threshold 0,5, jadi keunggulannya belum terlihat dan pilihan ini berdasarkan pertimbangan biaya kesalahan.
- Prediksi email baru (threshold 0,7): E1 → Bukan Spam (P = 0.0013), E2 → Bukan Spam (P = 0.4622), E3 → Spam (P = 0.9982), E4 → Spam (P = 0.8050).

### Bagian C – Pemahaman Konsep
Dijawab lengkap di notebook (sel Jawaban C1–C5).

### Bonus – Perbandingan dengan scikit-learn
- Linear Regression: sama persis dengan implementasi sendiri (intercept 28.540483, koefisien 8.985056).
- Logistic Regression: koefisien sklearn default berbeda (b = −8.3354, m1 = 0.7485, m2 = 0.7396) karena sklearn memakai regularisasi L2 dengan C = 1,0 secara default. Dengan `C=np.inf` (tanpa penalti) hasilnya mendekati implementasi sendiri. Metrik pada data testing sama untuk semua model.
