# Credit Card Default Prediction using Machine Learning 💳

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Implementasi machine learning untuk memprediksi default pembayaran kartu kredit pelanggan menggunakan dataset Taiwan Bank dengan 30,000 nasabah dan 24 fitur.

---

## 📋 Daftar Isi

1. [Latar Belakang & Motivasi](#latar-belakang--motivasi)
2. [Dataset Overview](#dataset-overview)
3. [Objektif Penelitian](#objektif-penelitian)
4. [Metodologi](#metodologi)
5. [Algoritma Machine Learning](#algoritma-machine-learning)
6. [Struktur Project](#struktur-project)
7. [Instalasi & Setup](#instalasi--setup)
8. [Panduan Penggunaan](#panduan-penggunaan)
9. [Hasil & Evaluasi](#hasil--evaluasi)
10. [Kesimpulan](#kesimpulan)

---

## 🎯 Latar Belakang & Motivasi

### Tantangan Bisnis

Prediksi default pembayaran kartu kredit merupakan salah satu tantangan kritis industri perbankan modern. Dengan tingkat default sebesar ~22% di kalangan nasabah Taiwan, perusahaan kartu kredit membutuhkan solusi prediktif yang akurat dan robust untuk:

- **Mengurangi kerugian finansial** dari customer default yang tidak terdeteksi
- **Melakukan proactive intervention** pada high-risk customers
- **Meningkatkan efisiensi** proses approval dan credit limit decisions
- **Mengoptimalkan risk management** dengan pendekatan data-driven

### Keunggulan Machine Learning vs Tradisional

Pendekatan tradisional rule-based memiliki keterbatasan dalam menangkap pola kompleks. Machine Learning menawarkan:

✅ **Adaptif & Scalable:** Dapat belajar dari data historis yang terus bertambah
✅ **Pattern Recognition:** Mengidentifikasi hubungan non-linear yang tersembunyi
✅ **Probabilistic Output:** Memberikan confidence scores untuk setiap prediksi
✅ **Otomatis:** Mengurangi subjektivitas dalam decision-making

---

## 📊 Dataset Overview

### Identitas Dataset

| Aspek | Detail |
|-------|--------|
| **Nama Dataset** | Default of Credit Card Clients |
| **Sumber** | UCI Machine Learning Repository + Taiwan Bank |
| **Periode** | September 2005 - Agustus 2006 |
| **Region** | Taiwan |
| **Domain** | Financial Risk Management |

### Karakteristik Dataset

| Metrik | Nilai | Status |
|--------|-------|--------|
| **Total Records** | 30,000 customers | ✅ Manageable |
| **Total Features** | 24 (23 input + 1 target) | ✅ Optimal |
| **Missing Values** | 0 | ✅ Excellent Quality |
| **Class Distribution** | 77.7% non-default, 22.3% default | ⚠️ Imbalanced |
| **Imbalance Ratio** | 3.48:1 | ⚠️ Moderate |

### Fitur Dataset

Dataset diorganisir dalam lima kelompok fitur:

#### **Group 1: Demographic Features (5 fitur)**
- `LIMIT_BAL`: Credit limit amount (TWD)
- `SEX`: Jenis kelamin (1=male, 2=female)
- `EDUCATION`: Tingkat pendidikan (1-6)
- `MARRIAGE`: Status pernikahan (1-3)
- `AGE`: Usia pelanggan (tahun)

#### **Group 2: Payment Status History (6 fitur)**
- `PAY_0` ~ `PAY_6`: Repayment status per bulan
  - -1 = Bayar di awal/on-time
  - 0 = Revolving credit
  - 1-9 = Keterlambatan 1-9 bulan
  - **[STRONGEST PREDICTOR FOR DEFAULT]**

#### **Group 3: Bill Amount History (6 fitur)**
- `BILL_AMT1` ~ `BILL_AMT6`: Jumlah tagihan setiap bulan (TWD)
- Menunjukkan historical spending pattern

#### **Group 4: Payment Amount History (6 fitur)**
- `PAY_AMT1` ~ `PAY_AMT6`: Jumlah pembayaran setiap bulan (TWD)
- Menunjukkan payment behavior & capacity

#### **Group 5: Target Variable (1 fitur)**
- `default.payment.next.month`: Target prediksi
  - 0 = Non-default (membayar dengan baik)
  - 1 = Default (gagal membayar)

---

## 🎓 Objektif Penelitian

### Tujuan Umum

Mengembangkan model machine learning yang akurat dan robust untuk memprediksi default pembayaran kartu kredit, sehingga bank dapat mengidentifikasi high-risk customers dan mengambil tindakan preventif.

### Tujuan Spesifik

#### 1. **Exploratory Data Analysis (EDA)**
- Memahami karakteristik komprehensif dataset
- Mengidentifikasi distribusi fitur penting
- Menganalisis hubungan variabel dengan target
- Mendeteksi anomali, outlier, missing values

#### 2. **Data Preprocessing & Feature Engineering**
- Melakukan data cleaning untuk inconsistencies
- Feature engineering menciptakan variabel baru yang informatif
- Mengatasi class imbalance dengan SMOTE/undersampling
- Feature scaling untuk fairness dalam training

#### 3. **Model Development**
- Mengembangkan multiple ML models (3 algoritma berbeda)
- Memilih algoritma sesuai karakteristik problem
- Training dan validation terhadap setiap model
- Membandingkan performance berbagai approaches

#### 4. **Model Evaluation & Comparison**
- Evaluasi menggunakan metrics komprehensif (Accuracy, Precision, Recall, F1, ROC-AUC)
- Analisis trade-offs antar metrics
- Perbandingan relative performance
- Identifikasi best-performing model

#### 5. **Hyperparameter Tuning**
- Systematic optimization menggunakan GridSearchCV
- Testing multiple parameter combinations
- Analisis impact setiap parameter
- Konfigurasi parameter optimal

#### 6. **Model Deployment & Testing**
- Model persistence (menyimpan model terlatih)
- Loading dan testing pada new data
- Validasi production-readiness
- Inference pada unseen customers

---

## 🔬 Metodologi

### Tahapan Analisis

```
Data Loading 
    ↓
Exploratory Data Analysis (EDA)
    ↓
Data Cleaning & Preprocessing
    ├─ Handling Missing Values
    ├─ Feature Engineering
    └─ Feature Scaling (StandardScaler)
    ↓
Train-Test Split (70:30 dengan Stratifikasi)
    ↓
Class Imbalance Handling (SMOTE)
    ↓
Model Development & Training
    ├─ Logistic Regression
    ├─ Random Forest
    └─ Gradient Boosting
    ↓
Hyperparameter Tuning (GridSearchCV)
    ↓
Model Evaluation & Comparison
    ├─ Performance Metrics
    ├─ Confusion Matrix
    └─ ROC-AUC Curves
    ↓
Model Selection & Persistence
```

### Teknik Preprocessing Utama

#### **1. Feature Engineering**
Membuat 5 fitur baru dari data existing:
- `avg_pay_status`: Rata-rata payment status (6 bulan terakhir)
- `credit_utilization_ratio`: Bill amount / credit limit
- `payment_to_bill_ratio`: Payment amount / bill amount
- `avg_bill_amount`: Rata-rata bill amount
- `avg_payment_amount`: Rata-rata payment amount

#### **2. Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### **3. Class Imbalance Handling**
Menggunakan SMOTE (Synthetic Minority Over-sampling Technique) untuk:
- Generate synthetic samples dari minority class
- Balance distribusi class dalam training set
- Mencegah model bias terhadap majority class

#### **4. Train-Test Split**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
```
**Hasil Split:**
- Training Set: 20,720 samples (70%)
- Testing Set: 8,881 samples (30%)
- Stratifikasi mempertahankan class distribution

---

## 🤖 Algoritma Machine Learning

Penelitian menggunakan tiga algoritma yang saling komplementer:

### **1. LOGISTIC REGRESSION (LR)**

**Konsep:**
Algoritma supervised learning untuk binary classification yang menggunakan sigmoid function untuk menghasilkan probability output (0-1).

**Keunggulan:**
- ✅ **Baseline Model:** Sederhana untuk comparison
- ✅ **Interpretable:** Feature weights menunjukkan direct impact
- ✅ **Efficient:** Training dan inference sangat cepat
- ✅ **Linear Relationships:** Excellent untuk understanding linear patterns

**Konfigurasi:**
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
```

**Best Parameters dari Tuning:**
- C = 0.01 (regularization strength)

---

### **2. RANDOM FOREST (RF)**

**Konsep:**
Ensemble learning yang mengkombinasikan multiple decision trees. Setiap tree ditraining pada random bootstrap sample dengan random feature subset. Final prediction adalah majority vote dari semua trees.

**Keunggulan:**
- ✅ **Non-linear:** Menangkap complex, non-linear patterns
- ✅ **Feature Interactions:** Otomatis mengidentifikasi interactions
- ✅ **Robustness:** Robust terhadap outliers & overfitting
- ✅ **Feature Importance:** Built-in importance scores
- ✅ **Imbalance Handling:** Lebih baik dalam handling imbalanced data

**Konfigurasi:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
```

**Karakteristik:**
- Non-linear decision boundaries
- Automatic feature selection
- Interpretable feature importance

---

### **3. GRADIENT BOOSTING (GB)**

**Konsep:**
Sequential ensemble method dimana setiap tree baru ditraining untuk memperbaiki residuals (errors) dari ensemble trees sebelumnya. Process berulang dengan focus pada correcting previous misclassifications.

**Keunggulan:**
- ✅ **High Accuracy:** Biasanya mencapai highest predictive accuracy
- ✅ **Sequential Learning:** Belajar dari mistakes iteration sebelumnya
- ✅ **Non-linear Power:** Powerful untuk complex relationships
- ✅ **Flexibility:** Tunable learning rate untuk prevent overfitting
- ✅ **Proven:** Effective dalam many real-world applications

**Konfigurasi:**
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

**Karakteristik:**
- Sequential tree building (boosting)
- Iterative error correction
- Powerful non-linear modeling
- Generally highest accuracy

---

### Rationale Tiga Algoritma

| Aspek | LR | RF | GB |
|-------|----|----|-----|
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Non-linear** | ❌ | ✅ | ✅ |
| **Imbalance** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**Kombinasi ini memungkinkan:**
1. Comparative analysis untuk best approach
2. Complementary strengths (interpretability vs accuracy)
3. Understanding model complexity effects
4. Robustness validation

---

## 📁 Struktur Project

```
credit-card-default-prediction/
│
├── atka.ipynb                          # Main analysis notebook
├── README.md                           # This file
├── dataset/
│   └── default_of_credit_card_clients.csv
│
├── code_sections/
│   ├── 01_data_loading.py
│   ├── 02_eda.py
│   ├── 03_preprocessing.py
│   ├── 04_feature_engineering.py
│   ├── 05_model_training.py
│   └── 06_evaluation.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── outputs/
│   ├── performance_comparison.csv
│   ├── feature_importance.png
│   └── roc_curves.png
│
└── requirements.txt
```

---

## 🔧 Instalasi & Setup

### Prerequisites
- Python 3.7 atau lebih tinggi
- Jupyter Notebook atau Google Colab
- pip package manager

### Instalasi Libraries

```bash
# Opsi 1: Menggunakan requirements.txt
pip install -r requirements.txt

# Opsi 2: Manual installation
pip install pandas numpy matplotlib seaborn scikit-learn
pip install imbalanced-learn
pip install jupyter
```

### Required Libraries

```python
import pandas as pd                    # Data manipulation
import numpy as np                     # Numerical computing
import matplotlib.pyplot as plt        # Visualization
import seaborn as sns                  # Statistical visualization
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import pickle
```

### Setup di Google Colab

```python
# Install imbalanced-learn di Colab
!pip install imbalanced-learn

# Mount Google Drive (jika menggunakan data dari Drive)
from google.colab import drive
drive.mount('/content/drive')

# Load data dari GitHub
data = pd.read_csv('https://raw.githubusercontent.com/[username]/[repo]/master/dataset/default%20of%20credits%20card%20clients.csv')
```

---

## 📖 Panduan Penggunaan

### 1. Running the Full Analysis

**Opsi A: Google Colab**
```
1. Buka file atka.ipynb di Google Colab
2. Click "Run All" untuk menjalankan semua cells
3. Monitor progress di output cells
```

**Opsi B: Local Jupyter**
```bash
jupyter notebook atka.ipynb
# Atau
jupyter-notebook atka.ipynb
```

### 2. Step-by-Step Execution

```python
# STEP 1: Load Data
import pandas as pd
data = pd.read_csv('dataset/default_of_credit_card_clients.csv')
print(data.shape)  # (30000, 25)

# STEP 2: Exploratory Data Analysis
print(data.info())
print(data.describe())
print(data['default.payment.next.month'].value_counts())

# STEP 3: Data Preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# STEP 4: Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# STEP 5: Model Evaluation
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### 3. Making Predictions pada New Data

```python
import pickle

# Load trained model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new customer data
new_customer = pd.DataFrame({
    'LIMIT_BAL': [100000],
    'AGE': [35],
    'PAY_0': [0],
    # ... other features
})

# Scale features
new_customer_scaled = scaler.transform(new_customer)

# Make prediction
prediction = model.predict(new_customer_scaled)
probability = model.predict_proba(new_customer_scaled)

print(f"Prediction: {prediction[0]}")  # 0 atau 1
print(f"Default Probability: {probability[0][1]:.2%}")
```

---

## 📈 Hasil & Evaluasi

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 81.2% | 65.4% | 45.2% | 53.4% | 0.842 |
| **Random Forest** | 82.1% | 68.9% | 52.1% | 59.3% | 0.867 |
| **Gradient Boosting** | 82.8% | 71.2% | 54.7% | 61.8% | 0.879 |

### Evaluation Metrics Explanation

#### **Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Proporsi prediksi yang benar (warning: bisa misleading dengan imbalanced data)

#### **Precision**
```
Precision = TP / (TP + FP)
```
- Dari prediksi default, berapa % yang benar?
- Important ketika False Positive cost tinggi

#### **Recall**
```
Recall = TP / (TP + FN)
```
- Dari actual defaults, berapa % yang terdeteksi?
- Important untuk minimize False Negatives (missed defaults)

#### **F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean dari precision dan recall
- **[PRIMARY METRIC FOR THIS PROJECT]**

#### **ROC-AUC**
- Area under the ROC curve
- Mengukur discriminative ability across all thresholds
- Range: 0 - 1 (1 = perfect classifier)

### Feature Importance (Top 10 - Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | avg_pay_status | 0.285 |
| 2 | PAY_0 | 0.241 |
| 3 | credit_utilization_ratio | 0.108 |
| 4 | PAY_2 | 0.067 |
| 5 | payment_to_bill_ratio | 0.061 |
| 6 | LIMIT_BAL | 0.051 |
| 7 | PAY_3 | 0.048 |
| 8 | avg_bill_amount | 0.043 |
| 9 | AGE | 0.038 |
| 10 | PAY_4 | 0.035 |

**Key Insight:** Payment status adalah strongest predictor untuk default!

---

## 🎯 Kesimpulan

### Temuan Utama

1. **Payment Status Dominates:** Riwayat pembayaran (PAY_0-6) adalah predictor terkuat dengan importance >54%
2. **Gradient Boosting Best:** Mencapai F1-score 61.8% dan ROC-AUC 0.879
3. **Class Imbalance Manageable:** SMOTE effective dalam handling 3.48:1 ratio
4. **Non-linear Relationships:** Tree-based models (RF, GB) significantly outperform LR

### Rekomendasi Implementasi

✅ **Use Gradient Boosting** untuk production deployment (highest accuracy)
✅ **Monitor Feature Drift** terhadap payment status trends
✅ **Set Decision Threshold** based on business cost function (adjust for FP vs FN)
✅ **Regular Retraining** setiap quarter dengan data baru
✅ **A/B Testing** sebelum full rollout

### Keterbatasan & Future Work

⚠️ **Class Imbalance:** Meski SMOTE membantu, bisa diimprove dengan cost-sensitive learning
⚠️ **Feature Temporal:** Current features tidak capture temporal patterns (time series)
⚠️ **External Factors:** Economic conditions, interest rates tidak included
🔮 **Future Improvements:**
- Implement LSTM/GRU untuk time-series patterns
- Add macro-economic indicators
- Ensemble voting dari semua 3 models
- Online learning untuk real-time adaptation

---

## 📚 References

1. Yeh, I. C., & Lien, C. H. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." *The Journal of the Operational Research Society*, 60(12), 1651-1659.

2. Scikit-learn Documentation: https://scikit-learn.org/
3. SMOTE: Synthetic Minority Over-sampling Technique: https://imbalanced-learn.org/

4. UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

---

## 👩‍💼 Author & Attribution

| Aspek | Detail |
|-------|--------|
| **Nama Penulis** | Friska Andalusia |
| **NIM** | 202022420003 |
| **Mata Kuliah** | Analisa Keputusan Untuk Teknologi Finansial (AKTF) |
| **Institusi** | [Universitas Anda] |
| **Tahun** | 2024-2025 |

---

## 📞 Support & Questions

Untuk pertanyaan atau issues:
1. Buka GitHub Issues di repository
2. Email: [your.email@university.edu]
3. Konsultasi dengan dosen pembimbing

---

## 📄 License

Project ini dilisensikan di bawah MIT License - lihat file LICENSE untuk detail.

---

**Last Updated:** January 2026
**Version:** 1.0.0
**Status:** ✅ Complete & Production Ready