"""
automate_Fina-Lestari.py
========================
Skrip otomasi preprocessing dataset Water Potability.
Mengonversi proses eksperimen dari Eksperimen_Fina-Lestari.ipynb
menjadi fungsi-fungsi modular yang siap digunakan untuk pelatihan model.

Tahapan preprocessing:
    1. Memuat dataset
    2. Menghapus duplikat
    3. Imputasi missing values (median)
    4. Penanganan outlier (IQR method)
    5. Normalisasi fitur (StandardScaler)
    6. Train-test split
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ─── Konfigurasi ────────────────────────────────────────────────────────────

DATASET_PATH = 'water_potability_raw.csv'
OUTPUT_DIR   = './'
OUTPUT_FILE  = 'water_potability_preprocessing.csv'

COLS_WITH_MISSING = ['ph', 'Sulfate', 'Trihalomethanes']
TARGET_COLUMN     = 'Potability'
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
IQR_FACTOR        = 1.5


# ─── Fungsi-fungsi Preprocessing ────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Memuat dataset CSV ke dalam DataFrame.

    Parameters
    ----------
    path : str
        Path ke file CSV dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame hasil pembacaan CSV.
    """
    df = pd.read_csv(path)
    print(f'[1] Dataset berhasil dimuat.')
    print(f'    Shape awal: {df.shape[0]} baris × {df.shape[1]} kolom')
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa duplikat.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f'[2] Duplikat dihapus: {removed} baris → Shape: {df.shape}')
    return df


def impute_missing_values(df: pd.DataFrame,
                          cols: list,
                          strategy: str = 'median') -> pd.DataFrame:
    """
    Melakukan imputasi missing values pada kolom-kolom tertentu.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.
    cols : list
        Daftar nama kolom yang akan diimputasi.
    strategy : str, optional
        Strategi imputasi ('median', 'mean', 'most_frequent').
        Default: 'median'.

    Returns
    -------
    pd.DataFrame
        DataFrame setelah imputasi.
    """
    imputer = SimpleImputer(strategy=strategy)
    df = df.copy()
    df[cols] = imputer.fit_transform(df[cols])
    print(f'[3] Imputasi {strategy} diterapkan pada: {cols}')
    print(f'    Missing values tersisa: {df.isnull().sum().sum()}')
    return df


def remove_outliers_iqr(df: pd.DataFrame,
                         exclude_cols: list,
                         factor: float = 1.5) -> pd.DataFrame:
    """
    Menghapus outlier menggunakan metode IQR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.
    exclude_cols : list
        Kolom yang dikecualikan dari penghapusan outlier (misal: target).
    factor : float, optional
        Faktor pengali IQR untuk menentukan batas outlier. Default: 1.5.

    Returns
    -------
    pd.DataFrame
        DataFrame setelah outlier dihapus.
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    before = len(df)

    for col in feature_cols:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask  = (df[col] >= lower) & (df[col] <= upper)
        df    = df[mask]

    total_removed = before - len(df)
    print(f'[4] Outlier dihapus: {total_removed} baris → Shape: {df.shape}')
    return df


def normalize_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Menormalisasi fitur menggunakan StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame fitur (tanpa kolom target).

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        - DataFrame fitur yang sudah dinormalisasi.
        - Objek scaler (untuk digunakan ulang saat inferensi).
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    print(f'[5] Normalisasi StandardScaler diterapkan.')
    print(f'    Mean fitur (≈ 0): {X_scaled.mean().round(4).to_dict()}')
    return X_scaled, scaler


def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Membagi data menjadi set pelatihan dan pengujian secara stratified.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame fitur yang sudah diproses.
    y : pd.Series
        Series target/label.
    test_size : float, optional
        Proporsi data uji. Default: 0.2.
    random_state : int, optional
        Seed untuk reprodusibilitas. Default: 42.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f'[6] Train-test split selesai.')
    print(f'    Train : {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.1f}%)')
    print(f'    Test  : {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.1f}%)')
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_scaled: pd.DataFrame,
                            y: pd.Series,
                            output_dir: str,
                            filename: str) -> str:
    """
    Menyimpan data hasil preprocessing ke file CSV.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        DataFrame fitur yang sudah dinormalisasi.
    y : pd.Series
        Series target/label.
    output_dir : str
        Direktori tujuan penyimpanan.
    filename : str
        Nama file CSV output.

    Returns
    -------
    str
        Path lengkap file yang disimpan.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_final = X_scaled.copy()
    df_final[TARGET_COLUMN] = y.values

    output_path = os.path.join(output_dir, filename)
    df_final.to_csv(output_path, index=False)
    print(f'[7] Data disimpan ke: {output_path}')
    return output_path


# ─── Fungsi Utama ───────────────────────────────────────────────────────────

def preprocess(dataset_path: str = DATASET_PATH,
               output_dir: str = OUTPUT_DIR,
               output_file: str = OUTPUT_FILE,
               save_output: bool = True) -> dict:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis.

    Tahapan:
        1. Memuat dataset
        2. Menghapus duplikat
        3. Imputasi missing values (median)
        4. Penanganan outlier (IQR method)
        5. Normalisasi fitur (StandardScaler)
        6. Train-test split (80/20, stratified)
        7. Menyimpan hasil preprocessing (opsional)

    Parameters
    ----------
    dataset_path : str
        Path ke file CSV dataset mentah.
    output_dir : str
        Direktori untuk menyimpan hasil preprocessing.
    output_file : str
        Nama file CSV output.
    save_output : bool
        Jika True, hasil disimpan ke CSV. Default: True.

    Returns
    -------
    dict
        Dictionary berisi:
            - 'X_train'  : pd.DataFrame
            - 'X_test'   : pd.DataFrame
            - 'y_train'  : pd.Series
            - 'y_test'   : pd.Series
            - 'scaler'   : StandardScaler (fitted)
            - 'df_clean' : pd.DataFrame (data setelah semua preprocessing, sebelum split)
    """
    print('=' * 55)
    print('PIPELINE PREPROCESSING — WATER POTABILITY')
    print('=' * 55)

    # 1. Load dataset
    df = load_dataset(dataset_path)

    # 2. Hapus duplikat
    df = remove_duplicates(df)

    # 3. Imputasi missing values
    df = impute_missing_values(df, cols=COLS_WITH_MISSING, strategy='median')

    # 4. Penanganan outlier
    df = remove_outliers_iqr(df, exclude_cols=[TARGET_COLUMN], factor=IQR_FACTOR)

    # 5. Pisahkan fitur dan target, lalu normalisasi
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_scaled, scaler = normalize_features(X)

    # 6. Train-test split
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 7. Simpan hasil preprocessing
    if save_output:
        save_preprocessed_data(X_scaled, y, output_dir, output_file)

    # Ringkasan
    print()
    print('=' * 55)
    print('RINGKASAN PREPROCESSING')
    print('=' * 55)
    print(f'Fitur yang digunakan  : {list(X_scaled.columns)}')
    print(f'Target                : {TARGET_COLUMN}')
    print(f'Metode imputasi       : Median ({COLS_WITH_MISSING})')
    print(f'Outlier handling      : IQR Method (factor={IQR_FACTOR})')
    print(f'Normalisasi           : StandardScaler')
    print(f'Split ratio           : {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test')
    print(f'Shape X_train         : {X_train.shape}')
    print(f'Shape X_test          : {X_test.shape}')
    print('=' * 55)
    print('Preprocessing selesai! Data siap untuk pelatihan model.')

    return {
        'X_train' : X_train,
        'X_test'  : X_test,
        'y_train' : y_train,
        'y_test'  : y_test,
        'scaler'  : scaler,
        'df_clean': pd.concat([X_scaled, y], axis=1),
    }


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    result = preprocess(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        output_file=OUTPUT_FILE,
        save_output=True,
    )

    X_train = result['X_train']
    X_test  = result['X_test']
    y_train = result['y_train']
    y_test  = result['y_test']
