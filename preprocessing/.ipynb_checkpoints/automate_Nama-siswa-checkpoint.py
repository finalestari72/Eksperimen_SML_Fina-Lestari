"""
automate_Nama-siswa.py
======================
Skrip otomatisasi preprocessing dataset untuk submission Machine Learning.
Mengonversi langkah-langkah eksperimen dari notebook menjadi fungsi modular
yang dapat dijalankan secara otomatis (CLI maupun impor sebagai modul).

Penggunaan:
    python automate_Fina-Lestari.py --input water_potability_raw.csv \\
                                   --output preprocessing/water_potability_preprocessing.csv

Penulis : Fina-Lestari
Tanggal : 2026-06-05
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Konfigurasi logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNGSI LOADING DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV/Excel ke dalam DataFrame.

    Parameters
    ----------
    file_path : str
        Path ke file dataset mentah.

    Returns
    -------
    pd.DataFrame
        DataFrame berisi data mentah.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()
    if ext in (".csv",):
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Format file tidak didukung: {ext}")

    logger.info("Dataset berhasil dimuat: %s baris × %s kolom", *df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNGSI PREPROCESSING (modular — sesuai notebook eksperimen)
# ══════════════════════════════════════════════════════════════════════════════

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info("Duplikat dihapus: %d baris", removed)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values:
      - Kolom numerik  → imputasi dengan median
      - Kolom kategori → imputasi dengan modus

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        logger.info("Imputasi median: %d kolom numerik", len(numeric_cols))

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        logger.info("Imputasi modus: %d kolom kategorikal", len(cat_cols))

    remaining = int(df.isnull().sum().sum())
    logger.info("Missing values setelah imputasi: %d", remaining)
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Menghapus baris yang mengandung outlier menggunakan metode IQR.

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list[str] | None
        Kolom yang akan dicek outlier-nya. Jika None, semua kolom numerik.
    factor  : float
        Pengali IQR (default 1.5).

    Returns
    -------
    pd.DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    before = len(df)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed = before - len(df)
    logger.info("Outlier dihapus: %d baris (IQR factor=%.1f)", removed, factor)
    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Mengodekan semua kolom kategorikal menggunakan Label Encoding.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame terenkode dan dict berisi LabelEncoder per kolom.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders: dict[str, LabelEncoder] = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Label Encoding: '%s' → %d kelas", col, len(le.classes_))

    return df, encoders


def normalize_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    method: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Melakukan normalisasi / standarisasi fitur numerik.

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str | None
        Nama kolom target yang TIDAK akan diskalakan. Jika None, semua kolom
        numerik akan diskalakan.
    method     : str
        'standard' untuk StandardScaler, 'minmax' untuk MinMaxScaler.

    Returns
    -------
    tuple[pd.DataFrame, scaler]
    """
    from sklearn.preprocessing import MinMaxScaler

    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in features:
        features.remove(target_col)

    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df[features] = scaler.fit_transform(df[features])
    logger.info(
        "Normalisasi (%s) diterapkan pada %d fitur", method.upper(), len(features)
    )
    return df, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE UTAMA
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    input_path: str,
    output_path: str,
    target_col: str | None = None,
    outlier_cols: list[str] | None = None,
    scale_method: str = "standard",
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Pipeline preprocessing lengkap:
        1. Load data
        2. Hapus duplikat
        3. Tangani missing values
        4. Hapus outlier
        5. Encode kategorikal
        6. Normalisasi fitur
        7. Simpan hasil

    Parameters
    ----------
    input_path   : str   – path ke dataset mentah
    output_path  : str   – path tujuan dataset hasil preprocessing
    target_col   : str   – kolom target (tidak diskalakan)
    outlier_cols : list  – kolom untuk deteksi outlier (None = semua numerik)
    scale_method : str   – 'standard' / 'minmax'
    iqr_factor   : float – pengali IQR untuk outlier

    Returns
    -------
    pd.DataFrame – dataset siap latih
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING")
    logger.info("=" * 60)

    # 1. Load
    df = load_dataset(input_path)
    shape_awal = df.shape

    # 2. Duplikat
    df = remove_duplicates(df)

    # 3. Missing values
    df = handle_missing_values(df)

    # 4. Outlier
    df = remove_outliers_iqr(df, columns=outlier_cols, factor=iqr_factor)

    # 5. Encoding
    df, _ = encode_categorical(df)

    # 6. Normalisasi
    df, _ = normalize_features(df, target_col=target_col, method=scale_method)

    # 7. Simpan
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info("PREPROCESSING SELESAI")
    logger.info("  Shape awal  : %s", shape_awal)
    logger.info("  Shape akhir : %s", df.shape)
    logger.info("  Output      : %s", output_path)
    logger.info("=" * 60)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing dataset untuk submission MSML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path ke file dataset mentah (CSV/XLSX).",
    )
    parser.add_argument(
        "--output", "-o",
        default="preprocessing/namadataset_preprocessing.csv",
        help="Path tujuan file hasil preprocessing.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Nama kolom target (tidak akan diskalakan).",
    )
    parser.add_argument(
        "--scale",
        choices=["standard", "minmax"],
        default="standard",
        help="Metode normalisasi fitur.",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="Pengali IQR untuk deteksi outlier.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    preprocess(
        input_path=args.input,
        output_path=args.output,
        target_col=args.target,
        scale_method=args.scale,
        iqr_factor=args.iqr_factor,
    )


if __name__ == "__main__":
    main()
