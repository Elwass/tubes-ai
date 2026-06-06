from __future__ import annotations

from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd


ARTIFACT_PATH = Path("model_artifacts.pkl")
OUTPUT_CSV = Path("sintetis_rekomendasi_1000.csv")
OUTPUT_XLSX = Path("sintetis_rekomendasi_1000.xlsx")
N_SAMPLES = 1000
RANDOM_SEED = 42


NUMERIC_FEATURES = [
    "Saya lebih suka kegiatan di luar ruangan (lapangan)",
    "Saya senang berdiskusi dan membahas suatu topik secara mendalam",
    "Saya tertarik untuk mengajar atau membimbing orang lain",
    "Saya tertarik melakukan penelitian atau mengolah data",
    "Saya lebih suka kegiatan praktik dibanding teori",
    "Saya memiliki kemampuan komunikasi yang baik",
    "Saya mampu bekerja sama dalam tim",
    "Saya memiliki jiwa kepemimpinan",
    "Saya mampu menganalisis masalah dengan baik",
    "Saya teliti dalam mengerjakan sesuatu",
    "Saya kreatif dalam mencari ide atau solusi",
    "Saya termasuk orang yang aktif dan energik",
    "Saya lebih nyaman bekerja dalam tim",
    "Saya memiliki empati terhadap orang lain",
    "Saya lebih suka pekerjaan yang terstruktur",
    "Saya mudah beradaptasi dengan lingkungan baru",
]


LIKERT_PROFILES = {
    "Bimbingan & Pengembangan": {
        "Saya lebih suka kegiatan di luar ruangan (lapangan)": 2.6,
        "Saya senang berdiskusi dan membahas suatu topik secara mendalam": 4.1,
        "Saya tertarik untuk mengajar atau membimbing orang lain": 4.7,
        "Saya tertarik melakukan penelitian atau mengolah data": 3.0,
        "Saya lebih suka kegiatan praktik dibanding teori": 3.2,
        "Saya memiliki kemampuan komunikasi yang baik": 4.5,
        "Saya mampu bekerja sama dalam tim": 4.3,
        "Saya memiliki jiwa kepemimpinan": 4.0,
        "Saya mampu menganalisis masalah dengan baik": 3.7,
        "Saya teliti dalam mengerjakan sesuatu": 3.8,
        "Saya kreatif dalam mencari ide atau solusi": 3.9,
        "Saya termasuk orang yang aktif dan energik": 4.0,
        "Saya lebih nyaman bekerja dalam tim": 4.2,
        "Saya memiliki empati terhadap orang lain": 4.6,
        "Saya lebih suka pekerjaan yang terstruktur": 3.3,
        "Saya mudah beradaptasi dengan lingkungan baru": 3.6,
    },
    "Kajian Pramuka": {
        "Saya lebih suka kegiatan di luar ruangan (lapangan)": 2.5,
        "Saya senang berdiskusi dan membahas suatu topik secara mendalam": 4.8,
        "Saya tertarik untuk mengajar atau membimbing orang lain": 3.6,
        "Saya tertarik melakukan penelitian atau mengolah data": 4.5,
        "Saya lebih suka kegiatan praktik dibanding teori": 3.0,
        "Saya memiliki kemampuan komunikasi yang baik": 4.0,
        "Saya mampu bekerja sama dalam tim": 3.8,
        "Saya memiliki jiwa kepemimpinan": 3.8,
        "Saya mampu menganalisis masalah dengan baik": 4.7,
        "Saya teliti dalam mengerjakan sesuatu": 4.6,
        "Saya kreatif dalam mencari ide atau solusi": 4.1,
        "Saya termasuk orang yang aktif dan energik": 3.4,
        "Saya lebih nyaman bekerja dalam tim": 3.8,
        "Saya memiliki empati terhadap orang lain": 3.9,
        "Saya lebih suka pekerjaan yang terstruktur": 4.6,
        "Saya mudah beradaptasi dengan lingkungan baru": 3.6,
    },
    "Kegiatan": {
        "Saya lebih suka kegiatan di luar ruangan (lapangan)": 4.6,
        "Saya senang berdiskusi dan membahas suatu topik secara mendalam": 3.0,
        "Saya tertarik untuk mengajar atau membimbing orang lain": 3.5,
        "Saya tertarik melakukan penelitian atau mengolah data": 2.8,
        "Saya lebih suka kegiatan praktik dibanding teori": 4.5,
        "Saya memiliki kemampuan komunikasi yang baik": 3.9,
        "Saya mampu bekerja sama dalam tim": 4.2,
        "Saya memiliki jiwa kepemimpinan": 3.8,
        "Saya mampu menganalisis masalah dengan baik": 3.2,
        "Saya teliti dalam mengerjakan sesuatu": 3.2,
        "Saya kreatif dalam mencari ide atau solusi": 3.8,
        "Saya termasuk orang yang aktif dan energik": 4.7,
        "Saya lebih nyaman bekerja dalam tim": 4.1,
        "Saya memiliki empati terhadap orang lain": 3.9,
        "Saya lebih suka pekerjaan yang terstruktur": 2.8,
        "Saya mudah beradaptasi dengan lingkungan baru": 4.0,
    },
    "Penelitian & Evaluasi": {
        "Saya lebih suka kegiatan di luar ruangan (lapangan)": 2.8,
        "Saya senang berdiskusi dan membahas suatu topik secara mendalam": 4.2,
        "Saya tertarik untuk mengajar atau membimbing orang lain": 3.3,
        "Saya tertarik melakukan penelitian atau mengolah data": 4.8,
        "Saya lebih suka kegiatan praktik dibanding teori": 3.3,
        "Saya memiliki kemampuan komunikasi yang baik": 3.9,
        "Saya mampu bekerja sama dalam tim": 3.9,
        "Saya memiliki jiwa kepemimpinan": 3.7,
        "Saya mampu menganalisis masalah dengan baik": 4.6,
        "Saya teliti dalam mengerjakan sesuatu": 4.7,
        "Saya kreatif dalam mencari ide atau solusi": 4.0,
        "Saya termasuk orang yang aktif dan energik": 3.6,
        "Saya lebih nyaman bekerja dalam tim": 3.8,
        "Saya memiliki empati terhadap orang lain": 3.7,
        "Saya lebih suka pekerjaan yang terstruktur": 4.3,
        "Saya mudah beradaptasi dengan lingkungan baru": 3.7,
    },
}


CAT_PROFILES = {
    "Status": {
        "Calon Dewan": 0.95,
        "Dewan Ambalan": 0.05,
    },
    "Jenis Kelamin": {
        "Laki-laki": 0.55,
        "Perempuan": 0.45,
    },
    "Pernah jadi panitia?": {
        "YA": 0.45,
        "TIDAK": 0.55,
    },
    "Pernah jadi leader?": {
        "YA": 0.45,
        "TIDAK": 0.55,
    },
    "Pernah ikut penelitian/evaluasi kegiatan?": {
        "YA": 0.55,
        "TIDAK": 0.45,
    },
    "Saya lebih suka bekerja di:": {
        "Balik layar": 0.45,
        "Lapangan": 0.55,
    },
    "Saya lebih tertarik pada:": {
        "Analisis atau perencanaan kegiatan": 0.58,
        "Kegiatan/event": 0.42,
    },
}


NUMERIC_BINARY_FEATURES = {
    "Pernah ikut organisasi Sebelumnya?": {
        "base": 0.45,
        "Bimbingan & Pengembangan": 0.60,
        "Kajian Pramuka": 0.45,
        "Kegiatan": 0.42,
        "Penelitian & Evaluasi": 0.55,
    },
}

CAT_PROFILES_BY_DIVISI = {
    "Bimbingan & Pengembangan": {
        "Pernah jadi panitia?": 0.55,
        "Pernah jadi leader?": 0.50,
        "Pernah ikut penelitian/evaluasi kegiatan?": 0.35,
        "Saya lebih suka bekerja di:": 0.40,  # 0.40 => Balik layar
        "Saya lebih tertarik pada:": 0.30,  # 0.30 => Analisis/perencanaan
    },
    "Kajian Pramuka": {
        "Pernah jadi panitia?": 0.35,
        "Pernah jadi leader?": 0.38,
        "Pernah ikut penelitian/evaluasi kegiatan?": 0.45,
        "Saya lebih suka bekerja di:": 0.55,  # 0.55 => balik layar
        "Saya lebih tertarik pada:": 0.78,  # 0.78 => Analisis/perencanaan
    },
    "Kegiatan": {
        "Pernah jadi panitia?": 0.60,
        "Pernah jadi leader?": 0.52,
        "Pernah ikut penelitian/evaluasi kegiatan?": 0.20,
        "Saya lebih suka bekerja di:": 0.85,  # 0.85 => Lapangan
        "Saya lebih tertarik pada:": 0.18,  # 0.18 => Kegiatan/event
    },
    "Penelitian & Evaluasi": {
        "Pernah jadi panitia?": 0.45,
        "Pernah jadi leader?": 0.40,
        "Pernah ikut penelitian/evaluasi kegiatan?": 0.70,
        "Saya lebih suka bekerja di:": 0.65,  # 0.65 => Balik layar
        "Saya lebih tertarik pada:": 0.80,  # 0.80 => Analisis/perencanaan
    },
}


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("model_artifacts.pkl tidak ditemukan. Jalankan ai.ipynb dulu untuk menyimpan model.")
    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def _pick_value(p: float, options: List[str], rng: np.random.Generator) -> str:
    p = float(np.clip(p, 0.0, 1.0))
    if len(options) == 0:
        return ""
    if len(options) == 1:
        return options[0]
    return options[0] if rng.random() > p else options[1]


def _likert_from_profile(mean: float, rng: np.random.Generator, std: float = 0.85) -> int:
    score = int(np.rint(rng.normal(loc=mean, scale=std)))
    return int(np.clip(score, 1, 5))


def _build_profiles(artifact: dict) -> dict:
    features = artifact["fitur_kolom"]
    encoders = artifact.get("encoders", {})
    target_col = artifact["target_col"]
    if target_col not in encoders:
        raise ValueError(f"Encoder untuk '{target_col}' tidak ditemukan.")
    if encoders[target_col].classes_.size == 0:
        raise ValueError("Encoder target tidak valid.")

    target_classes = list(encoders[target_col].classes_)

    # Pastikan semua nama feature tersedia
    missing_num = [col for col in NUMERIC_FEATURES if col not in features]
    if missing_num:
        raise ValueError(f"Kolom numerik berikut tidak ditemukan di fitur model: {missing_num}")
    missing_cat = [col for col in CAT_PROFILES if col not in encoders]
    if missing_cat:
        raise ValueError(f"Kolom kategorikal berikut tidak ada encoder: {missing_cat}")

    # Normalisasi nama kelas agar tidak gagal kalau dataset target lain
    available_targets = set(target_classes)
    selected_profiles = {
        k: LIKERT_PROFILES[k] for k in LIKERT_PROFILES if k in available_targets
    }
    if len(selected_profiles) < len(target_classes):
        for k in target_classes:
            if k not in selected_profiles:
                selected_profiles[k] = {f: 3.0 for f in NUMERIC_FEATURES}

    return {
        "features": features,
        "encoders": encoders,
        "targets": target_classes,
        "profiles": selected_profiles,
    }


def _sample_row_for_division(
    divisi: str,
    artifact_info: dict,
    encoders: dict,
    rng: np.random.Generator,
) -> tuple[Dict[str, object], str]:
    features = artifact_info["features"]
    row: Dict[str, object] = {}
    prof = artifact_info["profiles"][divisi]

    for col in features:
        if col in encoders:
            classes = list(encoders[col].classes_)
            if len(classes) >= 2:
                p_yes = 0.5

                if col in CAT_PROFILES:
                    p_yes = CAT_PROFILES[col].get(classes[1], 0.5)
                    if col in CAT_PROFILES_BY_DIVISI and divisi in CAT_PROFILES_BY_DIVISI:
                        p_yes = CAT_PROFILES_BY_DIVISI[divisi].get(col, p_yes)

                # NOTE: _pick_value menganggap prob terhadap opsi index ke-1.
                row[col] = _pick_value(p_yes, classes, rng)
            else:
                row[col] = classes[0]
        elif col in NUMERIC_BINARY_FEATURES:
            by_class = NUMERIC_BINARY_FEATURES[col]
            p_yes = by_class.get("base", 0.5)
            if divisi in by_class:
                p_yes = by_class[divisi]
            row[col] = 1 if rng.random() < p_yes else 0
        else:
            if col in prof:
                row[col] = _likert_from_profile(prof[col], rng)
            else:
                row[col] = _likert_from_profile(3.0, rng)

    return row, divisi


def _encode_for_model(df_rows: pd.DataFrame, artifact_info: dict) -> pd.DataFrame:
    encoders = artifact_info["encoders"]
    encoded = df_rows.copy()
    for col in artifact_info["features"]:
        if col in encoders:
            encoded[col] = encoders[col].transform(encoded[col].astype(str))
    return encoded


def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
    output_csv: Path = OUTPUT_CSV,
    output_xlsx: Path = OUTPUT_XLSX,
) -> None:
    artifact = _load_artifact()
    target_col = artifact["target_col"]
    model = artifact["model"]
    encoders = artifact.get("encoders", {})
    artifact_info = _build_profiles(artifact)
    rng = np.random.default_rng(seed + 1)

    target_cycle = artifact_info["targets"]
    rows: List[Dict[str, object]] = []
    for i in range(n_samples):
        div = target_cycle[i % len(target_cycle)]
        row, _ = _sample_row_for_division(div, artifact_info, encoders, rng)
        rows.append(row)

    df = pd.DataFrame(rows)
    encoded_for_pred = _encode_for_model(df, artifact_info)
    pred_idx = model.predict(encoded_for_pred)
    pred_divisi = encoders[target_col].inverse_transform(pred_idx)

    df[target_col] = pred_divisi

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    saved_xlsx = False
    try:
        df.to_excel(output_xlsx, index=False, engine="openpyxl")
        saved_xlsx = True
    except Exception:
        saved_xlsx = False

    print(f"[OK] Data sintetik berhasil dibuat: {output_csv} ({n_samples} baris)")
    if saved_xlsx:
        print(f"[OK] Juga disimpan sebagai: {output_xlsx}")
    else:
        print("[INFO] openpyxl belum terpasang, skip export xlsx.")

    print(f"[INFO] Distribusi target:")
    print(df[target_col].value_counts().to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data sintetik rekomendasi divisi.")
    parser.add_argument("--n", type=int, default=N_SAMPLES, help="Jumlah baris data sintetik (default: 1000)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Seed random")
    parser.add_argument("--csv", type=Path, default=OUTPUT_CSV, help="Nama file CSV output")
    parser.add_argument("--xlsx", type=Path, default=OUTPUT_XLSX, help="Nama file XLSX output (jika openpyxl tersedia)")
    args = parser.parse_args()

    generate_synthetic_dataset(
        n_samples=args.n,
        seed=args.seed,
        output_csv=args.csv,
        output_xlsx=args.xlsx,
    )
