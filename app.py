from __future__ import annotations

import os
import re
from html import escape
import smtplib
import ssl
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from email.message import EmailMessage


MODEL_ARTIFACT_PATH = "model_artifacts.pkl"
TARGET_LABELS = [
    "Bimbingan & Pengembangan",
    "Kajian Pramuka",
    "Kegiatan",
    "Penelitian & Evaluasi",
]
FORCED_STATUS = "Calon Dewan"
SMTP_DEFAULT_HOST = "smtp.gmail.com"

NUMERIC_SCALE_MIN = 1
NUMERIC_SCALE_MAX = 5

MANUAL_RULES: Dict[str, list[tuple[str, float, str, str]]] = {
    "Kegiatan": [
        ("Saya lebih suka kegiatan di luar ruangan (lapangan)", 1.5, "numeric", "high"),
        ("Saya lebih suka kegiatan praktik dibanding teori", 1.2, "numeric", "high"),
        ("Saya termasuk orang yang aktif dan energik", 1.1, "numeric", "high"),
        ("Saya lebih suka bekerja di:", 0.9, "categorical", "Lapangan"),
        ("Saya lebih tertarik pada:", 0.9, "categorical", "Kegiatan/event"),
    ],
    "Penelitian & Evaluasi": [
        ("Saya tertarik melakukan penelitian atau mengolah data", 1.6, "numeric", "high"),
        ("Saya mampu menganalisis masalah dengan baik", 1.3, "numeric", "high"),
        ("Saya teliti dalam mengerjakan sesuatu", 1.2, "numeric", "high"),
        ("Saya lebih suka pekerjaan yang terstruktur", 1.0, "numeric", "high"),
        ("Saya lebih suka bekerja di:", 0.8, "categorical", "Balik layar"),
    ],
    "Bimbingan & Pengembangan": [
        ("Saya tertarik untuk mengajar atau membimbing orang lain", 1.4, "numeric", "high"),
        ("Saya memiliki kemampuan komunikasi yang baik", 1.2, "numeric", "high"),
        ("Saya memiliki empati terhadap orang lain", 1.1, "numeric", "high"),
        ("Saya memiliki jiwa kepemimpinan", 1.1, "numeric", "high"),
        ("Saya mampu bekerja sama dalam tim", 1.0, "numeric", "high"),
    ],
    "Kajian Pramuka": [
        ("Saya senang berdiskusi dan membahas suatu topik secara mendalam", 1.4, "numeric", "high"),
        ("Saya mampu menganalisis masalah dengan baik", 1.2, "numeric", "high"),
        ("Saya kreatif dalam mencari ide atau solusi", 1.0, "numeric", "high"),
        ("Saya lebih tertarik pada:", 0.9, "categorical", "Analisis atau perencanaan kegiatan"),
    ],
}


st.set_page_config(
    page_title="Sistem Rekomendasi Divisi Pramuka",
    page_icon="??",
    layout="wide",
)


def _style() -> str:
    return """
    <style>
      .stApp {
        background: linear-gradient(155deg, #eef2ff 0%, #f7f9fc 45%, #eef8ff 100%);
        color: #0f172a;
      }
      .app-card { border: 1px solid #d4defe; border-radius: 14px; background: #fff; padding: 1rem; margin-top: .8rem; }
      .section-title { font-weight: 700; margin-bottom: .4rem; }
      .result-card { border: 1px solid #9dd5c7; background: linear-gradient(120deg, #ecfffb 0%, #f0fffb 100%); border-radius: 14px; padding: 1rem; }
      .reason-item { border: 1px solid #d8e3f7; border-radius: 10px; padding: .5rem; background: #f8fafc; margin-bottom: .5rem; }
      .pill { display: inline-block; padding: 4px 12px; border-radius: 999px; background: #e9f0ff; border: 1px solid #c4d7ff; font-weight: 700; }
      footer { visibility: hidden; }
    </style>
    """


def _load_env(path: str = ".env") -> None:
    candidate_paths = [
        Path(path),
        Path(__file__).resolve().parent / path,
        Path.cwd() / path,
    ]
    seen = set()
    for p in candidate_paths:
        if not p.exists() or str(p) in seen:
            continue
        seen.add(str(p))
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip() or line.strip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip().strip('"').strip("'")


def _secret_or_env(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]

    try:
        for key in keys:
            if key in st.secrets:
                value = st.secrets[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    for alias in ("spreadsheet_id", "SPREADSHEET_ID", "sheet_id"):
                        if alias in value:
                            return str(value[alias])
    except Exception:
        return default

    return default


def _resolve_spreadsheet_id() -> str | None:
    return (
        _secret_or_env("sheets", default=None)
        or _secret_or_env("spreadsheet_id", default=None)
        or _secret_or_env("SPREADSHEET_ID", default=None)
        or _secret_or_env("GOOGLE_SHEET_ID", default=None)
        or _secret_or_env("SHEET_ID", default=None)
    )


def _smtp_is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_smtp_config() -> dict[str, Any] | None:
    host = _secret_or_env("SMTP_HOST", "SMTP_SERVER", default=SMTP_DEFAULT_HOST)
    port_raw = _secret_or_env("SMTP_PORT", default="587")
    username = _secret_or_env("SMTP_USERNAME", "SMTP_USER", "EMAIL_USER", default=None)
    password = _secret_or_env("SMTP_PASSWORD", "EMAIL_PASSWORD", "SMTP_PASS", default=None)
    sender = _secret_or_env("SMTP_FROM", "EMAIL_FROM", default=username)
    use_tls = _smtp_is_truthy(_secret_or_env("SMTP_USE_TLS", default="true"))

    if not host or not username or not password or not sender:
        return None
    try:
        port = int(port_raw)
    except ValueError:
        return None

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "sender": sender,
        "use_tls": use_tls,
    }


def _is_valid_email(value: str) -> bool:
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, value.strip()))


def _safe_error_text(exc: Exception) -> str:
    if exc is None:
        return "unknown"

    text = str(exc).strip()
    if text:
        return text

    if exc.args:
        try:
            return "; ".join([str(a) for a in exc.args if str(a).strip()]) or "unknown"
        except Exception:
            pass

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            json_loader = getattr(response, "json", None)
            if callable(json_loader):
                payload = json_loader()
                if isinstance(payload, dict):
                    if isinstance(payload.get("error"), dict):
                        err_obj = payload["error"]
                        msg = str(err_obj.get("message", "")).strip()
                        reasons = err_obj.get("errors")
                        if isinstance(reasons, list) and reasons:
                            extra = [
                                str(r.get("reason", "")).strip()
                                for r in reasons
                                if isinstance(r, dict) and str(r.get("reason", "")).strip()
                            ]
                            if extra:
                                msg = f"{msg} (reason: {', '.join(extra)})"
                        if msg:
                            return msg
                    txt = str(payload).strip()
                    if txt:
                        return txt
        except Exception:
            pass

        for key in ("text", "content", "reason", "reason_phrase"):
            value = getattr(response, key, None)
            if value:
                if isinstance(value, (bytes, bytearray)):
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except Exception:
                        value = str(value)
                value = str(value).strip()
                if value:
                    return value

        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return f"HTTP {status_code}"

    err_type = type(exc).__name__
    return f"{err_type}: unknown"


def _build_recommendation_email_html(
    recipient_email: str,
    nama: str,
    kelas: str,
    recommendation: str,
    top3: list[tuple[str, float]],
    logic_ranked: list[tuple[str, float]],
    confidence: float,
) -> tuple[str, str]:
    safe_email = escape(recipient_email or "")
    safe_name = escape(nama or "-")
    safe_kelas = escape(kelas or "-")
    safe_recommendation = escape(recommendation or "-")

    top3_lines_text = (
        "\n".join([f"- {label}: {prob * 100:.2f}%" for label, prob in top3]) if top3 else "- Tidak tersedia"
    )
    logic_top = escape(logic_ranked[0][0] if logic_ranked else "-")
    logic_ranked_text = "\n".join(
        [f"- {i+1}. {escape(divisi)}: {score:.3f}" for i, (divisi, score) in enumerate(logic_ranked[:3])]
    ) if logic_ranked else "- Tidak tersedia"
    safe_confidence = f"{(confidence or 0.0) * 100:.2f}%"

    top3_rows = "".join(
        f"<tr>"
        f"<td style='padding:8px 10px; border:1px solid #dbe3eb;'>{escape(label)}</td>"
        f"<td style='padding:8px 10px; border:1px solid #dbe3eb; text-align:right'>{prob * 100:.2f}%</td>"
        f"</tr>"
        for label, prob in top3
    ) if top3 else "<tr><td style='padding:8px 10px; border:1px solid #dbe3eb;' colspan='2'>Tidak tersedia</td></tr>"

    logic_rows = "".join(
        f"<tr>"
        f"<td style='padding:8px 10px; border:1px solid #dbe3eb; text-align:center'>{idx + 1}</td>"
        f"<td style='padding:8px 10px; border:1px solid #dbe3eb;'>{escape(divisi)}</td>"
        f"<td style='padding:8px 10px; border:1px solid #dbe3eb; text-align:right'>{score:.3f}</td>"
        f"</tr>"
        for idx, (divisi, score) in enumerate(logic_ranked[:3])
    ) if logic_ranked else "<tr><td style='padding:8px 10px; border:1px solid #dbe3eb;' colspan='3'>Tidak tersedia</td></tr>"

    top3_chart_rows = "".join(
        f"<tr>"
        f"<td style='padding:10px 10px; border:1px solid #dbe3eb;'>{escape(label)}</td>"
        f"<td style='padding:10px 10px; border:1px solid #dbe3eb;'>"
        f"<div style='width:100%; background:#e2e8f0; border-radius:8px; overflow:hidden; height:12px;'>"
        f"<div style='height:12px; width:{min(prob * 100, 100):.2f}%; background:linear-gradient(90deg, #0f766e, #10b981); border-radius:8px;'></div>"
        f"</div>"
        f"</td>"
        f"<td style='padding:8px 10px; border:1px solid #dbe3eb; text-align:right; width:80px'>{prob*100:.2f}%</td>"
        f"</tr>"
        for label, prob in top3
    ) if top3 else "<tr><td style='padding:8px 10px; border:1px solid #dbe3eb;' colspan='3'>Tidak tersedia</td></tr>"

    logic_max_score = max((score for _, score in logic_ranked[:3]), default=0)
    if logic_max_score <= 0:
        logic_max_score = 1
    logic_chart_rows = "".join(
        f"<tr>"
        f"<td style='padding:10px 10px; border:1px solid #dbe3eb; text-align:center'>{idx + 1}</td>"
        f"<td style='padding:10px 10px; border:1px solid #dbe3eb;'>{escape(divisi)}</td>"
        f"<td style='padding:10px 10px; border:1px solid #dbe3eb;'>"
        f"<div style='width:100%; background:#e2e8f0; border-radius:8px; overflow:hidden; height:12px;'>"
        f"<div style='height:12px; width:{min((score / logic_max_score) * 100, 100):.2f}%; background:linear-gradient(90deg, #2563eb, #3b82f6); border-radius:8px;'></div>"
        f"</div>"
        f"</td>"
        f"<td style='padding:8px 10px; border:1px solid #dbe3eb; text-align:right; width:80px'>{score:.3f}</td>"
        f"</tr>"
        for idx, (divisi, score) in enumerate(logic_ranked[:3])
    ) if logic_ranked else "<tr><td style='padding:8px 10px; border:1px solid #dbe3eb;' colspan='4'>Tidak tersedia</td></tr>"

    text_body = (
        "Sistem Rekomendasi Divisi Pramuka\n\n"
        f"Yth. {nama}\n\n"
        f"Nama: {nama}\n"
        f"Kelas: {kelas}\n"
        f"Rekomendasi Model: {recommendation}\n"
        f"Confidence Top-1: {safe_confidence}\n"
        f"Rekomendasi Berdasarkan Rubrik: {logic_top}\n\n"
        "Top 3 Probability:\n"
        f"{top3_lines_text}\n\n"
        "Top 3 Rubrik Pendukung:\n"
        f"{logic_ranked_text}\n\n"
        "Catatan: hasil rekomendasi bersifat pendukung keputusan, bukan keputusan final."
    )

    html_body = f"""
    <html>
      <body style="margin:0; padding:20px; background:#f2f4f7; font-family:Arial, Helvetica, sans-serif; color:#1f2937;">
        <div style="max-width:700px; margin:0 auto;">
          <div style="background:#0f766e; color:#ffffff; padding:14px 18px; border-radius:12px 12px 0 0;">
            <h2 style="margin:0; font-size:22px;">Sistem Rekomendasi Divisi Pramuka</h2>
          </div>
          <div style="background:#ffffff; border:1px solid #e5e7eb; border-top:none; border-radius:0 0 12px 12px; padding:20px;">
            <p style="margin:0 0 10px 0;"><strong>Penerima:</strong> {safe_email}</p>
            <p style="margin:0 0 10px 0;"><strong>Nama:</strong> {safe_name}</p>
            <p style="margin:0 0 10px 0;"><strong>Kelas:</strong> {safe_kelas}</p>
            <p style="margin:0 0 10px 0;"><strong>Rekomendasi Model:</strong> {safe_recommendation}</p>
            <p style="margin:0 0 16px 0;"><strong>Confidence Top-1:</strong> {safe_confidence}</p>
            <p style="margin:0 0 16px 0;"><strong>Rekomendasi Berdasarkan Rubrik:</strong> {logic_top}</p>

            <h3 style="margin:0 0 10px 0; color:#0f766e;">Top 3 Probability</h3>
            <table style="border-collapse:collapse; width:100%; margin-bottom:16px;">
              <thead>
                <tr>
                  <th style="padding:8px 10px; background:#f8fafc; border:1px solid #dbe3eb; text-align:left;">Divisi</th>
                  <th style="padding:8px 10px; background:#f8fafc; border:1px solid #dbe3eb; text-align:center;" colspan="2">Grafik (0-100%)</th>
                </tr>
              </thead>
              <tbody>{top3_chart_rows}</tbody>
            </table>

            <h3 style="margin:0 0 10px 0; color:#0f766e;">Top 3 Rubrik Pendukung</h3>
            <table style="border-collapse:collapse; width:100%; margin-bottom:16px;">
              <thead>
                <tr>
                  <th style="padding:8px 10px; background:#f8fafc; border:1px solid #dbe3eb; text-align:center;">Rank</th>
                  <th style="padding:8px 10px; background:#f8fafc; border:1px solid #dbe3eb; text-align:left;">Divisi</th>
                  <th style="padding:8px 10px; background:#f8fafc; border:1px solid #dbe3eb; text-align:center;" colspan="2">Grafik Skor</th>
                </tr>
              </thead>
              <tbody>{logic_chart_rows}</tbody>
            </table>

            <p style="margin:0; font-size:13px; color:#6b7280; line-height:1.5;">
              Catatan: hasil rekomendasi ini bersifat pendukung keputusan, bukan keputusan final.
            </p>
          </div>
          <div style="margin-top:12px; padding:10px 14px; background:#1f2937; color:#e5e7eb; border-radius:12px; font-size:12px;">
            Email ini dikirim otomatis dari Sistem Rekomendasi Divisi Pramuka.
          </div>
        </div>
      </body>
    </html>
    """
    return text_body, html_body


def _send_recommendation_email(
    recipient_email: str,
    nama: str,
    kelas: str,
    recommendation: str,
    top3: list[tuple[str, float]],
    logic_ranked: list[tuple[str, float]],
    confidence: float,
) -> tuple[bool | None, str]:
    if not recipient_email.strip():
        return None, "NoRecipient"
    if not _is_valid_email(recipient_email):
        return False, "InvalidEmail"

    config = _resolve_smtp_config()
    if config is None:
        return None, "SmtpNotConfigured"

    text_body, html_body = _build_recommendation_email_html(
        recipient_email=recipient_email,
        nama=nama,
        kelas=kelas,
        recommendation=recommendation,
        top3=top3,
        logic_ranked=logic_ranked,
        confidence=confidence,
    )

    message = EmailMessage()
    message["Subject"] = "Hasil Rekomendasi Divisi Pramuka"
    message["From"] = config["sender"]
    message["To"] = recipient_email
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(config["host"], config["port"], timeout=12) as smtp:
            if config["use_tls"]:
                smtp.starttls(context=context)
            smtp.login(config["username"], config["password"])
            smtp.send_message(message)
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


def _load_gcp_credentials():
    env_path = _secret_or_env(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "GCP_SERVICE_ACCOUNT_FILE",
        "CREDENTIALS_PATH",
        default=None,
    )
    if env_path and Path(env_path).exists():
        return Path(env_path)

    for path_text in (
        "service_account.json",
        "credentials.json",
    ):
        candidate = Path(path_text)
        if candidate.exists():
            return candidate

    try:
        secrets_paths = (
            st.secrets.get("gcp_service_account_path"),
            st.secrets.get("google_service_account_path"),
            st.secrets.get("SERVICE_ACCOUNT_PATH"),
        )
        for candidate_text in secrets_paths:
            if not candidate_text:
                continue
            candidate = Path(str(candidate_text))
            if candidate.exists():
                return candidate
    except Exception:
        pass

    try:
        info = st.secrets.get("gcp_service_account")
        if info:
            return dict(info)
        if isinstance(info, str):
            return info
    except Exception:
        pass

    return None


def _init_connection():
    source = _load_gcp_credentials()
    if source is None:
        return None

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    if isinstance(source, Path):
        try:
            creds = Credentials.from_service_account_file(str(source), scopes=scopes)
        except Exception:
            return None
    else:
        try:
            creds = Credentials.from_service_account_info(source, scopes=scopes)
        except Exception:
            return None

    try:
        return gspread.authorize(creds)
    except Exception:
        return None


def _build_sheet_header() -> list[str]:
    return [
        "Nama Lengkap",
        "Kelas",
        "Status",
        "Waktu Input",
        "Rekomendasi Dari Form",
        "Rekomendasi Model",
        *FEATURE_COLUMNS,
        "Rekomendasi Final",
    ]


def _ensure_sheet_header(sheet) -> bool:
    expected_markers = {"Nama Lengkap", "Kelas", "Status", "Waktu Input"}
    header = _build_sheet_header()
    current = [str(cell).strip() for cell in sheet.row_values(1)]
    if not current:
        try:
            sheet.insert_row(header, index=1, value_input_option="USER_ENTERED")
            return True
        except Exception:
            return False

    has_markers = any(marker in set(current) for marker in expected_markers)
    if not has_markers:
        try:
            sheet.insert_row(header, index=1, value_input_option="USER_ENTERED")
            return True
        except Exception:
            return False
    return False


def _append_to_sheet(payload: dict[str, Any], recommendation: str) -> tuple[bool | None, str]:
    sheet_id = _resolve_spreadsheet_id()
    if not sheet_id:
        return None, "NoSpreadsheetId"

    client = _init_connection()
    if client is None:
        return None, "NoConnection"

    try:
        sheet = client.open_by_key(sheet_id).sheet1
        if sheet is None:
            return False, "WorksheetNotFound"

        _ensure_sheet_header(sheet)

        row = [
            payload.get("Nama Lengkap", ""),
            payload.get("Kelas", ""),
            payload.get("Status", "Calon Dewan"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            payload.get("recommended_division", ""),
        ]
        for col in FEATURE_COLUMNS:
            row.append(payload.get(col, ""))
        row.append(recommendation)
        sheet.append_row(row, value_input_option="USER_ENTERED")
        return True, "OK"
    except gspread.exceptions.APIError as exc:
        msg = str(exc).lower()
        if "permission" in msg or "forbidden" in msg or "insufficient" in msg:
            return False, "PermissionDenied: service account belum dibagikan dengan akses Editor."
        detail = _safe_error_text(exc)
        return False, f"GoogleAPIError: {detail}"
    except PermissionError as exc:
        message = _safe_error_text(exc).lower()
        if "sheets api has not been used" in message or "google sheets api has not been enabled" in message:
            return (
                False,
                "SheetsApiNotEnabled: aktifkan Google Sheets API di Google Cloud Project (APIs & Services > Library).",
            )
        if "permission" in message or "forbidden" in message or "insufficient" in message:
            return False, "PermissionDenied: service account belum dibagikan dengan akses Editor."
        if "not found" in message or "not accessible" in message:
            return False, "SpreadsheetNotFound: sheet_id tidak ditemukan / tidak boleh diakses."
        return False, f"PermissionError: {_safe_error_text(exc)}"
    except gspread.exceptions.SpreadsheetNotFound:
        return False, "SpreadsheetNotFound"
    except gspread.exceptions.WorksheetNotFound:
        return False, "WorksheetNotFound: tidak menemukan sheet pertama (sheet1) di spreadsheet."
    except Exception as exc:
        details = _safe_error_text(exc)
        return False, f"RuntimeError: {details}"


def _load_artifact() -> dict[str, Any] | None:
    try:
        with open(MODEL_ARTIFACT_PATH, "rb") as f:
            artifact = pickle.load(f)
    except FileNotFoundError:
        st.error("model_artifacts.pkl belum ada. Jalankan `python train_model.py` terlebih dahulu.")
        return None
    except Exception as exc:
        st.error(f"Gagal memuat artifact: {exc}")
        return None

    required = {
        "pipeline",
        "target_col",
        "feature_columns",
        "numeric_columns",
        "categorical_columns",
        "class_names",
        "metrics",
        "dataset_schema",
        "created_at",
    }
    missing = required - set(artifact.keys())
    if missing:
        st.error(f"Artifact tidak valid, missing: {', '.join(sorted(missing))}")
        return None

    return artifact


def _validate_row(row: pd.Series, artifact: dict[str, Any]) -> dict[str, Any]:
    feature_columns = artifact["feature_columns"]
    numeric_columns = set(artifact["numeric_columns"])
    category_columns = set(artifact["categorical_columns"])
    category_values = artifact.get("dataset_schema", {}).get("category_values", {})

    clean: dict[str, Any] = {}
    for col in feature_columns:
        if col not in row.index:
            raise ValueError(f"Kolom '{col}' tidak ditemukan.")

        raw = row[col]

        if col in numeric_columns:
            val = pd.to_numeric(raw, errors="coerce")
            if pd.isna(val):
                raise ValueError(f"{col} wajib angka.")
            int_val = int(round(float(val)))
            if int_val < NUMERIC_SCALE_MIN or int_val > NUMERIC_SCALE_MAX:
                raise ValueError(f"{col} harus dalam skala 1–5.")
            clean[col] = int_val
            continue

        val = "" if pd.isna(raw) else str(raw).strip()
        if not val:
            raise ValueError(f"{col} wajib diisi.")
        allowed = category_values.get(col)
        if allowed and val not in allowed:
            raise ValueError(f"{col}: nilai tidak valid. Pilihan: {', '.join(allowed)}")
        clean[col] = val

    return clean


def _predict_one(model, data: pd.DataFrame) -> Tuple[str, List[Tuple[str, float]]]:
    pred = str(model.predict(data)[0])
    top3: List[Tuple[str, float]] = []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data)[0]
        classes = list(model.classes_)
        idx = np.argsort(proba)[::-1][:3]
        top3 = [(str(classes[i]), float(proba[i])) for i in idx]
    return pred, top3


def _manual_scoring(values: dict[str, Any]) -> Tuple[list[tuple[str, float]], Dict[str, list[str]]]:
    scores: Dict[str, float] = {k: 0.0 for k in TARGET_LABELS}
    reasons: Dict[str, list[str]] = {k: [] for k in TARGET_LABELS}

    for divisi, rules in MANUAL_RULES.items():
        total = 0.0
        for feature, weight, ftype, expect in rules:
            val = values.get(feature)
            if val is None:
                continue

            if ftype == "numeric":
                num = int(float(val))
                if expect == "high" and num >= 4:
                    total += weight
                    reasons[divisi].append(f"{feature} bernilai {num} (tinggi)")
                elif expect == "low" and num <= 2:
                    total += weight
                    reasons[divisi].append(f"{feature} bernilai {num} (rendah)")
            else:
                if str(val) == expect:
                    total += weight
                    reasons[divisi].append(f"{feature} sesuai: {val}")

        scores[divisi] = round(total, 3)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked, reasons


def _format_percentage_table(top3: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Divisi": divisi, "Probability": f"{prob*100:.2f}%"} for divisi, prob in top3]
    )


def _build_sidebar() -> str:
    st.sidebar.subheader("Pengaturan")
    operator = st.sidebar.text_input("Nama Operator", value="Operator")
    email_ready = _resolve_smtp_config() is not None
    sheet_id = _resolve_spreadsheet_id()

    if email_ready:
        st.sidebar.success("Email (SMTP): aktif")
    else:
        st.sidebar.warning("Email (SMTP): belum siap (cek `.env` atau `st.secrets`).")

    if not sheet_id:
        st.sidebar.warning("Google Sheets: sheet ID belum disediakan.")
    else:
        credentials_ready = _load_gcp_credentials() is not None
        if not credentials_ready:
            st.sidebar.warning("Google Sheets: sheet ID ada, tapi credentials belum disiapkan.")
        else:
            sheet_client = _init_connection()
            if sheet_client is None:
                st.sidebar.warning("Google Sheets: credential ada, tetapi otentikasi gagal.")
            else:
                try:
                    _ = sheet_client.open_by_key(sheet_id).sheet1
                except PermissionError as exc:
                    msg = _safe_error_text(exc).lower()
                    if "sheets api has not been used" in msg or "google sheets api has not been enabled" in msg:
                        st.sidebar.error("Google Sheets: API belum aktif (Google Sheets API perlu diaktifkan).")
                    elif "permission" in msg or "forbidden" in msg or "insufficient" in msg:
                        st.sidebar.warning("Google Sheets: sheet ID ada, tapi credentials belum shared sebagai Editor.")
                    elif "not found" in msg:
                        st.sidebar.warning("Google Sheets: sheet ID tidak ditemukan.")
                    else:
                        st.sidebar.warning(f"Google Sheets: belum siap ({_safe_error_text(exc)}).")
                except Exception as exc:
                    st.sidebar.warning(f"Google Sheets: belum siap ({_safe_error_text(exc)}).")
                else:
                    st.sidebar.success("Google Sheets: aktif")

    st.sidebar.markdown("- Status input: otomatis `Calon Dewan`")
    if st.sidebar.button("Muat ulang model"):
        st.rerun()
    return operator


def _render_warning_if_class_imbalance(artifact: dict[str, Any]) -> None:
    dist = artifact.get("dataset_schema", {}).get("class_distribution", {})
    weak = [f"{k}: {v}" for k, v in dist.items() if v < 50]
    if weak:
        st.warning("Peringatan: kelas dengan data < 50: " + ", ".join(weak))


def _build_evaluation_metrics(artifact: dict[str, Any]) -> pd.DataFrame:
    metrics = artifact.get("metrics", {})
    holdout = metrics.get("holdout", {})
    rows = [
        ("Accuracy", holdout.get("accuracy", 0)),
        ("Precision (Micro)", holdout.get("precision_micro", 0)),
        ("Precision (Macro)", holdout.get("precision_macro", 0)),
        ("Recall (Macro)", holdout.get("recall_macro", 0)),
        ("F1 (Macro)", holdout.get("f1_macro", 0)),
    ]
    return pd.DataFrame(rows, columns=["Metrik", "Nilai"])


def _build_confusion_matrix(artifact: dict[str, Any]) -> pd.DataFrame:
    holdout = artifact.get("metrics", {}).get("holdout", {})
    labels = artifact.get("class_names", [])
    matrix = holdout.get("confusion_matrix", [])
    if not matrix:
        return pd.DataFrame()
    return pd.DataFrame(matrix, columns=labels, index=labels)


def _build_distribution_df(artifact: dict[str, Any]) -> pd.DataFrame:
    dist = artifact.get("dataset_schema", {}).get("class_distribution", {})
    rows = [[k, int(v)] for k, v in dist.items()]
    return pd.DataFrame(rows, columns=["Divisi", "Jumlah"])


@st.cache_resource
def _cached_load_artifact() -> dict[str, Any] | None:
    _load_env()
    return _load_artifact()


st.markdown(_style(), unsafe_allow_html=True)
st.markdown("<span class='pill'>Sistem Rekomendasi Divisi Pramuka</span>", unsafe_allow_html=True)
st.title("Sistem Rekomendasi Divisi Pramuka")
st.caption("Pipeline: ColumnTransformer + OneHotEncoder + model pilihan terbaik dari cross-validation.")

artifact = _cached_load_artifact()
if artifact is None:
    st.stop()

pipeline = artifact["pipeline"]
TARGET_COL = artifact["target_col"]
FEATURE_COLUMNS = artifact["feature_columns"]
NUMERIC_COLUMNS = set(artifact["numeric_columns"])
CAT_COLUMNS = set(artifact["categorical_columns"])
CLASS_NAMES = artifact["class_names"]
SCHEMA = artifact["dataset_schema"]
CREATED_AT = artifact["created_at"]
CATEGORY_VALUES = SCHEMA.get("category_values", {})

if "session_history" not in st.session_state:
    st.session_state["session_history"] = []

operator_name = _build_sidebar()

st.markdown("<div class='app-card'>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Model terpilih", artifact.get("best_model_name", "-"))
with m2:
    st.metric("Jumlah data latih", int(SCHEMA.get("rows", 0)))
with m3:
    st.metric("Jumlah fitur", len(FEATURE_COLUMNS))
with m4:
    st.metric("Model dibuat", CREATED_AT)
st.markdown("</div>", unsafe_allow_html=True)

_render_warning_if_class_imbalance(artifact)

dist_df = _build_distribution_df(artifact)
st.markdown("<div class='app-card'><div class='section-title'>Distribusi Data per Divisi</div>", unsafe_allow_html=True)
if dist_df.empty:
    st.info("Distribusi kelas tidak ditemukan di artifact.")
else:
    st.bar_chart(dist_df.set_index("Divisi")["Jumlah"], use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Dashboard Evaluasi")
metric_df = _build_evaluation_metrics(artifact)
cols = st.columns(5)
for col, (_, row) in zip(cols, metric_df.iterrows()):
    col.metric(row["Metrik"], f"{row['Nilai'] * 100:.2f}%")

with st.expander("Cross Validation (5-Fold)"):
    cv_rows = artifact.get("metrics", {}).get("cv_comparison", [])
    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        st.dataframe(
            cv_df.rename(
                columns={
                    "model_name": "Model",
                    "accuracy_mean": "Accuracy",
                    "precision_macro_mean": "Precision Macro",
                    "recall_macro_mean": "Recall Macro",
                    "f1_macro_mean": "F1 Macro",
                }
            ).style.format(
                {"Accuracy": "{:.4f}", "Precision Macro": "{:.4f}", "Recall Macro": "{:.4f}", "F1 Macro": "{:.4f}"}
            ),
            use_container_width=True,
        )
    else:
        st.info("Hasil CV belum tersedia.")

with st.expander("Confusion Matrix Hold-out"):
    cm_df = _build_confusion_matrix(artifact)
    if cm_df.empty:
        st.info("Confusion matrix tidak tersedia.")
    else:
        st.dataframe(cm_df, use_container_width=True)


tab_pred, tab_batch = st.tabs(["Prediksi Individu", "Upload Batch CSV"])

with tab_pred:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Input Peserta</div>", unsafe_allow_html=True)
    with st.form("form_single"):
        c1, c2 = st.columns(2)
        with c1:
            nama = st.text_input("Nama Lengkap")
        with c2:
            kelas = st.text_input("Kelas")
            email = st.text_input("Email (opsional)")

        st.text_input("Status", value=FORCED_STATUS, disabled=True)
        st.caption("Skala numerik: 1 (sangat tidak setuju) sampai 5 (sangat setuju).")

        left, right = st.columns(2)
        input_data: dict[str, Any] = {"Status": FORCED_STATUS}
        for i, col in enumerate(FEATURE_COLUMNS):
            if col == "Status":
                continue
            with left if i % 2 == 0 else right:
                if col in NUMERIC_COLUMNS:
                    input_data[col] = st.slider(col, 1, 5, 3)
                else:
                    options = CATEGORY_VALUES.get(col)
                    if not options:
                        options = ["Ya", "Tidak"]
                    input_data[col] = st.selectbox(col, options)

        submit = st.form_submit_button("Proses Rekomendasi", type="primary")

    if submit:
        errs: list[str] = []
        if not nama.strip():
            errs.append("Nama Lengkap wajib diisi.")
        if not kelas.strip():
            errs.append("Kelas wajib diisi.")

        try:
            clean_input = _validate_row(pd.Series(input_data), artifact)
        except Exception as exc:
            errs.append(str(exc))

        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        X = pd.DataFrame([clean_input], columns=FEATURE_COLUMNS)
        pred, top3 = _predict_one(pipeline, X)

        st.session_state["session_history"].append(
            {
                "waktu": datetime.now(),
                "nama": nama,
                "kelas": kelas,
                "hasil": pred,
                "confidence": top3[0][1] if top3 else 0.0,
            }
        )

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader(f"Hasil: {pred}")
        st.write(f"Peserta: {nama} ({kelas})")
        st.write(f"Rekomendasi oleh model: **{artifact.get('best_model_name', 'Model')}**")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='app-card'><div class='section-title'>Top 3 Probability (Model)</div>", unsafe_allow_html=True)
        st.table(_format_percentage_table(top3))
        st.markdown("</div>", unsafe_allow_html=True)

        ranked_logic, logic_reasons = _manual_scoring(clean_input)
        st.markdown("<div class='app-card'><div class='section-title'>Alasan Logika Rekomendasi</div>", unsafe_allow_html=True)
        for divisi, score in ranked_logic:
            st.markdown(f"**{divisi}** — skor {score:.2f}")
            for reason in logic_reasons.get(divisi, []):
                st.markdown(f"<div class='reason-item'>• {reason}</div>", unsafe_allow_html=True)
        if ranked_logic:
            st.caption(f"Rekomendasi logika: {ranked_logic[0][0]}")
        st.markdown("</div>", unsafe_allow_html=True)

        top_model = [i[0] for i in top3[:3]]
        top_logic = [i[0] for i in ranked_logic[:3]]
        match = len(set(top_model).intersection(top_logic))
        st.info(f"Kesesuaian Top-3: {match}/3")

        if email.strip():
            sent, email_status = _send_recommendation_email(
                recipient_email=email.strip(),
                nama=nama,
                kelas=kelas,
                recommendation=pred,
                top3=top3,
                logic_ranked=ranked_logic,
                confidence=top3[0][1] if top3 else 0.0,
            )
            if sent is True:
                st.success("Hasil rekomendasi berhasil dikirim ke email.")
            elif email_status == "SmtpNotConfigured":
                st.warning("Email tidak dikirim: SMTP belum disiapkan. Isi SMTP di `.env`/`st.secrets` untuk mengaktifkan.")
            elif email_status == "InvalidEmail":
                st.warning("Format email tidak valid.")
            else:
                st.warning(f"Gagal mengirim email: {email_status}")
        else:
            st.info("Email peserta kosong, hasil tidak dikirim melalui email.")

        payload_row = {
            "Nama Lengkap": nama,
            "Kelas": kelas,
            "Status": "Calon Dewan",
            **clean_input,
            "recommended_division": pred,
        }
        saved, status = _append_to_sheet(payload_row, pred)
        if saved is True:
            st.success("Data rekomendasi tersimpan di Google Sheets.")
        elif saved is None:
            st.info("Google Sheets belum aktif (credential/ID belum disiapkan).")
        else:
            st.warning(f"Gagal menyimpan ke sheet: {status}")
            if status.startswith("PermissionDenied") or "permission" in status.lower():
                st.info("Periksa apakah service account ini sudah di-share sebagai Editor pada spreadsheet.")
            elif status.startswith("SheetsApiNotEnabled"):
                st.info("Aktifkan Google Sheets API pada Google Cloud project service account.")
            elif status.startswith("SpreadsheetNotFound"):
                st.info("Pastikan `SPREADSHEET_ID` benar dan sheet tidak dibatasi akses.")
            elif status.startswith("WorksheetNotFound"):
                st.info("Pastikan ada sheet pertama (Sheet1) di Google Spreadsheet.")
                st.info("Jika nama Sheet tidak otomatis `Sheet1`, rename sheet pertama menjadi `Sheet1` atau buat ulang sheet.")

with tab_batch:
    st.markdown("<div class='app-card'><div class='section-title'>Upload Batch (CSV)</div>", unsafe_allow_html=True)
    st.caption("Kolom fitur mengikuti `feature_columns` artifact. Kolom target dapat ikut ada, akan diabaikan.")
    st.caption("Untuk kirim email per peserta, siapkan kolom `Email` (opsional).")
    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".xlsx"):
                batch_raw = pd.read_excel(uploaded)
            else:
                batch_raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Format file tidak valid: {exc}")
            batch_raw = None

        if batch_raw is not None:
            batch_df = batch_raw.copy()
            if TARGET_COL in batch_df.columns:
                batch_df = batch_df.drop(columns=[TARGET_COL])
            for req in ["Nama Lengkap", "Kelas"]:
                if req not in batch_df.columns:
                    batch_df[req] = ""
            if "Status" in FEATURE_COLUMNS and "Status" not in batch_df.columns:
                batch_df["Status"] = FORCED_STATUS

            missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
            if missing:
                st.error("Kolom fitur belum lengkap: " + ", ".join(missing))
            else:
                send_email_batch = st.checkbox("Kirim hasil ke email peserta (gunakan kolom `Email` jika tersedia)", value=False)
                if st.button("Proses Batch"):
                    email_send_targets = []
                    if send_email_batch:
                        email_send_targets = [c for c in batch_df.columns if str(c).strip().lower() in {"email", "alamat email", "email peserta"}]

                    if send_email_batch and not email_send_targets:
                        st.info("Batch email aktif, tetapi kolom email tidak ditemukan. Pastikan ada kolom `Email`/`Alamat Email`/`Email Peserta`.")

                    results = []
                    failed = []
                    email_summary = {
                        "total_with_email": 0,
                        "success": 0,
                        "invalid": 0,
                        "not_ready": 0,
                        "other_error": 0,
                        "not_filled": 0,
                    }
                    email_fail_rows = []
                    for idx, raw_row in batch_df.iterrows():
                        try:
                            clean = _validate_row(raw_row, artifact)
                            X = pd.DataFrame([clean], columns=FEATURE_COLUMNS)
                            pred, top3 = _predict_one(pipeline, X)
                            ranked_logic, _ = _manual_scoring(clean)
                            sent_status = "Tidak dikirim"
                            email_address = ""
                            if send_email_batch and email_send_targets:
                                email_address = str(raw_row.get(email_send_targets[0], "") or "").strip()
                                if not email_address:
                                    email_summary["not_filled"] += 1
                                else:
                                    email_summary["total_with_email"] += 1
                                    sent, status = _send_recommendation_email(
                                        recipient_email=email_address,
                                        nama=str(raw_row.get("Nama Lengkap", "")),
                                        kelas=str(raw_row.get("Kelas", "")),
                                        recommendation=pred,
                                        top3=top3,
                                        logic_ranked=ranked_logic,
                                        confidence=top3[0][1] if top3 else 0.0,
                                    )
                                    if sent is True:
                                        sent_status = "Terkirim"
                                        email_summary["success"] += 1
                                    elif status == "SmtpNotConfigured":
                                        sent_status = "Gagal: SMTP belum siap"
                                        email_summary["not_ready"] += 1
                                    elif status == "InvalidEmail":
                                        sent_status = "Gagal: format email tidak valid"
                                        email_summary["invalid"] += 1
                                        email_fail_rows.append((idx + 2, email_address, "Email tidak valid"))
                                    else:
                                        sent_status = f"Gagal: {status}"
                                        email_summary["other_error"] += 1
                                        email_fail_rows.append((idx + 2, email_address, status))

                            results.append(
                                {
                                    "Nama Lengkap": raw_row.get("Nama Lengkap", ""),
                                    "Kelas": raw_row.get("Kelas", ""),
                                    "Email": email_address,
                                    "Rekomendasi": pred,
                                    "Status Email": sent_status,
                                    "Confidence": f"{(top3[0][1]*100):.2f}%" if top3 else "",
                                    "Top 1": top3[0][0] if len(top3) > 0 else "",
                                    "Top 2": top3[1][0] if len(top3) > 1 else "",
                                    "Top 3": top3[2][0] if len(top3) > 2 else "",
                                }
                            )
                        except Exception as exc:
                            failed.append((idx + 2, str(exc)))

                    if send_email_batch:
                        if email_summary["not_ready"] > 0:
                            st.warning("Beberapa email gagal dikirim karena SMTP belum disiapkan.")
                        if email_summary["invalid"] > 0 or email_summary["other_error"] > 0:
                            st.warning(f"Email error: invalid {email_summary['invalid']}, lainnya {email_summary['other_error']}.")
                        if email_summary["success"] > 0 or email_summary["not_filled"] > 0:
                            st.info(
                                "Email batch: "
                                f"terkirim {email_summary['success']}, "
                                f"email kosong/tidak terisi {email_summary['not_filled']}, "
                                f"total dengan email {email_summary['total_with_email']}."
                            )
                        if email_fail_rows:
                            st.warning("Contoh kegagalan pengiriman email:")
                            for row_no, addr, reason in email_fail_rows[:5]:
                                st.caption(f"Baris {row_no} ({addr}): {reason}")

                    if failed:
                        st.warning(f"{len(failed)} baris gagal.")
                        for row_no, msg in failed[:5]:
                            st.caption(f"Baris {row_no}: {msg}")

                    out_df = pd.DataFrame(results)
                    if not out_df.empty:
                        st.success(f"Sukses prediksi {len(out_df)} baris.")
                        st.dataframe(out_df, use_container_width=True)
                        st.download_button(
                            "Download Hasil Batch CSV",
                            data=out_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"hasil_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Tidak ada data valid untuk diprediksi.")

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state["session_history"]:
    st.subheader("Riwayat Sesi")
    st.dataframe(
        pd.DataFrame(st.session_state["session_history"]).sort_values("waktu", ascending=False),
        use_container_width=True,
    )
