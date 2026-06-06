import os
import re
import json
from pathlib import Path

import gspread
import pandas as pd
import streamlit as st

try:
    from oauth2client.service_account import ServiceAccountCredentials as OAuth2ClientServiceAccountCredentials
    _HAS_OAUTH2CLIENT = True
except Exception:  # pragma: no cover
    OAuth2ClientServiceAccountCredentials = None
    _HAS_OAUTH2CLIENT = False

from google.oauth2.service_account import Credentials as GoogleServiceAccountCredentials


SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
DEFAULT_SERVICE_ACCOUNT = "service_account.json"
DEFAULT_SPREADSHEET_ID = "1WH79eI-Se-6TPJyhWCWmNnh8fWDtVHGnGrfCRXzKpBY"


def _load_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip().strip('"').strip("'"))


def _secret_or_env(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        if key in os.environ and os.environ[key].strip():
            return os.environ[key].strip()

    try:
        for key in keys:
            if key in st.secrets:
                value = st.secrets[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
    except Exception:
        pass
    return default


def _extract_sheet_id(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    if "spreadsheets/d/" in raw:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", raw)
        if match:
            return match.group(1)
    return raw


def _resolve_service_account_path() -> str:
    env_path = _secret_or_env(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "GCP_SERVICE_ACCOUNT_FILE",
        default=DEFAULT_SERVICE_ACCOUNT,
    )
    return str(Path(env_path))


def _resolve_spreadsheet_id(raw_input: str) -> str:
    env_value = _secret_or_env(
        "SPREADSHEET_ID",
        "GOOGLE_SHEET_ID",
        "SHEET_ID",
        default=raw_input,
    )
    return _extract_sheet_id(env_value or "")


def _is_google_sheets_permission_error(exc: Exception) -> bool:
    message = str(exc).lower()
    keywords = (
        "permission",
        "insufficient",
        "not have permission",
        "forbidden",
        "unauthorized",
        "request had insufficient authentication scopes",
    )
    return any(k in message for k in keywords)


@st.cache_resource(show_spinner=False)
def _build_gspread_client(service_account_path: str) -> gspread.Client:
    cred_path = Path(service_account_path)
    if not cred_path.exists():
        raise FileNotFoundError(f"File credential service account tidak ditemukan: {cred_path}")

    try:
        if _HAS_OAUTH2CLIENT:
            creds = OAuth2ClientServiceAccountCredentials.from_json_keyfile_name(
                str(cred_path), scopes=SCOPE
            )
        else:
            creds = GoogleServiceAccountCredentials.from_service_account_file(
                str(cred_path), scopes=SCOPE
            )
        return gspread.authorize(creds)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Format JSON service account tidak valid: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gagal memuat credentials: {exc}") from exc


@st.cache_data(show_spinner=True)
def _load_sheet1_data(sheet_id: str, service_account_path: str) -> pd.DataFrame:
    client = _build_gspread_client(service_account_path)
    try:
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.sheet1
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        if df.empty:
            # fallback bila sheet berisi header kosong tapi ada baris data
            raw = worksheet.get_all_values()
            if raw:
                header = raw[0]
                rows = raw[1:]
                df = pd.DataFrame(rows, columns=header)
            else:
                return pd.DataFrame()
        return df
    except gspread.exceptions.APIError as exc:
        if _is_google_sheets_permission_error(exc):
            raise PermissionError(
                "Akses ditolak ke Google Sheet. "
                "Pastikan service account sudah dibagikan sebagai Editor pada sheet."
            ) from exc
        raise
    except gspread.exceptions.SpreadsheetNotFound as exc:
        raise FileNotFoundError("Spreadsheet ID tidak ditemukan atau tidak dapat diakses.") from exc


def _refresh_sheet_data() -> None:
    st.cache_data.clear()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Google Sheets Viewer", layout="wide")
    st.title("Google Sheets Reader (sheet1)")
    st.caption("Menampilkan data dari sheet pertama (sheet1) Google Spreadsheet.")

    _load_env()

    default_sheet_id = _resolve_spreadsheet_id(DEFAULT_SPREADSHEET_ID)
    service_account_path = _resolve_service_account_path()

    st.sidebar.header("Konfigurasi")
    st.sidebar.text_input("Path Service Account JSON", value=service_account_path, key="service_account_path")
    st.sidebar.text_input(
        "Spreadsheet ID / URL",
        value=default_sheet_id,
        key="spreadsheet_id_input",
        help="Masukkan Spreadsheet ID atau URL lengkap Google Spreadsheet",
    )
    st.sidebar.button("Refresh Sheet", on_click=_refresh_sheet_data)

    path = st.session_state["service_account_path"]
    raw_sheet_id = st.session_state["spreadsheet_id_input"]
    sheet_id = _resolve_spreadsheet_id(raw_sheet_id)

    if not path:
        st.error("Path credential belum diset. Isi path file `service_account.json` di input sidebar.")
        return
    if not Path(path).exists():
        st.error(f"File credential tidak ditemukan: {path}")
        return
    if not sheet_id:
        st.error("Spreadsheet ID belum diisi.")
        return

    with st.expander("Info koneksi", expanded=True):
        st.write(f"Service account path: `{path}`")
        st.write(f"Spreadsheet ID: `{sheet_id}`")

    try:
        df = _load_sheet1_data(sheet_id=sheet_id, service_account_path=path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except PermissionError as exc:
        st.error(str(exc))
        st.info("Solusi: bagikan spreadsheet ke email service account (di file JSON) dengan akses Editor.")
        return
    except Exception as exc:
        st.error(f"Gagal memuat data: {exc}")
        return

    st.success(f"Data berhasil dimuat dari sheet1: {len(df)} baris")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
