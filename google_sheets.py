
"""
Google Sheets utilities.

This module centralizes all Google Sheets access used by the original Colab notebook.
Functions are annotated with READS SHEETS or WRITES SHEETS in docstrings.

Setup options:
1) Colab OAuth (interactive)
2) Service Account (non-interactive)

Never commit credentials. Use environment variables or a .env file (see .env.example).
"""
from __future__ import annotations
import os
import pandas as pd
from typing import Optional

# Colab OAuth
try:
    from google.colab import auth as colab_auth  # type: ignore
    from google.auth import default as colab_default  # type: ignore
except Exception:
    colab_auth = None
    colab_default = None

import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def client_from_service_account(json_path: str) -> gspread.Client:
    """Return an authorized gspread client using a service account (no interaction)."""
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_path, SCOPES)
    return gspread.authorize(creds)

def client_from_colab() -> gspread.Client:
    """Authorize a gspread client using Colab OAuth (interactive)."""
    if colab_auth is None or colab_default is None:
        raise RuntimeError("Colab auth not available outside Colab.")
    colab_auth.authenticate_user()
    creds, _ = colab_default()
    return gspread.authorize(creds)

def read_sheet_to_df(sheet_url_or_key: str, worksheet: str, client: Optional[gspread.Client]=None) -> pd.DataFrame:
    """READS SHEETS: Load a worksheet into a pandas DataFrame.

    Args:
        sheet_url_or_key: Full URL or the Sheet key.
        worksheet: Tab name.
        client: Optional authorized gspread client. If None, tries service account via ENV GSERVICE_JSON.
    """
    if client is None:
        json_path = os.getenv("GSERVICE_JSON")
        if not json_path:
            raise RuntimeError("Provide gspread client or set GSERVICE_JSON to a service account json path.")
        client = client_from_service_account(json_path)

    if sheet_url_or_key.startswith("http"):
        ss = client.open_by_url(sheet_url_or_key)
    else:
        ss = client.open_by_key(sheet_url_or_key)
    ws = ss.worksheet(worksheet)
    df = get_as_dataframe(ws)
    return df.dropna(how="all")

def write_df_to_sheet(df: pd.DataFrame, sheet_url_or_key: str, worksheet: str, client: Optional[gspread.Client]=None) -> None:
    """WRITES SHEETS: Upload/overwrite a pandas DataFrame to a worksheet.
    WARNING: This replaces the sheet's contents.
    """
    if client is None:
        json_path = os.getenv("GSERVICE_JSON")
        if not json_path:
            raise RuntimeError("Provide gspread client or set GSERVICE_JSON to a service account json path.")
        client = client_from_service_account(json_path)

    if sheet_url_or_key.startswith("http"):
        ss = client.open_by_url(sheet_url_or_key)
    else:
        ss = client.open_by_key(sheet_url_or_key)

    try:
        ws = ss.worksheet(worksheet)
    except gspread.exceptions.WorksheetNotFound:
        ws = ss.add_worksheet(title=worksheet, rows="100", cols="26")

    ws.clear()
    set_with_dataframe(ws, df)
