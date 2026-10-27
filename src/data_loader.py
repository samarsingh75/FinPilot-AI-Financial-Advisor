"""Loading and validating transaction CSV files."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

import config

REQUIRED_COLUMNS = ["date", "description", "category", "type", "amount"]
VALID_TYPES = {"income", "expense"}
MAX_REPORTED_ERRORS = 5

Source = Union[str, Path, bytes, BinaryIO]


class DataValidationError(ValueError):
    """Raised when an uploaded CSV is missing columns or has invalid values."""


def _read_raw(source: Source) -> pd.DataFrame:
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise DataValidationError(f"File not found: {path.name}")
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = source.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
        if len(data) > config.MAX_CSV_MB * 1024 * 1024:
            raise DataValidationError(f"File is larger than {config.MAX_CSV_MB} MB.")
        if not data.strip():
            raise DataValidationError("The file is empty.")
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except DataValidationError:
        raise
    except pd.errors.EmptyDataError:
        raise DataValidationError("The file is empty or has no header row.") from None
    except UnicodeDecodeError:
        raise DataValidationError("The file is not valid UTF-8 text. Save it as a UTF-8 CSV.") from None
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"The file could not be parsed as CSV: {exc}") from None


def validate_and_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate column names and value types; return a clean DataFrame."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Expected columns: " + ", ".join(REQUIRED_COLUMNS) + "."
        )
    df = df[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    # Drop completely blank rows.
    df = df[~(df == "").all(axis=1)].reset_index(drop=True)
    if df.empty:
        raise DataValidationError("The file has a header but no transactions.")

    problems: list[str] = []

    def row_label(idx: pd.Index) -> str:
        rows = [str(i + 2) for i in idx[:MAX_REPORTED_ERRORS]]  # +2: header line + 1-based
        more = "" if len(idx) <= MAX_REPORTED_ERRORS else f" (and {len(idx) - MAX_REPORTED_ERRORS} more)"
        return "row(s) " + ", ".join(rows) + more

    # date
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed", dayfirst=False)
    bad = df.index[dates.isna()]
    if len(bad):
        problems.append(f"Invalid or empty date in {row_label(bad)}. Use YYYY-MM-DD.")

    # type
    types = df["type"].str.lower()
    bad = df.index[~types.isin(VALID_TYPES)]
    if len(bad):
        problems.append(f"'type' must be 'income' or 'expense' in {row_label(bad)}.")

    # amount
    cleaned_amount = df["amount"].str.replace(",", "", regex=False).str.replace(
        config.CURRENCY_SYMBOL, "", regex=False
    ).str.strip()
    amounts = pd.to_numeric(cleaned_amount, errors="coerce")
    bad = df.index[amounts.isna() | ~amounts.map(lambda v: pd.notna(v) and abs(v) != float("inf"))]
    if len(bad):
        problems.append(f"'amount' must be a number in {row_label(bad)}.")
    else:
        bad = df.index[amounts < 0]
        if len(bad):
            problems.append(
                f"'amount' must not be negative in {row_label(bad)}. "
                "Use the 'type' column to mark income or expense."
            )

    # category
    bad = df.index[df["category"] == ""]
    if len(bad):
        problems.append(f"'category' is empty in {row_label(bad)}.")

    if problems:
        raise DataValidationError(" ".join(problems))

    out = pd.DataFrame(
        {
            "date": dates.dt.normalize(),
            "description": df["description"],
            "category": df["category"],
            "type": types,
            "amount": amounts.astype(float),
        }
    )
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    out["month"] = out["date"].dt.strftime("%Y-%m")
    return out


def load_transactions(source: Source) -> pd.DataFrame:
    """Read a CSV (path, bytes, or file-like object) and return validated data."""
    return validate_and_clean(_read_raw(source))


def load_sample_transactions() -> pd.DataFrame:
    return load_transactions(config.SAMPLE_CSV)


def sample_csv_bytes() -> bytes:
    return Path(config.SAMPLE_CSV).read_bytes()
