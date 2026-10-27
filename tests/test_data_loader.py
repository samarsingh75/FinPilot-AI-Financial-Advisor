import io

import pytest

import config
from src.data_loader import DataValidationError, load_sample_transactions, load_transactions

HEADER = "date,description,category,type,amount\n"


def csv(*rows):
    return (HEADER + "\n".join(rows) + "\n").encode()


def test_sample_loads_with_expected_types():
    df = load_sample_transactions()
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["amount"].dtype == float
    assert set(df["type"]) == {"income", "expense"}
    assert "month" in df.columns


def test_load_from_bytes_and_file_object():
    data = csv("2026-01-01,Pay,Salary,income,1000")
    assert len(load_transactions(data)) == 1
    assert len(load_transactions(io.BytesIO(data))) == 1


def test_headers_case_and_whitespace_insensitive_and_extra_columns_ignored():
    data = b" Date ,Description,CATEGORY,Type,Amount,notes\n2026-01-01,Pay,Salary,INCOME,\"1,000.50\",x\n"
    df = load_transactions(data)
    assert df.loc[0, "type"] == "income" and df.loc[0, "amount"] == 1000.50
    assert "notes" not in df.columns


def test_rupee_symbol_in_amount_is_accepted():
    assert load_transactions(csv("2026-01-01,Pay,Salary,income,\u20b95000")).loc[0, "amount"] == 5000


def test_missing_column_is_reported_by_name():
    with pytest.raises(DataValidationError, match="amount"):
        load_transactions(b"date,description,category,type\n2026-01-01,a,b,income\n")


def test_invalid_date_reports_row():
    with pytest.raises(DataValidationError, match=r"date.*row\(s\) 3"):
        load_transactions(csv("2026-01-01,a,b,income,1", "not-a-date,a,b,income,1"))


def test_invalid_type():
    with pytest.raises(DataValidationError, match="income' or 'expense"):
        load_transactions(csv("2026-01-01,a,b,transfer,1"))


def test_invalid_amount():
    with pytest.raises(DataValidationError, match="amount.*number"):
        load_transactions(csv("2026-01-01,a,b,income,abc"))


def test_negative_amount_rejected():
    with pytest.raises(DataValidationError, match="negative"):
        load_transactions(csv("2026-01-01,a,b,expense,-50"))


def test_empty_category_rejected():
    with pytest.raises(DataValidationError, match="category"):
        load_transactions(csv("2026-01-01,a,,expense,50"))


def test_multiple_problems_reported_together():
    with pytest.raises(DataValidationError) as exc:
        load_transactions(csv("bad,a,b,oops,xyz"))
    message = str(exc.value)
    assert "date" in message and "type" in message and "amount" in message


def test_empty_and_header_only_files():
    with pytest.raises(DataValidationError, match="empty"):
        load_transactions(b"")
    with pytest.raises(DataValidationError, match="no transactions"):
        load_transactions(HEADER.encode())


def test_binary_garbage_is_rejected_safely():
    with pytest.raises(DataValidationError):
        load_transactions(bytes(range(256)) * 4)


def test_oversized_file_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_CSV_MB", 0)
    with pytest.raises(DataValidationError, match="larger"):
        load_transactions(csv("2026-01-01,a,b,income,1"))


def test_missing_file_path():
    with pytest.raises(DataValidationError, match="not found"):
        load_transactions("does_not_exist.csv")


def test_blank_rows_are_dropped_and_data_sorted_by_date():
    df = load_transactions(csv("2026-03-01,a,b,income,1", ",,,,", "2026-01-01,c,d,expense,2"))
    assert len(df) == 2 and df.loc[0, "description"] == "c"
