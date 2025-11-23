import pandas as pd
import polars as pl


def get_year(row: pd.Series, table: pd.DataFrame) -> float:
    match = pd.Series(row["date"]).str.extract(r"(\d{4})", expand=False)

    year = match.iloc[0]

    if not pd.isna(year):
        return float(year)
    idx = row.name  # row.name contains the index

    if pd.isna(year):
        while idx > 0:
            idx -= 1
            prev_date = table.iloc[idx]
            match = pd.Series(prev_date["date"]).str.extract(r"(\d{4})", expand=False)
            year = match.iloc[0]
            if not pd.isna(year):
                break
    return float(year)


def get_year_vectorized(data_series: pd.Series) -> pd.Series:
    years = data_series.str.extract(r"(\d{4})", expand=False).astype(float)
    return years.ffill()


def check_year(row: pd.Series) -> float:
    years_cit = row["citation"].str.slice(0, 50).str.findall(r"(?<!\d)\d{4}(?!\d))")
    years_cit = [int(y) for y in years_cit]
    year = row["year"]
    if year in years_cit:
        return year
    else:
        return


def get_month(item: str) -> pd.Series:
    """
    Extracts the month from a date string.
    Args:
        item (str): The date string to extract the month from.
    Returns:
        pd.Series[int | bool]: A Series containing the month as an integer and a boolean indicating if a month was found.
    """
    # extract three letters that exist in month names
    match = pd.Series([item]).str.extract(r"([A-Za-z]{3})(?:[ ]|[^A-Za-z]|$)")
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    months = month_map.keys()
    if not match.empty and not pd.isna(match.iloc[0, 0]):
        month_abbrev = match.iloc[0, 0].str.lower()
        for m in months:
            if month_abbrev in m:
                return pd.Series([month_map[m.lower()], True])
        else:
            return pd.Series([1, False])
    else:
        return pd.Series([1, False])


def get_day(item: str) -> pd.Series:
    match = pd.Series([item]).str.extract(r"(?:^|\D)(\d{1,2})(?:\D|$)")
    if not match.empty and not pd.isna(match.iloc[0, 0]):
        day = int(match.iloc[0, 0])
        if 1 <= day <= 31:  # validate day range
            return pd.Series([day, True])
    return pd.Series([1, False])


def fill_na_dates(row: pd.Series, table: pd.DataFrame) -> str:
    date = row["date"]
    if pd.isna(date):
        idx = row.name
        while idx > 0:
            idx -= 1
            prev_date = table.iloc[idx]["date"]
            if not pd.isna(prev_date):
                return prev_date
    return date


def replace_same_as_previous(row: pd.Series) -> str:
    auth_match = pd.Series(row["Date"]).str.extract(r"same", expand=False)
    return auth_match.iloc[0]
