import pandas as pd
import polars as pl
import polars.selectors as cs

# from cluster import embed_items, create_cluster
import requests
from bs4 import BeautifulSoup


def parse_historical_table(html_content, session=None):
    """
    Parses the messy HTML table into a structured list of dictionaries.
    """
    if session is None:
        session = requests
    next_root = "https://www.british-history.ac.uk"
    html = session.get(next_root + html_content, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    next_row = soup.find("div", {"id": "block-componentpager"})
    rows = soup.find(class_="table-wrap").find("table").find_all(["th", "td"])
    idx = [
        i
        for i, element in enumerate(rows)
        if element.name == "th" and element.get_text(strip=True)
    ]
    citation = soup.find(class_="chicago").text.strip()
    cols = ["date", "text", "citation"]
    lines = []
    for index, i in enumerate(idx):
        row = []
        row.append(rows[i].text)
        if index == idx.index(idx[-1]):
            row.append("\n".join(t.text for t in rows[i + 1 :]))
        else:
            row.append("\n".join(t.text for t in rows[i + 1 : idx[index + 1]]))
        row.append(citation)
        lines.append(row)

    df1 = pl.DataFrame(data=lines, schema=cols)
    df1 = df1.with_columns(pl.col("text").str.replace_all(r"\n", " "))
    # df1["text"] = df1["text"].str.replace("\n", "")

    return df1


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from langchain_text_splitters import RecursiveCharacterTextSplitter

import tiktoken


def word_len(s: str):
    return len(s.split())


def explode_chunk_text(row):
    text = row["combined_text"]
    n_tokens = row["token_length"]
    id = row["ID"]
    max_tokens = 4000
    if n_tokens > max_tokens:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=200,
            length_function=word_len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_text(text)
        new_rows = []
        for i, chunk in enumerate(chunks):
            new_row = row.copy()
            new_row["combined_text"] = chunk
            new_row["ID"] += i / 10
            new_rows.append(new_row)
        return new_rows
    return [row]


def get_session():
    """Session with retries and backoff."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def clean_table(t: pl.DataFrame) -> pl.DataFrame:
    embedding_encoding = "cl100k_base"
    encoding = tiktoken.get_encoding(embedding_encoding)
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    t = t.with_columns(
        [
            pl.col("date")
            .str.to_lowercase()
            .str.extract(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)")
            .replace(month_map)
            .cast(pl.Int64)
            .fill_null(1)
            .alias("month"),
            pl.col("date")
            .str.extract(r"(?:^|\D)(\d{4})(?:\D|$)")
            .cast(pl.Int64)
            .forward_fill()
            .alias("year"),
        ]
    )
    t = t.with_columns(
        pl.col("citation")
        .str.slice(0, 50)
        .str.extract_all(r"(\d{4})")
        .alias("citation_years")
    )
    t = t.with_columns(
        pl.when(pl.col("citation_years").list.len() == 1)
        .then(
            pl.when(
                pl.col("year") != pl.col("citation_years").list.first().cast(pl.Int64)
            )
            .then(pl.col("citation_years").list.first().cast(pl.Int64))
            .otherwise(pl.col("year"))
        )
        .otherwise(pl.col("year"))
        .alias("year_adjusted")
    )
    t = t.with_columns(
        pl.col("date")
        .str.extract(r"(?:\b|^)(\d{1,2})(?:\b|$)")
        .cast(pl.Int64)
        .fill_null(1)
        .alias("day"),
        pl.col("date")
        .str.extract(r"(?:\b|^)(\d{1,2})(?:\b|$)")
        .is_not_null()
        .alias("has_day"),
        pl.col("date")
        .str.to_lowercase()
        .str.extract(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)")
        .is_not_null()
        .alias("has_month"),
    )
    t = t.with_columns(
        pl.when(pl.col("has_day"))
        .then(pl.lit("day"))
        .otherwise(
            pl.when(pl.col("has_month")).then(pl.lit("month")).otherwise(pl.lit("year"))
        )
        .alias("precision"),
    )
    t = t.drop(["year", "has_day", "has_month"])
    t = t.rename({"year_adjusted": "year"})
    t = t.with_columns(pl.row_index().cast(pl.Float64).alias("ID"))
    t = t.with_columns(
        (
            pl.when(pl.col("date").is_null()).then(pl.lit("")).otherwise(pl.col("date"))
            + pl.when(pl.col("text").is_null())
            .then(pl.lit(""))
            .otherwise(pl.col("text"))
        ).alias("combined_text")
    )
    t = (
        t.with_columns(
            pl.struct(pl.all())
            .map_elements(
                explode_chunk_text,
                returns_scalar=True,
                return_dtype=pl.List(pl.Struct(t.schema)),
            )
            .alias("new_rows")
        )
        .select(pl.col("new_rows"))
        .explode("new_rows")
        .unnest("new_rows")
    ).with_columns(
        pl.col("combined_text")
        .map_elements(lambda s: len(encoding.encode(s)))
        .alias("token_length")
    )
    return t


def generate_data():
    session = get_session()
    # from cluster import embed_items
    # url_num = 4

    with open("./data/data_links.txt", "r") as f:
        data_links = f.read().splitlines()

    # data_links = data_links[:url_num]

    dfs = []
    for url in data_links:
        df = parse_historical_table(url, session=session)
        dfs.append(df)
        print(f"finished {url}")
    table = pl.concat(dfs, how="vertical")
    table = clean_table(table)
    table.write_csv("./data/state_papers_ven-for-cleaned.csv", include_header=True)
    print("Data saved to ./data/state_papers_ven-for-cleaned.csv")
